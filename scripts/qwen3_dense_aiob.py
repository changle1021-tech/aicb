"""CUDA hardware profiler for dense Qwen3 inference blocks.

Timings are emitted in the AIOB inference format: microseconds in the profile
file, then converted to nanoseconds by the workload generator.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable


class ProfilingError(RuntimeError):
    """Raised when the requested hardware profile cannot be collected."""


def _load_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise ProfilingError(
            "AIOB profiling requires PyTorch with CUDA support; install the "
            "inference dependencies or run inside the AICB GPU container"
        ) from error

    if not torch.cuda.is_available():
        raise ProfilingError("AIOB profiling requires a CUDA-capable GPU")
    if not torch.cuda.is_bf16_supported():
        raise ProfilingError("Qwen3-32B AIOB profiling requires BF16-capable CUDA hardware")
    return torch, functional


def _benchmark(
    torch: Any,
    operation: Callable[[], Any],
    loops: int,
    warmup_loops: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup_loops):
            operation()
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(loops)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(loops)]
        for start, end in zip(starts, ends):
            start.record()
            operation()
            end.record()
        torch.cuda.synchronize()
    # CUDA events report milliseconds; AIOB inference profiles use microseconds.
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _filtered_stats(samples: list[float]) -> dict[str, float]:
    minimum = min(samples)
    filtered = [sample for sample in samples if sample <= 3 * minimum]
    return {
        "time_gpu_max": max(filtered),
        "time_gpu_min": minimum,
        "time_gpu_avg": sum(filtered) / len(filtered),
    }


def _combine_samples(*sample_groups: list[float]) -> list[float]:
    return [sum(samples) for samples in zip(*sample_groups)]


class DenseQwen3Profiler:
    def __init__(
        self,
        config: dict[str, Any],
        phase: str,
        seq_length: int,
        micro_batch: int,
        tensor_parallel_size: int,
        loops: int,
        warmup_loops: int = 3,
    ) -> None:
        self.torch, self.functional = _load_torch()
        self.config = config
        self.phase = phase
        self.seq_length = seq_length
        self.micro_batch = micro_batch
        self.tp = tensor_parallel_size
        self.loops = loops
        self.warmup_loops = warmup_loops
        self.dtype = self.torch.bfloat16
        self.device = self.torch.device("cuda", self.torch.cuda.current_device())

        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.head_dim = config["head_dim"]
        self.query_heads = config["num_attention_heads"] // self.tp
        # When TP exceeds the number of KV heads, vLLM replicates KV heads.
        self.kv_heads = max(1, config["num_key_value_heads"] // self.tp)
        self.query_size = self.query_heads * self.head_dim
        self.kv_size = self.kv_heads * self.head_dim
        self.local_intermediate_size = self.intermediate_size // self.tp
        self.active_tokens = micro_batch * (seq_length if phase == "prefill" else 1)

        if self.intermediate_size % self.tp:
            raise ProfilingError(
                "intermediate_size must be divisible by tensor parallel size"
            )
        if self.query_heads % self.kv_heads:
            raise ProfilingError("local query heads must be divisible by local KV heads")

        self._vllm_ops = None
        self._vllm_get_rope = None
        try:
            from vllm import _custom_ops as vllm_ops
            from vllm.model_executor.layers.rotary_embedding import get_rope

            self._vllm_ops = vllm_ops
            self._vllm_get_rope = get_rope
        except (ImportError, OSError):
            # torch.nn.functional.rms_norm remains a valid hardware kernel
            # fallback when vLLM custom operations are unavailable.
            pass

    def _release(self) -> None:
        gc.collect()
        self.torch.cuda.empty_cache()

    def _measure_linear(self, rows: int, input_size: int, output_size: int) -> list[float]:
        torch = self.torch
        x = torch.empty((rows, input_size), device=self.device, dtype=self.dtype)
        weight = torch.empty(
            (output_size, input_size), device=self.device, dtype=self.dtype
        )
        result = _benchmark(
            torch,
            lambda: self.functional.linear(x, weight),
            self.loops,
            self.warmup_loops,
        )
        del x, weight
        self._release()
        return result

    def _measure_residual_rms_norm(self) -> list[float]:
        torch = self.torch
        x = torch.empty(
            (self.active_tokens, self.hidden_size),
            device=self.device,
            dtype=self.dtype,
        )
        residual = torch.empty_like(x)
        weight = torch.ones(self.hidden_size, device=self.device, dtype=self.dtype)
        epsilon = self.config["rms_norm_eps"]

        if self._vllm_ops is not None:
            def operation() -> Any:
                x_work = x.clone()
                residual_work = residual.clone()
                self._vllm_ops.fused_add_rms_norm(
                    x_work, residual_work, weight, epsilon
                )
                return x_work
        else:
            def operation() -> Any:
                return self.functional.rms_norm(
                    x + residual, (self.hidden_size,), weight, epsilon
                )

        result = _benchmark(torch, operation, self.loops, self.warmup_loops)
        del x, residual, weight
        self._release()
        return result

    def _measure_head_rms_norm(self, heads: int) -> list[float]:
        torch = self.torch
        x = torch.empty(
            (self.active_tokens, heads, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        weight = torch.ones(self.head_dim, device=self.device, dtype=self.dtype)
        epsilon = self.config["rms_norm_eps"]

        if self._vllm_ops is not None:
            def operation() -> Any:
                output = torch.empty_like(x)
                self._vllm_ops.rms_norm(output, x, weight, epsilon)
                return output
        else:
            def operation() -> Any:
                return self.functional.rms_norm(x, (self.head_dim,), weight, epsilon)

        result = _benchmark(torch, operation, self.loops, self.warmup_loops)
        del x, weight
        self._release()
        return result

    def _measure_rope(self) -> list[float]:
        torch = self.torch
        q = torch.empty(
            (self.active_tokens, self.query_heads, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        k = torch.empty(
            (self.active_tokens, self.kv_heads, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        positions = torch.arange(self.seq_length, device=self.device)
        if self.phase == "prefill":
            positions = positions.repeat(self.micro_batch)
        else:
            positions = torch.full(
                (self.micro_batch,),
                self.seq_length - 1,
                device=self.device,
                dtype=torch.long,
            )

        if self._vllm_get_rope is not None:
            rotary_embedding = self._vllm_get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=self.config["max_position_embeddings"],
                base=self.config["rope_theta"],
                rope_scaling=self.config.get("rope_scaling"),
            )

            def operation() -> Any:
                return rotary_embedding(
                    positions,
                    q.view(self.active_tokens, -1),
                    k.view(self.active_tokens, -1),
                )

            result = _benchmark(torch, operation, self.loops, self.warmup_loops)
            del q, k, positions, rotary_embedding
            self._release()
            return result

        inv_freq = 1.0 / (
            self.config["rope_theta"]
            ** (
                torch.arange(0, self.head_dim, 2, device=self.device).float()
                / self.head_dim
            )
        )
        frequencies = torch.outer(positions.float(), inv_freq)
        cosine = frequencies.cos().to(self.dtype).unsqueeze(1)
        sine = frequencies.sin().to(self.dtype).unsqueeze(1)

        def rotate(x: Any) -> Any:
            first, second = x.chunk(2, dim=-1)
            return torch.cat((-second, first), dim=-1)

        def operation() -> Any:
            cosine_full = torch.cat((cosine, cosine), dim=-1)
            sine_full = torch.cat((sine, sine), dim=-1)
            return (
                q * cosine_full + rotate(q) * sine_full,
                k * cosine_full + rotate(k) * sine_full,
            )

        result = _benchmark(torch, operation, self.loops, self.warmup_loops)
        del q, k, positions, inv_freq, frequencies, cosine, sine
        self._release()
        return result

    def _measure_attention(self) -> list[float]:
        torch = self.torch
        if self.phase == "prefill":
            query_length = self.seq_length
            q = torch.empty(
                (self.micro_batch, self.query_heads, query_length, self.head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            k = torch.empty(
                (self.micro_batch, self.kv_heads, query_length, self.head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            v = torch.empty_like(k)
            is_causal = True
        else:
            q = torch.empty(
                (self.micro_batch, self.query_heads, 1, self.head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            k = torch.empty(
                (self.micro_batch, self.kv_heads, self.seq_length, self.head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            v = torch.empty_like(k)
            is_causal = False

        def operation() -> Any:
            return self.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
                enable_gqa=self.query_heads != self.kv_heads,
            )

        result = _benchmark(torch, operation, self.loops, self.warmup_loops)
        del q, k, v
        self._release()
        return result

    def _measure_mlp_activation(self) -> list[float]:
        torch = self.torch
        gate_up = torch.empty(
            (self.active_tokens, 2 * self.local_intermediate_size),
            device=self.device,
            dtype=self.dtype,
        )

        if self._vllm_ops is not None:
            def operation() -> Any:
                output = torch.empty(
                    (self.active_tokens, self.local_intermediate_size),
                    device=self.device,
                    dtype=self.dtype,
                )
                self._vllm_ops.silu_and_mul(output, gate_up)
                return output
        else:
            def operation() -> Any:
                gate, up = gate_up.chunk(2, dim=-1)
                return self.functional.silu(gate) * up

        result = _benchmark(torch, operation, self.loops, self.warmup_loops)
        del gate_up
        self._release()
        return result

    def run(self) -> dict[str, list[float]]:
        qkv_projection = self._measure_linear(
            self.active_tokens,
            self.hidden_size,
            self.query_size + 2 * self.kv_size,
        )
        query_norm = self._measure_head_rms_norm(self.query_heads)
        key_norm = self._measure_head_rms_norm(self.kv_heads)
        return {
            "atten_norm": self._measure_residual_rms_norm(),
            "atten_qkv": _combine_samples(qkv_projection, query_norm, key_norm),
            "atten_rotary_emb": self._measure_rope(),
            "atten_flash": self._measure_attention(),
            "atten_o": self._measure_linear(
                self.active_tokens, self.query_size, self.hidden_size
            ),
            "mlp_norm": self._measure_residual_rms_norm(),
            "mlp_gate_up": self._measure_linear(
                self.active_tokens,
                self.hidden_size,
                2 * self.local_intermediate_size,
            ),
            "mlp_act": self._measure_mlp_activation(),
            "mlp_down": self._measure_linear(
                self.active_tokens,
                self.local_intermediate_size,
                self.hidden_size,
            ),
        }


def profile_dense_qwen3(
    config: dict[str, Any],
    model_name: str,
    phase: str,
    seq_length: int,
    micro_batch: int,
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    loops: int,
    output_dir: Path,
) -> Path:
    profiler = DenseQwen3Profiler(
        config=config,
        phase=phase,
        seq_length=seq_length,
        micro_batch=micro_batch,
        tensor_parallel_size=tensor_parallel_size,
        loops=loops,
    )
    try:
        samples = profiler.run()
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise ProfilingError(
                "CUDA ran out of memory while profiling Qwen3-32B; reduce "
                "micro_batch or seq_length, or increase tensor parallel size"
            ) from error
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{model_name}-world_size{world_size}-tp{tensor_parallel_size}"
        f"-pp{pipeline_parallel_size}-ep1-bpg{micro_batch}"
        f"-seq{seq_length}-{phase}.txt"
    )
    output_path = output_dir / filename
    device_name = profiler.torch.cuda.get_device_name(profiler.device)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"inference_term:{phase}\n")
        output.write(f"device: {device_name}\n")
        output.write(f"dtype: bfloat16\n")
        output.write(f"profile_loops: {loops}\n")
        for section, section_samples in samples.items():
            output.write(f"{section}:\n")
            for name, value in _filtered_stats(section_samples).items():
                output.write(f"    {name}: {value}\n")
    print(f"AIOB compute profile saved in: {output_path}")
    return output_path
