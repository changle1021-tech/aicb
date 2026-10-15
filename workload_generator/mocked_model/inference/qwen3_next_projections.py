"""Qwen3-Next projection contracts and isolated device profiling.

Configuration/metadata helpers are CPU-only. CUDA dependencies are imported only
when an explicit profiling function is called.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from statistics import median


PROJECTION_PROFILE_VERSION = 2
CACHE_FLUSH_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class ProjectionShapes:
    hidden_size: int
    gdn_key_size: int
    gdn_value_size: int
    gdn_qkvz_size: int
    gdn_ba_size: int
    attention_query_size: int
    attention_kv_size: int
    attention_query_heads: int
    attention_kv_heads: int
    attention_qkv_size: int
    attention_output_gate: bool


def _positive_int(args, name):
    value = getattr(args, name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer, got {!r}".format(name, value))
    return value


def projection_shapes(args):
    """Match vLLM's local projection widths, including replicated KV heads."""
    tp = _positive_int(args, "tensor_model_parallel_size")
    key_heads = _positive_int(args, "linear_num_key_heads")
    value_heads = _positive_int(args, "linear_num_value_heads")
    query_heads = _positive_int(args, "num_attention_heads")
    kv_heads = _positive_int(args, "num_key_value_heads")
    if key_heads % tp or value_heads % tp or query_heads % tp:
        raise ValueError("TP must divide GDN key/value heads and attention query heads")
    if value_heads % key_heads:
        raise ValueError("GDN value heads must be a multiple of key heads")
    if (kv_heads >= tp and kv_heads % tp) or (kv_heads < tp and tp % kv_heads):
        raise ValueError("KV heads must be shardable or replicable across TP ranks")
    local_queries, local_kv = query_heads // tp, max(1, kv_heads // tp)
    if local_queries % local_kv:
        raise ValueError("Local query heads must be a multiple of local KV heads")
    gate = getattr(args, "attn_output_gate", None)
    gate = True if gate is None else gate
    if not isinstance(gate, bool):
        raise ValueError("attn_output_gate must be a boolean")
    key_size = key_heads * _positive_int(args, "linear_key_head_dim") // tp
    value_size = value_heads * _positive_int(args, "linear_value_head_dim") // tp
    head_dim = _positive_int(args, "head_dim")
    q_size, kv_size = local_queries * head_dim, local_kv * head_dim
    return ProjectionShapes(
        hidden_size=_positive_int(args, "hidden_size"),
        gdn_key_size=key_size,
        gdn_value_size=value_size,
        gdn_qkvz_size=2 * key_size + 2 * value_size,
        gdn_ba_size=2 * value_heads // tp,
        attention_query_size=q_size,
        attention_kv_size=kv_size,
        attention_query_heads=local_queries,
        attention_kv_heads=local_kv,
        attention_qkv_size=q_size * (1 + int(gate)) + 2 * kv_size,
        attention_output_gate=gate,
    )


def projection_precision(args):
    value = getattr(args, "projection_precision", None)
    if value is None:
        value = getattr(args, "torch_dtype", "bfloat16")
    value = str(value).lower()
    if value.startswith("torch."):
        value = value[len("torch."):]
    if value in ("bf16", "bfloat16"):
        return "bf16"
    if value == "fp8":
        return "fp8"
    raise ValueError("Qwen3-Next projections support bf16 or fp8, got {!r}".format(value))


def benchmark_gpu_region(fn, warmup=5, samples=30):
    """Return seconds for one device region, excluding Python launch gaps.

    Capture the entire callback so multi-kernel linear implementations and both
    sigmoid/multiply operations are included. Compilation and allocation of
    inputs belong outside this helper. This is not model-forward wall time.
    """
    import torch

    if warmup < 1 or samples < 1:
        raise ValueError("warmup and samples must be positive")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    flush = torch.empty(CACHE_FLUSH_BYTES, device="cuda", dtype=torch.uint8)
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flush.zero_()
        start.record()
        fn()
        end.record()
    timings = []
    for _ in range(samples):
        graph.replay()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return median(timings) / 1000.0


def benchmark_projection(args, m, k, n):
    import torch

    precision = projection_precision(args)
    if precision == "bf16":
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        return benchmark_gpu_region(lambda: torch.nn.functional.linear(x, weight))

    from utils.deepgemm_utils import construct, deep_gemm, calc_diff

    x, weight, c, out, reference = construct(m, k, n)

    def run():
        deep_gemm.fp8_gemm_nt(
            x, weight, out, c=c, disable_ue8m0_cast=True, recipe=None
        )

    run()
    diff = calc_diff(out, reference)
    if not diff < 0.001:
        raise ValueError("FP8 projection check failed for M/K/N={}/{}/{}: {}".format(m, k, n, diff))
    return benchmark_gpu_region(run)


def benchmark_input_quantization(args, m, width):
    if projection_precision(args) == "bf16":
        return 0.0
    import torch
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    x = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
    return benchmark_gpu_region(
        lambda: per_token_group_quant_fp8(x, 128, column_major_scales=True)
    )


def benchmark_attention_output_gate(args, m, width):
    if not projection_shapes(args).attention_output_gate:
        return 0.0
    import torch

    output = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(output)
    return benchmark_gpu_region(lambda: output * torch.sigmoid(gate))


def projection_metadata(args):
    """Identity of these projection components, not of all MoE compute."""
    sources = [Path(__file__), Path(__file__).with_name("AiobQwen3Next.py")]
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    return {
        "projection_profile_version": PROJECTION_PROFILE_VERSION,
        "scope": "qwen3_next_projections_only",
        "precision": projection_precision(args),
        "timing_scope": "cuda_graph_device_region_seconds",
        "cache_flush_bytes": CACHE_FLUSH_BYTES,
        "source_sha256": source_hash,
        "shapes": asdict(projection_shapes(args)),
        "invocation": {name: getattr(args, name, None) for name in (
            "model_name", "world_size", "tensor_model_parallel_size",
            "pipeline_model_parallel", "expert_model_parallel_size",
            "micro_batch", "seq_length", "phase",
        )},
    }


def write_projection_metadata(profile_path, args):
    path = Path(str(profile_path) + ".projections.json")
    metadata = projection_metadata(args)
    metadata["profile_sha256"] = hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def validate_projection_metadata(profile_path, args):
    path = Path(str(profile_path) + ".projections.json")
    if not profile_path or not Path(profile_path).is_file() or not path.is_file():
        raise ValueError("Qwen3-Next projection profile/metadata is missing; explicitly regenerate the profile with the corrected generator")
    expected = projection_metadata(args)
    expected["profile_sha256"] = hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()
    if json.loads(path.read_text()) != expected:
        raise ValueError("Qwen3-Next projection profile identity differs; explicitly regenerate it instead of reusing legacy timings")
