"""Single-GPU AIOB profiler for DeepSeek-V2-Lite/Chat.

The original AiobDeepSeek module models the Q-LoRA/FP8 671B path.  V2-Lite
uses BF16 weights, has no Q-LoRA, and has different dense and MoE FFN sizes,
so it gets a separate implementation and leaves the 671B profiler untouched.
"""

import os

import torch
import torch.nn.functional as F
import triton

import workload_generator.mocked_model.inference.AiobDeepSeek as AiobDeepSeek
from workload_generator.mocked_model.MockedModel import InferencePhase
from utils.utils import (
    Strategy,
    calculate_stats,
    get_ep_expected_m_per_group,
    process_all_keys,
    write_time,
)


def _bench_bf16_gemm(m, k, n):
    """Return BF16 GEMM latency in seconds (triton.do_bench returns ms)."""
    x = torch.randn((m, k), device="cuda:0", dtype=torch.bfloat16)
    weight = torch.randn((k, n), device="cuda:0", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda:0", dtype=torch.bfloat16)

    def gemm():
        torch.mm(x, weight, out=out)

    return triton.testing.do_bench(gemm) / 1000.0


def _bench_fp8_gemm(m, k, n):
    x_fp8, y_fp8, c, out, ref_out = AiobDeepSeek.construct(m, k, n)
    AiobDeepSeek.deep_gemm.fp8_gemm_nt(
        x_fp8,
        y_fp8,
        out,
        c=c,
        disable_ue8m0_cast=True,
        recipe=None,
    )
    diff = AiobDeepSeek.calc_diff(out, ref_out)
    assert diff < 0.001, f"{m=}, {k=}, {n=}, {diff:.5f}"

    def gemm():
        AiobDeepSeek.deep_gemm.fp8_gemm_nt(
            x_fp8,
            y_fp8,
            out,
            c=c,
            disable_ue8m0_cast=True,
            recipe=None,
        )

    return AiobDeepSeek.bench_kineto(
        gemm, "fp8_gemm", suppress_kineto_output=True
    )


def _bench_gemm(args, m, k, n):
    dtype = getattr(args, "aiob_dtype", "bfloat16").lower()
    if dtype in ("bfloat16", "bf16"):
        return _bench_bf16_gemm(m, k, n)
    if dtype in ("float8", "fp8"):
        return _bench_fp8_gemm(m, k, n)
    raise ValueError(f"Unsupported DeepSeek-V2-Lite AIOB dtype: {dtype}")


def _bench_bf16_grouped_gemm(num_groups, tokens_per_group, k, n):
    """Approximate BF16 grouped GEMM with one strided batched GEMM."""
    tokens_per_group = max(int(tokens_per_group), 1)
    x = torch.randn(
        (num_groups, tokens_per_group, k),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        (num_groups, k, n), device="cuda:0", dtype=torch.bfloat16
    )
    out = torch.empty(
        (num_groups, tokens_per_group, n),
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    def grouped_gemm():
        torch.bmm(x, weight, out=out)

    return triton.testing.do_bench(grouped_gemm) / 1000.0


class DeepSeekV2LiteAttention(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tp = args.tensor_model_parallel_size
        self.hidden_size = args.hidden_size
        self.num_heads = args.head_num
        self.batch_size = args.micro_batch
        self.kv_lora_rank = args.d_kv_c
        self.qk_nope_head_dim = args.d_q
        self.qk_rope_head_dim = args.d_r
        self.v_head_dim = args.d_kv

    def _q_proj(self, m):
        k = self.hidden_size
        n = (
            self.num_heads
            * (self.qk_nope_head_dim + self.qk_rope_head_dim)
            // self.tp
        )
        return _bench_gemm(self.args, m, k, n)

    def _kv_a_proj_with_mqa(self, m):
        k = self.hidden_size
        n = self.kv_lora_rank + self.qk_rope_head_dim
        return _bench_gemm(self.args, m, k, n)

    def _kv_b_proj(self, m):
        k = self.kv_lora_rank
        n = (
            self.num_heads
            * (self.qk_nope_head_dim + self.v_head_dim)
            // self.tp
        )
        return _bench_gemm(self.args, m, k, n)

    def _attention(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        batch = self.batch_size
        if phase == InferencePhase.PREFILL.value:
            q_length = self.args.seq_length
            kv_length = self.args.seq_length
            is_causal = True
        else:
            q_length = 1
            kv_length = self.args.seq_length + 1
            is_causal = False

        local_heads = self.num_heads // self.tp
        qk_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q = torch.randn(
            (batch, local_heads, q_length, qk_dim),
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            (batch, local_heads, kv_length, qk_dim),
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            (batch, local_heads, kv_length, self.v_head_dim),
            device="cuda:0",
            dtype=torch.bfloat16,
        )

        def attention():
            F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        return triton.testing.do_bench(attention)

    def _o_proj(self, m):
        k = self.num_heads * self.v_head_dim // self.tp
        n = self.hidden_size
        return _bench_gemm(self.args, m, k, n)

    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        m = (
            self.args.seq_length * self.batch_size
            if phase == InferencePhase.PREFILL.value
            else self.batch_size
        )
        q_time = self._q_proj(m)
        kv_a_time = self._kv_a_proj_with_mqa(m)
        kv_b_time = self._kv_b_proj(m)
        attention_time = self._attention()
        o_time = self._o_proj(m)
        return q_time + kv_a_time + kv_b_time, attention_time, o_time


class DeepSeekV2LiteMLP(AiobDeepSeek.DeepSeekMLP):
    def __init__(self, args, intermediate_size, tensor_parallel_size=None):
        torch.nn.Module.__init__(self)
        self.batch_size = args.micro_batch
        self.hidden_size = args.hidden_size
        self.expert_dim = intermediate_size
        self.seq_length = args.seq_length
        self.tp = (
            args.tensor_model_parallel_size
            if tensor_parallel_size is None
            else tensor_parallel_size
        )
        self.args = args

    def _up_gate(self, m):
        return _bench_gemm(
            self.args,
            m,
            self.hidden_size,
            self.expert_dim * 2 // self.tp,
        )

    def _down(self, m):
        return _bench_gemm(
            self.args,
            m,
            self.expert_dim // self.tp,
            self.hidden_size,
        )

    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        m = (
            self.seq_length * self.batch_size
            if phase == InferencePhase.PREFILL.value
            else self.batch_size
        )
        return self._up_gate(m), self._down(m)


class DeepSeekV2LiteMOE(AiobDeepSeek.DeepSeekMOE):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.batch_size = args.micro_batch
        self.hidden_size = args.hidden_size
        self.expert_dim = args.moe_intermediate_size
        self.num_experts = args.router_expert
        self.ep = args.expert_model_parallel_size
        self.topk = args.moe_router_topk
        self.seq_length = args.seq_length
        self.args = args

        if self.ep < 1 or self.num_experts % self.ep != 0:
            raise ValueError(
                "DeepSeek-V2-Lite requires router_expert to be divisible by "
                "expert_model_parallel_size"
            )

    def _tokens_per_local_expert(self, m):
        local_experts = self.num_experts // self.ep
        strategy = getattr(
            self.args, "moe_routing_strategy", Strategy.RoundRobin
        )
        return local_experts, get_ep_expected_m_per_group(
            m, local_experts, self.topk, self.ep, strategy
        )

    def _up_gate(self, m):
        local_experts, tokens = self._tokens_per_local_expert(m)
        if getattr(self.args, "aiob_dtype", "bfloat16").lower() in (
            "bfloat16",
            "bf16",
        ):
            return _bench_bf16_grouped_gemm(
                local_experts,
                tokens,
                self.hidden_size,
                self.expert_dim * 2,
            )
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        bench = (
            AiobDeepSeek.bench_contiguous
            if phase == InferencePhase.PREFILL.value
            else AiobDeepSeek.bench_masked
        )
        return bench(
            local_experts, tokens, self.hidden_size, self.expert_dim * 2
        )

    def _down(self, m):
        local_experts, tokens = self._tokens_per_local_expert(m)
        if getattr(self.args, "aiob_dtype", "bfloat16").lower() in (
            "bfloat16",
            "bf16",
        ):
            return _bench_bf16_grouped_gemm(
                local_experts,
                tokens,
                self.expert_dim,
                self.hidden_size,
            )
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        bench = (
            AiobDeepSeek.bench_contiguous
            if phase == InferencePhase.PREFILL.value
            else AiobDeepSeek.bench_masked
        )
        return bench(local_experts, tokens, self.expert_dim, self.hidden_size)

    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        m = (
            self.seq_length * self.batch_size
            if phase == InferencePhase.PREFILL.value
            else self.batch_size
        )
        return self._up_gate(m), self._down(m)


class DeepSeekV2LiteModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_list = {}
        self.args = args
        self.attention = DeepSeekV2LiteAttention(args)
        self.dense_mlp = DeepSeekV2LiteMLP(
            args, args.dense_intermediate_size
        )
        self.shared_expert = DeepSeekV2LiteMLP(
            args,
            args.moe_intermediate_size * args.shared_experts,
            tensor_parallel_size=1,
        )
        self.moe = DeepSeekV2LiteMOE(args)
        self.aiob_forward_loops = getattr(args, "aiob_forward_loops", 10)

    def _record(self, key, value, scale=1e6):
        self.time_list.setdefault(key, []).append({"time_gpu": value * scale})

    def forward(self):
        for _ in range(self.aiob_forward_loops):
            qkv, core, output = self.attention()
            self._record("atten_qkv", qkv)
            self._record("atten_flash", core, scale=1e3)
            self._record("atten_linear", output)

            dense_up, dense_down = self.dense_mlp()
            self._record("mlp_up", dense_up)
            self._record("mlp_down", dense_down)

            shared_up, shared_down = self.shared_expert()
            self._record("shared_experts_up", shared_up)
            self._record("shared_experts_down", shared_down)

            moe_up, moe_down = self.moe()
            self._record("moe_up_gate", moe_up)
            self._record("moe_down", moe_down)

        result_dir = "./results/aiob_outputs"
        os.makedirs(result_dir, exist_ok=True)
        stats_path = os.path.join(
            result_dir, f"{self.args.model_name}_time_list_stats.txt"
        )
        calculate_stats(self.time_list, stats_path)
        filepath = write_time(self.time_list, self.args)
        process_all_keys(filepath)
        return filepath
