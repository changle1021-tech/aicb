"""Qwen3-Next MoE workload contracts and isolated device profiling.

Load helpers are CPU-only. CUDA dependencies are imported only when an
explicit profiling function is called.
"""

import math
import random

from workload_generator.mocked_model.inference.qwen3_next_projections import (
    benchmark_gpu_region,
    projection_precision,
)


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer, got {!r}".format(name, value))
    return value


def moe_parallel_layout(args):
    """Return (dp, ep, local_experts) for the AG/RS EP layout used by vLLM."""
    tp = _positive_int("tensor_model_parallel_size", args.tensor_model_parallel_size)
    pp = _positive_int("pipeline_model_parallel", args.pipeline_model_parallel)
    world_size = _positive_int("world_size", args.world_size)
    num_experts = _positive_int("num_experts", args.num_experts)
    if world_size % (tp * pp):
        raise ValueError("world_size must be divisible by TP * PP")
    dp = world_size // tp // pp
    ep = tp * dp
    if num_experts % ep:
        raise ValueError("num_experts must be divisible by EP")
    return dp, ep, num_experts // ep


def gathered_tokens(local_tokens, dp):
    """Tokens seen by each rank after the DP all-gather.

    vLLM pads every DP rank to the same token count, so a rank processes
    local_tokens * dp rows even when only one rank holds a real request.
    """
    return _positive_int("local_tokens", local_tokens) * _positive_int("dp", dp)


def expected_local_routes(local_tokens, dp, topk, ep):
    """Expected token-expert routes landing on one EP rank."""
    return gathered_tokens(local_tokens, dp) * _positive_int("topk", topk) / _positive_int("ep", ep)


def sample_expert_loads(mean_per_expert, num_experts, rng=None):
    """Integer load per expert with E[load] == mean_per_expert (stochastic rounding).

    Small loads are not raised to one: with mean 10/128 most experts get zero
    rows, and the expected total equals mean_per_expert * num_experts.
    """
    if mean_per_expert < 0 or not math.isfinite(mean_per_expert):
        raise ValueError("mean_per_expert must be a finite non-negative number")
    _positive_int("num_experts", num_experts)
    rng = rng or random.Random()
    base = math.floor(mean_per_expert)
    frac = mean_per_expert - base
    return [base + int(rng.random() < frac) for _ in range(num_experts)]


def sample_routing(num_tokens, num_experts, topk, generator=None):
    """Uniform routing: each token picks topk distinct experts."""
    import torch

    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
    return scores.topk(topk, dim=1).indices.to(torch.int32)


def benchmark_routed_experts(args, local_tokens):
    """Device time of routed-expert Gate/Up, SiluAndMul and Down on one EP rank."""
    import torch

    dp, ep, local_experts = moe_parallel_layout(args)
    tokens = gathered_tokens(local_tokens, dp)
    hidden = _positive_int("hidden_size", args.hidden_size)
    inter = _positive_int("moe_intermediate_size", args.moe_intermediate_size)
    topk = _positive_int("num_experts_per_tok", args.num_experts_per_tok)
    num_experts = args.num_experts

    if projection_precision(args) == "fp8":
        return _benchmark_routed_experts_fp8(args, local_tokens, dp, ep, local_experts)

    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(local_experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(local_experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    topk_ids = sample_routing(tokens, num_experts, topk)
    topk_weights = torch.full((tokens, topk), 1.0 / topk, device="cuda", dtype=torch.float32)
    expert_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
    expert_map[:local_experts] = torch.arange(local_experts, device="cuda", dtype=torch.int32)

    def run():
        fused_experts(x, w1, w2, topk_weights, topk_ids,
                      global_num_experts=num_experts, expert_map=expert_map)

    return benchmark_gpu_region(run)


def _benchmark_routed_experts_fp8(args, local_tokens, dp, ep, local_experts):
    from utils.deepgemm_utils import bench_contiguous, bench_masked

    hidden, inter = args.hidden_size, args.moe_intermediate_size
    mean = expected_local_routes(local_tokens, dp, args.num_experts_per_tok, ep) / local_experts
    loads = sample_expert_loads(mean, local_experts)
    expected_m = max(1, math.ceil(mean))
    decode = getattr(args, "phase", "decode") == "decode"
    bench = bench_masked if decode else bench_contiguous
    up = bench(local_experts, expected_m, hidden, 2 * inter, loads=loads)
    down = bench(local_experts, expected_m, inter, hidden, loads=loads)
    return up + down
