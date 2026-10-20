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


def validate_routed_replay(args, local_tokens, inputs):
    """Check a replay's tensor shapes without importing CUDA dependencies.

    The expert map is supplied by the serving rank, rather than reconstructed
    as rank zero. Local rows include runtime padding; callers must not replace
    them with the number of live requests.
    """
    dp, ep, local_experts = moe_parallel_layout(args)
    tokens = gathered_tokens(local_tokens, dp)
    hidden = _positive_int("hidden_size", args.hidden_size)
    inter = _positive_int("moe_intermediate_size", args.moe_intermediate_size)
    topk = _positive_int("num_experts_per_tok", args.num_experts_per_tok)
    shapes = dict(hidden_states=(tokens, hidden),
                  w1=(local_experts, 2 * inter, hidden),
                  w2=(local_experts, hidden, inter),
                  topk_ids=(tokens, topk), topk_weights=(tokens, topk),
                  expert_map=(args.num_experts,))
    for name, expected in shapes.items():
        if name not in inputs or tuple(inputs[name].shape) != expected:
            raise ValueError("{} replay shape must be {}".format(name, expected))
    return inputs


def benchmark_routed_experts(args, local_tokens, *, replay_inputs=None,
                             cache_policy="cold", timing_details=None):
    """Device time of routed-expert Gate/Up, SiluAndMul and Down on one EP rank."""
    import torch

    dp, ep, local_experts = moe_parallel_layout(args)
    tokens = gathered_tokens(local_tokens, dp)
    hidden = _positive_int("hidden_size", args.hidden_size)
    inter = _positive_int("moe_intermediate_size", args.moe_intermediate_size)
    topk = _positive_int("num_experts_per_tok", args.num_experts_per_tok)
    num_experts = args.num_experts

    if projection_precision(args) == "fp8":
        if (replay_inputs is not None or cache_policy != "cold"
                or timing_details is not None):
            raise ValueError("Replay/cache timing controls require the BF16 routed-expert backend")
        return _benchmark_routed_experts_fp8(args, local_tokens, dp, ep, local_experts)

    from vllm.model_executor.layers.fused_moe.fused_moe import (
        disable_inplace, fused_experts,
    )

    if replay_inputs is None:
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(local_experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(local_experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02
        topk_ids = sample_routing(tokens, num_experts, topk)
        topk_weights = torch.full((tokens, topk), 1.0 / topk, device="cuda", dtype=torch.float32)
        expert_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
        expert_map[:local_experts] = torch.arange(local_experts, device="cuda", dtype=torch.int32)
    else:
        data = validate_routed_replay(args, local_tokens, replay_inputs)
        x, w1, w2 = (data[n] for n in ("hidden_states", "w1", "w2"))
        topk_ids, topk_weights, expert_map = (
            data[n] for n in ("topk_ids", "topk_weights", "expert_map"))

    # Match the serving call AND its runtime dispatch: vLLM disables inplace
    # on torch >= 2.9. Do not introduce an input-copy/cache change in that case.
    # On runtimes that do use inplace, restore a private working tensor outside
    # the timed region so replay never feeds back the previous output.
    prepare = None
    if not disable_inplace():
        reference = x
        x = reference.clone()

        def prepare():
            x.copy_(reference)

    def run():
        fused_experts(x, w1, w2, topk_weights, topk_ids,
                      inplace=True, global_num_experts=num_experts,
                      expert_map=expert_map)

    return benchmark_gpu_region(run, cache_policy=cache_policy,
                                timing_details=timing_details, prepare=prepare)


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


def benchmark_shared_expert(args, local_tokens):
    """Device time of the shared expert on one DP rank.

    The shared expert is replicated (not EP-sharded) and runs on this rank's
    own tokens: gate/up [hidden -> 2*inter], SiluAndMul, down [inter -> hidden],
    then multiplies by sigmoid(shared_expert_gate(x)).
    """
    import torch
    import torch.nn.functional as F

    hidden = _positive_int("hidden_size", args.hidden_size)
    inter = _positive_int("shared_expert_intermediate_size", getattr(
        args, "shared_expert_intermediate_size", args.moe_intermediate_size))
    tokens = _positive_int("local_tokens", local_tokens)

    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w_up = torch.randn(2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w_down = torch.randn(hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    w_gate = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16) * 0.02

    def run():
        up = F.linear(x, w_up)
        act = torch.empty(tokens, inter, device="cuda", dtype=torch.bfloat16)
        torch.ops._C.silu_and_mul(act, up)
        down = F.linear(act, w_down)
        gate = torch.sigmoid(F.linear(x, w_gate))
        return down * gate

    return benchmark_gpu_region(run)
