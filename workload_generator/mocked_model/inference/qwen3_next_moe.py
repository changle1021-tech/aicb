"""Qwen3-Next MoE workload contracts and isolated device profiling.

Load helpers are CPU-only. CUDA dependencies are imported only when an
explicit profiling function is called.
"""

import math
import random

from workload_generator.mocked_model.inference.qwen3_next_projections import (
    benchmark_component,
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
    generator = torch.Generator(device="cuda").manual_seed(getattr(args, "profile_seed", 42))
    topk_ids = sample_routing(tokens, num_experts, topk, generator=generator)
    topk_weights = torch.full((tokens, topk), 1.0 / topk, device="cuda", dtype=torch.float32)
    expert_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
    expert_map[:local_experts] = torch.arange(local_experts, device="cuda", dtype=torch.int32)

    def run():
        fused_experts(x, w1, w2, topk_weights, topk_ids,
                      global_num_experts=num_experts, expert_map=expert_map)

    return benchmark_component(args, run)


def _benchmark_routed_experts_fp8(args, local_tokens, dp, ep, local_experts):
    from utils.deepgemm_utils import bench_contiguous, bench_masked

    hidden, inter = args.hidden_size, args.moe_intermediate_size
    mean = expected_local_routes(local_tokens, dp, args.num_experts_per_tok, ep) / local_experts
    loads = sample_expert_loads(mean, local_experts, random.Random(getattr(args, "profile_seed", 42)))
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

    return benchmark_component(args, run)


def benchmark_moe_critical_compute(args, local_tokens, route_ids, ep_rank, samples=12):
    """Replay one layer's measured routes and native shared-stream dependency.

    Returns compute seconds for this EP shard. Communication is measured by a
    separate four-rank AG/RS benchmark; max across EP shards is taken per layer,
    then samples are averaged. Never sum the four ranks' elapsed times.
    """
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
    from workload_generator.mocked_model.inference.qwen3_next_projections import benchmark_gpu_region

    dp, ep, local_experts = moe_parallel_layout(args)
    if args.tensor_model_parallel_size != 1:
        raise ValueError("Measured MoE critical compute currently requires TP1")
    tokens = gathered_tokens(local_tokens, dp)
    hidden, inter, topk = args.hidden_size, args.moe_intermediate_size, args.num_experts_per_tok
    route_ids = route_ids.to(device="cuda", dtype=torch.int64)
    if tuple(route_ids.shape) != (tokens, topk):
        raise ValueError("Routing corpus shape differs from padded DP token layout")
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    local_x = x[:local_tokens]
    w1 = torch.randn(local_experts,2*inter,hidden,device="cuda",dtype=torch.bfloat16)*.02
    w2 = torch.randn(local_experts,hidden,inter,device="cuda",dtype=torch.bfloat16)*.02
    router_w = torch.randn(args.num_experts,hidden,device="cuda",dtype=torch.bfloat16)*.02
    logits = torch.full((tokens,args.num_experts),-10.,device="cuda",dtype=torch.bfloat16)
    logits.scatter_(1,route_ids,10.)
    expert_map = torch.full((args.num_experts,),-1,device="cuda",dtype=torch.int32)
    offset = ep_rank*local_experts
    expert_map[offset:offset+local_experts] = torch.arange(local_experts,device="cuda",dtype=torch.int32)
    shared_inter = args.shared_expert_intermediate_size
    up_w = torch.randn(2*shared_inter,hidden,device="cuda",dtype=torch.bfloat16)*.02
    down_w = torch.randn(hidden,shared_inter,device="cuda",dtype=torch.bfloat16)*.02
    gate_w = torch.randn(1,hidden,device="cuda",dtype=torch.bfloat16)*.02
    overlap = local_tokens <= getattr(args,"shared_expert_stream_token_threshold",256)
    stream = torch.cuda.Stream() if overlap else None
    def shared(z):
        up = F.linear(z,up_w)
        act = torch.empty(local_tokens,shared_inter,device="cuda",dtype=torch.bfloat16)
        torch.ops._C.silu_and_mul(act,up)
        out = F.linear(act,down_w)
        return torch.sigmoid(F.linear(z,gate_w))*out
    def run():
        if overlap:
            z = local_x.clone()
            z.record_stream(stream)
            stream.wait_stream(torch.cuda.current_stream())
        F.linear(local_x,router_w)
        weights, ids, _ = fused_topk(x,logits,topk,renormalize=True)
        routed = fused_experts(x,w1,w2,weights,ids,global_num_experts=args.num_experts,expert_map=expert_map)
        if overlap:
            with torch.cuda.stream(stream):out = shared(z)
            torch.cuda.current_stream().wait_stream(stream)
        else:
            out = shared(local_x)
        # Reduction/scatter is excluded. The final local add is included.
        out + local_x
        return routed
    with torch.inference_mode():
        return benchmark_gpu_region(run,warmup=3,samples=samples,
                                    timing_mode="eager" if args.phase=="prefill" else "cuda_graph")


def critical_profile_identity():
    import inspect, hashlib
    from workload_generator.mocked_model.inference.qwen3_next_projections import benchmark_gpu_region
    return hashlib.sha256("".join(inspect.getsource(fn) for fn in (
        benchmark_moe_critical_compute, benchmark_gpu_region,
        moe_parallel_layout, gathered_tokens)).encode()).hexdigest()


def critical_profile_includes_router(args):
    """Native FusedMoE replay starts after the external router linear.

    The compute-only helper above includes the router. Scope is explicit so
    the workload writer neither loses that linear nor adds it twice.
    """
    import json
    from pathlib import Path
    data = json.loads(Path(args.moe_compute_profile).read_text())
    scope = data.get("time_scope_by_phase", {}).get(args.phase, "compute_only")
    if scope not in ("compute_only", "compute_and_ep_comm"):
        raise ValueError("Unknown Qwen MoE timing scope")
    included = data.get("router_linear_included_by_phase", {}).get(
        args.phase, scope == "compute_only")
    if not isinstance(included, bool):
        raise ValueError("MoE router-linear scope must be a boolean")
    return included


def load_critical_moe_profile(args):
    import json, math
    from pathlib import Path
    path = getattr(args,"moe_compute_profile",None)
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if data.get("schema") not in ("qwen_moe_critical_compute_v1", "qwen_moe_critical_compute_v2") or data.get("producer_sha256") != critical_profile_identity():
        raise ValueError("Stale or incompatible Qwen MoE critical-compute profile")
    if projection_precision(args) != "bf16":
        raise ValueError("Native Qwen MoE replay is measured in BF16")
    scope = data.get("time_scope_by_phase",{}).get(args.phase,"compute_only")
    if scope not in ("compute_only", "compute_and_ep_comm"):
        raise ValueError("Unknown Qwen MoE timing scope")
    if scope == "compute_and_ep_comm" and data.get("native_runtime_source_sha256") != native_moe_runtime_identity():
        raise ValueError("Native vLLM MoE runtime changed; regenerate the replay profile")
    expected = dict(world_size=args.world_size,tp=args.tensor_model_parallel_size,
                    pp=args.pipeline_model_parallel,ep=args.expert_model_parallel_size,
                    hidden_size=args.hidden_size,num_experts=args.num_experts,
                    topk=args.num_experts_per_tok,intermediate_size=args.moe_intermediate_size,
                    shared_intermediate_size=args.shared_expert_intermediate_size,precision="bf16")
    if data.get("configuration") != expected:
        raise ValueError("Qwen MoE profile parallelism/shape/precision differs")
    tokens = args.seq_length if args.phase=="prefill" else args.micro_batch
    rows = data["phases"][args.phase].get(str(tokens))
    if rows is None or set(rows) != {str(i) for i in range(args.num_hidden_layers)}:
        raise ValueError("Qwen MoE profile is missing measured tokens or layers")
    if any(not math.isfinite(v) or v<=0 for v in rows.values()):
        raise ValueError("Invalid Qwen MoE compute times")
    return {int(k):v for k,v in rows.items()}


def native_moe_runtime_identity():
    """Native module/collective sources used by the four-rank replay producer."""
    import hashlib
    import vllm
    from pathlib import Path
    root = Path(vllm.__file__).parent
    paths = (
        "model_executor/models/qwen3_next.py", "model_executor/models/qwen2_moe.py",
        "model_executor/layers/fused_moe/layer.py", "model_executor/layers/fused_moe/fused_moe.py",
        "model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
        "distributed/parallel_state.py", "distributed/device_communicators/cuda_communicator.py",
        "distributed/device_communicators/all2all.py")
    return hashlib.sha256(b"".join((root/name).read_bytes() for name in paths)).hexdigest()
