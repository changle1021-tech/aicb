"""Qwen3-Next attention-chain shapes and isolated device profiling.

Shape helpers are CPU-only. CUDA dependencies are imported only when an
explicit profiling function is called.
"""

import types

from workload_generator.mocked_model.inference.qwen3_next_projections import (
    benchmark_gpu_region,
    projection_shapes,
)

CONV_BLOCK_M = 8


def gdn_conv_dim(args):
    """Channels of the GDN causal conv: q, k (key heads) and v (value heads)."""
    shapes = projection_shapes(args)
    return 2 * shapes.gdn_key_size + shapes.gdn_value_size


def conv_prefill_metadata(num_tokens, device="cuda"):
    """Precomputed Triton launch metadata for one sequence (graph-capturable)."""
    import torch

    count = (num_tokens + CONV_BLOCK_M - 1) // CONV_BLOCK_M
    batch_ptr = torch.zeros(count, device=device, dtype=torch.int32)
    offsets = torch.arange(count, device=device, dtype=torch.int32)
    entry = {"tot": count, "mlist": None, "mlist_len": count, "offsetlist": None,
             "batch_ptr": batch_ptr, "token_chunk_offset_ptr": offsets}
    return types.SimpleNamespace(batch_ptr=batch_ptr, token_chunk_offset_ptr=offsets,
                                 nums_dict={CONV_BLOCK_M: entry})


def _ensure_triton_allocator():
    import torch
    import triton

    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )


def benchmark_gdn_conv(args, phase, tokens):
    """Decode: causal_conv1d_update over `tokens` states. Prefill: one sequence."""
    import torch
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn, causal_conv1d_update)

    dim = gdn_conv_dim(args)
    width = args.linear_conv_kernel_dim
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    if phase == "decode":
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(tokens, dim, width - 1, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(tokens, device="cuda", dtype=torch.int32)
        return benchmark_gpu_region(lambda: causal_conv1d_update(
            x, state, weight, None, activation="silu", conv_state_indices=indices))

    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16).T
    state = torch.zeros(1, width - 1, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    query_start_loc = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    cache_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    has_initial_state = torch.zeros(1, device="cuda", dtype=torch.bool)
    metadata = conv_prefill_metadata(tokens)
    return benchmark_gpu_region(lambda: causal_conv1d_fn(
        x, weight, None, state, query_start_loc, cache_indices=cache_indices,
        has_initial_state=has_initial_state, activation="silu", metadata=metadata))


def benchmark_gdn_core(args, phase, batch, tokens):
    """Decode: fused recurrent rule per token. Prefill: chunked delta rule."""
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.layers.fla.ops import (
        chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)

    _ensure_triton_allocator()
    tp = args.tensor_model_parallel_size
    h, hv = args.linear_num_key_heads // tp, args.linear_num_value_heads // tp
    k_dim, v_dim = args.linear_key_head_dim, args.linear_value_head_dim
    seq_batch, seq_len = (batch, 1) if phase == "decode" else (1, tokens)

    q = torch.randn(seq_batch, seq_len, h, k_dim, device="cuda", dtype=torch.bfloat16)
    k = F.normalize(torch.randn_like(q), p=2, dim=-1)
    v = torch.randn(seq_batch, seq_len, hv, v_dim, device="cuda", dtype=torch.bfloat16)
    g = F.logsigmoid(torch.rand(seq_batch, seq_len, hv, device="cuda", dtype=torch.float32))
    beta = torch.rand(seq_batch, seq_len, hv, device="cuda", dtype=torch.bfloat16).sigmoid()
    state = torch.randn(seq_batch, hv, k_dim, v_dim, device="cuda", dtype=torch.bfloat16)

    if phase == "decode":
        indices = torch.arange(seq_batch, device="cuda", dtype=torch.int32)
        return benchmark_gpu_region(lambda: fused_recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state=state, inplace_final_state=True,
            ssm_state_indices=indices, use_qk_l2norm_in_kernel=True))

    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    return benchmark_gpu_region(lambda: chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
        cu_seqlens=cu_seqlens, head_first=False, use_qk_l2norm_in_kernel=True))


def benchmark_kv_cache_save(args, tokens, block_size=32):
    """Write this step's K/V into the paged cache (reshape_and_cache_flash)."""
    import torch
    from vllm import _custom_ops as ops

    shapes = projection_shapes(args)
    heads, head_dim = shapes.attention_kv_heads, args.head_dim
    num_blocks = (tokens + block_size - 1) // block_size + 1
    key = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key_cache = torch.zeros(num_blocks, block_size, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    return benchmark_gpu_region(lambda: ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slots, "auto", scale, scale))
