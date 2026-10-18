"""Qwen3-Next attention-chain shapes and isolated device profiling.

Shape helpers are CPU-only. CUDA dependencies are imported only when an
explicit profiling function is called.
"""

import types

from workload_generator.mocked_model.inference.qwen3_next_projections import (
    benchmark_component,
    projection_precision,
    projection_shapes,
)

CONV_BLOCK_M = 8


def hybrid_cache_block_size(args, alignment=16):
    """Match the hybrid manager's attention page to one BF16 GDN state page.

    The target model uses BF16 conv and SSM cache. Round to FlashAttention's
    token-block alignment; at TP1 this is 544, not the ordinary default 16.
    """
    explicit = getattr(args, "block_size", None)
    if explicit is not None:
        if explicit <= 0 or explicit % alignment:
            raise ValueError("block_size must be a positive multiple of {}".format(alignment))
        return explicit
    shapes = projection_shapes(args)
    tp = args.tensor_model_parallel_size
    conv_bytes = gdn_conv_dim(args) * (args.linear_conv_kernel_dim - 1) * 2
    state_bytes = (args.linear_num_value_heads // tp) * args.linear_key_head_dim * args.linear_value_head_dim * 2
    kv_token_bytes = 2 * shapes.attention_kv_heads * args.head_dim * 2
    page_tokens = (conv_bytes + state_bytes + kv_token_bytes - 1) // kv_token_bytes
    return (page_tokens + alignment - 1) // alignment * alignment


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
        return benchmark_component(args, lambda: causal_conv1d_update(
            x, state, weight, None, activation="silu", conv_state_indices=indices))

    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16).T
    state = torch.zeros(1, width - 1, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    query_start_loc = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    cache_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    has_initial_state = torch.zeros(1, device="cuda", dtype=torch.bool)
    metadata = conv_prefill_metadata(tokens)
    return benchmark_component(args, lambda: causal_conv1d_fn(
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
        return benchmark_component(args, lambda: fused_recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state=state, inplace_final_state=True,
            ssm_state_indices=indices, use_qk_l2norm_in_kernel=True))

    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    return benchmark_component(args, lambda: chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
        cu_seqlens=cu_seqlens, head_first=False, use_qk_l2norm_in_kernel=True))


def benchmark_kv_cache_save(args, tokens, block_size=None):
    """Write this step's K/V into the paged cache (reshape_and_cache_flash)."""
    import torch
    from vllm import _custom_ops as ops

    block_size = block_size or hybrid_cache_block_size(args)
    shapes = projection_shapes(args)
    heads, head_dim = shapes.attention_kv_heads, args.head_dim
    num_blocks = (tokens + block_size - 1) // block_size + 1
    key = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key_cache = torch.zeros(num_blocks, block_size, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    return benchmark_component(args, lambda: ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slots, "auto", scale, scale))


def benchmark_flash_attention(args, kv_lens, q_lens):
    """The paged FlashAttention path selected by the vLLM H100 backend.

    Plan/scheduler metadata belongs to input preparation, outside the model
    region, just as in FlashAttentionMetadataBuilder.build().
    """
    import torch
    from vllm import envs
    from vllm.attention.utils.fa_utils import (
        flash_attn_varlen_func, get_flash_attn_version, get_scheduler_metadata)

    shapes = projection_shapes(args)
    block_size = hybrid_cache_block_size(args)
    heads, kv_heads, head_dim = shapes.attention_query_heads, shapes.attention_kv_heads, args.head_dim
    num_blocks = sum((n + block_size - 1) // block_size for n in kv_lens)
    max_blocks = max((n + block_size - 1) // block_size for n in kv_lens)
    cache = torch.randn(2, num_blocks, block_size, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    query = torch.randn(sum(q_lens), heads, head_dim, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(query)
    block_table = torch.zeros(len(q_lens), max_blocks, device="cuda", dtype=torch.int32)
    offset = 0
    for i, length in enumerate(kv_lens):
        count = (length + block_size - 1) // block_size
        block_table[i, :count] = torch.arange(offset, offset + count, device="cuda", dtype=torch.int32)
        offset += count
    cu_q = torch.tensor([0] + list(__import__('itertools').accumulate(q_lens)), device="cuda", dtype=torch.int32)
    lengths = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    version = get_flash_attn_version()
    splits = envs.VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH if args.phase == "decode" else 0
    scheduler = None
    if version == 3:
        scheduler = get_scheduler_metadata(
            batch_size=len(q_lens), max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            num_heads_q=heads, num_heads_kv=kv_heads, headdim=head_dim,
            cache_seqlens=lengths, qkv_dtype=torch.bfloat16, cu_seqlens_q=cu_q,
            page_size=block_size, causal=True, window_size=(-1, -1), num_splits=splits)

    def run():
        return flash_attn_varlen_func(
            q=query, k=cache[0], v=cache[1], out=output,
            cu_seqlens_q=cu_q, max_seqlen_q=max(q_lens), seqused_k=lengths,
            max_seqlen_k=max(kv_lens), softmax_scale=head_dim ** -.5,
            causal=True, alibi_slopes=None, window_size=(-1, -1), softcap=0.0,
            block_table=block_table, scheduler_metadata=scheduler,
            fa_version=version, num_splits=splits)
    return benchmark_component(args, run)


def _make_gdn_native_core(args, phase, tokens):
    """Profile vLLM's complete GDN custom op, including cache state copies.

    The kernel-only chunk helper omits Q/K/V contiguous copies, initial-state
    gathering/zeroing, cache writeback and output copies. Prefill executes this
    Python custom-op body eagerly; decode captures the identical body in Graph.
    """
    import torch
    from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.forward_context import ForwardContext, override_forward_context

    _ensure_triton_allocator()
    tp = args.tensor_model_parallel_size
    obj = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(obj)
    obj.tp_size = tp
    obj.num_k_heads, obj.num_v_heads = args.linear_num_key_heads, args.linear_num_value_heads
    obj.head_k_dim, obj.head_v_dim = args.linear_key_head_dim, args.linear_value_head_dim
    obj.key_dim = obj.num_k_heads * obj.head_k_dim
    obj.value_dim = obj.num_v_heads * obj.head_v_dim
    obj.prefix = "aicb.gdn"
    obj.activation = "silu"
    dim = gdn_conv_dim(args)
    obj.conv1d = torch.nn.Conv1d(dim, dim, args.linear_conv_kernel_dim,
                               groups=dim, bias=False, device="cuda", dtype=torch.bfloat16)
    heads = obj.num_v_heads // tp
    obj.A_log = torch.nn.Parameter(torch.zeros(heads, device="cuda", dtype=torch.float32))
    obj.dt_bias = torch.nn.Parameter(torch.ones(heads, device="cuda", dtype=torch.float32))
    sequences = tokens if phase == "decode" else 1
    obj.kv_cache = [(torch.zeros(sequences, args.linear_conv_kernel_dim-1, dim,
                                device="cuda", dtype=torch.bfloat16),
                     torch.zeros(sequences, heads, obj.head_k_dim, obj.head_v_dim,
                                 device="cuda", dtype=torch.bfloat16))]
    conv = conv_prefill_metadata(tokens)
    metadata = GDNAttentionMetadata(
        num_prefills=int(phase == "prefill"), num_prefill_tokens=tokens if phase == "prefill" else 0,
        num_decodes=sequences if phase == "decode" else 0,
        num_decode_tokens=tokens if phase == "decode" else 0,
        num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=tokens,
        has_initial_state=torch.full((sequences,), bool(getattr(args,"context_length",0)), device="cuda",dtype=torch.bool),
        non_spec_query_start_loc=(torch.arange(tokens+1,device="cuda",dtype=torch.int32) if phase=="decode"
                                  else torch.tensor([0,tokens],device="cuda",dtype=torch.int32)),
        non_spec_state_indices_tensor=torch.arange(sequences,device="cuda",dtype=torch.int32),
        nums_dict=conv.nums_dict, batch_ptr=conv.batch_ptr,
        token_chunk_offset_ptr=conv.token_chunk_offset_ptr)
    context = ForwardContext(no_compile_layers={obj.prefix:obj},
                             attn_metadata={obj.prefix:metadata},virtual_engine=0)
    mixed = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(tokens, heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    output = torch.zeros(tokens,heads,obj.head_v_dim,device="cuda",dtype=torch.bfloat16)
    return obj, context, (mixed, b, a, output)


def benchmark_gdn_native_core(args, phase, tokens):
    import torch
    from vllm.forward_context import override_forward_context
    obj, context, inputs = _make_gdn_native_core(args, phase, tokens)
    with torch.inference_mode(), override_forward_context(context):
        return benchmark_component(args, lambda: obj._forward_core(*inputs))


def benchmark_gdn_compiled_chain(args, tokens):
    """Prefill's two compiled regions around the native GDN custom op.

    Include QKV ordering/materialization, zeroing the output, native gated RMS
    norm, projections and both decoder norms. Isolated GEMMs omit this launch
    and allocation work. Compilation/warmup are outside the timed callback.
    """
    if projection_precision(args) != "bf16":
        raise ValueError("The native GDN prefill chain is measured for BF16; an FP8 projection profile cannot label this chain")
    import torch
    from vllm.forward_context import override_forward_context
    from vllm.model_executor.layers.layernorm import RMSNormGated, GemmaRMSNorm

    class TupleLinear(torch.nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(outputs, inputs, device="cuda", dtype=torch.bfloat16)*.02,
                                             requires_grad=False)
        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight), None

    obj, context, _ = _make_gdn_native_core(args, "prefill", tokens)
    shapes = projection_shapes(args)
    obj.in_proj_qkvz = TupleLinear(args.hidden_size, shapes.gdn_qkvz_size)
    obj.in_proj_ba = TupleLinear(args.hidden_size, shapes.gdn_ba_size)
    obj.out_proj = TupleLinear(shapes.gdn_value_size, args.hidden_size)
    obj.norm = RMSNormGated(args.linear_value_head_dim, eps=args.rms_norm_eps,
                           group_size=None, norm_before_gate=True,
                           device="cuda", dtype=torch.bfloat16)
    # vLLM compiles the native expression into its surrounding FX region.
    # The FLA autograd wrapper is a different path and cannot be traced here.
    obj.norm._forward_method = obj.norm.forward_native
    pre = GemmaRMSNorm(args.hidden_size, args.rms_norm_eps).to(device="cuda", dtype=torch.bfloat16)
    post = GemmaRMSNorm(args.hidden_size, args.rms_norm_eps).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    heads = args.linear_num_value_heads // args.tensor_model_parallel_size

    def before_core():
        hidden, res = pre.forward_native(x, residual)
        qkv, _ = obj.in_proj_qkvz(hidden)
        ba, _ = obj.in_proj_ba(hidden)
        q, k, v, z, b, a = obj.fix_query_key_value_ordering(qkv, ba)
        mixed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        core = torch.zeros(tokens, heads, args.linear_value_head_dim, device="cuda", dtype=torch.bfloat16)
        return mixed, b, a, core, z, res

    def after_core(core, z, res):
        attn = obj.norm(core.reshape(-1, args.linear_value_head_dim),
                        z.reshape(-1, args.linear_value_head_dim)).reshape(tokens, shapes.gdn_value_size)
        out, _ = obj.out_proj(attn)
        return post.forward_native(out, res)

    with torch.inference_mode(), override_forward_context(context):
        before = torch.compile(before_core, fullgraph=True, dynamic=True)
        after = torch.compile(after_core, fullgraph=True, dynamic=True)
        def run():
            mixed, b, a, core, z, res = before()
            torch.ops.vllm.gdn_attention_core(mixed, b, a, core, obj.prefix)
            return after(core, z, res)
        return benchmark_component(args, run)
