import torch
# torch.set_default_device("cuda")
import workload_generator.mocked_model.inference.MockedQwen3Next as MockedQwen3Next
from workload_generator.mocked_model.MockedModel import InferencePhase
from utils.utils import *
from utils.deepgemm_utils import *
import torch.nn.functional as F
import random
from typing import Optional, Union, Tuple
from workload_generator.mocked_model.inference.qwen3_next_projections import (
    projection_shapes, projection_precision, benchmark_projection,
    benchmark_input_quantization, benchmark_attention_output_gate,
    write_projection_metadata, benchmark_gpu_region, benchmark_component,
)
from workload_generator.mocked_model.inference.qwen3_next_moe import (
    moe_parallel_layout, gathered_tokens, benchmark_routed_experts,
    benchmark_shared_expert,
    critical_profile_includes_router,
)
from workload_generator.mocked_model.inference.qwen3_next_attention import (
    benchmark_gdn_conv, benchmark_gdn_core, benchmark_kv_cache_save, benchmark_flash_attention, benchmark_gdn_native_core, benchmark_gdn_compiled_chain,
)
# import triton
import numpy as np
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm import _custom_ops as vllm_ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
import flashinfer
from vllm.model_executor.layers.fla.ops import (RMSNormGated, fused_recurrent_gated_delta_rule)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

def GemmaRMSNormTest(
        hidden_size,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: Optional[torch.Tensor], timing_mode="cuda_graph"):
    gemmaRMSNormImpl = GemmaRMSNorm(hidden_size, variance_epsilon)
    gemmaRMSNormImpl = gemmaRMSNormImpl.to(x.device)
    def test_func():
        gemmaRMSNormImpl.forward_cuda(x, residual)
    t = benchmark_gpu_region(test_func, timing_mode=timing_mode)
    return t

# modified from test_causal_conv1d.py
def test_causal_conv1d_update(batch, dim, width, seqlen, has_bias, silu_activation,
                              itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    current_platform.seed_everything(0)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    x_ref = x.clone()
    conv_state = torch.randn(batch, dim, width - 1, device=device, dtype=itype)

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_update(x,
                               conv_state,
                               weight,
                               bias,
                               activation=activation)
    return out

# modified from qwen3_next.py
# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr = 1.0,
    threshold: tl.constexpr = 20.0,
    BLK_HEADS: tl.constexpr = 8
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = (1 / beta) * tl.log(1 + tl.exp(beta * x))
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)


class Qwen3NextGatedDeltaNet(torch.nn.Module):
    def __init__(self, args=None):
        super(Qwen3NextGatedDeltaNet, self).__init__()
        self.args = args
        self.tp = self.args.tensor_model_parallel_size
        self.shapes = projection_shapes(self.args)

        self.norm = RMSNormGated(
            self.args.linear_value_head_dim,
            eps=self.args.rms_norm_eps,
            group_size=None,
            norm_before_gate=True,
            device='cuda', dtype=torch.bfloat16
        )

    def _attn_quant(self, qk_or_o, m):
        width = self.shapes.hidden_size if qk_or_o else self.shapes.gdn_value_size
        return benchmark_input_quantization(self.args, m, width)

    
    def _attn_proj(self, k, n, m):
        return benchmark_projection(self.args, m, k, n)


    def _causal_conv(self, m):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        return benchmark_gdn_conv(self.args, phase, m)

    def _fused_gdn_gating(self, m):
        from vllm.model_executor.models.qwen3_next import fused_gdn_gating
        heads = self.args.linear_num_value_heads // self.tp
        A_log = torch.zeros(heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.ones(heads, device="cuda", dtype=torch.float32)
        a = torch.randn(m, heads, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        return benchmark_component(self.args, lambda: fused_gdn_gating(A_log, a, b, dt_bias))

    def _fused_gdn_core(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        return benchmark_gdn_core(self.args, phase, self.args.micro_batch, self.args.seq_length)
        
    def _norm(self, m):
        k = self.args.linear_num_value_heads // self.tp
        n = self.args.linear_value_head_dim

        core_attn_out = torch.randn(m*k, n, device='cuda', dtype=torch.bfloat16)
        z = torch.randn(m*k, n, device='cuda', dtype=torch.bfloat16)

        def test_func():
            self.norm(core_attn_out, z)
        t = benchmark_component(self.args, test_func)
        return t
    
    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.args.micro_batch
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length

        if phase == InferencePhase.PREFILL.value:
            return 0.0, 0.0, benchmark_gdn_compiled_chain(self.args, m), 0.0

        # 1. qkv
        qkv_quant = self._attn_quant(True, m)
        
        qkvz_proj_k = self.args.hidden_size
        qkvz_proj_n = self.shapes.gdn_qkvz_size
        qkvz_proj = self._attn_proj(qkvz_proj_k, qkvz_proj_n, m)

        ba_proj_k = self.args.hidden_size
        ba_proj_n = self.shapes.gdn_ba_size
        ba_proj = self._attn_proj(ba_proj_k, ba_proj_n, m)
        
        attn_qkv = qkv_quant+qkvz_proj+ba_proj

        # 2. Causal Convolution
        attn_causal_conv = 0.0  # included in the native custom-op region

        # 3. Recurrent Attention
        gdn_gating = 0.0  # native custom op computes both g and beta
        #fused_recurrent_gated_delta_rule
        gdn_core = benchmark_gdn_native_core(self.args, phase, m)

        attn_gdn = gdn_gating + gdn_core

        # 4. output
        # attn_out_norm=RMSNormGated

        o_norm = self._norm(m)
        o_quant = self._attn_quant(False, m)
        o_proj_k = self.shapes.gdn_value_size
        o_proj_n = self.args.hidden_size
        o_proj = self._attn_proj(o_proj_k,o_proj_n, m)
        
        attn_o = o_norm + o_quant + o_proj

        return attn_qkv, attn_causal_conv, attn_gdn, attn_o

        
class Qwen3NextAttention(torch.nn.Module):
    def __init__(self, args=None):
        super(Qwen3NextAttention, self).__init__()
        self.args = args
        self.tp = self.args.tensor_model_parallel_size
        self.shapes = projection_shapes(self.args)
    
    def _attn_quant(self, qk_or_o, m):
        width = self.shapes.hidden_size if qk_or_o else self.shapes.attention_query_size
        return benchmark_input_quantization(self.args, m, width)


    def _attn_proj(self, qk_or_o, m):
        if qk_or_o:
            k, n = self.shapes.hidden_size, self.shapes.attention_qkv_size
        else:
            k, n = self.shapes.attention_query_size, self.shapes.hidden_size
        return benchmark_projection(self.args, m, k, n)

    def _qk_norm(self, q_or_k, m):
        head_dim = self.args.head_dim
        if q_or_k:
            n = self.shapes.attention_query_size
        else:
            n = self.shapes.attention_kv_size
        x = torch.randn(m,n, device='cuda', dtype=torch.bfloat16)
        x = x.view(*x.shape[:-1], x.shape[-1] // head_dim, head_dim)
        residual = torch.rand_like(x, device='cuda', dtype=torch.bfloat16)
        return GemmaRMSNormTest(head_dim, self.args.rms_norm_eps, x, None,
                                timing_mode="eager" if self.args.phase == "prefill" else "cuda_graph")
            
    def _rotary_emb(self, m):
        head_dim = self.args.head_dim
        n_q = self.shapes.attention_query_size
        n_k = self.shapes.attention_kv_size

        q = torch.randn(m, n_q, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(m, n_k, device='cuda', dtype=torch.bfloat16)
        pos = torch.arange(0, m, device='cuda')
        rotary_emb = get_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=self.args.max_position_embeddings,
                base=self.args.rope_theta,
                rope_scaling=self.args.rope_scaling,
                partial_rotary_factor=self.args.partial_rotary_factor,
                dual_chunk_attention_config = getattr(self.args, 
                                        "dual_chunk_attention_config",
                                        None),
            )
        def test_func():
            rotary_emb(pos, q, k)

        t = benchmark_component(self.args, test_func)
        return t

    def _attn_core(self, kv_lens, q_lens):
        return benchmark_flash_attention(self.args, kv_lens, q_lens)

    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.args.micro_batch
            kv_lens = [self.args.seq_length] * m
            q_lens = [1] * m
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length
            kv_lens = [self.args.seq_length + getattr(self.args, "context_length", 0)]
            q_lens = [m]

        # qkv
        qkv_quant = self._attn_quant(True,m)
        qkv_proj = self._attn_proj(True,m)
        q_norm = self._qk_norm(True,m)
        k_norm = self._qk_norm(False,m)
        attn_qkv = qkv_quant + qkv_proj + q_norm + k_norm

        # rotary_embedding
        rotary_emb = self._rotary_emb(m)

        # core: write this step's K/V into the paged cache, then attend
        kv_cache_save = benchmark_kv_cache_save(self.args, m)
        attn = kv_cache_save + self._attn_core(kv_lens, q_lens)

        # o
        output_gate = benchmark_attention_output_gate(
            self.args, m, self.shapes.attention_query_size
        )
        o_quant = self._attn_quant(False,m)
        o_proj = self._attn_proj(False,m)
        attn_o = o_quant + o_proj

        return attn_qkv, rotary_emb, attn, attn_o, output_gate

class Qwen3NextSparseMoeBlock(torch.nn.Module):
    def __init__(self, args):
        super(Qwen3NextSparseMoeBlock, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.topk = args.num_experts_per_tok
        self.dp, self.ep, self.local_experts = moe_parallel_layout(args)

    def _route_gate(self, tokens):
        # ReplicatedLinear [hidden -> num_experts], BF16 like the served model
        x = torch.randn(tokens, self.hidden_size, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(self.num_experts, self.hidden_size, device='cuda', dtype=torch.bfloat16)
        return benchmark_component(self.args, lambda: F.linear(x, w))

    def _select_experts(self, tokens):
        # FusedMoE.select_experts
        hidden_states = torch.randn(tokens, self.hidden_size, device='cuda', dtype=torch.bfloat16)
        router_logits = torch.randn(tokens, self.num_experts, device='cuda', dtype=torch.bfloat16)

        def test_func():
            fused_topk(hidden_states=hidden_states,
                       gating_output=router_logits,
                       topk=self.topk,
                       renormalize=True)
        return benchmark_component(self.args, test_func)

    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            local_tokens = self.args.micro_batch
        elif phase == InferencePhase.PREFILL.value:
            local_tokens = self.args.seq_length

        if getattr(self.args,"moe_compute_profile",None):
            # Per-layer measured critical compute is installed by the CSV writer.
            router = (0.0 if critical_profile_includes_router(self.args)
                      else self._route_gate(local_tokens))
            return router, 0.0, 0.0, 0.0
        # Router linear runs locally; top-k runs after the DP all-gather.
        tokens = gathered_tokens(local_tokens, self.dp)
        route_gate = self._route_gate(local_tokens)
        route_select_experts = self._select_experts(tokens)
        routed_experts = benchmark_routed_experts(self.args, local_tokens)
        shared_expert = benchmark_shared_expert(self.args, local_tokens)

        return route_gate, route_select_experts, routed_experts, shared_expert


class Qwen3NextModel(torch.nn.Module):
    def __init__(self, args=None):
        super(Qwen3NextModel, self).__init__()
        self.time_list = {}
        self.args = args
        projection_precision(args)
        # self.Embedding = Qwen3Embedding(self.args)
        self.FullAttention = Qwen3NextAttention(self.args)
        self.GDN = Qwen3NextGatedDeltaNet(self.args)
        # self.MLP = Qwen3NextMLP(self.args) # TODO add MLP only
        self.MoE = Qwen3NextSparseMoeBlock(self.args)
        self.aiob_forward_loops = getattr(args, "aiob_forward_loops", 10)
    
    def _norm(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.args.micro_batch
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length
        hidden_size = self.args.hidden_size
        x = torch.randn(m,hidden_size, device='cuda', dtype=torch.bfloat16)
        residual = torch.randn(m,hidden_size, device='cuda', dtype=torch.bfloat16)
        return GemmaRMSNormTest(hidden_size, self.args.rms_norm_eps, x, residual,
                                timing_mode="eager" if self.args.phase == "prefill" else "cuda_graph")
    
    def forward(self):
        for i in range(self.aiob_forward_loops):
            # print(f'======================{i}======================')
            # 1. input_layernorm
            pre_norm = self._norm()
            self.time_list.setdefault("atten_norm", []).append(
                        {"time_gpu": pre_norm* 1e6}
                    )
            # 2.A. self_attn
            atten_qkv, atten_rotary_emb, atten_core, atten_o, atten_output_gate = self.FullAttention()
            self.time_list.setdefault("atten_qkv", []).append(
                        {"time_gpu": atten_qkv* 1e6}
                    )
            self.time_list.setdefault("atten_rotary_emb", []).append(
                        {"time_gpu": atten_rotary_emb* 1e6}
                    )
            self.time_list.setdefault("atten_flash", []).append(
                        {"time_gpu": atten_core* 1e6}
                    )
            self.time_list.setdefault("atten_o", []).append(
                        {"time_gpu": atten_o* 1e6}
                    )
            
            self.time_list.setdefault("atten_output_gate", []).append(
                {"time_gpu": atten_output_gate * 1e6}
            )

            # 2.B. linear_attn
            gdn_qkv, gdn_causal_conv, gdn_core, gdn_o = self.GDN()
            self.time_list.setdefault("gdn_qkv", []).append(
                    {"time_gpu": gdn_qkv * 1e6}
                )
            self.time_list.setdefault("gdn_causal_conv", []).append(
                        {"time_gpu": gdn_causal_conv * 1e6}
                    )
            self.time_list.setdefault("gdn_core", []).append(
                        {"time_gpu": gdn_core * 1e6}
                    )
            self.time_list.setdefault("gdn_o", []).append(
                        {"time_gpu": gdn_o * 1e6}
                    )

            # 3. post_attention_layernorm
            post_norm = self._norm()
            self.time_list.setdefault("moe_norm", []).append(
                        {"time_gpu": post_norm* 1e6}
                    )

            # 4. mlp
            moe_route_gate, moe_route_select_experts, moe_routed, moe_shared = self.MoE()
            self.time_list.setdefault("moe_route_gate", []).append(
                        {"time_gpu": moe_route_gate* 1e6}
                    )
            self.time_list.setdefault("moe_route_select_experts", []).append(
                        {"time_gpu": moe_route_select_experts* 1e6}
                    )
            self.time_list.setdefault("moe_expert_routed", []).append(
                        {"time_gpu": moe_routed* 1e6}
                    )
            self.time_list.setdefault("moe_shared_expert", []).append(
                        {"time_gpu": moe_shared* 1e6}
                    )
        
        print(self.time_list)
        result_dir = "./results/aiob_outputs"
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        filename = f"{self.args.model_name}_time_list_stats.txt"
        file_name = os.path.join(result_dir, filename)
        calculate_stats(self.time_list, file_name)
        filepath = write_time(self.time_list, self.args)
        process_all_keys(filepath)
        write_projection_metadata(filepath, self.args)
        return filepath
    

if __name__ == "__main__":
    args = MockedQwen3Next.Qwen3NextParams()
    x = torch.randint(0, args.vocab_size, (2, 1))
    model = Qwen3NextModel(args)
    filepath = model(x)
    print(filepath)
