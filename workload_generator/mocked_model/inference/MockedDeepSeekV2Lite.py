"""Mocked inference model for DeepSeek-V2-Lite/DeepSeek-V2-Lite-Chat.

This module intentionally lives beside, rather than modifying, MockedDeepSeek.
DeepSeek-V2-Lite differs from the 671B model in two important ways:

* q_lora_rank is null, so Q is projected directly from the hidden state.
* the dense FFN and routed/shared experts use different intermediate sizes.
"""

import workload_generator.mocked_model.inference.MockedDeepSeek as MockedDeepSeek
from workload_generator.mocked_model.MockedModel import (
    MockedModel,
    MockedParam,
    MockedParamsBase,
)
from log_analyzer.log import Workload, LogItem


def _first_attr(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


class DeepSeekV2LiteParams(MockedParamsBase):
    """Load official-style V2-Lite fields and expose AICB-compatible aliases."""

    def __init__(self, config_file=None, args=None):
        super().__init__("DeepSeek-V2-Lite-Chat", "DeepSeek", config_file, args)

        self.num_layers = _first_attr(self, "num_layers", "num_hidden_layers")
        self.dense_layer = _first_attr(
            self, "dense_layer", "first_k_dense_replace", default=1
        )
        self.head_num = _first_attr(
            self, "head_num", "num_attention_heads"
        )
        self.d_q = _first_attr(self, "d_q", "qk_nope_head_dim")
        self.d_r = _first_attr(self, "d_r", "qk_rope_head_dim")
        self.d_kv = _first_attr(self, "d_kv", "v_head_dim")
        self.d_kv_c = _first_attr(self, "d_kv_c", "kv_lora_rank")
        self.q_lora_rank = _first_attr(self, "q_lora_rank", default=None)

        self.dense_intermediate_size = _first_attr(
            self,
            "dense_intermediate_size",
            "intermediate_size",
            default=_first_attr(self, "expert_dim"),
        )
        self.moe_intermediate_size = _first_attr(
            self,
            "moe_intermediate_size",
            default=_first_attr(self, "expert_dim"),
        )

        self.router_expert = _first_attr(
            self, "router_expert", "n_routed_experts"
        )
        self.duped_expert = _first_attr(self, "duped_expert", default=0)
        self.num_experts = self.router_expert + self.duped_expert
        self.shared_experts = _first_attr(
            self, "shared_experts", "n_shared_experts", default=0
        )
        self.moe_router_topk = _first_attr(
            self, "moe_router_topk", "num_experts_per_tok"
        )

        self.computation_enable = True
        self.add_bias_linear = False

        required = {
            "num_layers": self.num_layers,
            "head_num": self.head_num,
            "d_q": self.d_q,
            "d_r": self.d_r,
            "d_kv": self.d_kv,
            "d_kv_c": self.d_kv_c,
            "dense_intermediate_size": self.dense_intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "router_expert": self.router_expert,
            "moe_router_topk": self.moe_router_topk,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "DeepSeek-V2-Lite config is missing required fields: "
                + ", ".join(missing)
            )

        if self.q_lora_rank is not None:
            raise ValueError(
                "MockedDeepSeekV2Lite only supports q_lora_rank=null; "
                "use MockedDeepSeek for Q-LoRA models."
            )


class DeepSeekV2LiteAttention(MockedDeepSeek.DeepSeekAttention):
    """V2-Lite MLA projections for the no-Q-LoRA configuration."""

    def __init__(
        self,
        hidden_size,
        tp,
        head_num,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        seq_len,
        batch_size,
        layer_id,
        computation_enable=False,
        add_bias_linear=False,
        elem_size=2,
    ):
        # Do not call the 671B initializer: it always creates Q-LoRA matrices.
        self.layer_id = layer_id
        self.name = "attention_layer"
        self.hidden_size = hidden_size
        self.tp = tp
        self.n_heads = head_num // tp
        self.q_lora = None
        self.kv_lora = kv_lora_rank
        self.rope_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim
        self.kv_head_dim = v_head_dim

        # q_proj: hidden -> all local Q heads (no Q-LoRA compression/decompression).
        self.q_proj = MockedDeepSeek.DeepSeekColumnLinear(
            hidden_size,
            head_num * (qk_nope_head_dim + qk_rope_head_dim),
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention_q",
            computation_enable,
            add_bias_linear=add_bias_linear,
            elem_size=elem_size,
            name="q_proj",
        )

        # kv_a_proj_with_mqa: hidden -> compressed KV latent + decoupled RoPE key.
        self.kv_a_proj_with_mqa = MockedDeepSeek.DeepSeekColumnLinear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            1,
            seq_len,
            batch_size,
            layer_id,
            "attention_kv_a",
            computation_enable,
            add_bias_linear=add_bias_linear,
            elem_size=elem_size,
            name="kv_a_proj_with_mqa",
        )

        # kv_b_proj: compressed KV latent -> per-head non-RoPE K and V.
        self.kv_b_proj = MockedDeepSeek.DeepSeekColumnLinear(
            kv_lora_rank,
            head_num * (qk_nope_head_dim + v_head_dim),
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention_kv_b",
            computation_enable,
            add_bias_linear=add_bias_linear,
            elem_size=elem_size,
            name="kv_b_proj",
        )

        self.wo = MockedDeepSeek.DeepSeekRowLinear(
            head_num * v_head_dim,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention",
            computation_enable,
            add_bias_linear=add_bias_linear,
            elem_size=elem_size,
            name="o_proj",
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.q_proj.forward())
        workloads.extend(self.kv_a_proj_with_mqa.forward())
        workloads.extend(self.kv_b_proj.forward())
        workloads.extend(self.wo.forward())
        assert all(isinstance(item, LogItem) for item in workloads.workload)
        return workloads


class DeepSeekV2LiteMOE(MockedDeepSeek.DeepSeekMOE):
    """V2-Lite MoE with one fused shared-expert MLP."""

    def __init__(
        self,
        hidden_size,
        total_experts,
        expert_model_parallel_size,
        experts_topk,
        expert_dim,
        tp,
        seq_len,
        batch_size,
        layer_id,
        shared_experts_cnt,
        computation_enable,
        add_bias_linear,
        elem_size,
    ):
        self.tp = tp
        self.name = "sparse_moelayer"
        self.layer_id = layer_id
        self.expert_model_parallel_size = expert_model_parallel_size
        self.topk = experts_topk
        self.seq_length = seq_len
        self.num_experts = total_experts
        self.micro_batch = batch_size
        self.hidden_size = hidden_size
        self.w_gate = MockedParam(
            (self.num_experts, hidden_size), elem_size, name="moe_gate"
        )

        if total_experts % expert_model_parallel_size != 0:
            raise ValueError(
                "DeepSeek-V2-Lite total experts must be divisible by EP size"
            )
        num_local_experts = total_experts // expert_model_parallel_size
        self.expert = MockedDeepSeek.DeepSeekMLP(
            hidden_size,
            num_local_experts * expert_dim,
            1,
            seq_len,
            batch_size,
            layer_id,
            computation_enable,
            add_bias_linear=add_bias_linear,
            elem_size=elem_size,
            name="moe_expert",
        )

        # The reference V2 implementation concatenates all shared experts into
        # one MLP whose intermediate size is n_shared_experts * moe_inter_dim.
        self.shared_experts = []
        if shared_experts_cnt:
            self.shared_experts.append(
                MockedDeepSeek.DeepSeekMLP(
                    hidden_size,
                    shared_experts_cnt * expert_dim,
                    1,
                    seq_len,
                    batch_size,
                    layer_id,
                    computation_enable,
                    add_bias_linear=add_bias_linear,
                    elem_size=elem_size,
                    name="shared_experts",
                )
            )


class DeepSeekV2LiteTransformerLayer(MockedModel):
    def __init__(self, config, layer_id):
        self.id = layer_id
        self.dense_layer = config.dense_layer
        self.attention = DeepSeekV2LiteAttention(
            config.hidden_size,
            config.tensor_model_parallel_size,
            config.head_num,
            config.d_kv_c,
            config.d_r,
            config.d_q,
            config.d_kv,
            config.seq_length,
            config.micro_batch,
            layer_id,
            computation_enable=config.computation_enable,
            add_bias_linear=config.add_bias_linear,
            elem_size=2,
        )

        if layer_id < config.dense_layer:
            self.mlp = MockedDeepSeek.DeepSeekMLP(
                config.hidden_size,
                config.dense_intermediate_size,
                config.tensor_model_parallel_size,
                config.seq_length,
                config.micro_batch,
                layer_id,
                config.computation_enable,
                add_bias_linear=config.add_bias_linear,
                elem_size=2,
                name="dense_mlp",
            )
        else:
            self.mlp = DeepSeekV2LiteMOE(
                config.hidden_size,
                config.num_experts,
                config.expert_model_parallel_size,
                config.moe_router_topk,
                config.moe_intermediate_size,
                config.tensor_model_parallel_size,
                config.seq_length,
                config.micro_batch,
                layer_id,
                config.shared_experts,
                config.computation_enable,
                config.add_bias_linear,
                elem_size=2,
            )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.attention.forward())
        workloads.extend(self.mlp.forward())
        assert all(isinstance(item, LogItem) for item in workloads.workload)
        return workloads


class DeepSeekV2LiteModel(MockedModel):
    def __init__(self, config):
        self.layers = [
            DeepSeekV2LiteTransformerLayer(config, layer_id)
            for layer_id in range(config.num_layers)
        ]
        self.final = MockedDeepSeek.DeepSeekColumnLinear(
            config.hidden_size,
            config.vocab_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
            1,
            "final",
            computation_enable=config.computation_enable,
            add_bias_linear=config.add_bias_linear,
            elem_size=2,
            name="lm_head",
        )

    def forward(self, config):
        workloads = Workload()
        for layer in self.layers:
            workloads.extend(layer.forward())
        workloads.extend(self.final.forward())
        return workloads
