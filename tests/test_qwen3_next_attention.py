"""CPU tests for the Qwen3-Next attention-chain shape contracts."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from workload_generator.mocked_model.inference.qwen3_next_attention import (
    CONV_BLOCK_M,
    gdn_conv_dim,
)


ROOT = Path(__file__).resolve().parents[1]
AIOB = ROOT / "workload_generator/mocked_model/inference/AiobQwen3Next.py"
ATTENTION_HELPERS = ROOT / "workload_generator/mocked_model/inference/qwen3_next_attention.py"


def function_node(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name)


def called_functions(node):
    names = []
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            func = item.func
            names.append(func.id if isinstance(func, ast.Name) else
                         func.attr if isinstance(func, ast.Attribute) else "")
    return names


def args(tp=1):
    values = json.loads((ROOT / "scripts/inference_configs/qwen3_next_default.json").read_text())
    values.update(tensor_model_parallel_size=tp)
    return SimpleNamespace(**values)


class AttentionShapeTests(unittest.TestCase):
    def test_conv_dim_matches_vllm(self):
        # vLLM conv_dim = key_dim * 2 + value_dim = 2*16*128 + 32*128 per TP rank
        self.assertEqual(gdn_conv_dim(args(1)), 8192)
        self.assertEqual(gdn_conv_dim(args(2)), 4096)
        self.assertEqual(gdn_conv_dim(args(4)), 2048)

    def test_conv_block_size_matches_kernel(self):
        self.assertEqual(CONV_BLOCK_M, 8)

    def test_prefill_paths_are_not_decode_kernels(self):
        source = AIOB.read_text()
        tree = ast.parse(source)
        gdn = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3NextGatedDeltaNet")
        conv = next(n for n in gdn.body if isinstance(n, ast.FunctionDef) and n.name == "_causal_conv")
        core = next(n for n in gdn.body if isinstance(n, ast.FunctionDef) and n.name == "_fused_gdn_core")
        self.assertIn("benchmark_gdn_conv", called_functions(conv))
        self.assertIn("benchmark_gdn_core", called_functions(core))
        gdn_forward = next(n for n in gdn.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        gdn_calls = called_functions(gdn_forward)
        self.assertIn("benchmark_gdn_native_core", gdn_calls)
        self.assertNotIn("benchmark_gdn_conv", gdn_calls)
        self.assertNotIn("benchmark_gdn_core", gdn_calls)
        self.assertNotIn("_fused_gdn_gating", gdn_calls)
        attn_forward = function_node(AIOB, "Qwen3NextAttention", "forward")
        self.assertIn("benchmark_kv_cache_save", called_functions(attn_forward))
        attn_core = function_node(AIOB, "Qwen3NextAttention", "_attn_core")
        core_call = next(n for n in ast.walk(attn_core)
                         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                         and n.func.id == "benchmark_flash_attention")
        self.assertEqual(len(core_call.args), 3)
        self.assertIsInstance(core_call.args[0], ast.Attribute)
        self.assertEqual(core_call.args[0].attr, "args")
        self.assertEqual([arg.id for arg in core_call.args[1:]], ["kv_lens", "q_lens"])

    def test_flash_attention_uses_paged_flash_callback_with_scheduler_outside(self):
        helper = ast.parse(ATTENTION_HELPERS.read_text())
        flash = next(n for n in helper.body if isinstance(n, ast.FunctionDef)
                     and n.name == "benchmark_flash_attention")
        run = next(n for n in flash.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        flash_call = next(n for n in ast.walk(run) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name) and n.func.id == "flash_attn_varlen_func")
        self.assertIn("block_table", [kw.arg for kw in flash_call.keywords])
        self.assertIn("scheduler_metadata", [kw.arg for kw in flash_call.keywords])
        scheduler_calls = [n for n in ast.walk(flash) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Name) and n.func.id == "get_scheduler_metadata"]
        benchmark_calls = [n for n in ast.walk(flash) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Name) and n.func.id == "benchmark_component"]
        self.assertEqual(len(scheduler_calls), 1)
        self.assertEqual(len(benchmark_calls), 1)
        self.assertLess(scheduler_calls[0].lineno, benchmark_calls[0].lineno)

    def test_attention_core_routes_through_flash_helper(self):
        attn_core = function_node(AIOB, "Qwen3NextAttention", "_attn_core")
        core_call = next(n for n in ast.walk(attn_core) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name)
                         and n.func.id == "benchmark_flash_attention")
        self.assertEqual(len(core_call.args), 3)
        self.assertIsInstance(core_call.args[0], ast.Attribute)
        self.assertEqual(core_call.args[0].attr, "args")
        self.assertEqual([arg.id for arg in core_call.args[1:]], ["kv_lens", "q_lens"])

    def test_rotary_and_qk_norm_keep_vllm_arguments(self):
        rotary = function_node(AIOB, "Qwen3NextAttention", "_rotary_emb")
        rope_call = next(n for n in ast.walk(rotary) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "get_rope")
        rotary_factor = next(kw.value for kw in rope_call.keywords
                             if kw.arg == "partial_rotary_factor")
        self.assertIsInstance(rotary_factor, ast.Attribute)
        self.assertEqual(rotary_factor.attr, "partial_rotary_factor")
        qk_norm = function_node(AIOB, "Qwen3NextAttention", "_qk_norm")
        norm_call = next(n for n in ast.walk(qk_norm) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "GemmaRMSNormTest")
        self.assertGreaterEqual(len(norm_call.args), 4)
        self.assertIsInstance(norm_call.args[3], ast.Constant)
        self.assertIsNone(norm_call.args[3].value)

    def test_moe_router_and_expert_selection_use_expected_token_counts(self):
        forward = function_node(AIOB, "Qwen3NextSparseMoeBlock", "forward")
        calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)]
        route_gate = next(n for n in calls if n.func.attr == "_route_gate")
        select = next(n for n in calls if n.func.attr == "_select_experts")
        self.assertEqual(route_gate.args[0].id, "local_tokens")
        self.assertEqual(select.args[0].id, "tokens")
        gathered = next(n for n in ast.walk(forward) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == "gathered_tokens")
        self.assertEqual(gathered.args[0].id, "local_tokens")


if __name__ == "__main__":
    unittest.main()
