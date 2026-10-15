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
        self.assertIn("benchmark_gdn_conv", ast.unparse(conv))
        self.assertIn("benchmark_gdn_core", ast.unparse(core))
        attn = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3NextAttention")
        attn_src = ast.unparse(attn)
        self.assertIn("BatchPrefillWithPagedKVCacheWrapper", attn_src)
        self.assertIn("benchmark_kv_cache_save", attn_src)


if __name__ == "__main__":
    unittest.main()
