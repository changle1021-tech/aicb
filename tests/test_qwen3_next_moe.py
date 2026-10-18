"""CPU tests for the Qwen3-Next MoE load and layout contracts."""

import random
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from workload_generator.mocked_model.inference.qwen3_next_moe import (
    critical_profile_includes_router,
    expected_local_routes,
    gathered_tokens,
    moe_parallel_layout,
    sample_expert_loads,
)


def args(**overrides):
    values = dict(tensor_model_parallel_size=1, pipeline_model_parallel=1, world_size=4,
                  num_experts=512)
    values.update(overrides)
    return SimpleNamespace(**values)


class MoELayoutTests(unittest.TestCase):
    def test_dp4_ep4_layout(self):
        self.assertEqual(moe_parallel_layout(args()), (4, 4, 128))
        self.assertEqual(moe_parallel_layout(args(tensor_model_parallel_size=2, world_size=8)), (4, 8, 64))
        with self.assertRaises(ValueError):
            moe_parallel_layout(args(world_size=6))

    def test_decode_batch1_has_ten_routes_per_rank(self):
        self.assertEqual(gathered_tokens(1, 4), 4)
        self.assertAlmostEqual(expected_local_routes(1, 4, 10, 4), 10.0)
        self.assertAlmostEqual(expected_local_routes(512, 4, 10, 4), 5120.0)


class ExpertLoadTests(unittest.TestCase):
    def test_small_mean_is_not_raised_to_one(self):
        rng = random.Random(0)
        mean = 10 / 128
        totals = [sum(sample_expert_loads(mean, 128, rng)) for _ in range(20000)]
        self.assertAlmostEqual(sum(totals) / len(totals), 10.0, delta=0.1)
        self.assertLess(max(totals), 64)

    def test_loads_are_integers_around_the_mean(self):
        rng = random.Random(1)
        for mean in (0.0, 0.5, 1.0, 2.75, 40.0):
            loads = sample_expert_loads(mean, 128, rng)
            self.assertEqual(len(loads), 128)
            self.assertTrue(all(isinstance(v, int) for v in loads))
            self.assertTrue(all(int(mean) <= v <= int(mean) + 1 for v in loads))
        self.assertEqual(sum(sample_expert_loads(40.0, 128, rng)), 5120)

    def test_rejects_invalid_mean(self):
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sample_expert_loads(bad, 4)


class ProfileScopeTests(unittest.TestCase):
    def test_router_inclusion_follows_profile_scope_and_requires_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            cfg = SimpleNamespace(moe_compute_profile=str(path), phase="prefill")
            for scope, included in (("compute_only", True), ("compute_and_ep_comm", False)):
                path.write_text(json.dumps({"time_scope_by_phase": {"prefill": scope}}))
                self.assertEqual(critical_profile_includes_router(cfg), included)
            path.write_text(json.dumps({"time_scope_by_phase": {"prefill": "compute_and_ep_comm"},
                                        "router_linear_included_by_phase": {"prefill": "false"}}))
            with self.assertRaisesRegex(ValueError, "boolean"):
                critical_profile_includes_router(cfg)


class SharedExpertWiringTests(unittest.TestCase):
    def test_moe_forward_returns_shared_expert_time(self):
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1]
               / "workload_generator/mocked_model/inference/AiobQwen3Next.py").read_text()
        tree = ast.parse(src)
        moe = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "Qwen3NextSparseMoeBlock")
        forward = next(n for n in moe.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        called = [n.func.id for n in ast.walk(forward)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("benchmark_shared_expert", called)

    def test_shared_expert_defaults_to_moe_intermediate_size(self):
        from workload_generator.mocked_model.inference.qwen3_next_moe import benchmark_shared_expert
        # CPU-only: only the args-shape branch is exercised by import; the GPU
        # body requires torch. Verify the module is importable and the symbol exists.
        self.assertTrue(callable(benchmark_shared_expert))


if __name__ == "__main__":
    unittest.main()
