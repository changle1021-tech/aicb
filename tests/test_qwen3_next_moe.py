"""CPU tests for the Qwen3-Next MoE load and layout contracts."""

import random
from types import SimpleNamespace
import unittest

from workload_generator.mocked_model.inference.qwen3_next_moe import (
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


if __name__ == "__main__":
    unittest.main()
