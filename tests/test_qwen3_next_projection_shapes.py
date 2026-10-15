import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from workload_generator.mocked_model.inference.qwen3_next_projections import (
    projection_precision,
    projection_shapes,
    validate_projection_metadata,
    write_projection_metadata,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "scripts/inference_configs/qwen3_next_default.json"


def make_args(tp=1, **overrides):
    values = json.loads(CONFIG.read_text())
    values.update(
        tensor_model_parallel_size=tp,
        world_size=tp,
        pipeline_model_parallel=1,
        expert_model_parallel_size=1,
        micro_batch=1,
        seq_length=128,
        phase="decode",
        model_name="qwen3-next-test",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ProjectionShapeTests(unittest.TestCase):
    def test_tp_shapes_and_gate(self):
        expected = {
            1: (12288, 64, 4096, 9216, 512),
            2: (6144, 32, 2048, 4608, 256),
            4: (3072, 16, 1024, 2560, 256),
        }
        for tp, (qkvz, ba, value, gated_qkv, kv) in expected.items():
            with self.subTest(tp=tp):
                shape = projection_shapes(make_args(tp))
                self.assertEqual(shape.gdn_qkvz_size, qkvz)
                self.assertEqual(shape.gdn_ba_size, ba)
                self.assertEqual(shape.gdn_value_size, value)
                self.assertEqual(shape.attention_qkv_size, gated_qkv)
                self.assertEqual(shape.attention_kv_size, kv)
                self.assertTrue(shape.attention_output_gate)
                ungated = projection_shapes(make_args(tp, attn_output_gate=False))
                self.assertFalse(ungated.attention_output_gate)
                self.assertEqual(ungated.attention_qkv_size, {1: 5120, 2: 2560, 4: 1536}[tp])

    def test_kv_replication_and_output_width_uses_gdn_values(self):
        # TP=4 replicates each of the two KV heads; GDN's output width remains
        # based on its value-head count, not the attention query-head count.
        args = make_args(4, linear_num_value_heads=64)
        shape = projection_shapes(args)
        self.assertEqual(shape.attention_kv_heads, 1)
        self.assertEqual(shape.attention_kv_size, 256)
        self.assertEqual(shape.gdn_value_size, 2048)
        self.assertNotEqual(shape.gdn_value_size, shape.attention_query_size)

    def test_invalid_tp_heads_and_dimensions(self):
        cases = (
            make_args(3),
            make_args(4, linear_num_key_heads=6),
            make_args(4, linear_num_value_heads=30),
            make_args(4, num_attention_heads=14),
            make_args(4, num_key_value_heads=3),
            make_args(2, linear_key_head_dim=-1),
            make_args(2, hidden_size=0),
        )
        for args in cases:
            with self.subTest(args=vars(args)):
                with self.assertRaises(ValueError):
                    projection_shapes(args)


class ProjectionPrecisionTests(unittest.TestCase):
    def test_default_explicit_override_and_invalid_precision(self):
        self.assertEqual(projection_precision(make_args()), "bf16")
        self.assertEqual(projection_precision(make_args(projection_precision="fp8")), "fp8")
        self.assertEqual(projection_precision(make_args(projection_precision="torch.BF16")), "bf16")
        with self.assertRaises(ValueError):
            projection_precision(make_args(projection_precision="fp16"))


class ProjectionMetadataTests(unittest.TestCase):
    def test_missing_profile_or_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.csv"
            profile.write_text("timing\n1\n")
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_projection_metadata(profile, make_args())
            profile.with_name(profile.name + ".projections.json").write_text("{}")
            profile.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_projection_metadata(profile, make_args())

    def test_old_version_shape_or_precision_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.csv"
            profile.write_text("timing\n1\n")
            metadata = Path(str(profile) + ".projections.json")
            args = make_args()
            write_projection_metadata(profile, args)
            validate_projection_metadata(profile, args)

            baseline = json.loads(metadata.read_text())
            for changed in ("version", "shape", "precision"):
                candidate = json.loads(json.dumps(baseline))
                if changed == "version":
                    candidate["projection_profile_version"] -= 1
                    changed_args = args
                elif changed == "shape":
                    candidate["shapes"]["gdn_qkvz_size"] += 128
                    changed_args = args
                else:
                    candidate["precision"] = "fp8"
                    changed_args = args
                metadata.write_text(json.dumps(candidate))
                with self.subTest(change=changed), self.assertRaisesRegex(ValueError, "differs"):
                    validate_projection_metadata(profile, changed_args)


if __name__ == "__main__":
    unittest.main()
