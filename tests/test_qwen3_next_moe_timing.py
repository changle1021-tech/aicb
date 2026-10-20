"""CPU regressions for MoE profile parsing and shared-branch composition."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from workload_generator.mocked_model.inference import qwen3_next_moe_timing as timing


COMPONENTS = {
    "moe_route_gate": 10.0,
    "moe_route_select_experts": 20.0,
    "moe_expert_routed": 240.0,
    "moe_shared_expert": 50.0,
}


def write_profile(path, decode=None, prefill=None):
    sections = []
    for phase, values in (("decode", decode), ("prefill", prefill)):
        if values is None:
            continue
        sections.append(f"inference_term: {phase}")
        for name in timing.REQUIRED:
            sections.extend((f"{name}:", f"  time_gpu_avg: {values[name]}"))
    Path(path).write_text("\n".join(sections) + "\n", encoding="utf-8")


class ProfileReaderTests(unittest.TestCase):
    def test_converts_microseconds_to_nanoseconds_and_selects_phase(self):
        with TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile.txt"
            decode = {name: i + 1 for i, name in enumerate(timing.REQUIRED)}
            prefill = {name: (i + 1) * 10 for i, name in enumerate(timing.REQUIRED)}
            write_profile(profile, decode=decode, prefill=prefill)

            self.assertEqual(timing.read_moe_component_ns(profile, "decode"),
                             {name: value * 1000 for name, value in decode.items()})
            self.assertEqual(timing.read_moe_component_ns(profile, "prefill"),
                             {name: value * 1000 for name, value in prefill.items()})

    def test_rejects_duplicate_and_negative_values(self):
        with TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile.txt"
            profile.write_text(
                "inference_term: decode\n"
                "moe_route_gate:\n time_gpu_avg: 1\n"
                "moe_route_gate:\n time_gpu_avg: 2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                timing.read_moe_component_ns(profile, "decode")

            profile.write_text(
                "inference_term: decode\n"
                "moe_route_gate:\n time_gpu_avg: -0.5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid/duplicate"):
                timing.read_moe_component_ns(profile, "decode")

    def test_rejects_nan_timing(self):
        with TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile.txt"
            profile.write_text(
                "inference_term: decode\n"
                "moe_route_gate:\n time_gpu_avg: NaN\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid/duplicate"):
                timing.read_moe_component_ns(profile, "decode")


class SharedBranchCompositionTests(unittest.TestCase):
    def test_verified_native_bf16_overlap_counts_only_routed_expert(self):
        cache = {"moe_expert": 999.0, "other_component": 17.0}
        before = dict(cache)
        result, details = timing.compose_shared_expert(
            cache, COMPONENTS, local_tokens=2, threshold=8,
            enabled=True, precision="bf16",
        )
        self.assertTrue(details["applied"])
        self.assertEqual(result["moe_expert"], round(COMPONENTS["moe_expert_routed"]))
        self.assertEqual(result["other_component"], cache["other_component"])
        self.assertEqual(cache, before)

    def test_above_runtime_threshold_keeps_serial_cache(self):
        cache = {"moe_expert": 333.0}
        result, details = timing.compose_shared_expert(
            cache, COMPONENTS, local_tokens=9, threshold=8,
            enabled=True, precision="bf16",
        )
        self.assertFalse(details["applied"])
        self.assertEqual(details["reason"], "shared_expert_uses_main_stream")
        self.assertEqual(result, cache)

    def test_fp8_keeps_serial_cache(self):
        cache = {"moe_expert": 333.0}
        result, details = timing.compose_shared_expert(
            cache, COMPONENTS, local_tokens=2, threshold=8,
            enabled=True, precision="fp8",
        )
        self.assertFalse(details["applied"])
        self.assertEqual(details["reason"], "precision_path_not_verified")
        self.assertEqual(result, cache)

    def test_unverified_runtime_and_explicit_serial_keep_cache(self):
        with TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile.txt"
            write_profile(profile, decode=COMPONENTS)
            cache = {"moe_expert": 333.0}
            args = SimpleNamespace(phase="decode", micro_batch=1,
                                   seq_length=128, torch_dtype="bfloat16",
                                   moe_shared_overlap="auto")
            with patch("importlib.metadata.version", return_value="0.11.3"):
                result, details = timing.compose_qwen_moe_cache(cache, profile, args)
            self.assertEqual(result, cache)
            self.assertFalse(details["applied"])
            self.assertEqual(details["reason"], "serial_requested_or_unverified_runtime")

            args.moe_shared_overlap = "serial"
            with patch("importlib.metadata.version", return_value="0.11.2"):
                result, details = timing.compose_qwen_moe_cache(cache, profile, args)
            self.assertEqual(result, cache)
            self.assertFalse(details["applied"])
            self.assertEqual(details["reason"], "serial_requested_or_unverified_runtime")

    def test_shared_longer_than_main_conservatively_keeps_serial(self):
        components = dict(COMPONENTS, moe_route_gate=1.0,
                          moe_route_select_experts=2.0,
                          moe_expert_routed=3.0, moe_shared_expert=20.0)
        cache = {"moe_expert": 333.0}
        result, details = timing.compose_shared_expert(
            cache, components, local_tokens=2, threshold=8,
            enabled=True, precision="bf16",
        )
        self.assertFalse(details["applied"])
        self.assertEqual(details["reason"],
                         "shared_branch_requires_joint_communication_profile")
        self.assertEqual(result, cache)

    def test_batch_sizes_have_no_additional_hard_limit(self):
        for local_tokens in (1, 2, 8, 256):
            with self.subTest(local_tokens=local_tokens):
                result, details = timing.compose_shared_expert(
                    {"moe_expert": 1000.0}, COMPONENTS,
                    local_tokens=local_tokens, threshold=256,
                    enabled=True, precision="bf16",
                )
                self.assertTrue(details["applied"])
                self.assertEqual(result["moe_expert"], 240)


if __name__ == "__main__":
    unittest.main()
