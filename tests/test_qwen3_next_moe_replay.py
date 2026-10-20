"""CPU-only regressions for Qwen3-Next routed replay and cache timing API."""

import ast
import builtins
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "workload_generator" / "mocked_model" / "inference"


def args_for(batch=1, phase="decode", seq_length=512):
    return SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel=1,
        world_size=4,
        num_experts=512,
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts_per_tok=10,
        micro_batch=batch,
        seq_length=seq_length,
        phase=phase,
    )


def replay_inputs(local_tokens):
    rows = 4 * local_tokens
    return {
        "hidden_states": SimpleNamespace(shape=(rows, 2048)),
        "w1": SimpleNamespace(shape=(128, 1024, 2048)),
        "w2": SimpleNamespace(shape=(128, 2048, 512)),
        "topk_ids": SimpleNamespace(shape=(rows, 10)),
        "topk_weights": SimpleNamespace(shape=(rows, 10)),
        "expert_map": SimpleNamespace(shape=(512,)),
    }


class RoutedReplayValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moe = importlib.import_module(
            "workload_generator.mocked_model.inference.qwen3_next_moe"
        )

    def test_accepts_runtime_padded_batches(self):
        for batch in (1, 2, 8):
            with self.subTest(batch=batch):
                inputs = replay_inputs(batch)
                self.assertIs(
                    self.moe.validate_routed_replay(args_for(batch), batch, inputs),
                    inputs,
                )

    def test_rejects_missing_or_misshaped_replay_fields(self):
        cases = []
        missing = replay_inputs(1)
        del missing["w2"]
        cases.append((missing, "w2"))

        local_rows = replay_inputs(1)
        local_rows["hidden_states"] = SimpleNamespace(shape=(1, 2048))
        cases.append((local_rows, "hidden_states"))

        wrong_map = replay_inputs(1)
        wrong_map["expert_map"] = SimpleNamespace(shape=(128,))
        cases.append((wrong_map, "expert_map"))

        for inputs, field in cases:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.moe.validate_routed_replay(args_for(1), 1, inputs)


def load_production_moe_forward():
    """Compile only the production method so heavyweight module imports stay out."""
    source = INFERENCE / "AiobQwen3Next.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    moe_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3NextSparseMoeBlock"
    )
    forward = next(
        node for node in moe_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    module = ast.Module(body=[forward], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["forward"]


class SparseMoeForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forward = load_production_moe_forward()

    def run_case(self, args, local_tokens):
        calls = []
        returned = {
            "gate": object(), "select": object(),
            "routed": object(), "shared": object(),
        }
        helpers = {
            "gathered_tokens": lambda local, dp: local * dp,
            "benchmark_route_gate": lambda local: calls.append(("gate", local)) or returned["gate"],
            "benchmark_select_experts": lambda total: calls.append(("select", total)) or returned["select"],
            "benchmark_routed_experts": lambda _args, local: calls.append(("routed", local)) or returned["routed"],
            "benchmark_shared_expert": lambda _args, local: calls.append(("shared", local)) or returned["shared"],
        }
        # The patched forward calls instance methods for gate/select. Bind simple
        # recorders directly so this stays a test of the production dispatch.
        self_obj = SimpleNamespace(args=args, dp=4)
        self_obj._route_gate = lambda local: helpers["benchmark_route_gate"](local)
        self_obj._select_experts = lambda total: helpers["benchmark_select_experts"](total)
        # Resolve production global helpers without importing the CUDA-heavy file.
        fn_globals = self.forward.__globals__
        old = {key: fn_globals.get(key) for key in helpers}
        fn_globals.update(helpers)
        fn_globals["InferencePhase"] = SimpleNamespace(
            DECODE=SimpleNamespace(value="decode"),
            PREFILL=SimpleNamespace(value="prefill"),
        )
        try:
            output = type(self).forward(self_obj)
        finally:
            for key, value in old.items():
                if value is None:
                    fn_globals.pop(key, None)
                else:
                    fn_globals[key] = value
            fn_globals.pop("InferencePhase", None)
        return calls, output, returned

    def test_decode_gate_is_local_but_topk_uses_gathered_rows(self):
        for batch in (1, 2, 8):
            with self.subTest(batch=batch):
                calls, output, values = self.run_case(args_for(batch), batch)
                self.assertEqual(calls, [
                    ("gate", batch), ("select", 4 * batch),
                    ("routed", batch), ("shared", batch),
                ])
                self.assertEqual(output, tuple(values[name] for name in
                                               ("gate", "select", "routed", "shared")))

    def test_prefill_gate_uses_local_sequence_rows(self):
        args = args_for(phase="prefill", seq_length=512)
        calls, output, values = self.run_case(args, 512)
        self.assertEqual(calls, [
            ("gate", 512), ("select", 2048),
            ("routed", 512), ("shared", 512),
        ])
        self.assertEqual(output, tuple(values[name] for name in
                                       ("gate", "select", "routed", "shared")))


class FakeEvent:
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def record(self):
        self.events.append(self.label)

    def elapsed_time(self, _end):
        return 3.5


class FakeGraph:
    def __init__(self, events):
        self.events = events

    def replay(self):
        self.events.append("replay")


class FakeCuda:
    def __init__(self, events):
        self.events = events
        self.event_count = 0

    def synchronize(self):
        self.events.append("synchronize")

    def Event(self, enable_timing, external):
        self.events.append(("event", enable_timing, external))
        label = "start" if self.event_count == 0 else "end"
        self.event_count += 1
        return FakeEvent(label, self.events)

    def CUDAGraph(self):
        self.events.append("make_graph")
        return FakeGraph(self.events)

    class _GraphContext:
        def __init__(self, events):
            self.events = events

        def __enter__(self):
            self.events.append("graph_enter")

        def __exit__(self, *_exc):
            self.events.append("graph_exit")

    def graph(self, _graph):
        return self._GraphContext(self.events)


class FakeTorch:
    uint8 = "uint8"

    def __init__(self, events):
        self.events = events
        self.cuda = FakeCuda(events)

    def empty(self, *shape, **kwargs):
        self.events.append(("empty", shape, kwargs))
        return SimpleNamespace(zero_=lambda: self.events.append("flush_zero"))


class BenchmarkCachePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projections = importlib.import_module(
            "workload_generator.mocked_model.inference.qwen3_next_projections"
        )

    def benchmark(self, policy):
        events = []
        fake_torch = FakeTorch(events)

        def callback():
            events.append("fn")

        def prepare():
            events.append("prepare")

        details = {}
        with patch.dict(sys.modules, {"torch": fake_torch}):
            seconds = self.projections.benchmark_gpu_region(
                callback, warmup=1, samples=2,
                cache_policy=policy, timing_details=details, prepare=prepare,
            )
        return seconds, details, events

    def test_warm_policy_skips_flush_and_reports_event_span_ms(self):
        seconds, details, events = self.benchmark("warm")
        self.assertNotIn("empty", [event[0] for event in events if isinstance(event, tuple)])
        self.assertNotIn("flush_zero", events)
        self.assertLess(events.index("start"), events.index("fn", events.index("graph_enter")))
        self.assertLess(events.index("fn", events.index("graph_enter")), events.index("end"))
        graph_prepare = events.index("prepare", events.index("graph_enter"))
        self.assertLess(graph_prepare, events.index("start"))
        self.assertEqual(events.count("prepare"), 2)  # warmup and graph capture
        self.assertEqual(events.count("fn"), 2)  # warmup and graph capture
        self.assertEqual(seconds, 0.0035)
        self.assertEqual(details["metric"], "cuda_event_region_span_ms")
        self.assertEqual(details["samples_ms"], [3.5, 3.5])
        self.assertEqual(details["median_ms"], 3.5)
        self.assertEqual(details["flush_bytes"], 0)

    def test_cold_policy_keeps_flush_behavior(self):
        _seconds, details, events = self.benchmark("cold")
        self.assertIn("empty", [event[0] for event in events if isinstance(event, tuple)])
        self.assertIn("flush_zero", events)
        self.assertLess(events.index("flush_zero"), events.index("prepare", events.index("graph_enter")))
        self.assertEqual(details["flush_bytes"], 128 * 1024 * 1024)


class FakeTensor:
    def __init__(self, shape, values=None):
        self.shape = shape
        self.values = list(values or [7])

    def clone(self):
        return FakeTensor(self.shape, self.values)

    def copy_(self, other):
        self.values[:] = other.values
        return self


class RoutedReplayInputSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moe = importlib.import_module(
            "workload_generator.mocked_model.inference.qwen3_next_moe"
        )

    def test_bf16_replay_restores_working_input_and_passes_inplace(self):
        args = args_for(1)
        inputs = {
            "hidden_states": FakeTensor((4, 2048), [3, 5]),
            "w1": FakeTensor((128, 1024, 2048)),
            "w2": FakeTensor((128, 2048, 512)),
            "topk_ids": FakeTensor((4, 10)),
            "topk_weights": FakeTensor((4, 10)),
            "expert_map": FakeTensor((512,)),
        }
        original_values = list(inputs["hidden_states"].values)
        fused_calls = []

        def fake_fused(x, _w1, _w2, _weights, _ids, **kwargs):
            fused_calls.append((x, list(x.values), kwargs))
            x.values[:] = [-99]

        timing_call = {}

        def fake_benchmark(fn, **kwargs):
            timing_call.update(kwargs)
            timing_call["prepare"] = kwargs["prepare"]
            timing_call["run"] = fn
            for _ in range(2):
                kwargs["prepare"]()
                fn()
            return 0.012

        original_import = builtins.__import__

        def import_with_fake_fused(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "vllm.model_executor.layers.fused_moe.fused_moe":
                return SimpleNamespace(fused_experts=fake_fused,
                                       disable_inplace=lambda: False)
            return original_import(name, globals, locals, fromlist, level)

        # The replay path only needs the torch module to satisfy its import;
        # all tensor behavior is provided by CPU-only fakes.
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
             patch("builtins.__import__", side_effect=import_with_fake_fused), \
             patch.object(self.moe, "benchmark_gpu_region", side_effect=fake_benchmark):
            result = self.moe.benchmark_routed_experts(
                args, 1, replay_inputs=inputs, cache_policy="warm"
            )

        self.assertEqual(result, 0.012)
        self.assertEqual(len(fused_calls), 2)
        self.assertTrue(all(call[2]["inplace"] is True for call in fused_calls))
        self.assertTrue(all(call[1] == original_values for call in fused_calls))
        self.assertIsNot(fused_calls[0][0], inputs["hidden_states"])
        self.assertEqual(inputs["hidden_states"].values, original_values)
        self.assertEqual(timing_call["cache_policy"], "warm")
        self.assertIsNotNone(timing_call["prepare"])

    def test_disabled_inplace_uses_original_input_without_prepare(self):
        args = args_for(1)
        inputs = {
            "hidden_states": FakeTensor((4, 2048), [11, 13]),
            "w1": FakeTensor((128, 1024, 2048)),
            "w2": FakeTensor((128, 2048, 512)),
            "topk_ids": FakeTensor((4, 10)),
            "topk_weights": FakeTensor((4, 10)),
            "expert_map": FakeTensor((512,)),
        }
        original = inputs["hidden_states"]
        original_values = list(original.values)
        fused_calls = []
        timing_call = {}

        def fake_fused(x, _w1, _w2, _weights, _ids, **kwargs):
            fused_calls.append((x, list(x.values), kwargs))

        def fake_benchmark(fn, **kwargs):
            timing_call.update(kwargs)
            for _ in range(2):
                if kwargs["prepare"] is not None:
                    kwargs["prepare"]()
                fn()
            return 0.012

        original_import = builtins.__import__

        def import_with_non_inplace_fused(name, globals=None, locals=None,
                                          fromlist=(), level=0):
            if name == "vllm.model_executor.layers.fused_moe.fused_moe":
                return SimpleNamespace(fused_experts=fake_fused,
                                       disable_inplace=lambda: True)
            return original_import(name, globals, locals, fromlist, level)

        with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
             patch("builtins.__import__", side_effect=import_with_non_inplace_fused), \
             patch.object(self.moe, "benchmark_gpu_region", side_effect=fake_benchmark):
            result = self.moe.benchmark_routed_experts(
                args, 1, replay_inputs=inputs, cache_policy="warm"
            )

        self.assertEqual(result, 0.012)
        self.assertEqual(len(fused_calls), 2)
        self.assertTrue(all(call[0] is original for call in fused_calls))
        self.assertTrue(all(call[1] == original_values for call in fused_calls))
        self.assertTrue(all(call[2]["inplace"] is True for call in fused_calls))
        self.assertIsNone(timing_call["prepare"])
        self.assertEqual(original.values, original_values)


class ProjectionMetadataIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projections = importlib.import_module(
            "workload_generator.mocked_model.inference.qwen3_next_projections"
        )

    def test_moe_producer_change_invalidates_existing_sidecar(self):
        args = SimpleNamespace(
            hidden_size=2048,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            tensor_model_parallel_size=1,
            attn_output_gate=True,
            model_name=None,
            world_size=None,
            pipeline_model_parallel=None,
            expert_model_parallel_size=None,
            micro_batch=None,
            seq_length=None,
            phase=None,
        )
        producer = INFERENCE / "qwen3_next_moe.py"
        original_read_bytes = Path.read_bytes

        with TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile.csv"
            profile.write_text("profile data\n", encoding="utf-8")
            self.projections.write_projection_metadata(profile, args)
            # Successful validation returns normally (the API signals failure
            # with ValueError rather than a boolean result).
            self.projections.validate_projection_metadata(profile, args)

            def changed_producer_bytes(path):
                if path.resolve() == producer.resolve():
                    return b"producer contents changed for identity regression"
                return original_read_bytes(path)

            with patch.object(Path, "read_bytes", autospec=True,
                              side_effect=changed_producer_bytes):
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    self.projections.validate_projection_metadata(profile, args)

    def test_runtime_versions_are_recorded_and_invalidate_old_sidecar(self):
        args = SimpleNamespace(
            hidden_size=2048,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            tensor_model_parallel_size=1,
            attn_output_gate=True,
            model_name=None,
            world_size=None,
            pipeline_model_parallel=None,
            expert_model_parallel_size=None,
            micro_batch=None,
            seq_length=None,
            phase=None,
        )
        old_versions = {"torch": "2.8.0", "vllm": "0.11.1"}
        new_versions = {"torch": "2.9.0", "vllm": "0.11.2"}

        with TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile.csv"
            profile.write_text("profile data\n", encoding="utf-8")
            with patch.object(self.projections, "profile_runtime_versions",
                              return_value=old_versions):
                metadata = self.projections.projection_metadata(args)
                self.assertEqual(metadata["runtime_versions"], old_versions)
                self.projections.write_projection_metadata(profile, args)
                self.projections.validate_projection_metadata(profile, args)

            with patch.object(self.projections, "profile_runtime_versions",
                              return_value=new_versions):
                changed = self.projections.projection_metadata(args)
                self.assertEqual(changed["runtime_versions"], new_versions)
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    self.projections.validate_projection_metadata(profile, args)


if __name__ == "__main__":
    unittest.main()
