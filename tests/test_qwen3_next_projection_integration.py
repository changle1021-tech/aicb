"""Exercise production profiling callbacks with CPU fakes, without CUDA imports."""

import ast
import copy
from contextlib import contextmanager, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from workload_generator.mocked_model.MockedModel import InferencePhase
from workload_generator.mocked_model.inference import qwen3_next_projections as projections


ROOT = Path(__file__).resolve().parents[1]
AIOB = ROOT / "workload_generator/mocked_model/inference/AiobQwen3Next.py"


def config(**overrides):
    values = json.loads((ROOT / "scripts/inference_configs/qwen3_next_default.json").read_text())
    values.update(tensor_model_parallel_size=1, micro_batch=1, seq_length=512, phase="prefill")
    values.update(overrides)
    return SimpleNamespace(**values)


def load_profiling_classes():
    # Keep the actual method bodies. Replace only GPU base classes/imports so
    # the production forward/parameter wiring can run on a CPU-only machine.
    nodes = []
    names = ("Qwen3NextGatedDeltaNet", "Qwen3NextAttention", "Qwen3NextModel", "Qwen3NextSparseMoeBlock")
    for node in ast.parse(AIOB.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            node = copy.deepcopy(node)
            node.bases = [ast.Name(id="object", ctx=ast.Load())]
            nodes.append(node)
    env = {
        "InferencePhase": InferencePhase,
        "benchmark_projection": Mock(return_value=1.0),
        "critical_profile_includes_router": Mock(return_value=True),
        "benchmark_input_quantization": Mock(return_value=0.0),
        "benchmark_gdn_conv": Mock(return_value=100.0),
        "benchmark_gdn_core": Mock(return_value=200.0),
        "benchmark_gdn_gating": Mock(return_value=300.0),
        "benchmark_attention_output_gate": Mock(return_value=2.0),
        "benchmark_gdn_native_core": Mock(return_value=7.0),
        "benchmark_gdn_compiled_chain": Mock(return_value=13.0),
        "benchmark_kv_cache_save": Mock(return_value=0.0),
        "projection_precision": projections.projection_precision,
        "projection_shapes": projections.projection_shapes,
        "write_projection_metadata": Mock(),
        "calculate_stats": Mock(), "process_all_keys": Mock(),
        "write_time": Mock(return_value="stats.txt"), "os": os,
    }
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, str(AIOB), "exec"), env)
    return env


class ProjectionBackendTests(unittest.TestCase):
    def test_region_uses_device_events_and_flush_outside_timed_interval(self):
        order = []
        capture = {"graph": None}
        start, end = Mock(), Mock()
        def record(name):
            self.assertIsNone(capture["graph"], "CUDA events must be recorded outside graph capture")
            order.append((name, None))
        start.record.side_effect = lambda: record("start")
        end.record.side_effect = lambda: record("end")
        start.elapsed_time.side_effect = [102.0, 38.0, 42.0, 106.0, 110.0, 46.0]
        torch = Mock()
        torch.empty.return_value.zero_.side_effect = lambda: order.append(("flush", capture["graph"]))
        graph, baseline = Mock(), Mock()
        graph.replay.side_effect = lambda: order.append(("graph_replay", None))
        baseline.replay.side_effect = lambda: order.append(("baseline_replay", None))
        torch.cuda.CUDAGraph.side_effect = [graph, baseline]
        torch.cuda.Event.side_effect = [start, end]
        @contextmanager
        def graph_capture(value):
            self.assertIsNone(capture["graph"])
            capture["graph"] = value
            order.append(("capture_start", value))
            try:
                yield
            finally:
                order.append(("capture_end", value))
                capture["graph"] = None
        torch.cuda.graph.side_effect = graph_capture
        callback = Mock(side_effect=lambda: order.append(("kernel", capture["graph"])))
        with patch.dict(sys.modules, {"torch": torch}):
            seconds = projections.benchmark_gpu_region(callback, warmup=1, samples=3)
        first_event = order.index(("start", None))
        graph_kernels = [i for i, item in enumerate(order) if item == ("kernel", graph)]
        self.assertEqual(len(graph_kernels), 32)
        self.assertEqual(len([item for item in order if item == ("flush", graph)]), 32)
        self.assertEqual(len([item for item in order if item == ("flush", baseline)]), 32)
        self.assertEqual([order[i - 1] for i in graph_kernels], [("flush", graph)] * 32)
        self.assertEqual(order[:first_event].count(("kernel", None)), 1)
        self.assertEqual(callback.call_count, 33)
        self.assertEqual(order[first_event:].count(("graph_replay", None)), 3)
        self.assertEqual(order[first_event:].count(("baseline_replay", None)), 3)
        self.assertEqual(start.record.call_count, 6)
        self.assertEqual(end.record.call_count, 6)
        self.assertEqual([(total-base)/32.0 for total, base in
                          zip((102, 106, 110), (38, 42, 46))], [2.0, 2.0, 2.0])
        self.assertEqual(seconds, 0.002)
        self.assertEqual(torch.cuda.CUDAGraph.call_count, 2)

    def test_eager_region_calls_callback_each_sample_and_flushes_before_start(self):
        order = []
        start, end = Mock(), Mock()
        start.record.side_effect = lambda: order.append("start")
        end.record.side_effect = lambda: order.append("end")
        end.synchronize.side_effect = lambda: order.append("end_sync")
        start.elapsed_time.side_effect = [1.0, 3.0]
        torch = Mock()
        torch.cuda.synchronize.side_effect = lambda: order.append("sync")
        torch.empty.return_value.zero_.side_effect = lambda: order.append("flush")
        torch.cuda.Event.side_effect = [start, end]

        def callback():
            order.append("callback")

        with patch.dict(sys.modules, {"torch": torch}):
            seconds = projections.benchmark_gpu_region(
                callback, warmup=1, samples=2, timing_mode="eager"
            )

        self.assertEqual(order, [
            "callback", "sync",
            "flush", "sync", "start", "callback", "end", "end_sync",
            "flush", "sync", "start", "callback", "end", "end_sync",
        ])
        self.assertEqual(seconds, 0.002)
        torch.cuda.CUDAGraph.return_value.replay.assert_not_called()
        torch.cuda.graph.assert_not_called()

    def test_component_dispatches_timing_mode_by_phase(self):
        callback = Mock()
        with patch.object(projections, "benchmark_gpu_region", return_value=0.25) as timer:
            self.assertEqual(projections.benchmark_component(config(phase="prefill"), callback), 0.25)
            timer.assert_called_once_with(callback, timing_mode="eager")

            timer.reset_mock()
            self.assertEqual(projections.benchmark_component(config(phase="decode"), callback), 0.25)
            timer.assert_called_once_with(callback, timing_mode="cuda_graph")

            timer.reset_mock()
            self.assertEqual(projections.benchmark_component(SimpleNamespace(), callback), 0.25)
            timer.assert_called_once_with(callback, timing_mode="cuda_graph")

    def test_region_rejects_unknown_timing_mode(self):
        torch = Mock()
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(ValueError, "timing_mode must be cuda_graph or eager"):
                projections.benchmark_gpu_region(Mock(), timing_mode="invalid")
        torch.cuda.synchronize.assert_not_called()
        torch.empty.assert_not_called()

    def test_bf16_runs_linear_with_correct_weight_orientation(self):
        torch = Mock()
        torch.bfloat16 = "bf16"
        x, weight = object(), object()
        torch.randn.side_effect = [x, weight]
        with patch.dict(sys.modules, {"torch": torch}), patch.object(
            projections, "benchmark_gpu_region", side_effect=lambda fn, **kwargs: (fn(), 0.001)[1]
        ):
            self.assertEqual(projections.benchmark_projection(config(), 512, 2048, 12288), 0.001)
        self.assertEqual(torch.randn.call_args_list[0].args, (512, 2048))
        self.assertEqual(torch.randn.call_args_list[1].args, (12288, 2048))
        torch.nn.functional.linear.assert_called_once_with(x, weight)

    def test_bf16_quantization_never_imports_gpu_dependencies(self):
        with patch.dict(sys.modules, {"torch": None, "vllm": None, "deep_gemm": None}):
            self.assertEqual(projections.benchmark_input_quantization(config(), 512, 2048), 0.0)

    def test_fp8_preserves_deepgemm_and_checks_result_before_timing(self):
        backend = SimpleNamespace(
            construct=Mock(return_value=("x", "weight", None, "out", "reference")),
            deep_gemm=SimpleNamespace(fp8_gemm_nt=Mock()),
            calc_diff=Mock(return_value=0.0001),
        )
        torch = Mock()
        with patch.dict(sys.modules, {"torch": torch, "utils.deepgemm_utils": backend}), patch.object(
            projections, "benchmark_gpu_region", side_effect=lambda fn, **kwargs: (fn(), 0.002)[1]
        ):
            value = projections.benchmark_projection(config(projection_precision="fp8"), 1, 4096, 2048)
        self.assertEqual(value, 0.002)
        backend.construct.assert_called_once_with(1, 4096, 2048)
        self.assertEqual(backend.deep_gemm.fp8_gemm_nt.call_count, 2)
        backend.calc_diff.assert_called_once_with("out", "reference")
        torch.randn.assert_not_called()
        backend.calc_diff.return_value = float("nan")
        with patch.dict(sys.modules, {"torch": torch, "utils.deepgemm_utils": backend}), patch.object(
            projections, "benchmark_gpu_region"
        ) as timer:
            with self.assertRaisesRegex(ValueError, "FP8 projection check failed"):
                projections.benchmark_projection(config(projection_precision="fp8"), 1, 4096, 2048)
            timer.assert_not_called()

    def test_output_gate_includes_sigmoid_and_multiply(self):
        class Tensor:
            def __init__(self, values):
                self.values = values

            def __mul__(self, other):
                return Tensor([a * b for a, b in zip(self.values, other.values)])

        torch = Mock()
        torch.randn.return_value = Tensor([2.0, -3.0])
        torch.randn_like.return_value = Tensor([0.0, 1.0])
        torch.sigmoid.side_effect = lambda x: Tensor([1 / (1 + math.exp(-v)) for v in x.values])
        measured = []
        with patch.dict(sys.modules, {"torch": torch}), patch.object(
            projections, "benchmark_gpu_region", side_effect=lambda fn, **kwargs: (measured.append(fn()), 0.003)[1]
        ):
            projections.benchmark_attention_output_gate(config(), 1, 4096)
        self.assertAlmostEqual(measured[0].values[0], 1.0)
        self.assertAlmostEqual(measured[0].values[1], -3.0 / (1 + math.exp(-1)))
        torch.sigmoid.assert_called_once()
        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(projections.benchmark_attention_output_gate(config(attn_output_gate=False), 1, 4096), 0)


class ProjectionWiringTests(unittest.TestCase):
    def test_moe_profile_keeps_external_router_only_when_excluded_from_profile(self):
        for included in (False, True):
            with self.subTest(router_in_profile=included):
                env = load_profiling_classes()
                env["critical_profile_includes_router"].return_value = included
                cls = env["Qwen3NextSparseMoeBlock"]
                obj = cls.__new__(cls)
                obj.args = config(moe_compute_profile="profile.json")
                obj._route_gate = Mock(return_value=0.003)
                expected = 0.0 if included else 0.003
                self.assertEqual(obj.forward(), (expected, 0.0, 0.0, 0.0))
                if included:
                    obj._route_gate.assert_not_called()
                else:
                    obj._route_gate.assert_called_once_with(512)

    def test_gdn_forward_profiles_both_phases_and_uses_value_width(self):
        for tp in (1, 2, 4):
            for phase, m in (("prefill", 512), ("decode", 2)):
                with self.subTest(tp=tp, phase=phase):
                    env = load_profiling_classes()
                    cls = env["Qwen3NextGatedDeltaNet"]
                    obj = cls.__new__(cls)
                    obj.args = config(tensor_model_parallel_size=tp, phase=phase, micro_batch=2,
                                      linear_num_value_heads=64)
                    obj.tp = tp
                    obj.shapes = projections.projection_shapes(obj.args)
                    for name in ("_norm",):
                        setattr(obj, name, Mock(return_value=0))
                    values = obj.forward()
                    calls = env["benchmark_projection"].call_args_list
                    if phase == "prefill":
                        env["benchmark_gdn_compiled_chain"].assert_called_once_with(obj.args, m)
                        self.assertEqual(values, (0.0, 0.0, 13.0, 0.0))
                        self.assertEqual(calls, [])
                        env["benchmark_input_quantization"].assert_not_called()
                        env["benchmark_gdn_native_core"].assert_not_called()
                    else:
                        env["benchmark_gdn_compiled_chain"].assert_not_called()
                        self.assertEqual([c.args[1:] for c in calls], [
                            (m, 2048, 20480 // tp),
                            (m, 2048, 128 // tp),
                            (m, 8192 // tp, 2048),
                        ])
                        self.assertEqual([c.args[1:] for c in env["benchmark_input_quantization"].call_args_list],
                                         [(m, 2048), (m, 8192 // tp)])
                        env["benchmark_gdn_native_core"].assert_called_once_with(obj.args, phase, m)
                    env["benchmark_gdn_conv"].assert_not_called()
                    env["benchmark_gdn_core"].assert_not_called()
                    env["benchmark_gdn_gating"].assert_not_called()
                    self.assertEqual(values[1], 0.0)
                    self.assertEqual(values[2], 13.0 if phase == "prefill" else 7.0)

    def test_attention_forward_keeps_gate_separate_and_kv_replication(self):
        env = load_profiling_classes()
        cls = env["Qwen3NextAttention"]
        obj = cls.__new__(cls)
        obj.args = config(tensor_model_parallel_size=4)
        obj.tp = 4
        obj.shapes = projections.projection_shapes(obj.args)
        obj._qk_norm = Mock(return_value=0)
        obj._rotary_emb = Mock(return_value=0)
        obj._attn_core = Mock(return_value=0)
        values = obj.forward()
        self.assertEqual([c.args[1:] for c in env["benchmark_projection"].call_args_list],
                         [(512, 2048, 2560), (512, 1024, 2048)])
        env["benchmark_attention_output_gate"].assert_called_once_with(obj.args, 512, 1024)
        self.assertEqual(values[3:], (1.0, 2.0))

    def test_model_records_gate_once_and_existing_parser_consumes_it(self):
        env = load_profiling_classes()
        cls = env["Qwen3NextModel"]
        obj = cls.__new__(cls)
        obj.args = config(model_name="Qwen3-Next-80B")
        obj.time_list = {}
        obj.aiob_forward_loops = 1
        obj._norm = lambda: 0
        obj.FullAttention = lambda: (1e-6, 2e-6, 3e-6, 4e-6, 5e-6)
        obj.GDN = lambda: (0, 0, 0, 0)
        obj.MoE = lambda: (0, 0, 0, 0)
        # The existing parser is stdlib-only once extracted from the large
        # utils module; no GPU package has to be imported for this contract.
        parser_ast = ast.parse((ROOT / "utils/utils.py").read_text())
        parser_node = next(n for n in parser_ast.body if isinstance(n, ast.FunctionDef)
                           and n.name == "extract_inference_averages")
        parser_env = {"re": re}
        exec(compile(ast.Module(body=[parser_node], type_ignores=[]), "utils", "exec"), parser_env)
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                with redirect_stdout(io.StringIO()):
                    obj.forward()
            finally:
                os.chdir(previous)
            stats = Path(tmp) / "stats.txt"
            stats.write_text("".join("{}:\n    time_gpu_avg: {}\n".format(k, v[0]["time_gpu"])
                                     for k, v in obj.time_list.items()))
            cache = parser_env["extract_inference_averages"](str(stats), obj.args)
        self.assertEqual(obj.time_list["atten_output_gate"], [{"time_gpu": 5.0}])
        self.assertEqual(cache["attention_layer"], 15000)
        env["write_projection_metadata"].assert_called_once_with("stats.txt", obj.args)

    def test_qwen_params_preserves_config_and_passes_explicit_cli_options(self):
        from workload_generator.mocked_model.MockedModel import MockedParamsBase
        source = ROOT / "workload_generator/mocked_model/inference/MockedQwen3Next.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "Qwen3NextParams")
        env = {"MockedParamsBase": MockedParamsBase}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), env)
        cfg = ROOT / "scripts/inference_configs/qwen3_next_default.json"
        args = SimpleNamespace(world_size=4, tensor_model_parallel_size=1,
                               expert_model_parallel_size=4, projection_precision=None,
                               attn_output_gate=None)
        params = env["Qwen3NextParams"](str(cfg), args)
        self.assertTrue(params.attn_output_gate)
        self.assertEqual(projections.projection_precision(params), "bf16")
        args.attn_output_gate, args.projection_precision = False, "fp8"
        params = env["Qwen3NextParams"](str(cfg), args)
        self.assertFalse(params.attn_output_gate)
        self.assertEqual(projections.projection_precision(params), "fp8")

    def test_profile_payload_change_invalidates_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp) / "stats.txt"
            stats.write_text("original timings")
            projections.write_projection_metadata(stats, config())
            projections.validate_projection_metadata(stats, config())
            stats.write_text("unrelated or legacy timings")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                projections.validate_projection_metadata(stats, config())


if __name__ == "__main__":
    unittest.main()
