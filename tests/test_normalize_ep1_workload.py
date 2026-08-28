import importlib.util
import tempfile
import unittest
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "normalize_ep1_workload.py"
)
_SPEC = importlib.util.spec_from_file_location("normalize_ep1_workload", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class NormalizeEp1WorkloadTest(unittest.TestCase):
    def _write_workload(self, directory: str, ep_size: int) -> Path:
        path = Path(directory) / "workload.txt"
        path.write_text(
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
            f"model_parallel_NPU_group: 2 ep: {ep_size} pp: 1 all_gpus: 2\n"
            "1\n"
            "moe_route\t-1\t100\tALLTOALL_EP\t1024\t"
            f"{ep_size}\tALLTOALL_EP\t2048\t{ep_size}\tNONE\t0\t0\t100\n",
            encoding="utf-8",
        )
        return path

    def test_ep1_collectives_become_no_ops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_workload(directory, ep_size=1)

            changed = _MODULE.normalize_ep1_workload(path)

            self.assertEqual(changed, 2)
            self.assertIn(
                "\tNONE\t0\t0\tNONE\t0\t0\tNONE\t0\t0\t",
                path.read_text(encoding="utf-8"),
            )

    def test_ep_greater_than_one_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_workload(directory, ep_size=2)
            original = path.read_text(encoding="utf-8")

            changed = _MODULE.normalize_ep1_workload(path)

            self.assertEqual(changed, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
