"""Model-free regressions for the parity lane: python laya-dotnet/tools/test_ci_tools.py."""

import contextlib
import io
import types
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import mock_open, patch

from google.protobuf.message import DecodeError

import check_test_skips as guard
import regen_golden as regen


class SkipGuardTests(unittest.TestCase):
    def run_guard(self, outcomes, have=("english",), minimum=1):
        root = ET.Element(guard.NS + "TestRun")
        for outcome, reason in outcomes:
            result = ET.SubElement(root, guard.NS + "UnitTestResult", outcome=outcome, testName="parity")
            ET.SubElement(ET.SubElement(result, guard.NS + "Output"), guard.NS + "StdOut").text = reason
        argv = ["check_test_skips.py", "run.trx", "--min-passed", str(minimum), "--have", *have]
        with patch("sys.argv", argv), patch.object(guard.ET, "parse", return_value=ET.ElementTree(root)):
            with contextlib.redirect_stdout(io.StringIO()):
                return guard.main()

    def test_passed_run(self):
        self.assertEqual(0, self.run_guard([("Passed", "")]))

    def test_absent_checkpoint_is_allowed(self):
        self.assertEqual(0, self.run_guard([
            ("Passed", ""), ("NotExecuted", "no ONNX artifacts for 'Multilingual'; set LAYA_ONNX_ROOT"),
        ]))

    def test_owned_checkpoint_is_not_allowed(self):
        self.assertEqual(1, self.run_guard([
            ("Passed", ""), ("NotExecuted", "no split ONNX artifact for 'English'; export it"),
        ]))

    def test_missing_goldens_are_not_allowed(self):
        self.assertEqual(1, self.run_guard([
            ("Passed", ""), ("NotExecuted", "no golden data for 'Multilingual'; regenerate"),
        ]))

    def test_unknown_skip_fails_closed(self):
        self.assertEqual(1, self.run_guard([("Passed", ""), ("NotExecuted", "unexpected reason")]))

    def test_failure_and_unknown_outcome_fail(self):
        for outcome in [*guard.FAILING, "Unknown"]:
            with self.subTest(outcome=outcome):
                self.assertEqual(1, self.run_guard([("Passed", ""), (outcome, "")]))

    def test_empty_or_wrong_filter_fails(self):
        self.assertEqual(1, self.run_guard([]))
        self.assertEqual(1, self.run_guard([("Passed", "")], minimum=2))

    def test_router_requires_both_checkpoints(self):
        self.assertTrue(guard.expected_skip(guard.ROUTER, {"english"}))
        self.assertFalse(guard.expected_skip(guard.ROUTER, {"english", "multilingual"}))


class ExportCacheTests(unittest.TestCase):
    def validate(self, stamp="expected", sidecar_size=100, length=100, broken_graph=False):
        import onnx

        tensor = onnx.TensorProto(data_location=onnx.TensorProto.EXTERNAL)
        for key, value in [("location", "encoder.onnx.data"), ("length", str(length))]:
            entry = tensor.external_data.add()
            entry.key, entry.value = key, value
        model = types.SimpleNamespace(graph=types.SimpleNamespace(initializer=[tensor]))

        def size(path):
            if path.endswith(".data"):
                if sidecar_size is None:
                    raise FileNotFoundError(path)
                return sidecar_size
            return 100

        with patch("builtins.open", mock_open(read_data='{"inputs_sha256": "%s"}' % stamp)):
            with patch.object(regen.os.path, "getsize", side_effect=size):
                with patch.object(onnx, "load", return_value=model,
                                  side_effect=DecodeError("broken") if broken_graph else None):
                    return regen.is_valid("artifacts", ["encoder.onnx"], "expected")

    def test_complete_cache_is_reused(self):
        self.assertTrue(self.validate())

    def test_stale_stamp_is_rejected(self):
        self.assertFalse(self.validate(stamp="old"))

    def test_missing_empty_or_truncated_sidecar_is_rejected(self):
        for size in [None, 0, 99]:
            with self.subTest(size=size):
                self.assertFalse(self.validate(sidecar_size=size))

    def test_malformed_graph_is_rejected(self):
        self.assertFalse(self.validate(broken_graph=True))


if __name__ == "__main__":
    unittest.main()
