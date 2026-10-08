"""Model-free regressions: python laya-rust/tools/test_ci_tools.py."""

import contextlib
import io
import types
import unittest
from unittest.mock import mock_open, patch

from google.protobuf.message import DecodeError

import check_test_skips as guard
import regen_golden as regen
from repr_cases import build_repr_probe


def summary(passed=10, failed=0, ignored=0, filtered=0):
    outcome = "FAILED" if failed else "ok"
    return ("test result: %s. %d passed; %d failed; %d ignored; 0 measured; %d filtered out;\n"
            % (outcome, passed, failed, ignored, filtered))


class SkipGuardTests(unittest.TestCase):
    def check(self, text, minimum=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return guard.check(text, minimum)

    def test_passed_run_including_empty_sample_and_doc_binaries(self):
        self.assertEqual(0, self.check(summary() + summary(0)))

    def test_early_returns_counted_as_passes_are_rejected(self):
        for reason in [
            "skip: no ONNX artifacts for 'english'",
            "skip: no split-layout ONNX artifacts for 'multilingual'",
            "skip: no golden data for 'typed-decisions'",
            "test parity ... skip: unexpected reason",
        ]:
            with self.subTest(reason=reason):
                self.assertEqual(1, self.check(reason + "\n" + summary()))

    def test_ignored_and_failed_tests_are_rejected(self):
        for text in [summary(ignored=1), summary(failed=1), summary(filtered=1)]:
            with self.subTest(text=text):
                self.assertEqual(1, self.check(text))

    def test_empty_wrong_filter_and_build_errors_fail_closed(self):
        for text in ["", summary(0), summary(1), summary() + "error: compiler failed\n"]:
            with self.subTest(text=text):
                self.assertEqual(1, self.check(text, minimum=2))

    def test_unknown_or_incomplete_runs_fail_closed(self):
        for text in [
            summary() + "test result: Unknown. 1 passed;\n",
            "     Running tests/one.rs (target/one)\n" + summary()
            + "     Running tests/two.rs (target/two)\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(1, self.check(text))

    def test_missing_output_fails(self):
        with patch("sys.argv", ["guard", "missing.log"]), patch.object(
            guard.Path, "read_text", side_effect=FileNotFoundError("missing")
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(1, guard.main())


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

    def test_cached_exports_still_regenerate_goldens(self):
        with patch.object(regen, "is_valid", return_value=True), patch.object(
            regen, "inputs_hash", return_value="stamp"
        ), patch.object(regen, "run") as run, patch.object(regen, "snapshot") as snapshot:
            regen.regen_checkpoint("english", "artifacts", "revision", False)
        snapshot.assert_not_called()
        run.assert_called_once()
        self.assertIn("dump_golden.py", run.call_args.args[0][1])
        self.assertIn("--checkpoint", run.call_args.args[0])

    def test_routing_regeneration_is_unconditional(self):
        with patch("sys.argv", ["regen", "--checkpoint", "routing"]), patch.object(
            regen, "run"
        ) as run, patch.object(regen.shutil, "rmtree"):
            self.assertEqual(0, regen.main())
        self.assertIn("dump_routing_golden.py", run.call_args.args[0][1])
        self.assertIn("--force", run.call_args.args[0])

    def test_repr_regeneration_needs_no_models(self):
        with patch("sys.argv", ["regen", "--checkpoint", "repr"]), patch.object(
            regen, "run"
        ) as run, patch.object(regen, "regen_checkpoint") as model, patch.object(regen.shutil, "rmtree"):
            self.assertEqual(0, regen.main())
        model.assert_not_called()
        run.assert_called_once()
        self.assertIn("repr_cases.py", run.call_args.args[0][1])


class FloatProbeTests(unittest.TestCase):
    def test_sweep_is_reproducible_and_preserves_bit_patterns(self):
        import math
        import struct

        probe = build_repr_probe()
        self.assertEqual(probe, build_repr_probe())
        self.assertEqual(2261, len(probe))
        self.assertEqual(len(probe), len({entry["bits"] for entry in probe}))
        for entry in probe:
            value = struct.unpack(">d", bytes.fromhex(entry["bits"]))[0]
            self.assertTrue(math.isfinite(value))
            self.assertEqual(repr(value), entry["repr"])
        self.assertEqual(
            "1658206780088562.2",
            next(entry["repr"] for entry in probe if entry["bits"] == "43179085685d83c9"),
        )


if __name__ == "__main__":
    unittest.main()
