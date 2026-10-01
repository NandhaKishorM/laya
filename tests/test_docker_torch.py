"""Check the narrow NVIDIA metadata repair without installing CUDA."""
import base64
import contextlib
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "docker" / "check_torch.py"
spec = importlib.util.spec_from_file_location("check_torch", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class WheelFixture:
    metadata = {"Name": "nvidia-cusparselt-cu13"}
    version = "0.8.0"
    files = [Path("nvidia_cusparselt_cu13-0.8.0.dist-info/WHEEL")]

    def __init__(self, root):
        self.root = Path(root)
        self.wheel = self.locate_file(self.files[0])
        self.wheel.parent.mkdir()
        self.wheel.write_text("Wheel-Version: 1.0\nTag: py3-none-manylinux2014_sbsa\n")
        self.record = self.wheel.with_name("RECORD")
        self.record.write_text(f"{self.files[0]},sha256=old,0\nother-file,sha256=unchanged,12\n")
        self.library = self.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0")
        self.library.parent.mkdir(parents=True)
        self.library.write_bytes(b"\x7fELF\x02\x01" + bytes(12) + (183).to_bytes(2, "little"))

    def locate_file(self, path):
        return self.root / path


class WheelRepairTests(unittest.TestCase):
    def test_repair_rehashes_record_and_is_idempotent(self):
        for version in check.REVIEWED:
            with self.subTest(version=version):
                self._repair(version)

    def _repair(self, version):
        with tempfile.TemporaryDirectory() as root:
            dist = WheelFixture(root)
            dist.version = version
            with patch.object(check.ctypes, "CDLL") as loader:
                self.assertTrue(check.repair_cusparselt(dist))
                self.assertFalse(check.repair_cusparselt(dist))
                loader.assert_called_once_with(str(dist.library))
            data = dist.wheel.read_bytes()
            self.assertIn(b"Tag: py3-none-manylinux2014_aarch64", data)
            with dist.record.open() as handle:
                rows = list(csv.reader(handle))
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            self.assertEqual(rows[0][1:], ["sha256=" + digest, str(len(data))])
            self.assertEqual(rows[1], ["other-file", "sha256=unchanged", "12"])

    def test_unreviewed_version_is_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            dist = WheelFixture(root)
            before = dist.wheel.read_bytes(), dist.record.read_bytes()
            dist.version = "0.8.2"
            with self.assertRaisesRegex(RuntimeError, "Unreviewed"):
                check.repair_cusparselt(dist)
            self.assertEqual(before, (dist.wheel.read_bytes(), dist.record.read_bytes()))

    def test_wrong_architecture_or_linkage_is_untouched(self):
        for error in ("architecture", "linkage"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as root:
                dist = WheelFixture(root)
                before = dist.wheel.read_bytes(), dist.record.read_bytes()
                if error == "architecture":
                    dist.library.write_bytes(b"not an ARM64 library")
                with patch.object(check.ctypes, "CDLL", side_effect=OSError("cannot link")):
                    with self.assertRaises((RuntimeError, OSError)):
                        check.repair_cusparselt(dist)
                self.assertEqual(before, (dist.wheel.read_bytes(), dist.record.read_bytes()))


class MainEntryTests(unittest.TestCase):
    """`main()`'s own contract: a missing argument is a usage error, a missing torch is named.

    The build always calls this with the TORCH_INDEX it installed (`Dockerfile`:19), so both
    paths are for a person running the script by hand or in a bare environment -- where the
    old behaviour was `IndexError: list index out of range` and `ModuleNotFoundError`
    respectively, neither of which says what was wrong with the invocation.
    """

    def test_missing_expected_tag_is_a_usage_error(self):
        stderr = io.StringIO()
        with patch.object(sys, "argv", [str(SCRIPT)]), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                check.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("usage", stderr.getvalue())
        self.assertIn("cu128", stderr.getvalue())

    def test_missing_torch_is_reported_by_name(self):
        with patch.object(check, "repair_cusparselt"):   # never touch a real env in a unit test
            with patch.object(sys, "argv", [str(SCRIPT), "cpu"]):
                with patch.dict(sys.modules, {"torch": None}):   # `import torch` raises
                    with self.assertRaises(RuntimeError) as caught:
                        check.main()
        self.assertIn("not installed", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ImportError)


if __name__ == "__main__":
    unittest.main()
