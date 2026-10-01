"""`verify/checkpoints.py`: what `--fetch` repairs, and which remedy the failure banner offers.

The script promises the manifest catches "a truncated download or an upstream change ...
rather than turning into quietly wrong answers". Two defects stood between a failure and its
fix, both about the advice a user follows after one:

1. `--fetch` skipped any checkpoint whose file existed, so the banner's instruction -- "fetch
   it with `--fetch`" -- could never repair a truncated transfer. It now re-downloads when
   the size disagrees with the manifest, and leaves a right-sized file alone (no network call).
2. The banner offered one remedy for both causes: `--record`. For a truncated file that
   writes the truncation into the manifest as the new truth. It now separates the two: a size
   mismatch is re-fetched, an equal-sized digest change is the one to review and re-record.

No network and no real weights: the manifest is rewritten into a temp directory and
`huggingface_hub.snapshot_download` is a stub that writes the bytes the manifest expects.

Run: python tests/test_verify_checkpoints.py
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verify" / "checkpoints.py"

spec = importlib.util.spec_from_file_location("verify_checkpoints_under_test", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

GOOD = b"the checkpoint bytes, as fetched at the pinned revision"


def manifest_for(relpath, name="english", good=GOOD):
    return {
        "repo": "convaiinnovations/laya",
        "revision": "0" * 40,
        "note": "fixture",
        "checkpoints": [{
            "name": name, "repo": "convaiinnovations/laya", "subfolder": None,
            "path": relpath, "file": "model.safetensors",
            "bytes": len(good), "sha256": hashlib.sha256(good).hexdigest(),
            "params": 1, "encoder": "fixture", "context": 8,
        }],
    }


class FakeHub:
    """`huggingface_hub` stand-in: remembers the calls and writes the expected bytes."""

    def __init__(self, payload=GOOD):
        self.payload = payload
        self.calls = []

    def snapshot_download(self, repo, revision=None, local_dir=None, allow_patterns=None):
        self.calls.append({"repo": repo, "revision": revision})
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, "model.safetensors"), "wb") as handle:
            handle.write(self.payload)
        return local_dir


class CheckpointsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.models = self.root / "models"
        (self.models / "laya").mkdir(parents=True)

    def write_manifest(self, **kwargs):
        path = self.root / "checkpoints.json"
        path.write_text(json.dumps(manifest_for("laya/model.safetensors", **kwargs)))
        return path

    def run_script(self, manifest_path, *extra):
        argv = ["checkpoints.py", "--models", str(self.models), *extra]
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(check, "MANIFEST", str(manifest_path)), \
                contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = check.main()
        return code, buf.getvalue()

    @property
    def weights(self):
        return self.models / "laya" / "model.safetensors"

    def test_an_intact_checkpoint_passes(self):
        self.weights.write_bytes(GOOD)
        code, out = self.run_script(self.write_manifest())
        self.assertEqual(code, 0, out)
        self.assertIn("match the manifest", out)

    def test_missing_file_names_the_fetch_command(self):
        code, out = self.run_script(self.write_manifest())
        self.assertEqual(code, 1, out)
        self.assertIn("missing", out)
        self.assertIn("--fetch", out)

    def test_truncated_file_is_reported_with_the_refetch_remedy(self):
        self.weights.write_bytes(GOOD[:5])
        code, out = self.run_script(self.write_manifest())
        self.assertEqual(code, 1, out)
        self.assertIn("size mismatch means the file is incomplete", out)
        self.assertIn("--fetch", out)
        # `--record` here would write the truncation into the manifest as the new truth.
        self.assertNotIn("re-record", out)

    def test_equal_sized_digest_change_points_at_record(self):
        self.weights.write_bytes(b"x" * len(GOOD))
        code, out = self.run_script(self.write_manifest())
        self.assertEqual(code, 1, out)
        self.assertIn("review the files and re-record with", out)
        self.assertNotIn("size mismatch means", out)

    def test_fetch_repairs_a_truncated_file(self):
        self.weights.write_bytes(GOOD[:5])            # what a killed download leaves
        hub = FakeHub()
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            code, out = self.run_script(self.write_manifest(), "--fetch", "--quiet")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(hub.calls), 1, "the truncated file must be re-downloaded")
        self.assertEqual(hub.calls[0]["revision"], "0" * 40, "at the manifest's pinned revision")
        self.assertEqual(self.weights.read_bytes(), GOOD)

    def test_fetch_leaves_a_right_sized_file_alone(self):
        self.weights.write_bytes(GOOD)
        hub = FakeHub()
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            code, out = self.run_script(self.write_manifest(), "--fetch", "--quiet")
        self.assertEqual(code, 0, out)
        self.assertEqual(hub.calls, [], "a verifiable file needs no network call")


if __name__ == "__main__":
    unittest.main()
