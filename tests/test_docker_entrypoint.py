"""Exercise secret-file handling across the container command boundary."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker" / "entrypoint.py"


class EntrypointTests(unittest.TestCase):
    def invoke(self, extra, expected="direct"):
        env = {key: value for key, value in os.environ.items()
               if key not in {"HF_TOKEN", "HF_TOKEN_FILE"}}
        env.update(extra)
        child = "import os, sys; assert os.environ.get('HF_TOKEN') == sys.argv[1]; print('child ran')"
        return subprocess.run(
            [sys.executable, str(ENTRYPOINT), sys.executable, "-c", child, expected],
            env=env, capture_output=True, text=True, check=False,
        )

    def test_direct_value(self):
        result = self.invoke({"HF_TOKEN": "direct"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "child ran")

    def test_file_overrides_direct_and_trims_whitespace(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "token"
            secret.write_text("  synthetic-file-value\n", encoding="utf-8")
            result = self.invoke({"HF_TOKEN": "direct", "HF_TOKEN_FILE": str(secret)},
                                 "synthetic-file-value")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("synthetic-file-value", result.stdout + result.stderr)

    def test_invalid_files_stop_before_child_without_disclosure(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "private-filename"
            for contents in (None, b"", b" \n", b"synthetic-secret\x00", b"\xff"):
                with self.subTest(contents=contents):
                    if contents is not None:
                        secret.write_bytes(contents)
                    result = self.invoke({"HF_TOKEN": "direct", "HF_TOKEN_FILE": str(secret)})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("HF_TOKEN_FILE", result.stderr)
                    self.assertNotIn(str(secret), result.stderr)
                    self.assertNotIn("synthetic-secret", result.stderr)
                    self.assertNotIn("direct", result.stderr)

    def test_empty_file_setting_and_unknown_suffix_are_ignored(self):
        result = self.invoke({"HF_TOKEN": "direct", "HF_TOKEN_FILE": "", "UNKNOWN_FILE": "/missing"})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_command_fails(self):
        result = subprocess.run([sys.executable, str(ENTRYPOINT)], env={},
                                capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("command is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
