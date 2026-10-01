"""Exercise secret-file handling across the container command boundary.

`docker/entrypoint.py` loops over `SECRET_NAMES`, and the HTTP service's documented way to hand
over its bearer key is the second name in that loop -- `docs/docker.md`, "Bearer token from a
file": "`LAYA_API_KEY_FILE` is read once at startup, moved into `LAYA_API_KEY`, and the `_FILE`
variable is removed before the server execs." Every test in this file used to name `HF_TOKEN`
in its harness, its expected value and its error assertion, so the loop was exercised through
one of its two arms:

    $ sed 's/SECRET_NAMES = ("HF_TOKEN", "LAYA_API_KEY")/SECRET_NAMES = ("HF_TOKEN",)/' docker/entrypoint.py > ...
    $ python tests/test_docker_entrypoint.py
    Ran 5 tests in 0.165s
    OK

Run that mutated script against a live `laya-serve` (mounted over the image's own entrypoint) and
the service is left unauthenticated: `POST /v1/systemone` with no `Authorization` header answers
400 instead of 401. Nothing in the suite said so. The names are therefore read out of the script
itself and each one is driven through the whole contract, including the `_FILE` removal, which no
test checked for either name.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker" / "entrypoint.py"
COMPOSE_FILES = tuple(sorted(ROOT.glob("compose*.y*ml")))


def _secret_names():
    """The names the container entrypoint claims to load, read from the entrypoint itself."""
    spec = importlib.util.spec_from_file_location("laya_entrypoint_under_test", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the script only defines main() under a __main__ guard
    return tuple(module.SECRET_NAMES)


SECRET_NAMES = _secret_names()


def file_backed_secrets(text):
    """Names a Compose file forwards as `<NAME>_FILE` next to `<NAME>`.

    The pairing is what makes one: `<NAME>` is where the loaded value has to land, so a `_FILE`
    whose base name is not forwarded at all is a path handed to the program itself rather than a
    secret to read. That excludes `LAYA_REQUEST_FILE`, which the quickstart opens directly and
    which must *not* appear in `SECRET_NAMES`.
    """
    forwarded = set(re.findall(r"^\s+([A-Z][A-Z0-9_]*):", text, re.MULTILINE))
    return {key[:-len("_FILE")] for key in forwarded
            if key.endswith("_FILE") and key[:-len("_FILE")] in forwarded}


class EntrypointTests(unittest.TestCase):
    def invoke(self, name, extra, expected="direct"):
        """Run the entrypoint with `extra`, and report what the child saw without echoing it."""
        blocked = {prefix for secret in SECRET_NAMES for prefix in (secret, secret + "_FILE")}
        env = {key: value for key, value in os.environ.items() if key not in blocked}
        env.update(extra)
        child = ("import json, os, sys;"
                 "n = sys.argv[1];"
                 "print(json.dumps({'name': n,"
                 " 'matched': os.environ.get(n) == sys.argv[2],"
                 " 'file_var': n + '_FILE' in os.environ}))")
        return subprocess.run(
            [sys.executable, str(ENTRYPOINT), sys.executable, "-c", child, name, expected],
            env=env, capture_output=True, text=True, check=False,
        )

    def report(self, result, name):
        """The child's own account of its environment, or the entrypoint's failure."""
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("\n"), 1, result.stdout)
        seen = json.loads(result.stdout)
        self.assertEqual(seen.get("name"), name, result.stdout)
        return seen

    def test_every_secret_the_deployment_forwards_is_loaded(self):
        """A `<NAME>` / `<NAME>_FILE` pair in a shipped Compose file is a name the loop must have."""
        for compose in COMPOSE_FILES:
            with self.subTest(compose=compose.name):
                self.assertEqual(file_backed_secrets(compose.read_text(encoding="utf-8"))
                                 - set(SECRET_NAMES), set())

    def test_every_loaded_secret_is_reachable_from_the_deployment(self):
        """The reverse: a name in the loop that no Compose file forwards is one nobody can set."""
        for name in SECRET_NAMES:
            with self.subTest(secret=name):
                self.assertTrue(any(name in file_backed_secrets(compose.read_text(encoding="utf-8"))
                                    for compose in COMPOSE_FILES),
                                "%s is loaded by the entrypoint but forwarded by no Compose file"
                                % name)

    def test_direct_value(self):
        for name in SECRET_NAMES:
            with self.subTest(secret=name):
                result = self.invoke(name, {name: "direct"})
                self.assertTrue(self.report(result, name)["matched"], result.stdout)

    def test_file_overrides_direct_and_trims_whitespace(self):
        for name in SECRET_NAMES:
            with self.subTest(secret=name):
                with tempfile.TemporaryDirectory() as directory:
                    secret = Path(directory) / "token"
                    secret.write_text("  synthetic-file-value\n", encoding="utf-8")
                    result = self.invoke(name, {name: "direct", name + "_FILE": str(secret)},
                                         "synthetic-file-value")
                self.assertTrue(self.report(result, name)["matched"], result.stdout)
                self.assertNotIn("synthetic-file-value", result.stdout + result.stderr)

    def test_a_utf8_bom_is_not_part_of_the_secret(self):
        """A Windows editor's byte-order mark is file metadata, not the token's first character.

        PowerShell 5.1's `utf8` encoding and Notepad both write a UTF-8 BOM, and this
        entrypoint's audience includes the Windows path it runs a child on. Read as plain
        utf-8 the mark becomes U+FEFF, which `.strip()` does not remove (it is not
        whitespace), so the loaded variable carried one leading character no consumer --
        the Hub for `HF_TOKEN`, `laya-serve`'s bearer check for `LAYA_API_KEY` -- accepts.
        """
        for name in SECRET_NAMES:
            with self.subTest(secret=name):
                with tempfile.TemporaryDirectory() as directory:
                    secret = Path(directory) / "token"
                    secret.write_bytes(b"\xef\xbb\xbfsynthetic-file-value\n")
                    result = self.invoke(name, {name + "_FILE": str(secret)}, "synthetic-file-value")
                self.assertTrue(self.report(result, name)["matched"], result.stdout)
                self.assertNotIn("synthetic-file-value", result.stdout + result.stderr)

    def test_a_file_that_is_only_a_bom_is_empty(self):
        """Stripping the BOM leaves nothing, and nothing is the empty-secret refusal, not a load."""
        for name in SECRET_NAMES:
            with self.subTest(secret=name):
                with tempfile.TemporaryDirectory() as directory:
                    secret = Path(directory) / "token"
                    secret.write_bytes(b"\xef\xbb\xbf")
                    result = self.invoke(name, {name + "_FILE": str(secret)})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(name + "_FILE", result.stderr)

    def test_file_variable_is_removed_before_the_command_runs(self):
        """What "and the `_FILE` variable is removed before the server execs" promises, per name."""
        for name in SECRET_NAMES:
            with self.subTest(secret=name):
                with tempfile.TemporaryDirectory() as directory:
                    secret = Path(directory) / "token"
                    secret.write_text("synthetic-file-value\n", encoding="utf-8")
                    result = self.invoke(name, {name + "_FILE": str(secret)}, "synthetic-file-value")
                self.assertFalse(self.report(result, name)["file_var"], result.stdout)

    def test_invalid_files_stop_before_child_without_disclosure(self):
        for name in SECRET_NAMES:
            with self.subTest(secret=name):
                with tempfile.TemporaryDirectory() as directory:
                    secret = Path(directory) / "private-filename"
                    for contents in (None, b"", b" \n", b"synthetic-secret\x00", b"\xff"):
                        with self.subTest(contents=contents):
                            if contents is not None:
                                secret.write_bytes(contents)
                            result = self.invoke(name, {name: "direct", name + "_FILE": str(secret)})
                            self.assertNotEqual(result.returncode, 0)
                            self.assertEqual(result.stdout, "")
                            self.assertIn(name + "_FILE", result.stderr)
                            self.assertNotIn(str(secret), result.stderr)
                            self.assertNotIn("synthetic-secret", result.stderr)
                            self.assertNotIn("direct", result.stderr)

    def test_empty_file_setting_and_unknown_suffix_are_ignored(self):
        for name in SECRET_NAMES:
            with self.subTest(secret=name):
                result = self.invoke(name, {name: "direct", name + "_FILE": "",
                                            "UNKNOWN_FILE": "/missing"})
                self.assertTrue(self.report(result, name)["matched"], result.stdout)

    def test_missing_command_fails(self):
        result = subprocess.run([sys.executable, str(ENTRYPOINT)], env={},
                                capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("command is required", result.stderr)

    def test_child_exit_code_propagates(self):
        """A child that exits 42 hands that status to the entrypoint's caller.

        The exec path keeps it by replacing the process; the Windows path runs the child and
        propagates `returncode`, which the `os.execvp` emulation never did -- it exited 0
        whatever the child did.
        """
        blocked = {prefix for secret in SECRET_NAMES for prefix in (secret, secret + "_FILE")}
        env = {key: value for key, value in os.environ.items() if key not in blocked}
        result = subprocess.run(
            [sys.executable, str(ENTRYPOINT), sys.executable, "-c", "import sys; sys.exit(42)"],
            env=env, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 42, result.stderr)


if __name__ == "__main__":
    unittest.main()
