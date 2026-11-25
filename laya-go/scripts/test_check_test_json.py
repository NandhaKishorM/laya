#!/usr/bin/env python3
"""The checker has to be able to fail, or it is decoration: prove each way it can."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "check_test_json.py")


def event(action, test=None, package="example.com/m/pkg"):
    out = {"Action": action, "Package": package}
    if test:
        out["Test"] = test
    return json.dumps(out)


def run(lines, *extra):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        handle.write("\n".join(lines) + "\n")
    try:
        done = subprocess.run([sys.executable, CHECKER, handle.name, *extra],
                              capture_output=True, text=True)
    finally:
        os.unlink(handle.name)
    return done.returncode, done.stdout + done.stderr


PASSING = [event("run", "TestA"), event("pass", "TestA"), event("run", "TestB"),
           event("pass", "TestB"), event("pass")]


class CheckerTest(unittest.TestCase):
    def test_a_failed_test_shows_its_own_output(self):
        said = json.dumps({"Action": "output", "Package": "example.com/m/pkg", "Test": "TestB",
                           "Output": "    b_test.go:7: got 3, want 4\n"})
        other = json.dumps({"Action": "output", "Package": "example.com/m/pkg", "Test": "TestA",
                            "Output": "    a_test.go:1: a passing test's log\n"})
        code, out = run([event("run", "TestA"), other, event("pass", "TestA"), event("run", "TestB"), said,
                         event("fail", "TestB"), event("fail")], "--min-tests", "2")
        self.assertEqual(code, 1, out)
        self.assertIn("b_test.go:7: got 3, want 4", out)
        self.assertNotIn("a passing test's log", out)

    def test_all_pass(self):
        code, out = run(PASSING, "--min-tests", "2")
        self.assertEqual(code, 0, out)
        self.assertIn("2 ran, 2 passed, 0 skipped, 0 failed", out)

    def test_a_skip_fails_the_run(self):
        code, out = run(PASSING + [event("run", "TestC"), event("skip", "TestC")])
        self.assertEqual(code, 1)
        self.assertIn("skipped: pkg TestC", out)

    def test_a_skipped_subtest_fails_the_run(self):
        code, out = run(PASSING + [event("skip", "TestA/sub")])
        self.assertEqual(code, 1, out)

    def test_a_failure_fails_the_run(self):
        code, out = run(PASSING + [event("fail", "TestD")])
        self.assertEqual(code, 1)
        self.assertIn("failed: pkg TestD", out)

    def test_a_failed_package_fails_the_run(self):
        code, _ = run(PASSING + [event("fail")])
        self.assertEqual(code, 1)

    def test_a_build_failure_fails_the_run(self):
        code, out = run(PASSING + [json.dumps({"Action": "build-fail", "ImportPath": "m/pkg"})])
        self.assertEqual(code, 1)
        self.assertIn("build failed", out)

    def test_the_floor_catches_a_suite_that_stopped_registering(self):
        code, out = run(PASSING, "--min-tests", "3")
        self.assertEqual(code, 1)
        self.assertIn("only 2 tests ran", out)

    def test_an_empty_stream_is_not_a_pass(self):
        code, _ = run([], "--min-tests", "1")
        self.assertEqual(code, 1)

    def test_a_rerun_that_passes_after_failing_counts_its_last_word(self):
        code, _ = run([event("fail", "TestA"), event("pass", "TestA")])
        self.assertEqual(code, 0)

    def test_raw_text_between_events_is_tolerated(self):
        code, _ = run(["go: downloading example.com/x v1.0.0"] + PASSING)
        self.assertEqual(code, 0)

    def test_a_missing_file_says_the_test_step_did_not_run(self):
        done = subprocess.run([sys.executable, CHECKER, "/nonexistent/go-test.json"],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)
        self.assertIn("did not run", done.stderr)
        self.assertNotIn("Traceback", done.stderr)


if __name__ == "__main__":
    unittest.main()
