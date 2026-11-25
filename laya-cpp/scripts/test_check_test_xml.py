#!/usr/bin/env python3
"""The checker has to be able to fail, or it is decoration: prove each way it can."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "check_test_xml.py")


def report(*cases):
    return '<?xml version="1.0"?><testsuites><testsuite name="s">%s</testsuite></testsuites>' % "".join(cases)


PASS = '<testcase name="%s" classname="S" status="run"/>'
SKIP = '<testcase name="%s" classname="S" status="run"><skipped message="x"/></testcase>'
FAIL = '<testcase name="%s" classname="S" status="run"><failure message="x"/></testcase>'
NOTRUN = '<testcase name="%s" classname="S" status="notrun"/>'


def run(text, *extra):
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
        handle.write(text)
    try:
        done = subprocess.run([sys.executable, CHECKER, handle.name, *extra],
                              capture_output=True, text=True)
    finally:
        os.unlink(handle.name)
    return done.returncode, done.stdout + done.stderr


class CheckerTest(unittest.TestCase):
    def test_all_pass_meets_floor_exactly(self):
        code, out = run(report(PASS % "a", PASS % "b"), "--min-tests", "2")
        self.assertEqual(code, 0, out)

    def test_one_below_floor_fails(self):
        code, out = run(report(PASS % "a", PASS % "b"), "--min-tests", "3")
        self.assertEqual(code, 1)
        self.assertIn("floor is 3", out)

    def test_skip_fails(self):
        code, out = run(report(PASS % "a", SKIP % "b"))
        self.assertEqual(code, 1)
        self.assertIn("skipped: S.b", out)

    def test_disabled_test_counts_as_skip(self):
        code, _ = run(report(PASS % "a", NOTRUN % "b"))
        self.assertEqual(code, 1)

    def test_failure_fails(self):
        code, out = run(report(FAIL % "a"))
        self.assertEqual(code, 1)
        self.assertIn("failed: S.a", out)

    def test_malformed_xml_fails(self):
        code, out = run("<testsuites><testsuite>")
        self.assertEqual(code, 1)
        self.assertIn("not a readable XML report", out)

    def test_dtd_is_refused(self):
        code, out = run('<!DOCTYPE x [<!ENTITY a "b">]>' + report(PASS % "a"))
        self.assertEqual(code, 1)
        self.assertIn("DTD", out)

    def test_dtd_in_utf16_is_refused(self):
        text = '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE x [<!ENTITY a "b">]>' + report(PASS % "a")
        with tempfile.NamedTemporaryFile("wb", suffix=".xml", delete=False) as handle:
            handle.write(text.encode("utf-16"))
        try:
            done = subprocess.run([sys.executable, CHECKER, handle.name], capture_output=True, text=True)
        finally:
            os.unlink(handle.name)
        self.assertEqual(done.returncode, 1)
        self.assertIn("DTD", done.stdout + done.stderr)

    def test_status_fail_without_child_fails(self):
        code, out = run(report('<testcase name="a" classname="S" status="fail"/>'))
        self.assertEqual(code, 1)
        self.assertIn("failed: S.a", out)

    def test_error_child_fails(self):
        code, out = run(report('<testcase name="a" classname="S" status="run"><error/></testcase>'))
        self.assertEqual(code, 1)
        self.assertIn("failed: S.a", out)

    def test_root_must_be_testsuite(self):
        code, out = run('<other><testcase name="a" classname="S" status="run"/></other>')
        self.assertEqual(code, 1)
        self.assertIn("unexpected root", out)

    def test_real_ctest_disabled_and_skipped_fail(self):
        fixture = os.path.join(HERE, "fixtures", "ctest-disabled-and-skipped.xml")
        done = subprocess.run([sys.executable, CHECKER, fixture], capture_output=True, text=True)
        out = done.stdout + done.stderr
        self.assertEqual(done.returncode, 1)
        self.assertIn("status=disabled", out)
        self.assertIn("skipped: Extra.Skipped", out)

    def test_disabled_status_alone_fails(self):
        code, out = run(report(PASS % "a", '<testcase name="b" classname="S" status="disabled"/>'))
        self.assertEqual(code, 1)
        self.assertIn("status=disabled", out)

    def test_oversize_report_fails(self):
        big = report(PASS % "a") + "<!--" + "x" * (16 * 1024 * 1024) + "-->"
        code, out = run(big)
        self.assertEqual(code, 1)
        self.assertIn("over", out)

    def test_duplicate_report_path_fails(self):
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
            handle.write(report(PASS % "a"))
        try:
            done = subprocess.run([sys.executable, CHECKER, handle.name, handle.name],
                                  capture_output=True, text=True)
        finally:
            os.unlink(handle.name)
        self.assertEqual(done.returncode, 1)
        self.assertIn("twice", done.stderr)

    def test_negative_floor_fails(self):
        code, out = run(report(PASS % "a"), "--min-tests", "-1")
        self.assertEqual(code, 1)
        self.assertIn("negative", out)

    def test_wrong_root_fails(self):
        code, _ = run("<html/>")
        self.assertEqual(code, 1)

    def test_missing_file_fails(self):
        done = subprocess.run([sys.executable, CHECKER, "/nonexistent/x.xml"],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)

    def test_empty_report_fails_default_floor(self):
        code, _ = run(report())
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
