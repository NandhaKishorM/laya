#!/usr/bin/env python3
"""Fail a JUnit XML test report in which any test skipped, failed, or went missing.

A lane that exits 0 having skipped its tests proves nothing, and GoogleTest and CTest both treat a
skip as success. This reads the report (GoogleTest `--gtest_output=xml:` or `ctest --output-junit`)
and counts instead of trusting the exit status.

    ctest --test-dir build/debug --output-junit junit.xml
    python laya-cpp/scripts/check_test_xml.py junit.xml --min-tests 2

Exit 0 only when every test case has status `run` (or none), none failed or errored, and at least
--min-tests ran. Any other status fails: ctest writes `disabled` (a DISABLED_ test or a DISABLED
property, with no <skipped> child) and `notrun` (a skip, with a <skipped> child); GoogleTest's own
XML writes `notrun` for a disabled test. Several report files may be given; counts add up. A report
over 16 MiB or with a DTD (any encoding) is refused.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET  # nosemgrep: use-defused-xml
import xml.parsers.expat as expat  # nosemgrep: use-defused-xml

# The stdlib parser is used with DTDs refused (see _refuse_dtd) so the checker stays standard library.

MAX_BYTES = 16 * 1024 * 1024


def _refuse(*_args):
    raise ValueError("a DTD or entity declaration is not allowed in a test report")


def _refuse_dtd(data):
    """Parse once with expat, which decodes the document first, so a UTF-16 report cannot hide a
    DOCTYPE from it the way it hides one from a byte search. XXE and entity bombs need a DTD."""
    guard = expat.ParserCreate()
    guard.StartDoctypeDeclHandler = _refuse
    guard.EntityDeclHandler = _refuse
    guard.UnparsedEntityDeclHandler = _refuse
    guard.Parse(data, True)


def summarise(path):
    """(tests, problems) for one report file. Raises ValueError on a malformed report."""
    try:
        if os.path.getsize(path) > MAX_BYTES:
            raise ValueError("%s is over %d bytes" % (path, MAX_BYTES))
        with open(path, "rb") as handle:
            data = handle.read()
        _refuse_dtd(data)
        root = ET.fromstring(data)
    except (ET.ParseError, expat.ExpatError, OSError) as err:
        raise ValueError("%s is not a readable XML report: %s" % (path, err))
    if root.tag not in ("testsuites", "testsuite"):
        raise ValueError("%s: unexpected root element <%s>" % (path, root.tag))
    tests, problems = 0, []
    for case in root.iter("testcase"):
        tests += 1
        name = "%s.%s" % (case.get("classname") or "", case.get("name") or "?")
        children = {child.tag for child in case}
        status = case.get("status")
        if "skipped" in children:
            problems.append("skipped: " + name)
        elif status == "fail":
            problems.append("failed: " + name)
        elif status not in (None, "run"):
            problems.append("not run (status=%s): %s" % (status, name))
        if "failure" in children or "error" in children:
            problems.append("failed: " + name)
    return tests, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", help="JUnit XML report file(s)")
    parser.add_argument("--min-tests", type=int, default=1,
                        help="fail when fewer tests than this ran")
    args = parser.parse_args(argv)
    if args.min_tests < 0:
        print("check_test_xml: --min-tests must not be negative", file=sys.stderr)
        return 1
    if len(set(os.path.realpath(r) for r in args.reports)) != len(args.reports):
        print("check_test_xml: the same report is listed twice; its tests would count twice",
              file=sys.stderr)
        return 1
    total, problems = 0, []
    for path in args.reports:
        if not os.path.isfile(path):
            print("check_test_xml: %s does not exist: the step that writes it did not run or "
                  "failed before producing it" % path, file=sys.stderr)
            return 1
        try:
            tests, found = summarise(path)
        except ValueError as err:
            print("check_test_xml: %s" % err, file=sys.stderr)
            return 1
        total += tests
        problems += found
    if total < args.min_tests:
        problems.append("only %d tests ran, floor is %d" % (total, args.min_tests))
    for problem in problems:
        print("check_test_xml: " + problem, file=sys.stderr)
    print("check_test_xml: %d tests, %d problems" % (total, len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
