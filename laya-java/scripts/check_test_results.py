#!/usr/bin/env python3
"""Fail a Java test run that only looks green because the model-backed tests never ran.

The checkpoint-backed parity factories abort by assumption when `LAYA_CHECKPOINTS` or
`LAYA_ONNX_GRAPH` is unset, and an aborted test is not a failure to Gradle: a run with a broken
artifacts directory ends with BUILD SUCCESSFUL and zero parity coverage. In CI that turns the whole
lane into a rubber stamp, which is exactly how a wrong answer can sit behind a green suite.

Exits non-zero when

  * any test failed or errored,
  * a test was skipped or aborted and `--allow-aborted` was not passed, or
  * fewer than `--min-tests` tests ran at all -- which catches a filter that matched nothing and a
    factory that silently stopped registering its cases.

Usage:
    # the lane that deliberately has no checkpoints: aborting the parity factories is expected
    python laya-java/scripts/check_test_results.py <results-dir> --min-tests 80 --allow-aborted

    # the lane that exported the checkpoints: nothing may abort
    python laya-java/scripts/check_test_results.py <results-dir> --min-tests 200
"""
import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ElementTree


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="a directory of JUnit XML files, or one such file")
    parser.add_argument("--min-tests", type=int, default=1,
                        help="fail if fewer than this many tests ran")
    parser.add_argument("--allow-aborted", action="store_true",
                        help="permit skipped/aborted tests, for a lane with no checkpoints")
    parser.add_argument("--require-passed", action="append", default=[], metavar="CLASS.TEST",
                        help="a substring that must match at least one test that actually PASSED; "
                             "repeatable. For a guard that is conditionally disabled, where being "
                             "skipped would otherwise satisfy --allow-aborted")
    args = parser.parse_args(argv)

    if os.path.isdir(args.results):
        files = sorted(glob.glob(os.path.join(args.results, "**", "*.xml"), recursive=True))
    elif os.path.isfile(args.results):
        files = [args.results]
    else:
        # Reached whenever an earlier step failed and the test step was skipped, because this
        # check runs under `if: !cancelled()`. Crashing with a FileNotFoundError here buried the
        # real failure under a stack trace from the wrong script.
        print("check_test_results: %r does not exist, so no tests ran. The failure is in an "
              "earlier step -- look there, not here." % args.results, file=sys.stderr)
        return 1
    if not files:
        print("check_test_results: no JUnit XML under %r; the test task did not run"
              % args.results, file=sys.stderr)
        return 1

    tests = failures = errors = skipped = 0
    broken, aborted, passed = [], [], []
    for path in files:
        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError as problem:
            print("check_test_results: %s is not valid XML: %s" % (path, problem), file=sys.stderr)
            return 1
        suites = [root] if root.tag == "testsuite" else root.iter("testsuite")
        for suite in suites:
            tests += int(suite.get("tests", 0))
            failures += int(suite.get("failures", 0))
            errors += int(suite.get("errors", 0))
            skipped += int(suite.get("skipped", 0))
            for case in suite.iter("testcase"):
                name = "%s.%s" % (case.get("classname", "?"), case.get("name", "?"))
                if case.find("failure") is not None or case.find("error") is not None:
                    broken.append(name)
                elif case.find("skipped") is not None:
                    reason = case.find("skipped").get("message") or ""
                    aborted.append("%s (%s)" % (name, reason.strip()[:120]))
                else:
                    passed.append(name)

    print("check_test_results: %d tests, %d failed, %d errored, %d skipped, across %d file(s)"
          % (tests, failures, errors, skipped, len(files)))
    problems = []
    if broken:
        problems.append("%d test(s) failed or errored" % len(broken))
        for name in broken[:20]:
            print("  FAILED  " + name, file=sys.stderr)
    if aborted and not args.allow_aborted:
        problems.append("%d test(s) were skipped or aborted" % len(aborted))
        for name in aborted[:20]:
            print("  SKIPPED " + name, file=sys.stderr)
        print("  a checkpoint-backed test that aborts is not a pass: the artifacts were expected "
              "to be present in this lane", file=sys.stderr)
    elif aborted:
        print("  %d test(s) aborted, which this lane allows:" % len(aborted))
        for name in aborted[:5]:
            print("    " + name)
    # Executed, not merely registered. The JUnit `tests` attribute COUNTS SKIPPED tests, so a
    # suite in which every single test aborted satisfied this floor and both --allow-aborted
    # lanes -- the build gate and the Maven Central release gate -- went green with nothing
    # actually run. That is the exact failure this script's docstring says it exists to catch.
    executed = tests - skipped
    if executed < args.min_tests:
        problems.append("only %d of %d tests actually ran (%d skipped), expected at least %d"
                        % (executed, tests, skipped, args.min_tests))

    # A floor cannot protect a guard that is allowed to be absent. `--allow-aborted` exists so a
    # lane without the checkpoints can still gate on its model-free tests, but it also forgives a
    # CONDITIONALLY disabled test -- and a test disabled because the thing it checks was never
    # switched on is indistinguishable, to this script, from one that passed. The JDK matrix is
    # exactly that case: if `-PtestJavaVersion` stops reaching the build, no launcher override
    # happens, the one test that would notice is skipped rather than failed, and both cells go
    # green having run the compile JDK twice. Naming it here turns "silently absent" into a
    # failure.
    for wanted in args.require_passed:
        if not any(wanted in name for name in passed):
            where = ("it was skipped" if any(wanted in name for name in aborted)
                     else "it failed" if any(wanted in name for name in broken)
                     else "it did not run at all")
            problems.append("no PASSING test matched %r -- %s. This lane requires that test to "
                            "run, because it is the check that the lane is not vacuous."
                            % (wanted, where))

    if problems:
        for line in problems:
            print("check_test_results: " + line, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
