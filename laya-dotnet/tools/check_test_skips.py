"""Fail a .NET test run that only looks green because model-backed tests were skipped.

The test project skips (rather than fails) every test whose ONNX artifacts are missing, so a
run with a broken or empty artifacts directory still ends with "Passed!". In CI that would turn
the whole parity lane into a rubber stamp. This reads the TRX file `dotnet test` wrote and exits
non-zero when

  * any test failed, errored, timed out or was aborted,
  * a test was skipped for a reason that is not an expected one (see below), or
  * fewer than `--min-passed` tests passed (a wrong filter or a missing test project).

A skip is expected only when a missing-artifact reason names a checkpoint that this job deliberately has no
artifacts for (`--have` lists the ones it does have), or it is the router end-to-end skip and the
job lacks english or multilingual. Everything else, including every skip that names a checkpoint
the job *should* have, is reported and fails the run. The reason strings come from the skip
messages in tests/Laya.Tests; if one of them is reworded this fails closed, not open.

Usage:
    python laya-dotnet/tools/check_test_skips.py TestResults/run.trx --have english --min-passed 50

    # The model-free subset (model-backed classes filtered out): nothing at all may be skipped
    python laya-dotnet/tools/check_test_skips.py TestResults/run.trx --have english multilingual typed-decisions --min-passed 600
"""

import argparse
import collections
import re
import sys
import xml.etree.ElementTree as ET

NS = "{http://microsoft.com/schemas/VisualStudio/TeamTest/2010}"
FAILING = {"Failed", "Error", "Timeout", "Aborted", "Disconnected", "NotRunnable"}
CHECKPOINT_NAMES = {"english": "english", "multilingual": "multilingual", "typeddecisions": "typed-decisions"}
QUOTED = re.compile(r"'(English|Multilingual|TypedDecisions)'")
ROUTER = "router end-to-end tests need both english and multilingual"


def reason_of(result) -> str:
    out = result.find(NS + "Output")
    return " ".join("".join(out.itertext()).split()) if out is not None else ""


def expected_skip(reason: str, have: set) -> bool:
    named = {CHECKPOINT_NAMES[m.lower()] for m in QUOTED.findall(reason)}
    missing_artifact = any(message in reason for message in (
        "no tokenizer.json for ", "no model.onnx for ", "no ONNX artifacts for ",
        "no split ONNX artifact for ",
    ))
    if named and missing_artifact:
        return not (named & have)
    if ROUTER in reason:
        return not {"english", "multilingual"} <= have
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trx", nargs="+", help="TRX file(s) written by `dotnet test --report-xunit-trx`.")
    ap.add_argument(
        "--have",
        nargs="*",
        default=[],
        choices=sorted(set(CHECKPOINT_NAMES.values())),
        help="Checkpoints whose artifacts this job provides; their tests must run, not skip.",
    )
    ap.add_argument("--min-passed", type=int, default=1, help="Fail below this many passed tests (default: 1).")
    args = ap.parse_args()
    have = set(args.have)

    passed = failed = 0
    allowed = collections.Counter()
    bad_skips = collections.Counter()
    failures = []
    for path in args.trx:
        for r in ET.parse(path).getroot().iter(NS + "UnitTestResult"):
            outcome = r.get("outcome")
            if outcome == "Passed":
                passed += 1
            elif outcome in FAILING:
                failed += 1
                failures.append(r.get("testName", "<unnamed>"))
            elif outcome == "NotExecuted":
                reason = reason_of(r)
                if expected_skip(reason, have):
                    allowed[reason] += 1
                else:
                    bad_skips[(r.get("testName", "<unnamed>").split("(")[0], reason)] += 1
            else:
                bad_skips[(r.get("testName", "<unnamed>").split("(")[0], "outcome " + str(outcome))] += 1

    n_bad = sum(bad_skips.values())
    print("passed: %d  failed: %d  skipped (expected): %d  skipped (UNEXPECTED): %d"
          % (passed, failed, sum(allowed.values()), n_bad))
    print("checkpoints expected to run: %s" % (", ".join(sorted(have)) or "none"))
    for reason, n in allowed.most_common():
        print("  expected skip x%d: %s" % (n, reason[:140]))

    ok = True
    if failed:
        ok = False
        print("\nFAIL: %d test(s) failed, e.g. %s" % (failed, ", ".join(failures[:5])))
    if n_bad:
        ok = False
        print("\nFAIL: %d test(s) were skipped that should have run (missing or unusable artifacts?):" % n_bad)
        for (name, reason), n in bad_skips.most_common(10):
            print("  x%d %s: %s" % (n, name, reason[:200]))
    if passed < args.min_passed:
        ok = False
        print("\nFAIL: only %d test(s) passed, expected at least %d" % (passed, args.min_passed))
    if ok:
        print("\nOK: nothing that should have run was skipped")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
