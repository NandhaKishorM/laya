#!/usr/bin/env python3
"""Fail a `go test -json` run in which any test skipped, failed, or went missing.

A parity lane that exits 0 having skipped its checkpoint-backed tests proves nothing: the same
suite is green on a machine with no checkpoints at all. `go test` reports a skip as success, so
the exit status cannot be the gate. This reads the JSON event stream and counts instead.

    go test -json ./... > test.json; python laya-go/scripts/check_test_json.py test.json --min-tests 200

Exit 0 only when no test skipped, none failed, no package failed to build, and at least
--min-tests tests ran. The floor catches a suite that quietly stopped registering its cases.
"""
import argparse
import json
import os
import sys


def read_events(handle):
    for number, line in enumerate(handle, 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # `go test -json` also relays build errors and `go: downloading` notices as raw text.
            yield {"Action": "output", "Output": line + "\n", "Raw": number}


# How much of a failed test's own output is shown. The lane writes `go test -json` to a file, so
# without this a failure in CI names the test and says nothing about why.
OUTPUT_LINES = 40


def summarise(events):
    """(counts, problems, output of each failed test) for a stream of test2json events."""
    final = {}
    problems = []
    output = {}
    for event in events:
        action, package, test = event.get("Action"), event.get("Package", ""), event.get("Test")
        if action == "output" and test is not None:
            lines = output.setdefault((package, test), [])
            lines.append(event.get("Output", "").rstrip("\n"))
            del lines[:-OUTPUT_LINES]
        if action in ("build-fail",):
            problems.append("build failed: %s" % event.get("ImportPath", package))
        if test is None:
            if action == "fail":
                problems.append("package failed: %s" % package)
            continue
        if action in ("pass", "fail", "skip"):
            final[(package, test)] = action
    counts = {"pass": 0, "fail": 0, "skip": 0}
    for (package, test), action in sorted(final.items()):
        counts[action] += 1
        if action == "skip":
            problems.append("skipped: %s %s" % (package.rsplit("/", 1)[-1], test))
        elif action == "fail":
            problems.append("failed: %s %s" % (package.rsplit("/", 1)[-1], test))
    failed = {key: output.get(key, []) for key, action in sorted(final.items()) if action == "fail"}
    return counts, problems, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("events", help="file holding `go test -json` output, or - for stdin")
    parser.add_argument("--min-tests", type=int, default=1,
                        help="fail when fewer tests than this ran (tests and subtests both count)")
    args = parser.parse_args(argv)
    if args.events != "-" and not os.path.isfile(args.events):
        # The step that writes this file is a separate one and an earlier step's failure skips it.
        print("check_test_json: %s does not exist: the `go test` step that writes it did not run "
              "or failed before producing it; see that step's log" % args.events, file=sys.stderr)
        return 1
    handle = sys.stdin if args.events == "-" else open(args.events, encoding="utf-8")
    with handle:
        counts, problems, failed = summarise(read_events(handle))
    total = sum(counts.values())
    print("go tests: %d ran, %d passed, %d skipped, %d failed"
          % (total, counts["pass"], counts["skip"], counts["fail"]))
    if total < args.min_tests:
        problems.append("only %d tests ran, expected at least %d" % (total, args.min_tests))
    for problem in problems[:50]:
        print("  " + problem, file=sys.stderr)
    if len(problems) > 50:
        print("  ... and %d more" % (len(problems) - 50), file=sys.stderr)
    for (package, test), lines in list(failed.items())[:10]:
        print("--- output of %s %s (last %d lines)" % (package.rsplit("/", 1)[-1], test, len(lines)),
              file=sys.stderr)
        for line in lines:
            print("    " + line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
