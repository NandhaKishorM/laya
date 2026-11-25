"""Fail closed on cargo test skips, failures, empty or incomplete output.

Capture BOTH stdout and stderr with `cargo test --workspace --locked --
--nocapture --test-threads=1`. Rust's fixture helpers return early (libtest counts
that as a pass), so their `skip:` diagnostics must also be checked.

    python laya-rust/tools/check_test_skips.py cargo-test.log --min-passed 1
"""

import argparse
import re
from pathlib import Path

SUMMARY = re.compile(
    r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"(\d+) measured; (\d+) filtered out;", re.MULTILINE,
)
SKIP = re.compile(r"\bskip(?:ped)?\s*:", re.IGNORECASE)
ERROR = re.compile(r"^(?:error:|error\[|error: test failed)", re.MULTILINE)
RUN = re.compile(r"^\s*(?:Running .+ \(.+\)|Doc-tests .+)$", re.MULTILINE)


def check(text, minimum):
    summaries = SUMMARY.findall(text)
    passed = sum(int(s[1]) for s in summaries)
    failed = sum(int(s[2]) for s in summaries)
    ignored = sum(int(s[3]) for s in summaries)
    measured = sum(int(s[4]) for s in summaries)
    filtered = sum(int(s[5]) for s in summaries)
    skips = [line for line in text.splitlines() if SKIP.search(line)]
    print("%d passed; %d failed; %d ignored; %d skip diagnostics; %d filtered" %
          (passed, failed, ignored, len(skips), filtered))
    for line in skips:
        print(line)
    # Filtered or measured tests are not an all-tests parity run either.
    unknown = len(re.findall(r"^test result:", text, re.MULTILINE)) != len(summaries)
    incomplete = len(RUN.findall(text)) > len(summaries)
    bad = (not summaries or unknown or incomplete or passed < minimum or failed or ignored or measured or filtered
           or skips or ERROR.search(text) or any(s[0] != "ok" for s in summaries))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--min-passed", type=int, default=1)
    args = ap.parse_args()
    if args.min_passed < 1:
        ap.error("--min-passed must be positive")
    try:
        text = Path(args.log).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        print("Cannot read cargo test output: %s" % e)
        return 1
    return check(text, args.min_passed)


if __name__ == "__main__":
    raise SystemExit(main())
