"""Compare a fresh `laya_eval` report to a committed baseline and fail on drift.

Used by the scheduled `evals` workflow: it runs `research/eval/laya_eval.py`, then points this
at the result and a baseline in `research/results/`. The comparison uses
`laya.evals.EvalReport.compare`, so the tolerance semantics match the CLI.

    python research/evals/check_regression.py fresh.json baseline.json thresholds.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from laya.evals import EvalReport  # noqa: E402

METRICS = ("accuracy", "ece", "mean_confidence")


def _overall(document: Dict[str, Any], language: str, where: str) -> Dict[str, float]:
    report = (document or {}).get("report") or {}
    if language not in report:
        raise SystemExit("%s has no report for language %r" % (where, language))
    entry = report[language]
    return {metric: float(entry[metric]) for metric in METRICS if entry.get(metric) is not None}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh", help="the report produced by research/eval/laya_eval.py")
    parser.add_argument("baseline", help="a committed baseline report")
    parser.add_argument("thresholds", nargs="?", help="thresholds.json (languages + tolerance)")
    parser.add_argument("--lang", action="append", help="language to check; repeatable")
    args = parser.parse_args(argv)

    config: Dict[str, Any] = {}
    if args.thresholds:
        with open(args.thresholds, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    languages = args.lang or config.get("languages") or ["en"]
    tolerances = config.get("tolerance") or {}
    with open(args.fresh, "r", encoding="utf-8") as handle:
        fresh = json.load(handle)
    with open(args.baseline, "r", encoding="utf-8") as handle:
        baseline = json.load(handle)

    ok = True
    for language in languages:
        report = EvalReport(overall=_overall(fresh, language, args.fresh))
        base = {"overall": _overall(baseline, language, args.baseline)}
        passed, deltas = report.compare(base, tolerances)
        for metric, delta in sorted(deltas.items()):
            print("%-6s %-16s baseline=%.4f value=%.4f diff=%+.4f (tol %.4f)"
                  % (language, metric, delta["baseline"], delta["value"], delta["diff"], delta["tolerance"]))
        if not passed:
            ok = False
            print("%-6s REGRESSED" % language, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
