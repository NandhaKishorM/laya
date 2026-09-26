"""`laya-evals`: run a labelled evaluation, validate a dataset, or compare a report.

Exit codes: 0 on success, 1 when a threshold or a baseline tolerance fails, 2 on a usage error.

    laya-evals validate research/evals/fixture.jsonl
    laya-evals run research/evals/fixture.jsonl --model english --min-accuracy 0.8 --max-ece 0.1
    laya-evals run data.jsonl --baseline baseline.json --tolerance choice_accuracy=0.02 --json out.json

`laya eval ...` dispatches here from the main CLI, so both spellings work.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import evals
from .evals import EvalError


class RouterRunner:
    """Adapt a `Router` to the harness: one predict, or a batch of identical questions."""

    def __init__(self, router: Any):
        self.router = router

    def predict(self, state: Any, questions: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        return self.router.predict(state, questions, model=model)

    def predict_batch(self, states: Sequence[Any], questions: Dict[str, Any],
                      model: Optional[str] = None, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        requests = [{"state": state, "questions": questions, "model": model} for state in states]
        return self.router.predict_batch(requests, batch_size=batch_size)


def _parse_pairs(pairs: Optional[Sequence[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for pair in pairs or []:
        name, _, raw = pair.partition("=")
        if not name or not raw:
            raise EvalError("expected NAME=VALUE, got %r" % pair)
        try:
            out[name.strip()] = float(raw)
        except ValueError:
            raise EvalError("%r is not a number in %r" % (raw, pair))
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laya-evals",
                                     description="Evaluate a Laya checkpoint on a labelled dataset.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="check a dataset file without running a model")
    validate.add_argument("dataset")

    run = sub.add_parser("run", help="evaluate a dataset and apply thresholds")
    run.add_argument("dataset")
    run.add_argument("--model", help="force a checkpoint instead of auto-routing")
    run.add_argument("--device", help="torch device, e.g. cpu or cuda")
    run.add_argument("--batch-size", type=int, help="examples per forward pass when questions match")
    run.add_argument("--on-error", choices=("fail", "skip"), default="fail")
    run.add_argument("--baseline", help="a baseline report JSON to compare against")
    run.add_argument("--tolerance", action="append", metavar="METRIC=VALUE",
                     help="allowed absolute drift from the baseline; repeatable")
    run.add_argument("--min-accuracy", type=float,
                     help="minimum accuracy (choice, else noul) for the whole dataset")
    run.add_argument("--max-ece", type=float, help="maximum expected calibration error")
    run.add_argument("--min", action="append", metavar="METRIC=VALUE", help="minimum for any metric")
    run.add_argument("--max", action="append", metavar="METRIC=VALUE", help="maximum for any metric")
    run.add_argument("--slice", action="append", choices=("language", "model", "qid", "tag"),
                     help="also report this slice dimension; repeatable")
    run.add_argument("--json", dest="json_out", help="write the full report JSON here")
    run.add_argument("--markdown", dest="markdown_out", help="write a Markdown summary here")

    compare = sub.add_parser("compare", help="compare a report JSON against a baseline")
    compare.add_argument("report")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--tolerance", action="append", metavar="METRIC=VALUE",
                         help="allowed absolute drift; repeatable")

    return parser


def _load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_thresholds(overall: Dict[str, float], mins: Dict[str, float],
                      maxs: Dict[str, float]) -> List[str]:
    failures: List[str] = []
    if "choice_accuracy" not in overall and "noul_accuracy" in overall:
        overall = dict(overall, choice_accuracy=overall["noul_accuracy"])
    for name, limit in mins.items():
        value = overall.get(name)
        if value is None:
            failures.append("metric %r is not in the report" % name)
        elif value < limit:
            failures.append("%s=%.4f is below the minimum %.4f" % (name, value, limit))
    for name, limit in maxs.items():
        value = overall.get(name)
        if value is None:
            failures.append("metric %r is not in the report" % name)
        elif value > limit:
            failures.append("%s=%.4f is above the maximum %.4f" % (name, value, limit))
    return failures


def _cmd_validate(args) -> int:
    dataset = evals.Dataset.from_jsonl(args.dataset)
    questions = sorted({qid for example in dataset.examples for qid in example.questions})
    print("%d examples, %d question id(s): %s" % (len(dataset), len(questions), ", ".join(questions)))
    return 0


def _cmd_run(args) -> int:
    import laya

    dataset = evals.Dataset.from_jsonl(args.dataset)
    if args.model:
        for example in dataset.examples:      # --model is authoritative over per-row model
            example.model = args.model
    runner: Any = RouterRunner(laya.Router(device=args.device, preload=False))
    config = {"dataset": args.dataset, "model": args.model, "device": args.device}
    report = evals.evaluate(runner, dataset, batch_size=args.batch_size,
                            on_error=args.on_error, config=config)

    mins = _parse_pairs(args.min)
    maxs = _parse_pairs(args.max)
    if args.min_accuracy is not None:
        key = "choice_accuracy" if "choice_accuracy" in report.overall else "noul_accuracy"
        mins[key] = args.min_accuracy
    if args.max_ece is not None:
        maxs["ece"] = args.max_ece

    failures = _check_thresholds(report.overall, mins, maxs)
    if args.baseline:
        baseline = _load_report(args.baseline)
        ok, deltas = report.compare(baseline, _parse_pairs(args.tolerance))
        for metric, delta in sorted(deltas.items()):
            print("%-18s baseline=%.4f value=%.4f diff=%+.4f (tol %.4f)"
                  % (metric, delta["baseline"], delta["value"], delta["diff"], delta["tolerance"]))
        if not ok:
            failures.append("baseline comparison failed")

    for name in sorted(report.overall):
        print("%-18s %.4f" % (name, report.overall[name]))
    for dimension in args.slice or []:
        for value, metrics in sorted(report.slices.get(dimension, {}).items()):
            print("  %s=%s  %s" % (dimension, value,
                                   " ".join("%s=%.4f" % (k, v) for k, v in sorted(metrics.items()))))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report.to_json(), handle, ensure_ascii=False, indent=2, sort_keys=True)
    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as handle:
            handle.write(report.to_markdown())

    if failures:
        for failure in failures:
            print("FAIL: " + failure, file=sys.stderr)
        return 1
    return 0


def _cmd_compare(args) -> int:
    report = evals.EvalReport(**{k: v for k, v in _load_report(args.report).items()
                                 if k in ("config", "overall", "slices", "cases")})
    baseline = _load_report(args.baseline)
    ok, deltas = report.compare(baseline, _parse_pairs(args.tolerance))
    for metric, delta in sorted(deltas.items()):
        print("%-18s baseline=%.4f value=%.4f diff=%+.4f (tol %.4f)"
              % (metric, delta["baseline"], delta["value"], delta["diff"], delta["tolerance"]))
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "run":
            return _cmd_run(args)
        return _cmd_compare(args)
    except EvalError as exc:
        print("laya-evals: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
