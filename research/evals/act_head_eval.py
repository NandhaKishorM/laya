"""Score `act_probability` against labelled decisions, as a bar for a future act head.

`answer.action.act_probability` is what the released checkpoints emit from an act head they
were never trained to use, and #185 records that it sits at ~1.0 on almost every input and
that the raw signal runs *against* correctness. This harness does not try to fix that, and
it does not claim the signal is useful today. Its only job is to make the question
measurable, so that a retrained head has a number to beat:

    **Does `act_probability` rank correct decisions above incorrect ones on a labelled
    evaluation set?**

## Why this reads a report instead of running a model

`laya-evals run --json REPORT` already writes, per case, the full `answer` (which carries
`answer.action.act_probability`), the calibrated `confidence`, and the binary `correct`.
So the diagnostic is pure arithmetic over a report that already exists:

* no checkpoint is loaded and no forward pass is run a second time;
* model, routing and eval code are untouched, and `laya.evals` gains no public API;
* any future checkpoint or retraining run produces an apples-to-apples number, because the
  input format is the one the harness already commits.

## Case selection

Every input case lands in exactly one bucket, and nothing is silently folded into a
denominator:

| bucket | meaning |
|---|---|
| `paired` | usable for both AUCs |
| `no_binary_correctness` | `correct` is not a bool -- a `score` answer, where correctness is not unambiguous |
| `missing_act_probability` | `action.act_probability` absent, not a number, or not finite |
| `missing_confidence` | `correct` and `act_probability` fine, but `confidence` absent, not a number, or not finite |
| `malformed_case` | the entry is not a case object at all |

The `paired` set is used for **both** the `act_probability` AUC and the `confidence`
control, so the comparator covers exactly the same decisions by construction rather than by
convention. A `score` answer is skipped because `laya.evals` deliberately leaves `correct`
null for it; there is no canonical binary definition for a score, and inventing one here
would be a different experiment.

## Metric

`roc_auc` is the tie-correct ROC-AUC, the Mann-Whitney U form with mid-ranks for tied
scores. It agrees with `sklearn.metrics.roc_auc_score` but needs only numpy, which the
metric math in `laya.evals` already depends on. Concretely:

    AUC = (sum of mid-ranks of the correct cases - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

Ties get the average of the ranks they span, so a tied correct case is not credited with
beating a tied incorrect one; with every score tied and both classes present the result is
exactly 0.5.

ROC-AUC is undefined when one class is absent, and this harness reports `null` with a
`auc_null_reason` instead of inventing a number. A constant or near-constant
`act_probability` cannot rank anything, so the summary also reports the observed minimum,
maximum and unique-value count plus a `saturated` flag: a signal pinned at one value is a
failed diagnostic, not a 0.5 result to be averaged away.

A number is only accepted when it is finite. `json.load` reads the non-standard `NaN` /
`Infinity` / `-Infinity` tokens by default, so a malformed report can carry them in, and
ranking one returns a confident-looking number that means nothing -- 0.5 for `NaN` and
`+inf`, 0.0 for `-inf`, where 0.0 is easily misread as a real anti-correlated signal. They
are bucketed as missing evidence instead, `roc_auc` raises `ValueError` if called directly
with one, and the JSON writer sets `allow_nan=False` so none can leave this tool.

    python research/evals/act_head_eval.py report.json [--json out.json] [--markdown out.md]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SCHEMA = "laya-evals-report/act-head-eval/1"

# Below this spread every act_probability is the same number to the precision the report
# stores (4 decimals), so the signal cannot separate anything.
_SATURATION_EPS = 1e-9

_UNSCORABLE = "one class is absent: ROC-AUC is undefined"
_NO_CASES = "no scorable cases"


def _is_number(value: Any) -> bool:
    """True for a finite real number.

    ``bool`` is an ``int`` subclass and is not a measurement. NaN and the infinities are
    excluded too: ``json.load`` accepts the non-standard ``NaN`` / ``Infinity`` /
    ``-Infinity`` tokens, so a malformed report can carry them in, and they are not
    evidence of anything.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Tie-correct ROC-AUC of ``scores`` against binary ``labels``.

    None when either class is absent, or when there is nothing to score: ROC-AUC needs
    both a positive and a negative case to be defined at all.

    Raises ValueError on a non-finite score. This function is callable on its own, and
    ranking NaN or an infinity yields a confident-looking number -- 0.5 for NaN and for
    ``+inf``, 0.0 for ``-inf`` -- where 0.0 reads as "the signal is perfectly
    anti-correlated" rather than "the input was not a measurement". Refusing is the only
    honest answer.
    """
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("scores and labels must be the same length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must all be finite numbers")
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    # Mid-ranks over the sorted scores, so a group of tied scores shares the average of
    # the 1-based ranks it spans.
    order = np.argsort(s, kind="mergesort")
    ordered = s[order]
    ranks = np.empty(s.shape[0], dtype=float)
    start = 0
    while start < ordered.shape[0]:
        end = start
        while end + 1 < ordered.shape[0] and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1

    rank_sum = float(ranks[y].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _case_act_probability(case: Dict[str, Any]) -> Any:
    answer = case.get("answer")
    if not isinstance(answer, dict):
        return None
    action = answer.get("action")
    if not isinstance(action, dict):
        return None
    return action.get("act_probability")


def analyse_cases(cases: Sequence[Any]) -> Dict[str, Any]:
    """Score one report's cases, bucketing every case exactly once.

    Returns a dict that is JSON-serialisable as-is: counts, both AUCs (or ``None`` with a
    reason), and the spread of ``act_probability`` needed to spot a saturated signal.
    """
    paired: List[Dict[str, Any]] = []
    skipped = {
        "no_binary_correctness": 0,
        "missing_act_probability": 0,
        "missing_confidence": 0,
        "malformed_case": 0,
    }

    for case in cases:
        if not isinstance(case, dict):
            skipped["malformed_case"] += 1
            continue
        correct = case.get("correct")
        if not isinstance(correct, bool):
            # A score answer, or a report without the field: correctness is not binary here.
            skipped["no_binary_correctness"] += 1
            continue
        act = _case_act_probability(case)
        if not _is_number(act):
            # Never coerced to 0.0: a missing act_probability is missing evidence, and
            # reading it as "not acting" would quietly invent a decision.
            skipped["missing_act_probability"] += 1
            continue
        confidence = case.get("confidence")
        if not _is_number(confidence):
            # Dropped from the paired set rather than back-filled, so the act AUC and the
            # confidence control stay on exactly the same decisions.
            skipped["missing_confidence"] += 1
            continue
        paired.append({"act": float(act), "confidence": float(confidence), "correct": correct})

    labels = [p["correct"] for p in paired]
    acts = [p["act"] for p in paired]
    confs = [p["confidence"] for p in paired]

    act_auc = roc_auc(acts, labels) if paired else None
    confidence_auc = roc_auc(confs, labels) if paired else None
    if not paired:
        reason = _NO_CASES
    elif act_auc is None:
        reason = _UNSCORABLE
    else:
        reason = None

    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "total_cases": len(cases),
        "paired": len(paired),
        "paired_cases": len(paired),
        "correct": sum(1 for p in paired if p["correct"]),
        "incorrect": sum(1 for p in paired if not p["correct"]),
        "act_auc": act_auc,
        "confidence_auc": confidence_auc,
        "auc_null_reason": reason,
        "act_min": min(acts) if acts else None,
        "act_max": max(acts) if acts else None,
        "act_unique": len(set(acts)) if acts else 0,
    }
    summary["skipped"] = skipped
    summary["saturated"] = bool(
        acts and (len(set(acts)) <= 1 or (max(acts) - min(acts)) < _SATURATION_EPS)
    )
    return summary


def report_from_document(document: Any) -> Dict[str, Any]:
    """Analyse an `EvalReport.to_json()` document.

    Raises ValueError when the document is not a report, so a wrong file is a clear error
    rather than a set of zeros that reads like a result.
    """
    if not isinstance(document, dict):
        raise ValueError("report must be a JSON object")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("report has no 'cases' list: is this a laya-evals --json report?")
    analysis = analyse_cases(cases)
    return {
        "schema": SCHEMA,
        "config": document.get("config") or {},
        "source_overall": document.get("overall") or {},
        "analysis": analysis,
    }


def format_markdown(result: Dict[str, Any]) -> str:
    """A short human summary of an analysed report."""
    a = result["analysis"]
    lines = [
        "# act-head diagnostic",
        "",
        "Does `act_probability` rank correct decisions above incorrect ones?",
        "",
        "| measure | value |",
        "|---|---|",
        "| cases in report | %d |" % a["total_cases"],
        "| scorable (paired) decisions | %d |" % a["paired"],
        "| correct / incorrect | %d / %d |" % (a["correct"], a["incorrect"]),
        "| ROC-AUC of `act_probability` | %s |" % _fmt(a["act_auc"]),
        "| ROC-AUC of calibrated `confidence` (same cases) | %s |" % _fmt(a["confidence_auc"]),
        "| `act_probability` min / max / unique | %s / %s / %d |"
        % (_fmt(a["act_min"]), _fmt(a["act_max"]), a["act_unique"]),
        "| saturated signal | %s |" % ("yes" if a["saturated"] else "no"),
    ]
    if a["auc_null_reason"]:
        lines.append("")
        lines.append("No AUC reported: %s." % a["auc_null_reason"])
    skipped = a["skipped"]
    if any(skipped.values()):
        lines.append("")
        lines.append("Skipped cases, excluded from every count above:")
        for key in ("no_binary_correctness", "missing_act_probability",
                    "missing_confidence", "malformed_case"):
            if skipped[key]:
                lines.append("* `%s`: %d" % (key, skipped[key]))
    lines.extend([
        "",
        "This is a diagnostic only. It does not change `act_probability`, and a score",
        "here is not a claim that the current act head is informative.",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.4f" % value


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score act_probability against labelled decisions in a laya-evals report."
    )
    parser.add_argument("report", help="the JSON written by `laya-evals run --json`")
    parser.add_argument("--json", dest="json_out", help="write the full result here")
    parser.add_argument("--markdown", dest="md_out", help="write a human summary here")
    args = parser.parse_args(argv)

    with open(args.report, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    result = report_from_document(document)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            # allow_nan=False: this diagnostic refuses non-finite evidence on the way in,
            # so a NaN reaching the output would be a bug worth failing on, not a value
            # to write out as non-standard JSON.
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as handle:
            handle.write(format_markdown(result))
    if not args.json_out and not args.md_out:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
