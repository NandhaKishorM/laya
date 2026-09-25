"""Offline tests for the act-head diagnostic in `research/evals/act_head_eval.py`.

No checkpoint and no network: every function under test is pure arithmetic over a
report that `laya-evals --json` already wrote, so this runs offline.

The fixtures here are tiny and synthetic. They exist to pin the metric mechanics
only -- they are **not** a measurement of the released checkpoints and say nothing
about act-head quality. The real labelled sets are not committed to this repository.

Run: python research/evals/test_act_head_eval.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.evals.act_head_eval import (  # noqa: E402
    analyse_cases,
    format_markdown,
    roc_auc,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, (" -- " + detail) if detail else ""))


def check_close(name, got, want, tol=1e-9):
    if got is None:
        FAIL.append("%s:\n     got  None\n     want ~%r" % (name, want))
    elif abs(float(got) - want) > tol:
        FAIL.append("%s:\n     got  %r\n     want ~%r" % (name, got, want))
    else:
        PASS.append(name)


def case(act, correct, confidence=None, qid="q", answer_type="choice", drop_action=False):
    """One `laya-evals` report case, shaped like `EvalReport.to_json()["cases"][i]`."""
    answer = {"type": answer_type}
    if correct is not None and answer_type == "choice":
        answer["choice"] = "a"
    if not drop_action:
        answer["action"] = {} if act is None else {"act_probability": act}
    else:
        answer["action"] = {}
    case_obj = {"qid": qid, "answer": answer, "correct": correct}
    if confidence is not None:
        case_obj["confidence"] = confidence
    return case_obj


# ------------------------------------------------------------------ roc_auc math
# Perfect separation: every correct case outranks every incorrect one.
check("auc/perfect", roc_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1.0)
# Exactly reversed: the score runs against correctness, which is the failure #185 reports.
check("auc/reversed", roc_auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]), 0.0)
# All scores tied, both classes present: every pairing is a tie, so exactly 0.5.
check("auc/all_tied", roc_auc([1.0, 1.0, 1.0, 1.0], [True, True, False, False]), 0.5)
# Partial ties get mid-ranks, so a tied positive does not beat a tied negative outright.
check_close("auc/partial_tie", roc_auc([0.5, 0.5, 0.9], [True, False, True]), 0.75)
check_close("auc/single_positive", roc_auc([0.3, 0.7], [False, True]), 1.0)
# One class absent: ROC-AUC is undefined, and must not be faked as 0.5 or 0.0.
check("auc/all_positive", roc_auc([0.1, 0.9], [True, True]), None)
check("auc/all_negative", roc_auc([0.1, 0.9], [False, False]), None)
check("auc/empty", roc_auc([], []), None)
# A single tied pair is the smallest well-defined case.
check("auc/one_each_tied", roc_auc([0.7, 0.7], [True, False]), 0.5)

# ------------------------------------------------------------------ case bucketing
# A perfect ranking through the full report path, with the confidence control.
perfect = analyse_cases([
    case(0.99, True, 0.9), case(0.80, True, 0.8),
    case(0.20, False, 0.3), case(0.01, False, 0.1),
])
check("analyse/paired_all", perfect["paired"], 4)
check("analyse/correct", perfect["correct"], 2)
check("analyse/incorrect", perfect["incorrect"], 2)
check("analyse/act_auc", perfect["act_auc"], 1.0)
check("analyse/confidence_auc", perfect["confidence_auc"], 1.0)
check_true("analyse/not_saturated", perfect["saturated"] is False, repr(perfect))
check("analyse/unique", perfect["act_unique"], 4)

# Reversed act_probability, but a confidence that ranks correctly: this is the shape
# of the measurement in #185, where the raw signal runs against correctness.
reversed_act = analyse_cases([
    case(0.10, True, 0.9), case(0.20, True, 0.7),
    case(0.80, False, 0.3), case(0.90, False, 0.1),
])
check("analyse/reversed_act_auc", reversed_act["act_auc"], 0.0)
check("analyse/reversed_confidence_auc", reversed_act["confidence_auc"], 1.0)
check("analyse/reversed_same_case_set", reversed_act["paired"], 4)

# The documented failure mode: act_probability saturated at 1.0 on every input.
saturated = analyse_cases([
    case(1.0, True, 0.9), case(1.0, False, 0.4), case(1.0, True, 0.8), case(1.0, False, 0.2),
])
check("analyse/saturated_detected", saturated["saturated"], True)
check("analyse/saturated_unique", saturated["act_unique"], 1)
check("analyse/saturated_auc", saturated["act_auc"], 0.5)
check("analyse/saturated_min_max", (saturated["act_min"], saturated["act_max"]), (1.0, 1.0))
check_true("analyse/saturated_min_max_json_safe",
            json.dumps(saturated, sort_keys=True) is not None)

# `correct=null` (a score answer) is counted and skipped, never folded into the denominator.
with_nulls = analyse_cases([
    case(0.9, True, 0.8),
    case(0.1, False, 0.2),
    case(0.5, None, 0.6, answer_type="score"),
    case(0.4, None, 0.3, answer_type="score"),
])
check("analyse/null_correctness_skipped", with_nulls["paired"], 2)
check("analyse/null_correctness_counted", with_nulls["skipped"]["no_binary_correctness"], 2)
check("analyse/null_correctness_auc", with_nulls["act_auc"], 1.0)

# Missing act_probability is explicit, never silently read as 0.0.
missing = analyse_cases([
    case(0.9, True, 0.8),
    case(None, False, 0.3, drop_action=True),
    case(0.1, False, 0.2),
])
check("analyse/missing_act_counted", missing["skipped"]["missing_act_probability"], 1)
check("analyse/missing_act_paired", missing["paired"], 2)
check("analyse/missing_act_auc", missing["act_auc"], 1.0)
check_true("analyse/missing_act_min_ignores_missing",
           missing["act_min"] == 0.1 and missing["act_max"] == 0.9, repr(missing))

# A non-numeric act_probability is malformed, not zero.
malformed = analyse_cases([
    case("not-a-number", True, 0.8),
    case(0.2, False, 0.3),
])
check("analyse/malformed_act_counted", malformed["skipped"]["missing_act_probability"], 1)
check("analyse/malformed_act_paired", malformed["paired"], 1)

# A bool in the act_probability slot is malformed, not the integer 1.
boolish = analyse_cases([case(True, True, 0.8), case(0.2, False, 0.3)])
check("analyse/bool_act_counted", boolish["skipped"]["missing_act_probability"], 1)

# Missing confidence drops the case from BOTH AUCs, so the control stays on one set.
no_conf = analyse_cases([
    case(0.9, True, 0.8),
    case(0.1, False),
    case(0.2, True, 0.1),
])
check("analyse/missing_confidence_counted", no_conf["skipped"]["missing_confidence"], 1)
check("analyse/missing_confidence_paired", no_conf["paired"], 2)
check("analyse/missing_confidence_same_set", no_conf["paired_cases"], 2)

# All-correct or all-incorrect cannot produce a fake AUC.
all_correct = analyse_cases([case(0.9, True, 0.8), case(0.4, True, 0.6)])
check("analyse/all_correct_auc", all_correct["act_auc"], None)
check("analyse/all_correct_confidence_auc", all_correct["confidence_auc"], None)
check("analyse/all_correct_null_reason", all_correct["auc_null_reason"],
      "one class is absent: ROC-AUC is undefined")
all_wrong = analyse_cases([case(0.9, False, 0.8), case(0.4, False, 0.6)])
check("analyse/all_wrong_auc", all_wrong["act_auc"], None)

# An empty report is a clean null state, not a crash.
empty = analyse_cases([])
check("analyse/empty_paired", empty["paired"], 0)
check("analyse/empty_auc", empty["act_auc"], None)
check("analyse/empty_reason", empty["auc_null_reason"], "no scorable cases")

# Non-dict / malformed case entries are counted, not fatal.
junk = analyse_cases([None, "nope", 5, case(0.9, True, 0.8)])
check("analyse/junk_counted", junk["skipped"]["malformed_case"], 3)
check("analyse/junk_paired", junk["paired"], 1)

# ------------------------------------------------------------------ non-finite evidence
# `json.load` accepts the non-standard NaN / Infinity / -Infinity tokens by default, so a
# malformed report can carry them into the harness. They are not measurements: ranking
# them produced a confident-looking 0.5 for NaN and +Inf and 0.0 for -Inf, where 0.0 reads
# as "the signal is perfectly anti-correlated" rather than "the input was garbage".
NAN, INF, NEG_INF = float("nan"), float("inf"), float("-inf")

for label, value in (("nan", NAN), ("posinf", INF), ("neginf", NEG_INF)):
    nonfinite_act = analyse_cases([
        case(0.9, True, 0.8),
        case(value, False, 0.3),
        case(0.1, False, 0.2),
    ])
    check("analyse/nonfinite_act_counted_%s" % label,
          nonfinite_act["skipped"]["missing_act_probability"], 1)
    check("analyse/nonfinite_act_paired_%s" % label, nonfinite_act["paired"], 2)
    check("analyse/nonfinite_act_no_fake_auc_%s" % label, nonfinite_act["act_auc"], 1.0)
    check("analyse/nonfinite_act_excluded_from_spread_%s" % label,
          (nonfinite_act["act_min"], nonfinite_act["act_max"]), (0.1, 0.9))

    nonfinite_conf = analyse_cases([
        case(0.9, True, 0.8),
        case(0.2, False, value),
        case(0.1, False, 0.2),
    ])
    check("analyse/nonfinite_confidence_counted_%s" % label,
          nonfinite_conf["skipped"]["missing_confidence"], 1)
    check("analyse/nonfinite_confidence_paired_%s" % label, nonfinite_conf["paired"], 2)
    # The dropped case is the middle one; the two survivors still separate cleanly, and
    # the act AUC must be computed on exactly that reduced set.
    check("analyse/nonfinite_confidence_auc_%s" % label, nonfinite_conf["act_auc"], 1.0)

# `roc_auc` is independently callable, so it rejects non-finite scores itself rather than
# relying on the caller having filtered them.
for label, value in (("nan", NAN), ("posinf", INF), ("neginf", NEG_INF)):
    try:
        roc_auc([value, 0.5, 0.9], [True, False, True])
        FAIL.append("roc_auc should reject a %s score" % label)
    except ValueError as exc:
        check_true("roc_auc/rejects_%s" % label, "finite" in str(exc).lower(), str(exc))

# A report that is entirely non-finite leaves nothing to score, which is a clean null
# state rather than a fabricated number.
all_bad = analyse_cases([case(NAN, True, 0.8), case(0.5, False, INF)])
check("analyse/all_nonfinite_paired", all_bad["paired"], 0)
check("analyse/all_nonfinite_auc", all_bad["act_auc"], None)
check("analyse/all_nonfinite_reason", all_bad["auc_null_reason"], "no scorable cases")
check("analyse/all_nonfinite_counted",
      all_bad["skipped"]["missing_confidence"] + all_bad["skipped"]["missing_act_probability"], 2)

# ------------------------------------------------------------------ report plumbing
REPORT = {
    "config": {"model": "english"},
    "overall": {"accuracy": 0.5},
    "cases": [
        case(0.9, True, 0.8, qid="a"),
        case(0.1, False, 0.2, qid="b"),
        case(0.5, None, 0.6, qid="c", answer_type="score"),
    ],
}


def _load(payload):
    return json.loads(json.dumps(payload))


from research.evals.act_head_eval import report_from_document  # noqa: E402

loaded = report_from_document(_load(REPORT))
check("report/cases_read", loaded["analysis"]["total_cases"], 3)
check("report/config_kept", loaded["config"], {"model": "english"})
check("report/analyse_paired", loaded["analysis"]["paired"], 2)
check("report/schema", loaded["schema"], "laya-evals-report/act-head-eval/1")

# The same document twice gives byte-identical JSON, so a committed result is a real diff.
first = json.dumps(report_from_document(_load(REPORT)), sort_keys=True)
second = json.dumps(report_from_document(_load(REPORT)), sort_keys=True)
check("report/deterministic", first, second)

# A document without `cases` is rejected clearly instead of silently reporting zeros.
try:
    report_from_document({"overall": {}})
    FAIL.append("report/missing_cases should raise")
except ValueError as exc:
    check_true("report/missing_cases_raises", "cases" in str(exc), str(exc))

markdown = format_markdown(loaded)
check_true("report/markdown_has_question", "act_probability" in markdown, markdown[:200])
check_true("report/markdown_flags_saturation", "saturat" in markdown.lower(), markdown[:400])

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
if not FAIL:
    print("all act-head eval tests passed")
sys.exit(1 if FAIL else 0)
