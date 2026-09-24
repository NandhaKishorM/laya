"""Presentation-sensitivity regression checks for a Laya checkpoint.

Why this exists
---------------
``laya-multilingual``'s ``score`` head has a learned prior against the first-listed
level (#131). The fix is a position-balanced retrain, and this is the check that says
whether a retrained checkpoint removed the prior. It needs no labelled data: every
input is fixed in this file, and both checks compare the checkpoint against itself
under different presentations of the same question.

Checks
------
``score_slot0_identical``
    The identical-option control from @AlKor13 in #131. A ``score`` question whose K
    levels all carry the same text, so the rendered options differ only by position
    and the ``level N:`` prefix ``render_options`` always emits. The metric is the raw
    marker logit of slot 0 minus the mean over the K slots, averaged over every state
    and every (text, K) configuration. A checkpoint with no slot-0 deficit sits at or
    above 0; the gate is one-sided (see "Limits" in README.md).

``score_first_slot_permuted``
    Three real levels presented in all 3! = 6 orders for each state. Every level sits
    in every slot exactly twice per state, so a checkpoint whose answer does not
    depend on the order picks the first slot in exactly 1/3 of the decisions. The
    metric is the observed first-slot rate.

Both metrics read raw marker logits (before temperature) through
``laya_eval.score_cases``, the same forward pass ``research/scripts/bench_local.py``
uses. ``parity`` checks that path against ``Agent.system_one`` before anything is
reported.

Usage
-----
    python research/eval/presentation_checks.py --model convaiinnovations/laya
    python research/eval/presentation_checks.py --model convaiinnovations/laya \\
        --subfolder multilingual --out multilingual.json

Exit status: 0 every check passed, 1 a check failed, 2 the harness could not be
trusted (parity with ``Agent.system_one`` above ``PARITY_TOL``).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Same path handling as laya_eval.py: allow running this file directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.eval.laya_eval import score_cases, softmax_t, temperature_for  # noqa: E402

# Fixed English states written for this check. Urgency varies on purpose: a slot
# prior has to show through states whose honest answer is low, middle and high.
STATES = (
    "Hi, whenever you get a chance, could you update the billing address on our account "
    "to 12 Harbour Street? No rush at all.",
    "Our invoice for March shows two charges for the same seat. Could you look into it "
    "this week?",
    "The CSV export has been failing since this morning and our finance team cannot close "
    "the month until it works.",
    "Just wanted to say thanks for the onboarding call yesterday. Nothing needed from "
    "your side.",
    "We are thinking about adding ten more licences next quarter. Can someone send pricing "
    "when convenient?",
    "Login is down for every user in our company. Nobody can reach the dashboard and our "
    "customers are waiting.",
    "The mobile app sometimes shows the wrong time zone on reports. It is annoying but we "
    "can work around it.",
    "Our API key was revoked by mistake and every production request is returning 401 "
    "errors right now.",
    "Payroll runs tomorrow morning and the integration still rejects our employee file "
    "with a schema error.",
    "Could you add a dark mode to the admin screens at some point? Several of us would "
    "like it.",
)
INSTRUCTIONS = "How urgent is this request?"
IDENTICAL_TEXTS = ("moderate", "a request")
IDENTICAL_KS = (3, 4, 5)
LEVELS = ("Not urgent", "Soon", "Work is blocked")

# Gates. Set from the shipped checkpoints' leave-one-out ranges (README.md, "Thresholds").
SLOT0_MIN = -0.20
FIRST_SLOT_MIN = 0.15
PARITY_TOL = 1e-3

ScoreFn = Callable[[str, Sequence[Dict[str, Any]]], List[Any]]


def identical_question(text: str, k: int) -> Dict[str, Any]:
    return {"type": "score", "instructions": INSTRUCTIONS, "criteria": [text] * k}


def permuted_questions() -> List[Tuple[Tuple[int, ...], Dict[str, Any]]]:
    """Every order of LEVELS, as (order, question); order[slot] indexes LEVELS."""
    return [(order, {"type": "score", "instructions": INSTRUCTIONS,
                     "criteria": [LEVELS[i] for i in order]})
            for order in itertools.permutations(range(len(LEVELS)))]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def leave_one_out(per_state: Sequence[float]) -> Tuple[float, float]:
    """(min, max) of the metric recomputed with each state left out in turn."""
    n = len(per_state)
    total = sum(per_state)
    loo = [(total - x) / (n - 1) for x in per_state]
    return min(loo), max(loo)


def passes(value: float, threshold: float) -> bool:
    return value >= threshold


def check_score_slot0_identical(score_fn: ScoreFn, states: Sequence[str] = STATES) -> Dict[str, Any]:
    configs = [(text, k) for text in IDENTICAL_TEXTS for k in IDENTICAL_KS]
    questions = [identical_question(text, k) for text, k in configs]
    per_state: List[float] = []
    per_config: Dict[str, List[float]] = {"%s/K=%d" % c: [] for c in configs}
    per_slot: Dict[str, List[float]] = {"%s/K=%d" % (t, k): [0.0] * k for t, k in configs}
    for state in states:
        slot0 = []
        for (text, k), z in zip(configs, score_fn(state, questions)):
            name = "%s/K=%d" % (text, k)
            z = [float(v) for v in z]  # numpy float32 would survive into the JSON report
            centred = [v - mean(z) for v in z]
            slot0.append(centred[0])
            per_config[name].append(centred[0])
            per_slot[name] = [a + b / len(states) for a, b in zip(per_slot[name], centred)]
        per_state.append(mean(slot0))
    metric = mean(per_state)
    lo, hi = leave_one_out(per_state)
    return {
        "metric": round(metric, 4),
        "threshold": SLOT0_MIN,
        "passed": passes(metric, SLOT0_MIN),
        "leave_one_out": [round(lo, 4), round(hi, 4)],
        "per_state": [round(x, 4) for x in per_state],
        "per_config": {name: round(mean(v), 4) for name, v in per_config.items()},
        "per_slot_centred": {name: [round(x, 4) for x in v] for name, v in per_slot.items()},
    }


def check_score_first_slot_permuted(score_fn: ScoreFn, states: Sequence[str] = STATES) -> Dict[str, Any]:
    design = permuted_questions()
    questions = [q for _order, q in design]
    by_slot = [0] * len(LEVELS)
    by_label = {label: 0 for label in LEVELS}
    per_state: List[float] = []
    order_invariant = 0
    for state in states:
        picked = []
        for (order, _q), z in zip(design, score_fn(state, questions)):
            slot = max(range(len(z)), key=lambda i: float(z[i]))
            by_slot[slot] += 1
            by_label[LEVELS[order[slot]]] += 1
            picked.append((slot, LEVELS[order[slot]]))
        per_state.append(sum(1 for slot, _ in picked if slot == 0) / len(design))
        order_invariant += len({label for _, label in picked}) == 1
    metric = mean(per_state)
    lo, hi = leave_one_out(per_state)
    return {
        "metric": round(metric, 4),
        "threshold": FIRST_SLOT_MIN,
        "passed": passes(metric, FIRST_SLOT_MIN),
        "leave_one_out": [round(lo, 4), round(hi, 4)],
        "decisions": len(states) * len(design),
        "per_state": [round(x, 4) for x in per_state],
        "argmax_by_slot": by_slot,
        "picks_by_label": by_label,
        "order_invariant_states": order_invariant,
    }


CHECKS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "score_slot0_identical": check_score_slot0_identical,
    "score_first_slot_permuted": check_score_first_slot_permuted,
}


def run_checks(score_fn: ScoreFn, names: Optional[Sequence[str]] = None,
               states: Sequence[str] = STATES) -> Dict[str, Any]:
    names = list(names or CHECKS)
    results = {name: CHECKS[name](score_fn, states) for name in names}
    return {"checks": results, "passed": all(r["passed"] for r in results.values())}


def exit_code(report: Dict[str, Any], parity: float) -> int:
    if not parity <= PARITY_TOL:
        return 2
    return 0 if report["passed"] else 1


def agent_score_fn(agent) -> ScoreFn:
    """Raw marker logits for several questions on one state, one forward pass."""
    def score(state, questions):
        return score_cases(agent, [(state, {str(i): q for i, q in enumerate(questions)})])
    return score


def parity(agent, states: Sequence[str] = STATES) -> float:
    """Max |p| difference between this harness and Agent.system_one.

    system_one rounds probabilities to 4 decimals, so agreement shows up as <= 5e-5.
    """
    from laya.common import QTYPES

    questions = {"identical": identical_question(IDENTICAL_TEXTS[0], 3),
                 "levels": {"type": "score", "instructions": INSTRUCTIONS, "criteria": list(LEVELS)}}
    score = agent_score_fn(agent)
    worst = 0.0
    for state in states:
        public = agent.system_one(state, questions)["answers"]
        for (qid, _q), z in zip(questions.items(), score(state, list(questions.values()))):
            p = softmax_t(z, temperature_for(agent, QTYPES["score"], len(z)))
            got = [public[qid]["probabilities"][str(i)] for i in range(len(z))]
            worst = max(worst, max(abs(a - float(b)) for a, b in zip(got, p)))
    return worst


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="presentation-checks",
        description="Label-free presentation-sensitivity checks for a Laya checkpoint.")
    parser.add_argument("--model", default="convaiinnovations/laya",
                        help="checkpoint repo id or local path")
    parser.add_argument("--subfolder", default=None,
                        help="checkpoint subfolder, e.g. multilingual")
    parser.add_argument("--device", default="cpu",
                        help="default cpu: fp32 and deterministic, which is what the thresholds were set on")
    parser.add_argument("--checks", default=",".join(CHECKS),
                        help="comma-separated subset of: %s" % ", ".join(CHECKS))
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    names = [x.strip() for x in args.checks.split(",") if x.strip()]
    unknown = [x for x in names if x not in CHECKS]
    if unknown:
        print("unknown checks: %s" % ", ".join(unknown), file=sys.stderr)
        return 2

    import laya

    started = time.time()
    agent = laya.load(args.model, device=args.device, subfolder=args.subfolder)
    agent.model.eval()
    worst = parity(agent)
    report = run_checks(agent_score_fn(agent), names)
    payload = {
        "config": {
            "model": args.model,
            "subfolder": args.subfolder,
            "device": str(agent.device),
            "states": len(STATES),
            "instructions": INSTRUCTIONS,
            "identical_texts": list(IDENTICAL_TEXTS),
            "identical_ks": list(IDENTICAL_KS),
            "levels": list(LEVELS),
            "laya_version": getattr(laya, "__version__", "unknown"),
        },
        "parity_max_abs_diff": worst,
        **report,
        "seconds": round(time.time() - started, 1),
    }

    for name, r in report["checks"].items():
        print("  %-26s %8.4f  >= %5.2f  %s   (leave-one-out %.4f .. %.4f)"
              % (name, r["metric"], r["threshold"], "PASS" if r["passed"] else "FAIL",
                 r["leave_one_out"][0], r["leave_one_out"][1]))
    print("  parity vs Agent.system_one: max |dp| %.2e (tolerance %.0e)" % (worst, PARITY_TOL))
    code = exit_code(report, worst)
    print("  verdict: %s" % {0: "PASS", 1: "FAIL", 2: "HARNESS MISMATCH"}[code])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
    return code


if __name__ == "__main__":                      # pragma: no cover
    sys.exit(main())
