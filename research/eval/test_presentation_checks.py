"""Offline tests for the presentation checks.

No checkpoint and no network: the checks take a scoring function, and every test
here passes a scripted one. The model-dependent paths (`agent_score_fn`, `parity`)
run only from the CLI against a real checkpoint.

Run: python research/eval/test_presentation_checks.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from laya.common import render_options  # noqa: E402
from research.eval.presentation_checks import (  # noqa: E402
    CHECKS, FIRST_SLOT_MIN, IDENTICAL_KS, IDENTICAL_TEXTS, INSTRUCTIONS, LEVELS, PARITY_TOL,
    SLOT0_MIN, STATES, check_score_first_slot_permuted, check_score_slot0_identical, exit_code,
    identical_question, leave_one_out, passes, permuted_questions, run_checks,
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
        FAIL.append("%s %s" % (name, detail))


def scripted(position_bias, label_score=None):
    """A scoring function: logit = position_bias[slot] + label_score[option text]."""
    label_score = label_score or {}

    def score(_state, questions):
        return [[position_bias[i] + label_score.get(text, 0.0) for i, text in enumerate(q["criteria"])]
                for q in questions]
    return score


FLAT = scripted([0.0] * 5)
NO_SLOT0 = scripted([-2.0, 0.5, 0.5, 0.5, 0.5], {"Not urgent": 0.0, "Soon": 0.3, "Work is blocked": 0.6})
EARLY = scripted([1.5, 0.8, 0.0, -0.8, -1.5])                     # english-like: prefers early slots
BY_LABEL = scripted([0.0] * 5, {"Not urgent": 0.0, "Soon": 1.0, "Work is blocked": 2.0})


# ------------------------------------------------------------------- the inputs
check("states/ten fixed states", len(STATES), 10)
check("states/no duplicates", len(set(STATES)), len(STATES))
check_true("states/plain ASCII", all(s.isascii() for s in STATES))
check("const/instructions", INSTRUCTIONS, "How urgent is this request?")
check("const/identical texts", IDENTICAL_TEXTS, ("moderate", "a request"))
check("const/identical K", IDENTICAL_KS, (3, 4, 5))
check("const/levels", LEVELS, ("Not urgent", "Soon", "Work is blocked"))
check("const/slot-0 gate", SLOT0_MIN, -0.20)
check("const/first-slot gate", FIRST_SLOT_MIN, 0.15)
check("const/parity tolerance", PARITY_TOL, 1e-3)
check("const/registered checks", sorted(CHECKS), ["score_first_slot_permuted", "score_slot0_identical"])


# ------------------------------------------- identical options differ by position only
for text in IDENTICAL_TEXTS:
    for k in IDENTICAL_KS:
        q = identical_question(text, k)
        rendered = render_options({"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]})
        check("identical/%s K=%d renders level-indexed" % (text, k),
              rendered, ["level %d: %s" % (i, text) for i in range(k)])
        check("identical/%s K=%d one text after the prefix" % (text, k),
              {r.split(": ", 1)[1] for r in rendered}, {text})


# ------------------------------------------------------- the all-permutation design
design = permuted_questions()
check("perm/3! orders", len(design), 6)
check("perm/orders are distinct", len({order for order, _ in design}), 6)
for slot in range(len(LEVELS)):
    for li, label in enumerate(LEVELS):
        check("perm/%s in slot %d twice" % (label, slot),
              sum(1 for order, _ in design if order[slot] == li), 2)
check_true("perm/criteria follow the order",
           all(q["criteria"] == [LEVELS[i] for i in order] for order, q in design))


# ------------------------------------------------------------ slot-0, identical options
r = check_score_slot0_identical(FLAT)
check("slot0/flat logits sit at zero", r["metric"], 0.0)
check("slot0/flat passes", r["passed"], True)

r = check_score_slot0_identical(NO_SLOT0)
check_true("slot0/suppressed slot 0 is negative", r["metric"] < SLOT0_MIN, str(r["metric"]))
check("slot0/suppressed slot 0 fails", r["passed"], False)
# centred slot 0 = -2.0 - mean(bias[:k]); K=3: -5/3, K=4: -1.875, K=5: -2.0, same for both texts
check("slot0/suppressed metric by hand", r["metric"], round((-5 / 3 - 1.875 - 2.0) / 3, 4))
check("slot0/per-slot means are centred",
      all(abs(sum(v)) < 1e-3 for v in r["per_slot_centred"].values()), True)

r = check_score_slot0_identical(EARLY)
check_true("slot0/early-slot preference is positive", r["metric"] > 0, str(r["metric"]))
check("slot0/one-sided: early-slot preference passes", r["passed"], True)

check("slot0/one row per state", len(r["per_state"]), len(STATES))
check("slot0/one entry per (text, K)", len(r["per_config"]), len(IDENTICAL_TEXTS) * len(IDENTICAL_KS))
check("slot0/metric is the mean of per_state",
      round(sum(r["per_state"]) / len(r["per_state"]), 4), r["metric"])


# ------------------------------------------------------------ first slot, permuted
r = check_score_first_slot_permuted(BY_LABEL)
check("first/order-invariant model picks slot 0 in exactly 1/3", r["metric"], round(1 / 3, 4))
check("first/order-invariant passes", r["passed"], True)
check("first/every state order-invariant", r["order_invariant_states"], len(STATES))
check("first/picks follow the label", r["picks_by_label"],
      {"Not urgent": 0, "Soon": 0, "Work is blocked": 6 * len(STATES)})
check("first/decisions", r["decisions"], 6 * len(STATES))

r = check_score_first_slot_permuted(NO_SLOT0)
check("first/slot-0 hole gives zero", r["metric"], 0.0)
check("first/slot-0 hole fails", r["passed"], False)
check("first/argmax never lands in slot 0", r["argmax_by_slot"][0], 0)
check_true("first/order dependence is visible", r["order_invariant_states"] == 0,
           str(r["order_invariant_states"]))

r = check_score_first_slot_permuted(EARLY)
check("first/one-sided: always slot 0 passes", (r["metric"], r["passed"]), (1.0, True))


# ------------------------------------------------------------ leave-one-out and gates
check("loo/by hand", leave_one_out([1.0, 2.0, 3.0, 4.0]), (2.0, 3.0))
check("loo/constant", leave_one_out([0.5, 0.5, 0.5]), (0.5, 0.5))
r = check_score_slot0_identical(NO_SLOT0)
check_true("loo/brackets the metric",
           r["leave_one_out"][0] <= r["metric"] <= r["leave_one_out"][1], str(r))
check("gate/at the threshold passes", passes(SLOT0_MIN, SLOT0_MIN), True)
check("gate/just below fails", passes(SLOT0_MIN - 1e-6, SLOT0_MIN), False)


# ------------------------------------------------------------------- run and exit
calls = []


def counting(state, questions):
    calls.append(len(questions))
    return FLAT(state, questions)


run_checks(counting)
check("run/one forward per state per check", len(calls), 2 * len(STATES))
check("run/passes only when every check does", run_checks(BY_LABEL)["passed"], True)
check("run/one failing check fails the run", run_checks(NO_SLOT0)["passed"], False)
check("run/subset by name", sorted(run_checks(FLAT, ["score_slot0_identical"])["checks"]),
      ["score_slot0_identical"])
check("exit/pass", exit_code({"passed": True}, 4.9e-5), 0)
check("exit/fail", exit_code({"passed": False}, 4.9e-5), 1)
check("exit/parity outranks the verdict", exit_code({"passed": True}, 2e-3), 2)
check("exit/nan parity is a mismatch", exit_code({"passed": True}, float("nan")), 2)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
