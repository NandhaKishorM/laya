"""Tests for OptionStabilityHook and metamorphic reliability signal (#635).

Pure Python tests: runs offline without torch or checkpoint downloads.

Run: python tests/test_option_stability.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.hooks import (  # noqa: E402
    ALL_VARIANTS,
    DEFAULT_ACCEPT,
    DEFAULT_ESCALATE,
    OptionStabilityHook,
    PredictContext,
    VARIANT_PRESETS,
    calibrate_reliability,
    decide_reliability,
    make_variants,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
        return
    except BaseException as other:  # noqa: BLE001
        FAIL.append("%s: raised %r, want %r" % (name, other, exc))
        return
    FAIL.append("%s: did not raise %r" % (name, exc))


# --------------------------------------------------------------- constants & presets
check("constants/ALL_VARIANTS count", len(ALL_VARIANTS), 7)
check("constants/DEFAULT_ACCEPT", DEFAULT_ACCEPT, 0.77)
check("constants/DEFAULT_ESCALATE", DEFAULT_ESCALATE, 0.61)
check("constants/VARIANT_PRESETS keys", sorted(VARIANT_PRESETS.keys()), ["fast", "full"])


# --------------------------------------------------------------- make_variants
v1 = make_variants(["a", "b", "c"], "seed")
v2 = make_variants(["a", "b", "c"], "seed")
check("make_variants/deterministic", v1, v2)
check("make_variants/original is first", v1[0], ("original", ["a", "b", "c"], False))
check("make_variants/no duplicates", len({(tuple(o), r) for _, o, r in v1}), len(v1))

# 2 options: original + reversed + letters + letters_reversed = at most 4 variants
check_true("make_variants/two options capped at 4", len(make_variants(["x", "y"], "s", n_shuffles=10)) <= 4)

# Presets
fast = make_variants(["a", "b", "c"], "s", include=VARIANT_PRESETS["fast"])
check("make_variants/fast preset has 3 rows", [name for name, _, _ in fast], ["original", "reversed", "letters_reversed"])

# Relabel flag disabled
no_rel = make_variants(["a", "b", "c"], "s", relabel=False)
check_true("make_variants/no relabel has no letters variants", all(not r for _, _, r in no_rel))


# --------------------------------------------------------------- validation
check_raises("validation/negative n_shuffles", ValueError, lambda: OptionStabilityHook(n_shuffles=-1))
check_raises("validation/bool n_shuffles", ValueError, lambda: OptionStabilityHook(n_shuffles=True))
check_raises("validation/bool relabel", TypeError, lambda: OptionStabilityHook(relabel="yes"))
check_raises("validation/unknown preset", ValueError, lambda: OptionStabilityHook(variants="invalid"))
check_raises("validation/empty variants list", ValueError, lambda: OptionStabilityHook(variants=[]))
check_raises("validation/unknown variant name", ValueError, lambda: OptionStabilityHook(variants=["reversed", "nope"]))
check_raises("validation/invalid accept float", ValueError, lambda: OptionStabilityHook(accept=1.5))
check_raises("validation/negative accept", ValueError, lambda: OptionStabilityHook(accept=-0.1))
check_raises("validation/invalid escalate float", ValueError, lambda: OptionStabilityHook(escalate=1.1))
check_raises("validation/escalate greater than accept", ValueError, lambda: OptionStabilityHook(accept=0.5, escalate=0.8))


# --------------------------------------------------------------- decide & calibrate
check("decide/accept", decide_reliability(0.95), "ACCEPT")
check("decide/verify", decide_reliability(0.70), "VERIFY")
check("decide/escalate", decide_reliability(0.30), "ESCALATE")

scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
correct = [False, False, False, True, False, True, False, True, True, True, True, True]
acc, esc = calibrate_reliability(scores, correct, target_accuracy=0.90)
check_true("calibrate/escalate <= accept", esc <= acc)
kept = [c for s, c in zip(scores, correct) if s >= acc]
check_true("calibrate/target accuracy reached", sum(kept) / len(kept) >= 0.90)


# --------------------------------------------------------------- hook lifecycle: single state
hook = OptionStabilityHook(seed="fixed-seed")
q_choice = {
    "ask": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"refund": "money back", "tech": "bug fix", "sales": "pricing"},
    },
    "score_q": {"type": "score", "instructions": "Rate?", "criteria": ["low", "high"]},
    "noul_q": {"type": "noul", "instructions": "Yes or no?"},
}

ctx = PredictContext(states=["I want my money back."], questions=dict(q_choice))
hook.on_predict_start(ctx)

# Check variants added
check_true("start/original questions preserved", "ask" in ctx.questions)
check_true("start/non-choice unchanged", "score_q" in ctx.questions and "noul_q" in ctx.questions)
variant_keys = [k for k in ctx.questions if k.startswith("ask::rel")]
check_true("start/variants added for choice", len(variant_keys) >= 2)

# Check reserved separator rejected in caller question ids
bad_ctx = PredictContext(states=["test"], questions={"bad::rel1": q_choice["ask"]})
check_raises("start/rejects reserved separator", ValueError, lambda: hook.on_predict_start(bad_ctx))

# Simulate model execution where "refund" wins across all variants (stable agent)
simulated_answers = {}
for qk, qv in ctx.questions.items():
    if qk == "score_q":
        simulated_answers[qk] = {"type": "score", "score": 1.0, "confidence": 0.9, "answer_confidence": 0.9}
    elif qk == "noul_q":
        simulated_answers[qk] = {"type": "noul", "noul": 0.8, "confidence": 0.8, "answer_confidence": 0.8}
    else:
        # Choice question (original or variant)
        crit = qv["criteria"]
        # Find which key holds description "money back"
        target_key = [k for k, desc in crit.items() if desc == "money back"][0]
        other_keys = [k for k in crit if k != target_key]
        probs = {target_key: 0.85}
        for ok in other_keys:
            probs[ok] = round(0.15 / len(other_keys), 4)
        simulated_answers[qk] = {
            "type": "choice",
            "choice": target_key,
            "probabilities": probs,
            "confidence": 0.7,
            "answer_confidence": 0.85,
        }

ctx.results = [{
    "model": "fake",
    "answers": simulated_answers,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 0,
        "truncated_questions": ["ask::rel1"],
        "options": {"ask": {"total": 3, "distinct": 3}, "ask::rel1": {"total": 3, "distinct": 3}},
    },
}]

hook.on_predict_end(ctx)

# Verify results after on_predict_end
res = ctx.results[0]
ans = res["answers"]
check_true("end/variant keys removed from answers", not any(k.startswith("ask::rel") for k in ans))
check("end/original choice question retained", ans["ask"]["choice"], "refund")
check_true("end/reliability added to choice", "reliability" in ans["ask"])
rel = ans["ask"]["reliability"]
check("end/stable answer accepted", rel["decision"], "ACCEPT")
check("end/stability is 1.0", rel["stability"], 1.0)
check_true("end/soft_stability is ~0.85", abs(rel["soft_stability"] - 0.85) < 1e-3)
check("end/distinct_choices is single refund", rel["distinct_choices"], ["refund"])
check_true("end/non-choice has no reliability", "reliability" not in ans["score_q"])
check_true("end/usage truncated_questions cleaned", "ask::rel1" not in res["usage"]["truncated_questions"])
check_true("end/usage options cleaned", "ask::rel1" not in res["usage"]["options"])
check_true("end/internal plan cleaned", ctx.run_id not in hook._plans)


# --------------------------------------------------------------- fragile agent (picks first option)
ctx_fragile = PredictContext(states=["test"], questions={"ask": dict(q_choice["ask"])})
hook_fragile = OptionStabilityHook(seed="fixed-seed")
hook_fragile.on_predict_start(ctx_fragile)

sim_fragile = {}
for qk, qv in ctx_fragile.questions.items():
    crit = qv["criteria"]
    keys = list(crit.keys())
    probs = {keys[0]: 0.9}
    for ok in keys[1:]:
        probs[ok] = round(0.1 / (len(keys) - 1), 4)
    sim_fragile[qk] = {
        "type": "choice",
        "choice": keys[0],
        "probabilities": probs,
        "confidence": 0.8,
        "answer_confidence": 0.9,
    }

ctx_fragile.results = [{"model": "fake", "answers": sim_fragile, "usage": {}}]
hook_fragile.on_predict_end(ctx_fragile)

rel_f = ctx_fragile.results[0]["answers"]["ask"]["reliability"]
check_true("fragile/stability is low", rel_f["stability"] < 0.5)
check_true("fragile/decision is not ACCEPT", rel_f["decision"] in ("VERIFY", "ESCALATE"))
check_true("fragile/multiple distinct choices", len(rel_f["distinct_choices"]) > 1)


# --------------------------------------------------------------- batch execution
ctx_batch = PredictContext(states=["s1", "s2"], questions={"ask": dict(q_choice["ask"])})
hook_batch = OptionStabilityHook(seed="batch-test")
hook_batch.on_predict_start(ctx_batch)

# Simulate 2 states in results
results_batch = []
for state in ("s1", "s2"):
    sim_b = {}
    for qk, qv in ctx_batch.questions.items():
        crit = qv["criteria"]
        target = [k for k, desc in crit.items() if desc == "money back"][0]
        probs = {k: (0.9 if k == target else 0.05) for k in crit}
        sim_b[qk] = {"type": "choice", "choice": target, "probabilities": probs, "answer_confidence": 0.9}
    results_batch.append({"model": "fake", "answers": sim_b, "usage": {}})

ctx_batch.results = results_batch
hook_batch.on_predict_end(ctx_batch)

check("batch/len results is 2", len(ctx_batch.results), 2)
for i, r in enumerate(ctx_batch.results):
    a = r["answers"]["ask"]
    check("batch/%d variant keys cleaned" % i, [k for k in r["answers"] if "::rel" in k], [])
    check("batch/%d decision" % i, a["reliability"]["decision"], "ACCEPT")


# --------------------------------------------------------------- error handling cleanup
ctx_err = PredictContext(states=["err"], questions={"ask": dict(q_choice["ask"])})
hook_err = OptionStabilityHook()
hook_err.on_predict_start(ctx_err)
check_true("error/plan exists before error", ctx_err.run_id in hook_err._plans)
hook_err.on_error(ctx_err)
check_true("error/plan cleared on error", ctx_err.run_id not in hook_err._plans)


# --------------------------------------------------------------- uncalibrated / accept=None
check("decide/uncalibrated returns None", decide_reliability(0.99, accept=None), None)
hook_uncal = OptionStabilityHook(accept=None, escalate=None, seed="fixed-seed")
check("hook/uncalibrated accept is None", hook_uncal.accept, None)
check("hook/uncalibrated escalate is None", hook_uncal.escalate, None)
ctx_uncal = PredictContext(states=["I want my money back."], questions={"ask": dict(q_choice["ask"])})
hook_uncal.on_predict_start(ctx_uncal)
sim_uncal = {}
for qk, qv in ctx_uncal.questions.items():
    crit = qv["criteria"]
    target_key = [k for k, desc in crit.items() if desc == "money back"][0]
    sim_uncal[qk] = {
        "type": "choice",
        "choice": target_key,
        "probabilities": {target_key: 0.85},
        "confidence": 0.85,
        "answer_confidence": 0.85,
    }
ctx_uncal.results = [{"answers": sim_uncal, "usage": {}}]
hook_uncal.on_predict_end(ctx_uncal)
check("hook/uncalibrated decision is None", ctx_uncal.results[0]["answers"]["ask"]["reliability"]["decision"], None)


# --------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all option stability tests passed")
sys.exit(1 if FAIL else 0)
