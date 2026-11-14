"""Tests for OptionStabilityHook and metamorphic reliability signal (#635).

Pure Python tests: runs offline without torch or checkpoint downloads.

Run: python tests/test_option_stability.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.hooks import (  # noqa: E402
    ALL_VARIANTS,
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
check_raises("validation/missing required accept and escalate", TypeError, lambda: OptionStabilityHook())
check_raises("validation/negative n_shuffles", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, n_shuffles=-1))
check_raises("validation/bool n_shuffles", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, n_shuffles=True))
check_raises("validation/bool relabel", TypeError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, relabel="yes"))
check_raises("validation/unknown preset", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, variants="invalid"))
check_raises("validation/empty variants list", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, variants=[]))
check_raises("validation/unknown variant name", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, variants=["reversed", "nope"]))
check_raises("validation/invalid accept float", ValueError, lambda: OptionStabilityHook(accept=1.5, escalate=0.5))
check_raises("validation/negative accept", ValueError, lambda: OptionStabilityHook(accept=-0.1, escalate=0.0))
check_raises("validation/invalid escalate float", ValueError, lambda: OptionStabilityHook(accept=0.9, escalate=1.1))
check_raises("validation/escalate greater than accept", ValueError, lambda: OptionStabilityHook(accept=0.5, escalate=0.8))


# --------------------------------------------------------------- decide & calibrate
check("decide/accept", decide_reliability(0.95, accept=0.77, escalate=0.61), "ACCEPT")
check("decide/verify", decide_reliability(0.70, accept=0.77, escalate=0.61), "VERIFY")
check("decide/escalate", decide_reliability(0.30, accept=0.77, escalate=0.61), "ESCALATE")
check_raises("decide/missing accept", TypeError, lambda: decide_reliability(0.95))
check_raises("calibrate/empty pairs", ValueError, lambda: calibrate_reliability([], []))

scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
correct = [False, False, False, True, False, True, False, True, True, True, True, True]
acc, esc = calibrate_reliability(scores, correct, target_accuracy=0.90)
check_true("calibrate/escalate <= accept", esc <= acc)
kept = [c for s, c in zip(scores, correct) if s >= acc]
check_true("calibrate/target accuracy reached", sum(kept) / len(kept) >= 0.90)


# Invalid evidence must not fit a usable gate by silently losing rows (#1312).
complete_cuts = calibrate_reliability([0.9, 0.9], [True, False])
check("calibrate/complete tied evidence accepts nothing", complete_cuts, (1.01, 0.0))
check("calibrate/complete tied evidence replays as VERIFY",
      decide_reliability(0.9, *complete_cuts), "VERIFY")
for name, values, labels in (
    ("short labels", [0.9, 0.9], [True]),
    ("short scores", [0.9], [True, False]),
    ("empty labels", [0.9], []),
    ("empty scores", [], [True]),
    ("NaN first", [float("nan"), 0.9], [False, True]),
    ("NaN last", [0.9, float("nan")], [True, False]),
):
    check_raises("calibrate/rejects " + name, ValueError,
                 lambda: calibrate_reliability(values, labels))

invalid_probabilities = (
    -0.1, 1.01, float("nan"), float("inf"), -float("inf"),
    True, False, "0.9", None, 0.9 + 0j, 10 ** 1000,
)
for value in invalid_probabilities:
    check_raises("calibrate/rejects score %r" % (value,), ValueError,
                 lambda: calibrate_reliability([0.9, value], [True, False]))
    for parameter in ("target_accuracy", "max_escalate_accuracy"):
        check_raises("calibrate/rejects %s=%r" % (parameter, value), ValueError,
                     lambda: calibrate_reliability([0.9], [False], **{parameter: value}))

for label in (0, 1, 2, 0.0, 1.0, "false", None):
    check_raises("calibrate/rejects non-boolean label %r" % (label,), TypeError,
                 lambda: calibrate_reliability([0.9, 0.9], [True, label]))

# ... but a NumPy boolean IS a boolean, and `correct` is normally computed rather than written
# out: `preds == gold` on a NumPy array or a pandas column yields `np.bool_`, which fails a bare
# `isinstance(x, bool)`. Same for `np.float32`/`np.int64` scores, which `numbers.Real` accepts
# while still excluding `np.bool_`, so the score and label checks stay disjoint. The fitted
# cutoffs come back as plain floats either way, because a caller stores and replays them.
import numpy as _np  # noqa: E402

_want = calibrate_reliability([0.9, 0.4, 0.8], [True, False, True], target_accuracy=1.0)
check("calibrate/plain floats and bools", _want, (0.8, 0.0))
for _name, _scores, _labels in (
    ("numpy bool array labels", [0.9, 0.4, 0.8], _np.array([True, False, True])),
    ("numpy bool scalars", [0.9, 0.4, 0.8], list(_np.array([1, 0, 1]) == 1)),
    ("numpy float32 scores", list(_np.array([0.9, 0.4, 0.8], dtype=_np.float32)), [True, False, True]),
    ("numpy float64 scores", list(_np.array([0.9, 0.4, 0.8])), [True, False, True]),
    ("numpy int scores", list(_np.array([1, 0, 1], dtype=_np.int64)), [True, False, True]),
):
    _got = calibrate_reliability(_scores, _labels, target_accuracy=1.0)
    check_true("calibrate/%s fits" % _name, _got[0] <= 1.0 and _got[1] == 0.0, repr(_got))
    check("calibrate/%s returns plain floats" % _name,
          [type(_c).__name__ for _c in _got], ["float", "float"])
check("calibrate/numpy bools fit the same cutoffs as python bools",
      calibrate_reliability([0.9, 0.4, 0.8], _np.array([True, False, True]), target_accuracy=1.0), _want)
# A NumPy boolean is still not a score, and a NumPy float is still not a label.
check_raises("calibrate/rejects numpy bool score", ValueError,
             lambda: calibrate_reliability([_np.True_], [True]))
check_raises("calibrate/rejects numpy float label", TypeError,
             lambda: calibrate_reliability([0.9], [_np.float64(1.0)]))
# Sized sequences only: a length mismatch has to be detectable, so a generator is refused.
check_raises("calibrate/rejects a generator of scores", TypeError,
             lambda: calibrate_reliability((x for x in [0.9]), [True]))

# Integer endpoints and tuple sequences remain valid; fitting must not mutate callers.
for target in (0, 1):
    for max_escalate in (0, 1):
        cuts = calibrate_reliability((0, 1), (False, True), target, max_escalate)
        check("calibrate/target=%d max_escalate=%d endpoints" % (target, max_escalate),
              cuts, (target, 0.0))
        check("calibrate/target=%d max_escalate=%d replay" % (target, max_escalate),
              [decide_reliability(score, *cuts) for score in (0, 1)],
              ["ACCEPT", "ACCEPT"] if target == 0 else ["VERIFY", "ACCEPT"])
unsorted_scores, unsorted_labels = [0.9, 0.1, 0.5], [True, False, False]
check("calibrate/unsorted evidence", calibrate_reliability(unsorted_scores, unsorted_labels), (0.9, 0.5))
check("calibrate/does not mutate evidence", (unsorted_scores, unsorted_labels),
      ([0.9, 0.1, 0.5], [True, False, False]))


# --------------------------------------------------------------- hook lifecycle: single state
hook = OptionStabilityHook(accept=0.77, escalate=0.61, seed="fixed-seed")
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
hook_fragile = OptionStabilityHook(accept=0.77, escalate=0.61, seed="fixed-seed")
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
hook_batch = OptionStabilityHook(accept=0.77, escalate=0.61, seed="batch-test")
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
hook_err = OptionStabilityHook(accept=0.77, escalate=0.61)
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


# --------------------------------------------------------------- layout-aware probe selection (#635, #951)
class MockAgent:
    def __init__(self, option_layout="sequential"):
        self.config = {"option_layout": option_layout}

# 1. Parallel layout: auto-probes run rename-only
parallel_agent = MockAgent(option_layout="parallel")
ctx_par = PredictContext(states=["test parallel"], questions={"ask": dict(q_choice["ask"])}, agent=parallel_agent)
hook_par = OptionStabilityHook(accept=0.77, escalate=0.61, seed="fixed-seed")
hook_par.on_predict_start(ctx_par)

# Only rename variants should be injected (no reversed or shuffle)
par_variant_names = [k for k in ctx_par.questions.keys() if k != "ask"]
check_true("parallel/injects variants", len(par_variant_names) > 0)
check("parallel/plan probes_ran is rename only", hook_par._plans[ctx_par.run_id]["ask"]["probes_ran"], ["rename"])
check("parallel/plan layout is parallel", hook_par._plans[ctx_par.run_id]["ask"]["layout"], "parallel")

sim_par = {}
for qk, qv in ctx_par.questions.items():
    crit = qv["criteria"]
    target_key = [k for k, desc in crit.items() if desc == "money back"][0]
    sim_par[qk] = {
        "type": "choice",
        "choice": target_key,
        "probabilities": {target_key: 0.90},
        "confidence": 0.90,
        "answer_confidence": 0.90,
    }
ctx_par.results = [{"answers": sim_par, "usage": {}}]
hook_par.on_predict_end(ctx_par)

par_rel = ctx_par.results[0]["answers"]["ask"]["reliability"]
check("parallel/reliability probes is ['rename']", par_rel["probes"], ["rename"])
check("parallel/reliability layout is parallel", par_rel["option_layout"], "parallel")
check_true("parallel/rename_stability is not None", par_rel["rename_stability"] is not None)
check("parallel/reorder_stability is None", par_rel["reorder_stability"], None)

# 2. Sequential layout: auto-probes run both reorder and rename
seq_agent = MockAgent(option_layout="sequential")
ctx_seq = PredictContext(states=["test sequential"], questions={"ask": dict(q_choice["ask"])}, agent=seq_agent)
hook_seq = OptionStabilityHook(accept=0.77, escalate=0.61, seed="fixed-seed")
hook_seq.on_predict_start(ctx_seq)
check("sequential/plan probes_ran is ['reorder', 'rename']", hook_seq._plans[ctx_seq.run_id]["ask"]["probes_ran"], ["reorder", "rename"])

sim_seq = {}
for qk, qv in ctx_seq.questions.items():
    crit = qv["criteria"]
    target_key = [k for k, desc in crit.items() if desc == "money back"][0]
    sim_seq[qk] = {
        "type": "choice",
        "choice": target_key,
        "probabilities": {target_key: 0.90},
        "confidence": 0.90,
        "answer_confidence": 0.90,
    }
ctx_seq.results = [{"answers": sim_seq, "usage": {}}]
hook_seq.on_predict_end(ctx_seq)

seq_rel = ctx_seq.results[0]["answers"]["ask"]["reliability"]
check("sequential/reliability probes is ['reorder', 'rename']", seq_rel["probes"], ["reorder", "rename"])
check_true("sequential/rename_stability is not None", seq_rel["rename_stability"] is not None)
check_true("sequential/reorder_stability is not None", seq_rel["reorder_stability"] is not None)

# 3. Explicit probe override: probes="reorder" on parallel checkpoint overrides auto
ctx_forced = PredictContext(states=["test forced"], questions={"ask": dict(q_choice["ask"])}, agent=parallel_agent)
hook_forced = OptionStabilityHook(accept=0.77, escalate=0.61, probes="reorder", seed="fixed-seed")
hook_forced.on_predict_start(ctx_forced)
check("override/plan probes_ran is reorder", hook_forced._plans[ctx_forced.run_id]["ask"]["probes_ran"], ["reorder"])

# 4. Unprobed single-option question: returns None signals and probes=[]
single_q = {"single": {"type": "choice", "instructions": "only one", "criteria": {"a": "only"}}}
ctx_single = PredictContext(states=["test single"], questions=single_q, agent=seq_agent)
hook_single = OptionStabilityHook(accept=0.77, escalate=0.61)
hook_single.on_predict_start(ctx_single)
check("single_option/no variants injected", len(ctx_single.questions), 1)
ctx_single.results = [{"answers": {"single": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}}}, "usage": {}}]
hook_single.on_predict_end(ctx_single)
single_rel = ctx_single.results[0]["answers"]["single"]["reliability"]
check("single_option/probes is empty", single_rel["probes"], [])
check("single_option/stability is None", single_rel["stability"], None)
check("single_option/soft_stability is None", single_rel["soft_stability"], None)
check("single_option/decision is None", single_rel["decision"], None)

# 5. Validation of new parameters
check_raises("validation/invalid probe string", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, probes="nonsense"))
check_raises("validation/empty probe list", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, probes=[]))
check_raises("validation/invalid probe list item", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, probes=["reorder", "bad"]))
check_raises("validation/invalid rename_max_options", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, rename_max_options=1))
check_raises("validation/bool rename_max_options", ValueError, lambda: OptionStabilityHook(accept=0.8, escalate=0.6, rename_max_options=True))

# 6. Layout normalization and warning fallback
title_par_agent = MockAgent(option_layout="  Parallel  ")
ctx_title_par = PredictContext(states=["test case normalization"], questions={"ask": dict(q_choice["ask"])}, agent=title_par_agent)
hook_norm = OptionStabilityHook(accept=0.77, escalate=0.61, seed="fixed-seed")
hook_norm.on_predict_start(ctx_title_par)
check("normalization/Parallel maps to parallel layout", hook_norm._plans[ctx_title_par.run_id]["ask"]["layout"], "parallel")

import warnings as _test_warnings
with _test_warnings.catch_warnings(record=True) as _warn_records:
    _test_warnings.simplefilter("always")
    bogus_agent = MockAgent(option_layout="unsupported_layout")
    ctx_bogus = PredictContext(states=["test bogus layout"], questions={"ask": dict(q_choice["ask"])}, agent=bogus_agent)
    hook_bogus = OptionStabilityHook(accept=0.77, escalate=0.61, seed="fixed-seed")
    hook_bogus.on_predict_start(ctx_bogus)
    check("normalization/bogus falls back to sequential", hook_bogus._plans[ctx_bogus.run_id]["ask"]["layout"], "sequential")
    check_true("normalization/bogus emits warning", any("unknown option_layout" in str(w.message) for w in _warn_records))


# 7. Issue #1285: Preserve option labels when renaming choices without descriptions
q_dict_none = {"type": "choice", "instructions": "Action?", "criteria": {"approve": None, "reject": None}}
vq_none_rel, back_none_rel = OptionStabilityHook._variant_question(q_dict_none, ["approve", "reject"], relabel=True)
check("issue1285/none_desc relabel preserves labels as descriptions", vq_none_rel["criteria"], {"A": "approve", "B": "reject"})
check("issue1285/none_desc back map", back_none_rel, {"A": "approve", "B": "reject"})

vq_none_ord, back_none_ord = OptionStabilityHook._variant_question(q_dict_none, ["reject", "approve"], relabel=False)
check("issue1285/none_desc reorder preserves original keys and values", vq_none_ord["criteria"], {"reject": None, "approve": None})
check("issue1285/none_desc reorder back map", back_none_ord, {"reject": "reject", "approve": "approve"})

q_dict_empty = {"type": "choice", "instructions": "Action?", "criteria": {"approve": "", "reject": ""}}
vq_empty_rel, _ = OptionStabilityHook._variant_question(q_dict_empty, ["approve", "reject"], relabel=True)
check("issue1285/empty_str relabel preserves labels as descriptions", vq_empty_rel["criteria"], {"A": "approve", "B": "reject"})

q_dict_ws = {"type": "choice", "instructions": "Action?", "criteria": {"approve": "   ", "reject": "  "}}
vq_ws_rel, _ = OptionStabilityHook._variant_question(q_dict_ws, ["approve", "reject"], relabel=True)
check("issue1285/whitespace relabel preserves labels as descriptions", vq_ws_rel["criteria"], {"A": "approve", "B": "reject"})

q_list = {"type": "choice", "instructions": "Action?", "criteria": ["approve", "reject"]}
vq_list_rel, _ = OptionStabilityHook._variant_question(q_list, ["approve", "reject"], relabel=True)
check("issue1285/list and dict forms produce identical rename criteria", vq_none_rel["criteria"], vq_list_rel["criteria"])

q_mixed = {"type": "choice", "instructions": "Action?", "criteria": {"approve": "Confirm order", "reject": None}}
vq_mixed_rel, _ = OptionStabilityHook._variant_question(q_mixed, ["approve", "reject"], relabel=True)
check("issue1285/mixed keeps text description and fills missing with label", vq_mixed_rel["criteria"], {"A": "Confirm order", "B": "reject"})

q_falsy = {"type": "choice", "instructions": "Action?", "criteria": {"level_0": 0, "flag": False}}
vq_falsy_rel, _ = OptionStabilityHook._variant_question(q_falsy, ["level_0", "flag"], relabel=True)
check("issue1285/falsy numeric and boolean descriptions preserved", vq_falsy_rel["criteria"], {"A": 0, "B": False})

# End-to-end hook execution with undescribed dict criteria (#1285)
ctx_undescribed = PredictContext(
    states=["Please confirm my invoice"],
    questions={"action": {"type": "choice", "instructions": "Choose action", "criteria": {"approve": None, "reject": None}}},
)
hook_undescribed = OptionStabilityHook(accept=0.77, escalate=0.61, seed="undescribed-test")
hook_undescribed.on_predict_start(ctx_undescribed)

# Original question criteria must remain untouched
check("issue1285/e2e original question criteria unchanged", ctx_undescribed.questions["action"]["criteria"], {"approve": None, "reject": None})
# Injected rename variant questions have original labels as descriptions
rename_variants = [q for k, q in ctx_undescribed.questions.items() if k.startswith("action::rel") and list(q["criteria"].keys())[0] == "A"]
check_true("issue1285/e2e rename variants present", len(rename_variants) > 0)
for rv in rename_variants:
    check("issue1285/e2e rename variant preserves option semantic text", set(rv["criteria"].values()), {"approve", "reject"})

# Simulate inference response and verify back-mapping restores original labels
sim_answers = {}
for qk, qv in ctx_undescribed.questions.items():
    crit = qv["criteria"]
    # Model chooses option representing "approve"
    chosen_key = [k for k, v in crit.items() if (v == "approve" or k == "approve")][0]
    other_key = [k for k in crit if k != chosen_key][0]
    sim_answers[qk] = {
        "type": "choice",
        "choice": chosen_key,
        "probabilities": {chosen_key: 0.9, other_key: 0.1},
        "confidence": 0.8,
        "answer_confidence": 0.9,
    }

ctx_undescribed.results = [{"model": "fake", "answers": sim_answers, "usage": {}}]
hook_undescribed.on_predict_end(ctx_undescribed)

ans_e2e = ctx_undescribed.results[0]["answers"]["action"]
check("issue1285/e2e decoded choice is approve", ans_e2e["choice"], "approve")
check("issue1285/e2e distinct choices is approve", ans_e2e["reliability"]["distinct_choices"], ["approve"])
check("issue1285/e2e stability is 1.0", ans_e2e["reliability"]["stability"], 1.0)
check("issue1285/e2e decision is ACCEPT", ans_e2e["reliability"]["decision"], "ACCEPT")

# Parity check: list criteria vs dict-with-None criteria generate matching variants
ctx_list_parity = PredictContext(
    states=["Please confirm my invoice"],
    questions={"action": {"type": "choice", "instructions": "Choose action", "criteria": ["approve", "reject"]}},
)
hook_list_parity = OptionStabilityHook(accept=0.77, escalate=0.61, seed="undescribed-test")
hook_list_parity.on_predict_start(ctx_list_parity)
check("issue1285/parity injected variant count matches", len(ctx_undescribed.questions), len(ctx_list_parity.questions))
for (k1, q1), (k2, q2) in zip(sorted(ctx_undescribed.questions.items()), sorted(ctx_list_parity.questions.items())):
    if k1.startswith("action::rel"):
        c1, c2 = q1["criteria"], q2["criteria"]
        if isinstance(c1, dict) and isinstance(c2, dict):
            check("issue1285/parity rename criteria match %s" % k1, c1, c2)
        else:
            # Reorder probe: dict keys match list entries exactly
            check("issue1285/parity reorder ordering matches %s" % k1, list(c1.keys()), list(c2))
hook_list_parity.on_error(ctx_list_parity)


# --------------------------------------------------------------- calibration precision and replay
def precision_report(supports, accept=None, escalate=None):
    """Use real hook aggregation over simulated per-variant model probabilities."""
    precision_hook = OptionStabilityHook(accept=accept, escalate=escalate, variants="fast")
    precision_ctx = PredictContext(states=["Keep the request"], questions={
        "action": {"type": "choice", "instructions": "Which action?",
                   "criteria": {"keep": "keep request", "defer": "defer request"}},
    })
    precision_hook.on_predict_start(precision_ctx)
    assert len(precision_ctx.questions) == len(supports) == 3
    answers = {}
    for (qid, question), support in zip(precision_ctx.questions.items(), supports):
        target = next(k for k, value in question["criteria"].items() if value == "keep request")
        other = next(k for k in question["criteria"] if k != target)
        answers[qid] = {"type": "choice", "choice": target if support >= 0.5 else other,
                        "probabilities": {target: support, other: 1.0 - support}}
    precision_ctx.results = [{"answers": answers, "usage": {}}]
    precision_hook.on_predict_end(precision_ctx)
    return precision_ctx.results[0]["answers"]["action"]["reliability"]


# The individual probabilities have four decimal places; their mean generally does not.
# A saved report must reproduce the live decision without changing caller-supplied cuts.
for name, supports, accept, escalate, expected in (
    ("accept round-up", [0.7999, 0.8, 0.8], 0.8, 0.6, "VERIFY"),
    ("escalate round-up", [0.5999, 0.6, 0.6], 0.8, 0.6, "ESCALATE"),
    ("precise caller cut", [0.8001, 0.8, 0.8], 0.80002, 0.6, "ACCEPT"),
    ("uncalibrated", [0.7999, 0.8, 0.8], None, None, None),
):
    report = precision_report(supports, accept, escalate)
    restored = json.loads(json.dumps(report))
    check("precision/%s live decision" % name, report["decision"], expected)
    check("precision/%s reports the gated mean" % name, report["soft_stability"], sum(supports) / 3)
    check("precision/%s JSON replay" % name,
          decide_reliability(restored["soft_stability"], accept, escalate), expected)

# Fit from persisted reports, then replay the exact same model outputs. The lower two
# scores round to the same four-place value, but each is a distinct possible cutoff.
calibration_supports = [
    [0.8, 0.6, 0.7],
    [0.8, 0.6, 0.7001],
    [0.8, 0.6, 0.7002],
]
saved_reports = json.loads(json.dumps([precision_report(row) for row in calibration_supports]))
saved_scores = [report["soft_stability"] for report in saved_reports]
fitted = calibrate_reliability(saved_scores, [False, False, True],
                              target_accuracy=1.0, max_escalate_accuracy=0.0)
fitted = json.loads(json.dumps(fitted))
check("precision/fitted accept is the observed upper mean", fitted[0], sum(calibration_supports[2]) / 3)
check("precision/fitted escalate is the observed middle mean", fitted[1], sum(calibration_supports[1]) / 3)
check("precision/JSON fit and hook replay preserve all three bands",
      [precision_report(row, *fitted)["decision"] for row in calibration_supports],
      ["ESCALATE", "VERIFY", "ACCEPT"])
check("precision/escalate boundary is exclusive",
      decide_reliability(fitted[1], *fitted), "VERIFY")
check("precision/accept boundary is inclusive",
      decide_reliability(fitted[0], *fitted), "ACCEPT")

# Rounding a sole candidate up can leave no retained row; rounding nearby candidates
# down can merge a wrong row into a group that has a valid perfect-accuracy cutoff.
for name, values, labels, expected_cut in (
    ("round-up retains a lone correct row", [0.80006], [True], 0.80006),
    ("round-down keeps nearby scores separable", [0.80001, 0.80002], [False, True], 0.80002),
):
    accept, escalate = calibrate_reliability(values, labels, target_accuracy=1.0)
    check("precision/calibrate %s" % name, accept, expected_cut)
    accepted = [ok for value, ok in zip(values, labels)
                if decide_reliability(value, accept, escalate) == "ACCEPT"]
    check("precision/calibrate %s retained group" % name, accepted, [True])

check("precision/exact ties cannot be split",
      calibrate_reliability([0.8, 0.8], [False, True], target_accuracy=1.0), (1.01, 0.0))
check("precision/duplicate scores enter as a whole group",
      calibrate_reliability([0.8, 0.8], [False, True], target_accuracy=0.5), (0.8, 0.0))
check("precision/zero and one retain their boundary cuts",
      calibrate_reliability([0.0, 1.0], [False, True], target_accuracy=1.0), (1.0, 0.0))
check("precision/zero is accepted at a zero cutoff", decide_reliability(0.0, 0.0, 0.0), "ACCEPT")
check("precision/sentinel accepts no probability", decide_reliability(1.0, 1.01, 0.0), "VERIFY")


# --------------------------------------------------------------- issue #1493: no-op rename probe
# When the option keys are already A, B, ... the "letters" rewrite maps every label onto
# itself and renders the original question again. make_variants only dedups (order, relabel),
# so the duplicate survived and weighted the original rendering twice in soft_stability.
q_upper = {"type": "choice", "instructions": "Is this review positive?",
           "criteria": {"A": "yes, the review is positive", "B": "no, the review is negative"}}
_noop = []
for _name, _order, _rel in make_variants(list(q_upper["criteria"]), "issue-1493"):
    _vq, _back = OptionStabilityHook._variant_question(q_upper, _order, _rel)
    if _name != "original" and list(_vq["criteria"].items()) == list(q_upper["criteria"].items()):
        _noop.append(_name)
check("issue1493/no-op rename is the letters variant", _noop, ["letters"])

hook_upper = OptionStabilityHook(accept=0.77, escalate=0.61, seed="issue-1493")
ctx_upper = PredictContext(states=["The review is great."], questions={"review": dict(q_upper)})
hook_upper.on_predict_start(ctx_upper)
check("issue1493/duplicate probe dropped", len([k for k in ctx_upper.questions if k.startswith("review::rel")]), 2)
check("issue1493/injected questions render distinct options",
      len({tuple(qv["criteria"].items()) for qv in ctx_upper.questions.values()}), len(ctx_upper.questions))

answers_upper = {}
for _qk, _qv in ctx_upper.questions.items():
    _crit = _qv["criteria"]
    _yes = [k for k, d in _crit.items() if d == "yes, the review is positive"][0]
    _no = [k for k in _crit if k != _yes][0]
    # stub scorer from the report: 0.95 on the original rendering, 0.65 on every real variant
    _support = 0.95 if list(_crit.items()) == list(q_upper["criteria"].items()) else 0.65
    answers_upper[_qk] = {"type": "choice", "choice": _yes,
                          "probabilities": {_yes: _support, _no: round(1 - _support, 4)}}
ctx_upper.results = [{"model": "fake", "answers": answers_upper, "usage": {}}]
hook_upper.on_predict_end(ctx_upper)
rel_upper = ctx_upper.results[0]["answers"]["review"]["reliability"]
check("issue1493/original weighted once", rel_upper["n_variants"], 3)
check_true("issue1493/soft_stability no longer double-weights the original",
           abs(rel_upper["soft_stability"] - 0.75) < 1e-9, repr(rel_upper["soft_stability"]))
check("issue1493/borderline answer stays VERIFY", rel_upper["decision"], "VERIFY")

# If the dropped duplicate was the only rename probe, nothing was probed and the signal is None.
hook_only_letters = OptionStabilityHook(accept=0.77, escalate=0.61, n_shuffles=0, variants=["letters"])
ctx_only_letters = PredictContext(states=["s"], questions={"review": dict(q_upper)})
hook_only_letters.on_predict_start(ctx_only_letters)
check("issue1493/no usable probe injected", len(ctx_only_letters.questions), 1)
answers_only = {}
for _qk, _qv in ctx_only_letters.questions.items():
    _crit = _qv["criteria"]
    _yes = [k for k, d in _crit.items() if d == "yes, the review is positive"][0]
    _no = [k for k in _crit if k != _yes][0]
    _support = 0.95 if list(_crit.items()) == list(q_upper["criteria"].items()) else 0.65
    answers_only[_qk] = {"type": "choice", "choice": _yes,
                         "probabilities": {_yes: _support, _no: round(1 - _support, 4)}}
ctx_only_letters.results = [{"model": "fake", "answers": answers_only, "usage": {}}]
hook_only_letters.on_predict_end(ctx_only_letters)
rel_only = ctx_only_letters.results[0]["answers"]["review"]["reliability"]
check("issue1493/no probe means no probes reported", rel_only["probes"], [])
check("issue1493/no probe means no stability signal", rel_only["soft_stability"], None)
check("issue1493/no probe means no decision", rel_only["decision"], None)


# --------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all option stability tests passed")
sys.exit(1 if FAIL else 0)
