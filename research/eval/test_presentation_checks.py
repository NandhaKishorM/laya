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
    ALL_CHECKS, CHECKS, CHOICE_KS, FIRST_SLOT_MIN, IDENTICAL_KS, IDENTICAL_TEXTS, INSTRUCTIONS, LANGUAGES,
    LEVELS, PARITY_TOL, SLOT0_MIN, STATES, check_choice_slot0_identical, check_score_first_slot_permuted,
    check_score_slot0_identical, choice_identical_question, exit_code, identical_question, leave_one_out,
    main, parse_langs, passes, permuted_questions, routes, run_checks,
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


# ------------------------------------------------------- fixed states in other languages
check("lang/registered", list(LANGUAGES), ["en", "ja", "ko", "hi", "tr"])
check_true("lang/en is the English constants",
           LANGUAGES["en"]["states"] is STATES and LANGUAGES["en"]["levels"] is LEVELS
           and LANGUAGES["en"]["instructions"] == INSTRUCTIONS
           and LANGUAGES["en"]["identical_texts"] is IDENTICAL_TEXTS)
for lang, spec in LANGUAGES.items():
    if lang == "en":
        continue
    check("lang/%s ten states" % lang, len(spec["states"]), 10)
    check("lang/%s no duplicates" % lang, len(set(spec["states"])), 10)
    check_true("lang/%s no empty state" % lang, all(s.strip() for s in spec["states"]))
    check_true("lang/%s not the English states" % lang, not set(spec["states"]) & set(STATES))
    check_true("lang/%s written in the language, not ASCII" % lang,
               not any(s.isascii() for s in spec["states"]))
    check("lang/%s three distinct levels" % lang, len(set(spec["levels"])), 3)
    check("lang/%s two distinct identical texts" % lang, len(set(spec["identical_texts"])), 2)
    check_true("lang/%s instructions" % lang, bool(spec["instructions"].strip()))
    for text in spec["identical_texts"]:
        q = identical_question(text, 3, spec["instructions"])
        rendered = render_options({"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]})
        check("lang/%s identical %s renders level-indexed" % (lang, text),
              rendered, ["level %d: %s" % (i, text) for i in range(3)])
    design = permuted_questions(spec["levels"], spec["instructions"])
    check_true("lang/%s every level in every slot twice" % lang,
               all(sum(1 for order, _ in design if order[slot] == li) == 2
                   for slot in range(3) for li in range(3)))
    check_true("lang/%s questions carry the language's instructions" % lang,
               all(q["instructions"] == spec["instructions"] for _, q in design))

    lang_calls = []

    def lang_counting(state, questions, _calls=lang_calls, _states=spec["states"]):
        assert state in _states
        _calls.append(len(questions))
        return FLAT(state, questions)

    run_checks(lang_counting, lang=lang)
    check("lang/%s one forward per state per check, on its own states" % lang, len(lang_calls), 20)
    by_label = scripted([0.0] * 5, dict(zip(spec["levels"], (0.0, 1.0, 2.0))))
    r = check_score_first_slot_permuted(by_label, lang=lang)
    check("lang/%s order-invariant model scores exactly 1/3" % lang, r["metric"], round(1 / 3, 4))
    check("lang/%s picks are counted by its own levels" % lang,
          r["picks_by_label"], {spec["levels"][0]: 0, spec["levels"][1]: 0, spec["levels"][2]: 60})
    r = check_score_slot0_identical(NO_SLOT0, lang=lang)
    check("lang/%s suppressed slot 0 gives the English value" % lang,
          r["metric"], round((-5 / 3 - 1.875 - 2.0) / 3, 4))

# Routing only, no weights: the non-English states go to the checkpoint #131 is about.
for lang, spec in LANGUAGES.items():
    check("route/%s" % lang, routes(spec["states"]),
          {"english": 10} if lang == "en" else {"multilingual": 10})


# --------------------------------------------------- the default run is unchanged
for fn in (FLAT, NO_SLOT0, EARLY, BY_LABEL):
    check_true("default/run_checks equals lang=en and states=STATES",
               run_checks(fn) == run_checks(fn, lang="en") == run_checks(fn, states=STATES))
check("default/slot0 by hand still", check_score_slot0_identical(NO_SLOT0)["metric"],
      round((-5 / 3 - 1.875 - 2.0) / 3, 4))


# ------------------------------------------------------------------- --lang parsing
check("langs/none means English", parse_langs(None), ["en"])
check("langs/single", parse_langs(["ja"]), ["ja"])
check("langs/comma-separated", parse_langs(["ja,ko"]), ["ja", "ko"])
check("langs/repeated flag, order kept, no repeats", parse_langs(["tr", "ja", "tr,ko"]), ["tr", "ja", "ko"])
check("langs/case and spaces", parse_langs([" JA , Hi "]), ["ja", "hi"])
for bad in (["xx"], ["ja,xx"], [","]):
    try:
        parse_langs(bad)
        FAIL.append("langs/%r should raise" % bad)
    except ValueError as exc:
        check_true("langs/%r raises naming the known codes" % bad, "ja, ko, hi, tr" in str(exc), str(exc))
check("langs/unknown code exits 2 before loading a checkpoint", main(["--lang", "xx"]), 2)


# ------------------------------------------------ choice identical-option control (#602 a)
check("choice/option counts, gated first", CHOICE_KS, (4, 3))
check("choice/kept out of the default run", sorted(CHECKS), ["score_first_slot_permuted", "score_slot0_identical"])
check("choice/registered beside it", sorted(ALL_CHECKS),
      ["choice_slot0_identical", "score_first_slot_permuted", "score_slot0_identical"])
for lang, spec in LANGUAGES.items():
    for text in spec["identical_texts"]:
        for k in CHOICE_KS:
            q = choice_identical_question(text, k, spec["instructions"])
            check("choice/%s %s K=%d numbered keys" % (lang, text, k), list(q["criteria"]),
                  [str(i + 1) for i in range(k)])
            rendered = render_options({"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]})
            check("choice/%s %s K=%d renders numbered key + shared description" % (lang, text, k),
                  rendered, ["%d: %s" % (i + 1, text) for i in range(k)])

r = check_choice_slot0_identical(FLAT)
check("choice/flat logits sit at zero and pass", (r["metric"], r["passed"]), (0.0, True))
r = check_choice_slot0_identical(NO_SLOT0)
# slot 0 at -2, the rest at +0.5: K=4 -2 - (-0.5/4) = -1.875, K=3 -2 - (-1/3) = -5/3
check("choice/suppressed slot 0 gated at K=4", r["metric"], -1.875)
check("choice/K=3 reported beside it", r["by_k"], {"4": -1.875, "3": round(-5 / 3, 4)})
check("choice/suppressed slot 0 fails", (r["k"], r["passed"]), (4, False))
r = check_choice_slot0_identical(scripted([-0.1, 0.0, 0.0, 0.0, 0.0]))
check("choice/a slight deficit above the gate passes", (r["metric"], r["passed"]), (-0.075, True))
r = check_choice_slot0_identical(scripted([-0.3, 0.0, 0.0, 0.0, 0.0]))
check("choice/a deficit below the gate fails", (r["metric"], r["passed"]), (-0.225, False))
r = check_choice_slot0_identical(EARLY)
check_true("choice/one-sided: early-slot preference passes", r["metric"] > 0 and r["passed"], str(r))
check("choice/one row per state", len(r["per_state"]), len(STATES))
check("choice/one entry per (text, K)", len(r["per_config"]), len(IDENTICAL_TEXTS) * len(CHOICE_KS))
for lang, spec in LANGUAGES.items():
    seen = []

    def recording(state, questions, _seen=seen):
        _seen.extend(q["instructions"] for q in questions)
        return FLAT(state, questions)

    r = check_choice_slot0_identical(recording, lang=lang)
    check_true("choice/%s asks in its own language" % lang, set(seen) == {spec["instructions"]})
    check("choice/%s per-config keys use its texts" % lang, sorted(r["per_config"]),
          sorted("%s/K=%d" % (t, k) for t in spec["identical_texts"] for k in CHOICE_KS))

calls = []
run_checks(lambda s, q: calls.append(len(q)) or FLAT(s, q), list(ALL_CHECKS), lang="ja")
check("choice/all three checks: one forward per state per check", len(calls), 3 * 10)
check("choice/default run_checks still runs the two score checks only", sorted(run_checks(FLAT)["checks"]),
      ["score_first_slot_permuted", "score_slot0_identical"])
check("choice/unknown check name exits 2", main(["--checks", "nope"]), 2)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
