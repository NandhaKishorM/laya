"""Metric math, dataset parsing and regression comparison for laya.evals. No weights.

Run: python -m pytest tests/test_evals.py -q
"""
import json
import re
import warnings

import numpy as np

import pytest

import laya
from laya import evals
from laya.evals import (
    ChoiceAccuracy,
    Dataset,
    EvalError,
    EvalReport,
    Example,
    MeanConfidence,
    NoulAccuracy,
    ScoreMAE,
    ScoreWithin,
    assert_regression,
    aurc,
    brier,
    default_evaluators,
    ece,
    evaluate,
    is_confidence_metric,
    selective_accuracy,
)

Q = {"intent": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}}}
QSCORE = {"quality": {"type": "score", "instructions": "?", "criteria": ["low", "high"]}}
QNOUL = {"flag": {"type": "noul", "instructions": "?"}}


def choice_answer(label, confidence=0.9):
    return {"type": "choice", "choice": label, "probabilities": {label: confidence},
            "confidence": confidence}


def noul_answer(prob):
    return {"type": "noul", "noul": prob, "confidence": max(prob, 1 - prob)}


def score_answer(value, confidence=0.8):
    return {"type": "score", "score": value, "confidence": confidence}


class StubRunner:
    """Returns fixed answers per state, so evaluation is deterministic and weight-free."""

    def __init__(self, by_state):
        self.by_state = by_state

    def predict(self, state, questions, model=None):
        return {"model": model or "stub", "answers": self.by_state[state]}


# --------------------------------------------------------------- evaluator math
def test_choice_and_confidence_math():
    evaluator = ChoiceAccuracy()
    assert evaluator.score(choice_answer("a"), "a") == 1.0
    assert evaluator.score(choice_answer("b"), "a") == 0.0
    assert evaluator.score(noul_answer(0.9), "a") is None, "wrong answer type does not apply"
    assert MeanConfidence().score(choice_answer("a", 0.8), "a") == 0.8


def test_noul_and_score_math():
    assert NoulAccuracy().score(noul_answer(0.9), True) == 1.0
    assert NoulAccuracy().score(noul_answer(0.2), True) == 0.0
    assert ScoreMAE().score({"type": "score", "score": 0.4}, 0.7) == pytest.approx(0.3)
    within = ScoreWithin(0.5)
    assert within.score({"type": "score", "score": 0.4}, 0.7) == 1.0
    assert within.name == "score_within_0.5"


def test_calibration_uses_answer_confidence():
    answer = {"type": "choice", "choice": "a", "confidence": 0.2, "answer_confidence": 0.9}
    assert MeanConfidence().score(answer, "a") == pytest.approx(0.9), "calibrated, not entropy"
    report = evaluate(StubRunner({"s": {"intent": answer}}),
                      Dataset([Example("s", Q, {"intent": "a"})]))
    assert report.overall["mean_confidence"] == pytest.approx(0.9)


# ---------------------------------------------------------------- which confidence a metric reads
# `_answer_confidence` prefers `answer_confidence`, and an answer that carries none gets the entropy
# `confidence` -- a different quantity on a different scale (#394), which the docstring used to call
# "the calibrated confidence Laya reports ... on every answer type" while docs/evals.md told the
# operator the same. The shape that really arrives without the field is the strict Jev wire contract:
# `LAYA_JEV_STRICT` drops it (`laya/serve.py::_project_jev_strict`, documented in docs/http-api.md).
# So the stripped answers below are made by the server's own projector rather than hand-written -- a
# hand copy could drift from what the server serves and this gate would still pass. `laya.serve`
# imports with neither fastapi nor torch at module scope, which is why it is affordable here.

_CONTRACT_PROBS = {"choice": [0.95, 0.03, 0.02], "score": [0.05, 0.90, 0.05], "noul": [0.13, 0.87]}
_CONTRACT_KEYS = {"choice": ["billing", "sales", "other"], "score": ["low", "mid", "high"],
                  "noul": ["no", "yes"]}
_CONTRACT_LABEL = {"choice": "billing", "score": "mid", "noul": True}


def _agent_answer(kind):
    """The answer dict `laya/agent.py` publishes, built the way the decoder builds it: entropy
    `confidence` for `choice` and `score`, `max(p_true, 1 - p_true)` for `noul`, and
    `answer_confidence` = max(p) -- see laya/agent.py:1373, :1392, :1400."""
    p = _CONTRACT_PROBS[kind]
    argmax = max(range(len(p)), key=lambda i: p[i])
    from laya.common import confidence_from_probs
    conf = (round(max(p[1], 1.0 - p[1]), 4) if kind == "noul"
            else round(confidence_from_probs(np.asarray(p), len(p)), 4))
    answer = {"confidence": conf, "answer_confidence": round(float(p[argmax]), 4),
              "type": kind}
    if kind == "choice":
        return dict(answer, choice=_CONTRACT_KEYS[kind][argmax],
                    probabilities=dict(zip(_CONTRACT_KEYS[kind], p)))
    if kind == "score":
        return dict(answer, score=round(sum(i * v for i, v in enumerate(p)), 4),
                    legend={str(i): k for i, k in enumerate(_CONTRACT_KEYS[kind])},
                    probabilities={str(i): v for i, v in enumerate(p)})
    return dict(answer, noul=p[1])


def _contract_shapes():
    """`(full, strict)` -- the payload the agents publish, and the same payload put through the
    projector that serves the strict Jev contract."""
    from laya.serve import _project_jev_strict
    full = {kind: _agent_answer(kind) for kind in _CONTRACT_PROBS}
    strict = _project_jev_strict({"model": "stub", "answers": dict(full),
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}})["answers"]
    return full, strict


def _contract_report(shapes):
    questions = {"q": {"type": "choice", "instructions": "?",
                       "criteria": dict(zip(_CONTRACT_KEYS["choice"], ["c"] * 3))}}
    by_state = {"state-%s" % kind: {"q": shapes[kind]} for kind in shapes}
    dataset = Dataset([Example("state-%s" % kind, questions, {"q": _CONTRACT_LABEL[kind]})
                       for kind in shapes])
    return evaluate(StubRunner(by_state), dataset)


def test_serve_contract_projection_is_the_shape_without_the_field():
    full, strict = _contract_shapes()
    for kind in ("choice", "score", "noul"):
        assert "answer_confidence" in full[kind]
        assert "answer_confidence" not in strict[kind], (
            "the projector is what makes the fallback reachable; if it stopped dropping the field, "
            "the prose this test guards would need re-reading, not these numbers")
    assert "confidence" in strict["choice"] and "confidence" in strict["score"]
    assert "confidence" not in strict["noul"], "docs/http-api.md: noul keeps only `noul`"


def test_a_strict_answer_is_calibrated_against_the_entropy_number():
    full, strict = _contract_shapes()
    for kind in ("choice", "score"):
        assert evals._answer_confidence(full[kind]) == pytest.approx(full[kind]["answer_confidence"])
        assert evals._answer_confidence(strict[kind]) == pytest.approx(full[kind]["confidence"]), \
            "with the field gone the column silently becomes the entropy number"
    # A strict `noul` carries neither field, so it is the only fallback that lands on the right
    # quantity: max(p_true, 1 - p_true) is what the decoder reports as its answer_confidence.
    assert evals._answer_confidence(strict["noul"]) == pytest.approx(
        full["noul"]["answer_confidence"])


def test_the_two_shapes_do_not_score_the_same_report():
    full, strict = _contract_shapes()
    over_full, over_strict = _contract_report(full), _contract_report(strict)
    assert over_full.overall["mean_confidence"] == pytest.approx(0.9066666666666666)
    assert over_strict.overall["mean_confidence"] == pytest.approx(0.7665666666666667)
    # ece sees the two rows whose answer type yields a known label (choice, noul); the score row has
    # no correctness rule, so only mean_confidence moves for it.
    assert over_full.overall["ece"] == pytest.approx(0.09)
    assert over_strict.overall["ece"] == pytest.approx(0.17065)
    assert over_strict.overall["ece"] - over_full.overall["ece"] == pytest.approx(0.08065), \
        "the same answers, the same labels, a different field: 0.09 -> 0.17065"


def test_the_docs_name_the_fallback_and_refuse_to_call_the_shipped_number_calibrated():
    import ast
    import inspect
    import pathlib

    doc = inspect.getsource(evals._answer_confidence)
    flat = " ".join(doc.split())
    assert "falls through" in flat and "normalized entropy" in flat
    # The number of fallbacks the docstring teaches is read off the code's own Returns, so a fourth
    # fallback cannot be added without re-teaching the sentence that enumerates them.
    reads = [node for node in ast.walk(ast.parse(doc).body[0])
             if isinstance(node, ast.Return) and not isinstance(node.value, ast.Constant)]
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven")
    taught = words[len(reads) - 1]
    assert "falls through %s more reads, in order" % taught in flat, (
        "the docstring must enumerate the fallbacks in order, and the count it teaches must be the "
        "code's: %d returning reads, one of them the preferred `answer_confidence`" % len(reads))
    assert "LAYA_JEV_STRICT" in flat, "the shape that lacks the field has to be named"
    assert "over-confident" in flat, "the shipped level must not be called calibrated"
    assert "is not *calibrated* as shipped" in flat
    assert "The calibrated confidence" not in doc, "main's opening claim, verbatim"

    page = (pathlib.Path(__file__).resolve().parent.parent
            / "docs" / "evals.md").read_text(encoding="utf-8")
    prose = " ".join(page.split())
    assert "the calibrated probability Laya reports on every answer type" not in prose
    assert "`laya.evals._answer_confidence`" in prose, "the page names the function it describes"
    assert "then\n`max(p, 1 - p)`" in page or "then `max(p, 1 - p)`" in prose
    assert "LAYA_JEV_STRICT" in prose and "over-confident" in prose
    assert "## Which confidence a metric reads" in page, "the fallback needs a home, not a footnote"

    # Every metrics-table row that names the column has to carry the same hedge the prose does.
    rows = [line for line in page.splitlines()
            if line.startswith("| `") and 'answer["answer_confidence"]' in line]
    assert rows, "the page must tell the operator which field the metrics read"
    for row in rows:
        assert "where the answer carries it" in row or "#which-confidence-a-metric-reads" in row, (
            "a row that names only `answer_confidence` claims the fallback away: %s" % row[:100])

    # And the claim the evals page leans on has to still be the one the HTTP page makes.
    api = (pathlib.Path(__file__).resolve().parent.parent
           / "docs" / "http-api.md").read_text(encoding="utf-8")
    row = [line for line in api.splitlines() if "`LAYA_JEV_STRICT`" in line]
    assert row and "no per-answer `action` / `answer_confidence`" in row[0]



def test_compare_ignores_latency_by_default():
    report = EvalReport(overall={"choice_accuracy": 0.8, "latency_p50_ms": 12.0})
    baseline = {"overall": {"choice_accuracy": 0.8, "latency_p50_ms": 5.0}}
    ok, deltas = report.compare(baseline)
    assert ok and "latency_p50_ms" not in deltas, "timing noise is not a quality regression"
    bad, deltas = report.compare(baseline, {"latency_p50_ms": 1.0})
    assert not bad and "latency_p50_ms" in deltas


def test_ece_on_known_inputs():
    assert ece([1.0, 1.0], [True, False]) == pytest.approx(0.5)
    assert ece([0.0, 0.0], [False, False]) == pytest.approx(0.0)
    assert ece([], []) is None


def test_selective_metrics_on_known_inputs():
    # Brier of confidence-as-P(correct)
    assert brier([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert brier([1.0, 1.0], [True, False]) == pytest.approx(0.5)
    assert brier([], []) is None
    # A confidence that ranks right-from-wrong perfectly: top-2 both correct, bottom-2 both wrong.
    conf, corr = [0.9, 0.8, 0.2, 0.1], [True, True, False, False]
    # risk@coverage = 1 - cum_correct/k over k=1..4 -> [0, 0, 1/3, 1/2]; AURC = mean = 0.2083
    assert aurc(conf, corr) == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4, abs=1e-6)
    assert selective_accuracy(conf, corr, 0.5) == pytest.approx(1.0)   # top 2 are both correct
    assert selective_accuracy(conf, corr, 1.0) == pytest.approx(0.5)   # all four -> 2/4
    assert selective_accuracy([], [], 0.5) is None
    assert selective_accuracy(conf, corr, 0.0) is None                 # coverage must be in (0, 1]
    # a worse ranking has higher AURC on the same accuracy
    assert aurc([0.1, 0.2, 0.9, 0.8], corr) > aurc(conf, corr)
    for name in ("ece", "brier", "aurc", "selective_accuracy@50"):
        assert is_confidence_metric(name)
    assert not is_confidence_metric("choice_accuracy")


def test_coverage_cut_does_not_split_a_tie_group():
    """A coverage cut is a confidence threshold, so permuting the dataset cannot move it.

    Before this was fixed, a stable sort made the result reproducible but not order-independent:
    `y[:k]` took whichever tie members the dataset listed first. Ties are the normal case, not a
    corner -- `common.answer_confidence`'s own docstring records that the shipped `choice:11+`
    bucket "returns a point mass at 1.0".
    """
    conf = [1.0] * 8
    correct_first = [True, True, True, True, False, False, False, False]
    wrong_first = [False, False, False, False, True, True, True, True]
    interleaved = [True, False, True, False, True, False, True, False]
    # 8 rows, 4 correct, every confidence identical: the only honest answer is the group's own
    # accuracy, whatever order the rows arrive in. Before: 1.000 / 0.000 / 0.500 for sel@50.
    for y in (correct_first, wrong_first, interleaved):
        assert selective_accuracy(conf, y, 0.5) == pytest.approx(0.5)
        assert selective_accuracy(conf, y, 0.8) == pytest.approx(0.5)
        assert aurc(conf, y) == pytest.approx(0.5)
    # `brier` and `ece` do not cut, so a change to what a cut accepts cannot move them. Asserted on
    # DISTINCT, deliberately unsorted confidences and by comparing two orderings to each other: on
    # the all-tied data above, any confidence-sorting rewrite is the identity, so a mutant that
    # permuted the rows by confidence passed every one of these.
    jumbled_conf = [0.3, 0.95, 0.6, 0.1, 0.8]
    jumbled_corr = [False, True, True, False, False]
    order = [3, 0, 4, 2, 1]
    other_conf = [jumbled_conf[i] for i in order]
    other_corr = [jumbled_corr[i] for i in order]
    assert brier(jumbled_conf, jumbled_corr) == pytest.approx(
        brier(other_conf, other_corr), abs=1e-12)
    assert ece(jumbled_conf, jumbled_corr) == pytest.approx(
        ece(other_conf, other_corr), abs=1e-12)
    # and the values themselves, so a rewrite that made both sides equally wrong is still caught
    assert brier(jumbled_conf, jumbled_corr) == pytest.approx(
        sum((c - y) ** 2 for c, y in zip(jumbled_conf, jumbled_corr)) / 5, abs=1e-12)


def test_coverage_cut_extends_through_the_tie_it_lands_in():
    """The threshold is the confidence at the cut, and it accepts every row at that confidence."""
    # two groups: 0.9 (both correct) and 0.5 (two correct of four). The FIRST member of the 0.5
    # group is wrong on purpose: with a correct one there, accepting a row below the threshold
    # cannot change the mean, and an implementation that over-accepts passes unnoticed.
    conf = [0.9, 0.9, 0.5, 0.5, 0.5, 0.5]
    corr = [True, True, False, True, True, False]
    # coverage 1/3 -> k = ceil(2.0) = 2, the cut is 0.9, and nothing at 0.5 is accepted. If one
    # were, this would be 2/3 rather than 1.0.
    assert selective_accuracy(conf, corr, 1 / 3) == pytest.approx(1.0)
    # coverage 0.5 -> k = ceil(3.0) = 3, which lands inside the 0.5 group, so the threshold is
    # 0.5 and all six rows are accepted: 4 correct of 6. The figure therefore covers the group's
    # upper edge (6 answers) rather than the 3 asked for -- the only order-independent reading.
    assert selective_accuracy(conf, corr, 0.5) == pytest.approx(4 / 6)
    assert selective_accuracy(conf, corr, 0.8) == pytest.approx(4 / 6)
    # aurc integrates one risk per distinct level over the ROWS that level spans: 0.9 -> 2 rows
    # accepted, 0 errors, risk 0; 0.5 -> 6 accepted, 2 errors, risk 2/6, spanning the other 4 rows.
    # Weighted, not averaged: an unweighted mean of the two risks would weight a 2-row level the
    # same as a 4-row one, which is how a 999-row level and a 1-row level came to count alike.
    assert aurc(conf, corr) == pytest.approx((0.0 * 2 + (2 / 6) * 4) / 6)


def test_coverage_metrics_are_unchanged_without_ties():
    """Distinct confidences must produce exactly what they always did, to the last bit.

    The fix is only allowed to move a number a permutation of the same dataset could already
    move. With no ties there is one level per row, so every point of the risk-coverage curve --
    and `k` itself -- is what it was before.
    """
    conf, corr = [0.9, 0.8, 0.2, 0.1], [True, True, False, False]
    # the hand-computed values from `test_selective_metrics_on_known_inputs`, restated as floats
    # so this fails if the definition drifts rather than if that test is edited
    assert aurc(conf, corr) == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4, abs=1e-12)
    assert selective_accuracy(conf, corr, 0.5) == pytest.approx(1.0, abs=1e-12)
    assert selective_accuracy(conf, corr, 0.75) == pytest.approx(2 / 3, abs=1e-12)
    assert selective_accuracy(conf, corr, 1.0) == pytest.approx(0.5, abs=1e-12)
    # a single row, and a coverage that rounds below one row, still report that row
    assert selective_accuracy([0.4], [True], 0.01) == pytest.approx(1.0)
    assert aurc([0.4], [False]) == pytest.approx(1.0)
    # `coverage * n` is rarely an integer, and the fraction rounds UP so the answer covers at
    # least the fraction asked for. Five rows at 0.5 means three, not two -- every case above
    # happens to land on a whole row, so nothing here could see a floor in place of the ceiling.
    five, corr5 = [0.9, 0.8, 0.7, 0.6, 0.5], [True, True, False, True, True]
    assert selective_accuracy(five, corr5, 0.5) == pytest.approx(2 / 3)     # ceil(2.5) = 3 rows
    # 0.3 -> ceil(1.5) = 2 rows. The second row is WRONG, so a floor (1 row) would report 1.0.
    five_mixed = [0.9, 0.8, 0.7, 0.6, 0.5]
    corr_mixed = [True, False, True, True, True]
    assert selective_accuracy(five_mixed, corr_mixed, 0.3) == pytest.approx(0.5)
    assert selective_accuracy(five, corr5, 0.7) == pytest.approx(3 / 4)     # ceil(3.5) = 4 rows


def test_non_finite_confidences_are_order_independent_too():
    """A NaN confidence must not reintroduce the order-dependence this change removes.

    Confidences come from whatever a runner returns, so NaN and infinities are reachable --
    `_eval_policy._finite_number` exists because this codebase already expects that. `NaN != NaN`,
    so a boundary test written on `!=` alone would make every NaN row its own level and walk them
    in dataset order: the headline defect, surviving for exactly the input a custom runner is most
    likely to hand over by accident. All NaNs are one group, ranked last.

    These VALUES differ from what the old definition reported. They have to: the old ones were
    whichever rows the dataset listed first. `docs/evals.md` says so.
    """
    nan, inf = float("nan"), float("inf")
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # a RuntimeWarning here fails the test
        # all NaN: one group, so every coverage reports the group's own accuracy
        for y in ([True, False, True], [True, True, False], [False, True, True]):
            assert selective_accuracy([nan] * 3, y, 0.5) == pytest.approx(2 / 3)
            assert aurc([nan] * 3, y) == pytest.approx(1 / 3)
        # a NaN among finite values: the finite ones rank above it, the NaN group below
        assert selective_accuracy([0.9, nan, 0.5], [True, False, True], 0.5) == pytest.approx(1.0)
        # infinities are ordinary numbers to a comparison, and keep ranking normally
        assert selective_accuracy([inf, 0.5, -inf], [True, True, False], 0.5) == pytest.approx(1.0)
        assert aurc([0.9, 0.7, nan, 0.2], [True, False, True, True]) == pytest.approx(
            (0 * 1 + 0.5 * 1 + (1 - 2 / 3) * 1 + (1 - 3 / 4) * 1) / 4, abs=1e-12)

    # and the whole report, through `evaluate()`, with only the NaN rows reordered
    def report_for(rows):
        examples = [Example("s%d" % i, Q, {"intent": "a"}) for i, _ in enumerate(rows)]
        answers = {"s%d" % i: {"intent": dict(choice_answer("a" if ok else "b", conf),
                                              answer_confidence=conf)}
                   for i, (conf, ok) in enumerate(rows)}
        rep = evaluate(StubRunner(answers), Dataset(examples), evaluators=[ChoiceAccuracy()])
        return {n: rep.overall[n] for n in ("aurc", "selective_accuracy@50",
                                           "selective_accuracy@80")}

    finite = [(0.9, True), (0.8, True), (0.7, True), (0.6, True)]
    block = [(nan, True), (nan, False), (nan, True), (nan, False), (nan, True), (nan, False)]
    assert report_for(finite + block) == report_for(finite + block[::-1])


def test_confidences_group_on_exact_equality():
    """Two confidences that differ by one ULP are two levels, not one.

    A threshold comparison is exact, so grouping has to be too. Nothing else in this file puts two
    confidences closer together than 0.1, which let a tolerance window or an `isclose` boundary
    pass every check while changing which answers a cut accepts. The shipped agents round
    confidence to 4 decimals, so this is reachable only from a custom runner -- which is also true
    of every NaN case above.
    """
    one = 1.0
    just_below = 1.0 - 2 ** -52                      # the next double below 1.0
    assert just_below != one and round(just_below, 4) == round(one, 4)
    # two levels: the correct answer ranks first and the 50% cut takes only it
    assert selective_accuracy([one, just_below], [True, False], 0.5) == pytest.approx(1.0)
    assert aurc([one, just_below], [True, False]) == pytest.approx((0.0 * 1 + 0.5 * 1) / 2)
    # one level, were they treated as equal, would report the pair's own accuracy instead
    assert selective_accuracy([one, one], [True, False], 0.5) == pytest.approx(0.5)


def test_corrects_are_read_as_booleans():
    """`corrects` is a correctness flag, not a count.

    `_levels` reads it through `dtype=bool`; reading it as a float lets a value outside {0, 1}
    produce a NEGATIVE risk, and `brier` already reads the same argument as a float, so the two
    readings were unpinned and mutually inconsistent.
    """
    assert aurc([0.9, 0.8], [2, 0]) == pytest.approx(aurc([0.9, 0.8], [True, False]))
    assert aurc([0.9, 0.8], [2, 0]) >= 0.0
    assert selective_accuracy([0.9, 0.8], [2, 0], 1.0) == pytest.approx(0.5)


def test_a_0d_confidence_still_answers():
    """A bare scalar used to answer, and indexing it with an argsort result raises.

    Reachable only from the library entry point, and `_aggregate` always builds lists -- but an
    `IndexError` escaping `laya-evals` exits 1, which is the code that means "the model regressed".
    """
    assert selective_accuracy(np.float64(0.5), np.True_, 0.5) == pytest.approx(1.0)
    assert aurc(np.float64(0.5), np.False_) == pytest.approx(1.0)


def test_report_metrics_survive_a_permuted_dataset():
    """The gate reads these through `evaluate()`, so permuting a JSONL must not move them.

    `_eval_policy.check_policy` gates a slice on whatever metric a policy names, and
    `evals.is_confidence_metric` recognises `aurc` and `selective_accuracy@*` by name -- so a
    coverage metric that depended on row order meant a shuffled dataset could pass or fail a
    release gate on nothing at all.
    """
    examples = [Example("s%d" % i, Q, {"intent": "a"}) for i in range(1, 9)]
    # every answer at the same confidence, half of them wrong: the tie case, through the report
    answers = {"s%d" % i: {"intent": choice_answer("a" if i <= 4 else "b", 1.0)}
               for i in range(1, 9)}
    names = ("aurc", "selective_accuracy@50", "selective_accuracy@80", "ece", "brier")

    def metrics(order):
        report = evaluate(StubRunner(answers), Dataset([examples[i] for i in order]),
                          evaluators=[ChoiceAccuracy()])
        return {n: report.overall[n] for n in names}

    forward = metrics(range(8))
    assert metrics(range(7, -1, -1)) == forward
    assert metrics([4, 0, 5, 1, 6, 2, 7, 3]) == forward
    # and the values are the tie group's own accuracy, not whichever half came first
    assert forward["selective_accuracy@50"] == pytest.approx(0.5)
    assert forward["aurc"] == pytest.approx(0.5)


def test_a_relative_rule_refuses_a_baseline_from_the_old_coverage_definition(tmp_path):
    """A `max_drop` across definitions is unsafe in the PASS direction, so it must not be computed.

    The two definitions disagree by up to 0.500 on the same answers. A slice recorded at 0.033
    by the old one reads 0.517 under this one, so subtracting them lets a candidate that genuinely
    dropped 0.217 clear a `max_drop` of 0.05 -- a real regression shipping because a baseline is
    stale. `comparable_to` cannot see it: the dataset bytes and the question schema are identical,
    and its "a key missing on either side is unknown" rule exists so a patch release cannot
    invalidate a baseline. A definition change is not a patch release, so this refuses on the
    key's ABSENCE.
    """
    from laya import _eval_policy

    policy = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "selective_accuracy@50",
                   "min_count": 5, "max_drop": 0.05}))
    candidate = EvalReport(**_coverage_gate_report(0.30, evals.COVERAGE_METRIC_DEFINITION))

    stale = _coverage_gate_report(0.033333, None)
    failures = _eval_policy.check_policy(candidate, policy, stale)
    assert any("regenerate it before comparing" in f for f in failures), failures
    assert any("the baseline says 1 (unrecorded)" in f for f in failures), failures
    # an explicit version 1 is refused the same way
    assert any("regenerate it before comparing" in f for f in
               _eval_policy.check_policy(candidate, policy, _coverage_gate_report(0.033333, 1)))

    # regenerated under this definition, the same gate CATCHES the regression it was hiding
    fresh = _coverage_gate_report(0.516667, evals.COVERAGE_METRIC_DEFINITION)
    failures = _eval_policy.check_policy(candidate, policy, fresh)
    assert any("max_drop" in f and "exceeded" in f for f in failures), failures
    assert not any("regenerate" in f for f in failures), failures


def test_the_refusal_is_limited_to_metrics_a_cut_can_move(tmp_path):
    """`ece` and `brier` do not cut, so a baseline that predates the change stays comparable.

    Sweeping them into the refusal would break every committed baseline for no reason.
    """
    from laya import _eval_policy

    candidate = EvalReport(**_coverage_gate_report(0.10, evals.COVERAGE_METRIC_DEFINITION,
                                                   metric="ece"))
    stale = _coverage_gate_report(0.11, None, metric="ece")
    policy = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "ece",
                   "min_count": 5, "max_drop": 0.05}))
    assert _eval_policy.check_policy(candidate, policy, stale) == []
    assert evals.is_coverage_metric("aurc")
    assert evals.is_coverage_metric("selective_accuracy@80")
    assert not evals.is_coverage_metric("ece")
    assert not evals.is_coverage_metric("brier")
    assert not evals.is_coverage_metric("choice_accuracy")


def test_an_absolute_rule_needs_no_baseline_and_is_unaffected(tmp_path):
    """Only the relative comparison is across definitions; a `min` reads this run alone."""
    from laya import _eval_policy

    policy = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "selective_accuracy@50",
                   "min_count": 5, "min": 0.2}))
    candidate = EvalReport(**_coverage_gate_report(0.30, evals.COVERAGE_METRIC_DEFINITION))
    assert _eval_policy.check_policy(candidate, policy, None) == []


def test_a_stale_candidate_is_refused_as_well_as_a_stale_baseline(tmp_path):
    """The dangerous direction: an old-definition CANDIDATE can read BETTER than the truth.

    A report scored by the old definition carries a row-order artifact. Here the same slice reads
    1.000 under definition 1 (its correct rows happened to be listed first) where definition 2 says
    0.500, so subtracting it from a correctly regenerated 0.500 baseline shows no drop at all and
    the gate passes -- while the honestly scored candidate fails. Checking only the baseline leaves
    this open, and it is the easier of the two to reach by accident: a CI artifact produced by a
    pinned older `laya`, or an archived report re-checked after the baseline was regenerated.
    """
    from laya import _eval_policy

    policy = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "selective_accuracy@50",
                   "min_count": 5, "max_drop": 0.05}))
    fresh_baseline = _coverage_gate_report(0.50, evals.COVERAGE_METRIC_DEFINITION)
    stale_candidate = EvalReport(**_coverage_gate_report(1.0, None))

    failures = _eval_policy.check_policy(stale_candidate, policy, fresh_baseline)
    assert any("this report says 1 (unrecorded)" in f for f in failures), failures
    # Proof the refusal is what stops it: the subtraction itself sees 1.0 against 0.50, which is a
    # GAIN, so without the refusal there is no `max_drop` failure to catch the regression.
    assert not any("max_drop" in f for f in failures), failures
    # both sides stale is refused too, and says so about both
    both = _eval_policy.check_policy(stale_candidate, policy, _coverage_gate_report(0.50, None))
    assert any("this report says" in f and "the baseline says" in f for f in both), both


def test_compare_refuses_a_coverage_metric_across_definitions():
    """`EvalReport.compare` is the gate this project's docs lead with, and it also subtracts.

    `--gate-policy` is opt-in; `--baseline --tolerance` is the documented default. A refusal that
    covered only the policy gate would leave the usual path performing exactly the cross-definition
    subtraction the policy gate exists to refuse.
    """
    report = EvalReport(**_coverage_gate_report(0.50, evals.COVERAGE_METRIC_DEFINITION))
    stale = _coverage_gate_report(0.47, None)

    ok, deltas = report.compare(stale, {"selective_accuracy@50": 0.10})
    assert not ok, deltas
    assert "incomparable" in deltas["selective_accuracy@50"], deltas
    # Without the refusal this passes: 0.50 against 0.47 is a 0.03 move inside a 0.10 tolerance,
    # while the regenerated baseline for the same answers is 0.03 and the real move is 0.47.
    fresh = _coverage_gate_report(0.47, evals.COVERAGE_METRIC_DEFINITION)
    ok_same_definition, deltas_same = report.compare(fresh, {"selective_accuracy@50": 0.10})
    assert ok_same_definition, deltas_same
    assert "incomparable" not in deltas_same["selective_accuracy@50"]
    # `ece` does not cut, so an unstamped baseline stays comparable
    ece = EvalReport(**_coverage_gate_report(0.10, evals.COVERAGE_METRIC_DEFINITION, metric="ece"))
    ok_ece, deltas_ece = ece.compare(_coverage_gate_report(0.11, None, metric="ece"),
                                     {"ece": 0.05})
    assert ok_ece, deltas_ece


def test_the_refusal_bounds_a_hostile_recorded_definition(tmp_path):
    """The stamp is read from a report file, which on a pull-request gate the author writes.

    The refusal names the value it found, and that message is printed to a CI log, so the value is
    attacker-controlled content on its way into a log. Five megabytes in the field produced a
    five-megabyte failure line before this was bounded.
    """
    from laya import _eval_policy

    policy = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "selective_accuracy@50",
                   "min_count": 5, "max_drop": 0.05}))
    candidate = EvalReport(**_coverage_gate_report(0.30, evals.COVERAGE_METRIC_DEFINITION))

    failures = _eval_policy.check_policy(candidate, policy,
                                         _coverage_gate_report(0.03, "x" * 5_000_000))
    assert failures
    assert max(len(f) for f in failures) < 400, max(len(f) for f in failures)
    assert any("truncated" in f for f in failures), failures


def test_every_report_records_which_coverage_definition_produced_it():
    report = evaluate(StubRunner({"s1": {"intent": choice_answer("a")}}),
                      Dataset([Example("s1", Q, {"intent": "a"})]))
    assert report.config["coverage_metric_definition"] == evals.COVERAGE_METRIC_DEFINITION
    assert evals.COVERAGE_METRIC_DEFINITION == 2


def test_aurc_is_bit_identical_to_the_per_answer_mean_without_ties():
    """Not `approx`: the claim is bit-identity, and only an exact comparison can hold it.

    With no ties every level holds one answer, so weighting by rows and dividing once at the end
    reduces to the same pairwise sum over the same denominator. Weighting by the coverage width
    instead -- mathematically the same area -- loses the last bits on about half of all datasets,
    which would quietly move every committed number that has no tie in it.
    """
    import random

    random.seed(5)
    for _ in range(200):
        n = random.randint(1, 40)
        conf = random.sample([i / 10000 for i in range(1, 9999)], n)
        corr = [random.random() < 0.6 for _ in range(n)]
        order = np.argsort(-np.asarray(conf, dtype=float), kind="mergesort")
        y = np.asarray(corr, dtype=bool)[order].astype(float)
        per_answer = 1.0 - np.cumsum(y) / np.arange(1, n + 1)
        assert aurc(conf, corr) == float(np.mean(per_answer))


def test_a_single_cut_agrees_with_the_risk_coverage_curve():
    """The two routes to one rule must not drift apart, so a test holds them together.

    `selective_accuracy` answers for one cut (`_accepted_at_cut`, a binary search) what `_levels`
    answers for every level (a full scan, which `aurc` integrates). Deriving the single cut from
    the full scan was measured 19% to 58% slower for no change in any value, so both exist -- and
    a comment asserting they agree would be enforced by nothing. This is the enforcement: an edit
    to either side fails here.

    Exact equality, not `approx`: both routes divide one exact integer count by another, over the
    same accepted set, so any difference at all is a real divergence rather than rounding.
    """
    import math
    import random

    rng = random.Random(13)
    pool = [0.0, -0.0, 1.0, 0.5, 0.25, -1.0, 2.0,
            float("nan"), float("inf"), -float("inf"), 1.0 - 2.0 ** -52]
    for _ in range(4000):
        n = rng.randint(1, 14)
        conf = [rng.choice(pool) for _ in range(n)]
        corr = [rng.random() < 0.5 for _ in range(n)]
        accepted, hits, levels_n = evals._levels(conf, corr)
        assert levels_n == n
        for coverage in (0.01, 0.25, 0.5, 0.8, 1.0):
            k = max(1, int(math.ceil(coverage * n)))
            level = int(np.searchsorted(accepted, k, side="left"))
            # the two routes must agree on how many answers the cut accepts ...
            #
            # Sorted the way `_levels` sorts: `argsort(-c)` ranks NaN LAST, where
            # `np.sort(c)[::-1]` would rank it first and feed the lookup a different array.
            column = np.asarray(conf, dtype=float)
            sorted_desc = column[np.argsort(-column, kind="mergesort")]
            assert evals._accepted_at_cut(sorted_desc, k) == int(accepted[level]), (
                n, k, conf)
            # ... and therefore on the accuracy over them
            assert selective_accuracy(conf, corr, coverage) == float(hits[level] / accepted[level]), (
                n, coverage, conf, corr)


def test_selective_accuracy_is_bit_identical_to_the_accepted_slice_mean():
    """Not `approx`: the claim is bit-identity with what this function answered before.

    `selective_accuracy` now reads the same levels `aurc` reads, so it had to be shown that routing
    it through them does not move a value. It returns `hits / accepted` -- the exact count of
    correct answers over the exact count accepted -- because `corrects` is 0.0/1.0 and every
    partial sum is therefore an exact integer in float64.

    Recovering the same number as `1 - risk` instead, from a risk the levels had already divided,
    round-trips through a subtraction and differs in the last bits on about one value in eight. A
    gate would not notice; a user diffing two reports of the same no-tie data would, and this
    change is not allowed to move a number that no tie could move.
    """
    import random

    random.seed(11)
    for _ in range(200):
        n = random.randint(1, 40)
        conf = random.sample([i / 10000 for i in range(1, 9999)], n)   # distinct: no ties
        corr = [random.random() < 0.6 for _ in range(n)]
        order = np.argsort(-np.asarray(conf, dtype=float), kind="mergesort")
        y = np.asarray(corr, dtype=bool)[order].astype(float)
        for coverage in (0.1, 0.25, 0.5, 0.8, 1.0):
            k = max(1, int(np.ceil(coverage * n)))
            assert selective_accuracy(conf, corr, coverage) == float(np.mean(y[:k])), (
                n, coverage, conf, corr)


def test_selective_metrics_reach_the_report():
    # choice answers with a spread of confidence and correctness so the pairs exist
    ds = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"}),
                  Example("s3", Q, {"intent": "a"}), Example("s4", Q, {"intent": "a"})])
    answers = {"s1": {"intent": choice_answer("a", 0.95)}, "s2": {"intent": choice_answer("a", 0.80)},
               "s3": {"intent": choice_answer("b", 0.60)}, "s4": {"intent": choice_answer("b", 0.30)}}
    report = evaluate(StubRunner(answers), ds, evaluators=[ChoiceAccuracy()])
    for name in ("ece", "brier", "aurc", "selective_accuracy@50", "selective_accuracy@80"):
        assert name in report.overall, (name, sorted(report.overall))


def test_percentiles_use_nearest_rank():
    from laya.evals import _percentiles

    # Nearest-rank 95th percentile is the ceil(0.95 * n)-th smallest value (1-indexed).
    # `int(n * 0.95)` used to return the (0.95n + 1)-th value whenever n is a multiple of 20.
    # Expected values are written as literals (not the implementation's own formula) so a
    # formula that is wrong in both places at once still fails.
    for n, expected in ((20, 19), (40, 38), (100, 95), (30, 29), (50, 48)):
        values = list(range(1, n + 1))
        _, p95 = _percentiles(values)
        assert p95 == pytest.approx(expected), \
            "95th percentile of 1..%d must be the nearest-rank value %d, got %r" % (n, expected, p95)
    # A single value is its own percentile.
    assert _percentiles([7.0]) == (7.0, 7.0)


# --------------------------------------------------------------- dataset
def test_dataset_from_jsonl(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join([
        json.dumps({"state": "s1", "questions": Q, "expected": {"intent": "a"}, "language": "en"}),
        "# a comment line",
        json.dumps({"state": "s2", "questions": Q, "expected": {"intent": "b"}, "tags": ["t"]}),
    ]), encoding="utf-8")
    dataset = Dataset.from_jsonl(str(path))
    assert len(dataset) == 2
    assert dataset.examples[0].language == "en"
    assert dataset.examples[1].tags == ("t",)


@pytest.mark.parametrize("row, fragment", [
    ({"state": "s", "questions": Q}, "missing 'expected'"),
    ({"state": "s", "questions": Q, "expected": {"nope": "a"}}, "unknown question"),
    ({"state": "s", "questions": [], "expected": {}}, "'questions' must be an object"),
])
def test_dataset_rejects_bad_rows(row, fragment):
    with pytest.raises(EvalError) as exc:
        Example.from_dict(row)
    assert fragment in str(exc.value)


def test_dataset_rejects_malformed_json_and_empty(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(EvalError):
        Dataset.from_jsonl(str(bad))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(EvalError):
        Dataset.from_jsonl(str(empty))


# --------------------------------------------------------------- evaluate
def test_evaluate_overall_and_slices():
    dataset = Dataset([
        Example("s1", Q, {"intent": "a"}, language="en"),
        Example("s2", Q, {"intent": "b"}, language="en"),
        Example("s3", Q, {"intent": "a"}, language="de"),
    ])
    runner = StubRunner({
        "s1": {"intent": {"type": "choice", "choice": "a", "confidence": 1.0}},
        "s2": {"intent": {"type": "choice", "choice": "b", "confidence": 1.0}},
        "s3": {"intent": {"type": "choice", "choice": "b", "confidence": 1.0}},
    })
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()])
    assert report.overall["choice_accuracy"] == pytest.approx(2 / 3)
    assert report.slices["language"]["en"]["choice_accuracy"] == pytest.approx(1.0)
    assert report.slices["language"]["de"]["choice_accuracy"] == pytest.approx(0.0)
    assert report.slices["qid"]["intent"]["choice_accuracy"] == pytest.approx(2 / 3)
    assert "choice_accuracy" in report.to_markdown()


def test_model_slice_follows_the_routed_checkpoint():
    """A Router records the checkpoint it chose under `routing`, not at the top level.

    `result["model"]` is the payload's family tag -- `laya-rl-agent` -- on every Laya runner, so
    reading only that collapses `by model` into one bucket for a mixed-language run.
    """
    class RoutedRunner(StubRunner):
        def _result(self, state, model):
            # What this runner answers with. The pinned row is deliberately routed somewhere
            # else, so the case label can only come from the caller's pin.
            chosen = "english" if state == "s_en" else "multilingual"
            return {"model": "laya-rl-agent", "routing": {"model": chosen},
                    "answers": self.by_state[state]}

        def predict(self, state, questions, model=None):
            return self._result(state, model)

        def predict_batch(self, states, questions, model=None, batch_size=None):
            return [self._result(s, model) for s in states]

    dataset = Dataset([
        Example("s_en", Q, {"intent": "a"}),
        Example("s_de", Q, {"intent": "a"}),
        Example("pinned", Q, {"intent": "a"}, model="english"),
    ])
    runner = RoutedRunner({s: {"intent": choice_answer("a")} for s in ("s_en", "s_de", "pinned")})
    wanted = ["english", "multilingual", "english"]
    for kwargs in ({}, {"batch_size": 8}):
        report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], **kwargs)
        assert [c["model"] for c in report.cases] == wanted, kwargs
        assert sorted(report.slices["model"]) == ["english", "multilingual"], kwargs


def test_model_slice_falls_back_when_a_runner_reports_no_route():
    # Absent, empty, or not a usable checkpoint name -- none of these may label a case.
    UNUSABLE = [None, {}, {"model": None}, {"model": ""}, {"model": 5}, "english", ["english"]]

    class OddRunner(StubRunner):
        def __init__(self, by_state, routing):
            super().__init__(by_state)
            self.routing = routing

        def predict(self, state, questions, model=None):
            result = {"model": "laya-rl-agent", "answers": self.by_state[state]}
            if self.routing is not None:
                result["routing"] = self.routing
            return result

    answers = {"s": {"intent": choice_answer("a")}}
    dataset = Dataset([Example("s", Q, {"intent": "a"})])
    for routing in UNUSABLE:
        report = evaluate(OddRunner(answers, routing), dataset, evaluators=[ChoiceAccuracy()])
        assert sorted(report.slices["model"]) == ["laya-rl-agent"], routing
    # An Agent runner that never routes is labelled by its own payload, as before.
    report = evaluate(StubRunner({"s1": {"intent": choice_answer("a")}}),
                      Dataset([Example("s1", Q, {"intent": "a"})]), evaluators=[ChoiceAccuracy()])
    assert sorted(report.slices["model"]) == ["stub"]


def test_evaluate_batches_same_questions():
    class BatchRunner(StubRunner):
        def __init__(self, by_state):
            super().__init__(by_state)
            self.batches = []

        def predict_batch(self, states, questions, model=None, batch_size=None):
            self.batches.append(list(states))
            return [{"model": "m", "answers": self.by_state[s]} for s in states]

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    runner = BatchRunner({"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert runner.batches == [["s1", "s2"]], "identical questions share one forward pass"
    assert report.overall["choice_accuracy"] == 1.0


class RequestsRunner(StubRunner):
    """The `Router` shape: a list of per-request dicts, and no ``model=`` on the call."""

    def __init__(self, by_state):
        super().__init__(by_state)
        self.batches = []
        self.batch_sizes = []

    def predict_batch(self, requests, batch_size=None):
        self.batches.append(list(requests))
        self.batch_sizes.append(batch_size)
        return [{"model": r.get("model") or "m", "answers": self.by_state[r["state"]]}
                for r in requests]


def _labels(report):
    """The decided labels with their correctness: parity at decision level, not as floats."""
    return [(c["answer"]["choice"], c["correct"]) for c in report.cases]


def test_evaluate_drives_a_requests_shaped_batch():
    dataset = Dataset([Example("s1", Q, {"intent": "a"}, model="english"),
                       Example("s2", Q, {"intent": "a"}, model="english")])
    runner = RequestsRunner({"s1": {"intent": choice_answer("a")},
                             "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert len(runner.batches) == 1, "a request-dict batch is a forward pass the harness can run"
    expected = [{"state": "s1", "questions": Q, "model": "english"},
                {"state": "s2", "questions": Q, "model": "english"}]
    assert runner.batches[0] == expected, "each request carries its own state, questions, checkpoint"
    assert runner.batch_sizes == [8], "the requested batch size reaches the runner"
    assert report.overall["choice_accuracy"] == 1.0


def test_requests_shaped_batch_agrees_with_single_predicts():
    """`batch_size` changes how many forward passes a run makes, never what it scores."""
    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"}),
                       Example("s3", Q, {"intent": "b"})])
    answers = {"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("b")},
               "s3": {"intent": choice_answer("b")}}
    batched = evaluate(RequestsRunner(answers), dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    single = evaluate(RequestsRunner(answers), dataset, evaluators=[ChoiceAccuracy()])
    assert _labels(batched) == _labels(single) == [("a", True), ("b", False), ("b", True)]
    assert len(batched.cases) == len(single.cases) == 3


def test_evaluate_scores_a_batch_shape_it_cannot_call():
    """A `predict_batch` in neither documented shape must not fail the run row by row."""
    class UntypedBatch(StubRunner):
        def predict_batch(self, states, questions):
            raise AssertionError("the harness may not call this: no model=, no requests")

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    runner = UntypedBatch({"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8, on_error="skip")
    assert report.overall["choice_accuracy"] == 1.0, "scored one predict at a time"
    assert not report.config.get("errored"), "a batch entry point in an unknown shape is not an error"


def test_evaluate_scores_a_runner_with_no_batch_entry_point():
    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    runner = StubRunner({"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert report.overall["choice_accuracy"] == 1.0


def test_evaluate_batches_a_pass_through_wrapper_positionally():
    """A wrapper that forwards `*args, **kwargs` takes the positional call, whatever it names."""
    class PassThrough(StubRunner):
        def __init__(self, by_state):
            super().__init__(by_state)
            self.calls = []

        def predict_batch(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return [{"model": "m", "answers": self.by_state[s]} for s in args[0]]

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    runner = PassThrough({"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == (["s1", "s2"], Q)
    assert runner.calls[0][1] == {"model": None, "batch_size": 8}
    assert report.overall["choice_accuracy"] == 1.0


# ------------------------------------------------------- batch grouping (#294 knob)
#
# `Agent.predict_batch` and `Router.predict_batch` group similarly sized states inside a bounded
# `batch_size` since #294, and `research/` reports 2.15x over 10,000 tickets from it with no answer
# changing. `laya-evals run --batch-size` could bound a pass but not group one, so the harness that
# exists to measure cost could not ask for the cheaper shape of the same pass.

UNSENT = object()   # a value no caller can send, so a default distinguishes "absent" from False


class GroupingRunner(RequestsRunner):
    """A `Router`-shaped runner that has the knob and records exactly what the call carried."""

    def __init__(self, by_state):
        super().__init__(by_state)
        self.shapes = []

    def predict_batch(self, requests, batch_size=None, sort_by_length=UNSENT):
        self.shapes.append({"batch_size": batch_size, "sort_by_length": sort_by_length})
        return [{"model": r.get("model") or "m", "answers": self.by_state[r["state"]]}
                for r in requests]


def _three():
    return Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"}),
                    Example("s3", Q, {"intent": "b"})])


ANSWERS3 = {"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")},
            "s3": {"intent": choice_answer("b")}}


def test_grouping_reaches_a_requests_shaped_runner_when_asked():
    runner = GroupingRunner(ANSWERS3)
    report = evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], batch_size=2,
                      sort_by_length=True)
    assert runner.shapes == [{"batch_size": 2, "sort_by_length": True}], runner.shapes
    assert report.overall["choice_accuracy"] == 1.0


def test_grouping_is_not_invented_for_a_run_that_did_not_ask():
    """The control is dropped when unset, so an older runner sees the call it always saw."""
    runner = GroupingRunner(ANSWERS3)
    evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], batch_size=2)
    assert runner.shapes == [{"batch_size": 2, "sort_by_length": UNSENT}], runner.shapes


def test_grouping_reaches_a_states_shaped_runner_too():
    """Both batch call shapes the harness promises carry the knob; neither drops it silently."""
    class PassThrough(StubRunner):
        def __init__(self, by_state):
            super().__init__(by_state)
            self.calls = []

        def predict_batch(self, *args, **kwargs):
            self.calls.append(kwargs)
            return [{"model": "m", "answers": self.by_state[s]} for s in args[0]]

    runner = PassThrough(ANSWERS3)
    evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], batch_size=2, sort_by_length=True)
    assert runner.calls == [{"model": None, "batch_size": 2, "sort_by_length": True}], runner.calls


def test_grouping_changes_the_call_not_the_score():
    asked = GroupingRunner(ANSWERS3)
    sorted_report = evaluate(asked, _three(), evaluators=[ChoiceAccuracy()], batch_size=2,
                             sort_by_length=True)
    plain = GroupingRunner(ANSWERS3)
    unsorted_report = evaluate(plain, _three(), evaluators=[ChoiceAccuracy()], batch_size=2)
    assert _labels(sorted_report) == _labels(unsorted_report)
    assert asked.shapes != plain.shapes, "the knob really reached the runner"


def test_grouping_without_a_batch_to_reorder_is_recorded_as_not_sent():
    """A single-example chunk has nothing to group, and the report says so rather than agreeing."""
    runner = GroupingRunner(ANSWERS3)
    report = evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], sort_by_length=True)
    assert runner.shapes == [], "no batch call was made at all"
    assert report.config["timing"]["sort_by_length"] is True
    assert report.config["timing"]["sort_by_length_sent"] is False


def test_a_runner_without_the_knob_is_scored_unsorted():
    """`sort_by_length` is an optimisation: it must not end a run over a runner that predates it."""
    runner = RequestsRunner(ANSWERS3)          # `predict_batch(requests, batch_size=None)`
    report = evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], batch_size=2,
                      sort_by_length=True)
    assert runner.batch_sizes == [2], "the run still batched"
    assert report.overall["choice_accuracy"] == 1.0
    assert not report.config.get("errored")
    assert report.config["timing"]["sort_by_length_sent"] is False


def test_a_pass_through_wrapper_is_given_the_knob():
    """A runner that forwards `**kwargs` is a real runner behind it, so the grouping goes through."""
    class PassThrough(StubRunner):
        def predict_batch(self, *args, **kwargs):
            return [{"model": "m", "answers": self.by_state[s]} for s in args[0]]

    report = evaluate(PassThrough(ANSWERS3), _three(), evaluators=[ChoiceAccuracy()],
                      batch_size=2, sort_by_length=True)
    assert report.config["timing"]["sort_by_length_sent"] is True


# ------------------------------------------------------- abstention gate (#361 knob)
#
# `Router.predict` / `Router.predict_batch` and `ONNXAgent.predict` gate on `answer_confidence`
# and mark answers below `min_confidence` with `low_confidence: True` and `abstention:
# "abstained"` (#361). The gate is a *reporting* control: `apply_confidence_gate` leaves
# `answer["choice"] / ["noul"] / ["score"]` as the raw argmax, so every metric in `_aggregate`
# reads the same numbers at every threshold. What the harness forwards is the *claim* the
# report makes: `report.config["timing"]["min_confidence"]` names the threshold and
# `["min_confidence_sent"]` asserts the call carried it. A `laya-evals run` that could not pass
# the kwarg through had no way to publish that claim on a real Router -- the exact class of
# thing this harness exists to be.

UNSET = object()   # no caller sends it, so `mc is UNSET` distinguishes "absent" from 0.0 or None


class GatedRunner(RequestsRunner):
    """A `Router`-shaped runner that has the gate and records exactly what each call carried."""

    def __init__(self, by_state):
        super().__init__(by_state)
        self.thresholds = []

    def predict(self, state, questions, model=None, min_confidence=UNSET):
        self.thresholds.append({"path": "predict", "min_confidence": min_confidence})
        return {"model": model or "m", "answers": self.by_state[state]}

    def predict_batch(self, requests, batch_size=None, min_confidence=UNSET):
        self.thresholds.append({"path": "predict_batch", "min_confidence": min_confidence})
        return [{"model": r.get("model") or "m", "answers": self.by_state[r["state"]]}
                for r in requests]


class GatedStatesRunner(StubRunner):
    """The positional batch shape, same gate + same recording."""

    def __init__(self, by_state):
        super().__init__(by_state)
        self.thresholds = []

    def predict(self, state, questions, model=None, min_confidence=UNSET):
        self.thresholds.append({"path": "predict", "min_confidence": min_confidence})
        return {"model": model or "m", "answers": self.by_state[state]}

    def predict_batch(self, states, questions, model=None, batch_size=None,
                      min_confidence=UNSET):
        self.thresholds.append({"path": "predict_batch", "min_confidence": min_confidence})
        return [{"model": model or "m", "answers": self.by_state[s]} for s in states]


def test_min_confidence_reaches_a_requests_shaped_runner_when_asked():
    runner = GatedRunner(ANSWERS3)
    report = evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], batch_size=2,
                      min_confidence=0.7)
    assert runner.thresholds[0] == {"path": "predict_batch", "min_confidence": 0.7}
    assert report.config["timing"]["min_confidence"] == 0.7
    assert report.config["timing"]["min_confidence_sent"] is True


def test_min_confidence_reaches_a_states_shaped_runner_too():
    runner = GatedStatesRunner(ANSWERS3)
    report = evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], batch_size=2,
                      min_confidence=0.7)
    # The first two rows share a batch; the third stands alone because batch_size=2.
    assert runner.thresholds[0] == {"path": "predict_batch", "min_confidence": 0.7}
    assert runner.thresholds[1] == {"path": "predict", "min_confidence": 0.7}
    assert report.config["timing"]["min_confidence_sent"] is True


def test_min_confidence_reaches_the_single_predict_fallback():
    """A run with no `batch_size` still has to see the gate; the abstention path is per-decision."""
    runner = GatedRunner(ANSWERS3)
    report = evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], min_confidence=0.7)
    assert all(t["path"] == "predict" and t["min_confidence"] == 0.7 for t in runner.thresholds)
    assert report.config["timing"]["min_confidence_sent"] is True


def test_min_confidence_is_not_invented_for_a_run_that_did_not_ask():
    """The kwarg is dropped when unset, so an older runner sees the call it always saw."""
    runner = GatedRunner(ANSWERS3)
    evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], batch_size=2)
    # chunk 1 shares a batch (rows s1, s2), row s3 falls to `predict`. Neither sees the threshold.
    assert runner.thresholds == [{"path": "predict_batch", "min_confidence": UNSET},
                                  {"path": "predict", "min_confidence": UNSET}]


def test_min_confidence_zero_is_still_a_threshold():
    """`if min_confidence:` would drop 0.0; a gate at zero is a legal, meaningful ask."""
    runner = GatedRunner(ANSWERS3)
    evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], batch_size=2, min_confidence=0.0)
    assert runner.thresholds[0] == {"path": "predict_batch", "min_confidence": 0.0}


def test_a_runner_without_the_gate_is_refused_not_silently_scored():
    """Silently dropping the kwarg would let `report.config["timing"]["min_confidence"]` name a
    threshold the harness never applied -- a lying report even though the metric numbers stay
    identical. `RequestsRunner`'s `predict_batch(requests, batch_size=None)` is exactly what a
    pre-#361 runner looks like, so a run that asks for a threshold on it must fail loudly."""
    with pytest.raises(EvalError) as exc:
        evaluate(RequestsRunner(ANSWERS3), _three(), evaluators=[ChoiceAccuracy()],
                 batch_size=2, min_confidence=0.7)
    assert "min_confidence" in str(exc.value)
    assert "does not accept the abstention threshold" in str(exc.value)


def test_the_single_predict_path_refuses_too_when_only_predict_lacks_the_gate():
    """Batching on a gate-aware `predict_batch` does not excuse a gate-blind `predict`: a chunk
    of one still goes through that entry point, and would be scored without the threshold."""
    class OnlyBatchGated(GatedRunner):
        def predict(self, state, questions, model=None):        # predates #361
            return {"model": model or "m", "answers": self.by_state[state]}

    # batch_size=2 on a three-row dataset leaves the third row to `predict`; the guard sees that
    # before issuing anything.
    with pytest.raises(EvalError) as exc:
        evaluate(OnlyBatchGated(ANSWERS3), _three(), evaluators=[ChoiceAccuracy()],
                 batch_size=2, min_confidence=0.7)
    assert "predict" in str(exc.value)


def test_min_confidence_validated_by_core():
    """The accepted range is core's validator, so the harness cannot drift from the gate itself."""
    for bad in (1.5, -0.1, "high", True, float("nan")):
        with pytest.raises(EvalError) as exc:
            evaluate(GatedRunner(ANSWERS3), _three(), evaluators=[ChoiceAccuracy()],
                     min_confidence=bad)
        assert "min_confidence must be a float in [0.0, 1.0]" in str(exc.value)


def test_min_confidence_changes_the_call_not_the_score_of_a_stub():
    """The stub ignores the threshold, so the answers match. This is the parity check on the
    control-flow side -- the answer-shape witness is `flag_low_confidence`'s own suite (#361)."""
    asked = GatedRunner(ANSWERS3)
    gated = evaluate(asked, _three(), evaluators=[ChoiceAccuracy()], batch_size=2,
                     min_confidence=0.7)
    plain = GatedRunner(ANSWERS3)
    ungated = evaluate(plain, _three(), evaluators=[ChoiceAccuracy()], batch_size=2)
    assert _labels(gated) == _labels(ungated)
    # The calls really differed: one carried the threshold, one did not.
    assert asked.thresholds != plain.thresholds


def test_min_confidence_and_sort_by_length_are_independent_controls():
    """A run can ask for both: the sort is dropped for a runner that predates #294 while the
    gate is honoured for the same runner's #361 support, and vice versa."""
    class OnlyGated(GatedRunner):
        def predict_batch(self, requests, batch_size=None, min_confidence=UNSET):
            # No `sort_by_length` in the signature, but the threshold is real.
            self.thresholds.append({"path": "predict_batch", "min_confidence": min_confidence})
            return [{"model": r.get("model") or "m", "answers": self.by_state[r["state"]]}
                    for r in requests]

    runner = OnlyGated(ANSWERS3)
    report = evaluate(runner, _three(), evaluators=[ChoiceAccuracy()], batch_size=2,
                      sort_by_length=True, min_confidence=0.7)
    assert report.config["timing"]["sort_by_length_sent"] is False
    assert report.config["timing"]["min_confidence_sent"] is True


def test_cli_min_confidence_forwards_through_evaluate(monkeypatch, tmp_path):
    """The flag has to actually reach `evaluate`, not sit in the parser."""
    from laya import evals_cli

    seen = {}
    real_evaluate = evals_cli.evals.evaluate

    def spy(runner, dataset, **kwargs):
        seen.update(kwargs)
        return real_evaluate(runner, dataset, **kwargs)

    monkeypatch.setattr(evals_cli.evals, "evaluate", spy)
    _fake_router(monkeypatch, {})
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["run", dataset, "--min-confidence", "0.7"]) == 0
    assert seen["min_confidence"] == 0.7


def test_cli_min_confidence_out_of_range_is_a_usage_error(monkeypatch, tmp_path, capsys):
    """A mistyped `--min-confidence 1.5` must exit 2 with core's message, not traceback."""
    from laya import evals_cli

    _fake_router(monkeypatch, {})
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["run", dataset, "--min-confidence", "1.5"]) == 2
    err = capsys.readouterr().err
    assert "min_confidence must be a float" in err
    assert "Traceback" not in err


def test_cli_min_confidence_zero_is_accepted(monkeypatch, tmp_path):
    """A gate at zero is a legal ask; `if args.min_confidence` would drop it before the call."""
    from laya import evals_cli

    seen = {}
    real_evaluate = evals_cli.evals.evaluate

    def spy(runner, dataset, **kwargs):
        seen.update(kwargs)
        return real_evaluate(runner, dataset, **kwargs)

    monkeypatch.setattr(evals_cli.evals, "evaluate", spy)
    _fake_router(monkeypatch, {})
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["run", dataset, "--min-confidence", "0.0"]) == 0
    assert seen["min_confidence"] == 0.0


# --------------------------------------------------------------- timing (#585)
FORWARD_MS = 100.0
SHARED = 0.6                      # a batch of n costs SHARED * n * FORWARD_MS, as one call
QWIDE = {"intent": {"type": "choice", "instructions": "?",
                    "criteria": {"a": "x", "b": "y", "c": "z"}}}


class Clock:
    """A timer that advances only when the runner predicts, so every figure below is exact."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now / 1000.0


class TimedRunner(StubRunner):
    """Answers from the state alone, so grouping can never change a decision -- only the clock."""

    def __init__(self, by_state):
        super().__init__(by_state)
        self.clock = Clock()
        self.chunks = []
        self.singles = []

    def predict(self, state, questions, model=None):
        self.clock.now += FORWARD_MS
        self.singles.append(state)
        return StubRunner.predict(self, state, questions, model)

    def predict_batch(self, states, questions, model=None, batch_size=None):
        self.clock.now += len(states) * FORWARD_MS * SHARED
        self.chunks.append(len(states))
        return [StubRunner.predict(self, s, questions, model) for s in states]


def _timed_pair(monkeypatch, batch_size):
    """Score three shareable rows and one that cannot join them, on a fake clock.

    Returns (report, runner): the runner's own `chunks` is the witness that the grouping the test
    asserts is the grouping the harness really issued.
    """
    import laya.evals as evals_module

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"}),
                       Example("s3", Q, {"intent": "a"}), Example("s4", QWIDE, {"intent": "a"})])
    answers = {state: {"intent": choice_answer("a")} for state in ("s1", "s2", "s3", "s4")}
    runner = TimedRunner(answers)
    monkeypatch.setattr(evals_module, "time", runner.clock)
    return evals_module.evaluate(runner, dataset, evaluators=[ChoiceAccuracy()],
                                 batch_size=batch_size), runner


def test_batched_latency_is_what_a_request_waited(monkeypatch):
    solo, solo_runner = _timed_pair(monkeypatch, None)
    batched, runner = _timed_pair(monkeypatch, 8)

    assert _labels(batched) == _labels(solo), "identical decisions; only the timing moved"
    assert (solo_runner.chunks, solo_runner.singles) == ([], ["s1", "s2", "s3", "s4"])
    assert (runner.chunks, runner.singles) == ([3], ["s4"]), \
        "one shared call of three, and the row whose questions match nothing left alone"

    assert solo.overall["latency_p50_ms"] == pytest.approx(FORWARD_MS)
    # The chunk of three returns all three requests together at +180 ms, so that is their latency.
    assert batched.overall["latency_p50_ms"] == pytest.approx(3 * FORWARD_MS * SHARED)
    assert batched.overall["latency_p50_ms"] > solo.overall["latency_p50_ms"], \
        "batching trades request latency for throughput; the report has to say so"


def test_the_throughput_share_keeps_its_own_metric(monkeypatch):
    solo, _ = _timed_pair(monkeypatch, None)
    batched, _ = _timed_pair(monkeypatch, 8)

    # Unbatched, the two quantities are the same number, so every report without the flag is
    # unchanged by this fix.
    assert solo.overall["cost_per_decision_p50_ms"] == pytest.approx(solo.overall["latency_p50_ms"])
    assert solo.overall["cost_per_decision_p95_ms"] == pytest.approx(solo.overall["latency_p95_ms"])
    # Batched, the share is the figure the old `latency_p50_ms` published: 180 ms over three rows.
    assert batched.overall["cost_per_decision_p50_ms"] == pytest.approx(FORWARD_MS * SHARED)
    assert batched.overall["cost_per_decision_p95_ms"] == pytest.approx(FORWARD_MS)


def test_a_latency_gate_cannot_pass_a_run_where_nothing_finished_in_time(monkeypatch):
    from laya import evals_cli

    solo, _ = _timed_pair(monkeypatch, None)
    batched, _ = _timed_pair(monkeypatch, 8)
    # 80 ms is below every wait in either run (100 ms alone, 180 ms shared). The old report passed
    # the batched run at 60 ms, which was 1/3 of a call no request could see the end of.
    for name, report in (("unbatched", solo), ("--batch-size 8", batched)):
        assert evals_cli._check_thresholds(report.overall, {}, {"latency_p50_ms": 80.0}), \
            "%s abstains: no request in it was served inside the limit" % name
    for name, report in (("unbatched", solo), ("--batch-size 8", batched)):
        assert not evals_cli._check_thresholds(report.overall, {}, {"latency_p50_ms": 200.0}), \
            "%s passes a bound every request beat" % name
    # The throughput win is still gateable, under the name that measures it.
    assert not evals_cli._check_thresholds(batched.overall, {}, {"cost_per_decision_p50_ms": 80.0})
    assert evals_cli._check_thresholds(solo.overall, {}, {"cost_per_decision_p50_ms": 80.0})


def test_timing_facts_record_what_the_harness_did(monkeypatch):
    solo, _ = _timed_pair(monkeypatch, None)
    batched, _ = _timed_pair(monkeypatch, 8)

    assert solo.config["timing"]["batch_size"] is None
    assert solo.config["timing"]["batch_form"] is None
    assert solo.config["timing"]["rows_grouped"] == 0
    assert solo.config["timing"]["rows_alone"] == 4
    assert solo.config["timing"]["max_chunk"] == 1
    assert batched.config["timing"]["batch_size"] == 8
    assert batched.config["timing"]["batch_form"] == "states"
    assert batched.config["timing"]["chunks"] == 2
    assert batched.config["timing"]["rows_grouped"] == 3
    assert batched.config["timing"]["rows_alone"] == 1
    assert batched.config["timing"]["max_chunk"] == 3
    assert batched.config["timing"]["latency_metric"] != batched.config["timing"]["cost_metric"]

    # `--json` is the artifact a reviewer reads, so the two runs must differ there, not only in
    # the flag they were asked with.
    assert json.dumps(batched.to_json()) != json.dumps(solo.to_json())


def test_a_requested_batch_size_is_not_a_batched_run(monkeypatch):
    """The report records the grouping it achieved, so a runner that cannot batch cannot hide it."""
    import laya.evals as evals_module

    class Untimed(StubRunner):
        pass

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    monkeypatch.setattr(evals_module, "time", Clock())
    report = evals_module.evaluate(Untimed({"s1": {"intent": choice_answer("a")},
                                            "s2": {"intent": choice_answer("a")}}),
                                   dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert report.config["timing"]["batch_form"] is None
    assert report.config["timing"]["rows_grouped"] == 0, "asked to batch, and the report says it did not"
    assert report.config["timing"]["rows_alone"] == 2


def test_a_batched_call_that_raises_is_still_recorded_as_batched(monkeypatch):
    """`config["timing"]` counts the calls the harness issued, not only the calls that returned.

    A raised chunk used to contribute to none of the shape counters, which made a run that issued
    one shared forward and lost it indistinguishable from a run that never shared a call at all --
    the one mode where `docs/evals.md` says the block has to be right.
    """
    import laya.evals as evals_module

    class RaisingBatch(TimedRunner):
        def predict_batch(self, states, questions, model=None, batch_size=None):
            self.chunks.append(len(states))
            raise RuntimeError("the shared forward failed")

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"}),
                       Example("s3", Q, {"intent": "a"}), Example("s4", QWIDE, {"intent": "a"})])
    answers = {state: {"intent": choice_answer("a")} for state in ("s1", "s2", "s3", "s4")}
    runner = RaisingBatch(answers)
    monkeypatch.setattr(evals_module, "time", runner.clock)
    report = evals_module.evaluate(runner, dataset, evaluators=[ChoiceAccuracy()],
                                   batch_size=8, on_error="skip")

    # Why the runner records it itself: the counters under test are the harness's own account, so
    # an independent witness that a three-row forward really went out is what makes the assertion
    # mean something.
    assert runner.chunks == [3], "a shared forward was issued for the three matching rows"
    timing = report.config["timing"]
    assert (timing["chunks"], timing["rows_grouped"], timing["rows_alone"]) == (2, 3, 1)
    assert timing["max_chunk"] == 3
    assert len(report.config["errored"]) == 3

    # Why the metric lists stay below the `continue`: a call that returned nothing has no request
    # latency to publish, so counting the attempt must not invent one. Only s4's own `predict`
    # reaches the percentiles here.
    assert report.overall["latency_p50_ms"] == pytest.approx(FORWARD_MS)
    assert [case["correct"] for case in report.cases] == [True]

    # The ambiguity this closes: a runner with no `predict_batch` gave the identical three
    # counters, so the artifact could not tell "nothing was batched" from "the batch raised".
    plain = evaluate(StubRunner(answers), dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    for key in ("chunks", "rows_grouped", "max_chunk"):
        assert timing[key] != plain.config["timing"][key], key


def test_compare_leaves_the_cost_metrics_alone():
    report = EvalReport(overall={"choice_accuracy": 0.8, "latency_p50_ms": 12.0,
                                 "cost_per_decision_p50_ms": 4.0})
    baseline = {"overall": {"choice_accuracy": 0.8, "latency_p50_ms": 5.0,
                            "cost_per_decision_p50_ms": 2.0}}
    ok, deltas = report.compare(baseline)
    assert ok and not {"latency_p50_ms", "cost_per_decision_p50_ms"} & set(deltas)
    bad, deltas = report.compare(baseline, {"cost_per_decision_p50_ms": 0.5})
    assert not bad, "a 2 ms drift is outside the 0.5 ms tolerance it named"
    assert "latency_p50_ms" not in deltas, "the metric nobody named stays out of the comparison"
    assert deltas["cost_per_decision_p50_ms"]["diff"] == pytest.approx(2.0)


def test_docs_and_the_harness_name_the_same_timing_metrics(monkeypatch):
    """`docs/evals.md` must list exactly the metrics `evaluate` publishes, with their quantities."""
    import pathlib
    import re

    page = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "evals.md").read_text(
        encoding="utf-8")
    batched, _ = _timed_pair(monkeypatch, 8)
    published = {metric for metric in batched.overall if metric.endswith("_ms")}

    rows = {}
    for line in page.splitlines():
        if line.startswith("| `"):
            for metric in re.findall(r"`([a-z_0-9]+_ms)`", line):
                rows[metric] = line
    assert set(rows) == published, "no metric documented that is not emitted, and none missed"
    assert "per request" in rows["latency_p50_ms"]
    assert "waited" in rows["latency_p50_ms"], "the row has to say a request waits the whole call"
    assert "divided by the rows it carried" in rows["cost_per_decision_p50_ms"]
    assert "## Batching and timing" in page, "the trade-off the two numbers encode is written down"
    assert "`config.timing`" in page, "the report's own run facts are documented"


def test_evaluate_skips_errors_when_asked():
    class Boom(StubRunner):
        def predict(self, state, questions, model=None):
            if state == "bad":
                raise RuntimeError("no model")
            return super().predict(state, questions, model)

    dataset = Dataset([Example("ok", Q, {"intent": "a"}), Example("bad", Q, {"intent": "a"})])
    runner = Boom({"ok": {"intent": choice_answer("a")}})
    with pytest.raises(RuntimeError):
        evaluate(runner, dataset, on_error="fail")
    report = evaluate(runner, dataset, on_error="skip")
    assert len(report.cases) == 1
    assert report.config["errored"][0]["error"].startswith("RuntimeError")


# --------------------------------------------------------------- compare
def test_compare_and_assert_regression():
    report = EvalReport(overall={"choice_accuracy": 0.8, "ece": 0.10})
    baseline = {"overall": {"choice_accuracy": 0.79, "ece": 0.08}}
    ok, deltas = report.compare(baseline, {"choice_accuracy": 0.02, "ece": 0.03})
    assert ok and deltas["choice_accuracy"]["diff"] == pytest.approx(0.01)
    bad, bad_deltas = report.compare(baseline, {"ece": 0.01})
    assert not bad and bad_deltas["ece"]["diff"] == pytest.approx(0.02)
    with pytest.raises(AssertionError):
        assert_regression(report, baseline, {"ece": 0.01})


def test_compare_fails_when_a_baseline_metric_is_missing():
    # Every example errors under on_error="skip", so `overall` is empty. The gate used to skip
    # each baseline metric absent from the report and pass with nothing compared.
    class Broken:
        def predict(self, state, questions, model=None):
            raise RuntimeError("checkpoint failed to load")

    report = evaluate(Broken(), Dataset([Example("s", Q, {"intent": "a"})] * 3), on_error="skip")
    assert "choice_accuracy" not in report.overall
    baseline = {"overall": {"choice_accuracy": 0.9, "ece": 0.05, "latency_p50_ms": 3.0}}
    ok, deltas = report.compare(baseline, {"choice_accuracy": 1.0})
    assert not ok
    assert deltas["choice_accuracy"]["missing"] and deltas["ece"]["missing"]
    assert "latency_p50_ms" not in deltas, "latency stays informational without a tolerance"
    with pytest.raises(AssertionError, match="choice_accuracy"):
        assert_regression(report, baseline)

    # A partial loss fails too, and names only the metric that went missing.
    partial = EvalReport(overall={"choice_accuracy": 0.9})
    ok, deltas = partial.compare({"overall": {"choice_accuracy": 0.9, "noul_accuracy": 0.8}})
    assert not ok and set(deltas) == {"choice_accuracy", "noul_accuracy"}
    assert "missing" not in deltas["choice_accuracy"] and deltas["noul_accuracy"]["missing"]


def test_a_nan_metric_fails_every_gate():
    """Every comparison with NaN is False, so a NaN metric used to pass --min, --max and the
    baseline comparison alike."""
    from laya import evals_cli

    nan = float("nan")
    report = EvalReport(overall={"score_mae": nan, "mean_confidence": nan})
    failures = evals_cli._check_thresholds(report.overall, {"mean_confidence": 0.9}, {"score_mae": 0.1})
    assert len(failures) == 2 and all("NaN" in failure for failure in failures)

    ok, deltas = report.compare({"overall": {"score_mae": 0.05, "mean_confidence": 0.95}},
                                {"score_mae": 0.1, "mean_confidence": 0.1})
    assert not ok and set(deltas) == {"score_mae", "mean_confidence"}

    # A NaN in the baseline, or as the tolerance, is the same case from the other side.
    healthy = EvalReport(overall={"choice_accuracy": 0.9})
    assert not healthy.compare({"overall": {"choice_accuracy": nan}}, {"choice_accuracy": 1.0})[0]
    assert not healthy.compare({"overall": {"choice_accuracy": 0.9}}, {"choice_accuracy": nan})[0]
    assert healthy.compare({"overall": {"choice_accuracy": 0.9}})[0]


def test_cli_rejects_a_nan_limit_or_tolerance(capsys):
    from laya import evals_cli

    with pytest.raises(EvalError, match="not a number"):
        evals_cli._parse_pairs(["choice_accuracy=nan"])

    # The two dedicated flags bypass `_parse_pairs`; a NaN there disabled the gate the same way.
    for flag in ("--min-accuracy", "--max-ece"):
        with pytest.raises(SystemExit) as exc:
            evals_cli.main(["run", "data.jsonl", flag, "nan"])
        assert exc.value.code == 2
        assert "not a number" in capsys.readouterr().err
    args = evals_cli._build_parser().parse_args(["run", "data.jsonl", "--min-accuracy", "0.8",
                                                 "--max-ece", "0.1"])
    assert (args.min_accuracy, args.max_ece) == (0.8, 0.1)


def test_default_evaluators_cover_the_three_types():
    names = {e.name for e in default_evaluators()}
    assert {"choice_accuracy", "noul_accuracy", "score_mae", "mean_confidence"} <= names


# ------------------------------------------------------- run identity / comparability
def _identified(**config):
    return EvalReport(config={"schema": evals.REPORT_SCHEMA, **config},
                      overall={"choice_accuracy": 0.8})


def test_evaluate_records_what_it_measured():
    """`config` is the artifact a reviewer reads, so it has to say which run produced it.

    `config.dataset` is the path as typed. Two datasets share a path across a rebase, a CI
    cache or a colleague's checkout, and `docs/evals.md` tells the reviewer to commit the
    dataset and the baseline together and read the result as a diff -- which is only reviewable
    if the report says which bytes were scored.
    """
    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "b"})])
    report = evaluate(StubRunner({"s1": {"intent": choice_answer("a")},
                                  "s2": {"intent": choice_answer("a")}}), dataset)
    config = report.config
    assert config["schema"] == evals.REPORT_SCHEMA
    assert config["laya_version"] == laya.__version__
    assert len(config["questions_sha256"]) == 64, "the question schema is part of the identity"

    # Determinism: the identity must carry nothing time-bearing, or a re-run of the same dataset
    # stops producing the same report and the artifact a reviewer diffs becomes noise. The
    # `*_ms` metrics are wall clock and always differ; only the config block is asserted here.
    again = evaluate(StubRunner({"s1": {"intent": choice_answer("a")},
                                 "s2": {"intent": choice_answer("a")}}), dataset)
    assert {k: v for k, v in config.items() if k != "timing"} == \
           {k: v for k, v in again.config.items() if k != "timing"}


def test_questions_fingerprint_follows_the_question_not_the_row():
    """The fingerprint covers the question schema, so it is a function of what was asked.

    A run that scores a different dataset must not collide with a baseline, and a run that
    scores the same questions on more rows must still be comparable to it. Renaming a `choice`
    label is a different question, not a different row: `research/eval/metamorphic.py` exists
    because option order and label rename flip answers.

    The scope of the last claim is the fingerprint, not the gate. `dataset_sha256` is compared
    too, and adding a row changes the file's bytes, so through `laya-evals run` a longer dataset
    is correctly refused. The fingerprint being row-independent is what lets a programmatic
    caller see that the *questions* did not change.
    """
    base = Dataset([Example("s1", Q, {"intent": "a"})])
    more_rows = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    assert evals.questions_fingerprint(base) == evals.questions_fingerprint(more_rows)

    renamed = Dataset([Example("s1", {"intent": {"type": "choice", "instructions": "?",
                                                 "criteria": {"a": "x", "c": "y"}}},
                               {"intent": "a"})])
    assert evals.questions_fingerprint(renamed) != evals.questions_fingerprint(base)

    retyped = Dataset([Example("s1", {"flag": {"type": "noul", "instructions": "?"}},
                               {"flag": True})])
    assert evals.questions_fingerprint(retyped) != evals.questions_fingerprint(base)


def test_questions_fingerprint_covers_the_instructions():
    """`instructions` is the prompt. Excluding it made the fingerprint say "same question" for
    two questions the model answers differently.

    `laya/common.py:158-159` renders `"%s question: %s" % (q["type"], instructions)` into the
    tokenized head, `laya/agent.py:633-634` makes the field mandatory ("add the text the model
    should answer"), and Laya's own identity key for sharing forward passes
    (`Router._question_schema`, `laya/router.py:141`) hashes the whole questions dict,
    instructions included -- as does this module's own batch grouping at `laya/evals.py:529`.
    `tests/test_router_batch.py:543` pins that rewording alone moves a row to its own batch group.
    Leaving it out made the one field this contract exists to protect the one field it ignored.
    """
    judged = Dataset([Example("s1", {"verdict": {"type": "choice", "instructions": "Judge whether a refund is justified",
                                                "criteria": {"yes": "approved", "no": "denied"}}},
                               {"verdict": "yes"})])
    conservative = Dataset([Example("s1", {"verdict": {"type": "choice", "instructions": "Be conservative and only approve explicit refund requests",
                                                      "criteria": {"yes": "approved", "no": "denied"}}},
                                 {"verdict": "yes"})])
    assert evals.questions_fingerprint(judged) != evals.questions_fingerprint(conservative)

    # Same wording, different question id, is still a different question.
    renamed_id = Dataset([Example("s1", {"decision": judged.examples[0].questions["verdict"]},
                                  {"decision": "yes"})])
    assert evals.questions_fingerprint(renamed_id) != evals.questions_fingerprint(judged)

    # Whitespace is not decoration: it is tokenized. `common.py:158` strips only the tokenizer's
    # mask token, and this module has no tokenizer to know what that is, so nothing else is
    # normalized away.
    spaced = Dataset([Example("s1", {"verdict": dict(judged.examples[0].questions["verdict"],
                                                     instructions="Judge whether a refund is justified ")},
                              {"verdict": "yes"})])
    assert evals.questions_fingerprint(spaced) != evals.questions_fingerprint(judged)


def test_questions_fingerprint_normalizes_instructions_the_way_the_engine_does():
    """Two questions that render the same text are the same question, whatever their JSON shape.

    `Agent._to_internal` (`laya/agent.py:736-748`) turns a non-string `instructions` into
    `json.dumps(ins, ensure_ascii=False)` before tokenizing, and `tests/test_criteria.py:229-251`
    pins why: the default `ensure_ascii=True` escaped non-ASCII to literal `\\uXXXX` and a German
    question answered noul=0.1652 as a dict against 0.2650 as the identical plain string. The
    fingerprint has to mirror that step, or it hashes the input's JSON shape instead of the text
    the model reads.
    """
    criteria = {"yes": "approved", "no": "denied"}
    as_object = Dataset([Example("s1", {"verdict": {"type": "choice",
                                                    "instructions": {"task": "Bittet der Kunde um eine Rückerstattung?"},
                                                    "criteria": criteria}},
                               {"verdict": "yes"})])
    as_text = Dataset([Example("s1", {"verdict": {"type": "choice",
                                                  "instructions": '{"task": "Bittet der Kunde um eine Rückerstattung?"}',
                                                  "criteria": criteria}},
                               {"verdict": "yes"})])
    assert evals.questions_fingerprint(as_object) == evals.questions_fingerprint(as_text)


def test_a_reworded_question_cannot_pass_the_gate():
    """The end-to-end contract: a different question is not a baseline drift, it is a new run.

    Reproduces the hole the fingerprint's exclusion opened. `evaluate(runner, Dataset(...))` has
    no dataset file to hash, so `questions_sha256` is the only identity a programmatic run has --
    and it used to be blind to the one field that decides the answer. The metric gate is a
    separate, tolerance-dependent safety net; this is the check that does not depend on having
    guessed the right tolerance.
    """
    criteria = {"yes": "approved", "no": "denied"}

    def twenty(instr):
        q = {"verdict": {"type": "choice", "instructions": instr, "criteria": criteria}}
        return Dataset([Example("s%d" % i, q, {"verdict": "yes"}) for i in range(20)])

    class OneRowFlips:
        """The answer depends on the instruction on exactly one state, so the metric delta can
        sit inside a realistic tolerance while the question asked is a different one."""
        def predict(self, state, questions, model=None):
            reworded = "conservative" in questions["verdict"]["instructions"]
            label = "no" if (reworded and state == "s19") else "yes"
            return {"model": "stub", "answers": {"verdict": {
                "type": "choice", "choice": label,
                "probabilities": {label: 0.9}, "confidence": 0.9}}}

    baseline = evaluate(OneRowFlips(), twenty("Judge whether a refund is justified"),
                        evaluators=[ChoiceAccuracy()])
    candidate = evaluate(OneRowFlips(), twenty("Be conservative and only approve explicit refunds"),
                        evaluators=[ChoiceAccuracy()])
    assert candidate.overall["choice_accuracy"] == pytest.approx(0.95)

    ok, reasons = candidate.comparable_to(baseline.to_json())
    assert not ok, "a reworded question is a different experiment, not an identical one"
    assert any("questions_sha256" in reason for reason in reasons), reasons

    # And the same question over more rows is still the same experiment, which is the property
    # that would be lost if the fingerprint were simply "the whole dataset".
    def same_question(rows):
        q = {"verdict": {"type": "choice", "instructions": "Judge whether a refund is justified",
                         "criteria": criteria}}
        return Dataset([Example("s%d" % i, q, {"verdict": "yes"}) for i in range(rows)])

    longer = evaluate(OneRowFlips(), same_question(25), evaluators=[ChoiceAccuracy()])
    assert longer.comparable_to(baseline.to_json())[0], "more rows is not a different question"


def test_comparable_to_refuses_two_different_runs():
    """`compare` reads `overall` only, so without this a gate passes two different experiments.

    Both reports score 0.8, so the arithmetic is identical -- but they are not the same
    measurement, and a promotion decision made on that is unfalsifiable. Every field
    `docs/staged-adoption.md` tells the operator to record with the policy has to be able to
    stop the comparison.
    """
    baseline = _identified(dataset_sha256="a" * 64, questions_sha256="q" * 64)
    same = _identified(dataset_sha256="a" * 64, questions_sha256="q" * 64)
    ok, reasons = same.comparable_to(baseline.to_json())
    assert ok and reasons == [], reasons

    for field, other in (("schema", "laya-evals-report/act-head-eval/1"),
                         ("dataset_sha256", "b" * 64),
                         ("questions_sha256", "z" * 64)):
        candidate = _identified(dataset_sha256="a" * 64, questions_sha256="q" * 64)
        candidate.config[field] = other
        ok, reasons = candidate.comparable_to(baseline.to_json())
        assert not ok, field
        # The key is named, so the failure is greppable against the `config` a reviewer reads.
        assert any(field in reason for reason in reasons), (field, reasons)

    # The reason names both values, or a reviewer still has to open two files to act on it.
    swapped = _identified(dataset_sha256="c" * 64, questions_sha256="q" * 64)
    ok, reasons = swapped.comparable_to(baseline.to_json())
    assert "c" * 64 in " ".join(reasons) and "a" * 64 in " ".join(reasons)


def test_comparable_to_ignores_what_it_cannot_see():
    """A field missing on one side is unknown, not a conflict.

    Every baseline committed before this existed -- including the scheduled gate's
    `research/results/eval_english_51_languages.json`, which comes from `research/eval/` and
    has no `config.schema` at all -- has to keep comparing exactly as it did.
    """
    legacy = {"overall": {"choice_accuracy": 0.8}, "config": {"dataset": "old.jsonl"}}
    report = _identified(dataset_sha256="a" * 64, questions_sha256="q" * 64)
    ok, reasons = report.comparable_to(legacy)
    assert ok and reasons == [], reasons
    assert report.compare(legacy, {"choice_accuracy": 0.0})[0], "the metric gate still runs"

    # Two reports that both predate the identity fields are as comparable as they ever were.
    ok, reasons = EvalReport(config={"dataset": "a.jsonl"}).comparable_to(legacy)
    assert ok and reasons == [], reasons

    # A bare metric dict is a baseline `compare` already accepts, so it stays acceptable here.
    ok, reasons = report.comparable_to({"choice_accuracy": 0.8})
    assert ok and reasons == [], reasons


def test_a_comparing_report_does_not_read_a_shape_it_does_not_know():
    # `research/evals/act_head_eval.py` publishes a report under its own schema tag. A consumer
    # that cannot read it must be told, not left to compare a different metric space.
    foreign = {"schema": "laya-evals-report/act-head-eval/1",
               "overall": {"choice_accuracy": 0.8}, "cases": [], "slices": {}}
    ok, reasons = _identified(dataset_sha256="a" * 64).comparable_to(foreign)
    assert not ok and any("act-head-eval" in reason for reason in reasons), reasons


def test_docs_document_the_run_identity_and_the_refusal():
    """The docs table is derived against the code, not transcribed, so it cannot drift.

    `docs/evals.md` is what an operator reads before deciding what to record with a policy, so
    every key the report actually writes has to appear there.
    """
    from pathlib import Path

    page = (Path(__file__).resolve().parent.parent / "docs" / "evals.md").read_text(encoding="utf-8")
    dataset = Dataset([Example("s1", Q, {"intent": "a"})])
    report = evaluate(StubRunner({"s1": {"intent": choice_answer("a")}}), dataset)
    for key in report.config:
        if key != "timing":
            assert "`%s`" % key in page, key
    # The keys the CLI adds, and the programmatic entry points, are part of the same contract.
    for key in ("dataset_sha256", "thresholds", "gate_policy", "revisions"):
        assert "`%s`" % key in page, key
    for name in ("REPORT_SCHEMA", "questions_fingerprint", "file_fingerprint", "comparable_to"):
        assert name in page, name
    assert "not comparable" in page, "the refusal the gate prints is documented"


# --------------------------------------------------------------- CLI
def _write_dataset(tmp_path, rows):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_cli_validate_and_dispatch(tmp_path):
    from laya import cli, evals_cli

    good = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["validate", good]) == 0
    assert cli.main(["eval", "validate", good]) == 0, "`laya eval` dispatches to laya-evals"

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    # A dataset that is not JSONL is a usage error, so exit 2 (docs/evals.md:28) -- not the 1
    # that a threshold or baseline-tolerance failure uses. A CI job has to be able to tell
    # "my dataset is broken" from "the model regressed".
    assert evals_cli.main(["validate", str(bad)]) == 2


def test_cli_compare_exit_codes(tmp_path):
    from laya import evals_cli

    (tmp_path / "report.json").write_text(json.dumps({"overall": {"choice_accuracy": 0.80}}))
    (tmp_path / "baseline.json").write_text(json.dumps({"overall": {"choice_accuracy": 0.79}}))
    report, baseline = str(tmp_path / "report.json"), str(tmp_path / "baseline.json")
    assert evals_cli.main(["compare", report, "--baseline", baseline,
                           "--tolerance", "choice_accuracy=0.02"]) == 0
    assert evals_cli.main(["compare", report, "--baseline", baseline,
                           "--tolerance", "choice_accuracy=0.001"]) == 1


def test_cli_separates_a_usage_error_from_a_quality_failure(tmp_path, capsys):
    """docs/evals.md:28 promises 0 / 1 / 2, and the three must stay distinguishable.

    Everything the caller got wrong exits 2; only a real threshold or baseline-tolerance
    failure exits 1. Before, `main` had a single error path returning 1, so a CI job could
    not tell "my dataset is malformed" from "the model regressed" -- and a missing file escaped
    as a raw traceback, a third outcome the docs never mention.
    """
    from laya import evals_cli

    report = tmp_path / "report.json"
    report.write_text(json.dumps({"overall": {"choice_accuracy": 0.80}}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"overall": {"choice_accuracy": 0.79}}))

    good = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])

    # --- exit 0: the request was well-formed ---
    assert evals_cli.main(["validate", good]) == 0

    # --- exit 2: the caller's mistake, and no traceback ---
    usage = [
        ("a dataset that is not JSONL", ["validate", _write_raw(tmp_path, "x.jsonl", "{oops\n")]),
        ("a dataset with no examples", ["validate", _write_raw(tmp_path, "empty.jsonl", "")]),
        ("a row missing 'expected'",
         ["validate", _write_raw(tmp_path, "noexp.jsonl", json.dumps(
             {"qid": "intent", "state": "s", "questions": Q}) + "\n")]),
        ("a missing dataset file", ["validate", str(tmp_path / "nope.jsonl")]),
        ("a missing baseline report", ["compare", str(report), "--baseline",
                                       str(tmp_path / "nope.json")]),
        ("an unparseable tolerance", ["compare", str(report), "--baseline", str(baseline),
                                      "--tolerance", "choice_accuracy=abc"]),
    ]
    for label, argv in usage:
        capsys.readouterr()
        assert evals_cli.main(argv) == 2, label
        assert "Traceback" not in capsys.readouterr().err, "%s leaked a traceback" % label

    # --- exit 1: a real quality failure, still not a usage error ---
    assert evals_cli.main(["compare", str(report), "--baseline", str(baseline),
                           "--tolerance", "choice_accuracy=0.001"]) == 1


def _write_raw(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_cli_compare_fails_on_an_empty_report(tmp_path, capsys):
    from laya import evals_cli

    (tmp_path / "report.json").write_text(json.dumps({"overall": {}}))
    (tmp_path / "baseline.json").write_text(json.dumps({"overall": {"choice_accuracy": 0.79}}))
    assert evals_cli.main(["compare", str(tmp_path / "report.json"),
                           "--baseline", str(tmp_path / "baseline.json")]) == 1
    out = capsys.readouterr().out
    assert "choice_accuracy" in out and "missing from the report" in out


def test_cli_rejects_a_malformed_tolerance():
    from laya import evals_cli

    with pytest.raises(EvalError):
        evals_cli._parse_pairs(["choice_accuracy"])


def test_cli_run_records_the_run_identity(monkeypatch, tmp_path):
    """The `--json` artifact is what a reviewer reads, so it has to say which run produced it.

    `docs/evals.md` tells the reviewer to commit the dataset and a reviewed baseline together and
    read the result as a diff. That only works if the report carries the dataset's bytes and the
    question schema, not just the path, and if it records the gate the numbers came out of.
    """
    from laya import evals_cli

    _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    out = tmp_path / "report.json"
    assert evals_cli.main(["run", dataset, "--min-accuracy", "0.5", "--max-ece", "0.9",
                           "--tolerance", "choice_accuracy=0.02", "--json", str(out)]) == 0
    config = json.loads(out.read_text(encoding="utf-8"))["config"]
    assert config["schema"] == evals.REPORT_SCHEMA
    assert config["laya_version"] == laya.__version__
    assert config["dataset"] == dataset, "the path stays a name; the hash carries the bytes"
    assert config["dataset_sha256"] == evals.file_fingerprint(dataset)
    assert config["questions_sha256"] == evals.questions_fingerprint(evals.Dataset.from_jsonl(dataset))
    assert config["thresholds"]["min"] == {"choice_accuracy": 0.5}
    assert config["thresholds"]["max"] == {"ece": 0.9}
    assert config["thresholds"]["baseline_tolerance"] == {"choice_accuracy": 0.02}


def test_cli_run_refuses_a_baseline_from_a_different_dataset(monkeypatch, tmp_path, capsys):
    """The gate that a promotion decision rests on: two different runs cannot pass as one.

    Both runs score the same rows with the same stub, so every metric matches and the baseline
    comparison passes. But the baseline was recorded against different bytes, so the number being
    promoted was never measured on the data the gate claims to have validated. A pass here is
    arithmetic, not evidence.
    """
    from laya import evals_cli

    _patch_router(monkeypatch)
    original = _write_dataset(tmp_path, RUN_ROWS)
    baseline_out = tmp_path / "baseline.json"
    assert evals_cli.main(["run", original, "--json", str(baseline_out)]) == 0
    capsys.readouterr()

    # The same file name, one extra row: exactly what an edited dataset looks like in a diff.
    edited = tmp_path / "edited.jsonl"
    edited.write_text("\n".join(json.dumps(row) for row in RUN_ROWS)
                      + "\n" + json.dumps(RUN_ROWS[0]), encoding="utf-8")
    assert evals_cli.main(["run", str(edited), "--baseline", str(baseline_out),
                           "--tolerance", "choice_accuracy=0.05"]) == 1
    err = capsys.readouterr().err
    assert "baseline is not comparable" in err
    assert "dataset_sha256" in err, "the failure names the key a reviewer can check"


def test_cli_run_still_accepts_a_baseline_with_no_identity(monkeypatch, tmp_path, capsys):
    """Every baseline committed before the identity existed keeps passing exactly as it did.

    The scheduled gate compares against `research/results/eval_english_51_languages.json`, which
    comes from `research/eval/` and has no `config.schema`. A key missing on one side is unknown,
    not a conflict -- otherwise this change would break the repo's own CI on its first run.
    """
    from laya import evals_cli

    _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"config": {"dataset": "somewhere/else.jsonl"},
                                  "overall": {"choice_accuracy": 1.0}}), encoding="utf-8")
    assert evals_cli.main(["run", dataset, "--baseline", str(legacy),
                           "--tolerance", "choice_accuracy=0.05"]) == 0
    assert "not comparable" not in capsys.readouterr().err


def test_cli_compare_refuses_two_different_runs(tmp_path, capsys):
    from laya import evals_cli

    report = {"config": {"schema": evals.REPORT_SCHEMA, "dataset_sha256": "a" * 64,
                         "questions_sha256": "q" * 64},
              "overall": {"choice_accuracy": 0.8}, "cases": [], "slices": {}}
    baseline = {"config": {"schema": evals.REPORT_SCHEMA, "dataset_sha256": "b" * 64,
                           "questions_sha256": "q" * 64},
                "overall": {"choice_accuracy": 0.8}, "cases": [], "slices": {}}
    (tmp_path / "r.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(baseline), encoding="utf-8")
    # Identical `overall`, and the default tolerance is an exact match, so the metric gate passes.
    assert evals_cli.main(["compare", str(tmp_path / "r.json"),
                           "--baseline", str(tmp_path / "b.json")]) == 1
    captured = capsys.readouterr()
    assert "baseline is not comparable" in captured.err and "dataset_sha256" in captured.err
    # The deltas are still printed, so the reviewer sees what moved as well as why it is not a pass.
    assert "diff=" in captured.out and "choice_accuracy" in captured.out


def test_questions_fingerprint_covers_labels():
    """`labels` decides the option text, so it belongs in the key.

    The same argument `test_questions_fingerprint_covers_the_instructions` makes for
    `instructions`. `labels` is validated (`laya/agent.py:722-726` -> `_resolve_noul_labels`),
    carried into the internal question (`laya/agent.py:748-749`), and resolved into the option
    text the model actually reads (`laya/common.py:92`). Two question sets differing only in
    `labels` therefore ask the model different things, and hashing them alike let the gate pass
    a comparison it exists to refuse.
    """
    def with_labels(labels):
        q = {"verdict": {"type": "noul", "instructions": "Is this a refund?"}}
        if labels is not None:
            q["verdict"]["labels"] = labels
        return Dataset([Example("s", q, {"verdict": "true"})])

    default = evals.questions_fingerprint(with_labels(None))
    custom = evals.questions_fingerprint(with_labels(
        {"false": "denied: no money was requested",
         "true": "approved: the customer asked for money back"}))
    assert custom != default, "a change in `labels` left the fingerprint unchanged"
    # And the change is real: these are not two spellings of one thing.
    from laya.common import _resolve_noul_labels
    assert _resolve_noul_labels() != _resolve_noul_labels(
        {"false": "denied: no money was requested",
         "true": "approved: the customer asked for money back"})


def test_questions_fingerprint_keeps_criteria_order():
    """A choice question's criteria order is positional, so two orders are two questions.

    `examples/hooks/cache.py:22-25` states the rule and `Router._question_schema`
    (`laya/router.py:141`) applies it with `sort_keys=False`. Worth pinning here for the dict
    form: `json.dumps(sort_keys=True)` reorders *dict keys* and leaves *lists* alone, so a
    list-valued `criteria` -- the shape a dataset row has -- was already safe. The flag only
    mattered for a dict-valued one, where folding is the wrong direction to be wrong in.
    """
    def with_criteria(criteria):
        q = {"pick": {"type": "choice", "instructions": "Pick one", "criteria": criteria}}
        return Dataset([Example("s", q, {"pick": "a"})])

    assert evals.questions_fingerprint(with_criteria(["alpha", "beta"])) != \
        evals.questions_fingerprint(with_criteria(["beta", "alpha"]))
    assert evals.questions_fingerprint(with_criteria({"a": "alpha", "b": "beta"})) != \
        evals.questions_fingerprint(with_criteria({"b": "beta", "a": "alpha"}))


def test_a_top_level_schema_on_the_candidate_still_refuses(tmp_path, capsys):
    """The identity fallback is two-sided.

    `_identity_of` reads a report's identity from `config` *or* the top level, and its own comment
    says why: `research/evals/act_head_eval.py` puts `schema` there. But `comparable_to` applied
    it to the baseline only -- the candidate's identity came from `self.config`, and
    `laya/evals_cli.py` had already dropped a top-level `schema` when it built the `EvalReport`.
    So the *same* disagreement was refused when the candidate stated it in `config` and passed
    when it stated it at the top level.
    """
    from laya import evals_cli

    def write(name, doc):
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
        return str(tmp_path / name)

    shared = {"overall": {"choice_accuracy": 0.8}, "cases": [], "slices": {}}
    baseline = write("b.json", dict(shared, schema="act-head-eval/2"))

    # Same mismatch, candidate states it in `config` -- refused.
    assert evals_cli.main(["compare", write("in_config.json", dict(
        shared, config={"schema": "act-head-eval/1"})), "--baseline", baseline]) == 1
    capsys.readouterr()

    # Candidate states it at the top level -- must refuse too, not pass.
    assert evals_cli.main(["compare", write("top_level.json", dict(
        shared, schema="act-head-eval/1")), "--baseline", baseline]) == 1
    err = capsys.readouterr().err
    assert "schema" in err


# ------------------------------------------------------------------ CLI run
# One score row inside 0.25 of its label, one 0.3 away (inside 0.5 but not 0.25), plus a choice and
# an noul row so every default metric is in the report too.
RUN_ROWS = [
    {"state": "near", "questions": QSCORE, "expected": {"quality": 4}},
    {"state": "far", "questions": QSCORE, "expected": {"quality": 4}},
    {"state": "intent", "questions": Q, "expected": {"intent": "a"}},
    {"state": "flag", "questions": QNOUL, "expected": {"flag": True}},
]
RUN_ANSWERS = {
    "near": {"quality": score_answer(4.0)},
    "far": {"quality": score_answer(3.7)},
    "intent": {"intent": choice_answer("a")},
    "flag": {"flag": noul_answer(0.9)},
}


def _patch_router(monkeypatch, answers=None):
    """Replace the checkpoint-loading `Router`, so `laya-evals run` needs no weights.

    Returns the list the stand-in appends to when it is constructed, so a test can show a bad flag
    fails before anything loads.
    """
    import laya

    built: list = []

    class FakeRouter:
        def __init__(self, device=None, preload=False):
            built.append(device)

        def predict(self, state, questions, model=None):
            return {"model": model or "stub", "answers": (answers or RUN_ANSWERS)[state]}

    monkeypatch.setattr(laya, "Router", FakeRouter)
    return built


def _overall(stdout):
    return {name: float(value) for name, value in
            re.findall(r"^(\S+)\s+([0-9.]+)$", stdout, flags=re.M)}


def test_cli_score_within_publishes_the_documented_metric(monkeypatch, tmp_path, capsys):
    from laya import evals_cli

    _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    assert evals_cli.main(["run", dataset, "--score-within", "0.25"]) == 0
    overall = _overall(capsys.readouterr().out)
    assert overall["score_within_0.25"] == pytest.approx(0.5), "one of the two score rows is inside"
    for name in ("choice_accuracy", "noul_accuracy", "score_mae", "mean_confidence", "ece"):
        assert name in overall, "--score-within adds to the defaults, it does not replace them"


def test_cli_score_within_is_repeatable_per_column(monkeypatch, tmp_path, capsys):
    from laya import evals_cli

    _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    assert evals_cli.main(["run", dataset, "--score-within", "0.25", "--score-within", "0.5"]) == 0
    overall = _overall(capsys.readouterr().out)
    assert overall["score_within_0.25"] == pytest.approx(0.5)
    assert overall["score_within_0.5"] == pytest.approx(1.0), "the far row is inside 0.5"


def test_cli_score_within_gate_decides_on_the_number(monkeypatch, tmp_path, capsys):
    from laya import evals_cli

    _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    assert evals_cli.main(["run", dataset, "--score-within", "0.25",
                           "--min", "score_within_0.25=0.5"]) == 0
    capsys.readouterr()
    assert evals_cli.main(["run", dataset, "--score-within", "0.25",
                           "--min", "score_within_0.25=0.6"]) == 1
    assert "below the minimum" in capsys.readouterr().err


def test_cli_score_within_without_score_rows_says_so(monkeypatch, tmp_path, capsys):
    from laya import evals_cli

    _patch_router(monkeypatch)
    rows = [row for row in RUN_ROWS if row["state"] in ("intent", "flag")]
    dataset = _write_dataset(tmp_path, rows)
    assert evals_cli.main(["run", dataset, "--score-within", "0.25"]) == 0
    captured = capsys.readouterr()
    assert "score_within_0.25" not in captured.out, "no value is invented for it"
    assert "score_within_0.25 has no value" in captured.err
    assert "0 of 2 answered case(s) are score answers" in captured.err


def test_cli_rejects_an_unusable_tolerance_before_loading_a_checkpoint(monkeypatch, tmp_path, capsys):
    from laya import evals_cli

    built = _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    for bad in ("0.25", "-0.1", "nan", "inf"):
        capsys.readouterr()
        code = evals_cli.main(["run", dataset, "--score-within", bad])
        if bad == "0.25":
            assert code == 0 and built == [None]
            continue
        assert code == 2, "%r would name a metric that is always 1.0 or always 0.0" % bad
        assert "--score-within" in capsys.readouterr().err
        assert built == [None], "the flag is rejected before a checkpoint is loaded"


def test_cli_records_the_requested_tolerances(monkeypatch, tmp_path):
    from laya import evals_cli

    _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    out = tmp_path / "report.json"
    assert evals_cli.main(["run", dataset, "--score-within", "0.25", "--score-within", "0.5",
                           "--json", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["config"]["score_within"] == [0.25, 0.5]
    assert "score_within_0.5" in report["overall"]


def test_docs_and_the_cli_name_the_same_flags():
    import argparse
    from pathlib import Path

    from laya import evals_cli

    page = (Path(__file__).resolve().parent.parent / "docs" / "evals.md").read_text(encoding="utf-8")
    registered: set = set()
    for action in evals_cli._build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                registered.update(sub._option_string_actions)
    quickstart = page.split("```bash", 1)[1].split("```", 1)[0]
    # Lines that run another script (the ONNX exporter) teach that script's flags, not these.
    evals_lines = "\n".join(line for line in page.splitlines() if "export_onnx.py" not in line)
    taught = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", evals_lines))
    assert taught <= registered, "the page teaches %s, which no subcommand registers" % sorted(
        taught - registered)
    # The other half of parity: a registered flag nobody documents is unreachable in practice. The
    # page that teaches `--batch-size` has to teach the grouping that makes a bounded pass cheaper,
    # and the abstention threshold the report names even though the gate flags rather than rescores.
    assert {"--batch-size", "--sort-by-length", "--min-confidence"} <= taught, \
        "the evals page teaches the batch size but not the grouping or abstention knobs"
    assert "--score-within" in quickstart, "the tolerance metric has to be reachable from the quickstart"
    metrics = page.split("## Metrics", 1)[1].split("\n## ", 1)[0]
    assert "score_within" in metrics and "--score-within" in metrics, \
        "the section that publishes the metric has to carry the flag that reaches it"
    grouping = page.split("### Grouping the rows inside a batch", 1)[1].split("\n#", 1)[0]
    assert "--sort-by-length" in grouping and "sort_by_length_sent" in grouping, \
        "the subsection that explains the grouping has to name the flag and what it reports"
    gate = page.split("### The abstention gate at a threshold", 1)[1].split("\n#", 1)[0]
    assert "--min-confidence" in gate and "min_confidence_sent" in gate, \
        "the subsection that explains the gate has to name the flag and what it reports"
    assert "refused" in gate, "the docs have to teach that a gate-blind runner is refused, not silently scored"


# --------------------------------------------------------------- --revision pinning
SHA = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def _fake_router(monkeypatch, recorded):
    """Replace `laya.Router` with a weight-free stand-in that records how `_cmd_run` built it.

    It validates checkpoint names the way `Router.__init__` does, by calling the same
    `normalise_name`, so the fake refuses a typo for the real reason.
    """
    import laya

    class FakeRouter:
        def __init__(self, device=None, preload=None, revision=None, revisions=None):
            from laya.router import normalise_name
            recorded.update(device=device, preload=preload, revision=revision,
                            revisions={normalise_name(k): v for k, v in (revisions or {}).items()})

        def predict(self, state, questions, model=None, min_confidence=None):
            # The gate is a real part of the Router interface since #361; a stub without it would
            # turn a `--min-confidence` run into a TypeError and hide whether the CLI really
            # forwarded, rather than only parsing, the flag.
            return {"model": model or "english", "answers": {"intent": choice_answer("a")}}

        loaded_revisions = {"english": SHA}

    monkeypatch.setattr(laya, "Router", FakeRouter)


def test_cli_forwards_the_pin_and_records_the_commit_that_answered(tmp_path, monkeypatch):
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    out = tmp_path / "report.json"
    assert evals_cli.main(["run", dataset, "--model", "english", "--device", "cpu",
                           "--revision", "english=" + SHA, "--json", str(out)]) == 0
    assert recorded["revision"] is None
    assert recorded["revisions"] == {"english": SHA}
    # The report is the artifact a reviewer commits, so it has to carry the commit itself.
    assert json.loads(out.read_text())["config"]["revisions"] == {"english": SHA}


def test_cli_bare_revision_pins_every_checkpoint(tmp_path, monkeypatch):
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["run", dataset, "--revision", SHA, "--json",
                           str(tmp_path / "r.json")]) == 0
    # A bare SHA is one commit for every checkpoint, so nothing is pinned per name.
    assert recorded["revision"] == SHA and recorded["revisions"] == {}


def test_cli_records_an_unpinned_run_too(tmp_path, monkeypatch):
    """The default branch is what an unpinned baseline was taken on; the report must say so."""
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    out = tmp_path / "r.json"
    assert evals_cli.main(["run", dataset, "--json", str(out)]) == 0
    assert recorded["revision"] is None and recorded["revisions"] == {}
    assert json.loads(out.read_text())["config"]["revisions"] == {"english": SHA}


def test_cli_rejects_a_typo_in_a_pinned_checkpoint_name(tmp_path, monkeypatch, capsys):
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    # The code's own comment calls this "a usage error", and docs/evals.md:28 gives usage
    # errors exit 2. It used to exit 1, the code a threshold failure uses.
    assert evals_cli.main(["run", dataset, "--revision", "englishg=" + SHA]) == 2
    err = capsys.readouterr().err
    assert "unknown model 'englishg'" in err
    assert "choose one of" in err, "the message comes from core, with the option list"
    assert "Traceback" not in err, "a mistyped pin is a usage error, not a crash"
    assert not recorded, "and it fails before any checkpoint is loaded"


def test_cli_accepts_a_checkpoint_alias_in_a_pin(tmp_path, monkeypatch):
    """`Router` owns the alias table, so `en=` must reach it rather than be second-guessed here."""
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["run", dataset, "--revision", "en=" + SHA]) == 0
    assert recorded["revisions"] == {"english": SHA}


# --------------------------------------------------------------- opt-in slice gates
def _slice_gate_report(en_correct, zh_correct):
    cases = [{"language": language, "scores": {"choice_accuracy": float(i < correct)},
              "confidence": 0.8, "correct": i < correct}
             for language, size, correct in (("en", 180, en_correct), ("zh", 20, zh_correct))
             for i in range(size)]
    return {"config": {"schema": evals.REPORT_SCHEMA, "dataset_sha256": "a" * 64,
                       "questions_sha256": "b" * 64},
            "overall": {"choice_accuracy": (en_correct + zh_correct) / 200},
            "slices": {"language": {"en": {"choice_accuracy": en_correct / 180},
                                    "zh": {"choice_accuracy": zh_correct / 20}}},
            "cases": cases}


def _coverage_gate_report(value, definition, metric="selective_accuracy@50"):
    """A slice-gate report carrying a coverage metric and a definition stamp.

    `definition=None` is a report written before `coverage_metric_definition` existed, which is
    exactly what every committed baseline is.
    """
    cases = [{"language": "zh", "scores": {}, "confidence": 0.8, "correct": True}
             for _ in range(20)]
    config = {"schema": evals.REPORT_SCHEMA, "dataset_sha256": "a" * 64,
              "questions_sha256": "b" * 64}
    if definition is not None:
        config["coverage_metric_definition"] = definition
    return {"config": config, "overall": {metric: value},
            "slices": {"language": {"zh": {metric: value}}}, "cases": cases}


def _write_gate_policy(tmp_path, rule):
    path = tmp_path / "gates.json"
    path.write_text(json.dumps({"version": 1, "rules": [rule]}), encoding="utf-8")
    return str(path)


def test_cli_slice_gate_catches_regression_hidden_by_overall_gain(tmp_path, capsys):
    from laya import evals_cli

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_slice_gate_report(162, 18)), encoding="utf-8")
    candidate.write_text(json.dumps(_slice_gate_report(171, 12)), encoding="utf-8")
    policy = _write_gate_policy(tmp_path, {"slice": {"language": "zh"},
                                          "metric": "choice_accuracy", "min_count": 20,
                                          "max_drop": 0.05})
    args = ["compare", str(candidate), "--baseline", str(baseline),
            "--tolerance", "choice_accuracy=0.02"]
    assert evals_cli.main(args) == 0, "90% -> 91.5% overall passes the existing gate"
    capsys.readouterr()
    assert evals_cli.main(args + ["--gate-policy", policy]) == 1
    error = capsys.readouterr().err
    assert "language=zh" in error and "candidate=0.6000 (n=20)" in error
    assert "baseline=0.9000 (n=20)" in error and "max_drop=0.0500" in error


def test_slice_gate_requires_scored_evidence_on_both_sides(tmp_path):
    from laya import _eval_policy

    policy = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "choice_accuracy",
                   "min_count": 20, "max_drop": 0.05}))
    base = _slice_gate_report(162, 18)
    candidate = _slice_gate_report(171, 12)
    report = EvalReport(**candidate)
    assert not any("count" in f for f in _eval_policy.check_policy(report, policy, base))
    policy["rules"][0]["max_drop"] = 0.3
    assert _eval_policy.check_policy(report, policy, base) == [], "the exact drift boundary passes"
    policy["rules"][0]["max_drop"] = 0.05
    report.cases = [c for c in report.cases if c["language"] != "zh"] + report.cases[-19:]
    assert any("candidate count 19" in f for f in _eval_policy.check_policy(report, policy, base))
    report = EvalReport(**candidate)
    base["cases"] = [c for c in base["cases"] if c["language"] != "zh"] + base["cases"][-19:]
    assert any("baseline count 19" in f for f in _eval_policy.check_policy(report, policy, base))
    del base["slices"]["language"]["zh"]
    assert any("baseline metric or slice is missing" in f
               for f in _eval_policy.check_policy(report, policy, base))


def test_slice_gate_absolute_and_increase_direction(tmp_path):
    from laya import _eval_policy

    candidate = _slice_gate_report(171, 12)
    report = EvalReport(**candidate)
    absolute = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "choice_accuracy",
                   "min_count": 20, "min": 0.7}))
    assert any("min=0.7000 failed" in f for f in _eval_policy.check_policy(report, absolute))
    candidate["slices"]["language"]["zh"].pop("choice_accuracy")
    assert any("metric or slice is missing" in f for f in _eval_policy.check_policy(report, absolute))
    candidate = _slice_gate_report(171, 12)
    candidate["slices"]["language"]["zh"]["ece"] = 0.3
    for case in candidate["cases"]:
        if case["language"] == "zh":
            case["confidence"] = 0.8
    base = _slice_gate_report(162, 18)
    base["slices"]["language"]["zh"]["ece"] = 0.1
    increase = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "ece",
                   "min_count": 20, "max_increase": 0.05}))
    assert any("max_increase=0.0500" in f
               for f in _eval_policy.check_policy(EvalReport(**candidate), increase, base))


def test_slice_gate_ece_requires_boolean_correct_evidence(tmp_path):
    from laya import _eval_policy

    candidate = _slice_gate_report(171, 12)
    candidate["slices"]["language"]["zh"]["ece"] = 0.2
    for case in candidate["cases"]:
        if case["language"] == "zh":
            case["correct"] = int(case["correct"])
    policy = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "ece",
                   "min_count": 20, "max": 0.3}))
    assert any("candidate count 0 is below min_count 20" in failure
               for failure in _eval_policy.check_policy(EvalReport(**candidate), policy))
    for case in candidate["cases"]:
        if case["language"] == "zh":
            case["correct"] = bool(case["correct"])
    assert _eval_policy.check_policy(EvalReport(**candidate), policy) == []


def test_slice_gate_counts_each_tagged_answer_once_and_accepts_boundary(tmp_path):
    from laya import _eval_policy

    candidate = _slice_gate_report(171, 12)
    for case in candidate["cases"]:
        case.update(qid="intent", model="stub", tags=["critical", "critical"])
    candidate["slices"].update({
        "tag": {"critical": {"choice_accuracy": 183 / 200}},
        "qid": {"intent": {"choice_accuracy": 183 / 200}},
        "model": {"stub": {"choice_accuracy": 183 / 200}},
    })
    policy = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"tag": "critical"}, "metric": "choice_accuracy",
                   "min_count": 200, "min": 0.915}))
    report = EvalReport(**candidate)
    policy["rules"][0]["min_count"] = 201
    assert any("candidate count 200 is below min_count 201" in failure
               for failure in _eval_policy.check_policy(report, policy))
    policy["rules"][0]["min_count"] = 200
    assert _eval_policy.check_policy(report, policy) == []
    for dimension, value in (("qid", "intent"), ("model", "stub")):
        policy["rules"][0]["slice"] = {dimension: value}
        assert _eval_policy.check_policy(report, policy) == []


@pytest.mark.parametrize("rule", [
    {"slice": {"language": "zh"}, "metric": "choice_accuracy", "min_count": 20,
     "min": 0.6, "max_drop": 0.1},
    {"slice": {"language": "zh", "tag": "critical"}, "metric": "choice_accuracy",
     "min_count": 20, "min": 0.6},
    {"slice": {"region": "cn"}, "metric": "choice_accuracy", "min_count": 20,
     "min": 0.6},
    {"slice": {"language": "zh"}, "metric": "choice_accuracy", "min_count": 0,
     "min": 0.6},
    {"slice": {"language": "zh"}, "metric": "choice_accuracy", "min_count": 20,
     "max_drop": -0.1},
])
def test_cli_rejects_malformed_slice_policy_before_loading(monkeypatch, tmp_path, rule):
    from laya import evals_cli

    built = _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    policy = _write_gate_policy(tmp_path, rule)
    assert evals_cli.main(["run", dataset, "--gate-policy", policy]) == 2
    assert not built


def test_cli_relative_slice_rule_requires_baseline_before_loading(monkeypatch, tmp_path):
    from laya import evals_cli

    built = _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    policy = _write_gate_policy(tmp_path, {"slice": {"language": "zh"},
                                          "metric": "choice_accuracy", "min_count": 1,
                                          "max_drop": 0.1})
    assert evals_cli.main(["run", dataset, "--gate-policy", policy]) == 2
    assert not built


def test_cli_rejects_invalid_policy_json_and_version(monkeypatch, tmp_path):
    from laya import evals_cli

    built = _patch_router(monkeypatch)
    dataset = _write_dataset(tmp_path, RUN_ROWS)
    path = tmp_path / "gates.json"
    for content in ("{not json", json.dumps({"version": 2, "rules": [
            {"slice": {"language": "zh"}, "metric": "choice_accuracy",
             "min_count": 1, "min": 0.5}]})):
        path.write_text(content, encoding="utf-8")
        assert evals_cli.main(["run", dataset, "--gate-policy", str(path)]) == 2
    assert not built


def test_cli_run_slice_gate_and_saved_compare_share_policy(monkeypatch, tmp_path, capsys):
    from laya import evals_cli

    # Choice answers without confidence keep this gate-wiring test weight-free.
    def answer(label):
        return {"type": "choice", "choice": label}
    rows = [{"state": language, "questions": Q, "expected": {"intent": "a"},
             "language": language} for language in ("en", "zh")]
    dataset = _write_dataset(tmp_path, rows)
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _patch_router(monkeypatch, {"en": {"intent": answer("a")},
                              "zh": {"intent": answer("a")}})
    assert evals_cli.main(["run", dataset, "--json", str(baseline)]) == 0
    _patch_router(monkeypatch, {"en": {"intent": answer("a")},
                              "zh": {"intent": answer("b")}})
    policy = _write_gate_policy(tmp_path, {"slice": {"language": "zh"},
                                          "metric": "choice_accuracy", "min_count": 1,
                                          "max_drop": 0.1})
    flags = ["--baseline", str(baseline), "--tolerance", "choice_accuracy=1",
             "--gate-policy", policy]
    assert evals_cli.main(["run", dataset, *flags, "--json", str(candidate)]) == 1
    assert json.loads(candidate.read_text())["config"]["gate_policy"]["rules"][0]["max_drop"] == 0.1
    assert evals_cli.main(["compare", str(candidate), *flags]) == 1
    assert "language=zh" in capsys.readouterr().err


def test_cli_compare_reports_a_different_recorded_gate_policy(tmp_path, capsys):
    from laya import evals_cli

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    policy_path = tmp_path / "policy.json"
    strict = {"version": 1, "rules": [{"slice": {"language": "zh"},
                                       "metric": "choice_accuracy", "min_count": 20,
                                       "min": 0.99}]}
    lax = {"version": 1, "rules": [{"slice": {"language": "zh"},
                                    "metric": "choice_accuracy", "min_count": 20,
                                    "min": 0.0}]}
    baseline.write_text(json.dumps(_slice_gate_report(162, 18)), encoding="utf-8")
    document = _slice_gate_report(171, 12)
    document["config"]["gate_policy"] = strict
    candidate.write_text(json.dumps(document), encoding="utf-8")
    args = ["compare", str(candidate), "--baseline", str(baseline),
            "--tolerance", "choice_accuracy=0.02"]
    policy_path.write_text(json.dumps(lax), encoding="utf-8")
    assert evals_cli.main(args + ["--gate-policy", str(policy_path)]) == 0
    assert "warning: --gate-policy differs from the report's recorded gate_policy" in capsys.readouterr().err
    policy_path.write_text(json.dumps(strict), encoding="utf-8")
    assert evals_cli.main(args + ["--gate-policy", str(policy_path)]) == 1
    assert "warning: --gate-policy differs" not in capsys.readouterr().err
    assert evals_cli.main(args) == 0
    assert capsys.readouterr().err == ""


def test_slice_gate_fails_closed_on_identity_or_skipped_cases(tmp_path):
    from laya import _eval_policy

    candidate = _slice_gate_report(171, 12)
    base = _slice_gate_report(162, 18)
    policy = _eval_policy.load_policy(_write_gate_policy(
        tmp_path, {"slice": {"language": "zh"}, "metric": "choice_accuracy",
                   "min_count": 20, "max_drop": 1.0}))
    base["config"].pop("dataset_sha256")
    assert any("need dataset_sha256 identity" in f
               for f in _eval_policy.check_policy(EvalReport(**candidate), policy, base))
    base["config"]["dataset_sha256"] = "c" * 64
    assert any("not comparable" in f
               for f in _eval_policy.check_policy(EvalReport(**candidate), policy, base))
    base["config"]["dataset_sha256"] = "a" * 64
    candidate["config"]["errored"] = [{"index": 1, "error": "missing answer"}]
    assert any("skipped/errored" in f
               for f in _eval_policy.check_policy(EvalReport(**candidate), policy, base))


@pytest.mark.parametrize("pairs, expected", [
    (None, (None, {})),
    ([], (None, {})),
    ([SHA], (SHA, {})),
    ([SHA, SHA], (SHA, {})),                      # repeating one commit is agreement, not a clash
    (["en=" + SHA], (None, {"en": SHA})),
    (["en=" + SHA, SHA], (SHA, {"en": SHA})),     # both forms at once: Router lets the pair win
])
def test_parse_revisions_accepts_both_forms(pairs, expected):
    from laya import evals_cli

    assert evals_cli._parse_revisions(pairs) == expected


@pytest.mark.parametrize("pair, fragment", [
    ("=abc", "NAME=REVISION"),
    ("en=", "NAME=REVISION"),
    ("   ", "commit SHA"),
    ("abc", None),
])
def test_parse_revisions_rejects_half_a_pair(pair, fragment):
    from laya import evals_cli

    if fragment is None:
        assert evals_cli._parse_revisions([pair]) == (pair, {})
        return
    with pytest.raises(EvalError) as exc:
        evals_cli._parse_revisions([pair])
    assert fragment in str(exc.value)


def test_parse_revisions_rejects_two_different_bare_commits():
    from laya import evals_cli

    with pytest.raises(EvalError) as exc:
        evals_cli._parse_revisions(["abc", "def"])
    assert "two commits" in str(exc.value)
