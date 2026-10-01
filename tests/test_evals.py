"""Metric math, dataset parsing and regression comparison for laya.evals. No weights.

Run: python -m pytest tests/test_evals.py -q
"""
import json
import re

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
    default_evaluators,
    ece,
    evaluate,
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
# and mark answers below `min_confidence` with `low_confidence: True` (#361). That is a scoring
# control: an answer below the threshold is not the same decision as one above. A `laya-evals run`
# that could not pass it through had no way to measure `precision@coverage` at any threshold
# except by wrapping a Router by hand -- the exact class of thing this harness exists to be.

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
    """Silently dropping a scoring control would report `precision@coverage` for a policy that
    never ran. `RequestsRunner`'s `predict_batch(requests, batch_size=None)` is exactly what a
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

    page = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "evals.md").read_text()
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
    # and the abstention threshold that changes which answers score at all.
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
