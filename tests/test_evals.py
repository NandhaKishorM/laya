"""Metric math, dataset parsing and regression comparison for laya.evals. No weights.

Run: python -m pytest tests/test_evals.py -q
"""
import json
import re

import pytest

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


def test_p95_is_nearest_rank_when_the_sample_size_is_a_multiple_of_20():
    # 0.95 * 20 = 19 exactly: the 19th smallest, not the 20th (#680 -- int(n * 0.95) landed
    # one rank too high whenever the product was whole).
    from laya.evals import _percentiles

    assert _percentiles([float(i) for i in range(1, 21)])[1] == 19.0
    assert _percentiles([float(i) for i in range(1, 41)])[1] == 38.0
    # Away from the exact boundary the rank is unchanged, and the ends hold.
    assert _percentiles([float(i) for i in range(1, 22)])[1] == 20.0
    assert _percentiles([10.0])[1] == 10.0
    assert _percentiles([3.0, 7.0])[1] == 7.0


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
    assert evals_cli.main(["validate", str(bad)]) == 1


def test_cli_compare_exit_codes(tmp_path):
    from laya import evals_cli

    (tmp_path / "report.json").write_text(json.dumps({"overall": {"choice_accuracy": 0.80}}))
    (tmp_path / "baseline.json").write_text(json.dumps({"overall": {"choice_accuracy": 0.79}}))
    report, baseline = str(tmp_path / "report.json"), str(tmp_path / "baseline.json")
    assert evals_cli.main(["compare", report, "--baseline", baseline,
                           "--tolerance", "choice_accuracy=0.02"]) == 0
    assert evals_cli.main(["compare", report, "--baseline", baseline,
                           "--tolerance", "choice_accuracy=0.001"]) == 1


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
        assert code == 1, "%r would name a metric that is always 1.0 or always 0.0" % bad
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
    assert "--score-within" in quickstart, "the tolerance metric has to be reachable from the quickstart"
    metrics = page.split("## Metrics", 1)[1].split("\n## ", 1)[0]
    assert "score_within" in metrics and "--score-within" in metrics, \
        "the section that publishes the metric has to carry the flag that reaches it"


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

        def predict(self, state, questions, model=None):
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
    assert evals_cli.main(["run", dataset, "--revision", "englishg=" + SHA]) == 1
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
