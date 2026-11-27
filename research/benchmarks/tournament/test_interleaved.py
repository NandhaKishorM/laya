"""Behavioral timing contracts with local fake models, without Hub downloads."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from research.benchmarks.tournament import interleaved as timing
from research.benchmarks.tournament.embedders import EmbeddingSpec


SPEC = EmbeddingSpec("fake", "a" * 40, "mean", 2, 32, "Q:", "D:")


class Encoder:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.device = "cpu"
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[1., (sum(map(ord, text)) % 23) / 23] for text in texts], dtype=np.float32)


class Agent:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.device = "cpu"
        self.dtype = torch.float32
        self.calls = []

    def dtype_for(self, rows):
        return self.dtype

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return {"answers": {qid: {"type": "choice", "choice": next(iter(q["criteria"]))}
                            for qid, q in questions.items()}}


def inputs():
    questions = {"q": {"type": "choice", "instructions": "Which?",
                       "criteria": dict.fromkeys("label%d" % i for i in range(32))}}
    cases = [{"id": "test:%d" % i, "state": {"message": "test%d" % i}} for i in range(2)]
    warmup = [{"id": "dev:%d" % i, "state": {"message": "warm%d" % i}} for i in range(3)]
    return cases, questions, warmup


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(timing, "memory_snapshot", lambda device: {"scope": "fake"})
    monkeypatch.setattr(timing, "infeasible_budget", lambda *args: None)
    def invoke(arms, encoder=None, agent=None, **kwargs):
        cases, questions, warmup = inputs()
        return timing.run_interleaved(agent or Agent(), kwargs.pop("cases", cases), questions, arms,
                                      {"qd": encoder or Encoder()}, {"qd": SPEC},
                                      kwargs.pop("warmup", warmup), **kwargs)
    return invoke


def test_each_arm_encodes_every_query_and_labels_only_once(run):
    encoder, agent = Encoder(), Agent()
    arms = [timing.Arm("top", "top1", embedder="qd"),
            timing.Arm("short", "shortlist", embedder="qd"),
            timing.Arm("combo", "combined", embedder="qd")]
    result = run(arms, encoder, agent, passes=3)
    assert len([call for call in encoder.calls if len(call) == 32]) == 1
    # 3 warmups and 2 measured queries * 3 passes, independently for each arm.
    queries = [call for call in encoder.calls if len(call) == 1]
    assert len(queries) == 27
    assert all(text.startswith("Q:") for call in queries for text in call)
    assert all(sample["query_encodings"] == 1 for sample in result["samples"])
    assert all(sample["decision_calls"] == {"top": 0, "short": 1, "combo": 2}[sample["arm"]]
               for sample in result["samples"])
    # No decision calls for top1; one for shortlist; two for combined k20/group16.
    assert len(agent.calls) == 9 * (0 + 1 + 2)
    assert result["independent_requests"] == 2
    assert all(len(values) == 2 for arm in result["arms"].values() for values in arm["passes_ms"])
    assert result["schedule"][:3] == [
        {"pass": 0, "id": "test:0", "arms": ["top", "short", "combo"]},
        {"pass": 0, "id": "test:1", "arms": ["short", "combo", "top"]},
        {"pass": 1, "id": "test:0", "arms": ["short", "combo", "top"]}]


def test_query_equal_to_label_is_still_encoded_and_label_storage_is_immutable():
    encoder = Encoder()
    labels = timing._options(encoder, ["same", "other"], "D:")
    adapter = timing.ColdQuery(encoder, labels, "Q:")
    first = adapter(["same", "same"])
    adapter(["same", "same"])
    assert encoder.calls == [["D:same", "D:other"], ["Q:same"], ["Q:same"]]
    np.testing.assert_array_equal(first[1], labels["same"])
    with pytest.raises(ValueError):
        labels["same"][0] = 123
    with pytest.raises(TypeError):
        labels["other"] = np.zeros(2)


def test_prefix_variants_share_weights_but_keep_separate_label_vectors(monkeypatch):
    monkeypatch.setattr(timing, "memory_snapshot", lambda device: {})
    encoder = Encoder()
    cases, questions, warmup = inputs()
    arms = [timing.Arm("qd", "shortlist", embedder="qd"),
            timing.Arm("qq", "combined", embedder="qq")]
    result = timing.run_interleaved(Agent(), cases, questions, arms, {"qd": encoder, "qq": encoder},
                                    {"qd": SPEC, "qq": replace(SPEC, option_prefix="Q:")}, warmup, passes=1)
    setups = [call for call in encoder.calls if len(call) == 32]
    assert len(setups) == 2
    assert all(text.startswith("D:") for text in setups[0])
    assert all(text.startswith("Q:") for text in setups[1])
    assert len([call for call in encoder.calls if len(call) == 1]) == 10
    assert set(result["arms"]) == {"qd", "qq"}


def test_builtin_has_cold_queries_and_resident_labels(run, monkeypatch):
    builtin = Encoder()
    monkeypatch.setattr(timing, "embed_fn_from_agent", lambda agent: builtin)
    result = run([timing.Arm("builtin", "builtin"), timing.Arm("top", "top1", embedder="qd")], passes=2)
    assert len([call for call in builtin.calls if len(call) == 32]) == 1
    assert len([call for call in builtin.calls if len(call) == 1]) == 7
    assert result["label_setup"]["builtin"]["unique_texts"] == 32


def test_sync_brackets_only_live_request_and_reporting_is_outside_timer(run, monkeypatch):
    events = []
    agent = Agent()
    original = agent.predict
    def predict(*args, **kwargs):
        events.append("predict")
        return original(*args, **kwargs)
    agent.predict = predict
    monkeypatch.setattr(timing, "synchronize", lambda device: events.append("sync"))
    ticks = iter([0., .002, 1., 1.004])
    def clock():
        events.append("clock")
        return next(ticks)
    result = run([timing.Arm("single", "single")], agent=agent, passes=1, clock=clock,
                 progress=lambda report: events.append("progress"))
    assert events == ["sync", "predict", "sync"] * 3 + [
        "sync", "clock", "predict", "sync", "clock", "progress"] * 2
    np.testing.assert_allclose(result["arms"]["single"]["passes_ms"], [[2., 4.]])
    assert result["arms"]["single"]["p50_ms"] == pytest.approx(3.)
    assert result["arms"]["single"]["p95_ms"] == pytest.approx(3.9)


def test_full_eight_arm_schedule_contains_every_arm(run, monkeypatch):
    monkeypatch.setattr(timing, "embed_fn_from_agent", lambda agent: Encoder())
    arms = [timing.Arm(strategy, strategy, embedder="qd" if strategy in timing.DEDICATED else None)
            for strategy in timing.STRATEGIES]
    result = run(arms, passes=1)
    assert set(result["arms"]) == set(timing.STRATEGIES)
    assert len(result["samples"]) == 16
    assert all(set(row["arms"]) == set(timing.STRATEGIES) for row in result["schedule"])


@pytest.mark.parametrize("arms", [[], [timing.Arm("x", "missing")],
                                    [timing.Arm("x", "top1", embedder="missing")],
                                    [timing.Arm("x", "single"), timing.Arm("x", "wide")]])
def test_missing_unknown_and_duplicate_arms_are_rejected(run, arms):
    with pytest.raises(ValueError):
        run(arms)


def test_device_mismatch_and_fallback_fail_without_complete_samples(run):
    arms = [timing.Arm("short", "shortlist", embedder="qd")]
    with pytest.raises(timing.LatencyFailure, match="actual device cpu") as failed:
        run(arms, requested_device="mps")
    assert failed.value.report["status"] == "failed"
    assert not failed.value.report["samples"]
    agent = Agent()
    original = agent.predict
    def fallback(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(agent.calls) == 4:  # After the three warmups, inside the first measured call.
            agent.device = "meta"
        return result
    agent.predict = fallback
    with pytest.raises(timing.LatencyFailure, match="changed within a decision round") as failed:
        run(arms, agent=agent)
    assert failed.value.report["status"] == "failed"
    assert not failed.value.report["samples"]


def test_tensor_device_is_checked_even_if_owner_claims_cpu(run):
    agent = Agent()
    agent.model = agent.model.to("meta")
    with pytest.raises(timing.LatencyFailure, match="model tensors disagree"):
        run([timing.Arm("single", "single")], agent=agent)


@pytest.mark.parametrize("counter", ["cpu_fallback_count", "_amp_failures"])
def test_transient_fallback_is_rejected_even_if_device_is_restored(run, counter):
    agent = Agent()
    original = agent.predict
    def fallback(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(agent.calls) == 4:
            setattr(agent, counter, 1)
        return result
    agent.predict = fallback
    with pytest.raises(timing.LatencyFailure, match="changed within a decision round") as failed:
        run([timing.Arm("single", "single")], agent=agent)
    assert not failed.value.report["samples"]


def test_declared_encoder_spec_must_match_actual_loaded_model(run):
    encoder = Encoder()
    encoder.spec = replace(SPEC, pooling="cls")
    with pytest.raises(ValueError, match="weight/pooling contract"):
        run([timing.Arm("top", "top1", embedder="qd")], encoder=encoder)


def test_failure_preserves_completed_samples_but_never_marks_arm_complete(run):
    def fail_on_second(report):
        if report["samples"][-1]["id"] == "test:1":
            raise RuntimeError("interrupted")
    with pytest.raises(timing.LatencyFailure, match="interrupted") as failed:
        run([timing.Arm("top", "top1", embedder="qd")], passes=3, progress=fail_on_second)
    assert len(failed.value.report["samples"]) == 2
    assert failed.value.report["arms"]["top"]["status"] == "running"
    assert "p50_ms" not in failed.value.report["arms"]["top"]


def test_progress_contains_all_completed_observations(run):
    checkpoints = []
    def progress(report):
        checkpoints.append((report["status"], [dict(sample) for sample in report["samples"]]))
    result = run([timing.Arm("top", "top1", embedder="qd")], passes=2, progress=progress)
    assert [len(samples) for status, samples in checkpoints] == [1, 2, 3, 4]
    assert all(status == "running" for status, samples in checkpoints)
    assert checkpoints[-1][1] == result["samples"]


@pytest.mark.parametrize("bad", [None, {}, {"answers": {}},
                                {"answers": {"q": {"type": "choice", "choice": "unknown"}}},
                                {"answers": {"q": {"type": "score", "choice": "label0"}}}])
def test_invalid_answer_is_failed_request_not_a_fast_sample(run, bad):
    agent = Agent()
    original = agent.predict
    def predict(*args, **kwargs):
        result = original(*args, **kwargs)
        return bad if len(agent.calls) == 4 else result
    agent.predict = predict
    with pytest.raises(timing.LatencyFailure, match="decision returned") as failed:
        run([timing.Arm("single", "single")], agent=agent)
    assert failed.value.report["status"] == "failed"
    assert not failed.value.report["samples"]
    assert "p50_ms" not in failed.value.report["arms"]["single"]


def test_tournament_round_cannot_hide_amp_fallback_before_final_call(run):
    agent = Agent()
    original = agent.predict
    def predict(state, questions, **kwargs):
        result = original(state, questions, **kwargs)
        if len(agent.calls) > 6:  # The three warmups each make two decisions.
            agent._amp_failures = 1 if len(questions) > 1 else 0
        return result
    agent.predict = predict
    with pytest.raises(timing.LatencyFailure, match="changed within a decision round") as failed:
        run([timing.Arm("combined", "combined", embedder="qd")], agent=agent)
    assert len(agent.calls) == 7  # Abort before a successful final call could reset the counter.
    assert not failed.value.report["samples"]


def test_tournament_round_choice_must_belong_to_that_group(run):
    agent = Agent()
    original = agent.predict
    def predict(state, questions, **kwargs):
        result = original(state, questions, **kwargs)
        if len(agent.calls) == 7:
            other_group_label = next(iter(questions["1"]["criteria"]))
            result["answers"]["0"]["choice"] = other_group_label
        return result
    agent.predict = predict
    with pytest.raises(timing.LatencyFailure, match="out-of-set choice") as failed:
        run([timing.Arm("combined", "combined", embedder="qd")], agent=agent)
    assert len(agent.calls) == 7
    assert not failed.value.report["samples"]


def test_warmup_overlap_and_repeated_requests_are_rejected(run):
    cases, _, warmup = inputs()
    warmup[0]["state"] = cases[0]["state"]
    with pytest.raises(ValueError, match="disjoint"):
        run([timing.Arm("single", "single")], warmup=warmup)
    cases[1]["state"] = cases[0]["state"]
    with pytest.raises(ValueError, match="distinct requests"):
        run([timing.Arm("single", "single")], cases=cases)


def test_infeasible_arm_remains_visible_without_fabricated_samples(run, monkeypatch):
    monkeypatch.setattr(timing, "infeasible_budget", lambda agent, state, questions, strategy:
                        {"status": "infeasible", "reason": "budget"} if strategy == "single" else None)
    result = run([timing.Arm("single", "single"), timing.Arm("top", "top1", embedder="qd")], passes=1)
    assert result["arms"]["single"]["status"] == "infeasible"
    assert result["arms"]["single"]["passes_ms"] == [[]]
    assert "p50_ms" not in result["arms"]["single"]
    assert all(sample["arm"] == "top" for sample in result["samples"])
