"""Offline checks for the amendment's row identity and order controls."""
import copy
import sys

import numpy as np
import pytest
import torch

from laya.evals import Dataset, Example, questions_fingerprint
from research.benchmarks.tournament import amended_comparison as amended
from research.benchmarks.tournament.diagnostics import fingerprint


class Agent:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.device = "cpu"
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((copy.deepcopy(state), copy.deepcopy(questions)))
        return {"answers": {key: {"type": "choice", "choice": next(iter(q["criteria"])),
                                  "answer_confidence": .75} for key, q in questions.items()}}


class Embedder:
    def cache_clear(self):
        pass

    def warm_options(self, texts):
        pass

    def __call__(self, texts):
        return np.array([[1., 0.]] + [[float(text), 1.] for text in texts[1:]])


def data(count=32, rows=2):
    qs = {"q": {"type": "choice", "instructions": "Which?", "criteria": dict.fromkeys(map(str, range(count)))}}
    cases = [{"id": "test:%d" % i, "state": {"message": "request%d" % i}, "expected": str(count - 1 - i)}
             for i in range(rows)]
    for case in cases:
        case["state_sha256"] = fingerprint(case["state"])
    dataset = Dataset([Example(case["state"], qs, {"q": case["expected"]}) for case in cases])
    identity = {"case_ids_sha256": fingerprint([case["id"] for case in cases]),
                "questions_sha256": questions_fingerprint(dataset), "checkpoint_revision": "a" * 40}
    return cases, qs, identity


def test_permutation_reorders_same_retrieval_membership_before_grouping():
    cases, questions, identity = data()
    natural = amended.quality(Agent(), cases, questions, identity, "combined", Embedder())
    order = amended.label_order(questions, 11)
    changed = amended.quality(Agent(), cases, questions, identity, "combined", Embedder(), order=order)
    assert questions["q"]["criteria"] == dict.fromkeys(map(str, range(32)))
    for baseline, observed in zip(natural["cases"], changed["cases"]):
        assert observed["shortlist"] == baseline["shortlist"]
        assert observed["decision_order"] == [label for label in order if label in baseline["shortlist"]]
        assert observed["gold_rank"] == baseline["gold_rank"]
        assert observed["groups"][0] == [observed["decision_order"][:10], observed["decision_order"][10:]]
        assert len(observed["finalists"]) == 2
        assert observed["decision_calls"] == 2
    assert sum(changed["stages"]["counts"].values()) == len(cases)


def test_shortlist_is_a_presentation_order_control_and_gold_never_enters_predict():
    cases, qs, identity = data()
    agent = Agent()
    order = list(qs["q"]["criteria"])
    result = amended.quality(agent, cases, qs, identity, "shortlist", Embedder(), order=order)
    for index, row in enumerate(result["cases"]):
        assert row["shortlist"] == list(map(str, range(31, 11, -1)))
        assert row["decision_order"] == list(map(str, range(12, 32)))
        assert row["choice"] == "12"
        assert row["confidence"] == .75
        assert row["decision_calls"] == 1
        assert agent.calls[index][0] == cases[index]["state"]
        assert set(agent.calls[index][0]) == {"message"}
        assert "expected" not in str(agent.calls[index][1])


@pytest.mark.parametrize("k", [5, 8, 12, 16])
def test_one_group_combined_is_exactly_one_shortlist_decision_and_alias(k):
    cases, qs, identity = data()
    order = amended.label_order(qs, 29)
    short_agent, combined_agent = Agent(), Agent()
    short = amended.quality(short_agent, cases, qs, identity, "shortlist", Embedder(), k=k, order=order)
    combined = amended.quality(combined_agent, cases, qs, identity, "combined", Embedder(), k=k, order=order)
    assert short_agent.calls == combined_agent.calls
    assert short["stages"] == combined["stages"]
    assert [row["choice"] for row in short["cases"]] == [row["choice"] for row in combined["cases"]]
    assert all(row["decision_calls"] == 1 for row in combined["cases"])
    alias = amended.alias_single_pass(short)
    assert alias["degenerate_single_pass"] and alias["equivalent_to"] == "shortlist"
    assert alias["strategy"] == "combined" and short["strategy"] == "shortlist"
    alias["cases"][0]["choice"] = "changed"
    assert short["cases"][0]["choice"] != "changed"


def test_selected_cases_are_exact_permutation_slice_with_fresh_identity(monkeypatch):
    calls = []
    all_cases, qs, identity = data(rows=10)
    def load(name, split, limit, seed):
        calls.append((name, split, limit, seed))
        cases = all_cases[:limit]
        dataset = Dataset([Example(c["state"], qs, {"q": c["expected"]}) for c in cases])
        return cases, dataset, identity
    monkeypatch.setattr(amended.fixed, "load_cases", load)
    cases, dataset, selected = amended.selected_cases("banking77", "dev", 4, 3, 0)
    assert calls == [("banking77", "dev", 7, 0)]
    assert cases == all_cases[3:7]
    assert [e.state for e in dataset.examples] == [c["state"] for c in cases]
    assert selected["case_ids_sha256"] == fingerprint([c["id"] for c in cases])
    assert selected["offset"] == 3
    assert identity["case_ids_sha256"] != selected["case_ids_sha256"]


@pytest.mark.parametrize("failure", ["case_ids", "questions", "state", "duplicate", "gold"])
def test_identity_corruption_is_rejected(failure):
    cases, qs, identity = data()
    if failure == "case_ids":
        cases.reverse()
    elif failure == "questions":
        qs["q"]["instructions"] = "different"
    elif failure == "state":
        cases[0]["state"]["message"] = "different"
    elif failure == "duplicate":
        cases[1]["id"] = cases[0]["id"]
    else:
        cases[0]["expected"] = "unknown"
    with pytest.raises(ValueError):
        amended.quality(Agent(), cases, qs, identity, "shortlist", Embedder())


@pytest.mark.parametrize("order", [["0", "0"], ["0"], ["0", "outside"]])
def test_invalid_global_permutations_are_rejected(order):
    cases, qs, identity = data(count=2)
    with pytest.raises(ValueError, match="permutation"):
        amended.quality(Agent(), cases, qs, identity, "tournament", None, order=order)


def test_full_tournament_has_no_retrieval_rank():
    cases, qs, identity = data()
    result = amended.quality(Agent(), cases, qs, identity, "tournament", None)
    assert all(row["gold_rank"] is None for row in result["cases"])


def test_label_order_deterministic_and_stale_option_order_rejected():
    cases, qs, identity = data()
    assert amended.label_order(qs, 11) == amended.label_order(qs, 11)
    assert amended.label_order(qs, 11) != amended.label_order(qs, 29)
    qs["q"]["option_order"] = list(range(32))
    identity["questions_sha256"] = questions_fingerprint(Dataset([Example(cases[0]["state"], qs, {})]))
    with pytest.raises(ValueError, match="canonical criteria"):
        amended.quality(Agent(), cases, qs, identity, "tournament", None, order=amended.label_order(qs, 11))


def test_missing_shortlist_metadata_is_rejected_even_for_passthrough(monkeypatch):
    cases, qs, identity = data(count=4)
    original = amended.fixed.predict_strategy
    def omit(*args, **kwargs):
        result = original(*args, **kwargs)
        result.pop("shortlist")
        return result
    monkeypatch.setattr(amended.fixed, "predict_strategy", omit)
    with pytest.raises(ValueError, match="metadata"):
        amended.quality(Agent(), cases, qs, identity, "shortlist", Embedder(), k=8)


@pytest.mark.parametrize("counter", ["cpu_fallback_count", "_amp_failures"])
def test_transient_fallback_prevents_complete_quality_report(counter):
    cases, qs, identity = data()
    agent = Agent()
    original = agent.predict
    def fallback(*args, **kwargs):
        result = original(*args, **kwargs)
        setattr(agent, counter, 1)
        return result
    agent.predict = fallback
    with pytest.raises(RuntimeError, match="fallback counters"):
        amended.quality(agent, cases, qs, identity, "shortlist", Embedder())


def test_combined_catches_amp_failure_before_later_round_resets_counter():
    cases, qs, identity = data()
    agent = Agent()
    original = agent.predict
    def fallback(*args, **kwargs):
        result = original(*args, **kwargs)
        agent._amp_failures = 1 if len(agent.calls) == 1 else 0
        return result
    agent.predict = fallback
    with pytest.raises(RuntimeError, match="fallback counters"):
        amended.quality(agent, cases, qs, identity, "combined", Embedder())
    assert len(agent.calls) == 1


def test_actual_tensor_device_and_requested_device_are_checked():
    cases, qs, identity = data()
    with pytest.raises(ValueError, match="requested device"):
        amended.quality(Agent(), cases, qs, identity, "shortlist", Embedder(), requested_device="mps")
    agent = Agent()
    agent.model = agent.model.to("meta")
    with pytest.raises(ValueError, match="model tensors"):
        amended.quality(agent, cases, qs, identity, "shortlist", Embedder())


@pytest.mark.parametrize("strategy", ["builtin", "single", "wide", "shortlist-wide", "unknown"])
def test_quality_cli_rejects_unsupported_strategies_before_loading_models(monkeypatch, tmp_path, strategy):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid CLI must fail before model loading")
    monkeypatch.setattr(amended, "TextEncoder", forbidden)
    monkeypatch.setattr(sys, "argv", ["amended", "--split", "dev", "--limit", "2",
                                    "--strategies", strategy, "--out", str(tmp_path / "report.json")])
    with pytest.raises(SystemExit) as exc:
        amended.main()
    assert exc.value.code == 2
    assert not (tmp_path / "report.json").exists()


def test_diagnostic_trace_keeps_groups(monkeypatch):
    cases, qs, identity = data()
    monkeypatch.setattr(amended.fixed, "diagnose_calls", lambda *args: [{"offline": True}])
    result = amended.quality(Agent(), cases, qs, identity, "combined", Embedder(), diagnose=True)
    assert all(row["groups"] and row["calls"] == [{"offline": True}] for row in result["cases"])
