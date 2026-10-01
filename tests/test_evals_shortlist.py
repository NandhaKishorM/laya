"""Weight-free checks for two-stage shortlist evaluation."""
import json

import pytest

from laya.evals import Dataset, EvalError, Example, evaluate
from laya.evals_shortlist import SHORTLIST_REPORT_SCHEMA, evaluate_shortlist


QUESTION = {"intent": {"type": "choice", "instructions": "Which?",
                       "criteria": ["a", "b", "c"]}}
VECTORS = {
    "Which?\none": [1.0, 0.0],
    "Which?\ntwo": [1.0, 0.0],
    "Which?\nthree": [-1.0, 0.0],
    "a": [1.0, 0.0],
    "b": [0.0, 1.0],
    "c": [-1.0, 0.0],
}


class TableEmbed:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [VECTORS[text] for text in texts]


class ChoiceAgent:
    def __init__(self, chosen):
        self.chosen = chosen
        self.received = []

    def system_one(self, state, questions):
        labels = list(questions["intent"]["criteria"])
        self.received.append(labels)
        return {"model": "stub", "answers": {"intent": {
            "type": "choice", "choice": self.chosen[state],
            "answer_confidence": 0.7,
        }}}


def sample(state, gold, tag):
    return Example(state, QUESTION, {"intent": gold}, tags=(tag,))


def run(agent, dataset, embed, k=2):
    return evaluate_shortlist(agent, dataset, embed, k=k,
                              checkpoint_id="checkpoint@123", embedder_id="embedder@456")


def test_stage_attribution_and_slices():
    agent = ChoiceAgent({"one": "a", "two": "a", "three": "c"})
    dataset = Dataset([sample("one", "a", "ok"), sample("two", "b", "bad"),
                       sample("three", "a", "bad")])
    report = run(agent, dataset, TableEmbed())
    assert [case["shortlist_status"] for case in report.cases] == [
        "correct", "decision_miss", "retrieval_miss"]
    assert report.overall["shortlist_recall_at_k"] == pytest.approx(2 / 3)
    assert report.overall["shortlist_accuracy_on_recalled"] == pytest.approx(1 / 2)
    assert report.overall["choice_accuracy"] == pytest.approx(1 / 3)
    assert report.slices["tag"]["bad"]["shortlist_recall_at_k"] == 0.5
    assert report.slices["tag"]["bad"]["shortlist_accuracy_on_recalled"] == 0.0
    assert report.config["shortlist"]["counts"] == {
        "correct": 1, "retrieval_miss": 1, "decision_miss": 1}
    assert report.config["schema"] == SHORTLIST_REPORT_SCHEMA
    assert report.config["shortlist"]["k"] == 2
    assert json.loads(json.dumps(report.to_json()))["cases"][2]["shortlist_status"] == "retrieval_miss"
    assert agent.received == [["a", "b"], ["a", "b"], ["c", "b"]]


def test_zero_recall_has_no_conditional_accuracy():
    report = run(ChoiceAgent({"three": "c"}), Dataset([sample("three", "a", "bad")]),
                 TableEmbed(), k=1)
    assert report.overall["shortlist_recall_at_k"] == 0.0
    assert "shortlist_accuracy_on_recalled" not in report.overall
    assert "shortlist_accuracy_on_recalled" not in report.slices["tag"]["bad"]
    assert report.overall["choice_accuracy"] == 0.0


def test_passthrough_skips_embedding():
    def no_embedding(_texts):
        raise AssertionError("k >= n should not embed")

    report = run(ChoiceAgent({"one": "a"}), Dataset([sample("one", "a", "ok")]),
                 no_embedding, k=3)
    assert report.overall["shortlist_recall_at_k"] == 1.0
    assert report.cases[0]["shortlist"]["passthrough"] is True
    assert report.cases[0]["shortlist"]["scores"] is None


def test_invalid_gold_and_out_of_shortlist_answer_fail():
    with pytest.raises(EvalError, match="not in criteria"):
        run(ChoiceAgent({}), Dataset([sample("one", "missing", "bad")]), TableEmbed())
    with pytest.raises(EvalError, match="outside the shortlist"):
        run(ChoiceAgent({"one": "c"}), Dataset([sample("one", "a", "bad")]), TableEmbed())
    with pytest.raises(EvalError, match="positive integer"):
        run(ChoiceAgent({}), Dataset([sample("one", "a", "ok")]), TableEmbed(), k=True)


def test_malformed_shortlist_metadata_fails(monkeypatch):
    def malformed(*_args, **_kwargs):
        return {"answers": {"intent": {"type": "choice", "choice": "a"}},
                "shortlist": {"intent": {"labels": ["a", "a"], "k": 2, "n": 3,
                                         "passthrough": False, "scores": [1.0, 1.0]}}}

    monkeypatch.setattr("laya.evals_shortlist.predict_shortlist", malformed)
    with pytest.raises(EvalError, match="inconsistent shortlist metadata"):
        run(ChoiceAgent({"one": "a"}), Dataset([sample("one", "a", "ok")]), TableEmbed())


def test_dataset_bytes_are_recorded_when_path_is_supplied(tmp_path):
    path = tmp_path / "intents.jsonl"
    path.write_text(json.dumps({"state": "one", "questions": QUESTION,
                                "expected": {"intent": "a"}}) + "\n", encoding="utf-8")
    dataset = Dataset.from_jsonl(str(path))
    report = evaluate_shortlist(ChoiceAgent({"one": "a"}), dataset, TableEmbed(), k=2,
                                checkpoint_id="checkpoint@123", embedder_id="embedder@456",
                                dataset_path=str(path))
    from laya.evals import file_fingerprint

    assert report.config["dataset_sha256"] == file_fingerprint(str(path))
    assert report.config["questions_sha256"]


def test_plain_evaluate_keeps_its_schema_and_metrics():
    class Runner:
        def predict(self, state, questions, model=None):
            return {"answers": {"intent": {"type": "choice", "choice": "a"}}}

    report = evaluate(Runner(), Dataset([sample("one", "a", "ok")]))
    assert report.config["schema"] == "laya-evals-report/1"
    assert report.overall["choice_accuracy"] == 1.0
    assert "shortlist_recall_at_k" not in report.overall
    assert "shortlist" not in report.cases[0]
