"""Weight-free checks for tournament stage attribution and report identity."""
import json

import pytest

from laya.evals import Dataset, EvalError, Example, evaluate
from laya.evals_tournament import TOURNAMENT_REPORT_SCHEMA, evaluate_tournament


WIDE = {"intent": {"type": "choice", "instructions": "Which?",
                   "criteria": ["a", "b", "c", "d", "e", "f"]}}
NARROW = {"intent": {"type": "choice", "instructions": "Which?", "criteria": ["a", "b"]}}


class ChoiceAgent:
    def __init__(self):
        self.received = []

    def system_one(self, state, questions):
        self.received.append((state, {qid: list(q["criteria"]) for qid, q in questions.items()}))
        answers = {}
        for qid, question in questions.items():
            labels = list(question["criteria"])
            if state == "early" and labels == ["a", "b"]:
                choice = "a"
            elif state == "late" and labels == ["c", "e"]:
                choice = "e"
            elif (state == "final" and labels == ["a", "c"]) or (state == "pass" and labels == ["a", "b"]):
                choice = labels[-1]
            else:
                choice = labels[0]
            answers[qid] = {"type": "choice", "choice": choice, "answer_confidence": 0.7}
        return {"model": "stub", "answers": answers}


def row(state, gold, *, narrow=False, tag="test"):
    return Example(state, NARROW if narrow else WIDE, {"intent": gold}, tags=(tag,))


def run(dataset, group_size=2):
    return evaluate_tournament(ChoiceAgent(), dataset, group_size=group_size,
                               checkpoint_id="checkpoint@123")


def test_stage_attribution_and_slices():
    dataset = Dataset([row("ok", "a", tag="kept"), row("final", "a", tag="kept"),
                       row("early", "b", tag="lost"), row("late", "c", tag="lost"),
                       row("pass", "a", narrow=True, tag="kept")])
    report = run(dataset)
    assert [case["tournament_status"] for case in report.cases] == [
        "correct", "final_choice_miss", "elimination_miss", "elimination_miss",
        "final_choice_miss"]
    assert [case["tournament"]["rounds"] for case in report.cases] == [2, 2, 2, 2, 0]
    assert report.overall["tournament_survival_rate"] == pytest.approx(3 / 5)
    assert report.overall["tournament_accuracy_on_survivors"] == pytest.approx(1 / 3)
    assert report.overall["choice_accuracy"] == pytest.approx(1 / 5)
    assert report.slices["tag"]["lost"]["tournament_survival_rate"] == 0.0
    assert "tournament_accuracy_on_survivors" not in report.slices["tag"]["lost"]
    assert report.config["schema"] == TOURNAMENT_REPORT_SCHEMA
    assert report.config["tournament"]["counts"] == {
        "correct": 1, "elimination_miss": 2, "final_choice_miss": 2}
    assert json.loads(json.dumps(report.to_json()))["cases"][3]["tournament_status"] == "elimination_miss"


def test_passthrough_and_group_size_are_reported_and_compared():
    dataset = Dataset([row("ok", "a", narrow=True)])
    report = run(dataset)
    assert report.cases[0]["tournament"]["labels"] == ["a", "b"]
    assert report.cases[0]["tournament"]["rounds"] == 0
    assert report.overall["tournament_survival_rate"] == 1.0
    same = run(dataset)
    assert report.comparable_to(same.to_json())[0]
    wider = run(dataset, group_size=3)
    ok, reasons = report.comparable_to(wider.to_json())
    assert not ok
    assert any("tournament group size" in reason for reason in reasons)
    missing = same.to_json()
    missing["config"].pop("tournament_group_size")
    assert not report.comparable_to(missing)[0]


def test_invalid_inputs_fail_before_inference():
    with pytest.raises(EvalError, match="not in criteria"):
        run(Dataset([row("ok", "missing")]))
    with pytest.raises(EvalError, match="choice question"):
        run(Dataset([Example("ok", {"intent": {"type": "score"}}, {"intent": 1})]))
    with pytest.raises(EvalError, match="at least 2"):
        run(Dataset([row("ok", "a")]), group_size=True)
    with pytest.raises(EvalError, match="checkpoint_id"):
        evaluate_tournament(ChoiceAgent(), Dataset([row("ok", "a")]), checkpoint_id=" ")


def test_a_row_built_in_python_names_the_row_instead_of_crashing():
    # Same on this door: `Example.from_dict` makes these three checks for a JSONL row and
    # `_validate_dataset` reads the same two fields as mappings, so a row built in Python died on
    # `.items()` / `.get()` as an unhandled AttributeError, and an `expected` key with no question
    # behind it was refused for the wrong reason ("tournament evaluation requires a choice question").
    for row_shape, fragment in ((Example("ok", WIDE, True), "'expected' must be an object"),
                                (Example("ok", ["intent"], {"intent": "a"}), "'questions' must be an object"),
                                (Example("ok", WIDE, {"nope": "a"}), "'expected' names unknown question")):
        agent = ChoiceAgent()
        with pytest.raises(EvalError, match=r"example 0 " + fragment):
            evaluate_tournament(agent, Dataset([row_shape]), group_size=2,
                                checkpoint_id="checkpoint@123")
        assert agent.received == [], row_shape


def test_malformed_tournament_metadata_fails(monkeypatch):
    def malformed(*_args, **_kwargs):
        return {"answers": {"intent": {"type": "choice", "choice": "a"}},
                "tournament": {"intent": {"labels": ["a", "a"], "n": 6, "rounds": 2}}}

    monkeypatch.setattr("laya.evals_tournament.predict_tournament", malformed)
    with pytest.raises(EvalError, match="inconsistent tournament metadata"):
        run(Dataset([row("ok", "a")]))


def test_explicit_model_reaches_every_round():
    class ModelAgent(ChoiceAgent):
        def predict(self, state, questions, model=None):
            assert model == "english"
            return self.system_one(state, questions)

    agent = ModelAgent()
    example = Example("ok", WIDE, {"intent": "a"}, model="english")
    report = evaluate_tournament(agent, Dataset([example]), group_size=2,
                                 checkpoint_id="checkpoint@123")
    assert len(agent.received) == 3  # two elimination rounds and the final call
    assert report.slices["model"]["english"]["choice_accuracy"] == 1.0


def test_multiple_questions_and_described_criteria():
    questions = {
        "route": {"type": "choice", "criteria": {label: label.upper() for label in "abcdef"}},
        "fallback": NARROW["intent"],
    }
    report = run(Dataset([Example("ok", questions, {"route": "a", "fallback": "a"})]))
    assert [case["qid"] for case in report.cases] == ["route", "fallback"]
    assert [case["tournament"]["rounds"] for case in report.cases] == [2, 0]
    assert report.slices["qid"]["route"]["tournament_survival_rate"] == 1.0
    assert report.slices["qid"]["fallback"]["choice_accuracy"] == 1.0


def test_dataset_hash_and_plain_evaluate(tmp_path):
    path = tmp_path / "intents.jsonl"
    path.write_text(json.dumps({"state": "ok", "questions": WIDE,
                                "expected": {"intent": "a"}}) + "\n", encoding="utf-8")
    dataset = Dataset.from_jsonl(str(path))
    report = evaluate_tournament(ChoiceAgent(), dataset, group_size=2,
                                 checkpoint_id="checkpoint@123", dataset_path=str(path))
    from laya.evals import file_fingerprint

    assert report.config["dataset_sha256"] == file_fingerprint(str(path))
    assert report.config["questions_sha256"]

    class PlainRunner:
        def predict(self, state, questions, model=None):
            return {"answers": {"intent": {"type": "choice", "choice": "a"}}}

    plain = evaluate(PlainRunner(), Dataset([row("ok", "a", narrow=True)]))
    assert plain.config["schema"] == "laya-evals-report/1"
    assert "tournament_survival_rate" not in plain.overall
    assert "tournament" not in plain.cases[0]
