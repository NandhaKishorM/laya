"""Decision outcome log: SQLite records for offline per-question calibration.

Weight-free throughout: hand-built `PredictContext` objects for the unit shape, and a
`Router` with an attached stub agent for the installed-hook path, so nothing here loads
a checkpoint or touches the Hub.
"""
import json
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import OutcomeLogHook, Router  # noqa: E402
from laya.hooks import PredictContext  # noqa: E402
from laya.outcome_log import _canonical_state_hash  # noqa: E402

QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": None, "tech": None}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["calm", "firm", "angry"]},
    "threat": {"type": "noul", "instructions": "Is it a threat?"},
}

ANSWERS = {
    "dept": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.94, "tech": 0.06}, "confidence": 0.94},
    "urgency": {"type": "score", "score": 1.7,
                "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
                "confidence": 0.5, "legend": {"0": "calm", "1": "firm", "2": "angry"}},
    "threat": {"type": "noul", "noul": 0.2, "confidence": 0.8},
}

RESULT = {"model": "laya-rl-agent", "answers": dict(ANSWERS),
          "usage": {"input_tokens": 42, "output_tokens": 0}}


def _ctx(**overrides):
    base = {"states": ["billed twice"], "questions": dict(QUESTIONS),
            "results": [dict(RESULT)], "decision": {"model": "english"},
            "model": "english", "elapsed_ms": 12.5, "error": None,
            "usage": {"input_tokens": 42}}
    base.update(overrides)
    return PredictContext(**{k: v for k, v in base.items()
                             if k in PredictContext.__dataclass_fields__})


class StubAgent:
    """Answers every question with the fixed payload above, recording the call."""

    def __init__(self):
        self.calls = []

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        self.calls.append((list(states), questions))
        if "boom" in states:
            raise RuntimeError("inference failed")
        return [{"model": "fake", "answers": dict(ANSWERS),
                 "usage": {"input_tokens": 42, "output_tokens": 0}} for _ in states]

    def system_one(self, state, questions, **overrides):
        return self.predict_batch([state], questions, **overrides)[0]


def test_logs_one_row_per_question_with_values():
    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_ctx(run_id="r1"))
        rows = hook.decisions()
    finally:
        hook.close()
    assert len(rows) == 3
    assert {row["qid"] for row in rows} == set(QUESTIONS)
    assert {row["run_id"] for row in rows} == {"r1"}
    by_qid = {row["qid"]: row for row in rows}
    assert by_qid["dept"]["value_text"] == "billing"
    assert by_qid["dept"]["value_real"] is None
    assert by_qid["urgency"]["value_real"] == 1.7
    assert by_qid["threat"]["value_real"] == 0.2
    assert by_qid["dept"]["confidence"] == 0.94
    assert by_qid["threat"]["qtype"] == "noul"
    assert by_qid["dept"]["model"] == "english"
    assert by_qid["dept"]["elapsed_ms"] == 12.5
    assert all(row["error"] is None for row in rows)
    assert json.loads(by_qid["urgency"]["answer_json"])["legend"]["1"] == "firm"


def test_raw_state_never_stored_by_default():
    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_ctx(run_id="r1"))
        rows = hook.decisions()
    finally:
        hook.close()
    assert rows and all(row["state_json"] is None for row in rows)
    assert {row["state_hash"] for row in rows} == {_canonical_state_hash("billed twice")}
    blob = " ".join(json.dumps(row, default=str) for row in rows)
    assert "billed twice" not in blob


def test_include_state_is_an_explicit_opt_in():
    hook = OutcomeLogHook(":memory:", include_state=True)
    try:
        hook.on_predict_end(_ctx(run_id="r1"))
        rows = hook.decisions()
    finally:
        hook.close()
    assert {row["state_json"] for row in rows} == {'"billed twice"'}


def test_failed_call_logs_an_error_row_not_answers():
    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_ctx(run_id="r9", results=None, error=RuntimeError("boom")))
        rows = hook.decisions()
    finally:
        hook.close()
    assert len(rows) == 1
    assert rows[0]["qid"] is None and rows[0]["error"] == "boom"


def test_hook_never_breaks_inference():
    """A closed database costs a warning; the decision still goes through."""
    router = Router()
    router.attach("english", StubAgent())
    hook = OutcomeLogHook(":memory:")
    hook.close()
    with pytest.warns(RuntimeWarning, match="OutcomeLogHook failed"):
        out = router.predict("billed twice", dict(QUESTIONS), model="english",
                             hooks=[hook])
    assert out["answers"]["dept"]["choice"] == "billing"


def test_installed_hook_logs_router_calls_end_to_end():
    router = Router()
    stub = StubAgent()
    router.attach("english", stub)
    hook = OutcomeLogHook(":memory:")
    try:
        router.predict("billed twice", dict(QUESTIONS), model="english", hooks=[hook])
        router.predict("billed twice", dict(QUESTIONS), model="english", hooks=[hook])
        rows = hook.decisions()
    finally:
        hook.close()
    assert len(rows) == 6
    assert len({row["run_id"] for row in rows}) == 2


def test_batch_logs_every_state_with_its_own_hash():
    router = Router()
    router.attach("english", StubAgent())
    hook = OutcomeLogHook(":memory:")
    try:
        router.predict_batch([{"state": "one", "questions": dict(QUESTIONS)},
                              {"state": "two", "questions": dict(QUESTIONS)}],
                             hooks=[hook])
        rows = hook.decisions()
    finally:
        hook.close()
    assert len(rows) == 6
    assert {row["state_hash"] for row in rows} == {_canonical_state_hash("one"),
                                                  _canonical_state_hash("two")}


def test_record_outcome_replaces_by_provenance():
    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_ctx(run_id="r1"))
        hook.record_outcome("r1", "dept", "billing", source="review")
        assert hook.outcomes()[0]["weight"] == 1.0
        hook.record_outcome("r1", "dept", {"correct": False}, source="shadow-eval",
                            weight=0.5)
        assert hook.outcomes() == [{"run_id": "r1", "qid": "dept",
                                    "label_json": '{"correct": false}',
                                    "source": "shadow-eval", "weight": 0.5,
                                    "ts": hook.outcomes()[0]["ts"]}]
    finally:
        hook.close()


def test_record_outcome_rejects_a_non_finite_weight():
    hook = OutcomeLogHook(":memory:")
    try:
        for bad in (float("nan"), float("inf"), "heavy", True, None):
            with pytest.raises(ValueError, match="weight must be a finite number"):
                hook.record_outcome("r1", "dept", "billing", source="review", weight=bad)
        assert hook.outcomes() == []
    finally:
        hook.close()


def test_unknown_answer_shapes_log_nothing_not_errors():
    hook = OutcomeLogHook(":memory:")
    try:
        weird_questions = {"odd": {"type": "noul", "instructions": "?"},
                           "missing": {"type": "noul", "instructions": "?"}}
        weird = dict(RESULT)
        weird["answers"] = {"odd": {"type": "weird", "payload": object()},
                            "missing": None}
        with warnings.catch_warnings(record=True) as warnings_made:
            warnings.simplefilter("always")
            hook.on_predict_end(_ctx(run_id="r1", questions=weird_questions,
                                     results=[weird]))
        assert [w for w in warnings_made if issubclass(w.category, RuntimeWarning)] == []
        rows = hook.decisions()
    finally:
        hook.close()
    assert {row["qid"] for row in rows} == {"odd", "missing"}
    assert all(row["value_text"] is None and row["value_real"] is None
               and row["confidence"] is None for row in rows)


# ------------------------------------------------- slice 2: weak labels ----------

def _stability_answer(choice, stability):
    """A choice answer carrying the block an annotating hook would have attached."""
    return {"type": "choice", "choice": choice,
            "probabilities": {choice: 1.0}, "confidence": 0.9,
            "reliability": {"stability": stability, "decision": "ACCEPT",
                            "probes": ["rename", "reorder"]}}


def _stability_ctx(run_id="r1"):
    questions = {"a": {"type": "choice", "instructions": "?",
                       "criteria": {"x": None, "y": None}},
                 "b": {"type": "choice", "instructions": "?",
                       "criteria": {"x": None, "y": None}},
                 "c": {"type": "choice", "instructions": "?",
                       "criteria": {"x": None, "y": None}}}
    answers = {"a": _stability_answer("x", 1.0),
               "b": _stability_answer("x", 0.5),
               "c": {"type": "choice", "choice": "x",
                     "probabilities": {"x": 1.0}, "confidence": 0.9}}
    return _ctx(run_id=run_id, questions=questions,
                results=[{"model": "m", "answers": answers, "usage": {}}])


def test_stability_export_labels_only_unanimous_rows_by_default():
    from laya.outcome_labels import export_stability_labels

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_stability_ctx())
        assert export_stability_labels(hook) == 1
        assert hook.outcomes() == [{"run_id": "r1", "qid": "a",
                                    "label_json": '{"value": "x"}',
                                    "source": "stability", "weight": 1.0,
                                    "ts": hook.outcomes()[0]["ts"]}]
    finally:
        hook.close()


def test_stability_export_follows_a_lowered_threshold():
    from laya.outcome_labels import export_stability_labels

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_stability_ctx())
        assert export_stability_labels(hook, min_stability=0.5) == 2
        by_qid = {row["qid"]: row for row in hook.outcomes()}
        assert by_qid["b"]["weight"] == 0.5
        assert json.loads(by_qid["b"]["label_json"]) == {"value": "x"}
    finally:
        hook.close()


def test_stability_export_never_overwrites_an_existing_outcome():
    from laya.outcome_labels import export_stability_labels

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_stability_ctx())
        hook.record_outcome("r1", "a", "y", source="review")
        assert export_stability_labels(hook) == 0
        assert hook.outcomes() == [{"run_id": "r1", "qid": "a",
                                    "label_json": '"y"', "source": "review",
                                    "weight": 1.0, "ts": hook.outcomes()[0]["ts"]}]
    finally:
        hook.close()


def test_stability_export_validates_its_threshold():
    from laya.outcome_labels import export_stability_labels

    hook = OutcomeLogHook(":memory:")
    try:
        for bad in (float("nan"), True, "unanimous"):
            with pytest.raises(ValueError, match="min_stability must be a finite number"):
                export_stability_labels(hook, min_stability=bad)
    finally:
        hook.close()


def _gated_ctx(run_id="r2"):
    def gated(qid, confidence, abstention):
        return {"type": "noul", "noul": 0.4, "confidence": confidence,
                "abstention": abstention,
                "abstention_threshold": 0.9} if abstention is not None else \
            {"type": "noul", "noul": 0.4, "confidence": confidence}

    questions = {qid: {"type": "noul", "instructions": "?"}
                 for qid in ("abstained", "unevaluated", "weak", "passed", "confident")}
    answers = {"abstained": gated("abstained", 0.4, "abstained"),
               "unevaluated": gated("unevaluated", None, "unevaluated"),
               "weak": gated("weak", 0.3, "passed"),
               "passed": gated("passed", 0.95, "passed"),
               "confident": gated("confident", 0.95, None)}
    return _ctx(run_id=run_id, questions=questions,
                results=[{"model": "m", "answers": answers, "usage": {}}])


def test_review_queue_lists_everything_the_gate_did_not_pass():
    from laya.outcome_labels import export_review_queue

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_gated_ctx())
        queue = export_review_queue(hook, min_confidence=0.9)
        assert [(entry["qid"], entry["reason"]) for entry in queue] == [
            ("abstained", "abstained"), ("unevaluated", "unevaluated"),
            ("weak", "low_confidence")]
        assert all(entry["run_id"] == "r2" for entry in queue)
    finally:
        hook.close()


def test_review_queue_without_a_threshold_lists_only_gate_failures():
    from laya.outcome_labels import export_review_queue

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_gated_ctx())
        assert [(entry["qid"], entry["reason"]) for entry in export_review_queue(hook)] == [
            ("abstained", "abstained"), ("unevaluated", "unevaluated")]
    finally:
        hook.close()


def test_review_queue_skips_already_labeled_rows():
    from laya.outcome_labels import export_review_queue

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_gated_ctx())
        hook.record_outcome("r2", "abstained", "no", source="review")
        assert [entry["qid"] for entry in export_review_queue(hook)] == ["unevaluated"]
    finally:
        hook.close()


# ------------------------------------------------- slice 3: fit driver -----------

def test_build_fit_pairs_joins_labels_to_targets():
    from laya.outcome_fit import build_fit_pairs

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_ctx(run_id="r1"))
        hook.record_outcome("r1", "dept", "billing", source="review")
        hook.record_outcome("r1", "urgency", 1, source="review")
        hook.record_outcome("r1", "threat", 0.2, source="review")
        pairs, report = build_fit_pairs(hook, [{"state": "billed twice",
                                                "questions": dict(QUESTIONS)}])
    finally:
        hook.close()
    assert report["n_outcomes"] == 3 and report["n_pairs"] == 1
    assert all(v == 0 for v in report["skipped"].values())
    assert list(pairs) == ["english"]
    (state, questions, targets), = pairs["english"]
    assert state == "billed twice" and sorted(questions) == ["dept", "threat", "urgency"]
    assert targets["dept"] == [1.0, 0.0]
    assert targets["urgency"] == [0.0, 1.0, 0.0]
    assert targets["threat"] == [0.8, 0.2]


def test_build_fit_pairs_value_shape_and_correct_true():
    from laya.outcome_fit import build_fit_pairs

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_ctx(run_id="r1"))
        hook.record_outcome("r1", "dept", {"value": "tech"}, source="stability", weight=0.9)
        hook.record_outcome("r1", "urgency", {"correct": True}, source="review")
        pairs, report = build_fit_pairs(hook, [{"state": "billed twice",
                                                "questions": dict(QUESTIONS)}])
    finally:
        hook.close()
    # {"value": ...} unwraps; {"correct": true} on a score question has no level to
    # adopt (the logged value is a continuous prediction), so it is skipped, not guessed
    assert report["n_pairs"] == 1
    assert pairs["english"][0][2] == {"dept": [0.0, 1.0]}
    assert report["skipped"]["label_unusable"] == 1


def test_build_fit_pairs_correct_true_adopts_a_choice():
    from laya.outcome_fit import build_fit_pairs

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_ctx(run_id="r1"))
        hook.record_outcome("r1", "dept", {"correct": True}, source="review")
        hook.record_outcome("r1", "urgency", {"correct": False}, source="review")
        pairs, report = build_fit_pairs(hook, [{"state": "billed twice",
                                                "questions": dict(QUESTIONS)}])
    finally:
        hook.close()
    assert pairs["english"][0][2] == {"dept": [1.0, 0.0]}
    assert report["skipped"]["label_unusable"] == 1


def test_build_fit_pairs_filters_and_counts_skips():
    from laya.outcome_fit import build_fit_pairs

    hook = OutcomeLogHook(":memory:")
    hook.on_predict_end(_ctx(run_id="r1"))
    hook.record_outcome("r1", "dept", "billing", source="review")
    hook.record_outcome("r1", "urgency", {"value": "x"}, source="stability", weight=0.9)
    hook.record_outcome("r1", "threat", True, source="review")
    hook.record_outcome("ghost", "dept", "billing", source="review")
    items = [{"state": "billed twice", "questions": dict(QUESTIONS)}]
    pairs, report = build_fit_pairs(hook, items, min_weight=0.95)
    # the weak label falls below the floor; the rest are joined
    assert report["n_outcomes"] == 4 and report["n_pairs"] == 1
    assert report["skipped"]["weight_below_floor"] == 1
    assert report["skipped"]["no_decision_row"] == 1
    assert sorted(pairs["english"][0][2]) == ["dept", "threat"]
    pairs, report = build_fit_pairs(hook, items, source="stability", min_weight=0.0)
    # the only stability outcome names "x", which is no score level: skipped, not guessed
    assert pairs == {} and report["skipped"]["label_unusable"] == 1
    hook.close()


def test_build_fit_pairs_rejects_malformed_items_and_models():
    from laya.outcome_fit import build_fit_pairs

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_ctx(run_id="r1"))
        hook.record_outcome("r1", "dept", "billing", source="review")
        items = [{"state": "billed twice", "questions": dict(QUESTIONS)}]
        with pytest.raises(ValueError, match="item 1 must be"):
            build_fit_pairs(hook, [{"state": "ok", "questions": {}}, {"state": "bad"}])
        pairs, report = build_fit_pairs(hook, items, models="multilingual")
        assert pairs == {} and report["skipped"]["model_filtered"] == 1
        with pytest.raises(ValueError, match="min_weight must be a finite number"):
            build_fit_pairs(hook, items, min_weight=float("nan"))
    finally:
        hook.close()


def test_build_fit_pairs_groups_by_model():
    from laya.outcome_fit import build_fit_pairs

    hook = OutcomeLogHook(":memory:")
    try:
        hook.on_predict_end(_ctx(run_id="r1", model="english"))
        hook.on_predict_end(_ctx(run_id="r2", states=["Guten Tag"],
                                 model="multilingual"))
        hook.record_outcome("r1", "dept", "billing", source="review")
        hook.record_outcome("r2", "dept", "tech", source="review")
        pairs, report = build_fit_pairs(
            hook, [{"state": "billed twice", "questions": dict(QUESTIONS)},
                   {"state": "Guten Tag", "questions": dict(QUESTIONS)}])
    finally:
        hook.close()
    assert report["n_pairs"] == 2
    assert pairs["english"][0][2] == {"dept": [1.0, 0.0]}
    assert pairs["multilingual"][0][2] == {"dept": [0.0, 1.0]}


def _synthetic_choice_records(n=30, k=2, seed=3):
    rng = np.random.RandomState(seed)
    recs = []
    for _ in range(n):
        y = int(rng.randint(0, k))
        z = [-4.0] * k
        z[y if rng.rand() < 0.8 else (y + 1) % k] = 4.0
        t = [0.0] * k
        t[y] = 1.0
        recs.append((0, z, t, k))
    return recs


def test_fit_payload_from_records_fits_synthetic():
    from laya.outcome_fit import fit_payload_from_records, write_payload

    out = fit_payload_from_records(_synthetic_choice_records())
    assert out["n_records"] == 30
    assert len(out["temperature"]["temperature"]) == 3
    assert out["payload"]["version"] >= 1
    assert out["payload"]["temperature"] == out["temperature"]["temperature"]
    # 30 rows clear the type floor (10) but no 100-row abstention bucket: no gate
    assert out["abstention_thresholds"] == {}
    write_payload(out["payload"], "/tmp/opencode-test-payload.json")
    assert json.load(open("/tmp/opencode-test-payload.json"))["temperature"] \
        == out["payload"]["temperature"]


def test_fit_payload_rejects_empty_and_bad_error():
    from laya.outcome_fit import fit_payload_from_records

    with pytest.raises(ValueError, match="at least one record"):
        fit_payload_from_records([])
    with pytest.raises(ValueError, match="target_error"):
        fit_payload_from_records(_synthetic_choice_records(), target_error=1.5)


# ----------------------------------------------- slice 4: shadow gate -------

def test_shadow_evaluate_summarizes_records():
    from laya.outcome_shadow import evaluate_records

    rows, summary = evaluate_records(_synthetic_choice_records(n=200, seed=11),
                                     [1.0, 1.0, 1.0], {})
    assert summary["n"] == 200 and len(rows) == 200
    assert 0.0 < summary["accuracy"] < 1.0
    assert 0.0 <= summary["ece"] <= 1.0
    assert summary["coverage"] == 1.0
    assert summary["gated_error"] == pytest.approx(1.0 - summary["accuracy"])
    assert all(row["bucket"] == "choice:2" for row in rows)


def _overconfident_records(n=400, k=2, acc=0.6, mag=8.0, seed=21):
    import numpy as _np

    rng = _np.random.RandomState(seed)
    recs = []
    for _ in range(n):
        y = int(rng.randint(0, k))
        pred = y if rng.rand() < acc else (y + 1) % k
        z = [-mag] * k
        z[pred] = mag
        t = [0.0] * k
        t[y] = 1.0
        recs.append((0, z, t, k))
    return recs


def test_shadow_ship_accepts_a_calibration_improvement():
    from laya.outcome_shadow import compare_reports, evaluate_records, should_ship

    recs = _overconfident_records()
    _, before = evaluate_records(recs, [1.0, 1.0, 1.0], {})
    _, after = evaluate_records(recs, [8.0, 8.0, 8.0], {})
    assert after["ece"] < before["ece"] - 0.05
    assert compare_reports(before, after)["ece"] == pytest.approx(after["ece"] - before["ece"])
    ship, reasons = should_ship(before, after)
    assert ship, reasons
    assert all("ok" in reason for reason in reasons)


def test_shadow_ship_blocks_an_ece_regression():
    from laya.outcome_shadow import evaluate_records, should_ship

    recs = _overconfident_records()
    _, good = evaluate_records(recs, [8.0, 8.0, 8.0], {})
    _, bad = evaluate_records(recs, [1.0, 1.0, 1.0], {})
    ship, reasons = should_ship(good, bad)
    assert not ship
    assert any(reason.startswith("ece BLOCKED") for reason in reasons)


def test_shadow_gate_abstains_below_cut_and_fails_closed_when_empty():
    from laya.outcome_shadow import evaluate_records, should_ship

    recs = _synthetic_choice_records(n=50, seed=5)
    _, gated = evaluate_records(recs, [1.0, 1.0, 1.0], {}, min_confidence=1.0)
    assert gated["coverage"] == 0.0 and gated["gated_error"] is None
    _, full = evaluate_records(recs, [1.0, 1.0, 1.0], {})
    ship, reasons = should_ship(full, gated)
    assert not ship
    assert any("undefined" in reason for reason in reasons)


def test_shadow_validates_its_inputs():
    from laya.outcome_shadow import evaluate_records

    recs = _synthetic_choice_records(n=10)
    with pytest.raises(ValueError, match="at least one record"):
        evaluate_records([], [1.0, 1.0, 1.0], {})
    with pytest.raises(ValueError, match="one scale per question type"):
        evaluate_records(recs, [1.0, 1.0], {})
    with pytest.raises(ValueError, match=r"min_confidence must be a float"):
        evaluate_records(recs, [1.0, 1.0, 1.0], {}, min_confidence=True)
