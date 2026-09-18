"""P2: pattern helpers, tested against a stub agent -- no weights required."""
import pytest

from conftest import FakeAgent, choice, noul, score
from laya import patterns
from laya.errors import QuestionError

CHOICE_Q = {"type": "choice", "instructions": "which team?", "criteria": {"a": None, "b": None}}
SCORE_Q = {"type": "score", "instructions": "how urgent?", "criteria": ["low", "mid", "high", "max"]}
NOUL_Q = {"type": "noul", "instructions": "is it spam?"}


class TestConfidenceGate:
    def test_splits_on_threshold(self):
        agent = FakeAgent(lambda s, q: {"hi": noul(0.99, conf=0.95), "lo": noul(0.5, conf=0.4)})
        out = patterns.confidence_gate(agent, "s", {"hi": NOUL_Q, "lo": NOUL_Q}, threshold=0.85)
        assert set(out["automatic"]) == {"hi"}
        assert set(out["escalate"]) == {"lo"}

    def test_boundary_is_automatic(self):
        agent = FakeAgent(lambda s, q: {"x": noul(0.9, conf=0.85)})
        out = patterns.confidence_gate(agent, "s", {"x": NOUL_Q}, threshold=0.85)
        assert "x" in out["automatic"]

    def test_rejects_bad_threshold(self):
        with pytest.raises(ValueError):
            patterns.confidence_gate(FakeAgent(), "s", {"x": NOUL_Q}, threshold=1.5)


class TestRoute:
    def test_dispatches_to_winner(self):
        agent = FakeAgent(lambda s, q: {"route": choice("billing", {"billing": 0.9})})
        assert patterns.route(agent, "s", CHOICE_Q, {"billing": lambda a: "went-billing"}) == "went-billing"

    def test_falls_back_when_unhandled(self):
        agent = FakeAgent(lambda s, q: {"route": choice("other", {"other": 0.9})})
        assert patterns.route(agent, "s", CHOICE_Q, {"billing": lambda a: "b"},
                              default=lambda a: "fallback") == "fallback"

    def test_low_confidence_uses_default(self):
        agent = FakeAgent(lambda s, q: {"route": choice("billing", {"billing": 0.5}, conf=0.3)})
        out = patterns.route(agent, "s", CHOICE_Q, {"billing": lambda a: "b"},
                             default=lambda a: "escalated", min_confidence=0.8)
        assert out == "escalated"

    def test_rejects_non_choice(self):
        with pytest.raises(QuestionError):
            patterns.route(FakeAgent(), "s", NOUL_Q, {})


class TestCompositeScore:
    def test_normalizes_to_unit_range(self):
        agent = FakeAgent(lambda s, q: {"a": noul(1.0), "b": score(3.0, n_levels=4)})
        out = patterns.composite_score(agent, "s", {"a": NOUL_Q, "b": SCORE_Q})
        assert out["score"] == pytest.approx(1.0)

    def test_weights_shift_the_result(self):
        agent = FakeAgent(lambda s, q: {"a": noul(1.0), "b": noul(0.0)})
        hi = patterns.composite_score(agent, "s", {"a": NOUL_Q, "b": NOUL_Q}, weights={"a": 3, "b": 1})
        assert hi["score"] == pytest.approx(0.75)

    def test_choice_answers_are_skipped(self):
        agent = FakeAgent(lambda s, q: {"c": choice("x", {"x": 1.0}), "n": noul(1.0)})
        out = patterns.composite_score(agent, "s", {"c": CHOICE_Q, "n": NOUL_Q})
        assert "c" not in out["components"] and out["score"] == pytest.approx(1.0)

    def test_score_normalized_by_level_count(self):
        agent = FakeAgent(lambda s, q: {"b": score(2.0, n_levels=5)})
        out = patterns.composite_score(agent, "s", {"b": SCORE_Q})
        assert out["score"] == pytest.approx(0.5)


class TestSelfConsistency:
    def test_unanimous_agreement(self):
        agent = FakeAgent(lambda s, q: {k: noul(0.9) for k in q})
        out = patterns.self_consistency(agent, "s", NOUL_Q, ["a?", "b?", "c?"])
        assert out["agreement"] == 1.0 and not out["needs_review"]

    def test_disagreement_flags_review(self):
        vals = iter([0.9, 0.1, 0.9])
        agent = FakeAgent(lambda s, q: {k: noul(next(vals)) for k in q})
        out = patterns.self_consistency(agent, "s", NOUL_Q, ["a?", "b?", "c?"])
        assert out["agreement"] == pytest.approx(2 / 3, abs=1e-4)
        assert out["needs_review"]

    def test_runs_one_question_per_variant(self):
        agent = FakeAgent(lambda s, q: {k: noul(0.9) for k in q})
        patterns.self_consistency(agent, "s", NOUL_Q, ["a?", "b?", "c?"])
        assert len(agent.calls[0][1]) == 3

    def test_requires_variants(self):
        with pytest.raises(ValueError):
            patterns.self_consistency(FakeAgent(), "s", NOUL_Q, [])


class TestRerank:
    def test_sorts_best_first(self):
        scores = {"a": 0.2, "b": 0.9, "c": 0.5}
        agent = FakeAgent(lambda s, q: {"rank": noul(scores[s])})
        out = patterns.rerank(agent, ["a", "b", "c"], NOUL_Q)
        assert [c for c, _ in out] == ["b", "c", "a"]

    def test_top_k(self):
        scores = {"a": 0.2, "b": 0.9, "c": 0.5}
        agent = FakeAgent(lambda s, q: {"rank": noul(scores[s])})
        assert len(patterns.rerank(agent, ["a", "b", "c"], NOUL_Q, top_k=2)) == 2

    def test_state_fn_maps_candidates(self):
        agent = FakeAgent(lambda s, q: {"rank": noul(len(s) / 10)})
        out = patterns.rerank(agent, [{"t": "xx"}, {"t": "xxxxx"}], NOUL_Q, state_fn=lambda c: c["t"])
        assert out[0][0]["t"] == "xxxxx"

    def test_rejects_choice(self):
        with pytest.raises(QuestionError):
            patterns.rerank(FakeAgent(), ["a"], CHOICE_Q)


class TestCascade:
    def test_stops_when_gate_fails(self):
        agent = FakeAgent(lambda s, q: {k: noul(0.0) for k in q})
        out = patterns.cascade(agent, "s", [
            {"questions": {"cheap": NOUL_Q}, "gate": lambda a: a["cheap"]["noul"] > 0.5},
            {"questions": {"expensive": NOUL_Q}},
        ])
        assert out["stopped_at"] == 0 and not out["completed"]
        assert "expensive" not in out["answers"]
        assert len(agent.calls) == 1, "later stage must not run"

    def test_runs_all_when_gates_pass(self):
        agent = FakeAgent(lambda s, q: {k: noul(1.0) for k in q})
        out = patterns.cascade(agent, "s", [
            {"questions": {"cheap": NOUL_Q}, "gate": lambda a: a["cheap"]["noul"] > 0.5},
            {"questions": {"expensive": NOUL_Q}},
        ])
        assert out["completed"] and set(out["answers"]) == {"cheap", "expensive"}

    def test_rejects_stage_without_questions(self):
        with pytest.raises(QuestionError):
            patterns.cascade(FakeAgent(), "s", [{"gate": lambda a: True}])


def test_fan_out_one_call_per_state():
    agent = FakeAgent(lambda s, q: {"x": noul(0.5)})
    out = patterns.fan_out(agent, ["a", "b", "c"], {"x": NOUL_Q})
    assert len(out) == 3 and len(agent.calls) == 3
