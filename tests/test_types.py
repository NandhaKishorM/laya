"""P1: typed answers expose attributes while staying dict-compatible for existing callers."""
import pytest

from laya.types import Action, ChoiceAnswer, NoulAnswer, Response, ScoreAnswer, Truncation, Usage


@pytest.fixture
def answer():
    return ChoiceAnswer(
        choice="billing",
        probabilities={"billing": 0.9, "sales": 0.1},
        confidence=0.94,
        action=Action(act_probability=0.88),
    )


class TestAttributeAccess:
    def test_attributes(self, answer):
        assert answer.choice == "billing"
        assert answer.confidence == 0.94
        assert answer.type == "choice"

    def test_nested_action(self, answer):
        assert answer.action.act_probability == 0.88

    def test_frozen(self, answer):
        with pytest.raises(Exception):
            answer.choice = "sales"


class TestDictCompatibility:
    """Code written against the old raw-dict return must keep working."""

    def test_index_access(self, answer):
        assert answer["choice"] == "billing"
        assert answer["confidence"] == 0.94
        assert answer["type"] == "choice"

    def test_nested_action_index(self, answer):
        assert answer["action"]["act_probability"] == 0.88

    def test_get(self, answer):
        assert answer.get("choice") == "billing"
        assert answer.get("nope", "dflt") == "dflt"

    def test_in_operator(self, answer):
        assert "choice" in answer and "nope" not in answer

    def test_unknown_key_raises_keyerror(self, answer):
        with pytest.raises(KeyError):
            answer["nope"]

    def test_double_star_unpacking(self, answer):
        assert {**answer}["choice"] == "billing"

    def test_to_dict_is_json_ready(self, answer):
        import json
        d = answer.to_dict()
        assert d["action"]["act_probability"] == 0.88
        assert json.loads(json.dumps(d))["choice"] == "billing"


class TestNoul:
    def test_is_true_above_half(self):
        assert NoulAnswer(noul=0.9, confidence=0.9, action=Action(0.5)).is_true

    def test_is_false_below_half(self):
        assert not NoulAnswer(noul=0.1, confidence=0.9, action=Action(0.5)).is_true

    def test_boundary_is_true(self):
        assert NoulAnswer(noul=0.5, confidence=0.5, action=Action(0.5)).is_true


class TestResponse:
    def test_defaults(self):
        r = Response(answers={}, usage=Usage(input_tokens=5))
        assert r.model == "laya-rl-agent"
        assert r.truncated == []
        assert r["usage"]["input_tokens"] == 5

    def test_independent_truncated_lists(self):
        a = Response(answers={}, usage=Usage(1))
        b = Response(answers={}, usage=Usage(1))
        a.truncated.append(Truncation("q", 1, 2))
        assert b.truncated == [], "default_factory must not share state"

    def test_dropped_tokens(self):
        assert Truncation("q", kept_tokens=100, state_tokens=350).dropped_tokens == 250

    def test_dropped_never_negative(self):
        assert Truncation("q", kept_tokens=10, state_tokens=5).dropped_tokens == 0


def test_score_answer_shape():
    s = ScoreAnswer(
        score=1.84, legend={"0": "calm"}, probabilities={"0": 1.0},
        confidence=0.7, action=Action(0.5),
    )
    assert s.score == 1.84 and s["legend"]["0"] == "calm" and s.type == "score"
