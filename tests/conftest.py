"""Shared fixtures. Nothing here loads model weights -- these tests run on CPU with no download."""
import pytest

from laya.types import Action, ChoiceAnswer, NoulAnswer, Response, ScoreAnswer, Usage


def choice(name, probs, conf=0.9, act=0.9):
    return ChoiceAnswer(
        choice=name, probabilities=probs, confidence=conf, action=Action(act_probability=act)
    )


def score(value, n_levels=4, conf=0.9, act=0.9):
    return ScoreAnswer(
        score=value,
        legend={str(i): "level %d" % i for i in range(n_levels)},
        probabilities={str(i): 1.0 / n_levels for i in range(n_levels)},
        confidence=conf,
        action=Action(act_probability=act),
    )


def noul(p, conf=None, act=0.9):
    return NoulAnswer(
        noul=p,
        confidence=conf if conf is not None else max(p, 1 - p),
        action=Action(act_probability=act),
    )


class FakeAgent:
    """Agent stub returning scripted answers, for testing compositions without a model."""

    def __init__(self, answers_for=None):
        self.answers_for = answers_for or (lambda state, questions: {})
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        answers = self.answers_for(state, questions)
        return Response(
            answers=answers,
            usage=Usage(input_tokens=10),
            request_id="req_test",
        )

    predict = system_one


@pytest.fixture
def fake_agent():
    return FakeAgent
