"""End-to-end tests against the real checkpoint.

These download weights (~600MB) and run on CPU, so they are opt-in:

    LAYA_INTEGRATION=1 pytest tests/test_integration.py

Everything else in the suite runs without a model.
"""
import os

import pytest

import laya
from laya.errors import OptionsTooLongError, QuestionError
from laya.types import ChoiceAnswer, NoulAnswer, Response, ScoreAnswer

pytestmark = pytest.mark.skipif(
    not os.environ.get("LAYA_INTEGRATION"),
    reason="set LAYA_INTEGRATION=1 to run tests that download the model",
)

MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")


@pytest.fixture(scope="module")
def agent():
    return laya.Agent(MODEL, device=os.environ.get("LAYA_DEVICE", "cpu"))


QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {"billing": "invoices and refunds", "technical": "bugs and outages"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["no rush", "soon", "blocking"],
    },
    "needs_reply": {"type": "noul", "instructions": "Does the sender expect a reply?"},
}

STATE = "My invoice was charged twice this month and I need the duplicate refunded today."


def test_returns_typed_answers(agent):
    res = agent.system_one(STATE, QUESTIONS)
    assert isinstance(res, Response)
    assert isinstance(res.answers["department"], ChoiceAnswer)
    assert isinstance(res.answers["urgency"], ScoreAnswer)
    assert isinstance(res.answers["needs_reply"], NoulAnswer)


def test_dict_access_still_works(agent):
    """The pre-0.2 calling convention used by the notebooks."""
    result = agent.predict(STATE, QUESTIONS)
    answers = result["answers"]
    assert answers["department"]["choice"] in ("billing", "technical")
    assert isinstance(result["usage"]["input_tokens"], int)


def test_request_id_present(agent):
    assert agent.system_one(STATE, QUESTIONS).request_id.startswith("req_")


def test_probabilities_are_normalized(agent):
    res = agent.system_one(STATE, QUESTIONS)
    assert sum(res.answers["department"].probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= res.answers["needs_reply"].noul <= 1.0
    assert 0.0 <= res.answers["department"].confidence <= 1.0


def test_structured_criteria_run_end_to_end(agent):
    """P0 #1: the path that used to crash with TypeError."""
    questions = {
        "cat": {
            "type": "choice",
            "instructions": "Classify this message.",
            "criteria": {
                "billing": {"what": "invoices, refunds", "not_for": ["outages"]},
                "technical": {"what": "bugs, outages", "examples": ["500 error"]},
            },
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Is this urgent?",
            "criteria": {"true": {"what": "same-day"}, "false": {"what": "can wait"}},
        },
    }
    res = agent.system_one(STATE, questions)
    assert res.answers["cat"].choice in ("billing", "technical")
    assert 0.0 <= res.answers["is_urgent"].noul <= 1.0


def test_truncation_is_reported(agent):
    """P0 #2: overlong state is recorded rather than silently dropped."""
    long_state = "word " * 20000
    res = agent.system_one(long_state, {"q": QUESTIONS["needs_reply"]})
    assert res.truncated, "long state must be reported as truncated"
    t = res.truncated[0]
    assert t.question_id == "q" and t.dropped_tokens > 0


def test_truncate_left_keeps_the_tail(agent):
    """The needle is at the end, so only left-truncation can see it."""
    tail_agent = laya.Agent(MODEL, device="cpu", truncate_left=True)
    needle = " The customer explicitly demands an immediate refund."
    state = ("filler text. " * 5000) + needle
    left = tail_agent.system_one(state, {"q": QUESTIONS["needs_reply"]})
    right = agent.system_one(state, {"q": QUESTIONS["needs_reply"]})
    assert left.answers["q"].noul != right.answers["q"].noul


def test_too_many_options_raises_typed_error(agent):
    huge = {"option_number_%03d" % i: "a fairly long description " * 4 for i in range(400)}
    with pytest.raises(OptionsTooLongError, match="do not fit"):
        agent.system_one(STATE, {"q": {"type": "choice", "instructions": "pick", "criteria": huge}})


def test_malformed_question_raises_typed_error(agent):
    with pytest.raises(QuestionError):
        agent.system_one(STATE, {"q": {"type": "nonsense", "instructions": "x"}})


def test_patterns_work_against_the_real_model(agent):
    out = laya.patterns.confidence_gate(agent, STATE, QUESTIONS, threshold=0.5)
    assert len(out["automatic"]) + len(out["escalate"]) == len(QUESTIONS)


def test_two_stage_beats_the_single_head_ceiling(agent):
    """P0 #3: 300 options cannot fit one head, but two-stage handles them."""
    options = {"category_%03d" % i: "handles case %d" % i for i in range(300)}
    with pytest.raises(OptionsTooLongError):
        agent.system_one(STATE, {"q": {"type": "choice", "instructions": "pick", "criteria": options}})
    out = laya.patterns.two_stage_choice(agent, STATE, "pick", options, group_size=16)
    assert out["choice"] in options
    assert 0.0 <= out["confidence"] <= 1.0


def test_async_matches_sync(agent):
    import asyncio

    from laya.aio import AsyncAgent

    sync = agent.system_one(STATE, QUESTIONS)
    out = asyncio.run(AsyncAgent(agent).system_one(STATE, QUESTIONS))
    assert out.answers["department"].choice == sync.answers["department"].choice


def test_async_batch(agent):
    import asyncio

    from laya.aio import AsyncAgent

    states = [STATE, "The server returns a 500 error on every login attempt."]
    out = asyncio.run(AsyncAgent(agent).batch(states, QUESTIONS, max_concurrency=2))
    assert len(out) == 2
    assert all(isinstance(r, Response) for r in out)


def test_select_tool_end_to_end(agent):
    tools = {
        "run_sql_query": "run a SQL query against the analytics warehouse",
        "send_email": "compose and send an email to a customer",
        "search_docs": "search the internal documentation",
    }
    out = laya.patterns.select_tool(agent, "How many users signed up last week?", tools)
    assert out["tool"] in list(tools) + [None]
    assert 0.0 <= out["confidence"] <= 1.0


def test_select_tool_can_decline(agent):
    tools = {"run_sql_query": "run a SQL query", "send_email": "send an email"}
    out = laya.patterns.select_tool(agent, "hello there, how are you today?", tools)
    assert out["tool"] is None or out["tool"] in tools


def test_check_citation_distinguishes_support_from_contradiction(agent):
    supported = laya.patterns.check_citation(
        agent,
        "The service had an outage on Tuesday.",
        "On Tuesday the service was unavailable for three hours due to an outage.",
    )
    unrelated = laya.patterns.check_citation(
        agent,
        "The service had an outage on Tuesday.",
        "The cafeteria introduced a new lunch menu this week.",
    )
    assert supported["verdict"] in ("supported", "unsupported", "contradicted")
    assert 0.0 <= supported["supported"] <= 1.0
    assert 0.0 <= unrelated["supported"] <= 1.0
    # the supporting source should score at least as high as the unrelated one
    assert supported["supported"] >= unrelated["supported"]


def test_check_citations_batch(agent):
    pairs = [
        ("The sky is blue.", "The sky appears blue on a clear day."),
        ("Revenue doubled.", "The cafeteria menu changed on Monday."),
    ]
    out = laya.patterns.check_citations(agent, pairs)
    assert out["n_checked"] == 2
    assert isinstance(out["unsupported"], list)
