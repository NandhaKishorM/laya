"""Schema-driven decisions: mapping, projection and rejections. No model, no download.

Run: python tests/test_structured.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya  # noqa: E402
from laya.structured import (  # noqa: E402
    SchemaError,
    answers_to_json,
    answer_to_pydantic,
    decide,
    plan_from_json_schema,
    questions_from_json_schema,
    questions_from_pydantic,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond):
    if cond:
        PASS.append(name)
    else:
        FAIL.append(name)


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
        return
    except BaseException as other:  # noqa: BLE001
        FAIL.append("%s: raised %r, want %r" % (name, other, exc))
        return
    FAIL.append("%s: did not raise %r" % (name, exc))


def plan_or_fail(name, model):
    """Plan a pydantic model, recording a FAIL instead of letting the suite die.

    A schema the planner rejects raises out of the module-level block below, which would end
    the run before it prints its summary. Returning None turns that into an ordinary failure
    so the counts stay readable.
    """
    try:
        return questions_from_pydantic(model)
    except SchemaError as exc:
        FAIL.append("%s: %s" % (name, exc))
        return None


SCHEMA = {
    "type": "object",
    "properties": {
        "department": {"type": "string", "enum": ["billing", "support", "sales"],
                       "description": "Which team?"},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"},
        "priority": {"enum": [1, 2, 3]},
    },
}


# --------------------------------------------------------------- mapping
questions = questions_from_json_schema(SCHEMA)
check("map/enum is a choice", questions["department"]["type"], "choice")
check("map/description becomes instructions", questions["department"]["instructions"], "Which team?")
check("map/enum labels", list(questions["department"]["criteria"]), ["billing", "support", "sales"])
check("map/bounded number is a score", questions["urgency"]["type"], "score")
check("map/score levels", questions["urgency"]["criteria"], ["0", "1", "2"])
check("map/boolean is noul", questions["needs_human"]["type"], "noul")
check("map/integer enum becomes a choice", questions["priority"]["type"], "choice")
check("map/integer enum labels are strings", list(questions["priority"]["criteria"]), ["1", "2", "3"])
check("map/plan has one field per property", len(plan_from_json_schema(SCHEMA)), 4)


# --------------------------------------------------------------- projection
ANSWERS = {
    "department": {"type": "choice", "choice": "billing", "confidence": 0.9,
                   "probabilities": {"billing": 0.9, "support": 0.1, "sales": 0.0}},
    "urgency": {"type": "score", "score": 1.2, "confidence": 0.5,
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "legend": {}},
    "needs_human": {"type": "noul", "noul": 0.8, "confidence": 0.8},
    "priority": {"type": "choice", "choice": "2", "confidence": 0.7,
                 "probabilities": {"1": 0.2, "2": 0.7, "3": 0.1}},
}
values = answers_to_json(ANSWERS, SCHEMA)
check("project/choice value", values["department"], "billing")
check("project/score is the argmax level", values["urgency"], 2)
check("project/noul is a bool", values["needs_human"], True)
check("project/integer enum keeps its type", values["priority"], 2)
check_true("project/integer enum is an int", isinstance(values["priority"], int))
check("project/false noul",
      answers_to_json({"x": {"type": "noul", "noul": 0.2}},
                      {"type": "object", "properties": {"x": {"type": "boolean"}}})["x"],
      False)


# --------------------------------------------------------------- rejections
def _bad(schema):
    return lambda: questions_from_json_schema(schema)


check_raises("reject/free string", SchemaError,
             _bad({"type": "object", "properties": {"a": {"type": "string"}}}))
check_raises("reject/array", SchemaError,
             _bad({"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string"}}}}))
check_raises("reject/nested object", SchemaError,
             _bad({"type": "object", "properties": {"a": {"type": "object", "properties": {}}}}))
check_raises("reject/$ref", SchemaError,
             _bad({"type": "object", "properties": {"a": {"$ref": "#/$defs/X"}}}))
check_raises("reject/unbounded number", SchemaError,
             _bad({"type": "object", "properties": {"a": {"type": "integer", "minimum": 0}}}))
check_raises("reject/score too wide", SchemaError,
             _bad({"type": "object", "properties": {"a": {"type": "integer", "minimum": 0, "maximum": 100}}}))
check_raises("reject/too many properties", SchemaError,
             _bad({"type": "object", "properties": {("p%d" % i): {"type": "boolean"} for i in range(33)}}))
check_raises("reject/too many options", SchemaError,
             _bad({"type": "object", "properties": {"a": {"type": "string", "enum": ["v%d" % i for i in range(33)]}}}))
check_raises("reject/non-object root", SchemaError, _bad({"type": "array"}))
check_raises("reject/empty properties", SchemaError, _bad({"type": "object", "properties": {}}))


# --------------------------------------------------------------- decide
class FakeRunner:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append({"state": state, "questions": questions, "kwargs": kwargs})
        return {"answers": self.answers, "usage": {"input_tokens": 1, "output_tokens": 0},
                "routing": {"model": "english"}}


runner = FakeRunner(ANSWERS)
out = decide(runner, "some state", schema=SCHEMA)
check("decide/values", out["department"], "billing")
check("decide/builds questions", runner.calls[0]["questions"]["department"]["type"], "choice")
check("decide/forwards state", runner.calls[0]["state"], "some state")

details = decide(runner, "some state", schema=SCHEMA, return_details=True)
check("decide/details confidence", details.confidence["department"], 0.9)
check("decide/details noul probabilities", details.probabilities["needs_human"], {"false": 0.2, "true": 0.8})
check("decide/details usage", details.usage, {"input_tokens": 1, "output_tokens": 0})
check("decide/details routing", details.routing, {"model": "english"})

runner = FakeRunner({"a": {"type": "noul", "noul": 0.9, "confidence": 0.9}})
out = decide(runner, "s", questions={"a": {"type": "noul", "instructions": "?"}})
check("decide/questions pass-through returns answers", out,
      {"a": {"type": "noul", "noul": 0.9, "confidence": 0.9}})

check_raises("decide/requires schema or questions", ValueError, lambda: decide(runner, "s"))
check_raises("decide/rejects both", ValueError, lambda: decide(runner, "s", schema=SCHEMA, questions={}))

# kwargs are forwarded to predict
runner = FakeRunner(ANSWERS)
decide(runner, "s", schema=SCHEMA, hooks_raise=False)
check("decide/forwards predict kwargs", runner.calls[0]["kwargs"], {"hooks_raise": False})


# --------------------------------------------------------------- pydantic (optional)
try:
    from typing import Literal, Optional, Union

    import pydantic

    class Ticket(pydantic.BaseModel):
        department: Literal["billing", "support", "sales"]
        urgency: Literal[0, 1, 2]
        needs_human: bool

    pq = questions_from_pydantic(Ticket)
    check("pydantic/choice", pq["department"]["type"], "choice")
    check("pydantic/int literal choice", pq["urgency"]["type"], "choice")
    check("pydantic/bool noul", pq["needs_human"]["type"], "noul")

    ticket = answer_to_pydantic(Ticket, {
        "department": {"type": "choice", "choice": "support", "confidence": 0.9, "probabilities": {}},
        "urgency": {"type": "choice", "choice": "1", "confidence": 0.6, "probabilities": {}},
        "needs_human": {"type": "noul", "noul": 0.7, "confidence": 0.7},
    })
    check("pydantic/instance", (ticket.department, ticket.urgency, ticket.needs_human),
          ("support", 1, True))

    # pydantic v2 spells "optional" as a union, not as the type list a hand-written schema uses:
    #   {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": None, "title": "..."}
    # `_field` only understood the type-list form, so every Optional[...] field reached the final
    # `unsupported schema` raise and the whole model was unusable (#357).
    from typing import Optional

    class OptionalTicket(pydantic.BaseModel):
        urgent: Optional[bool] = None
        priority: Optional[int] = pydantic.Field(default=None, ge=0, le=2)
        channel: Optional[Literal["email", "chat"]] = pydantic.Field(
            default=None, description="what the caller sees")

    oq = plan_or_fail("pydantic-nullable/OptionalTicket plans", OptionalTicket)
    if oq:
        check("pydantic-nullable/Optional[bool] is a noul", oq["urgent"]["type"], "noul")
        check("pydantic-nullable/Optional[int] keeps its bounds", oq["priority"]["criteria"], ["0", "1", "2"])
        check("pydantic-nullable/Optional[Literal] is a choice",
              sorted(oq["channel"]["criteria"]), ["chat", "email"])
        check("pydantic-nullable/description beside the union survives the unwrap",
              oq["channel"]["instructions"], "what the caller sees")
        check("pydantic-nullable/every field is planned", sorted(oq),
              ["channel", "priority", "urgent"])

    # A required field next to optional ones must be unaffected.
    class Mixed(pydantic.BaseModel):
        department: Literal["billing", "support"]
        urgent: Optional[bool] = None

    mq = plan_or_fail("pydantic-nullable/Mixed plans", Mixed)
    if mq:
        check("pydantic-nullable/required Literal is untouched", mq["department"]["type"], "choice")
        check("pydantic-nullable/optional next to it works", mq["urgent"]["type"], "noul")

    # A union of two real variants is not a nullable field and cannot be one option set. The
    # error must name the path and say how many variants it could not reconcile, rather than
    # the bare "unsupported schema" the anyOf form used to produce.
    class Ambiguous(pydantic.BaseModel):
        value: Union[int, str, None] = None

    check_raises("pydantic-nullable/Union[int,str,None] is rejected", SchemaError,
                 lambda: questions_from_pydantic(Ambiguous))
    try:
        questions_from_pydantic(Ambiguous)
        FAIL.append("pydantic-nullable/Union[int,str,None]: no error raised")
    except SchemaError as exc:
        check_true("pydantic-nullable/rejection names the path", "value" in str(exc))
        check_true("pydantic-nullable/rejection counts the variants", "2 variants" in str(exc))
except ImportError:
    PASS.append("pydantic/skipped (not installed)")


# --------------------------------------------------------------- exports and methods
check("export/decide", callable(laya.decide), True)
check("export/DecisionResult", hasattr(laya, "DecisionResult"), True)
check("method/Agent.decide", callable(getattr(laya.Agent, "decide", None)), True)
check("method/Router.decide", callable(getattr(laya.Router, "decide", None)), True)
from laya.onnx_agent import ONNXAgent  # noqa: E402

check("method/ONNXAgent.decide", callable(getattr(ONNXAgent, "decide", None)), True)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all structured tests passed")
sys.exit(1 if FAIL else 0)
