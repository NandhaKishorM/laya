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
    decide_batch,
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
check("map/nullable boolean remains noul",
      questions_from_json_schema({"type": "object", "properties": {"a": {"type": ["null", "boolean"]}}})["a"]["type"],
      "noul")


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

# A score answer with no `probabilities` -- a cached/replayed decision, or a minimal runner
# that only reports the expected level index -- must still add `minimum`: `score` is the
# 0-based level index (DecisionModel._decode_answers), not the absolute field value, so a
# schema whose levels do not start at 0 must not be projected as if they did.
SEVERITY_SCHEMA = {"type": "object", "properties": {"x": {"type": "integer", "minimum": 3, "maximum": 7}}}
for score, want in ((0, 3), (2.4, 5), (4, 7)):
    got = answers_to_json({"x": {"type": "score", "score": score}}, SEVERITY_SCHEMA)["x"]
    check("project/score with no probabilities honours minimum (score=%r)" % score, got, want)


# --------------------------------------------------------------- rejections
def _bad(schema):
    return lambda: questions_from_json_schema(schema)


check_raises("reject/free string", SchemaError,
             _bad({"type": "object", "properties": {"a": {"type": "string"}}}))
check_raises("reject/multiple non-null types", SchemaError,
             _bad({"type": "object", "properties": {"a": {"type": ["boolean", "integer"],
                                                              "minimum": 0, "maximum": 2}}}))
check_raises("reject/nullable union of multiple types", SchemaError,
             _bad({"type": "object", "properties": {"a": {"type": ["null", "integer", "boolean"],
                                                              "minimum": 0, "maximum": 2}}}))
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
for colliding in ([1, "1"], [None, "null"], [True, "True"]):
    check_raises("reject/colliding enum %r" % colliding, SchemaError,
                 _bad({"type": "object", "properties": {"x": {"enum": colliding}}}))
try:
    questions_from_json_schema({"type": "object", "properties": {"x": {"enum": [1, "1"]}}})
except SchemaError as exc:
    check("reject/colliding enum names the field", str(exc),
          "properties.x: enum values produce duplicate choice labels")
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
# The pass-through is the raw answer, and no `min_confidence` was passed, so it comes back exactly
# as the runner produced it -- the gate adds nothing to a call it was not asked to gate.
check("decide/questions pass-through returns answers", out,
      {"a": {"type": "noul", "noul": 0.9, "confidence": 0.9}})

check_raises("decide/requires schema or questions", ValueError, lambda: decide(runner, "s"))
check_raises("decide/rejects both", ValueError, lambda: decide(runner, "s", schema=SCHEMA, questions={}))

# kwargs are forwarded to predict
runner = FakeRunner(ANSWERS)
decide(runner, "s", schema=SCHEMA, hooks_raise=False)
check("decide/forwards predict kwargs", runner.calls[0]["kwargs"], {"hooks_raise": False})

# --------------------------------------------------------------- min_confidence abstention (#361)
# Default behavior unchanged: all fields project normally when min_confidence is omitted
runner_default = FakeRunner(ANSWERS)
out_default = decide(runner_default, "s", schema=SCHEMA)
check("decide/default department unchanged", out_default["department"], "billing")
check("decide/default urgency unchanged", out_default["urgency"], 2)
check("decide/default needs_human unchanged", out_default["needs_human"], True)
check("decide/default priority unchanged", out_default["priority"], 2)
check_true("decide/default has no low_confidence flags",
           not any(a.get("low_confidence") for a in ANSWERS.values()))

# With min_confidence=0.85: answers below 0.85 project as None, raw confidence preserved
runner_gated = FakeRunner({k: dict(v) for k, v in ANSWERS.items()})
out_gated = decide(runner_gated, "s", schema=SCHEMA, min_confidence=0.85)
check("decide/gated high-confidence field kept", out_gated["department"], "billing")
check("decide/gated low-confidence score becomes None", out_gated["urgency"], None)
check("decide/gated low-confidence noul becomes None", out_gated["needs_human"], None)
check("decide/gated low-confidence choice becomes None", out_gated["priority"], None)

# With return_details=True, details keep raw confidence and answers dict
runner_det = FakeRunner({k: dict(v) for k, v in ANSWERS.items()})
det = decide(runner_det, "s", schema=SCHEMA, min_confidence=0.85, return_details=True)
check("decide/details values has None for low conf", det.values["urgency"], None)
check("decide/details raw confidence preserved", det.confidence["urgency"], 0.5)
check_true("decide/details answers flag set", det.answers["urgency"].get("low_confidence") is True)
check("decide/details high conf value kept", det.values["department"], "billing")
check_true("decide/details high conf flag unset", det.answers["department"].get("low_confidence") is not True)

# --------------------------------------------------------------------- answer_confidence
# The decision: "which of my structured decisions are safe to automate?" is answered by filtering
# the details artifact, so the number that filter reads has to be the one the gate uses.
#
# `DecisionResult.confidence` is built from the answer's `confidence` field, which is normalized
# entropy: `laya/common.py` calls it "not calibrated: it is not what temperature scaling fits and
# not what the reported ECE measures", `tests/test_confidence.py` pins that a two-option
# distribution reads 0.90 on a `noul` and 0.53 on an equivalent `choice`, and #394 is an open issue
# saying a confidence threshold does not transfer across option counts -- which is the entropy
# definition's failure mode, since `log(k)` is in the denominator.
#
# So a caller filtering `details.confidence` to decide what to escalate filters on a different
# quantity from the one `min_confidence` compares against, and on one that moves with the shape of
# the question rather than with how right the answer is.
#
# Terminology, deliberately careful: `answer_confidence` is `max(p)`, the quantity calibration
# fits and every calibration figure is computed on, and the quantity the gate is defined against.
# That makes it the right number to *filter* on. It is NOT a claim that the number is right --
# "about c of the answers returned at c are correct" holds only after temperatures are fitted and
# validated for that checkpoint and question shape, and the shipped checkpoints are over-confident
# as shipped (README, Calibration).
CAL = {"department": {"type": "choice", "choice": "billing", "confidence": 0.30, "answer_confidence": 0.95},
       "urgency": {"type": "score", "score": 2.0, "confidence": 0.90, "answer_confidence": 0.40,
                   "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4}, "legend": {}}}
cal_det = decide(FakeRunner(CAL), "s", schema=SCHEMA, return_details=True)

# The two numbers differ on the same field, which is the whole point: the entropy value ranks
# `urgency` above `department` and max(p) reverses it.
check_true("details/the two confidences really do disagree",
           cal_det.confidence["department"] < cal_det.confidence["urgency"])
check_true("details/…and answer_confidence reverses that order",
           cal_det.answer_confidence["department"] > cal_det.answer_confidence["urgency"])
check("details/answer_confidence is reported per field", cal_det.answer_confidence["department"], 0.95)

# The name means what it says, so a caller that sorts by it sorts on the same quantity the gate
# and the eval harness use.
check("details/entropy field is left exactly as it was", cal_det.confidence["department"], 0.30)

# A field with no usable answer_confidence is `None`, not 0.0. `confidence` defaults to 0.0, so
# today an absent confidence and a genuinely zero one are the same value, and a caller filtering on
# "below 0.4, escalate" escalates both without being able to tell why.
NO_CONF = {"department": {"type": "choice", "choice": "billing", "confidence": 0.0}}
no_det = decide(FakeRunner(NO_CONF), "s", schema=SCHEMA, return_details=True)
check("details/absent answer_confidence is None, not 0.0", no_det.answer_confidence["department"], None)
check_true("details/…and stays distinct from a reported zero",
           no_det.answer_confidence["department"] is not cal_det.answer_confidence["urgency"])

# A `bool` is not a confidence, and a NaN is not a decision.
BAD_CONF = {"department": {"type": "choice", "choice": "billing", "answer_confidence": True}}
bad_det = decide(FakeRunner(BAD_CONF), "s", schema=SCHEMA, return_details=True)
check("details/a bool is not reported as a confidence", bad_det.answer_confidence["department"], None)
NAN_CONF = {"department": {"type": "choice", "choice": "billing", "answer_confidence": float("nan")}}
nan_det = decide(FakeRunner(NAN_CONF), "s", schema=SCHEMA, return_details=True)
check("details/a NaN is not reported as a confidence", nan_det.answer_confidence["department"], None)

# Every field is accounted for, so a caller can iterate `answer_confidence` without a KeyError on
# a field the model did not answer.
check_true("details/one entry per field", set(cal_det.answer_confidence) == set(CAL))

# Direct projection with low_confidence: True in answer
answers_with_flag = {
    "department": {"type": "choice", "choice": "billing", "confidence": 0.4, "low_confidence": True},
    "urgency": {"type": "score", "score": 2.0, "confidence": 0.9, "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9}, "legend": {}},
}
proj = answers_to_json(answers_with_flag, SCHEMA)
check("project/flagged answer becomes None", proj["department"], None)
check("project/unflagged answer keeps value", proj["urgency"], 2)

# Invalid min_confidence validation
check_raises("decide/rejects min_confidence > 1", ValueError,
             lambda: decide(runner, "s", schema=SCHEMA, min_confidence=1.2))
check_raises("decide/rejects min_confidence < 0", ValueError,
             lambda: decide(runner, "s", schema=SCHEMA, min_confidence=-0.1))
check_raises("decide/rejects bool min_confidence", ValueError,
             lambda: decide(runner, "s", schema=SCHEMA, min_confidence=True))
check_raises("decide/rejects str min_confidence", ValueError,
             lambda: decide(runner, "s", schema=SCHEMA, min_confidence="0.85"))


# --------------------------------------------------------------- decide_batch
class FakeBatchRunner:
    """predict_batch(states, questions, **kwargs) like Agent's/Router's: one
    result dict per state, in input order."""

    def __init__(self, answers_by_state):
        self.answers_by_state = answers_by_state
        self.calls = []

    def predict_batch(self, states, questions, **kwargs):
        self.calls.append({"states": states, "questions": questions, "kwargs": kwargs})
        # like Router.predict_batch: empty answers only for a state that was
        # genuinely skipped by a hook -- the fake keys them off an unknown text.
        return [{"answers": self.answers_by_state.get(s, {}), "routing": {"model": "english"}}
                for s in states]


STATES = ["first ticket", "second ticket"]
BATCH_ANSWERS = {
    "first ticket": ANSWERS,
    "second ticket": {
        "department": {"type": "choice", "choice": "sales", "confidence": 0.8,
                       "probabilities": {"billing": 0.0, "support": 0.1, "sales": 0.9}},
        "urgency": {"type": "score", "score": 0.4, "confidence": 0.5,
                    "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1}},
        "needs_human": {"type": "noul", "noul": 0.1, "confidence": 0.8},
        "priority": {"type": "choice", "choice": "1", "confidence": 0.6,
                    "probabilities": {"1": 0.6, "2": 0.3, "3": 0.1}},
    },
}

batch_runner = FakeBatchRunner(BATCH_ANSWERS)
outs = decide_batch(batch_runner, STATES, schema=SCHEMA)
check("batch/one predict_batch call", len(batch_runner.calls), 1)
check("batch/states forwarded", batch_runner.calls[0]["states"], STATES)
check("batch/questions planned once", batch_runner.calls[0]["questions"],
      {f.name: f.question for f in plan_from_json_schema(SCHEMA)})
check("batch/input order", [o["department"] for o in outs], ["billing", "sales"])
check("batch/projection matches decide", outs[0], decide(FakeRunner(ANSWERS), "x", schema=SCHEMA))
check("batch/per-state values", outs[1],
      {"department": "sales", "urgency": 0, "needs_human": False, "priority": 1})

# kwargs are forwarded to predict_batch (batch_size, model, hooks, ...)
batch_runner = FakeBatchRunner(BATCH_ANSWERS)
decide_batch(batch_runner, STATES, schema=SCHEMA, batch_size=4, model="english")
check("batch/forwards predict kwargs", batch_runner.calls[0]["kwargs"],
      {"batch_size": 4, "model": "english"})

# return_details: one DecisionResult per state with the usual fields
details = decide_batch(FakeBatchRunner(BATCH_ANSWERS), STATES, schema=SCHEMA, return_details=True)
check_true("batch/details are DecisionResult", all(isinstance(d, laya.DecisionResult) for d in details))
check("batch/details values", [d.values["department"] for d in details], ["billing", "sales"])
check("batch/details confidence", details[0].confidence["department"], 0.9)
check("batch/details routing", details[0].routing, {"model": "english"})

# questions= is the raw-answers pass-through, exactly like decide()
raw = decide_batch(FakeBatchRunner({"a": {"x": {"type": "noul", "noul": 0.9, "confidence": 0.9}}}),
                   ["a"], questions={"x": {"type": "noul", "instructions": "?"}})
check("batch/questions pass-through", raw,
      [{"x": {"type": "noul", "noul": 0.9, "confidence": 0.9}}])


class FakeRouterLike:
    """predict_batch(requests, **kwargs) like Router's: one request dict per
    state (no positional questions), routed through route_batch."""

    def __init__(self, answers_by_state):
        self.answers_by_state = answers_by_state
        self.calls = []

    def route_batch(self, requests):
        return [{"model": "english"} for _ in requests]

    def predict_batch(self, requests, **kwargs):
        self.calls.append({"requests": requests, "kwargs": kwargs})
        return [{"answers": self.answers_by_state.get(r["state"], {}),
                 "routing": {"model": "english"}} for r in requests]


router_like = FakeRouterLike(BATCH_ANSWERS)
router_outs = decide_batch(router_like, STATES, schema=SCHEMA, batch_size=3)
expected_questions = {f.name: f.question for f in plan_from_json_schema(SCHEMA)}
check("batch/router one call", len(router_like.calls), 1)
check("batch/router request dicts", [r["state"] for r in router_like.calls[0]["requests"]], STATES)
check("batch/router questions per request",
      all(r["questions"] == expected_questions for r in router_like.calls[0]["requests"]), True)
check("batch/router kwargs forwarded", router_like.calls[0]["kwargs"], {"batch_size": 3})
check("batch/router projection matches agent path", router_outs, outs)

# a string is a sequence of states only accidentally: refuse it like route_batch does
check_raises("batch/rejects string states", TypeError,
             lambda: decide_batch(FakeBatchRunner({}), "abc", schema=SCHEMA))
check_raises("batch/requires schema or questions", ValueError,
             lambda: decide_batch(FakeBatchRunner({}), ["a"]))
check_raises("batch/rejects both", ValueError,
             lambda: decide_batch(FakeBatchRunner({}), ["a"], schema=SCHEMA, questions={}))
check_raises("batch/rejects bad schema", SchemaError,
             lambda: decide_batch(FakeBatchRunner({}), ["a"],
                                  schema={"type": "object", "properties": {"s": {"type": "string"}}}))


class NoBatchRunner(FakeRunner):
    pass  # has predict() but no predict_batch


check_raises("batch/runner without predict_batch", TypeError,
             lambda: decide_batch(NoBatchRunner(ANSWERS), ["a"], schema=SCHEMA))

# a runner that answers nothing (hook-skipped states surface as empty values)
noop_runner = FakeBatchRunner({})
check("batch/skipped states", decide_batch(noop_runner, ["unmatched"], schema=SCHEMA), [{}])
check("batch/noop still batches", len(noop_runner.calls), 1)


# --------------------------------------------------------------- pydantic (optional)
try:
    from typing import Literal

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
except ImportError:
    PASS.append("pydantic/skipped (not installed)")


# --------------------------------------------------------------- exports and methods
check("export/decide", callable(laya.decide), True)
check("export/decide_batch", callable(laya.decide_batch), True)
check("export/DecisionResult", hasattr(laya, "DecisionResult"), True)
check("method/Agent.decide", callable(getattr(laya.Agent, "decide", None)), True)
check("method/Agent.decide_batch", callable(getattr(laya.Agent, "decide_batch", None)), True)
check("method/Router.decide", callable(getattr(laya.Router, "decide", None)), True)
check("method/Router.decide_batch", callable(getattr(laya.Router, "decide_batch", None)), True)
from laya.onnx_agent import ONNXAgent  # noqa: E402

check("method/ONNXAgent.decide", callable(getattr(ONNXAgent, "decide", None)), True)


# --------------------------------------------------------------- nullable via anyOf (pydantic v2)
# Pydantic v2 renders Optional[X] as {"anyOf": [<X>, {"type": "null"}]}, the same nullable intent
# as the list form type:["string","null"]. Both must map to the underlying field, not raise.
NULLABLE = {
    "type": "object",
    "properties": {
        "dept": {"anyOf": [{"type": "string", "enum": ["billing", "sales"]}, {"type": "null"}],
                 "description": "Which team?"},
        "score": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 2}, {"type": "null"}]},
        "flag": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
    },
}
nq = questions_from_json_schema(NULLABLE)
check("nullable/anyOf enum is a choice", nq["dept"]["type"], "choice")
check("nullable/anyOf carries the outer description", nq["dept"]["instructions"], "Which team?")
check("nullable/anyOf bounded integer is a score", nq["score"]["type"], "score")
check("nullable/anyOf boolean is a noul", nq["flag"]["type"], "noul")
# oneOf is accepted the same way
check("nullable/oneOf enum is a choice",
      questions_from_json_schema(
          {"type": "object", "properties": {"a": {"oneOf": [{"enum": ["x", "y"]}, {"type": "null"}]}}}
      )["a"]["type"], "choice")
# a union of two real types stays ambiguous and is rejected
check_raises("nullable/two real branches rejected", SchemaError,
             _bad({"type": "object",
                   "properties": {"a": {"anyOf": [{"type": "boolean"}, {"type": "integer"}]}}}))


# --------------------------------------------------------------- local $ref (pydantic Enum)
# Pydantic renders every `Enum` field as `{"$ref": "#/$defs/<Name>"}` (v1 and draft-07 use
# `#/definitions/`, and wrap a described ref in a one-item `allOf`), so a local ref to an enum
# definition must plan as the enum it points at, with the property's own keys kept on top.
def _schema_with_defs(props, key="$defs"):
    return {"type": "object", "properties": props, key: {
        "Dept": {"enum": ["billing", "support"], "type": "string", "title": "Dept",
                 "description": "The team that owns it."},
        "Prio": {"enum": [0, 1, 2], "type": "integer", "title": "Prio"},
        "Alias": {"$ref": "#/%s/Prio" % key},
        "Node": {"type": "object", "properties": {"next": {"$ref": "#/%s/Node" % key}}},
        "Loop": {"$ref": "#/%s/Loop" % key},
        "Maybe": {"anyOf": [{"$ref": "#/%s/Maybe" % key}, {"type": "null"}]},
    }}


def _ref_error(prop, key="$defs"):
    try:
        plan_from_json_schema(_schema_with_defs({"a": prop}, key))
    except SchemaError as exc:
        return str(exc)
    return None


REFS = _schema_with_defs({
    "dept": {"$ref": "#/$defs/Dept"},
    "described": {"$ref": "#/$defs/Dept", "description": "Which team?"},
    "optional": {"anyOf": [{"$ref": "#/$defs/Dept"}, {"type": "null"}], "default": None,
                 "description": "Which team, if any?"},
    "wrapped": {"allOf": [{"$ref": "#/$defs/Dept"}], "description": "Which team (v1)?"},
    "prio": {"$ref": "#/$defs/Prio", "default": 0},
    "alias": {"$ref": "#/$defs/Alias"},
})
rq = questions_from_json_schema(REFS)
check("ref/enum definition is a choice", rq["dept"]["type"], "choice")
check("ref/enum labels come from the definition", list(rq["dept"]["criteria"]), ["billing", "support"])
# The definition's description is pydantic's copy of the enum docstring ("An enumeration." on v1
# when there is none): it describes the type, not this field, so it is not the question wording.
check("ref/definition's own description is not the wording", rq["dept"]["instructions"],
      "What is `dept`?")
check("ref/sibling description is the wording", rq["described"]["instructions"], "Which team?")
check("ref/Optional ref is a choice", rq["optional"]["type"], "choice")
check("ref/Optional ref carries the outer description", rq["optional"]["instructions"], "Which team, if any?")
check("ref/allOf-wrapped ref is a choice", rq["wrapped"]["type"], "choice")
check("ref/allOf-wrapped ref keeps the sibling description", rq["wrapped"]["instructions"], "Which team (v1)?")
check("ref/integer enum labels", list(rq["prio"]["criteria"]), ["0", "1", "2"])
check("ref/a ref to a ref resolves", list(rq["alias"]["criteria"]), ["0", "1", "2"])
check("ref/draft-07 #/definitions/ resolves",
      questions_from_json_schema(_schema_with_defs({"a": {"$ref": "#/definitions/Dept"}}, "definitions"))
      ["a"]["type"], "choice")
check("ref/allOf wraps an inline schema too",
      questions_from_json_schema({"type": "object", "properties": {"a": {"allOf": [{"type": "boolean"}]}}})
      ["a"]["type"], "noul")
rv = answers_to_json({"dept": {"choice": "support"}, "prio": {"choice": "2"}, "alias": {"choice": "1"}}, REFS)
check("ref/projects the definition's values", rv, {"dept": "support", "prio": 2, "alias": 1})
_before = repr(REFS)
plan_from_json_schema(REFS)
check("ref/planning does not mutate the schema", repr(REFS), _before)

check("ref/missing definition is rejected", _ref_error({"$ref": "#/$defs/Nope"}),
      "properties.a: $ref '#/$defs/Nope' does not resolve to an entry of this schema's '$defs' or 'definitions'")
check("ref/wrong definitions key is rejected", _ref_error({"$ref": "#/definitions/Dept"}),
      "properties.a: $ref '#/definitions/Dept' does not resolve to an entry of this schema's '$defs' or "
      "'definitions'")
check("ref/non-local ref is rejected", _ref_error({"$ref": "other.json#/$defs/Dept"}),
      "properties.a: $ref 'other.json#/$defs/Dept' does not resolve to an entry of this schema's '$defs' or "
      "'definitions'")
check("ref/a ref with no definitions at all is rejected", _ref_error({"$ref": "#/$defs/X"}, "unused"),
      "properties.a: $ref '#/$defs/X' does not resolve to an entry of this schema's '$defs' or 'definitions'")
check("ref/self-reference is rejected", _ref_error({"$ref": "#/$defs/Loop"}),
      "properties.a: $ref '#/$defs/Loop' is recursive; flatten the schema")
check("ref/recursion through Optional is rejected", _ref_error({"$ref": "#/$defs/Maybe"}),
      "properties.a: $ref '#/$defs/Maybe' is recursive; flatten the schema")
check("ref/an object definition is a nested object", _ref_error({"$ref": "#/$defs/Node"}),
      "properties.a: nested objects are not supported; flatten the schema")

try:
    import enum
    from typing import Optional

    import pydantic

    class Dept(str, enum.Enum):
        billing = "billing"
        support = "support"
        sales = "sales"

    class Prio(enum.IntEnum):
        low = 0
        high = 1

    class Colour(enum.Enum):
        red = "red"
        blue = "blue"

    class EnumTicket(pydantic.BaseModel):
        department: Dept = pydantic.Field(description="Which team should handle this?")
        prio: Prio
        colour: Colour
        backup: Optional[Dept] = None

    eq = questions_from_pydantic(EnumTicket)
    check("pydantic-enum/str Enum is a choice", eq["department"]["type"], "choice")
    check("pydantic-enum/Field description is the wording", eq["department"]["instructions"],
          "Which team should handle this?")
    check("pydantic-enum/labels are the member values", list(eq["department"]["criteria"]),
          ["billing", "support", "sales"])
    check("pydantic-enum/IntEnum is a choice", (eq["prio"]["type"], list(eq["prio"]["criteria"])),
          ("choice", ["0", "1"]))
    check("pydantic-enum/plain Enum is a choice", eq["colour"]["type"], "choice")
    check("pydantic-enum/Optional[Enum] is a choice", eq["backup"]["type"], "choice")

    enum_answers = {
        "department": {"type": "choice", "choice": "support", "confidence": 0.9, "probabilities": {}},
        "prio": {"type": "choice", "choice": "1", "confidence": 0.8, "probabilities": {}},
        "colour": {"type": "choice", "choice": "blue", "confidence": 0.8, "probabilities": {}},
    }
    et = answer_to_pydantic(EnumTicket, enum_answers)
    check("pydantic-enum/round trip to members", (et.department, et.prio, et.colour, et.backup),
          (Dept.support, Prio.high, Colour.blue, None))
    check_true("pydantic-enum/IntEnum comes back as the member", et.prio is Prio.high)
    et = answer_to_pydantic(EnumTicket, dict(enum_answers, backup={"type": "choice", "choice": "sales"}))
    check("pydantic-enum/Optional[Enum] answered", et.backup, Dept.sales)
    check("pydantic-enum/decide projects values",
          decide(FakeRunner(enum_answers), "s", schema=EnumTicket),
          {"department": "support", "prio": 1, "colour": "blue"})

    class Inner(pydantic.BaseModel):
        flag: bool

    class Outer(pydantic.BaseModel):
        inner: Inner

    try:
        questions_from_pydantic(Outer)
        FAIL.append("pydantic-enum/nested model: did not raise")
    except SchemaError as exc:
        check("pydantic-enum/nested model is still a nested object", str(exc),
              "properties.inner: nested objects are not supported; flatten the schema")
except ImportError:
    PASS.append("pydantic-enum/skipped (not installed)")


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all structured tests passed")
sys.exit(1 if FAIL else 0)
