"""API-stability guard for schema-driven decisions.

Pins the public signatures, constants, the DecisionResult fields and the exports, so a change
that would break callers fails here first. Update this file in the same commit as an intentional
change.

Run: python tests/test_structured_api.py
"""
import dataclasses
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya  # noqa: E402
from laya import structured  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def check_param(name, fn, param, default):
    params = inspect.signature(fn).parameters
    if param not in params:
        FAIL.append("%s/%s: missing parameter" % (name, param))
        return
    check("%s/%s default" % (name, param), params[param].default, default)


def check_kwonly(name, fn, param):
    params = inspect.signature(fn).parameters
    if param not in params:
        FAIL.append("%s/%s: missing parameter" % (name, param))
        return
    check("%s/%s kind" % (name, param), params[param].kind, inspect.Parameter.KEYWORD_ONLY)


# --------------------------------------------------------------- constants
check("MAX_PROPERTIES", structured.MAX_PROPERTIES, 32)
check("MAX_OPTIONS", structured.MAX_OPTIONS, 32)
check("MAX_SCORE_LEVELS", structured.MAX_SCORE_LEVELS, 10)

# --------------------------------------------------------------- functions
check_param("questions_from_json_schema", structured.questions_from_json_schema, "schema", inspect.Parameter.empty)
check_param("questions_from_pydantic", structured.questions_from_pydantic, "model", inspect.Parameter.empty)
check_param("answers_to_json", structured.answers_to_json, "answers", inspect.Parameter.empty)
check_param("answers_to_json", structured.answers_to_json, "schema", inspect.Parameter.empty)
check_param("answer_to_pydantic", structured.answer_to_pydantic, "model", inspect.Parameter.empty)
check_param("answer_to_pydantic", structured.answer_to_pydantic, "answers", inspect.Parameter.empty)
check_param("plan_from_json_schema", structured.plan_from_json_schema, "schema", inspect.Parameter.empty)

check_param("decide", structured.decide, "runner", inspect.Parameter.empty)
check_param("decide", structured.decide, "state", inspect.Parameter.empty)
check_param("decide", structured.decide, "schema", None)
check_kwonly("decide", structured.decide, "questions")
check_kwonly("decide", structured.decide, "return_details")
check_param("decide", structured.decide, "questions", None)
check_param("decide", structured.decide, "return_details", False)
check_true("decide/accepts **predict_kwargs",
           any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(structured.decide).parameters.values()))

# --------------------------------------------------------------- DecisionResult
FIELDS = ["values", "confidence", "probabilities", "answers", "usage", "routing"]
check("DecisionResult fields", [f.name for f in dataclasses.fields(structured.DecisionResult)], FIELDS)
for optional in ("usage", "routing"):
    check("DecisionResult/%s default None" % optional,
          structured.DecisionResult.__dataclass_fields__[optional].default, None)

# --------------------------------------------------------------- methods
for label, cls in (("Agent", laya.Agent), ("Router", laya.Router)):
    fn = getattr(cls, "decide", None)
    check_true("%s.decide exists" % label, callable(fn))
    if fn is not None:
        check_param("%s.decide" % label, fn, "schema", None)
        check_kwonly("%s.decide" % label, fn, "questions")
        check_kwonly("%s.decide" % label, fn, "return_details")

from laya.onnx_agent import ONNXAgent  # noqa: E402

check_true("ONNXAgent.decide exists", callable(getattr(ONNXAgent, "decide", None)))

# --------------------------------------------------------------- exports
for name in ("decide", "DecisionResult"):
    check_true("__all__/%s" % name, name in laya.__all__)
    check_true("laya.%s exists" % name, hasattr(laya, name))
check_true("laya.structured.SchemaError is a ValueError", issubclass(structured.SchemaError, ValueError))


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all structured API tests passed")
sys.exit(1 if FAIL else 0)
