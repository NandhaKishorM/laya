"""API-stability guard for the OpenAI-compatible translation.

Pins the public signatures, constants, the ChatPlan fields and the exports. Update this file in
the same commit as an intentional change.

Run: python tests/test_openai_api.py
"""
import dataclasses
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import openai_api  # noqa: E402

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
check("MAX_TOOLS", openai_api.MAX_TOOLS, 8)
check("TOOL_QUESTION", openai_api.TOOL_QUESTION, "tool")
check("MODERATION_THRESHOLD", openai_api.MODERATION_THRESHOLD, 0.5)
check_true("UnsupportedRequest is a ValueError",
           issubclass(openai_api.UnsupportedRequest, ValueError))

# --------------------------------------------------------------- functions
check_param("parse_chat_request", openai_api.parse_chat_request, "body", inspect.Parameter.empty)
check_param("parse_responses_request", openai_api.parse_responses_request, "body", inspect.Parameter.empty)
check_param("format_chat_response", openai_api.format_chat_response, "plan", inspect.Parameter.empty)
check_param("format_chat_response", openai_api.format_chat_response, "result", inspect.Parameter.empty)
check_param("format_responses_response", openai_api.format_responses_response, "plan", inspect.Parameter.empty)
check_param("format_responses_response", openai_api.format_responses_response, "result", inspect.Parameter.empty)
check_param("models_payload", openai_api.models_payload, "names", inspect.Parameter.empty)
check_param("parse_moderation_request", openai_api.parse_moderation_request, "body", inspect.Parameter.empty)
check_param("moderation_payload", openai_api.moderation_payload, "results", inspect.Parameter.empty)
check_kwonly("moderation_payload", openai_api.moderation_payload, "model")
check_kwonly("moderation_payload", openai_api.moderation_payload, "threshold")
check_kwonly("moderation_payload", openai_api.moderation_payload, "questions")
check_param("moderation_payload", openai_api.moderation_payload, "model", "laya")
check_param("moderation_payload", openai_api.moderation_payload, "threshold", openai_api.MODERATION_THRESHOLD)
check_param("moderation_payload", openai_api.moderation_payload, "questions", None)

# --------------------------------------------------------------- ChatPlan
FIELDS = ["mode", "state", "questions", "model", "schema", "tool_names", "tool_schemas", "forced_tool"]
check("ChatPlan fields", [f.name for f in dataclasses.fields(openai_api.ChatPlan)], FIELDS)
check("ChatPlan/model default", openai_api.ChatPlan.__dataclass_fields__["model"].default, None)
check("ChatPlan/forced_tool default", openai_api.ChatPlan.__dataclass_fields__["forced_tool"].default, None)

# --------------------------------------------------------------- exports
for name in openai_api.__all__:
    check_true("openai_api/%s exists" % name, hasattr(openai_api, name))


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all openai API tests passed")
sys.exit(1 if FAIL else 0)
