"""OpenAI-compatible translation: pure request/response mapping. No fastapi, no model.

Run: python tests/test_openai.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.openai_api import (  # noqa: E402
    MAX_TOOLS,
    TOOL_QUESTION,
    UnsupportedRequest,
    format_chat_response,
    format_responses_response,
    moderation_payload,
    models_payload,
    parse_chat_request,
    parse_moderation_request,
    parse_responses_request,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


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
        "department": {"type": "string", "enum": ["billing", "support"]},
        "urgent": {"type": "boolean"},
    },
}

TOOLS = [
    {"type": "function", "function": {
        "name": "route", "description": "Route a ticket",
        "parameters": {"type": "object", "properties": {
            "department": {"type": "string", "enum": ["billing", "support"]},
            "urgent": {"type": "boolean"}}}}} ,
    {"type": "function", "function": {
        "name": "escalate", "description": "Escalate to a human",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "enum": ["anger", "sla"]}}}}} ,
]


# --------------------------------------------------------------- schema mode
plan = parse_chat_request({
    "model": "laya",
    "messages": [{"role": "user", "content": "billed twice"}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": SCHEMA}},
})
check("chat/schema mode", plan.mode, "schema")
check("chat/schema state", plan.state, "billed twice")
check("chat/schema model", plan.model, "laya")
check("chat/schema questions", plan.questions["department"]["type"], "choice")

result = {"answers": {
    "department": {"type": "choice", "choice": "billing", "confidence": 0.9, "probabilities": {}},
    "urgent": {"type": "noul", "noul": 0.8, "confidence": 0.8}},
    "usage": {"input_tokens": 7, "output_tokens": 0}}
response = format_chat_response(plan, result)
check("chat/schema content", json.loads(response["choices"][0]["message"]["content"]),
      {"department": "billing", "urgent": True})
check("chat/schema finish", response["choices"][0]["finish_reason"], "stop")
check("chat/schema object", response["object"], "chat.completion")
check("chat/schema usage", response["usage"],
      {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7})

# multiple messages become a turn list
multi = parse_chat_request({
    "messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}],
    "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}})
check("chat/multi-message state is a turn list", multi.state,
      [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}])

# content parts are joined
parts = parse_chat_request({
    "messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
    "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}})
check("chat/content parts joined", parts.state, "a\nb")


# --------------------------------------------------------------- tools mode
tplan = parse_chat_request({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS})
check("tools/mode", tplan.mode, "tools")
check("tools/selection question", TOOL_QUESTION in tplan.questions, True)
check("tools/namespaced arg question", "route__department" in tplan.questions, True)
check("tools/second tool arg question", "escalate__reason" in tplan.questions, True)
check("tools/names", tplan.tool_names, ["route", "escalate"])

tresult = {"answers": {
    TOOL_QUESTION: {"type": "choice", "choice": "escalate", "confidence": 0.7, "probabilities": {}},
    "route__department": {"type": "choice", "choice": "billing", "confidence": 0.6, "probabilities": {}},
    "route__urgent": {"type": "noul", "noul": 0.9, "confidence": 0.9},
    "escalate__reason": {"type": "choice", "choice": "anger", "confidence": 0.8, "probabilities": {}}},
    "usage": {"input_tokens": 3, "output_tokens": 0}}
tresp = format_chat_response(tplan, tresult)
call = tresp["choices"][0]["message"]["tool_calls"][0]
check("tools/finish", tresp["choices"][0]["finish_reason"], "tool_calls")
check("tools/chosen tool", call["function"]["name"], "escalate")
check("tools/arguments use original field names", json.loads(call["function"]["arguments"]),
      {"reason": "anger"})
check("tools/call id prefix", call["id"].startswith("call_"), True)

# one tool: no selection question
one = parse_chat_request({"messages": [{"role": "user", "content": "hi"}], "tools": [TOOLS[0]]})
check("tools/single tool has no selection question", TOOL_QUESTION in one.questions, False)

# forced tool via tool_choice
forced = parse_chat_request({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS,
                             "tool_choice": {"type": "function", "function": {"name": "route"}}})
check("tools/forced tool skips selection", TOOL_QUESTION in forced.questions, False)
check("tools/forced tool recorded", forced.forced_tool, "route")


# --------------------------------------------------------------- responses API
rplan = parse_responses_request({
    "model": "laya", "input": "billed twice",
    "text": {"format": {"type": "json_schema", "name": "t", "schema": SCHEMA}}})
check("responses/schema mode", rplan.mode, "schema")
check("responses/state", rplan.state, "billed twice")
rresp = format_responses_response(rplan, result)
check("responses/object", rresp["object"], "response")
check("responses/status", rresp["status"], "completed")
check("responses/output type", rresp["output"][0]["type"], "message")
check("responses/output text", json.loads(rresp["output"][0]["content"][0]["text"]),
      {"department": "billing", "urgent": True})
check("responses/usage", rresp["usage"], {"input_tokens": 7, "output_tokens": 0, "total_tokens": 7})

rtools = parse_responses_request({"input": ["a", "b"], "tools": TOOLS})
check("responses/tools turn list", rtools.state, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
rout = format_responses_response(rtools, tresult)
check("responses/tools function_call", rout["output"][0]["type"], "function_call")
check("responses/tools name", rout["output"][0]["name"], "escalate")


# --------------------------------------------------------------- unsupported
check_raises("unsupported/no schema or tools", UnsupportedRequest,
             lambda: parse_chat_request({"messages": [{"role": "user", "content": "hi"}]}))
check_raises("unsupported/json_object", UnsupportedRequest,
             lambda: parse_chat_request({"messages": [{"role": "user", "content": "hi"}],
                                         "response_format": {"type": "json_object"}}))
check_raises("unsupported/both", UnsupportedRequest,
             lambda: parse_chat_request({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS,
                                         "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}}))
check_raises("unsupported/too many tools", UnsupportedRequest,
             lambda: parse_chat_request({"messages": [{"role": "user", "content": "hi"}],
                                         "tools": [TOOLS[0]] * (MAX_TOOLS + 1)}))
check_raises("unsupported/duplicate tool", UnsupportedRequest,
             lambda: parse_chat_request({"messages": [{"role": "user", "content": "hi"}],
                                         "tools": [TOOLS[0], TOOLS[0]]}))
check_raises("unsupported/non-function tool", UnsupportedRequest,
             lambda: parse_chat_request({"messages": [{"role": "user", "content": "hi"}],
                                         "tools": [{"type": "retrieval"}]}))
check_raises("unsupported/bad tool_choice", UnsupportedRequest,
             lambda: parse_chat_request({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS,
                                         "tool_choice": {"function": {"name": "nope"}}}))
check_raises("unsupported/responses without format", UnsupportedRequest,
             lambda: parse_responses_request({"input": "hi"}))


# --------------------------------------------------------------- models and moderations
check("models/object", models_payload(["english"])["object"], "list")
check("models/data", models_payload(["english", "multilingual"])["data"][0],
      {"id": "english", "object": "model", "created": 0, "owned_by": "laya"})

check("moderation/parse string", parse_moderation_request({"input": "hi"}), ["hi"])
check("moderation/parse list", parse_moderation_request({"input": ["a", "b"]}), ["a", "b"])
check_raises("moderation/reject object", UnsupportedRequest, lambda: parse_moderation_request({"input": {}}))

mod = moderation_payload([
    {"answers": {"toxic": {"type": "noul", "noul": 0.8, "confidence": 0.8},
                 "severity": {"type": "score", "score": 2.1, "probabilities": {}},
                 "spam": {"type": "noul", "noul": 0.1, "confidence": 0.9}}},
    {"answers": {"toxic": {"type": "noul", "noul": 0.05, "confidence": 0.95}}},
])
check("moderation/flagged first", mod["results"][0]["flagged"], True)
check("moderation/category bool", mod["results"][0]["categories"]["toxic"], True)
check("moderation/category score", mod["results"][0]["category_scores"]["severity"], 2.1)
check("moderation/spam not flagged", mod["results"][0]["categories"]["spam"], False)
check("moderation/second clean", mod["results"][1]["flagged"], False)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all openai translation tests passed")
sys.exit(1 if FAIL else 0)
