"""OpenAI-compatible request and response translation for ``laya-serve``.

Pure Python: no fastapi, no torch, so the wire mapping can be unit tested on its own. Laya is a
structured-decision engine, not a chat model, so only the structured subset of the OpenAI surface
is supported: ``response_format`` with a JSON schema, and ``tools`` (function calling). Anything
that needs free text, streaming or sampling raises ``UnsupportedRequest`` and becomes a 400.

The supported shapes:

* ``POST /v1/chat/completions``: ``response_format={"type": "json_schema", ...}``, or ``tools``.
* ``POST /v1/responses``: ``text.format`` (json_schema) or ``tools``.
* ``POST /v1/moderations``: the ``moderation_questions`` preset.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .presets import moderation_questions
from .structured import answers_to_json, plan_from_json_schema

MAX_TOOLS = 8
TOOL_QUESTION = "tool"
MODERATION_THRESHOLD = 0.5


class UnsupportedRequest(ValueError):
    """The request asks for something a decision model cannot do; the message says what to send."""


@dataclass
class ChatPlan:
    """A parsed OpenAI request, ready to answer."""

    mode: str                                  # "schema" or "tools"
    state: Any
    questions: Dict[str, Any]
    model: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None    # schema mode
    tool_names: List[str] = field(default_factory=list)
    tool_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # namespaced, for projection
    forced_tool: Optional[str] = None


# ---------------------------------------------------------------- request helpers
def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in (None, "text", "input_text", "output_text"):
                    parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _state_from_messages(messages: Any) -> Any:
    if not isinstance(messages, list) or not messages:
        raise UnsupportedRequest("'messages' must be a non-empty list")
    turns = []
    for message in messages:
        if not isinstance(message, dict) or "role" not in message:
            raise UnsupportedRequest("each message needs a 'role'")
        turns.append({"role": str(message["role"]), "content": _text_from_content(message.get("content"))})
    if len(turns) == 1:
        return turns[0]["content"]
    return turns


def _state_from_input(input_value: Any) -> Any:
    if isinstance(input_value, str):
        return input_value
    if not isinstance(input_value, list) or not input_value:
        raise UnsupportedRequest("'input' must be a string or a non-empty list")
    turns = []
    for item in input_value:
        if isinstance(item, str):
            turns.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise UnsupportedRequest("each input item must be a string or an object")
        role = str(item.get("role", "user"))
        turns.append({"role": role, "content": _text_from_content(item.get("content"))})
    if len(turns) == 1:
        return turns[0]["content"]
    return turns


def _schema_from_response_format(response_format: Any) -> Dict[str, Any]:
    if not isinstance(response_format, dict):
        raise UnsupportedRequest("'response_format' must be an object")
    kind = response_format.get("type")
    if kind == "json_object":
        raise UnsupportedRequest(
            "Laya needs options to choose from; use response_format json_schema, not json_object")
    if kind != "json_schema":
        raise UnsupportedRequest("unsupported response_format type %r" % kind)
    inner = response_format.get("json_schema")
    schema = inner.get("schema") if isinstance(inner, dict) else None
    if not isinstance(schema, dict):
        raise UnsupportedRequest("response_format.json_schema.schema must be an object")
    return schema


def _namespaced(schema: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Copy a parameter schema with every property key prefixed, so tools do not collide."""
    props = schema.get("properties")
    if not isinstance(props, dict):
        raise UnsupportedRequest("a tool needs an object 'parameters' schema with 'properties'")
    renamed = {prefix + name: sub for name, sub in props.items()}
    out = dict(schema)
    out["properties"] = renamed
    if "required" in out and isinstance(out["required"], list):
        out["required"] = [prefix + r for r in out["required"]]
    return out


def _forced_tool(tool_choice: Any, names: Sequence[str]) -> Optional[str]:
    """The tool named by `tool_choice`, if it names one. Only function-type choices are honoured."""
    if not isinstance(tool_choice, dict):
        return None
    name = (tool_choice.get("function") or {}).get("name")
    if name is None:
        return None
    if name not in names:
        raise UnsupportedRequest("tool_choice names unknown tool %r" % name)
    return name


def _responses_tools(tools: Any) -> Any:
    """Map the Responses tool shape onto the nested Chat shape `_tool_questions` reads.

    Responses uses a flat function object (``{"type": "function", "name": ..., "parameters":
    ...}``); Chat Completions nests it under ``function``. An already-nested entry is passed
    through, and anything else is left for `_tool_questions` to reject with its own message.
    """
    if not isinstance(tools, list):
        return tools
    normalised = []
    for tool in tools:
        if (isinstance(tool, dict) and "function" not in tool
                and tool.get("type", "function") == "function" and isinstance(tool.get("name"), str)):
            normalised.append({"type": "function", "function": {
                "name": tool["name"],
                "description": tool.get("description"),
                "parameters": tool.get("parameters") or {},
            }})
        else:
            normalised.append(tool)
    return normalised


def _responses_tool_choice(tool_choice: Any) -> Any:
    """Map the Responses named tool_choice onto the nested Chat shape `_forced_tool` reads.

    ``{"type": "function", "name": "x"}`` becomes ``{"type": "function", "function": {"name":
    "x"}}``; ``"auto"`` / ``"none"`` / ``"required"`` are strings and pass through.
    """
    if (isinstance(tool_choice, dict) and "function" not in tool_choice
            and tool_choice.get("type", "function") == "function" and isinstance(tool_choice.get("name"), str)):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice


def _tool_questions(tools: Any, tool_choice: Any) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, Any], Optional[str]]:
    if not isinstance(tools, list) or not tools:
        raise UnsupportedRequest("'tools' must be a non-empty list")
    if len(tools) > MAX_TOOLS:
        raise UnsupportedRequest("%d tools exceeds MAX_TOOLS=%d" % (len(tools), MAX_TOOLS))

    names: List[str] = []
    schemas: Dict[str, Dict[str, Any]] = {}
    questions: Dict[str, Any] = {}
    descriptions: Dict[str, Optional[str]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise UnsupportedRequest("only 'function' tools are supported")
        function = tool.get("function") or {}
        name = function.get("name")
        if not name:
            raise UnsupportedRequest("every tool needs function.name")
        if name in schemas:
            raise UnsupportedRequest("duplicate tool name %r" % name)
        schema = _namespaced(function.get("parameters") or {}, name + "__")
        for field_plan in plan_from_json_schema(schema):   # validates the tool schema now
            questions[field_plan.name] = field_plan.question
        names.append(name)
        schemas[name] = schema
        descriptions[name] = function.get("description") or None

    forced = _forced_tool(tool_choice, names)
    if forced is None and len(names) > 1:
        questions[TOOL_QUESTION] = {
            "type": "choice",
            "instructions": "Which tool should be used for this request?",
            "criteria": descriptions,
        }
    return names, schemas, questions, forced


def _plan(mode: str, state: Any, model: Optional[str], questions: Dict[str, Any],
          schema: Optional[Dict[str, Any]] = None, tool_names: Optional[List[str]] = None,
          tool_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
          forced_tool: Optional[str] = None) -> ChatPlan:
    if not questions:
        raise UnsupportedRequest("the request produced no questions to answer")
    return ChatPlan(mode=mode, state=state, questions=questions, model=model, schema=schema,
                    tool_names=tool_names or [], tool_schemas=tool_schemas or {},
                    forced_tool=forced_tool)


def _reject_stream_and_n(body: Dict[str, Any]) -> None:
    """Reject the bits of the OpenAI surface a decision engine cannot honour.

    Both are documented as unsupported, but silently answering a `stream: true` request with one
    non-streamed body leaves the client waiting for `data:` chunks that never arrive, and `n`
    other than 1 would silently return fewer choices than asked for.
    """
    if body.get("stream"):
        raise UnsupportedRequest("streaming is not supported; resend without 'stream'")
    n = body.get("n")
    if n is not None and n != 1:
        raise UnsupportedRequest("'n' is not supported; Laya returns a single choice")


def parse_chat_request(body: Dict[str, Any]) -> ChatPlan:
    if not isinstance(body, dict):
        raise UnsupportedRequest("request body must be an object")
    _reject_stream_and_n(body)
    state = _state_from_messages(body.get("messages"))
    model = body.get("model")
    tools = body.get("tools")
    response_format = body.get("response_format")
    if tools and response_format:
        raise UnsupportedRequest("send either tools or response_format, not both")
    if tools:
        names, schemas, questions, forced = _tool_questions(tools, body.get("tool_choice"))
        return _plan("tools", state, model, questions, tool_names=names,
                     tool_schemas=schemas, forced_tool=forced)
    if response_format is not None:
        schema = _schema_from_response_format(response_format)
        questions = {f.name: f.question for f in plan_from_json_schema(schema)}
        return _plan("schema", state, model, questions, schema=schema)
    raise UnsupportedRequest(
        "Laya is a structured-decision engine: send response_format json_schema or tools")


def parse_responses_request(body: Dict[str, Any]) -> ChatPlan:
    if not isinstance(body, dict):
        raise UnsupportedRequest("request body must be an object")
    _reject_stream_and_n(body)
    state = _state_from_input(body.get("input"))
    model = body.get("model")
    tools = body.get("tools")
    text = body.get("text") or {}
    fmt = text.get("format") if isinstance(text, dict) else None
    if tools and fmt:
        raise UnsupportedRequest("send either tools or text.format, not both")
    if tools:
        names, schemas, questions, forced = _tool_questions(
            _responses_tools(tools), _responses_tool_choice(body.get("tool_choice")))
        return _plan("tools", state, model, questions, tool_names=names,
                     tool_schemas=schemas, forced_tool=forced)
    if fmt is not None:
        if not isinstance(fmt, dict):
            raise UnsupportedRequest("'text.format' must be an object")
        if fmt.get("type") == "json_object":
            raise UnsupportedRequest(
                "Laya needs options to choose from; use text.format json_schema, not json_object")
        if fmt.get("type") != "json_schema":
            raise UnsupportedRequest("unsupported text.format type %r" % fmt.get("type"))
        schema = fmt.get("schema")
        if not isinstance(schema, dict):
            raise UnsupportedRequest("text.format.schema must be an object")
        questions = {f.name: f.question for f in plan_from_json_schema(schema)}
        return _plan("schema", state, model, questions, schema=schema)
    raise UnsupportedRequest("Laya is a structured-decision engine: send text.format json_schema or tools")


# ---------------------------------------------------------------- response helpers
def _usage(result: Dict[str, Any], *, responses: bool = False) -> Dict[str, int]:
    usage = result.get("usage") or {}
    prompt = int(usage.get("input_tokens", 0) or 0)
    completion = int(usage.get("output_tokens", 0) or 0)
    if responses:
        return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": prompt + completion}
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _values_for_tool(plan: ChatPlan, answers: Dict[str, Any], tool: str) -> Dict[str, Any]:
    schema = plan.tool_schemas[tool]
    projected = answers_to_json(answers, schema)
    prefix = tool + "__"
    return {name[len(prefix):]: value for name, value in projected.items()}


def _tool_call(plan: ChatPlan, answers: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    tool = plan.forced_tool
    if tool is None and len(plan.tool_names) > 1:
        selection = answers.get(TOOL_QUESTION) or {}
        tool = str(selection.get("choice") or plan.tool_names[0])
    if tool is None:
        tool = plan.tool_names[0]
    return tool, _values_for_tool(plan, answers, tool)


def format_chat_response(plan: ChatPlan, result: Dict[str, Any]) -> Dict[str, Any]:
    answers = result.get("answers") or {}
    if plan.mode == "schema":
        content = json.dumps(answers_to_json(answers, plan.schema))
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        finish_reason = "stop"
    else:
        tool, args = _tool_call(plan, answers)
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)},
            }],
        }
        finish_reason = "tool_calls"
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": plan.model or "laya",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": _usage(result),
    }


def format_responses_response(plan: ChatPlan, result: Dict[str, Any]) -> Dict[str, Any]:
    answers = result.get("answers") or {}
    if plan.mode == "schema":
        text = json.dumps(answers_to_json(answers, plan.schema))
        output = [{"type": "message", "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": text}]}]
    else:
        tool, args = _tool_call(plan, answers)
        output = [{"type": "function_call", "call_id": "call_" + uuid.uuid4().hex,
                   "name": tool, "arguments": json.dumps(args)}]
    return {
        "id": "resp_" + uuid.uuid4().hex,
        "object": "response",
        "created_at": int(time.time()),
        "model": plan.model or "laya",
        "status": "completed",
        "output": output,
        "usage": _usage(result, responses=True),
    }


def models_payload(names: Sequence[str]) -> Dict[str, Any]:
    return {"object": "list", "data": [
        {"id": name, "object": "model", "created": 0, "owned_by": "laya"} for name in names]}


def parse_moderation_request(body: Dict[str, Any]) -> List[Any]:
    input_value = body.get("input")
    if isinstance(input_value, str):
        return [input_value]
    if isinstance(input_value, list) and input_value and all(isinstance(i, str) for i in input_value):
        return list(input_value)
    raise UnsupportedRequest("'input' must be a string or a list of strings")


def _score_levels(question: Any, answer: Dict[str, Any]) -> Optional[int]:
    """How many levels a `score` question has, from its criteria or the answer's probabilities.

    Returns None when neither names more than one level, in which case the raw score is used.
    """
    criteria = (question or {}).get("criteria") if isinstance(question, dict) else None
    if isinstance(criteria, list) and len(criteria) > 1:
        return len(criteria)
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and len(probabilities) > 1:
        return len(probabilities)
    return None


def moderation_payload(results: Sequence[Dict[str, Any]], *, model: str = "laya",
                       threshold: float = MODERATION_THRESHOLD,
                       questions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rows = []
    for result in results:
        answers = result.get("answers") or {}
        categories: Dict[str, bool] = {}
        scores: Dict[str, float] = {}
        for name, answer in answers.items():
            if answer.get("type") == "noul":
                value = float(answer.get("noul", 0.0))
                scores[name] = round(value, 4)
                categories[name] = value >= threshold
            elif answer.get("type") == "score":
                # A score answer's `score` is the expected level on the question's 0..n-1
                # scale, not a probability, so it is normalized onto [0, 1] before it shares
                # the `threshold` with the noul categories. Without this a 4-level severity
                # trends above 0 and flags every input.
                value = float(answer.get("score", 0.0))
                levels = _score_levels((questions or {}).get(name), answer)
                if levels:
                    value /= levels - 1
                scores[name] = round(value, 4)
                categories[name] = value >= threshold
        rows.append({"flagged": any(categories.values()), "categories": categories,
                     "category_scores": scores})
    return {"id": "modr-" + uuid.uuid4().hex, "model": model, "results": rows}


def moderation_questions_preset() -> Dict[str, Any]:
    return moderation_questions()


__all__ = [
    "ChatPlan",
    "MAX_TOOLS",
    "MODERATION_THRESHOLD",
    "TOOL_QUESTION",
    "UnsupportedRequest",
    "format_chat_response",
    "format_responses_response",
    "moderation_payload",
    "moderation_questions_preset",
    "models_payload",
    "parse_chat_request",
    "parse_moderation_request",
    "parse_responses_request",
]
