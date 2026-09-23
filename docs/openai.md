# OpenAI-compatible API

`laya-serve` speaks the OpenAI wire format for structured decisions. Point any OpenAI client at it
and get fast, local, schema-shaped answers from Laya, with no LLM and no free text.

Laya is a decision engine, not a chat model, so the supported surface is the structured subset:
`response_format` with a JSON schema, and `tools` (function calling). A request that needs free
text, sampling or streaming is rejected with a 400 that says what to send.

| method | path | purpose |
|---|---|---|
| `GET` | `/v1/models` | the available checkpoints |
| `POST` | `/v1/chat/completions` | `response_format` (json_schema) or `tools` |
| `POST` | `/v1/responses` | the newer Responses shape, `text.format` or `tools` |
| `POST` | `/v1/moderations` | the moderation preset, OpenAI `results` shape |

The Jev-compatible `POST /v1/systemone` route stays as it is.

## Start the server

```bash
pip install "laya[serve]"
laya-serve            # http://127.0.0.1:8000
```

Set `LAYA_API_KEY` to require a bearer token, exactly as the chat clients send it.

## Structured output

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "laya",
  "messages": [{"role": "user", "content": "I was charged twice, refund me."}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "ticket", "schema": {
    "type": "object",
    "properties": {
      "department": {"type": "string", "enum": ["billing", "support", "sales"]},
      "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
      "needs_human": {"type": "boolean"}
    }
  }}}
}' | python -m json.tool
```

The response is a normal chat completion whose `message.content` is the JSON:

```json
{"object": "chat.completion", "model": "laya",
 "choices": [{"index": 0, "finish_reason": "stop",
   "message": {"role": "assistant",
     "content": "{\"department\": \"billing\", \"urgency\": 2, \"needs_human\": true}"}}],
 "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12}}
```

This is the subset the [schema-driven decisions](structured.md) layer supports. A schema it cannot
express raises a 422 naming the exact path.

## Function calling

Send `tools` and Laya chooses a tool and fills its typed arguments in one forward pass.

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "The site is down, this is urgent"}],
  "tools": [{"type": "function", "function": {
    "name": "route_ticket", "description": "Route a support ticket",
    "parameters": {"type": "object", "properties": {
      "department": {"type": "string", "enum": ["billing", "support"]},
      "urgent": {"type": "boolean"}}}}}]
}'
```

```json
{"choices": [{"index": 0, "finish_reason": "tool_calls",
  "message": {"role": "assistant", "content": null,
    "tool_calls": [{"id": "call_...", "type": "function",
      "function": {"name": "route_ticket",
        "arguments": "{\"department\": \"support\", \"urgent\": true}"}}]}}]}
```

With several tools, Laya first answers a `choice` over the tool names and then reads only the
chosen tool's arguments, all in the same pass. Up to `MAX_TOOLS = 8` tools are accepted; each
tool's parameters must fit the supported subset. Pass `tool_choice` naming a function to skip the
selection step.

## Responses API

`POST /v1/responses` takes `input` and `text.format`, and returns an `output` array:

```bash
curl -s localhost:8000/v1/responses -H 'content-type: application/json' -d '{
  "input": "I was charged twice",
  "text": {"format": {"type": "json_schema", "name": "ticket", "schema": {
    "type": "object", "properties": {"department": {"type": "string", "enum": ["billing", "support"]}}}}}
}'
```

The output is a `message` item with an `output_text` part, or a `function_call` item when `tools`
are sent. `usage` uses the `input_tokens` / `output_tokens` / `total_tokens` names.

## Moderations

```bash
curl -s localhost:8000/v1/moderations -H 'content-type: application/json' \
  -d '{"input": "you are an idiot"}'
```

```json
{"id": "modr-...", "model": "laya",
 "results": [{"flagged": true,
   "categories": {"toxic": true, "harassment": false, "threat": false, "spam": false, "severity": true},
   "category_scores": {"toxic": 0.81, "harassment": 0.12, "threat": 0.03, "spam": 0.05, "severity": 1.7}}]}
```

The categories are Laya's moderation preset (`toxic`, `harassment`, `threat`, `spam`, `severity`),
not OpenAI's fixed taxonomy. A category is flagged at or above `LAYA_MODERATION_THRESHOLD`
(default `0.5`). `input` may be a list, and `results` is aligned with it.

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

completion = client.chat.completions.create(
    model="laya",
    messages=[{"role": "user", "content": "I was charged twice, refund me."}],
    response_format={"type": "json_schema", "json_schema": {"name": "ticket", "schema": SCHEMA}},
)
print(completion.choices[0].message.content)
```

See [`examples/openai_client.py`](../examples/openai_client.py) for a runnable version with tools.

## Limits and rejections

| case | status |
|---|---|
| no `tools` and no schema | 400, the message says to send one |
| `response_format` type `json_object` | 400, Laya needs options, not free JSON |
| both `tools` and a schema | 400, send one |
| more than `MAX_TOOLS = 8` tools, duplicate or non-function tools | 400 |
| a schema the structured layer cannot express | 422, naming the path |
| body larger than `MAX_BODY_BYTES` | 413 |
| `LAYA_API_KEY` set and the bearer token is wrong or missing | 401 |

## Honest limits

- Not a chat model: no free text, no reasoning chains, no variable-length or nested output.
- No streaming, no `n`, no sampling parameters; they are ignored or rejected.
- Inference is serialized on one worker, like the rest of `laya-serve`, so throughput is one
  decision at a time per process.

## See also

- [Schema-driven decisions](structured.md): the mapping and the direct Python API.
- [Prediction hooks](hooks/index.md): observe or shape these decisions.
