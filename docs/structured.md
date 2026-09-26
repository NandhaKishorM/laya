# Schema-driven decisions

Turn a JSON schema, or a pydantic model, into Laya questions, and get back typed values with
calibrated confidence. This is the bridge that makes Laya a structured-output engine: you describe
the shape you want, Laya answers it in one forward pass.

```python
import laya

schema = {
    "type": "object",
    "properties": {
        "department": {"type": "string", "enum": ["billing", "support", "sales"],
                       "description": "Which team should handle this?"},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"},
    },
}

agent = laya.load("convaiinnovations/laya")
values = agent.decide("I was charged twice, refund me.", schema=schema)
# {"department": "billing", "urgency": 2, "needs_human": True}
```

With pydantic (install `laya[structured]`):

```python
from typing import Literal
from pydantic import BaseModel

class Ticket(BaseModel):
    department: Literal["billing", "support", "sales"]
    urgency: Literal[0, 1, 2]
    needs_human: bool

ticket = agent.decide("I was charged twice, refund me.", schema=Ticket)
```

## The supported subset

The top level must be an object with `properties`. Each property becomes one question.

| JSON schema | Laya question | Returned value |
|---|---|---|
| `enum`, `Literal`, `const` | `choice` | the chosen value, with its original type |
| `boolean` | `noul` | `true` / `false` |
| `integer` or `number` with `minimum` and `maximum`, span up to `MAX_SCORE_LEVELS` | `score` | the highest-probability level, as an integer |
| `string` with `enum` | `choice` | the chosen string |
| `description` | question instructions | |
| `title` | option label | |

Projection is exact: an `enum: [1, 2, 3]` returns `2`, not `"2"`; a bounded integer returns a
level between `minimum` and `maximum`; a boolean is `noul >= 0.5`.

## Rejections

A schema that cannot be answered from a fixed option set raises `laya.structured.SchemaError`
(a `ValueError`) naming the exact path:

| case | message shape |
|---|---|
| free `string` without `enum` | `properties.name: a free string cannot be a fixed option set; use 'enum' or a boolean` |
| `array` | `properties.name: arrays are not supported; ask one field per element` |
| nested `object` | `properties.name: nested objects are not supported; flatten the schema` |
| `$ref` / recursion | `properties.name: $ref/recursion is not supported; flatten the schema` |
| enum values with the same choice label, such as `1` and `"1"` | `properties.name: enum values produce duplicate choice labels` |
| unbounded number | `properties.name: a numeric field needs integer 'minimum' and 'maximum' to become a score` |
| more than `MAX_PROPERTIES` / `MAX_OPTIONS` / `MAX_SCORE_LEVELS` | the limit is named in the message |

Limits: `MAX_PROPERTIES = 32`, `MAX_OPTIONS = 32`, `MAX_SCORE_LEVELS = 10`.

## The API

| function | purpose |
|---|---|
| `laya.decide(runner, state, schema=..., *, questions=..., return_details=..., **predict_kwargs)` | the free function, works for `Agent` and `Router` |
| `agent.decide(state, schema=..., ...)` / `router.decide(state, schema=..., ...)` | convenience methods |
| `questions_from_json_schema(schema)` | schema to Laya questions |
| `questions_from_pydantic(model)` | pydantic model to questions (requires pydantic) |
| `answers_to_json(answers, schema)` | project raw answers onto schema values |
| `answer_to_pydantic(model, answers)` | project raw answers into a pydantic instance |
| `plan_from_json_schema(schema)` | the validated field plan (advanced) |

Pass exactly one of `schema` or `questions`. With `questions`, `decide` returns the raw answers
instead of projecting. Extra keyword arguments are forwarded to `predict`, so hooks, `model=`,
`task=` and the token budget all work:

```python
router.decide(state, schema=Ticket, model="multilingual", hooks=[Metrics()])
```

## Confidence and probabilities

By default `decide` returns only the values. Pass `return_details=True` for a `DecisionResult`
with per-field confidence, probabilities, the raw answers, and the usage and routing of the call:

```python
result = agent.decide(state, schema=Ticket, return_details=True)
result.values["department"]        # "billing"
result.confidence["department"]    # 0.94
result.probabilities["department"] # {"billing": 0.94, "support": 0.06, "sales": 0.0}
result.usage                       # {"input_tokens": 42, "output_tokens": 0}
result.routing                     # the Router decision, when a Router answered
```

You can gate on it, for example escalate a field whose confidence is below a threshold:

```python
if result.confidence["department"] < 0.6:
    result.values["department"] = "human-review"
```

## How it maps internally

- Enum and `Literal` become `choice` questions with the values as string labels; the label is
  mapped back to the original value on the way out, so integers stay integers.
- A bounded integer becomes a `score` question with one level per value; the returned value is
  `minimum + argmax`.
- A boolean becomes a `noul` question; the value is `noul >= 0.5`.
- A `description` becomes the question instructions, so a good description is what makes the
  decision accurate. This follows the same rule as the [hooks guide](hooks/index.md): be explicit
  about what each option means.

## See also

- [Prediction hooks](hooks/index.md): observe, shape, cache or gate the decisions this produces.
- [Decision primitives](index.md): `choice`, `score` and `noul` in depth.
