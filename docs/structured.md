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

Every row below is a real schema: `tests/test_structured_docs.py` compiles the first column and
asserts the question the compiler actually produces, so this table cannot drift from the code. A cell
is either a property schema on its own, or a call to an entry point.

| JSON schema | Laya question | Returned value |
|---|---|---|
| `{"enum": ["billing", "support"]}` | `choice` | the chosen value, with its original type |
| `{"const": "billing"}` | `choice` | that one value |
| `{"type": "boolean"}` | `noul` | `true` / `false` |
| `{"type": "integer", "minimum": 0, "maximum": 5}` | `score` | the highest-probability level, as an integer |
| `{"type": "number", "minimum": 0, "maximum": 5}` | `score` | the level, as an integer |
| `{"type": "string", "enum": ["low", "high"], "description": "How urgent?"}` | `choice` | `How urgent?` is the question wording |
| `{"anyOf": [{"enum": ["x", "y"]}, {"type": "null"}]}` | `choice` | as the plain `enum` row; no answer leaves the key out |
| `{"oneOf": [{"type": "boolean"}, {"type": "null"}]}` | `noul` | as the plain `boolean` row |
| `{"type": ["integer", "null"], "minimum": 1, "maximum": 3}` | `score` | as the plain bounded-integer row |

`Literal[...]` and `Optional[...]` are the pydantic spellings of the `enum` and `anyOf` rows:
`questions_from_pydantic` renders them to those shapes and the same rows apply.

`title` is **not** read. pydantic puts one on every field of `model_json_schema()` whether you asked
for it or not, and a per-property name cannot label the per-option choices a question is built from,
so the wording lever is `description` — see *How it maps internally* below.

Projection is exact: an `enum: [1, 2, 3]` returns `2`, not `"2"`; a bounded integer returns a
level between `minimum` and `maximum`; a boolean is `noul >= 0.5`.

## Rejections

A schema that cannot be answered from a fixed option set raises `laya.structured.SchemaError`
(a `ValueError`) naming the exact path. Each row is executed too, with the field named `name`:

| property schema | message |
|---|---|
| `{"type": "string"}` | `properties.name: a free string cannot be a fixed option set; use 'enum' or a boolean` |
| `{"type": "array", "items": {"type": "string"}}` | `properties.name: arrays are not supported; ask one field per element` |
| `{"type": "object", "properties": {"inner": {"type": "boolean"}}}` | `properties.name: nested objects are not supported; flatten the schema` |
| `{"$ref": "#/definitions/node"}` | `properties.name: $ref/recursion is not supported; flatten the schema` |
| `{"enum": [1, "1"]}` | `properties.name: enum values produce duplicate choice labels` |
| `{"enum": []}` | `properties.name: 'enum' must not be empty` |
| `{"type": "number"}` | `properties.name: a numeric field needs integer 'minimum' and 'maximum' to become a score` |
| `{"type": "integer", "minimum": 5, "maximum": 2}` | `properties.name: 'maximum' 2 is below 'minimum' 5` |
| `{"type": "integer", "minimum": 0, "maximum": 10}` | `properties.name: 11 levels exceeds MAX_SCORE_LEVELS=10; narrow the range or use an enum` |
| `{"anyOf": [{"type": "string"}, {"type": "integer"}]}` | `properties.name: only 'Optional[...]' unions (one non-null branch) are supported, got 2` |
| `{"type": ["string", "integer"]}` | `properties.name: 'type' has multiple non-null types; unions are not supported` |
| `{"format": "date"}` | `properties.name: unsupported schema {'format': 'date'}` |
| `"boolean"` | `properties.name: property must be an object, got str` |

The entry points themselves reject these:

| call | message |
|---|---|
| `plan_from_json_schema("not a schema")` | `expected a JSON schema object, got str` |
| `plan_from_json_schema({"type": "object"})` | `the top level must be an object with 'properties'` |
| `plan_from_json_schema({"type": "object", "properties": {}})` | `'properties' must be a non-empty object` |
| `plan_from_json_schema({"type": "object", "properties": {"p%d" % i: {"type": "boolean"} for i in range(33)}})` | `33 properties exceeds MAX_PROPERTIES=32` |
| `plan_from_json_schema({"type": "object", "properties": {"name": {"enum": ["v%d" % i for i in range(33)]}}})` | `properties.name: 33 options exceeds MAX_OPTIONS=32` |
| `decide(None, "I was charged twice.", schema=42)` | `expected a JSON schema dict or a pydantic model, got int` |

Limits: `MAX_PROPERTIES = 32`, `MAX_OPTIONS = 32`, `MAX_SCORE_LEVELS = 10`.

## The API

| function | purpose |
|---|---|
| `laya.decide(runner, state, schema=..., *, questions=..., return_details=..., min_confidence=..., **predict_kwargs)` | the free function, works for `Agent` and `Router` |
| `agent.decide(state, schema=..., ...)` / `router.decide(state, schema=..., ...)` | convenience methods |
| `laya.decide_batch(runner, states, schema=..., ...)` / `agent.decide_batch(...)` / `router.decide_batch(...)` | the same over many states, one batched call |
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

## Scoring many states

`decide_batch` is the throughput form: the schema is planned once and its questions run over every
state through `predict_batch`, so states share forward passes instead of one per call. Results come
back in input order, projected exactly as `decide` projects, and `return_details=True` gives one
`DecisionResult` per state:

```python
values = agent.decide_batch(ticket_texts, schema=Ticket)          # values[i] matches ticket_texts[i]
results = router.decide_batch(states, schema=Ticket, return_details=True, batch_size=64)
```

On a `Router` each state is still routed on its own, so one call can span checkpoints. Keyword
arguments reach `predict_batch`, so `batch_size=`, `model=` and hooks work as they do for `decide`.
`Agent`, `ONNXAgent` and `Router` all have it; a runner without `predict_batch` raises
`TypeError` rather than silently falling back to a loop — call `decide` per state there. Batching can shift borderline argmaxes the
same way `predict_batch` does; the README records the measured speedups for both devices.

## Confidence and probabilities

By default `decide` returns only the values. Pass `return_details=True` for a `DecisionResult`
with per-field confidence, probabilities, the raw answers, and the usage and routing of the call:

```python
result = agent.decide(state, schema=Ticket, return_details=True)
result.values["department"]            # "billing"
result.answer_confidence["department"] # 0.94  max(p): the quantity min_confidence gates on
result.confidence["department"]        # 0.71  normalized entropy, which depends on label count
result.probabilities["department"]     # {"billing": 0.94, "support": 0.06, "sales": 0.0}
result.usage                           # {"input_tokens": 42, "output_tokens": 0}
result.routing                         # the Router decision, when a Router answered
```

`confidence` and `answer_confidence` are different quantities, and the names follow the definitions.
`answer_confidence` is `max(p)`, the probability mass on the answer being reported. It is what
temperature scaling fits, what every calibration figure in this repository is computed on, and what
`min_confidence` is compared against — which is the reason to gate on it rather than on
`confidence`. `confidence` is normalized entropy, which depends on how many options the question
had: `tests/test_confidence.py` pins that a two-option distribution comes back as 0.90 on a `noul`
and 0.53 on an equivalent `choice`, so it does not compare against a threshold. A field that
reported no usable `answer_confidence` maps to `None`, which is not the same as a reported `0.0`.

Gate on the same quantity the gate uses:

```python
if result.answer_confidence["department"] < 0.6:
    result.values["department"] = "human-review"
```

`answer_confidence` being the right number to *filter* on is not the same as it being a trustworthy
probability. Reading it as "about c of the answers returned at c are correct" holds only after
temperatures have been fitted and validated on held-out data for that checkpoint and question shape.
The shipped checkpoints are over-confident as shipped and `laya-multilingual` ships with no fitted
temperatures at all — see the README's
[Calibration](https://github.com/NandhaKishorM/laya#calibration) and
[Honest limits](https://github.com/NandhaKishorM/laya#honest-limits) sections, and the
[fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
for the fitting loop. Fit before you rely on the level; report it because it is the quantity the
gate and the eval harness both use.

## How it maps internally

- Enum and `Literal` become `choice` questions with the values as string labels; the label is
  mapped back to the original value on the way out, so integers stay integers.
- A bounded integer becomes a `score` question with one level per value; the returned value is
  `minimum + argmax`.
- A boolean becomes a `noul` question; the value is `noul >= 0.5`.
- A `description` becomes the question instructions, so a good description is what makes the
  decision accurate. This follows the same rule as the [hooks guide](hooks/index.md): be explicit
  about what each option means.
- A `null` branch is dropped before the field is planned, so `Optional[X]` asks exactly the question
  `X` asks. The field's key is simply absent from the values when there is no answer for it, which is
  what makes it safe to declare a field optional without changing what the model sees.

## See also

- [Prediction hooks](hooks/index.md): observe, shape, cache or gate the decisions this produces.
- [Decision primitives](index.md): `choice`, `score` and `noul` in depth.
- [LangChain and LangGraph](langchain.md): `LayaDecision` is this call as a runnable in a chain.
