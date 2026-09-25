# Routing

`Router` chooses a checkpoint for each request and loads its `Agent` when prediction needs it.
The default router sends English text to the English checkpoint and other supported languages to
the multilingual checkpoint. You can override that choice, supply your own language hint, or select
the typed-decisions checkpoint explicitly.

This guide covers model selection and lifecycle. For the question types accepted by prediction,
see [Structured decisions](structured.md); for lifecycle callbacks, see [Prediction hooks](hooks/index.md).

## Quick start

```python
from laya import Router

router = Router()

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    }
}

result = router.predict("We were billed twice. Please refund the duplicate charge.", questions)
print(result["answers"]["department"]["choice"])
print(result["routing"]["model"])
```

Creating `Router()` does not download checkpoints by default. `predict()` routes the request, then
loads the selected checkpoint on first use. The first prediction can therefore take longer while
files are downloaded and the model is initialized; later predictions reuse the loaded agent.

## How a checkpoint is selected

`Router.route(state, questions, ...)` returns a `RouteDecision` without loading a checkpoint or
running inference. The decision contains the selected model, a human-readable reason, and language
detection details when built-in detection was used.

Routing checks inputs in this order:

1. `model=` selects a checkpoint directly.
2. `task=` selects a checkpoint for the named task.
3. If `auto_task_detection=True`, an exact match against one of the known typed-decisions question
   ID sets selects `typed-decisions`.
4. A recognized `lang=` value selects English or multilingual.
5. A per-call `lang_guess=` or the router's configured `lang_guess` is consulted.
6. Built-in script and language analysis selects a checkpoint. If there is no reliable language
   signal, the router uses its configured `default` (English by default).

The first matching rule wins. For example, `model="multilingual"` overrides `lang="en"`. Invalid
model names raise `ValueError` instead of falling through to detection.

```python
decision = router.route(
    "La aplicación se cierra cada vez que abro la configuración.",
    questions,
)
print(decision.model)   # multilingual
print(decision.reason)  # why that checkpoint was selected
```

`RouteDecision` is dict-compatible, so its fields are also available with keys such as
`decision["model"]` and `decision["reason"]`. `Router.predict()` includes the same decision under
the result's `"routing"` key.

## Override language routing

Use `lang=` when the application already knows the request language. Language tags such as
`"en"`, `"en-US"`, and `"en_US.UTF-8"` are accepted. English routes to `english`; other recognized
language codes route to `multilingual`.

```python
result = router.predict(state, questions, lang="de")
assert result["routing"]["model"] == "multilingual"
```

If the application has its own language detector, pass its result as a language code with
`lang_guess=`. A callable receives the state and can return a code or `None` to abstain:

```python
def detect_request_language(state):
    # Replace this with the application's detector.
    return "en" if "invoice" in str(state).lower() else None

router = Router(lang_guess=detect_request_language)
```

An unrecognized or empty hint does not force a checkpoint; routing continues to the next rule.
This keeps an abstaining detector from silently pinning requests to the wrong model.

Built-in detection is a lightweight script and language heuristic, not a general-purpose language
identification model. It analyzes string values in text, dict, and list states; dictionary keys are
ignored because they are often field names in English. Short or ambiguous text can use the default
checkpoint. For known workloads, an explicit language or application-provided hint is more
predictable.

## Selecting typed-decisions

The typed-decisions checkpoint is not selected automatically by default. Select it explicitly:

```python
result = router.predict(state, questions, model="typed-decisions")
# `task="typed_decisions"` is also accepted.
```

Alternatively, set `auto_task_detection=True`. The router then checks whether the question IDs are
an exact match for one of its known typed-decision workflows. It does not infer the task from the
question wording, and adding unrelated question IDs prevents an exact match.

```python
router = Router(auto_task_detection=True)
```

## Inspect routing without loading models

Use `route()` to inspect one decision or `route_batch()` to inspect a sequence. Neither method
loads checkpoints, so both are useful for debugging routing rules before running inference.

```python
requests = [
    {"state": "Please refund the duplicate charge.", "questions": questions},
    {"state": "Necesito ayuda con mi factura.", "questions": questions},
]

decisions = router.route_batch(requests)
for decision in decisions:
    print(decision.model, decision.reason)
```

Every `route_batch()` item needs `state` and `questions`; optional routing overrides (`model`,
`task`, `lang`, and `lang_guess`) are specified per item. Decisions stay in input order.

## Loading and memory

By default, the router loads a checkpoint the first time it is needed and keeps at most two agents
resident. Automatic language routing normally needs only the English and multilingual checkpoints.
If requests can also select `typed-decisions`, a small `max_loaded` may evict another agent and
cause it to be loaded again the next time it is needed.

```python
# Load only the checkpoints this process serves, before accepting requests.
router = Router()
router.preload(["english", "multilingual"])

print(router.loaded)  # currently resident checkpoint names
router.unload("multilingual")
```

`Router(preload=True)` preloads all configured checkpoints. Preloading raises the resident-model
limit to fit the requested set. To control a three-checkpoint workload without preloading, set
`max_loaded=3`. Use `unload()` to release one agent or all agents (`router.unload()`). A router used
as a context manager unloads its agents when the block exits:

```python
with Router(preload=True) as router:
    result = router.predict(state, questions)
```

You can also pass `device="cpu"`, `device="cuda"`, or another supported PyTorch device when
constructing the router. Availability and memory determine which devices can run a given model.

## Mixed batches

`predict_batch()` accepts requests with different states, models, languages, and question schemas.
The router first makes a decision for each request, groups work by checkpoint and compatible
question schema, then restores the results to the original input order.

```python
requests = [
    {"state": "Please refund the duplicate charge.", "questions": questions},
    {"state": "Mi cuenta fue cobrada dos veces.", "questions": questions},
    {"state": "A third request", "questions": questions, "model": "typed-decisions"},
]

results = router.predict_batch(requests, batch_size=8)
```

Each request requires `state` and `questions`; it may also include `model`, `task`, `lang`, or
`lang_guess`. Requests sharing a checkpoint and question schema can share an Agent batch forward
pass. Different schemas or checkpoints are handled in separate groups. `batch_size` limits the
number of states passed together to the Agent; results still correspond to the request order.

## Choosing an entry point

- Use `route()` or `route_batch()` when you need to inspect decisions without loading models.
- Use `predict()` for a single request and `predict_batch()` for multiple possibly heterogeneous
  requests.
- Use `Agent` directly when the application already chose and loaded one checkpoint and does not
  need automatic routing.

See the [Router API reference](reference/router.md) for constructor and method details.
