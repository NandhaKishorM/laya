# Prediction Hooks

Hooks let you observe or shape every decision without forking: audit logging, PII redaction
before inference, caching, metrics, confidence gating, routing overrides, and forwarding to an
external service. They are opt-in; an unset hook is a no-op.

```python
import laya

def log(ctx):
    print(ctx.model, ctx.results[0]["answers"], ctx.elapsed_ms)

agent = laya.load("convaiinnovations/laya", on_predict_end=log)
agent.system_one("I was charged twice.", {"urgent": {"type": "noul", "instructions": "Urgent?"}})
```

## Two ways to write a hook

A hook is either a plain callable or an object implementing any subset of the lifecycle
methods. Both are configured the same way.

```python
# callable
laya.load("convaiinnovations/laya", on_predict_start=lambda ctx: ...)

# object with any subset of the events
class Audit:
    def on_predict_start(self, ctx): ...
    def on_predict_end(self, ctx): ...
    def on_error(self, ctx): ...

laya.load("convaiinnovations/laya", hooks=[Audit()])
```

`hooks=` accepts a single object or a sequence. `on_predict_start=` / `on_predict_end=` accept a
single callable or a sequence. Installed hooks run first; per-call hooks are appended after them.
On the `Router`, per-call `hooks=` also apply to `on_route` (so a hook can pin a checkpoint for a
single call), and `route()` itself takes `hooks=`.

## The context

Every hook receives a mutable `PredictContext`:

| field | meaning |
|---|---|
| `states` | the states for this call (`system_one` = one). A start hook may rewrite it. |
| `questions` | the questions. A start hook may rewrite them. |
| `run_id` | a unique id shared by every hook of this call, for tracing and correlation. |
| `results` | per-state result dicts. Set before `on_predict_end`; an end hook may rewrite it. |
| `decision` | Router only: the routing decision. `on_route` may rewrite it. |
| `model` | the resolved checkpoint name. |
| `agent` / `router` | the runtime that is running the call. |
| `usage` | aggregated `{"input_tokens", "output_tokens"}`. |
| `elapsed_ms` | wall time for the call. |
| `error` | the exception, on the failure path. |

`ctx.skip(results)` from a start hook short-circuits inference: the forward pass is skipped, the
end hooks still run, and the cached results are returned.

## Events

| event | where | populated |
|---|---|---|
| `on_predict_start` | Agent and Router | `states`, `questions`, `model`, `agent`, `router` |
| `on_predict_end` | Agent and Router | plus `results`, `usage`, `elapsed_ms`, `error` |
| `on_route` | Router | `decision`, `states`, `questions` |
| `on_load` | Router | `model`, `agent` |
| `on_evict` | Router | `model` |
| `on_error` | Agent and Router | predict fields with `error` set |

### Agent vs Router

- **Agent hooks** wrap the forward pass (`predict_batch`, so `system_one` / `predict` inherit).
- **Router hooks** wrap route + infer and additionally see `ctx.decision`. They are not forwarded
  into the `Agent`; if you attach an `Agent` that has its own hooks, both run.

`laya.serve` and the MCP server call `Router.predict`, so Router hooks fire for them too.

## Use cases

### Audit logging

```python
import json

def audit(ctx):
    print(json.dumps({
        "model": ctx.model,
        "answers": ctx.results[0]["answers"] if ctx.results else None,
        "routing": ctx.results[0].get("routing") if ctx.results else None,
        "elapsed_ms": round(ctx.elapsed_ms or 0.0, 2),
    }))

laya.load("convaiinnovations/laya", on_predict_end=audit)
```

See [`examples/hooks/audit.py`](../examples/hooks/audit.py).

### Redacting PII before inference

```python
import re
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

def redact(ctx):
    ctx.states = [EMAIL.sub("[email]", s) for s in ctx.states]

laya.load("convaiinnovations/laya", on_predict_start=redact)
```

See [`examples/hooks/redact.py`](../examples/hooks/redact.py).

### Caching

```python
def cache_read(ctx):
    hit = CACHE.get(key(ctx.states[0], ctx.questions))
    if hit is not None:
        ctx.skip([hit])

def cache_write(ctx):
    if ctx.results:
        CACHE[key(ctx.states[0], ctx.questions)] = ctx.results[0]

laya.load("convaiinnovations/laya", on_predict_start=cache_read, on_predict_end=cache_write)
```

See [`examples/hooks/cache.py`](../examples/hooks/cache.py).

### Confidence gating

An end hook can reject or replace a low-confidence answer:

```python
def gate(ctx):
    answer = ctx.results[0]["answers"]["dept"]
    if answer["confidence"] < 0.6:
        answer["choice"] = "human-review"

laya.load("convaiinnovations/laya", on_predict_end=gate)
```

### Metrics

```python
def metrics(ctx):
    COUNTERS[ctx.model] = COUNTERS.get(ctx.model, 0) + 1
    HISTOGRAM.append(ctx.elapsed_ms or 0.0)

laya.load("convaiinnovations/laya", on_predict_end=metrics)
```

See [`examples/hooks/otel.py`](../examples/hooks/otel.py).

### Routing override

`on_route` may replace `ctx.decision` to pin a checkpoint:

```python
from laya.router import RouteDecision

def pin(ctx):
    if "refund" in str(ctx.states[0]).lower():
        ctx.decision = RouteDecision(model="typed-decisions", repo="convaiinnovations/laya/typed-decisions",
                                     reason="refund workflow", detection=None, workflow=None)

Router(hooks=[pin])
```

## Errors

By default a failing hook propagates (`hooks_raise=True`). Set `hooks_raise=False` to warn and
continue, which is what you usually want for telemetry:

```python
laya.load("convaiinnovations/laya", on_predict_end=metrics, hooks_raise=False)
```

`on_error` fires when inference raises; the end hooks still run with `ctx.error` set, and the
original exception is re-raised. A failing `on_error` / `on_predict_end` hook cannot mask it.

## Concurrency

Hooks may run concurrently: `Agent` and `Router` are safe to call from many threads, and each
call gets its own context. If your hook is not thread-safe, serialise it:

```python
laya.load("convaiinnovations/laya", on_predict_end=metrics, hooks_concurrent=False)
```

## Synchronous only

Hooks are synchronous, like the rest of the core. Keep them fast and non-blocking: they run on
the calling thread, and `laya.serve` runs inference on a single worker, so a slow hook delays
other requests. For work that must be async or slow, offload it from the hook or do it in
`laya.serve` middleware. An `AsyncHook` adapter is a possible follow-up.

## Tracing

Every hook of one call receives the same `run_id`, so a tracer can correlate the start, end and
error events (and any spans it opens) without threading its own state:

```python
class Trace:
    def __init__(self):
        self.spans = {}

    def on_predict_start(self, ctx):
        self.spans[ctx.run_id] = start_span(ctx.model, ctx.run_id)

    def on_predict_end(self, ctx):
        end_span(self.spans.pop(ctx.run_id, None), ctx.usage, ctx.elapsed_ms)

    def on_error(self, ctx):
        end_span(self.spans.pop(ctx.run_id, None), error=ctx.error)

laya.load("convaiinnovations/laya", hooks=[Trace()])
```
