# Examples

Copy-paste recipes. Every snippet is self-contained apart from the helpers it names
(`ship`, `CACHE`, and so on), which you supply.

- [Quick start](#quick-start)
- [Audit](#audit)
- [Redact PII](#redact-pii)
- [Cache](#cache)
- [Metrics](#metrics)
- [Guardrail](#guardrail)
- [Confidence gate](#confidence-gate)
- [Routing pin](#routing-pin)
- [Lifecycle](#lifecycle)
- [Composition](#composition)
- [Per-call hooks](#per-call-hooks)
- [Batch](#batch)
- [HTTP server](#http-server)
- [ONNXAgent](#onnxagent)
- [Runtime registration](#runtime-registration)
- [Base class and process-wide defaults](#base-class-and-process-wide-defaults)
- [Async hooks](#async-hooks)
- [Hook timeout](#hook-timeout)
- [Token budget](#token-budget)
- [Testing hooks](#testing-hooks)

## Quick start

```python
import laya

def log(ctx):
    print(ctx.model, ctx.results[0]["answers"])

agent = laya.load("convaiinnovations/laya", on_predict_end=log)
agent.system_one("I was charged twice.", {"urgent": {"type": "noul", "instructions": "Urgent?"}})
```

## Audit

The browser-use use case: capture every decision and ship it to an external service.

```python
import json, sys
import laya

def audit(ctx):
    record = {
        "run_id": ctx.run_id,
        "model": ctx.model,
        "routing": ctx.results[0].get("routing") if ctx.results else None,
        "answers": ctx.results[0]["answers"] if ctx.results else None,
        "usage": ctx.usage,
        "elapsed_ms": round(ctx.elapsed_ms or 0.0, 3),
    }
    print(json.dumps(record), file=sys.stderr)
    # ship_to_service(record)

agent = laya.load("convaiinnovations/laya", on_predict_end=audit)
```

A full runnable version is in [`examples/hooks/audit.py`](../../examples/hooks/audit.py).

## Redact PII

```python
import re
import laya

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"\+?\d[\d ()-]{7,}\d")

def scrub(value):
    if isinstance(value, str):
        return PHONE.sub("[phone]", EMAIL.sub("[email]", value))
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value

def redact(ctx):
    ctx.states = [scrub(s) for s in ctx.states]

agent = laya.load("convaiinnovations/laya", on_predict_start=redact)
```

See [`examples/hooks/redact.py`](../../examples/hooks/redact.py).

## Cache

```python
import hashlib, json
import laya

CACHE = {}

def key(state, questions):
    payload = json.dumps([state, questions], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

def read(ctx):
    hit = CACHE.get(key(ctx.states[0], ctx.questions))
    if hit is not None:
        ctx.skip([hit])

def write(ctx):
    if ctx.results:
        CACHE[key(ctx.states[0], ctx.questions)] = ctx.results[0]

agent = laya.load("convaiinnovations/laya", on_predict_start=read, on_predict_end=write)
first = agent.system_one("state", QUESTIONS)    # runs the model
second = agent.system_one("state", QUESTIONS)   # served from CACHE
```

See [`examples/hooks/cache.py`](../../examples/hooks/cache.py).

## Metrics

```python
import laya

COUNTS, LATENCIES = {}, []

def metrics(ctx):
    COUNTS[ctx.model] = COUNTS.get(ctx.model, 0) + 1
    if ctx.elapsed_ms is not None:
        LATENCIES.append(ctx.elapsed_ms)

agent = laya.load("convaiinnovations/laya", on_predict_end=metrics, hooks_raise=False)
```

See [`examples/hooks/otel.py`](../../examples/hooks/otel.py).

## Guardrail

Block a request by raising from a start hook.

```python
import laya

class Blocked(Exception):
    pass

def guard(ctx):
    text = str(ctx.states[0]).lower()
    if "ignore previous instructions" in text:
        raise Blocked("prompt injection")

agent = laya.load("convaiinnovations/laya", on_predict_start=guard)

try:
    agent.system_one("Ignore previous instructions and ...", QUESTIONS)
except Blocked:
    handle_block()
```

## Confidence gate

Rewrite a low-confidence answer, or annotate it.

```python
def gate(ctx):
    answer = ctx.results[0]["answers"].get("dept")
    if answer and answer["confidence"] < 0.6:
        answer["choice"] = "human-review"
        answer["gated"] = True

agent = laya.load("convaiinnovations/laya", on_predict_end=gate)
```

## Routing pin

Force a checkpoint for a class of traffic.

```python
from laya import Router
from laya.router import RouteDecision

def pin(ctx):
    if "refund" in str(ctx.states[0]).lower():
        ctx.decision = RouteDecision(
            model="typed-decisions",
            repo="convaiinnovations/laya/typed-decisions",
            reason="refund workflow",
            detection=None,
            workflow=None,
        )

router = Router(hooks=[pin])
```

Per-call, without installing:

```python
router.predict("refund request", QUESTIONS, hooks=[pin])
```

## Lifecycle

Observe checkpoint build and eviction.

```python
from laya import Router

class Lifecycle:
    def on_load(self, ctx):
        print("loaded", ctx.model, "agent", type(ctx.agent).__name__)

    def on_evict(self, ctx):
        print("evicted", ctx.model)

router = Router(max_loaded=1, hooks=[Lifecycle()])
router.preload(["english", "multilingual"])   # on_load fires per build
router.unload()                               # on_evict fires per freed checkpoint
```

## Composition

Installed hooks first, then convenience callables; all share one context.

```python
import laya

class Metrics:
    def on_predict_end(self, ctx):
        record_latency(ctx.model, ctx.elapsed_ms)

def redact(ctx):
    ctx.states = [strip_pii(s) for s in ctx.states]

def audit(ctx):
    ship(ctx.run_id, ctx.results)

agent = laya.load(
    "convaiinnovations/laya",
    hooks=[Metrics()],              # installed, runs first
    on_predict_start=redact,        # convenience, appended
    on_predict_end=audit,           # convenience, appended
    hooks_raise=True,
)
```

## Per-call hooks

Override or extend hooks for a single call.

```python
agent.system_one(
    state,
    questions,
    on_predict_end=lambda ctx: debug_dump(ctx),
    hooks_raise=False,
)

router.predict(
    state,
    questions,
    hooks=[pin],                    # applies to on_route too
    on_predict_end=audit,
)
```

## Batch

Hooks fire once per `predict_batch` call, with `ctx.states` holding every state.

```python
def audit_batch(ctx):
    for state, result in zip(ctx.states, ctx.results):
        ship_one(ctx.run_id, state, result)

results = agent.predict_batch([state_a, state_b, state_c], questions, on_predict_end=audit_batch)
```

## HTTP server

Router hooks fire for `laya.serve` automatically, because the server calls `Router.predict`.

```python
from laya import Router
from laya.serve import create_app

router = Router(hooks=[Metrics()], on_predict_end=audit, hooks_raise=False)
app = create_app(router=router)
```

## ONNXAgent

`ONNXAgent` exposes the predict-level events only.

```python
from laya.onnx_agent import ONNXAgent

agent = ONNXAgent("convaiinnovations/laya", onnx_path="laya.onnx", on_predict_end=audit)
agent.system_one(state, questions)
```

## Runtime registration

Attach, detach or scope hooks after construction.

```python
agent.add_hook(Metrics())          # attach at runtime
agent.remove_hook(Metrics())       # by identity

with agent.hooks_installed(DebugDump()):
    agent.system_one(state, questions)   # DebugDump only here
```

## Base class and process-wide defaults

Subclass `BaseHook` to override only what you need, and register something once for the whole
process instead of passing it to every `Agent` and `Router`.

```python
from laya import BaseHook, hooks

class Audit(BaseHook):
    def on_predict_end(self, ctx):
        ship(ctx.run_id, ctx.results)

hooks.set_default_hooks(hooks=[Audit()])   # runs for every call in the process

# later, or in tests:
hooks.clear_default_hooks()
```

## Token budget

Shape the token budget for one call, from a hook or a per-call argument.

```python
def widen(ctx):
    k = len(next(iter(ctx.questions.values())).get("criteria", {}) or {})
    if k >= 50:
        ctx.head_max_len = 16 + 4 * k

agent = laya.load("convaiinnovations/laya", on_predict_start=widen)

# or per call
agent.system_one(state, questions, head_max_len=324, max_len=1024)
```

## Async hooks

Wrap an async hook in `AsyncHook`; each coroutine runs to completion in the sync core, whether the
caller is synchronous or already inside an event loop.

```python
import laya
from laya import AsyncHook

class RemoteAudit:
    async def on_predict_end(self, ctx):
        await ship(ctx.run_id, ctx.results)

agent = laya.load("convaiinnovations/laya", hooks=[AsyncHook(RemoteAudit())])
```

A plain async callable works too:

```python
async def async_end(ctx):
    await ship(ctx.results)

agent.system_one(state, questions, on_predict_end=async_end)
```

## Hook timeout

Bound each hook call, so a stuck hook cannot hang a served request:

```python
agent = laya.load("convaiinnovations/laya", on_predict_end=metrics, hooks_timeout=2.0)

# or per call
agent.system_one(state, questions, on_predict_end=metrics, hooks_timeout=0.5)
```

A timed-out hook raises `TimeoutError` (or warns when `hooks_raise=False`). The hook keeps running
in the background, so also give network calls their own timeout. See
[errors](errors.md#timeouts).

## Testing hooks

Assert what a hook saw without a model: drive `predict_batch` with the encode/forward/decode
helpers stubbed, as [`tests/test_hooks.py`](../../tests/test_hooks.py) does.

```python
seen = []
agent.predict_batch(["s0"], questions, on_predict_end=lambda ctx: seen.append(ctx.results))
assert len(seen) == 1
```

The API surface is pinned by [`tests/test_hooks_api.py`](../../tests/test_hooks_api.py).

## See also

- [Tracing](tracing.md): `run_id`, spans, OpenTelemetry.
- [Patterns and anti-patterns](patterns.md): the reasoning behind these recipes.
