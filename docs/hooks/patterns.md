# Patterns and anti-patterns

Hooks are a small seam, and it is easy to use them well or badly. This page collects the shapes
that hold up in production and the ones that bite.

- [Patterns](#patterns)
  - [Audit log](#audit-log)
  - [PII redaction](#pii-redaction)
  - [Caching](#caching)
  - [Metrics](#metrics)
  - [Guardrails](#guardrails)
  - [Confidence gating](#confidence-gating)
  - [Routing override](#routing-override)
  - [Model lifecycle](#model-lifecycle)
  - [Multi-tenant context](#multi-tenant-context)
  - [Composition](#composition)
  - [Scoped instrumentation](#scoped-instrumentation)
  - [Process-wide instrumentation](#process-wide-instrumentation)
  - [Token-budget shaping](#token-budget-shaping)
- [Anti-patterns](#anti-patterns)
  - [Blocking work](#blocking-work)
  - [Raising from end hooks for control flow](#raising-from-end-hooks-for-control-flow)
  - [Shared mutable state without a lock](#shared-mutable-state-without-a-lock)
  - [Silent failure](#silent-failure)
  - [Retaining contexts](#retaining-contexts)
  - [Redacting too late](#redacting-too-late)
  - [Per-question logic in a per-call hook](#per-question-logic-in-a-per-call-hook)
  - [Recursive predict](#recursive-predict)
  - [Plain callables in `hooks=`](#plain-callables-in-hooks)
  - [Assuming results exist in end hooks](#assuming-results-exist-in-end-hooks)
  - [Order-dependent hooks](#order-dependent-hooks)

## Patterns

### Audit log

Record every decision with enough to reconstruct it: the state, the questions, the answers, the
model, the routing decision, usage and latency.

```python
import json

def audit(ctx):
    for state, result in zip(ctx.states, ctx.results or []):
        json.dump({
            "run_id": ctx.run_id,
            "model": ctx.model,
            "state": state,
            "routing": result.get("routing"),
            "answers": result["answers"],
            "usage": result.get("usage"),
            "call_usage": ctx.usage,
            "call_elapsed_ms": round(ctx.elapsed_ms or 0.0, 3),
        }, sys.stdout)
        sys.stdout.write("\n")

laya.load("convaiinnovations/laya", on_predict_end=audit)
```

A hook fires once per call, and a `predict_batch` call carries every state in it, so the record is
written per decision: `ctx.states` and `ctx.results` are aligned by index. `ctx.usage` and
`ctx.elapsed_ms` are totals for the whole call; each result carries its own `usage`.

Make it lenient if losing a log line must not fail a request: `hooks_raise=False`. Make it
strict if the audit trail is a compliance requirement.

### PII redaction

Redaction must happen in `on_predict_start`, before tokenization, or the model has already seen
the data.

```python
import re
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

def redact(ctx):
    ctx.states = [
        EMAIL.sub("[email]", s) if isinstance(s, str) else s
        for s in ctx.states
    ]

laya.load("convaiinnovations/laya", on_predict_start=redact)
```

A redaction hook is a policy hook: keep `hooks_raise=True`, because a silently broken redactor
is a data leak.

### Caching

A start hook checks the cache and calls `ctx.skip(...)`; an end hook fills it. The forward pass
is skipped on a hit.

```python
import hashlib, json

CACHE = {}

def key(ctx, index):
    # Not sort_keys=True: criteria order is positional, so two orders are two questions,
    # and the checkpoint and token budget change the answer too.
    payload = json.dumps([ctx.states[index], ctx.questions, ctx.model,
                          ctx.max_len, ctx.head_max_len], default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

def read(ctx):
    hits = [CACHE.get(key(ctx, i)) for i in range(len(ctx.states))]
    if all(hit is not None for hit in hits):
        ctx.skip(hits)   # one per state: skip replaces the whole call

def write(ctx):
    for i, result in enumerate(ctx.results or []):
        CACHE[key(ctx, i)] = result

laya.load("convaiinnovations/laya", on_predict_start=read, on_predict_end=write)
```

Hooks fire once per call, so keying on one state is not enough on `predict_batch`: `ctx.skip()`
replaces every result the call would have returned. Guard the cache with a lock when serving
concurrently. On the Router the cached payload still gets a `routing` key, so the return shape is
unchanged.

The same pair works on a single [LangChain](../langchain.md) node via its `hooks=` argument, which
is the way to cache one hot step in a graph without changing what every other caller of that agent
sees.

### Metrics

Counters and histograms from `ctx.model`, `ctx.usage` and `ctx.elapsed_ms`. Keep it lenient.

```python
COUNTS, LATENCIES = {}, []

def metrics(ctx):
    COUNTS[ctx.model] = COUNTS.get(ctx.model, 0) + 1
    if ctx.elapsed_ms is not None:
        LATENCIES.append(ctx.elapsed_ms)

laya.load("convaiinnovations/laya", on_predict_end=metrics, hooks_raise=False)
```

### Guardrails

A policy hook raises to block a request. `hooks_raise=True` (the default) lets the block reach
the caller; `on_error` and `on_predict_end` still run, so the audit trail records it.

```python
class Blocked(Exception):
    pass

def guard(ctx):
    if any("ssn" in str(state).lower() for state in ctx.states):
        raise Blocked("possible PII in state")

laya.load("convaiinnovations/laya", on_predict_start=guard)
```

Test it against the state shape. A guard that only reads `ctx.states[0]` blocks a single call and
lets a `predict_batch` call put every remaining state through the forward pass.

### Confidence gating

An end hook rewrites a low-confidence answer to a safe fallback, or annotates it for downstream
logic. This is a result mutation, not a rejection.

```python
def gate(ctx):
    for result in ctx.results or []:
        answer = result["answers"].get("dept")
        if answer and answer["confidence"] < 0.6:
            answer["choice"] = "human-review"
            answer["gated"] = True

laya.load("convaiinnovations/laya", on_predict_end=gate)
```

Mutate through `ctx.results`, which holds one dict per state of the call: gating only the first
one ships every other low-confidence answer unannotated.

### Routing override

`on_route` may replace `ctx.decision` to pin a checkpoint for a class of traffic.

```python
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

Router(hooks=[pin])
```

### Model lifecycle

`on_load` and `on_evict` observe checkpoints. Use them for warmup logs, memory accounting, or
eviction alerts. They run outside the Router lock, so a hook may call back into the Router.

```python
class Lifecycle:
    def on_load(self, ctx):
        print("loaded", ctx.model)

    def on_evict(self, ctx):
        print("evicted", ctx.model)

Router(hooks=[Lifecycle()])
```

### Multi-tenant context

Thread a tenant id through by capturing it in the hook closure, or by reading it from a
context-local. Do not store per-request state on the hook object without a lock.

```python
def make_audit(tenant):
    def audit(ctx):
        ship(tenant, ctx.run_id, ctx.results)
    return audit

agent = laya.load("convaiinnovations/laya", on_predict_end=make_audit("acme"))
```

### Composition

Several hooks of different kinds compose naturally; installed hooks run first, in order.

```python
agent = laya.load(
    "convaiinnovations/laya",
    hooks=[Metrics(), Guardrail()],     # metrics first, then policy
    on_predict_start=redact,            # convenience callables appended after hooks
    hooks_raise=True,                   # policy failures are fatal
)
```

Keep the ordering deliberate and documented, because a later hook sees the mutations of an
earlier one.

### Scoped instrumentation

Attach a tracer or debug hook only for the code that needs it, instead of reconstructing the
agent. `hooks_installed` restores the previous list on exit, even if the block raises.

```python
with agent.hooks_installed(DebugDump()):
    agent.system_one(state, questions)   # DebugDump only here
```

`add_hook`/`remove_hook` do the same without a block, for a tracer that lives as long as the
process.

### Process-wide instrumentation

A tracer or metrics hook that every decision should see can be registered once, instead of being
passed to each `Agent` and `Router`. Defaults run before the instance and per-call hooks.

```python
from laya import BaseHook, hooks

class Metrics(BaseHook):
    def on_predict_end(self, ctx):
        record(ctx.model, ctx.elapsed_ms)

hooks.set_default_hooks(hooks=[Metrics()])
```

This is global state, so scope it deliberately: set it once at startup, and `clear_default_hooks()`
in tests so one test cannot leak a hook into the next.

### Token-budget shaping

A start hook can raise the token budget for one call, for example when a question has many options
and the default head budget would collapse the labels. Four details decide whether the hook helps
or quietly makes the call worse:

* A start hook's `ctx.head_max_len` *replaces* the budget for the call. What is in force before it
  is the caller's own per-call value, or the checkpoint default in `ctx.agent.cfg` -- so compare
  against that, and writing a plain number can lower the budget a caller already set.
* One call answers every question it carries, so size on the widest of them rather than on
  whichever happens to come first.
* Once the options no longer fit the head, `laya/common.py` gives each of them
  `max(4, (head_max_len - 16) // k)` tokens. `16 + 4 * k` therefore lands exactly on that floor:
  every label is still cut down to the tokens it shares with the others, which is the collapse the
  hook was written to avoid. `16 + 8 * k` leaves them distinguishable.
* The state gets `max_len - head_max_len - 8` tokens, so a widened head has to widen `max_len`
  with it or the state loses its window.

```python
def widen_for_high_cardinality(ctx):
    k = max((len(q.get("criteria", {}) or {}) for q in ctx.questions.values()), default=0)
    if k < 50:
        return
    cfg = getattr(ctx.agent, "cfg", None) or {}
    head = ctx.head_max_len if ctx.head_max_len is not None else cfg.get("head_max_len", 192)
    window = ctx.max_len if ctx.max_len is not None else cfg.get("max_len", 512)
    need = 16 + 8 * k                          # 8 tokens per label, not the core's floor of 4
    if need > head:                            # only ever widen, never lower
        ctx.head_max_len = need
        ctx.max_len = max(window, need + 8 + 64)   # 8 reserved, then room for the state

agent = laya.load("convaiinnovations/laya", on_predict_start=widen_for_high_cardinality)
```

This does not touch the shared agent config, so concurrent calls are unaffected. The same knobs are
available per call: `agent.system_one(state, questions, head_max_len=512, max_len=1024)`.

Widening is not free: a longer window means a larger tensor, and the checkpoints were trained at
512 (`laya`) and 1,024 tokens. Past that, narrowing the candidates with
[`predict_shortlist`](../reference/helpers.md) beats stretching the budget.

## Anti-patterns

### Blocking work

Hooks run on the calling thread, and `laya.serve` uses a single inference worker. A hook that
sleeps, waits on a network round-trip, or calls `input()` stalls every other request behind it.

```python
# bad: blocks the whole server
def audit(ctx):
    requests.post("https://slow.example/decisions", json=..., timeout=30)

# better: enqueue, let a background worker ship it
def audit(ctx):
    QUEUE.put_nowait(record(ctx))
```

If you must do slow work, set `hooks_concurrent=False` to at least keep the hook itself from
overlapping, and run `laya.serve` behind a queue.

### Raising from end hooks for control flow

`on_predict_end` runs after inference. Raising there throws away a computed result and, on the
success path, surfaces to the caller. Use a start hook to block before paying for inference, or
rewrite `ctx.results` to change the answer.

### Shared mutable state without a lock

The same hook instance runs on many threads. `self.counter += 1` races.

```python
# bad
class Count:
    def __init__(self): self.n = 0
    def on_predict_end(self, ctx): self.n += 1

# good
import threading
class Count:
    def __init__(self):
        self.n = 0
        self._lock = threading.Lock()
    def on_predict_end(self, ctx):
        with self._lock:
            self.n += 1
```

### Silent failure

`hooks_raise=False` warns once per failure, but a hook that catches everything itself hides
real problems.

```python
# bad: no one will ever know the audit trail stopped
def audit(ctx):
    try:
        ship(record(ctx))
    except Exception:
        pass
```

If a hook is optional, let `hooks_raise=False` handle it and watch the warnings. If it is not,
let it raise.

### Retaining contexts

A hook that appends `ctx` to a list keeps the whole state, questions, results and agent alive.

```python
# bad: unbounded memory growth
SEEN = []
def audit(ctx):
    SEEN.append(ctx)

# good: keep only what you need
SEEN = []
def audit(ctx):
    SEEN.append((ctx.run_id, ctx.model, ctx.elapsed_ms))
```

### Redacting too late

By `on_predict_end` the model has already tokenized the state. Redact in `on_predict_start`.

### Per-question logic in a per-call hook

There is one `PredictContext` per call, and one forward pass answers every question. There are no
per-question events. Iterate the answers inside `on_predict_end`, and iterate the states too: on a
batch, one context carries every state of the call.

```python
def flag(ctx):
    for result in ctx.results or []:
        for qid, answer in result["answers"].items():
            if answer.get("confidence", 1.0) < 0.5:
                alert(qid, ctx.run_id)
```

### Recursive predict

A hook that calls `agent.predict`/`system_one` runs the hooks again. Without a depth guard this
recurses.

```python
# bad
def enrich(ctx):
    ctx.results = [agent.predict(ctx.states[0], EXTRA_QUESTIONS)]

# good: guard, or use a separate agent with no hooks
def enrich(ctx):
    if getattr(ctx, "_enriched", False):
        return
    ctx._enriched = True
    ctx.results = [enricher.predict(state, EXTRA_QUESTIONS) for state in ctx.states]
```

### Plain callables in `hooks=`

`hooks=` takes hook objects; a bare callable does not say which event it is for, so it is
rejected. Use `on_predict_start=` / `on_predict_end=`.

```python
# bad: TypeError
laya.load("convaiinnovations/laya", hooks=[lambda ctx: None])

# good
laya.load("convaiinnovations/laya", on_predict_end=lambda ctx: None)
```

### Assuming results exist in end hooks

On the failure path `ctx.results` is `None` unless a start hook set it. Always check.

```python
def audit(ctx):
    if ctx.results is None:
        log_failure(ctx.run_id, ctx.error)
        return
    log_success(ctx.run_id, ctx.results)
```

### Order-dependent hooks

A hook that reads a mutation from another hook is fragile unless the order is pinned. Installed
hooks run in list order, then convenience callables; document any coupling, or merge the coupled
hooks into one object.

## See also

- [Errors](errors.md): the failure matrix behind several of these anti-patterns.
- [Examples](examples.md): fuller versions of the patterns above.
