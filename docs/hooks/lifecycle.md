# Lifecycle

This page is the precise order of events for each entry point. If you only read one diagram,
read the [Router](#routerpredict) one; it is the superset.

- [Agent.predict_batch](#agentpredict_batch)
- [Agent.system_one / predict](#agentsystem_one-predict)
- [Router.predict](#routerpredict)
- [Router.predict_batch](#routerpredict_batch)
- [Model lifecycle](#model-lifecycle)
- [Caching with skip](#caching-with-skip)
- [Empty inputs](#empty-inputs)
- [Ordering rules](#ordering-rules)
- [Concurrency](#concurrency)

## Agent.predict_batch

`predict_batch` is the single implementation; `system_one` and `predict` call it with one state.

```
predict_batch(states, questions, batch_size=..., hooks=..., ...)
  │
  ├─ active  = installed hooks + per-call hooks         (installed first)
  ├─ ctx     = PredictContext(states, questions, model=self.model_id, agent=self)
  │
  ├─ try:
  │    │
  │    ├─ on_predict_start ─────────────────────────────┐
  │    │      a hook may:                                │
  │    │        • rewrite ctx.states / ctx.questions     │
  │    │        • set ctx.max_len / ctx.head_max_len     │
  │    │        • ctx.skip(results) ─────────────┐       │
  │    │        • raise (aborts; see errors)     │       │
  │    │                                         │      │
  │    ├─ if ctx.results is not None:  ◄─────────┘      │  cache hit
  │    │      skip tokenization and forward             │
  │    ├─ else:                                         │
  │    │      validate states is a list                 │
  │    │      for each batch chunk:                     │
  │    │        _encode_state ─► collate ─► _forward    │
  │    │      _decode_answers                           │
  │    │      ctx.results = [...]                       │
  │    │                                                │
  │    └─ (any failure here) ──► except BaseException:  │
  │              ctx.error = exc                        │
  │              on_error                               │
  │              re-raise                               │
  │                                                     │
  │   finally:                                          │
  │     ctx.elapsed_ms = now - ctx.started_at           │
  │     if ctx.results: ctx.usage = aggregate_usage(...)│
  │     on_predict_end ─────────────────────────────────┘
  │
  └─ return ctx.results
```

The same `ctx` object flows through start, error and end, so `run_id` correlates them and an end
hook can read `ctx.error`.

## Agent.system_one / predict

```
system_one(state, questions, hooks=..., ...)
  └─ predict_batch([state], questions, hooks=..., ...)[0]
```

So `system_one` inherits every hook and the same lifecycle, with `ctx.states == [state]`.

## Router.predict

```
Router.predict(state, questions, model=..., hooks=..., on_predict_start=..., on_predict_end=...)
  │
  ├─ active = installed hooks + per-call hooks
  │
  ├─ route(state, questions, ..., hooks=per-call, hooks_raise=...)
  │    │
  │    ├─ _route(...)                      detect script / language / workflow
  │    ├─ on_route  ──► ctx.decision       a hook may replace the decision
  │    └─ return ctx.decision
  │
  ├─ load(decision["model"])
  │    │
  │    ├─ already resident? ──► return it
  │    ├─ else build Agent(...) ──► on_load   (after the Router lock is released)
  │    └─ evict LRU checkpoints ──► on_evict  (after the Router lock is released)
  │
  ├─ ctx = PredictContext(states=[state], questions, decision, model=decision.model,
  │                       agent=agent, router=self)
  ├─ try:
  │    ├─ on_predict_start
  │    ├─ if ctx.results is None:
  │    │      result = agent.system_one(ctx.states[0], ctx.questions)
  │    │        └─ the Agent's own hooks run here (start / forward / end)
  │    │      result["routing"] = decision
  │    │      ctx.results = [result]
  │    └─ else:
  │           for each cached result: result.setdefault("routing", decision)
  │    └─ (any failure) ──► except: on_error, re-raise
  │    └─ finally: elapsed_ms, usage, on_predict_end
  │
  └─ return ctx.results[0]
```

Key points:

- `on_route` runs before the model is loaded, so a hook can pin a checkpoint and avoid loading
  another one.
- Router-level predict hooks wrap the whole call. They are **not** forwarded into the Agent;
  an attached Agent with its own hooks runs those too, which is expected.
- A Router-level `ctx.skip()` still adds `routing`, so the return shape is stable.

## Router.predict_batch

Each result is what `predict` returns for that request, so Router-level predict hooks run per
request here too: every request gets its own `PredictContext`, `run_id` and `elapsed_ms`.

```
Router.predict_batch(requests, batch_size=...)
  │
  ├─ route_batch(requests) ──► on_route, once per request   (no checkpoint loaded yet)
  │
  └─ for each checkpoint, in order of first appearance:
       │
       ├─ load(checkpoint) ──► on_load / on_evict
       ├─ for each request of this checkpoint, in input order:
       │      ctx = PredictContext(states=[state], questions, decision, model, agent, router)
       │      on_predict_start       a hook may redact, rewrite, set a token budget or skip
       ├─ group the requests left to infer by (questions, ctx.max_len, ctx.head_max_len)
       │      agent.predict_batch(states, questions, ...)  ──► one shared forward pass per group
       │      result["routing"] = decision;  ctx.results = [result]
       ├─ (any failure) ──► on_error for every started request without a result,
       │                    on_predict_end for every started request, re-raise
       └─ on_predict_end, once per request of this checkpoint, in input order
```

Key points:

- Every start hook of a checkpoint's requests runs before any of their end hooks, because they
  share forward passes. A cache that fills in `on_predict_end` therefore cannot serve a duplicate
  state within the same checkpoint group; it can across calls.
- A start hook that replaces `ctx.states`, `ctx.questions` or the token budget changes its own
  request only: requests are grouped for the forward pass after their start hooks have run.
  Mutating a questions dict in place changes it for every request that shares that dict, and for
  the caller, as it would with `predict`.
- Every started request gets exactly one `on_predict_end`, even when an earlier request's end hook
  raises; the first such error is raised after all of them have run.
- If the batch fails, a request that did not get a result is reported as failed (`on_error`, with
  `ctx.error` set to the exception that failed the batch), because the caller gets no result for
  it. Requests of checkpoint groups that already finished have ended with their results, as the
  earlier calls of `[router.predict(...) for ...]` would have.

## Model lifecycle

`on_load` fires when a checkpoint is built; `on_evict` when one is freed. Both run **after** the
Router's internal lock is released, so a hook may safely call back into the Router.

```
load("multilingual")
  │
  ├─ [lock]
  │    build Agent(...)          (seconds: download + weights)
  │    register in _agents / _order
  │    evict LRU if over max_loaded ──► evicted = ["english"]
  ├─ [unlock]
  ├─ on_evict("english")
  └─ on_load("multilingual")

unload("english")
  ├─ [lock] remove from _agents / _order
  ├─ [unlock]
  └─ on_evict("english")
```

`attach(name, agent)` registers an existing agent and does **not** fire `on_load`, because no
checkpoint was built.

## Caching with skip

```
on_predict_start
  ├─ cache hit?  ctx.skip([cached_result])
  │     └─ forward pass skipped
  │     └─ on_predict_end still runs
  │     └─ Router adds `routing` if missing
  └─ cache miss? nothing
        └─ forward pass runs
        └─ on_predict_end can store the result
```

See [`examples/hooks/cache.py`](../../examples/hooks/cache.py) for a working cache.

## Empty inputs

Hooks still fire so audit sees every call:

| input | `ctx.results` at end |
|---|---|
| `predict_batch([])` | `[]` |
| `predict_batch(states, {})` (no questions) | one empty-answer payload per state |
| `system_one(state, {})` | a single empty-answer payload |

No tokenization or forward pass happens in these cases, but `on_predict_start` and
`on_predict_end` run.

## Ordering rules

1. Installed hooks run before per-call hooks, always.
2. Within a list, hooks run in list order.
3. For one event, every hook that implements it runs, in that order, before the next event.
4. `on_error` runs before `on_predict_end` on the failure path.
5. `on_evict` runs before `on_load` when a single `load` both evicts and builds.

```
installed: [A, B]   per-call: [C]
on_predict_start: A, B, C
on_predict_end:   A, B, C
```

## Concurrency

`Agent` and `Router` are safe to call from many threads. Each call creates its own
`PredictContext`, so contexts never leak across requests. The only shared state is the hook
objects themselves, so a hook that is not thread-safe must either guard its own state or be
installed with `hooks_concurrent=False`.

```
hooks_concurrent=True (default)      hooks_concurrent=False
  thread 1 ─┐                          thread 1 ─┐
  thread 2 ─┼─ hooks run in parallel   thread 2 ─┼─ one hook at a time
  thread 3 ─┘                          thread 3 ─┘   (RLock)
```

`hooks_concurrent=False` serialises each hook invocation, not whole calls: two calls can still
interleave between events. It uses a re-entrant lock, so a hook may call back into the same
`Agent`/`Router` without deadlocking.

## See also

- [Errors](errors.md): what happens when a hook raises, per event.
- [Patterns and anti-patterns](patterns.md): how to use the lifecycle well.
