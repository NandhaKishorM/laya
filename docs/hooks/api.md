# API reference

Everything on this page is importable from `laya` (the common names) or `laya.hooks` (the
whole surface).

```python
from laya import PredictContext, PredictHook, Hook
from laya.hooks import HOOK_EVENTS, normalise_hooks, dispatch, aggregate_usage
```

- [PredictContext](#predictcontext)
- [Hook protocol](#hook-protocol)
- [Convenience types](#convenience-types)
- [Configuration surface](#configuration-surface)
- [Event payloads](#event-payloads)
- [Validation](#validation)
- [Advanced helpers](#advanced-helpers)

## PredictContext

A `PredictContext` is created once per public call and passed to every hook of that call. It is
**mutable**: hooks may rewrite `states`, `questions` and `results`, and `on_route` may rewrite
`decision`. It uses identity equality (`eq=False`), so a context is hashable and two contexts are
never equal.

```python
@dataclass(eq=False)
class PredictContext:
    states: List[Any]
    questions: Dict[str, Any]
    run_id: str = <uuid4 hex>
    results: Optional[List[Dict[str, Any]]] = None
    decision: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    agent: Any = None
    router: Any = None
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    usage: Optional[Dict[str, int]] = None
    started_at: float = <perf_counter()>
    elapsed_ms: Optional[float] = None
    error: Optional[BaseException] = None
```

| field | type | set when | mutable | meaning |
|---|---|---|---|---|
| `states` | `list` | always | yes (start) | the states for this call. `system_one`/`Router.predict` pass one; `predict_batch` passes many. A start hook may replace the list. |
| `questions` | `dict` | always | yes (start) | the questions. A start hook may replace the dict. |
| `run_id` | `str` | always | no | a unique id shared by every hook of this call. Use it to correlate events and spans. |
| `results` | `list \| None` | end (and on a skip) | yes (end) | per-state result dicts, each shaped like `system_one`'s return. `None` until inference finishes. |
| `decision` | `dict \| None` | Router only | yes (route) | the `RouteDecision` (a `dict`) that selected the checkpoint. |
| `model` | `str \| None` | always | no | the checkpoint id: `Agent.model_id` for an Agent, the resolved alias (for example `"english"`) for a Router. |
| `agent` | `Agent \| ONNXAgent \| None` | predict events | no | the runtime answering the call. |
| `router` | `Router \| None` | Router events | no | the Router, when one is in play. |
| `max_len` | `int \| None` | always | yes (start) | per-call token budget for the encoder. `None` uses the agent config. |
| `head_max_len` | `int \| None` | always | yes (start) | per-call token budget for the question head. `None` uses the agent config. |
| `usage` | `dict \| None` | end | yes (end) | `{"input_tokens", "output_tokens"}`, summed over the states of the call. |
| `started_at` | `float` | always | no | `time.perf_counter()` when the call began. |
| `elapsed_ms` | `float \| None` | end | no | wall time for the whole call, milliseconds. |
| `error` | `BaseException \| None` | failure path | no | the exception, set before `on_error` and `on_predict_end`. |

### `PredictContext.skip(results)`

Short-circuits inference. Called from `on_predict_start`, it sets `ctx.results` so the forward
pass is skipped; `on_predict_end` still runs and the supplied results are returned.

```python
def cache_read(ctx):
    hit = CACHE.get(key(ctx.states[0], ctx.questions))
    if hit is not None:
        ctx.skip([hit])   # list of per-state results, same shape as predict_batch's return
```

On the `Router`, a skipped payload gets a `routing` key added (without overwriting one it
already has), so `Router.predict` keeps its documented return shape.

## Hook protocol

`Hook` is a `typing.Protocol`. Implement any subset of the methods; the rest are skipped.

```python
class Hook(Protocol):
    def on_predict_start(self, ctx: PredictContext) -> None: ...
    def on_predict_end(self, ctx: PredictContext) -> None: ...
    def on_route(self, ctx: PredictContext) -> None: ...
    def on_load(self, ctx: PredictContext) -> None: ...
    def on_evict(self, ctx: PredictContext) -> None: ...
    def on_error(self, ctx: PredictContext) -> None: ...
```

| event | where | runs | can change |
|---|---|---|---|
| `on_predict_start` | Agent, Router | before tokenization/forward | `states`, `questions`, or `skip()` |
| `on_predict_end` | Agent, Router | after results exist, success or failure | `results` |
| `on_route` | Router | after detection, before loading | `decision` |
| `on_load` | Router | after a checkpoint is built | nothing (observe) |
| `on_evict` | Router | after a checkpoint is freed | nothing (observe) |
| `on_error` | Agent, Router | when a predict call fails | nothing (observe) |

A hook is free to define extra attributes and methods; only the six event names are consulted.
If a hook defines one of the six as a non-callable, configuration fails fast (see
[Validation](#validation)).

## Convenience types

```python
PredictHook = Callable[[PredictContext], None]
```

`PredictHook` is the type of a plain callable used with `on_predict_start=` / `on_predict_end=`.
Pass a single callable or a sequence of them; each is wrapped into a minimal hook.

## Runtime registration

Every runtime mixes in `HookRegistry`, so hooks can be added, removed or scoped after
construction. Mutation is thread-safe; a call reads a snapshot of the list, so adding or
removing a hook never disturbs a call in flight.

```python
agent.add_hook(tracer)              # one hook or a sequence; returns self for chaining
agent.remove_hook(tracer)           # by identity; True if it was installed

with agent.hooks_installed(debug):  # installed for the block, removed on exit
    agent.system_one(state, questions)
```

`add_hook` accepts the same objects as `hooks=` (not plain callables). `hooks_installed` takes
any number of hook objects or sequences and restores the previous list on exit, including when
the block raises.

## Configuration surface

Every entry point accepts the same five hook parameters. `hooks` takes an object or a sequence
of objects; `on_predict_start` / `on_predict_end` take a callable or a sequence.

| parameter | type | default | meaning |
|---|---|---|---|
| `hooks` | `Hook \| Sequence[Hook] \| None` | `None` | lifecycle hooks (any of the six events). |
| `on_predict_start` | `PredictHook \| Sequence[PredictHook] \| None` | `None` | convenience callables for one event. |
| `on_predict_end` | `PredictHook \| Sequence[PredictHook] \| None` | `None` | convenience callables for one event. |
| `hooks_raise` | `bool` | `True` | `True`: a hook exception propagates. `False`: warn and continue. |
| `hooks_concurrent` | `bool` | `True` | `False`: dispatch hooks under a lock, one at a time. |

### Agent

```python
Agent(
    model_id_or_path="convaiinnovations/laya",
    device=None, token=None, subfolder=None, fast=False, compile=False,
    hooks=None, on_predict_start=None, on_predict_end=None,
    hooks_raise=True, hooks_concurrent=True,
)

load(..., hooks=None, on_predict_start=None, on_predict_end=None,
     hooks_raise=True, hooks_concurrent=True)

agent.predict_batch(states, questions, batch_size=None,
                    hooks=None, on_predict_start=None, on_predict_end=None, hooks_raise=None,
                    max_len=None, head_max_len=None)

agent.system_one(state, questions,
                 hooks=None, on_predict_start=None, on_predict_end=None, hooks_raise=None,
                 max_len=None, head_max_len=None)

agent.predict(...)          # alias of system_one
```

- `hooks_raise` on a per-call method defaults to `None`, meaning "use the instance value".
- `hooks_concurrent` is instance-level only.

### Router

```python
Router(
    models=None, device=None, token=None, max_loaded=2, default="english",
    auto_task_detection=False, standalone_repos=False, preload=False, lang_guess=None,
    hooks=None, on_predict_start=None, on_predict_end=None,
    hooks_raise=True, hooks_concurrent=True,
)

router.route(state, questions=None, model=None, task=None, lang=None, lang_guess=None,
             hooks=None, hooks_raise=None)

router.predict(state, questions, model=None, task=None, lang=None, lang_guess=None,
               hooks=None, on_predict_start=None, on_predict_end=None, hooks_raise=None,
               max_len=None, head_max_len=None)

router.system_one(...)      # alias of predict
router.load(name)           # builds on first use; fires on_load
router.preload(names=None)  # builds several; fires on_load per build
router.unload(name=None)    # frees one or all; fires on_evict
router.attach(name, agent)  # registers an existing agent; does not fire on_load
router.loaded               # list of resident checkpoint names
```

- Per-call `hooks=` on `route` and `predict` apply to the whole call, including `on_route`.
- `route()` is public: calling it dispatches `on_route` with the installed hooks plus any
  per-call `hooks`.

### ONNXAgent

```python
ONNXAgent(model_id_or_path, onnx_path="laya.onnx", subfolder=None,
          hooks=None, on_predict_start=None, on_predict_end=None,
          hooks_raise=True, hooks_concurrent=True)

onnx_agent.system_one(state, questions,
                      hooks=None, on_predict_start=None, on_predict_end=None, hooks_raise=None,
                      max_len=None, head_max_len=None)

onnx_agent.predict(...)     # alias of system_one
```

`ONNXAgent` has no Router, so it exposes only the predict-level events.

## Event payloads

Which fields are populated, per event and runtime:

| event | runtime | `states` | `questions` | `decision` | `model` | `agent` | `router` | `results` | `usage` | `elapsed_ms` | `error` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `on_predict_start` | Agent | ✓ | ✓ | – | ✓ | ✓ | – | – | – | – | – |
| `on_predict_start` | Router | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – |
| `on_predict_end` | Agent | ✓ | ✓ | – | ✓ | ✓ | – | ✓ | on success | ✓ | on failure |
| `on_predict_end` | Router | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | on success | ✓ | on failure |
| `on_error` | both | ✓ | ✓ | ✓ (Router) | ✓ | ✓ | ✓ (Router) | – | – | – | ✓ |
| `on_route` | Router | ✓ | ✓ | ✓ | – | – | ✓ | – | – | – | – |
| `on_load` | Router | `[]` | `{}` | – | ✓ | ✓ | ✓ | – | – | – | – |
| `on_evict` | Router | `[]` | `{}` | – | ✓ | – | ✓ | – | – | – | – |

Timing details:

- `on_predict_end` sees `results` on the success path. On the failure path `results` is `None`
  unless a start hook had set them via `skip()`, and `usage` is therefore `None` too (it is
  derived from `results`); `elapsed_ms` is always set.
- `on_error` runs before the `finally` block that computes `elapsed_ms` and `usage`, so both are
  `None` there. Read timing and usage from `on_predict_end` instead.
- `run_id` is always populated.

## Validation

Configuration is validated when hooks are normalised, which happens at construction for
installed hooks and at call time for per-call hooks. These raise `TypeError`:

| case | message |
|---|---|
| a class is passed instead of an instance | `hooks entries must be instances, not classes; ...` |
| an object implements none of the six events | `hooks entries must implement at least one of ...` |
| an event attribute is not callable | `hooks entry X.on_predict_start must be callable, got int` |
| `on_predict_start=` / `on_predict_end=` is not callable | `on_predict_start must be callable, got int` |

`hooks=` does not accept plain callables, because a bare callable does not say *which* event it
is for. Use `on_predict_start=` / `on_predict_end=` for those.

## Advanced helpers

These are used internally and are stable, but most users do not need them.

```python
HOOK_EVENTS          # tuple of the six event names, in dispatch order
normalise_hooks(hooks=None, on_predict_start=None, on_predict_end=None) -> list
dispatch(hooks, event, ctx, *, raise_errors=True, lock=None) -> None
aggregate_usage(results) -> {"input_tokens": int, "output_tokens": int}
```

`normalise_hooks` flattens a `hooks` object/sequence and the two callables into one ordered list.
`dispatch` calls `event` on every hook that implements it, applying the raise policy and lock.
`aggregate_usage` sums per-state usage blocks.

```python
from laya.hooks import normalise_hooks, dispatch, PredictContext

hooks = normalise_hooks(on_predict_start=[log, redact])
ctx = PredictContext(states=["..."], questions={...})
dispatch(hooks, "on_predict_start", ctx)
```

## See also

- [Lifecycle](lifecycle.md): when each event runs, with flowcharts.
- [Errors](errors.md): the failure matrix and chaining rules.
- [Patterns and anti-patterns](patterns.md): how to structure hooks well.
- [Examples](examples.md): recipes for every use case.
