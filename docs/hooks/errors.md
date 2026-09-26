# Error handling

Hooks run inside the request they observe, so how their failures are handled matters. This page
is the exact policy.

- [The two policies](#the-two-policies)
- [What runs when something fails](#what-runs-when-something-fails)
- [Exception chaining](#exception-chaining)
- [BaseException](#baseexception)
- [Configuration errors](#configuration-errors)
- [Warnings](#warnings)
- [Choosing a policy](#choosing-a-policy)

## The two policies

`hooks_raise` controls what happens when a hook raises.

| `hooks_raise` | behaviour |
|---|---|
| `True` (default) | the hook exception propagates out of the call. |
| `False` | the hook is skipped with a `RuntimeWarning` and the call continues. |

It is set per instance and can be overridden per call (`hooks_raise=` on `predict_batch`,
`system_one`, `Router.route`, `Router.predict`). Per-call `None` means "use the instance value".

```python
# strict: a broken audit hook fails the request
laya.load("convaiinnovations/laya", on_predict_end=audit, hooks_raise=True)

# lenient: telemetry must never take down a served request
laya.load("convaiinnovations/laya", on_predict_end=metrics, hooks_raise=False)
```

`dispatch` catches `Exception`. Anything that is not an `Exception` (see
[BaseException](#baseexception)) is never swallowed, even with `hooks_raise=False`.

## What runs when something fails

The predict lifecycle is wrapped in `try / except / finally`, so cleanup hooks run on failure.

```
try:
    on_predict_start
    inference
except BaseException as exc:
    ctx.error = exc
    on_error            (best effort; cannot mask exc)
    raise
finally:
    elapsed_ms, usage
    on_predict_end      (best effort; cannot mask exc on the failure path)
```

Failure matrix, per event and runtime:

| event | runtime | if it raises |
|---|---|---|
| `on_predict_start` | Agent / Router | `hooks_raise=True`: `on_error` and `on_predict_end` still run, then the exception propagates. `False`: warn and continue (mutations made before the raise remain). |
| inference | Agent / Router | `ctx.error` set, `on_error` runs, `on_predict_end` runs, exception propagates. |
| `on_error` | Agent / Router | never masks the original exception; chained as `__context__`. |
| `on_predict_end` (success path) | Agent / Router | `hooks_raise=True`: propagates (the result is computed but the call fails). `False`: warn. |
| `on_predict_end` (failure path) | Agent / Router | never masks the original exception; chained as `__context__`. |
| `on_route` | Router | propagates directly; there is no predict context yet. |
| `on_load` | Router | propagates directly; the checkpoint stays built and resident. |
| `on_evict` | Router | propagates directly; the checkpoint is already freed. |

Consequences worth knowing:

- A failing `on_load` leaves the model cached, so the next `load` returns it without firing
  `on_load` again.
- A failing `on_predict_end` on the success path means the caller gets an exception instead of a
  result, even though inference succeeded. Use `hooks_raise=False` for end hooks that are pure
  side effects.

## Exception chaining

When a hook fails while another exception is already propagating, the original exception is
re-raised and the hook's exception is attached as `__context__`. The root cause is never lost.

```python
class BadTelemetry:
    def on_error(self, ctx):
        raise RuntimeError("telemetry down")

try:
    agent.system_one(state, questions, hooks=[BadTelemetry()])
except RuntimeError as exc:
    assert exc.__context__ is not None   # the telemetry failure
```

The same rule applies to a failing `on_predict_end` on the failure path.

## BaseException

`dispatch` catches `Exception`, not `BaseException`, so `KeyboardInterrupt` and `SystemExit`
always propagate. They still trigger the `except BaseException` branch of the predict lifecycle,
which means `on_error` and `on_predict_end` run before the process unwinds. Keep those hooks
fast and non-blocking if you care about interrupt latency.

## Configuration errors

Bad configuration fails fast with `TypeError`, before any inference:

| case | raised at | example |
|---|---|---|
| class instead of instance | construction | `hooks=[MyHook]` |
| no lifecycle method | construction | `hooks=[object()]` |
| non-callable event | construction | `on_predict_start = 5` |
| non-callable convenience hook | construction | `on_predict_start=123` |
| plain callable in `hooks=` | construction | `hooks=[lambda ctx: None]` |

Per-call hooks are validated when the call is made, so a bad per-call hook raises `TypeError`
from `predict`/`system_one` rather than at construction.

## Warnings

With `hooks_raise=False`, each failing hook emits one `RuntimeWarning` naming the hook type and
event:

```
laya: hook Metrics.on_predict_end failed: connection reset
```

The warning is emitted once per failure, not once per hook definition, so a flaky hook under
load can be noisy. Aggregate or rate-limit inside the hook if that matters.

## Timeouts

`hooks_timeout` bounds each hook call in seconds. A hook still running after the limit is treated
as a hook failure: `TimeoutError` when `hooks_raise=True`, a `RuntimeWarning` when `False`. `None`
(the default) means no limit.

```python
laya.load("convaiinnovations/laya", on_predict_end=metrics, hooks_timeout=2.0)
```

It can be set per instance or overridden per call on `predict_batch`, `system_one`,
`Router.route`, `Router.predict` and `ONNXAgent.system_one`. The value must be positive; `0` or a
negative number raises `ValueError` at the point it is set, rather than racing on a zero-length
`join`.

A timed hook runs on a worker thread in a copy of the caller's `contextvars` context, so a
request id or tracing span set by the caller is visible to the hook.

One honest caveat: Python cannot interrupt a thread, so a timed-out hook keeps running in the
background. The timeout bounds how long the request waits, not how long the hook lives. Use it to
keep a served request responsive, not to reclaim the work. For a hook that can hang, also give the
underlying call its own timeout (a socket or HTTP timeout). Because the thread cannot be
reclaimed, a hook that hangs on every call grows threads one per call; give a hook that can hang
its own bound rather than relying on `hooks_timeout` to stop it.

For an async hook, the coroutine runs on the event loop; a timeout on the calling side still
returns after the limit, and the coroutine keeps running on the loop.

The timeout also releases the `hooks_concurrent=False` lock: dispatch waits for the hook only up
to the limit, then moves on, while the timed-out hook keeps running outside the lock. So the lock
serialises the hooks that finish in time, not every hook that was ever started; a hook that
overruns no longer blocks the ones behind it.

## Choosing a policy

| hook kind | recommended | why |
|---|---|---|
| policy / guardrail / redaction | `hooks_raise=True` | a policy that silently fails is a security hole. |
| audit / logging | `hooks_raise=True` in tests, often `False` in production | losing an audit record should be loud, but not necessarily fatal. |
| metrics / tracing | `hooks_raise=False` | observability must not fail the request. |
| cache read/write | `hooks_raise=True` | a broken cache should surface, not silently miss. |

You can mix: install a guardrail with the instance default and give a telemetry hook its own
try/except, or use a separate `hooks_raise` per call.

## See also

- [Lifecycle](lifecycle.md): the `try/except/finally` shape in context.
- [Patterns and anti-patterns](patterns.md): common mistakes with error handling.
