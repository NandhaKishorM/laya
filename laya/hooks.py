"""Opt-in prediction hooks: observe or shape every decision without forking.

A hook is either a plain callable or an object implementing any subset of the lifecycle
methods on `Hook`. Hooks are configured on `Agent` / `Router` and can be overridden per call.
Everything here is pure Python: importing `laya` must not start pulling torch.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union


@dataclass(eq=False)
class PredictContext:
    """Mutable state passed to every hook for one call.

    `states` / `questions` may be rewritten by `on_predict_start`; `results` may be rewritten
    by `on_predict_end`. A start hook can call `skip()` to short-circuit inference with a
    cached result.

    Identity semantics (`eq=False`): two contexts are never equal, and a context is hashable
    by identity, so a hook can keep one in a set without comparing agent/router internals.
    """

    states: List[Any]
    questions: Dict[str, Any]
    results: Optional[List[Dict[str, Any]]] = None
    decision: Optional[Dict[str, Any]] = None       # Router: the RouteDecision dict
    model: Optional[str] = None                     # resolved checkpoint name
    agent: Any = None
    router: Any = None
    usage: Optional[Dict[str, int]] = None          # aggregated input/output tokens
    started_at: float = field(default_factory=time.perf_counter)
    elapsed_ms: Optional[float] = None
    error: Optional[BaseException] = None

    def skip(self, results: List[Dict[str, Any]]) -> None:
        """Set cached results from a start hook; inference is skipped, end hooks still run."""
        self.results = results


class Hook(Protocol):
    """Optional lifecycle methods. Implement any subset; missing methods are skipped."""

    def on_predict_start(self, ctx: PredictContext) -> None: ...
    def on_predict_end(self, ctx: PredictContext) -> None: ...
    def on_route(self, ctx: PredictContext) -> None: ...
    def on_load(self, ctx: PredictContext) -> None: ...
    def on_evict(self, ctx: PredictContext) -> None: ...
    def on_error(self, ctx: PredictContext) -> None: ...


PredictHook = Callable[[PredictContext], None]
HookArg = Union[Hook, Sequence[Hook], None]
PredictHookArg = Union[PredictHook, Sequence[PredictHook], None]

HOOK_EVENTS = (
    "on_predict_start",
    "on_predict_end",
    "on_route",
    "on_load",
    "on_evict",
    "on_error",
)


class _StartAdapter:
    """Wrap a plain `on_predict_start` callable as a `Hook`."""

    __slots__ = ("fn",)

    def __init__(self, fn: PredictHook):
        if not callable(fn):
            raise TypeError("on_predict_start must be callable, got %s" % type(fn).__name__)
        self.fn = fn

    def on_predict_start(self, ctx: PredictContext) -> None:
        self.fn(ctx)


class _EndAdapter:
    """Wrap a plain `on_predict_end` callable as a `Hook`."""

    __slots__ = ("fn",)

    def __init__(self, fn: PredictHook):
        if not callable(fn):
            raise TypeError("on_predict_end must be callable, got %s" % type(fn).__name__)
        self.fn = fn

    def on_predict_end(self, ctx: PredictContext) -> None:
        self.fn(ctx)


def _as_sequence(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def normalise_hooks(
    hooks: HookArg = None,
    on_predict_start: PredictHookArg = None,
    on_predict_end: PredictHookArg = None,
) -> List[Any]:
    """Flatten `hooks` and the two convenience callables into one ordered hook list.

    Installed hooks are normalised once at construction; per-call arguments are appended
    after them, so per-call hooks always run last.
    """
    result: List[Any] = []
    for hook in _as_sequence(hooks):
        if isinstance(hook, type):
            raise TypeError(
                "hooks entries must be instances, not classes; got %s. Instantiate it first."
                % hook.__name__
            )
        if not any(hasattr(hook, event) for event in HOOK_EVENTS):
            raise TypeError(
                "hooks entries must implement at least one of %s; got %s. Pass a plain "
                "callable as on_predict_start=/on_predict_end= instead."
                % (", ".join(HOOK_EVENTS), type(hook).__name__)
            )
        for event in HOOK_EVENTS:
            method = getattr(hook, event, None)
            if method is not None and not callable(method):
                raise TypeError(
                    "hooks entry %s.%s must be callable, got %s"
                    % (type(hook).__name__, event, type(method).__name__)
                )
        result.append(hook)
    for fn in _as_sequence(on_predict_start):
        result.append(_StartAdapter(fn))
    for fn in _as_sequence(on_predict_end):
        result.append(_EndAdapter(fn))
    return result


def aggregate_usage(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sum the per-state usage blocks so a hook sees one total for the call."""
    def total(key: str) -> int:
        return sum(int((r.get("usage") or {}).get(key, 0) or 0) for r in results)
    return {"input_tokens": total("input_tokens"), "output_tokens": total("output_tokens")}


def dispatch(
    hooks: Sequence[Any],
    event: str,
    ctx: PredictContext,
    *,
    raise_errors: bool = True,
    lock: Any = None,
) -> None:
    """Call `event` on every hook that implements it.

    `raise_errors=False` warns and continues, for hooks (telemetry) that must not fail a
    request. `lock` serialises dispatch for hooks that are not safe to run concurrently.
    """
    for hook in hooks:
        method = getattr(hook, event, None)
        if method is None:
            continue
        try:
            if lock is not None:
                with lock:
                    method(ctx)
            else:
                method(ctx)
        except Exception as exc:  # noqa: BLE001 -- policy depends on raise_errors
            if raise_errors:
                raise
            warnings.warn(
                "laya: hook %s.%s failed: %s" % (type(hook).__name__, event, exc),
                RuntimeWarning,
                stacklevel=2,
            )
