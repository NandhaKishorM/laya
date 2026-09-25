"""Bounded request queue and dynamic batcher for ``laya-serve``.

Pure Python (stdlib only): the queue and the batcher can be unit tested without FastAPI or
torch. ``serve.create_app`` wires a ``Batcher`` around ``Router.predict_batch`` so concurrent
``POST /v1/systemone`` requests share forward passes instead of running one at a time.

Batching is opt-in and behavior-preserving by default: with ``LAYA_BATCH_MAX=1`` and
``LAYA_BATCH_WINDOW_MS=0`` each request is dispatched immediately through the same single
inference worker the endpoint used before. Set ``LAYA_BATCH_WINDOW_MS`` to collect requests
arriving in a short window and ``LAYA_BATCH_MAX`` to size the batch; ``LAYA_BATCH_MAX_ROWS``
bounds ``requests x questions`` rows per forward (a forward holds that many rows, not just the
requests); set ``LAYA_QUEUE_MAX`` to bound the queue and reject overflow with 503.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

log = logging.getLogger("laya.serve")

# Opt-in defaults: batch_max=1 and window_ms=0 dispatch each request immediately, so an
# unconfigured server behaves as it did before batching existed. `queue_max=0` is an
# unbounded queue (no backpressure) and `request_timeout_s=0` disables the per-request timeout.
DEFAULT_WINDOW_MS = 0
DEFAULT_BATCH_MAX = 1
DEFAULT_QUEUE_MAX = 0
DEFAULT_REQUEST_TIMEOUT_S = 0.0
# A forward pass holds `requests x questions` rows, so bounding requests is not enough. 64
# matches MAX_QUESTIONS: one maximal single request still fits, and a batch cannot grow past
# this many rows however many concurrent callers arrive.
DEFAULT_BATCH_MAX_ROWS = 64
DEFAULT_RETRY_AFTER = 1
# Upper bound on how long shutdown waits for the in-flight batch before abandoning it.
DEFAULT_SHUTDOWN_GRACE_S = 30.0


class ConfigError(ValueError):
    """A ``LAYA_*`` value is unusable; the message names the variable and the value."""


class RequestQueueFull(Exception):
    """The bounded queue is full; the caller should retry after ``retry_after`` seconds."""

    def __init__(self, retry_after: int = DEFAULT_RETRY_AFTER, reason: str = "request queue is full"):
        super().__init__(reason)
        self.retry_after = retry_after


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ConfigError("%s must be an integer, got %r" % (name, raw))
    if value < minimum:
        raise ConfigError("%s must be >= %d, got %d" % (name, minimum, value))
    return value


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        raise ConfigError("%s must be a number, got %r" % (name, raw))
    if value < minimum:
        raise ConfigError("%s must be >= %s, got %s" % (name, minimum, value))
    return value


@dataclass
class ServingConfig:
    """Batching and queue settings, read from ``LAYA_*`` once per app."""

    batch_window_ms: int = DEFAULT_WINDOW_MS
    batch_max: int = DEFAULT_BATCH_MAX
    batch_max_rows: int = DEFAULT_BATCH_MAX_ROWS
    queue_max: int = DEFAULT_QUEUE_MAX
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S

    @classmethod
    def from_env(cls) -> "ServingConfig":
        return cls(
            batch_window_ms=_env_int("LAYA_BATCH_WINDOW_MS", DEFAULT_WINDOW_MS, 0),
            batch_max=_env_int("LAYA_BATCH_MAX", DEFAULT_BATCH_MAX, 1),
            batch_max_rows=_env_int("LAYA_BATCH_MAX_ROWS", DEFAULT_BATCH_MAX_ROWS, 0),
            queue_max=_env_int("LAYA_QUEUE_MAX", DEFAULT_QUEUE_MAX, 0),
            request_timeout_s=_env_float("LAYA_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S, 0.0),
        )


@dataclass
class PendingRequest:
    """One queued request and the future its HTTP handler is waiting on."""

    payload: Dict[str, Any]
    future: "asyncio.Future"
    enqueued_at: float = field(default_factory=time.perf_counter)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class Batcher:
    """Collects concurrent requests into one ``Router.predict_batch`` call.

    A bounded queue provides backpressure; each request waits on its own future. When the
    router isolates batch failures (``supports_batch_isolation``), a request that fails is
    answered with its own exception and the rest of the batch is unaffected, under a single
    Router lifecycle. A router without that support falls back to one ``Router.predict`` per
    request, and a router with no ``predict_batch`` runs ``predict`` for every request.
    """

    def __init__(self, router: Any, config: ServingConfig, executor: Any):
        self.router = router
        self.config = config
        self.executor = executor
        self.queue: "asyncio.Queue[Optional[PendingRequest]]" = asyncio.Queue(maxsize=config.queue_max)
        self._task: Optional["asyncio.Task"] = None
        self._running = False
        # A request pulled to start a batch that the row cap would not let in; it leads the
        # next batch so arrival order is preserved and it is never dropped.
        self._carry: Optional[PendingRequest] = None

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    def start(self) -> None:
        """Start the worker. Idempotent, so the request path can start it without a lifespan."""
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def submit(self, payload: Dict[str, Any]) -> PendingRequest:
        """Enqueue a request and return its ``PendingRequest``.

        Raises `RequestQueueFull` when the bounded queue is full.
        """
        pending = PendingRequest(payload=payload, future=asyncio.get_running_loop().create_future())
        try:
            self.queue.put_nowait(pending)
        except asyncio.QueueFull:
            raise RequestQueueFull()
        return pending

    async def _run(self) -> None:
        while self._running:
            if self._carry is not None:
                pending, self._carry = self._carry, None
            else:
                try:
                    pending = await self.queue.get()
                except asyncio.CancelledError:
                    break
                if pending is None:                  # shutdown sentinel
                    break
            batch = [pending]
            rows = len(pending.payload.get("questions") or {})
            deadline = time.perf_counter() + self.config.batch_window_ms / 1000.0
            while len(batch) < self.config.batch_max:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if nxt is None:                      # shutdown sentinel arrived mid-collection
                    self._running = False
                    break
                nxt_rows = len(nxt.payload.get("questions") or {})
                if self.config.batch_max_rows and rows + nxt_rows > self.config.batch_max_rows:
                    # Carry `nxt` into the next batch: it keeps arrival order, and a single
                    # request larger than the cap still dispatches alone instead of starving.
                    self._carry = nxt
                    break
                batch.append(nxt)
                rows += nxt_rows
            await self._dispatch(batch)

    async def _dispatch(self, batch: Sequence[PendingRequest]) -> None:
        for pending in batch:
            pending.started_at = time.perf_counter()
        predict_batch = getattr(self.router, "predict_batch", None)
        if predict_batch is None:
            for pending in batch:
                await self._dispatch_one(pending)
            return
        payloads = [p.payload for p in batch]
        loop = asyncio.get_running_loop()
        isolates = bool(getattr(self.router, "supports_batch_isolation", False))

        def call():
            if isolates:
                return predict_batch(payloads, batch_size=self.config.batch_max, isolate_errors=True)
            return predict_batch(payloads, batch_size=self.config.batch_max)

        try:
            results = await loop.run_in_executor(self.executor, call)
        except Exception as exc:  # noqa: BLE001 -- surfaced per request
            if isolates:
                # A per-request failure is returned in `results`, not raised, so an exception
                # here is router-level (for example a checkpoint load error) and fails the batch.
                for pending in batch:
                    self._fail(pending, exc)
                return
            log.warning("batch of %d failed, retrying per request: %s: %s",
                        len(batch), type(exc).__name__, exc)
            for pending in batch:
                await self._dispatch_one(pending)
            return

        for pending, result in zip(batch, results):
            if isinstance(result, BaseException):
                self._fail(pending, result)
            else:
                self._resolve(pending, result)

    async def _dispatch_one(self, pending: PendingRequest) -> None:
        payload = pending.payload
        kwargs = {key: payload[key] for key in ("model", "task", "lang", "lang_guess")
                  if payload.get(key) is not None}
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.router.predict(payload["state"], payload["questions"], **kwargs),
            )
            self._resolve(pending, result)
        except Exception as exc:  # noqa: BLE001 -- surfaced to this one request
            if not pending.future.done():
                pending.future.set_exception(exc)

    @staticmethod
    def _resolve(pending: PendingRequest, result: Dict[str, Any]) -> None:
        # A timed-out or cancelled request already resolved its future; do not set it twice.
        if pending.future.done():
            return
        pending.finished_at = time.perf_counter()
        pending.future.set_result(result)

    @staticmethod
    def _fail(pending: PendingRequest, exc: BaseException) -> None:
        # Same guard as `_resolve`: a request that timed out already resolved its future.
        if pending.future.done():
            return
        pending.finished_at = time.perf_counter()
        pending.future.set_exception(exc)

    async def aclose(self, grace_s: float = DEFAULT_SHUTDOWN_GRACE_S) -> None:
        """Stop the worker and fail anything still queued, so no request is left waiting."""
        self._running = False
        if self._carry is not None:                  # pulled but not yet dispatched
            carry, self._carry = self._carry, None
            if not carry.future.done():
                carry.future.set_exception(RequestQueueFull(reason="server is shutting down"))
        if self._task is not None:
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            try:
                await asyncio.wait_for(self._task, timeout=grace_s)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        while True:
            try:
                pending = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if pending is not None and not pending.future.done():
                pending.future.set_exception(RequestQueueFull(reason="server is shutting down"))
