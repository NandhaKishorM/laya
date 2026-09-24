"""Bounded request queue and dynamic batcher for ``laya-serve``.

Pure Python (stdlib only): the queue and the batcher can be unit tested without FastAPI or
torch. ``serve.create_app`` wires a ``Batcher`` around ``Router.predict_batch`` so concurrent
``POST /v1/systemone`` requests share forward passes instead of running one at a time.

Batching is opt-in and behavior-preserving by default: with ``LAYA_BATCH_MAX=1`` and
``LAYA_BATCH_WINDOW_MS=0`` each request is dispatched immediately through the same single
inference worker the endpoint used before. Set ``LAYA_BATCH_WINDOW_MS`` to collect requests
arriving in a short window and ``LAYA_BATCH_MAX`` to size the batch; set ``LAYA_QUEUE_MAX`` to
bound the queue and reject overflow with 503 instead of letting it grow without limit.
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
    queue_max: int = DEFAULT_QUEUE_MAX
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S

    @classmethod
    def from_env(cls) -> "ServingConfig":
        return cls(
            batch_window_ms=_env_int("LAYA_BATCH_WINDOW_MS", DEFAULT_WINDOW_MS, 0),
            batch_max=_env_int("LAYA_BATCH_MAX", DEFAULT_BATCH_MAX, 1),
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

    A bounded queue provides backpressure; each request waits on its own future. A batch
    that raises falls back to one ``Router.predict`` per request, so a single bad state
    cannot fail the whole batch. If the router has no ``predict_batch``, every request is
    dispatched through ``predict`` (the pre-batching behavior).
    """

    def __init__(self, router: Any, config: ServingConfig, executor: Any):
        self.router = router
        self.config = config
        self.executor = executor
        self.queue: "asyncio.Queue[Optional[PendingRequest]]" = asyncio.Queue(maxsize=config.queue_max)
        self._task: Optional["asyncio.Task"] = None
        self._running = False

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
            try:
                pending = await self.queue.get()
            except asyncio.CancelledError:
                break
            if pending is None:                      # shutdown sentinel
                break
            batch = [pending]
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
                batch.append(nxt)
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
        try:
            results = await loop.run_in_executor(
                self.executor,
                lambda: predict_batch(payloads, batch_size=self.config.batch_max),
            )
            for pending, result in zip(batch, results):
                self._resolve(pending, result)
        except Exception as exc:  # noqa: BLE001 -- isolate the poison request from the batch
            log.warning("batch of %d failed, retrying per request: %s: %s",
                        len(batch), type(exc).__name__, exc)
            for pending in batch:
                await self._dispatch_one(pending)

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

    async def aclose(self, grace_s: float = DEFAULT_SHUTDOWN_GRACE_S) -> None:
        """Stop the worker and fail anything still queued, so no request is left waiting."""
        self._running = False
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
