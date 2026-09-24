"""Production serving primitives for ``laya-serve``: batching, backpressure and metrics.

Pure Python: no FastAPI, no torch, so the queue and the batcher can be unit tested on their own.
``serve.create_app`` wires these into the HTTP layer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from .state import LocalState, StateStore

try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
    from prometheus_client import generate_latest
    _PROMETHEUS = True
except ImportError:  # pragma: no cover - depends on the environment
    _PROMETHEUS = False

DEFAULT_WINDOW_MS = 10
DEFAULT_BATCH_MAX = 16
DEFAULT_QUEUE_MAX = 64
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_SHUTDOWN_GRACE_S = 30.0
DEFAULT_INFER_WORKERS = 1
DEFAULT_RETRY_AFTER = 1
DEFAULT_BREAKER_COOLDOWN_S = 30.0

log = logging.getLogger("laya.serve")


class ConfigError(ValueError):
    """A ``LAYA_*`` value is unusable; the message names the variable and the value."""


class RequestQueueFull(Exception):
    """The bounded queue is full; the caller should retry after ``retry_after`` seconds."""

    def __init__(self, retry_after: int = DEFAULT_RETRY_AFTER, reason: str = "request queue is full",
                 code: str = "queue_full"):
        super().__init__(reason)
        self.retry_after = retry_after
        self.code = code


class CircuitOpen(Exception):
    """Inference kept failing; the breaker is open and the request was not dispatched."""

    def __init__(self, retry_after: int = DEFAULT_RETRY_AFTER):
        super().__init__("inference circuit breaker is open")
        self.retry_after = retry_after
        self.code = "circuit_open"


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


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    value = str(raw).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    # Silently treating a typo like LAYA_METRICS=enabled as false hides a disabled metrics
    # endpoint until someone notices the dashboards are empty; fail fast instead.
    raise ConfigError("%s must be a boolean (one of %s), got %r" % (name, ", ".join(_TRUE + _FALSE), raw))


@dataclass
class ServingConfig:
    """Batching, queue and lifecycle settings, read from ``LAYA_*`` once per app."""

    batch_window_ms: int = DEFAULT_WINDOW_MS
    batch_max: int = DEFAULT_BATCH_MAX
    queue_max: int = DEFAULT_QUEUE_MAX
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    shutdown_grace_s: float = DEFAULT_SHUTDOWN_GRACE_S
    infer_workers: int = DEFAULT_INFER_WORKERS
    metrics_enabled: bool = True
    log_format: str = "text"
    warmup: bool = True
    # Per-client in-flight cap. 0 disables admission control (the single global queue still
    # bounds total load with 503); set it when several callers share one server so one burst
    # cannot starve the rest. Over-limit callers get 429 + Retry-After.
    client_max_inflight: int = 0
    # Circuit breaker: 0 consecutive-failure threshold means disabled. It is keyed per client.
    breaker_threshold: int = 0
    breaker_cooldown_s: float = DEFAULT_BREAKER_COOLDOWN_S
    # Successful half-open probes required before a client's breaker closes. >1 avoids flapping
    # on a service that is still warming up and passes one probe then fails the next.
    breaker_probe_successes: int = 2
    # Trust X-Forwarded-For/Proto from `forwarded_allow_ips` (uvicorn). Off by default: only
    # enable behind a proxy you control, or a client can spoof its own identity in logs and
    # per-IP admission.
    proxy_headers: bool = False
    forwarded_allow_ips: str = "127.0.0.1"
    # Shared state backend (e.g. redis://host:6379/0). Unset keeps everything in-process.
    state_url: Optional[str] = None
    # With a shared store, share the per-client breaker too; False keeps it per-replica while
    # admission is still shared.
    shared_breaker: bool = True
    # Breaker entry TTL: None auto-derives max(cooldown, 60); a positive value is explicit; 0
    # disables the TTL (only the size backstop sweeps).
    breaker_entry_ttl_s: Optional[float] = None
    # What to do when a shared state store errors: "local" degrades to per-replica enforcement
    # (still capped, no cluster coordination), "open" lets every request through unenforced.
    state_failure_policy: str = "local"

    @classmethod
    def from_env(cls) -> "ServingConfig":
        log_format = (os.environ.get("LAYA_LOG_FORMAT", "text") or "text").strip().lower()
        if log_format not in ("text", "json"):
            raise ConfigError("LAYA_LOG_FORMAT must be 'text' or 'json', got %r" % log_format)
        return cls(
            batch_window_ms=_env_int("LAYA_BATCH_WINDOW_MS", DEFAULT_WINDOW_MS, 0),
            batch_max=_env_int("LAYA_BATCH_MAX", DEFAULT_BATCH_MAX, 1),
            queue_max=_env_int("LAYA_QUEUE_MAX", DEFAULT_QUEUE_MAX, 1),
            request_timeout_s=_env_float("LAYA_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S, 0.0),
            shutdown_grace_s=_env_float("LAYA_SHUTDOWN_GRACE_S", DEFAULT_SHUTDOWN_GRACE_S, 0.0),
            infer_workers=_env_int("LAYA_INFER_WORKERS", DEFAULT_INFER_WORKERS, 1),
            metrics_enabled=_env_bool("LAYA_METRICS", True),
            log_format=log_format,
            warmup=_env_bool("LAYA_WARMUP", True),
            client_max_inflight=_env_int("LAYA_CLIENT_MAX_INFLIGHT", 0, 0),
            breaker_threshold=_env_int("LAYA_BREAKER_THRESHOLD", 0, 0),
            breaker_cooldown_s=_env_float("LAYA_BREAKER_COOLDOWN_S", DEFAULT_BREAKER_COOLDOWN_S, 0.0),
            breaker_probe_successes=_env_int("LAYA_BREAKER_PROBE_SUCCESSES", 2, 1),
            proxy_headers=_env_bool("LAYA_PROXY_HEADERS", False),
            forwarded_allow_ips=(os.environ.get("LAYA_FORWARDED_ALLOW_IPS", "").strip() or "127.0.0.1"),
            state_url=(os.environ.get("LAYA_STATE_URL", "").strip() or None),
            shared_breaker=_env_bool("LAYA_SHARED_BREAKER", True),
            breaker_entry_ttl_s=_env_entry_ttl(),
            state_failure_policy=_env_failure_policy(),
        )


def _env_failure_policy() -> str:
    """LAYA_STATE_FAILURE_POLICY: 'local' (degrade to per-replica) or 'open' (unenforced)."""
    value = (os.environ.get("LAYA_STATE_FAILURE_POLICY", "local") or "local").strip().lower()
    if value not in ("local", "open"):
        raise ConfigError("LAYA_STATE_FAILURE_POLICY must be 'local' or 'open', got %r" % value)
    return value


def _env_entry_ttl() -> Optional[float]:
    """LAYA_BREAKER_ENTRY_TTL_S: 'auto'/unset -> None; a non-negative number as given."""
    raw = os.environ.get("LAYA_BREAKER_ENTRY_TTL_S")
    if raw is None or raw.strip() == "" or raw.strip().lower() == "auto":
        return None
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):
        raise ConfigError("LAYA_BREAKER_ENTRY_TTL_S must be 'auto', a number, or 0, got %r" % raw)
    if value < 0:
        raise ConfigError("LAYA_BREAKER_ENTRY_TTL_S must be >= 0, got %s" % value)
    return value


@dataclass
class PendingRequest:
    payload: Dict[str, Any]
    future: "asyncio.Future"
    enqueued_at: float = field(default_factory=time.perf_counter)
    started_at: Optional[float] = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Admission/breaker identity of the caller, set by the HTTP layer; "global" when unknown.
    client_key: str = "global"


class _Noop:
    """A stand-in instrument for `LAYA_METRICS=0`; swallows every recorded operation."""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        return self

    def dec(self, *args, **kwargs):
        return self

    def set(self, *args, **kwargs):
        return self

    def observe(self, *args, **kwargs):
        return self


class Metrics:
    """A per-app Prometheus registry and instruments.

    A fresh registry per app means tests can build many apps without the duplicate
    registration error the default global registry raises.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled) and _PROMETHEUS
        if enabled and not _PROMETHEUS:
            raise RuntimeError("prometheus-client is required for /metrics; install 'laya[serve]'")
        self.registry = CollectorRegistry() if self.enabled else None
        if not self.enabled:
            noop = _Noop()
            self.requests = self.latency = self.queue_wait = self.inference = self.batch_size = noop
            self.queue_depth = self.active = self.batches = self.loaded = noop
            self.queue_oldest = self.rejections = noop
            self.circuits_open = self.clients_active = self.circuits_tracked = noop
            self.shared_state_errors = self.inflight_total = noop
            self.state_degraded = self.state_backend = noop
            return
        self.requests = Counter("laya_requests_total", "Decisions by model and outcome",
                                ["model", "status"], registry=self.registry)
        self.latency = Histogram("laya_request_latency_seconds", "End-to-end request latency",
                                 registry=self.registry,
                                 buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
        self.queue_wait = Histogram("laya_queue_wait_seconds", "Time from arrival to dispatch",
                                    registry=self.registry,
                                    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1))
        self.inference = Histogram("laya_inference_seconds", "Time for one batch forward",
                                   registry=self.registry,
                                   buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
        self.batch_size = Histogram("laya_batch_size", "Requests per forward pass",
                                    registry=self.registry, buckets=(1, 2, 4, 8, 16, 32, 64))
        self.queue_depth = Gauge("laya_queue_depth", "Requests waiting in the queue", registry=self.registry)
        self.active = Gauge("laya_active_requests", "Requests currently dispatching", registry=self.registry)
        self.batches = Counter("laya_batches_total", "Forward passes by outcome",
                               ["result"], registry=self.registry)
        self.loaded = Gauge("laya_models_loaded", "Resident checkpoints", registry=self.registry)
        self.queue_oldest = Gauge("laya_queue_oldest_seconds",
                                  "Age of the oldest request still waiting in the queue",
                                  registry=self.registry)
        self.rejections = Counter("laya_rejections_total", "Requests shed before inference, by reason",
                                  ["reason"], registry=self.registry)
        self.circuits_open = Gauge("laya_circuits_open",
                                   "Clients whose inference breaker is currently open",
                                   registry=self.registry)
        self.clients_active = Gauge("laya_clients_active",
                                    "Clients with at least one request in flight",
                                    registry=self.registry)
        self.circuits_tracked = Gauge("laya_circuits_tracked",
                                      "Clients with breaker state tracked (local view)",
                                      registry=self.registry)
        self.shared_state_errors = Counter("laya_shared_state_errors_total",
                                           "Shared-state backend errors (see state_failure_policy)",
                                           ["op"], registry=self.registry)
        self.inflight_total = Gauge("laya_inflight_total",
                                    "In-flight requests; cluster-wide under a shared store",
                                    registry=self.registry)
        self.state_degraded = Gauge("laya_state_degraded",
                                    "1 when the last shared-state op fell back",
                                    registry=self.registry)
        self.state_backend = Gauge("laya_state_backend", "Active state backend (value is 1)",
                                   ["mode"], registry=self.registry)

    def render(self) -> Tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    def count(self, model: Optional[str], status: str) -> None:
        if self.enabled:
            self.requests.labels(model=model or "auto", status=status).inc()

    def reject(self, reason: str) -> None:
        """Count a request shed before inference. `reason` is queue_full/draining/client_limit/circuit_open."""
        if self.enabled:
            self.rejections.labels(reason=reason).inc()

    def observe_queue_age(self, seconds: float) -> None:
        if self.enabled:
            self.queue_oldest.set(seconds)

    def shared_state_error(self, op: str) -> None:
        """A shared-state backend call failed; policy decides whether it fell back."""
        if self.enabled:
            self.shared_state_errors.labels(op=op).inc()
            self.state_degraded.set(1)

    def state_recovered(self) -> None:
        if self.enabled:
            self.state_degraded.set(0)

    def set_state_backend(self, mode: str) -> None:
        if self.enabled:
            self.state_backend.labels(mode=mode).set(1)

    def observe_success(self, pending: PendingRequest, result: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        model = ((result or {}).get("routing") or {}).get("model") or (result or {}).get("model") or "auto"
        self.requests.labels(model=model, status="ok").inc()
        self.latency.observe(now - pending.enqueued_at)
        if pending.started_at is not None:
            self.queue_wait.observe(pending.started_at - pending.enqueued_at)

    def observe_failure(self, pending: PendingRequest, status: str = "error") -> None:
        if not self.enabled:
            return
        model = pending.payload.get("model") or "auto"
        self.requests.labels(model=model, status=status).inc()
        self.latency.observe(time.perf_counter() - pending.enqueued_at)


class JsonFormatter(logging.Formatter):
    """One JSON object per log line, for pipelines that parse structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "path", "status", "model", "queue_wait_ms", "infer_ms", "total_ms",
                    "input_tokens"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class Batcher:
    """Collects concurrent requests into one ``Router.predict_batch`` call.

    A bounded queue provides backpressure; each request waits on its own future. A forward
    that raises falls back to one call per request, so a single bad state cannot fail the
    batch. ``drain`` stops intake and lets in-flight work finish within a grace period.
    """

    def __init__(self, router: Any, config: ServingConfig, metrics: Metrics, executor: Any,
                 state: "StateStore" = None):
        self.router = router
        self.config = config
        self.metrics = metrics
        self.executor = executor
        self.state = state if state is not None else LocalState(config)
        self.queue: "asyncio.Queue[Optional[PendingRequest]]" = asyncio.Queue(maxsize=config.queue_max)
        self._task: Optional["asyncio.Task"] = None
        self._running = False
        self._draining = False

    @property
    def draining(self) -> bool:
        return self._draining

    def begin_drain(self) -> None:
        """Stop accepting new work but keep the worker running to finish what is queued.

        This is the hook a Kubernetes ``preStop`` calls before SIGTERM: readiness and
        ``submit`` start failing with 503 while the load balancer drops the pod, so the queue
        can drain cleanly before the real shutdown signal arrives.
        """
        self._draining = True

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    async def refresh_metrics(self) -> None:
        """Publish live queue and state gauges. Called before a scrape so values are current."""
        self.metrics.queue_depth.set(self.queue.qsize())
        waiting = getattr(self.queue, "_queue", None)
        if waiting and waiting[0] is not None:
            self.metrics.observe_queue_age(max(0.0, time.perf_counter() - waiting[0].enqueued_at))
        else:
            self.metrics.observe_queue_age(0.0)
        snap = await self.state.snapshot()
        self.metrics.clients_active.set(snap.get("clients_active", 0.0))
        self.metrics.circuits_open.set(snap.get("circuits_open", 0.0))
        self.metrics.circuits_tracked.set(snap.get("circuits_tracked", 0.0))
        self.metrics.inflight_total.set(snap.get("inflight_total", 0.0))

    def start(self) -> None:
        """Start the worker. Idempotent, so a request path can start it without a lifespan."""
        if self._task is not None and not self._task.done():
            return
        if self._draining:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def submit(self, payload: Dict[str, Any], client_key: str = "global") -> "asyncio.Future":
        """Enqueue a request.

        Raises `RequestQueueFull` when the queue is full (``queue_full``) or the server is
        draining (``draining``), and `CircuitOpen` when this client's breaker is open.
        """
        if self._draining:
            raise RequestQueueFull(reason="server is shutting down", code="draining")
        decision = await self.state.breaker_allow(
            client_key, self.config.breaker_threshold, self.config.breaker_cooldown_s,
            self.config.breaker_probe_successes)
        if not decision.allowed:
            raise CircuitOpen(retry_after=decision.retry_after)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        pending = PendingRequest(payload=payload, future=future, client_key=client_key)
        try:
            self.queue.put_nowait(pending)
        except asyncio.QueueFull:
            # The HTTP layer counts the rejection once, from the payload's model label; counting
            # it here too made every shed request land twice in `laya_requests_total`.
            raise RequestQueueFull()
        self.metrics.queue_depth.set(self.queue.qsize())
        return future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                pending = await self.queue.get()
            except asyncio.CancelledError:
                break
            if pending is None:                      # drain sentinel
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
                if nxt is None:                      # drain sentinel arrived mid-collection
                    self._running = False
                    break
                batch.append(nxt)
            self.metrics.queue_depth.set(self.queue.qsize())
            await self._dispatch(loop, batch)

    async def _dispatch(self, loop: Any, batch: Sequence[PendingRequest]) -> None:
        payloads = [p.payload for p in batch]
        for pending in batch:
            pending.started_at = time.perf_counter()
            self.metrics.active.inc()
        started = time.perf_counter()
        try:
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.router.predict_batch(payloads, batch_size=self.config.batch_max),
            )
            self.metrics.batches.labels(result="ok").inc()
            self.metrics.inference.observe(time.perf_counter() - started)
            self.metrics.batch_size.observe(len(batch))
            for pending, result in zip(batch, results):
                await self._resolve(pending, result)
        except Exception as exc:  # noqa: BLE001 -- isolate the poison request from the batch
            self.metrics.batches.labels(result="fallback").inc()
            log.warning("batch of %d failed, retrying per request: %s: %s",
                        len(batch), type(exc).__name__, exc)
            for pending in batch:
                await self._dispatch_one(loop, pending)
        finally:
            for _ in batch:
                self.metrics.active.dec()

    async def _dispatch_one(self, loop: Any, pending: PendingRequest) -> None:
        payload = pending.payload
        kwargs = {key: payload[key] for key in ("model", "task", "lang", "lang_guess")
                  if payload.get(key) is not None}
        try:
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.router.predict(payload["state"], payload["questions"], **kwargs),
            )
            await self._resolve(pending, result)
        except Exception as exc:  # noqa: BLE001 -- surfaced to this one request
            self.metrics.batches.labels(result="error").inc()
            self.metrics.observe_failure(pending)
            await self.state.breaker_failure(
                pending.client_key, self.config.breaker_threshold, self.config.breaker_cooldown_s)
            log.warning("request failed: %s: %s", type(exc).__name__, exc, exc_info=True)
            if not pending.future.done():
                pending.future.set_exception(exc)

    async def _resolve(self, pending: PendingRequest, result: Dict[str, Any]) -> None:
        if pending.future.done():
            return
        pending.future.set_result(result)
        await self.state.breaker_success(pending.client_key, self.config.breaker_probe_successes)
        self.metrics.observe_success(pending, result)

    async def drain(self) -> None:
        """Stop intake, let the in-flight work finish within the grace budget, then stop the worker.

        Work still queued when SIGTERM arrives is failed with ``RequestQueueFull`` (503) rather
        than waited for: the grace bounds shutdown, and a caller is answered instead of being left
        to time out. ``shutdown_grace_s`` is the single budget for the in-flight pass.
        """
        self._draining = True
        self._running = False
        try:
            self.queue.put_nowait(None)              # wake a worker blocked on get()
        except asyncio.QueueFull:
            pass
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self.config.shutdown_grace_s)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        # Whatever the worker did not pick up is failed now, so a caller is answered during
        # shutdown instead of waiting out the request timeout.
        while True:
            try:
                pending = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if pending is not None and not pending.future.done():
                pending.future.set_exception(
                    RequestQueueFull(reason="server is shutting down", code="draining"))
        await self.refresh_metrics()


def log_extra(request_id: str, **fields: Any) -> Dict[str, Any]:
    """Build the ``extra`` mapping for a structured log record."""
    out = {"request_id": request_id}
    out.update(fields)
    return out
