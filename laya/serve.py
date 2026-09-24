"""HTTP server exposing Laya over TypeSafe Jev's ``/v1/systemone`` wire protocol.

Laya's ``predict()`` output is already schema-compatible with the Jev decision
API -- ``choice`` / ``score`` / ``noul`` answers and a ``{input_tokens,
output_tokens}`` usage block -- so a client written against Jev (for example the
`hs-jev` Haskell client) can point its ``baseUrl`` at this server and keep
working unchanged. It also ships the production serving layer: a bounded request
queue with backpressure, dynamic batching through ``Router.predict_batch``,
Prometheus metrics, readiness and liveness probes, graceful drain, and request ids.

Configuration is entirely via environment variables so the same entry point
serves a laptop dev run and a systemd unit:

======================  ============================================  =========
env var                 meaning                                        default
======================  ============================================  =========
``LAYA_HOST``           bind address                                   0.0.0.0
``LAYA_PORT``           bind port                                      8000
``LAYA_DEVICE``         torch device for every checkpoint              (auto)
``LAYA_PRELOAD``        preload checkpoints in the background at startup   1
                        (0 = load lazily on first request)
``LAYA_MODELS``         comma list to preload (english,multilingual,   (all)
                        typed-decisions); empty = every checkpoint
``LAYA_THREADS``        cap torch intra-op threads (CPU inference).    (torch
                        Keep <= physical cores; oversubscribing the     default)
                        logical/hyperthread count is a large regression.
``LAYA_AUTO_TASK``      auto-route to the typed-decisions checkpoint   0
``LAYA_API_KEY``        if set, require ``Authorization: Bearer <it>``  (none)
``LAYA_BATCH_WINDOW_MS``  collection window for a batch; 0 is immediate 10
``LAYA_BATCH_MAX``      max requests per forward pass                  16
``LAYA_QUEUE_MAX``      bounded queue; overflow is 503                 64
``LAYA_REQUEST_TIMEOUT_S``  queue + inference budget per request       120
``LAYA_SHUTDOWN_GRACE_S``   drain budget on SIGTERM                    30
``LAYA_INFER_WORKERS``  batch dispatch threads (1 is the right count)  1
``LAYA_WARMUP``         warmup inference per checkpoint after preload 1
``LAYA_CLIENT_MAX_INFLIGHT``  per-client in-flight cap; 429 over it   0 (off)
``LAYA_BREAKER_THRESHOLD``    per-client failures to open breaker      0 (off)
``LAYA_BREAKER_COOLDOWN_S``   breaker open duration before a probe      30
``LAYA_BREAKER_PROBE_SUCCESSES``  half-open probes to close a breaker   2
``LAYA_PROXY_HEADERS``  trust X-Forwarded-* from LAYA_FORWARDED_ALLOW_IPS 0
``LAYA_FORWARDED_ALLOW_IPS``  peers whose proxy headers are trusted     127.0.0.1
``LAYA_STATE_URL``      shared admission/breaker backend (redis://...)  (in-process)
``LAYA_SHARED_BREAKER`` share the per-client breaker across replicas    1
``LAYA_BREAKER_ENTRY_TTL_S``  breaker entry TTL: auto/number/0          (auto)
``LAYA_STATE_FAILURE_POLICY``  on store error: local degrade / open      local
``LAYA_METRICS``        expose ``/metrics``                            1
``LAYA_LOG_FORMAT``     ``text`` or ``json`` structured logs           text
``LAYA_LOG_LEVEL``      uvicorn log level                              info
======================  ============================================  =========

Imports of heavy dependencies (fastapi, uvicorn, torch via Router) are all
deferred into the functions that need them, so ``import laya.serve`` stays cheap
and touches no GPU -- which is what keeps the Nix ``pythonImportsCheck`` honest.
"""
import hmac
import json
import logging
import os
from typing import Any, Dict, Optional

# The three checkpoint names the router understands; used to decide whether a
# client's `model` field names a Laya checkpoint (honour it) or is some other
# Jev model id (ignore it and let the router auto-select).
_KNOWN_MODELS = {"english", "multilingual", "typed-decisions"}

# Guardrails for unauthenticated remote input. The state is tokenized once per
# question and collated into one tensor, so an unbounded body can OOM the worker;
# the single-worker pool means one large request would also starve /health.
MAX_QUESTIONS = 64
MAX_STATE_CHARS = 50000
MAX_BODY_BYTES = 2 * 1024 * 1024
# Public Hugging Face ids, accepted so a client can name a checkpoint. The root bundle is
# deliberately absent: the documented ``convaiinnovations/laya`` value means
# "let the Router choose", rather than pinning the English checkpoint.
_PUBLISHED_MODEL_IDS = {
    "convaiinnovations/laya-multilingual": "multilingual",
    "convaiinnovations/laya-typed-decisions": "typed-decisions",
}


def _env_bool(name: str, default: bool) -> bool:
    from .serving import _env_bool as _strict_env_bool

    return _strict_env_bool(name, default)


def _resolve_model(model: Optional[str]) -> Optional[str]:
    """Map a client's `model` field onto a Laya checkpoint, or None to auto-route."""
    if not model:
        return None
    published = _PUBLISHED_MODEL_IDS.get(str(model).strip().lower())
    if published is not None:
        return published
    from .router import normalise_name

    # normalise_name raises ValueError on anything that is not a known checkpoint
    # or alias. A Jev client's `model` field (e.g. "jev-1") is expected to miss;
    # treat that as "no explicit checkpoint" and let the router auto-select.
    try:
        key = normalise_name(model)
    except Exception:
        return None
    return key if key in _KNOWN_MODELS else None


def _resolve_port() -> int:
    """Port from LAYA_PORT, validated. Exits with a message instead of a traceback."""
    raw = os.environ.get("LAYA_PORT", "8000")
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError):
        raise SystemExit("invalid LAYA_PORT %r: must be an integer 1-65535" % (raw,))
    if not 1 <= port <= 65535:
        raise SystemExit("invalid LAYA_PORT %r: must be an integer 1-65535" % (raw,))
    return port


def _check_request_limits(state: Any, questions: Any) -> None:
    """Reject oversized inference requests before tokenization (413)."""
    from fastapi import HTTPException

    if not isinstance(questions, dict):
        raise HTTPException(status_code=400, detail="'questions' must be an object")
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=413,
                            detail="too many questions (%d > %d)" % (len(questions), MAX_QUESTIONS))
    try:
        state_len = len(state) if isinstance(state, str) else len(str(state))
    except Exception:
        state_len = MAX_STATE_CHARS + 1
    if state_len > MAX_STATE_CHARS:
        raise HTTPException(status_code=413,
                            detail="state too large (%d > %d chars)" % (state_len, MAX_STATE_CHARS))


async def _read_body_capped(request: Any) -> bytes:
    """Read the request body, refusing to buffer more than ``MAX_BODY_BYTES``.

    ``Content-Length`` cannot be the only gate. It is a value the client chooses,
    and under ``Transfer-Encoding: chunked`` it is absent altogether -- HTTP/2 and
    HTTP/3 have no such header at all -- so a request that simply omits it was
    read into memory in full, whatever its size. The body is streamed here and
    abandoned as soon as it exceeds the cap, so the limit holds for every framing
    rather than only for clients that announce their length honestly.
    """
    from fastapi import HTTPException

    total = 0
    chunks = []
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            # Stop reading rather than draining the rest: the peer is already over
            # the limit and nothing further can make the request acceptable.
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _apply_thread_limit():
    """Honour LAYA_THREADS by capping torch's intra-op thread count for CPU
    inference. Returns the value applied, or None if unset/invalid. torch is
    imported only when a limit is actually requested."""
    raw = os.environ.get("LAYA_THREADS")
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    if n <= 0:
        return None
    import torch

    torch.set_num_threads(n)
    return n


def _preload_names():
    """Checkpoints named by ``LAYA_MODELS`` to preload; ``None`` means every checkpoint."""
    models_env = os.environ.get("LAYA_MODELS", "").strip()
    return [m.strip() for m in models_env.split(",") if m.strip()] or None


# A tiny representative request run once per resident checkpoint after preload, to pay the
# first-call JIT/compile cost at startup instead of on the first user request.
_WARMUP_STATE = "laya warmup"
_WARMUP_QUESTIONS = {"warmup": {"type": "noul", "instructions": "is this a warmup?"}}


def _warmup(router: Any) -> None:
    """Best-effort warmup pass over the resident checkpoints; never fatal."""
    log = logging.getLogger("laya.serve")
    for name in list(getattr(router, "loaded", []) or [None]):
        try:
            router.predict(_WARMUP_STATE, _WARMUP_QUESTIONS, **({"model": name} if name else {}))
        except Exception:  # noqa: BLE001 -- warmup is an optimization, not a health gate
            log.warning("warmup inference failed for %s", name or "default", exc_info=True)


def build_router(preload: bool = True):
    """Build a Router from the environment.

    ``preload=False`` lets the server bind and answer liveness while it loads checkpoints in
    the background; callers that want the old "resident before returning" behavior keep it.
    """
    from .router import Router

    _apply_thread_limit()
    device = os.environ.get("LAYA_DEVICE") or None
    router = Router(device=device, auto_task_detection=_env_bool("LAYA_AUTO_TASK", False))
    if preload and _env_bool("LAYA_PRELOAD", True):
        router.preload(_preload_names())
    return router


def create_app(router: Optional[Any] = None, config: Optional[Any] = None, state: Optional[Any] = None):
    """Build the FastAPI app.

    Pass a Router to inject one (tests); otherwise one is built from the environment.
    Checkpoints preload in the background once the server is live, so ``/health`` answers
    during a slow load and ``/ready`` stays 503 until they are resident. ``config``
    overrides ``ServingConfig.from_env()``; ``state`` overrides the admission/breaker
    backend (default ``LocalState``, or the shared backend named by ``LAYA_STATE_URL``).
    """
    import asyncio
    import logging
    import time
    import uuid
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, Header, HTTPException, Request, Response

    from .serving import (
        DEFAULT_RETRY_AFTER, Batcher, CircuitOpen, ConfigError, JsonFormatter, Metrics,
        RequestQueueFull, ServingConfig, log_extra,
    )
    from .state import build_state

    preload_enabled = False
    if router is None:
        router = build_router(preload=False)
        preload_enabled = _env_bool("LAYA_PRELOAD", True)
    if config is None:
        config = ServingConfig.from_env()
    api_key = os.environ.get("LAYA_API_KEY") or None

    log = logging.getLogger("laya.serve")
    if config.log_format == "json" and not any(isinstance(h.formatter, JsonFormatter) for h in log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
        log.setLevel(logging.INFO)

    metrics = Metrics(config.metrics_enabled)
    # Inference is synchronous torch, and a CPU call takes hundreds of milliseconds to
    # seconds, so it must not run on the event loop: one request would stall every other
    # client, `GET /health` included. One worker, because the batch width is the
    # concurrency: one `predict_batch` at a time is what a single CPU or GPU Agent wants.
    # `LAYA_INFER_WORKERS` raises it, but 1 is documented as the right default.
    pool = ThreadPoolExecutor(max_workers=config.infer_workers, thread_name_prefix="laya-infer")
    if state is not None:
        store = state
    else:
        try:
            store = build_state(config, metrics)
        except ImportError as exc:
            raise ConfigError(str(exc)) from exc
    metrics.set_state_backend("redis" if config.state_url else "local")
    batcher = Batcher(router, config, metrics, pool, store)
    state = {"ready": False, "draining": False, "preload_error": None}
    preload = {"task": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        batcher.start()
        if preload_enabled:
            async def _preload():
                loop = asyncio.get_running_loop()
                try:
                    # Off the event loop: loading checkpoints is minutes of blocking work, and the
                    # server must keep answering `/health` and `/ready` while it happens.
                    await loop.run_in_executor(pool, router.preload, _preload_names())
                    if config.warmup:
                        # Weights resident is not the same as first-request latency: a JIT/CUDA
                        # kernel pass is paid by whoever calls first unless we absorb it here.
                        await loop.run_in_executor(pool, _warmup, router)
                except Exception as exc:  # noqa: BLE001 -- surfaced on /health for a restart
                    state["preload_error"] = "%s: %s" % (type(exc).__name__, exc)
                    log.exception("preload failed; /health stays 503 so the orchestrator restarts")
                    return
                state["ready"] = True
                if metrics.enabled:
                    metrics.loaded.set(len(router.loaded))
                log.info("preload complete: loaded=%s", router.loaded)

            preload["task"] = asyncio.create_task(_preload())
        else:
            state["ready"] = True
            if metrics.enabled:
                metrics.loaded.set(len(router.loaded))
        log.info("laya-serve listening: batch_max=%d window_ms=%d queue_max=%d preload=%s",
                 config.batch_max, config.batch_window_ms, config.queue_max, preload_enabled)
        try:
            yield
        finally:
            state["draining"] = True
            if preload["task"] is not None and not preload["task"].done():
                preload["task"].cancel()
            await batcher.drain()
            await store.close()
            pool.shutdown(wait=False)
            log.info("laya-serve drained and stopped")

    app = FastAPI(
        title="laya-serve",
        summary="Laya System-1 decisions over the TypeSafe Jev /v1/systemone protocol",
        lifespan=lifespan,
    )

    # A header is latin-1 on the wire, so encode both sides before comparing: a header such as
    # `Authorization: Bearer s\xe9cret` is legal and would otherwise make the comparison raise
    # (HTTP 500) instead of answering 401. Constant-time for every header a client can send.
    expected_credential = api_key.encode("utf-8", "surrogateescape") if api_key else b""

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is None:
            return
        # RFC 7235: the auth-scheme is case-insensitive and the credentials are separated from it
        # by one or more spaces. Gateways normalize header casing, so an exact "Bearer <key>"
        # comparison rejected legitimate clients; only the key itself is compared constant-time.
        scheme, _, credential = (authorization or "").partition(" ")
        supplied = credential.strip().encode("utf-8", "surrogateescape")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected_credential):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def _declared_length_too_large(request: Request) -> None:
        if request.headers.get("content-length"):
            try:
                if int(request.headers["content-length"]) > MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="request body too large")
            except ValueError:
                pass

    def _client_key(request: Request, authorization: Optional[str]) -> str:
        """Identity for per-client admission: an explicit tenant header, else the credential, else IP."""
        header = request.headers.get("x-laya-client")
        if header:
            return "client:" + header[:200]
        if authorization:
            _, _, credential = authorization.partition(" ")
            if credential.strip():
                return "key:" + credential.strip()[:200]
        if request.client and request.client.host:
            return "ip:" + request.client.host
        return "unknown"

    async def _run_batched(payload: Dict[str, Any], client_key: str) -> Dict[str, Any]:
        batcher.start()                      # idempotent; covers tests without a lifespan
        if not await store.acquire(client_key, config.client_max_inflight):
            metrics.count(payload.get("model"), "rate_limited")
            metrics.reject("client_limit")
            raise HTTPException(status_code=429, detail="client concurrency limit exceeded",
                                headers={"Retry-After": str(DEFAULT_RETRY_AFTER)})
        try:
            try:
                future = await batcher.submit(payload, client_key)
            except RequestQueueFull as exc:
                metrics.count(payload.get("model"), "rejected")
                metrics.reject(exc.code)
                raise HTTPException(status_code=503, detail=str(exc),
                                    headers={"Retry-After": str(exc.retry_after)})
            except CircuitOpen as exc:
                metrics.count(payload.get("model"), "rejected")
                metrics.reject(exc.code)
                raise HTTPException(status_code=503, detail=str(exc),
                                    headers={"Retry-After": str(exc.retry_after)})
            try:
                return await asyncio.wait_for(future, timeout=config.request_timeout_s)
            except RequestQueueFull as exc:
                # Resolved during drain: the queue was abandoned before dispatch.
                metrics.count(payload.get("model"), "rejected")
                metrics.reject(exc.code)
                raise HTTPException(status_code=503, detail=str(exc),
                                    headers={"Retry-After": str(exc.retry_after)})
            except asyncio.TimeoutError:
                metrics.count(payload.get("model"), "timeout")
                raise HTTPException(status_code=504, detail="inference timed out")
            except asyncio.CancelledError:
                raise HTTPException(status_code=503, detail="server is shutting down")
        finally:
            await store.release(client_key)

    @app.middleware("http")
    async def _request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info("request", extra=log_extra(
            request_id, path=request.url.path, status=response.status_code,
            total_ms=round((time.perf_counter() - started) * 1000.0, 2)))
        return response

    @app.get("/health")
    def health() -> Dict[str, Any]:
        # A failed preload is terminal until restarted, so liveness must fail too; otherwise a
        # pod whose weights never loaded stays up and never serves.
        if state["preload_error"]:
            raise HTTPException(status_code=503, detail={
                "status": "error", "error": state["preload_error"], "loaded": router.loaded})
        return {
            "status": "ok",
            "loaded": router.loaded,
            "device": os.environ.get("LAYA_DEVICE") or "auto",
        }

    @app.get("/ready")
    async def ready():
        if not state["ready"] or state["draining"] or state["preload_error"]:
            raise HTTPException(status_code=503, detail={
                "ready": False, "draining": state["draining"],
                "preload_error": state["preload_error"], "loaded": router.loaded})
        metrics.loaded.set(len(router.loaded))
        await batcher.refresh_metrics()
        return {"status": "ready", "loaded": router.loaded,
                "device": os.environ.get("LAYA_DEVICE") or "auto",
                "queue_depth": batcher.queue_depth}

    @app.post("/drain")
    def drain_endpoint(authorization: Optional[str] = Header(default=None)):
        """Operational drain hook for a Kubernetes preStop, before SIGTERM arrives.

        Stops intake and fails readiness so the endpoint controller stops routing new work,
        while in-flight and queued requests finish within LAYA_SHUTDOWN_GRACE_S. Idempotent.
        """
        _check_auth(authorization)
        batcher.begin_drain()
        state["draining"] = True
        log.info("drain requested via /drain")
        return {"status": "draining", "loaded": router.loaded}

    @app.get("/metrics")
    async def metrics_endpoint(authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        if not metrics.enabled:
            raise HTTPException(status_code=404, detail="metrics are disabled (LAYA_METRICS=0)")
        await batcher.refresh_metrics()    # publish current queue/state gauges before scraping
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        _declared_length_too_large(request)   # fast reject; _read_body_capped enforces every framing
        raw = await _read_body_capped(request)
        try:
            # Every parse failure a client can cause is a ValueError: JSONDecodeError for
            # malformed/empty/truncated bodies, UnicodeDecodeError for invalid UTF-8. A
            # broader catch would also swallow ClientDisconnect and Starlette's own
            # stream errors, reporting a transport or server fault as the client's.
            body = json.loads(raw)
        except ValueError:
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="request body must be an object with a 'questions' field")
        state_payload = body.get("state")
        questions = body["questions"]
        _check_request_limits(state_payload, questions)
        model = _resolve_model(body.get("model"))
        payload = {"state": state_payload, "questions": questions, "model": model}
        try:
            # Laya's result is already Jev-shaped: {model, answers, usage, routing}.
            # hs-jev decodes `answers` and `usage` and ignores the rest.
            return await _run_batched(payload, _client_key(request, authorization))
        except HTTPException:
            raise
        except ValueError as e:
            # Question validation errors name the question and what to fix: safe for clients.
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:  # noqa: BLE001 -- never leak paths/weights/OOM text to clients
            # The client gets a generic detail; the operator gets the traceback server-side, or
            # a 500 with no diagnostic (the original failure) is impossible to act on.
            log.exception("inference failed for /v1/systemone")
            raise HTTPException(status_code=500, detail="inference failed")

    return app


def main() -> None:
    import uvicorn

    from .serving import ServingConfig

    config = ServingConfig.from_env()
    # Without feeding the grace to uvicorn it waits for every in-flight connection before the
    # lifespan drain runs, so a backlog at SIGTERM outlives LAYA_SHUTDOWN_GRACE_S and the
    # orchestrator SIGKILLs the pod mid-drain. proxy_headers let request.client be the real
    # caller behind a trusted proxy, which is what per-IP admission and access logs key on.
    uvicorn.run(
        create_app(config=config),
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=_resolve_port(),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
        timeout_graceful_shutdown=int(config.shutdown_grace_s),
        proxy_headers=config.proxy_headers,
        forwarded_allow_ips=config.forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()
