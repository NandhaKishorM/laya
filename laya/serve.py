"""HTTP server exposing Laya over TypeSafe Jev's ``/v1/systemone`` wire protocol.

Laya's ``predict()`` output is already schema-compatible with the Jev decision
API -- ``choice`` / ``score`` / ``noul`` answers and a ``{input_tokens,
output_tokens}`` usage block -- so a client written against Jev (for example the
`hs-jev` Haskell client) can point its ``baseUrl`` at this server and keep
working unchanged. All this module adds is the HTTP surface Laya itself does not
ship: a ``POST /v1/systemone`` route, an optional bearer check, and a health
probe.

Configuration is entirely via environment variables so the same entry point
serves a laptop dev run and a systemd unit:

======================  ============================================  =========
env var                 meaning                                        default
======================  ============================================  =========
``LAYA_HOST``           bind address                                   0.0.0.0
``LAYA_PORT``           bind port                                      8000
``LAYA_DEVICE``         torch device for every checkpoint              (auto)
``LAYA_PRELOAD``        build the checkpoints at startup, not lazily   1
``LAYA_MODELS``         comma list to preload (english,multilingual,   (all)
                        typed-decisions); empty = every checkpoint
``LAYA_THREADS``        cap torch intra-op threads (CPU inference).    (torch
                        Keep <= physical cores; oversubscribing the     default)
                        logical/hyperthread count is a large regression.
``LAYA_AUTO_TASK``      auto-route to the typed-decisions checkpoint   0
``LAYA_API_KEY``        if set, require ``Authorization: Bearer <it>``  (none)
``LAYA_LOG_LEVEL``      uvicorn log level                              info
``LAYA_BACKEND``        inference backend for every checkpoint         (eager)
                        (auto, eager, compile, tilelang; laya.backends)
``LAYA_BATCH_WINDOW_MS``  collect concurrent requests for this long    2
                        and run them as one forward; 0 = one request
                        per forward, as before
``LAYA_BATCH_MAX``      most requests merged into one forward          32
======================  ============================================  =========

Dynamic batching: a forward pass at Laya's sizes is launch-bound, so eight requests in one
``predict_batch`` cost about what one costs (see ``laya.backends``). Requests that arrive
within ``LAYA_BATCH_WINDOW_MS`` of each other -- or while a forward is already running --
are merged into one ``Router.predict_batch`` call, per checkpoint and question schema, and
each gets its own result and its own error: a request that fails validation answers 422
without failing the rest of its batch.

Imports of heavy dependencies (fastapi, uvicorn, torch via Router) are all
deferred into the functions that need them, so ``import laya.serve`` stays cheap
and touches no GPU -- which is what keeps the Nix ``pythonImportsCheck`` honest.
"""
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

# A failed inference is reported to the client as a fixed 500 so nothing about paths,
# weights or memory state leaks, which leaves the server log as the only place the
# actual cause can appear. Uvicorn configures the root logger, so a module logger
# propagates there without this module setting up any handlers.
_log = logging.getLogger("laya.serve")

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
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_number(name: str, default: float, minimum: float = 0.0) -> float:
    """A non-negative number from the environment; anything unparsable is the default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, value)


DEFAULT_BATCH_WINDOW_MS = 2.0
DEFAULT_BATCH_MAX = 32


class RequestBatcher:
    """Merge concurrent ``/v1/systemone`` requests into one ``Router.predict_batch`` call.

    Every request is appended to ``pending`` and awaits its own future. The first arrival
    schedules a flush: it waits ``window_s`` (so requests arriving together join), then takes
    the queue -- at most ``max_batch`` at a time -- and runs it on the inference executor
    under the app's gate. Requests that arrive while a batch is running queue up for the
    next one, so a busy server batches however short the window is.

    The window only opens during a burst: when the request that starts a flush arrived
    within ``window_s`` of the previous one. A single client sending requests back to back
    is spaced by its own round trip, so it is served at once and never pays the window; a
    burst of clients firing together is ``window_s`` apart at most and is collected.

    Per-request semantics are kept: results are handed back by position, and if the batched
    call raises (one request's questions failed validation, say), every request in it is
    re-run on its own so the failure reaches only its owner and the rest still answer. A
    router without ``predict_batch`` (an injected fake) runs requests one at a time too.
    """

    def __init__(self, router: Any, pool: Any, window_s: float, max_batch: int):
        self.router = router
        self.pool = pool
        self.window_s = max(0.0, float(window_s))
        self.max_batch = max(1, int(max_batch))
        self.pending: list = []
        self._flush = None
        self.gate = None                       # created on the running loop, see create_app
        self.batches: list = []                # sizes of the batches run, for tests and metrics
        self._last_arrival = float("-inf")
        self._burst = False

    async def submit(self, state: Any, questions: Any, model: Optional[str]) -> Dict[str, Any]:
        import asyncio

        loop = asyncio.get_running_loop()
        if self.gate is None:
            self.gate = asyncio.Lock()
        future = loop.create_future()
        now = time.perf_counter()
        if not self.pending:
            self._burst = (now - self._last_arrival) < self.window_s
        self._last_arrival = now
        self.pending.append((state, questions, model, future))
        if self._flush is None:
            self._flush = loop.create_task(self._flush_later())
        return await future

    async def _flush_later(self):
        import asyncio

        try:
            if self.window_s > 0 and self._burst:
                await asyncio.sleep(self.window_s)
            while self.pending:
                items, self.pending = self.pending[:self.max_batch], self.pending[self.max_batch:]
                loop = asyncio.get_running_loop()
                async with self.gate:
                    outcomes = await loop.run_in_executor(self.pool, self._run, items)
                for (_, _, _, future), (ok, value) in zip(items, outcomes):
                    if future.cancelled():
                        continue
                    if ok:
                        future.set_result(value)
                    else:
                        future.set_exception(value)
        finally:
            self._flush = None
            if self.pending:                   # arrivals during the last run start the next flush
                self._flush = asyncio.get_running_loop().create_task(self._flush_later())

    def _run(self, items) -> list:
        """On the executor: one predict_batch, falling back to per-request predict on failure."""
        self.batches.append(len(items))
        batched = getattr(self.router, "predict_batch", None)
        if len(items) > 1 and callable(batched):
            requests = []
            for state, questions, model, _ in items:
                req = {"state": state, "questions": questions}
                if model is not None:
                    req["model"] = model
                requests.append(req)
            try:
                results = batched(requests, batch_size=self.max_batch)
                if len(results) == len(items):
                    return [(True, r) for r in results]
                _log.error("predict_batch returned %d results for %d requests; re-running singly",
                           len(results), len(items))
            except Exception:  # noqa: BLE001 -- isolate the failure to the request that caused it
                pass
        outcomes = []
        for state, questions, model, _ in items:
            try:
                outcomes.append((True, self.router.predict(state, questions, model=model)))
            except Exception as e:  # noqa: BLE001 -- every request gets its own answer or error
                outcomes.append((False, e))
        return outcomes


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


def build_router():
    """Build a Router from the environment, preloading unless told otherwise."""
    from .router import Router

    _apply_thread_limit()
    device = os.environ.get("LAYA_DEVICE") or None
    models_env = os.environ.get("LAYA_MODELS", "").strip()
    preload_names = [m.strip() for m in models_env.split(",") if m.strip()] or None
    router = Router(device=device, auto_task_detection=_env_bool("LAYA_AUTO_TASK", False),
                    backend=os.environ.get("LAYA_BACKEND") or None)
    if _env_bool("LAYA_PRELOAD", True):
        router.preload(preload_names)
    return router


def create_app(router: Optional[Any] = None):
    """Build the FastAPI app. Pass a Router to inject one (tests); otherwise one
    is built from the environment (and preloaded) at app-creation time."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, Header, HTTPException, Request

    if router is None:
        router = build_router()
    api_key = os.environ.get("LAYA_API_KEY") or None

    # Inference is synchronous torch, and a CPU call takes hundreds of milliseconds to
    # seconds, so it must not run on the event loop: one request would stall every
    # other client, `GET /health` included. One worker, because one forward pass at a
    # time is what a single CPU or GPU Agent wants (the Router already guards checkpoint
    # lifecycle, and leaves `Agent.system_one` unguarded deliberately so concurrent
    # predictions can share a checkpoint -- a GPU-shaped choice this endpoint does not
    # rely on). `loop.run_in_executor` is the API the issue asked for.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")
    # Created on first request, not here: an `asyncio.Lock` binds to the loop that is
    # running when it is first awaited, and `create_app` may be called before that loop
    # exists (module scope, TestClient startup, a preload script).
    gate: Optional[asyncio.Lock] = None
    # Concurrent requests share a forward pass (see RequestBatcher); LAYA_BATCH_WINDOW_MS=0
    # keeps one request per forward.
    window_ms = _env_number("LAYA_BATCH_WINDOW_MS", DEFAULT_BATCH_WINDOW_MS)
    batcher = RequestBatcher(router, pool, window_ms / 1000.0,
                             int(_env_number("LAYA_BATCH_MAX", DEFAULT_BATCH_MAX, minimum=1)))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            # TestClient, embedded ASGI apps, and process supervisors all need
            # the executor to drain when the app stops.
            pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="laya-serve",
        summary="Laya System-1 decisions over the TypeSafe Jev /v1/systemone protocol",
        lifespan=lifespan,
    )

    # Compared as bytes, not str. `hmac.compare_digest` raises TypeError when a str
    # operand holds a non-ASCII character, and Starlette decodes request headers as
    # latin-1 -- so `Authorization: Bearer s\xe9cret`, which is legal on the wire,
    # made the comparison itself raise. That surfaced as HTTP 500 plus a traceback
    # in the log, reachable by any unauthenticated client with one byte. Encoding
    # both sides first keeps the comparison constant-time and total: every header a
    # client can send now answers 401.
    expected_auth = ("Bearer " + api_key).encode("utf-8", "surrogateescape") if api_key else b""

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is None:
            return
        supplied = (authorization or "").encode("utf-8", "surrogateescape")
        if not hmac.compare_digest(supplied, expected_auth):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "loaded": router.loaded,
            "device": os.environ.get("LAYA_DEVICE") or "auto",
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        nonlocal gate
        _check_auth(authorization)
        # A declared length over the cap is rejected before anything is read; the
        # streaming cap below is what actually enforces it, for bodies that declare
        # no length or understate it.
        if request.headers.get("content-length"):
            try:
                if int(request.headers["content-length"]) > MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="request body too large")
            except ValueError:
                pass
        raw = await _read_body_capped(request)
        try:
            # A client can cause ValueError (JSONDecodeError for malformed/empty/truncated
            # bodies, UnicodeDecodeError for invalid UTF-8) or RecursionError (deeply nested
            # arrays/objects). A broader catch would also swallow ClientDisconnect and
            # Starlette's own stream errors, reporting a transport or server fault as the client's.
            body = json.loads(raw)
        except (ValueError, RecursionError):
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="request body must be an object with a 'questions' field")
        state = body.get("state")
        questions = body["questions"]
        _check_request_limits(state, questions)
        model = _resolve_model(body.get("model"))
        if gate is None:
            gate = asyncio.Lock()
            batcher.gate = gate
        try:
            # Laya's result is already Jev-shaped: {model, answers, usage, routing}.
            # hs-jev decodes `answers` and `usage` and ignores the rest.
            t0 = time.perf_counter()
            if window_ms > 0:
                # The batcher takes the gate itself; a request's inference time then covers
                # the window it waited and the forward it shared.
                result = await batcher.submit(state, questions, model)
            else:
                async with gate:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        pool, lambda: router.predict(state, questions, model=model))
            infer_ms = (time.perf_counter() - t0) * 1000.0
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=result,
                headers={
                    "Server-Timing": f"inference;dur={infer_ms:.2f}",
                    "X-Inference-Time-Ms": f"{infer_ms:.2f}"
                }
            )
        except HTTPException:
            raise
        except ValueError as e:
            # Question validation errors name the question and what to fix: safe for clients.
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:  # noqa: BLE001 -- never leak paths/weights/OOM text to clients
            # The client still learns nothing, but the operator gets the traceback. Without
            # this the container logs show only the 500, so a deterministic failure such as a
            # missing C compiler for triton's JIT (#365) is invisible from the running server
            # and has to be reproduced in-process to be diagnosed at all.
            _log.exception("inference failed for model=%s", model)
            raise HTTPException(status_code=500, detail="inference failed")

    app.state.batcher = batcher
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=_resolve_port(),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
