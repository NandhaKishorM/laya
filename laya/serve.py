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
``LAYA_MAX_CONCURRENT`` cap on requests past auth at once; excess      16
                        gets 503 (see below)
======================  ============================================  =========

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
# HTTP-only amplification guard; the library keeps its head_max_len-aware budget.
MAX_CHOICE_OPTIONS = 100
MAX_SCORE_LEVELS = 32
MAX_TOTAL_OPTIONS = 512

# Cap on requests past auth at once. Each one can buffer up to MAX_BODY_BYTES
# before inference, so without a bound many concurrent near-cap requests OOM
# the worker even though every request is individually valid (#330).
DEFAULT_MAX_CONCURRENT = 16
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


def _resolve_max_concurrent() -> int:
    """Bound on requests past auth at once, from LAYA_MAX_CONCURRENT."""
    raw = os.environ.get("LAYA_MAX_CONCURRENT")
    if not raw:
        return DEFAULT_MAX_CONCURRENT
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_MAX_CONCURRENT
    return n if n > 0 else DEFAULT_MAX_CONCURRENT


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
    """Reject absent or oversized inference requests before tokenization (400/413)."""
    from fastapi import HTTPException

    # `serialize_state(None)` is `json.dumps(None)` == the four characters `null`, so a body with
    # no `state` key, or `"state": null`, was answered as a decision about the literal text
    # "null" -- HTTP 200, and at ~0.94 confidence here, byte-identical to sending `"state":
    # "null"`. Nothing downstream can tell that apart from a real string, so the check has to
    # happen before serialization. The repo's other two surfaces already require a state:
    # examples/server.py declares it as a required field and mcp/tools.py rejects an empty one.
    if state is None:
        raise HTTPException(status_code=400, detail="'state' is required")
    if not isinstance(questions, dict):
        raise HTTPException(status_code=400, detail="'questions' must be an object")
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=413,
                            detail="too many questions (%d > %d)" % (len(questions), MAX_QUESTIONS))

    total_options = 0
    for qid, question in questions.items():
        if not isinstance(question, dict):
            continue
        crit = question.get("criteria")
        qtype = question.get("type")
        if qtype == "choice" and isinstance(crit, (dict, list)):
            count = len(crit)
            total_options += count
            if count > MAX_CHOICE_OPTIONS:
                raise HTTPException(
                    status_code=413,
                    detail="too many choice options for %r (%d > %d)" % (qid, count, MAX_CHOICE_OPTIONS),
                )
        elif qtype == "score" and isinstance(crit, list):
            count = len(crit)
            total_options += count
            if count > MAX_SCORE_LEVELS:
                raise HTTPException(
                    status_code=413,
                    detail="too many score levels for %r (%d > %d)" % (qid, count, MAX_SCORE_LEVELS),
                )
    if total_options > MAX_TOTAL_OPTIONS:
        raise HTTPException(
            status_code=413,
            detail="too many answer options across questions (%d > %d)" % (total_options, MAX_TOTAL_OPTIONS),
        )

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
    router = Router(device=device, auto_task_detection=_env_bool("LAYA_AUTO_TASK", False))
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
    # Admission bound, same late-creation reason as the gate. Checked before any
    # body byte is read and held through inference, so the bodies buffered at
    # once stay bounded no matter how many clients connect (#330). The inference
    # gate is still joined only after the body is complete, so a slow client
    # holds an admission slot but never an inference slot.
    max_concurrent = _resolve_max_concurrent()
    admission: Optional[asyncio.Semaphore] = None

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
            "revisions": getattr(router, "loaded_revisions", {}),
            "device": os.environ.get("LAYA_DEVICE") or "auto",
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        nonlocal gate, admission
        _check_auth(authorization)
        if admission is None:
            admission = asyncio.Semaphore(max_concurrent)
        if admission.locked():
            # Non-blocking: excess load is refused rather than queued, so the
            # buffered bodies stay within the bound above.
            raise HTTPException(status_code=503, detail="server busy, try again later")
        await admission.acquire()
        try:
            return await _systemone_inner(request)
        finally:
            admission.release()

    async def _systemone_inner(request: Request):
        nonlocal gate
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
        try:
            # Laya's result is already Jev-shaped: {model, answers, usage, routing}.
            # hs-jev decodes `answers` and `usage` and ignores the rest.
            async with gate:
                loop = asyncio.get_running_loop()
                t0 = time.perf_counter()
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
