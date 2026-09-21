"""
Laya REST API server — OpenAI-compatible decision inference endpoint.

Wraps ``laya.Router`` behind a FastAPI application so that Laya can be
called from any language (JavaScript, Go, Rust, cURL, …) and submitted
to OpenRouter as a drop-in replacement for TypeSafe Jev.

Quick start
-----------
Install extras::

    pip install "laya[server]"

Run::

    laya serve --port 8000                        # via CLI
    python -m laya.server                         # via module
    uvicorn laya.server:create_app --factory      # via uvicorn directly

Environment variables
---------------------
LAYA_MODEL          HuggingFace model ID or local path (default: convaiinnovations/laya)
LAYA_DEVICE         cuda | cpu | mps | auto (default: auto)
LAYA_HF_TOKEN       HuggingFace token for private model repos
LAYA_WORKERS        Number of inference threads in the thread-pool (default: 4)
LAYA_CORS_ORIGINS   Comma-separated allowed origins (default: *)
LAYA_LOG_LEVEL      Uvicorn log level: debug|info|warning|error (default: info)
LAYA_API_KEY        Optional bearer token — when set, all inference endpoints
                    require an ``Authorization: Bearer <key>`` header.
                    Also accepts TYPESAFE_API_KEY for drop-in Jev SDK compat.

Endpoints
---------
GET  /health                  Liveness + model info
GET  /v1/models               List available model(s)  [OpenRouter-compatible]
POST /v1/decide               Single-state decision     [OpenAI-compatible naming]
POST /v1/systemone            Drop-in alias — 100% wire-compatible with
                              TypeSafe Python + JavaScript SDKs, Vercel AI SDK
POST /v1/decide/batch         Multi-state batch decision (up to 256 states)

All responses mirror the OpenAI JSON envelope convention so that
existing OpenAI-compatible clients need only change the base URL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import laya
from laya.schemas import (
    AnyQuestion,
    BatchDecideRequest,
    BatchDecisionResponse,
    DecideRequest,
    DecisionResponse,
    ErrorDetail,
    ErrorResponse,
    UsageInfo,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("laya.server")

# ---------------------------------------------------------------------------
# Settings — read once from environment
# ---------------------------------------------------------------------------


class _Settings:
    model: str = os.environ.get("LAYA_MODEL", "english")
    device: Optional[str] = os.environ.get("LAYA_DEVICE") or None  # None → auto
    hf_token: Optional[str] = os.environ.get("LAYA_HF_TOKEN") or None
    workers: int = int(os.environ.get("LAYA_WORKERS", "4"))
    cors_origins: List[str] = [
        o.strip() for o in os.environ.get("LAYA_CORS_ORIGINS", "*").split(",") if o.strip()
    ]
    log_level: str = os.environ.get("LAYA_LOG_LEVEL", "info")
    # Accept LAYA_API_KEY or TYPESAFE_API_KEY for drop-in Jev SDK compat.
    api_key: Optional[str] = (
        os.environ.get("LAYA_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or None
    )


settings = _Settings()

# ---------------------------------------------------------------------------
# Global router holder
# ---------------------------------------------------------------------------

_router: Optional[laya.Router] = None


def _init_router() -> laya.Router:
    """
    Build and preload the Laya Router.

    This is a module-level function so tests can patch it easily:

        with patch("laya.server._init_router", return_value=mock_router):
            ...

    In tests, the mock bypasses model download entirely.
    """
    r = laya.Router(max_loaded=1, device=settings.device, token=settings.hf_token)
    r.preload([settings.model])
    return r


def _get_or_raise() -> laya.Router:
    """Return the loaded Router, or raise 503 if startup hasn't finished."""
    if _router is None:
        raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model is still loading.",
    )
    return _router


# ---------------------------------------------------------------------------
# Lifespan — load model once on startup, release on shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _router

    logger.info(
        "Loading Laya model '%s' on device '%s' …",
        settings.model,
        settings.device or "auto",
    )
    t0 = time.perf_counter()

    # Run the blocking model load in a thread-pool executor so the event loop stays free.
    loop = asyncio.get_running_loop()
    _router = await loop.run_in_executor(None, _init_router)

    elapsed = time.perf_counter() - t0
    logger.info("Model ready in %.2fs", elapsed)

    yield  # ⌛ server handles requests here

    logger.info("Shutting down — releasing model.")
    if _router is not None:
        _router.unload()
    _router = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Called by uvicorn when launched with ``--factory``::

        uvicorn laya.server:create_app --factory --port 8000
    """
    app = FastAPI(
        title="Laya Decision API",
        summary="Fast, non-autoregressive System 1 decision engine — open-source alternative to TypeSafe Jev.",
        version=laya.__version__,
        license_info={"name": "Apache 2.0", "url": "https://opensource.org/licenses/Apache-2.0"},
        contact={
            "name": "Convai Innovations",
            "url": "https://huggingface.co/convaiinnovations/laya",
        },
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def _add_timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Response-Time-Ms"] = f"{ms:.2f}"
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        body = ErrorResponse(
            error=ErrorDetail(
                code="internal_error",
                message="Internal server error",
            )
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(),
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    _register_routes(app)

    return app


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


async def _check_auth(credentials: Optional[HTTPAuthorizationCredentials] = None) -> None:
    """Optional bearer-token guard. Skipped when LAYA_API_KEY is not set."""
    if not settings.api_key:
        return
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error=ErrorDetail(code="unauthorized", message="Invalid or missing API key.")
            ).model_dump(),
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:

    # ── /health ───────────────────────────────────────────────────────────────

    @app.get(
        "/health",
        summary="Liveness check",
        tags=["System"],
        response_description="Server status and loaded model info.",
    )
    async def health():
        """
        Returns ``200 OK`` when the model is loaded and ready.
        Use this as the liveness/readiness probe in Kubernetes / Docker.
        """
        _get_or_raise()  # raises 500 if not loaded yet
        return {
            "status": "ok",
            "model": settings.model,
            "version": laya.__version__,
        }

    # ── /v1/models ────────────────────────────────────────────────────────────

    @app.get(
        "/v1/models",
        summary="List available models",
        tags=["Models"],
        response_description="OpenAI-style model list.",
    )
    async def list_models():
        """
        Returns the currently loaded model in an OpenAI-compatible ``/v1/models`` envelope,
        so existing OpenAI-SDK clients can enumerate models without modification.
        """
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "convai-innovations",
                    "capabilities": {
                        "question_types": ["choice", "score", "noul"],
                        "batch": True,
                    },
                }
            ],
        }

    # ── POST /v1/decide ───────────────────────────────────────────────────────

    @app.post(
        "/v1/decide",
        response_model=DecisionResponse,
        summary="Single-state decision",
        tags=["Inference"],
        responses={
            200: {"description": "Typed answers for every question in one forward pass."},
            422: {"model": ErrorResponse, "description": "Validation error in the request body."},
            500: {"model": ErrorResponse, "description": "Inference error."},
        },
    )
    async def decide(request: DecideRequest) -> DecisionResponse:
        """
        Evaluate all ``questions`` over ``state`` in a **single parallel forward pass**.

        - ``choice`` → best label + full probability distribution + confidence
        - ``score``  → expected ordinal level + distribution + confidence
        - ``noul``   → calibrated P(true) ∈ [0, 1]

        Latency is typically **33–38 ms on GPU** regardless of the number of questions.

        ### Example (cURL)
        ```bash
        curl -X POST http://localhost:8000/v1/decide \\
          -H "Content-Type: application/json" \\
          -d '{
            "state": {"subject": "Charged twice!", "body": "Please refund asap."},
            "questions": {
              "department": {
                "type": "choice",
                "instructions": "Which department should handle this email?",
                "criteria": {"billing": "invoices and refunds", "technical": "bugs"}
              },
              "is_urgent": {
                "type": "noul",
                "instructions": "Does the message communicate time pressure?"
              }
            }
          }'
        ```
        """
        router = _get_or_raise()
        raw_questions = _to_raw_questions(request.questions)

        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(
                None,
                lambda: router.predict(request.state, raw_questions, model=request.model),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorResponse(
                    error=ErrorDetail(code="invalid_question", message=str(exc))
                ).model_dump(),
            ) from exc

        return DecisionResponse.from_raw(raw)

    # ── POST /v1/decide/batch ─────────────────────────────────────────────────

    @app.post(
        "/v1/decide/batch",
        response_model=BatchDecisionResponse,
        summary="Multi-state batch decision",
        tags=["Inference"],
        responses={
            200: {"description": "One DecisionResponse per input state, in order."},
            422: {"model": ErrorResponse, "description": "Validation error."},
            500: {"model": ErrorResponse, "description": "Inference error."},
        },
    )
    async def decide_batch(request: BatchDecideRequest) -> BatchDecisionResponse:
        """
        Evaluate the **same question set** over multiple states concurrently.

        Each state is processed independently; results are returned in the same
        order as the input ``states`` list. Up to **256 states** per request.

        This is the fastest way to process a queue of tickets, emails, or log lines
        in bulk — all states run in the thread-pool concurrently.
        """
        router = _get_or_raise()
        raw_questions = _to_raw_questions(request.questions)

        loop = asyncio.get_running_loop()

        async def _infer_one(state) -> Dict[str, Any]:
            return await loop.run_in_executor(
                None,
                lambda: router.predict(state, raw_questions, model=request.model),
            )

        try:
            raws = await asyncio.gather(*[_infer_one(s) for s in request.states])
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorResponse(
                    error=ErrorDetail(code="invalid_question", message=str(exc))
                ).model_dump(),
            ) from exc

        results = [DecisionResponse.from_raw(r) for r in raws]
        total_tokens = sum(r.usage.input_tokens for r in results)

        return BatchDecisionResponse(
            model=settings.model,
            results=results,
            total_usage=UsageInfo(input_tokens=total_tokens, output_tokens=0),
        )

    # ── POST /v1/systemone ────────────────────────────────────────────────────
    # 100% wire-compatible alias for the TypeSafe Jev SDK.
    # The TypeSafe Python SDK calls POST /v1/systemone with the same body shape
    # as our /v1/decide, so this single alias makes laya-sdk and @typesafe-ai/sdk
    # work against this server with ZERO code changes.

    @app.post(
        "/v1/systemone",
        response_model=DecisionResponse,
        summary="TypeSafe Jev drop-in alias",
        tags=["Inference"],
        responses={
            200: {"description": "Identical to POST /v1/decide — Jev SDK compatible."},
            401: {"model": ErrorResponse, "description": "Invalid or missing API key."},
            422: {"model": ErrorResponse, "description": "Validation error."},
            500: {"model": ErrorResponse, "description": "Inference error."},
        },
    )
    async def system_one(request: DecideRequest) -> DecisionResponse:
        """
        **Drop-in replacement for the TypeSafe Jev API.**

        This endpoint is 100% wire-compatible with:
        - TypeSafe Python SDK (`client.system_one(...)`)
        - TypeSafe JavaScript SDK (`client.systemOne(...)`)
        - Vercel AI SDK TypeSafe integration
        - Any OpenRouter client pointing at this server

        The request and response shapes are identical to ``POST /v1/decide``.
        Simply point your existing SDK at this server's base URL — no other
        code changes required.

        ### Migration (Python)
        ```python
        # Before — paid TypeSafe cloud
        from typesafe import TypeSafeClient
        client = TypeSafeClient()  # uses api.typesafe.ai

        # After — self-hosted Laya (zero cost, full privacy)
        from typesafe import TypeSafeClient
        client = TypeSafeClient(base_url="http://localhost:8000")
        ```

        ### Migration (JavaScript)
        ```javascript
        // Before
        const client = new TypeSafeClient();

        // After
        const client = new TypeSafeClient({ baseURL: "http://localhost:8000" });
        ```
        """
        router = _get_or_raise()
        raw_questions = _to_raw_questions(request.questions)

        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(
                None,
                lambda: router.predict(request.state, raw_questions, model=request.model),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorResponse(
                    error=ErrorDetail(code="invalid_question", message=str(exc))
                ).model_dump(),
            ) from exc

        return DecisionResponse.from_raw(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_raw_questions(questions: Dict[str, AnyQuestion]) -> Dict[str, Dict]:
    """
    Convert validated Pydantic question models back to the raw dict format
    expected by ``Router.predict()`` → ``Agent.system_one()``.
    """
    out: Dict[str, Dict] = {}
    for qid, q in questions.items():
        d: Dict[str, Any] = {"type": q.type, "instructions": q.instructions}
        if hasattr(q, "criteria") and q.criteria is not None:
            if q.type == "noul":
                # NoulCriteria is a Pydantic model; convert to plain dict, drop None values.
                d["criteria"] = {k: v for k, v in q.criteria.model_dump().items() if v is not None}
            else:
                d["criteria"] = q.criteria
        out[qid] = d
    return out


# ---------------------------------------------------------------------------
# CLI entry-point:  python -m laya.server
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m laya.server",
        description="Launch the Laya Decision API server.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--model",
        default=settings.model,
        help=f"HuggingFace model ID or local path (default: {settings.model})",
    )
    parser.add_argument("--device", default=None, help="cuda | cpu | mps | auto (default: auto)")
    parser.add_argument("--workers", type=int, default=settings.workers, help="Thread-pool workers (default: 4)")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload for development.")
    parser.add_argument("--log-level", default=settings.log_level, help="Uvicorn log level (default: info)")
    args = parser.parse_args()

    # Propagate CLI args back to settings so the lifespan picks them up.
    settings.model = args.model
    if args.device:
        settings.device = args.device
    settings.workers = args.workers
    settings.log_level = args.log_level

    logging.basicConfig(level=args.log_level.upper())

    uvicorn.run(
        "laya.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    _main()
