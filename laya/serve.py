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
======================  ============================================  =========

Imports of heavy dependencies (fastapi, uvicorn, torch via Router) are all
deferred into the functions that need them, so ``import laya.serve`` stays cheap
and touches no GPU -- which is what keeps the Nix ``pythonImportsCheck`` honest.
"""
import os
from typing import Any, Dict, Optional

# The three checkpoint names the router understands; used to decide whether a
# client's `model` field names a Laya checkpoint (honour it) or is some other
# Jev model id (ignore it and let the router auto-select).
_KNOWN_MODELS = {"english", "multilingual", "typed-decisions"}


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _resolve_model(model: Optional[str]) -> Optional[str]:
    """Map a client's `model` field onto a Laya checkpoint, or None to auto-route."""
    if not model:
        return None
    from .router import normalise_name

    # normalise_name raises ValueError on anything that is not a known checkpoint
    # or alias. A Jev client's `model` field (e.g. "jev-1") is expected to miss;
    # treat that as "no explicit checkpoint" and let the router auto-select.
    try:
        key = normalise_name(model)
    except Exception:
        return None
    return key if key in _KNOWN_MODELS else None


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
    from fastapi import FastAPI, Header, HTTPException, Request

    if router is None:
        router = build_router()
    api_key = os.environ.get("LAYA_API_KEY") or None

    app = FastAPI(
        title="laya-serve",
        summary="Laya System-1 decisions over the TypeSafe Jev /v1/systemone protocol",
    )

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is None:
            return
        if authorization != "Bearer " + api_key:
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
        _check_auth(authorization)
        body = await request.json()
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="request body must be an object with a 'questions' field")
        state = body.get("state")
        questions = body["questions"]
        model = _resolve_model(body.get("model"))
        try:
            # Laya's result is already Jev-shaped: {model, answers, usage, routing}.
            # hs-jev decodes `answers` and `usage` and ignores the rest.
            return router.predict(state, questions, model=model)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 -- surface model/tokenizer errors as 422
            raise HTTPException(status_code=422, detail=str(e))

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=int(os.environ.get("LAYA_PORT", "8000")),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
