"""Optional HTTP adapter for the JavaScript SDK. Install with `pip install '.[server]'`."""
import argparse
import logging
import os
import secrets
import threading
from typing import Annotated, Dict, List, Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .router import Router, _repo_str

logger = logging.getLogger(__name__)


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    instructions: JsonValue


class ChoiceQuestion(QuestionBase):
    type: Literal["choice"]
    criteria: Union[Dict[str, JsonValue], List[str]] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def unique_labels(cls, value):
        if isinstance(value, list) and len(value) != len(set(value)):
            raise ValueError("choice labels must be unique")
        return value


class ScoreQuestion(QuestionBase):
    type: Literal["score"]
    criteria: List[JsonValue] = Field(min_length=1)


class NoulQuestion(QuestionBase):
    type: Literal["noul"]
    criteria: Optional[Dict[Literal["true", "false"], JsonValue]] = None


Question = Annotated[Union[ChoiceQuestion, ScoreQuestion, NoulQuestion], Field(discriminator="type")]
State = Union[str, Dict[str, JsonValue], List[JsonValue], None]


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    state: State
    questions: Dict[str, Question] = Field(default_factory=dict)
    model: Optional[str] = None
    task: Optional[str] = None
    lang: Optional[str] = None

    def routing_options(self):
        return {key: getattr(self, key) for key in ("model", "task", "lang")}

    def question_definitions(self):
        return {key: value.model_dump() for key, value in self.questions.items()}


class PredictRequest(RouteRequest):
    questions: Dict[str, Question] = Field(min_length=1)


def _routing_payload(decision):
    result = dict(decision)
    # The opt-in workflow branch currently returns a (repo, subfolder) tuple.
    # Keep the wire contract consistent with every other Router.route branch.
    if isinstance(result.get("repo"), (list, tuple)):
        result["repo"] = _repo_str(result["repo"])
    return result


def create_app(router=None, *, api_key=None, cors_origins=None):
    """Create an app around one shared Router; construction downloads no weights.

    Inject a configured/preloaded Router to choose checkpoints, devices and memory limits.
    Auth and CORS are opt-in. CLI configuration is separate from this factory.
    """
    runtime = router if router is not None else Router()
    app = FastAPI(title="Laya", version=__version__)
    app.state.router = runtime
    bearer = HTTPBearer(auto_error=False)
    # Agent's GPU fallback mutates the shared model/device. Serialize predictions
    # while keeping the ASGI event loop free (FastAPI runs sync endpoints in threads).
    inference_lock = threading.Lock()

    def authorize(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
        if api_key is not None and (
            credentials is None
            or not secrets.compare_digest(credentials.credentials.encode(), api_key.encode())
        ):
            raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo potentially sensitive state or credentials in validation responses.
        details = [{"location": list(err["loc"]), "message": err["msg"], "type": err["type"]}
                   for err in exc.errors()]
        return JSONResponse(status_code=422, content={"error": {
            "code": "validation_error", "message": "Invalid request", "details": details,
        }})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        code = {401: "unauthorized", 422: "validation_error"}.get(exc.status_code, "http_error")
        return JSONResponse(status_code=exc.status_code, headers=exc.headers,
                            content={"error": {"code": code, "message": str(exc.detail)}})

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logger.error("Laya request failed", exc_info=(type(exc), exc, exc.__traceback__))
        return JSONResponse(status_code=500, content={"error": {
            "code": "inference_error", "message": "Laya could not complete the request; check server logs",
        }})

    def route_request(body):
        try:
            return runtime.route(body.state, body.question_definitions(), **body.routing_options())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/health", dependencies=[Depends(authorize)])
    def health():
        # Liveness only: an empty list is normal before the first prediction.
        return {"status": "ok", "version": __version__, "loaded_models": runtime.loaded}

    @app.post("/v1/route", dependencies=[Depends(authorize)])
    def route(body: RouteRequest):
        return _routing_payload(route_request(body))

    @app.post("/v1/predict", dependencies=[Depends(authorize)])
    def predict(body: PredictRequest):
        # Validate routing before loading, so unknown model names are client errors.
        decision = route_request(body)
        with inference_lock:
            agent = runtime.load(decision["model"])
            try:
                result = agent.system_one(body.state, body.question_definitions())
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        return {**result, "routing": _routing_payload(decision)}

    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
    return app


def main():
    """Run a single inference worker, bound to localhost by default."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve Laya for JavaScript/TypeScript clients")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-loaded", type=int, default=1)
    parser.add_argument("--preload", nargs="+", choices=["english", "multilingual", "typed-decisions"])
    parser.add_argument("--auto-task-detection", action="store_true")
    args = parser.parse_args()
    router = Router(device=args.device, max_loaded=args.max_loaded,
                    auto_task_detection=args.auto_task_detection)
    if args.preload:
        router.preload(args.preload)
    origins = [s.strip() for s in os.environ.get("LAYA_CORS_ORIGINS", "").split(",") if s.strip()]
    app = create_app(router, api_key=os.environ.get("LAYA_API_KEY") or None, cors_origins=origins)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
