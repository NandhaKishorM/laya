"""Server-shim tests: verify the Jev /v1/systemone surface without a GPU.

A fake Router is injected so nothing loads a checkpoint; we only assert that the
HTTP layer maps requests/responses and enforces auth as hs-jev expects.
"""
import json
import logging

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from laya.serve import (  # noqa: E402
    MAX_BODY_BYTES,
    _apply_thread_limit,
    _env_bool,
    _resolve_model,
    create_app,
)
from laya.serving import ServingConfig  # noqa: E402


class FakeRouter:
    """Records the last predict() call and returns a Jev-shaped payload."""

    loaded = ["english"]

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        return {
            "model": "laya-rl-agent",
            "answers": {
                "dept": {"type": "choice", "choice": "billing",
                         "probabilities": {"billing": 0.94, "tech": 0.06}, "confidence": 0.94},
            },
            "usage": {"input_tokens": 42, "output_tokens": 0},
            "routing": {"model": "english", "reason": "English Latin text"},
        }

    def predict_batch(self, requests, batch_size=None):
        return [self.predict(r["state"], r["questions"], model=r.get("model")) for r in requests]


def _client(monkeypatch, api_key=None):
    if api_key is None:
        monkeypatch.delenv("LAYA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("LAYA_API_KEY", api_key)
    fake = FakeRouter()
    return TestClient(create_app(router=fake)), fake


REQ = {
    "model": "jev-1",  # a non-Laya model id -> should be ignored, router auto-routes
    "state": {"body": "billed twice, refund please"},
    "questions": {"dept": {"type": "choice", "instructions": "which team?",
                           "criteria": {"billing": None, "tech": None}}},
}


def test_predict_passthrough_shape(monkeypatch):
    client, fake = _client(monkeypatch)
    r = client.post("/v1/systemone", json=REQ)
    assert r.status_code == 200
    body = r.json()
    # exactly the fields hs-jev's Response/Usage decoders require
    assert set(["answers", "usage"]).issubset(body)
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert body["answers"]["dept"]["choice"] == "billing"
    # unknown model id was dropped -> router asked to auto-route
    assert fake.calls[0]["model"] is None


def test_known_model_is_honoured(monkeypatch):
    client, fake = _client(monkeypatch)
    client.post("/v1/systemone", json={**REQ, "model": "multilingual"})
    assert fake.calls[0]["model"] == "multilingual"


@pytest.mark.parametrize(("model", "expected"), [
    ("convaiinnovations/laya-multilingual", "multilingual"),
    ("convaiinnovations/laya-typed-decisions", "typed-decisions"),
])
def test_published_model_id_is_honoured(monkeypatch, model, expected):
    client, fake = _client(monkeypatch)
    client.post("/v1/systemone", json={**REQ, "model": model})
    assert fake.calls[0]["model"] == expected


def test_missing_questions_is_400(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/systemone", json={"state": "hi"})
    assert r.status_code == 400


@pytest.mark.parametrize("payload", [
    b"not json",
    b"",                    # empty body
    b"\xff\xfe\x00bad",     # invalid UTF-8
    b'{"questions": ',      # truncated
    pytest.param(b"[" * 100000, id="deeply-nested"),  # raises RecursionError, not ValueError
])
def test_malformed_json_body_is_400(monkeypatch, payload):
    """A body that isn't valid JSON must not fall through to an unstyled 500."""
    client, _ = _client(monkeypatch)
    r = client.post("/v1/systemone", content=payload,
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    # pin which 400: the other branch below also answers 400, so the status alone
    # would not notice the parse guard disappearing.
    assert r.json()["detail"] == "request body must be valid JSON"


@pytest.mark.parametrize("payload", [b"[1,2,3]", b'"hello"', b"null"])
def test_json_that_is_not_an_object_is_400(monkeypatch, payload):
    """Valid JSON that isn't an object is the other 400, not a parse failure."""
    client, _ = _client(monkeypatch)
    r = client.post("/v1/systemone", content=payload,
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert "questions" in r.json()["detail"]


def test_auth_required_when_key_set(monkeypatch):
    client, _ = _client(monkeypatch, api_key="s3cret")
    assert client.post("/v1/systemone", json=REQ).status_code == 401
    ok = client.post("/v1/systemone", json=REQ, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_auth_rejects_a_non_ascii_header(monkeypatch):
    """A hostile Authorization header must answer 401, not raise.

    `hmac.compare_digest` raises TypeError when a str operand holds a non-ASCII
    character, and Starlette decodes request headers as latin-1. So
    `Authorization: Bearer s\xe9cret` -- legal on the wire -- used to make the
    comparison itself raise, which FastAPI turned into HTTP 500 with a traceback
    in the log, reachable by any unauthenticated client.
    """
    client, _ = _client(monkeypatch, api_key="s3cret")
    for header in (
        "Bearer s\u00e9cret".encode("latin-1"),   # non-ASCII inside the token
        "B\u00ebarer s3cret".encode("latin-1"),   # non-ASCII in the scheme
        b"Bearer \xff\xfe",                      # bytes that are not valid UTF-8
    ):
        r = client.post("/v1/systemone", json=REQ, headers={"Authorization": header})
        assert r.status_code == 401, (header, r.status_code)


def _chunked(payload: bytes):
    """Send `payload` with no Content-Length, i.e. Transfer-Encoding: chunked."""
    yield payload


def test_body_limit_holds_without_content_length(monkeypatch):
    """The body cap must not depend on the client declaring its length.

    Content-Length is a value the client chooses and chunked transfer-encoding
    omits it entirely (HTTP/2 and /3 have no such header), so checking only the
    header let a request of any size be read into memory in full. The state and
    question-count guards do not cover this: state stays tiny and there is one
    question -- the payload is large because the question's own text is.
    """
    client, fake = _client(monkeypatch)
    oversized = json.dumps({
        "state": "ok",
        "questions": {"a": {"type": "choice",
                            "instructions": "A" * (MAX_BODY_BYTES + 1024),
                            "criteria": {"y": None, "z": None}}},
    }).encode()
    assert len(oversized) > MAX_BODY_BYTES

    declared = client.post("/v1/systemone", content=oversized,
                           headers={"content-type": "application/json"})
    assert declared.status_code == 413

    undeclared = client.post("/v1/systemone", content=_chunked(oversized),
                             headers={"content-type": "application/json"})
    assert undeclared.status_code == 413
    # And it was refused before reaching inference, which is the point: the pool
    # is one worker wide, so a body that gets that far blocks every other client.
    assert fake.calls == []


def test_a_request_within_the_limit_still_works_without_content_length(monkeypatch):
    """The cap must not break legitimate chunked clients."""
    client, fake = _client(monkeypatch)
    body = json.dumps(REQ).encode()
    r = client.post("/v1/systemone", content=_chunked(body),
                    headers={"content-type": "application/json"})
    assert r.status_code == 200
    assert len(fake.calls) == 1


def test_body_read_preserves_parse_error_codes(monkeypatch):
    """Reading the body ourselves must keep 400 for anything unparseable."""
    client, _ = _client(monkeypatch)
    for payload in (b"", b"{not json", b'{"questions":{},"state":"\xff\xfe"}'):
        r = client.post("/v1/systemone", content=payload,
                        headers={"content-type": "application/json"})
        assert r.status_code == 400, (payload, r.status_code)


def test_health(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_helpers():
    assert _resolve_model("multilingual") == "multilingual"
    assert _resolve_model("convaiinnovations/laya-multilingual") == "multilingual"
    assert _resolve_model("convaiinnovations/laya-typed-decisions") == "typed-decisions"
    assert _resolve_model("convaiinnovations/laya") is None
    assert _resolve_model("jev-1") is None
    assert _resolve_model(None) is None
    import os
    os.environ.pop("X_FLAG", None)
    assert _env_bool("X_FLAG", True) is True


def test_thread_limit(monkeypatch):
    monkeypatch.delenv("LAYA_THREADS", raising=False)
    assert _apply_thread_limit() is None  # unset -> no-op, no torch import
    for bad in ("0", "-4", "abc", ""):
        monkeypatch.setenv("LAYA_THREADS", bad)
        assert _apply_thread_limit() is None
    monkeypatch.setenv("LAYA_THREADS", "8")
    assert _apply_thread_limit() == 8
    import torch
    assert torch.get_num_threads() == 8


# The endpoint is `async def` and inference is synchronous torch, which on CPU takes
# hundreds of milliseconds to seconds. Calling it from the coroutine puts that work on
# the event loop, so every other client -- `GET /health` included -- waits for it.
# Driving the app directly on a loop (`httpx.ASGITransport`) makes the difference
# observable: offloaded work runs on a worker thread, inline work runs on the loop's own
# `MainThread`. `TestClient` cannot see this, because it runs the loop in a portal thread
# and hands each call its own, so a blocking endpoint still looks concurrent there.
class SlowRouter(FakeRouter):
    """Sleeps like a CPU forward pass and records the thread it ran on."""

    def __init__(self, seconds=0.25):
        super().__init__()
        self.seconds = seconds
        self.threads = []

    def predict(self, state, questions, model=None):
        import threading
        import time
        self.threads.append(threading.current_thread().name)
        time.sleep(self.seconds)
        return super().predict(state, questions, model=model)


def test_inference_runs_off_the_event_loop(monkeypatch):
    import asyncio
    import threading

    import httpx

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    fake = FakeRouter()
    seen = []
    real_predict = fake.predict

    def recording_predict(state, questions, model=None):
        seen.append(threading.current_thread().name)
        return real_predict(state, questions, model=model)

    fake.predict = recording_predict
    app = create_app(router=fake)

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.post("/v1/systemone", json=REQ)

    response = asyncio.run(drive())

    assert response.status_code == 200, response.text
    assert seen, "predict was never called"
    assert "MainThread" not in seen, (
        "predict ran on the event loop thread: %s -- one request would stall every "
        "other client, including GET /health" % seen)


def test_health_stays_available_during_inference(monkeypatch):
    """A request in flight must not stop the app answering `GET /health`."""
    import asyncio

    import httpx

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    fake = SlowRouter(seconds=0.25)
    app = create_app(router=fake)
    seen = {}

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            await client.post("/v1/systemone", json=REQ)          # warm up

            async def slow_request():
                seen["slow"] = (await client.post("/v1/systemone", json=REQ)).status_code

            async def health():
                r = await client.get("/health")
                seen["health"] = r.status_code
                seen["payload"] = r.json()

            await asyncio.gather(slow_request(), health())

    asyncio.run(drive())

    assert seen["slow"] == 200
    assert seen["health"] == 200 and seen["payload"]["status"] == "ok"
    assert fake.threads and "MainThread" not in fake.threads, fake.threads


class ExplodingRouter:
    """Fails the way a container missing triton's C compiler does (#365).

    The message is the shape a real failure takes: it names a path and a tool, which is
    exactly what must not reach the client and exactly what the operator needs.
    """

    loaded = ["multilingual"]

    def __init__(self, message):
        self.message = message

    def predict(self, state, questions, model=None):
        raise RuntimeError(self.message)


def test_inference_failure_is_logged_and_not_leaked(monkeypatch, caplog):
    """A failed inference still returns a bare 500, but the cause reaches the log.

    The client-facing message is deliberately fixed, so the server log is the only place
    the real exception can appear. Before this, the log carried nothing at all: a
    deterministic failure was visible only as `POST /v1/systemone HTTP/1.1" 500`, and the
    cause had to be reproduced in-process to be found.
    """
    secret = ("Failed to find C compiler. Please specify via CC environment variable "
              "or set triton.knobs.build.impl (/opt/venv/lib/python3.11/site-packages/triton)")
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=ExplodingRouter(secret)), raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="laya.serve"):
        response = client.post("/v1/systemone", json=REQ)

    assert response.status_code == 500
    assert response.json() == {"detail": "inference failed"}
    for leaked in ("C compiler", "triton", "/opt/venv", "site-packages"):
        assert leaked not in response.text, response.text

    logged = "\n".join(r.getMessage() if isinstance(r.getMessage(), str) else str(r.msg)
                       for r in caplog.records)
    assert any(r.levelno == logging.ERROR for r in caplog.records), caplog.records
    # the traceback has to be in the record, not only the summary line
    assert any(r.exc_info for r in caplog.records), "no exc_info on the failure record"
    assert "inference failed" in logged


def test_validation_errors_are_not_logged_as_failures(monkeypatch, caplog):
    """A 422 is the caller's mistake and must not be logged as a server error.

    `ValueError` from the router is mapped to 422 with its message intact, because those
    messages name the question and what to fix. Only the bare `except Exception` below it
    reports a server fault, so only that branch logs.
    """
    class RejectingRouter:
        loaded = ["english"]

        def predict(self, state, questions, model=None):
            raise ValueError("question 'q': a choice question needs at least one criterion")

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=RejectingRouter()), raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="laya.serve"):
        response = client.post("/v1/systemone", json=REQ)

    assert response.status_code == 422, response.text
    assert "at least one criterion" in response.text, response.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], caplog.records


def test_inference_timing_headers():
    """POST /v1/systemone returns Server-Timing and X-Inference-Time-Ms headers."""
    router = FakeRouter()
    client = TestClient(create_app(router=router))
    res = client.post("/v1/systemone", json={
        "state": "test timing",
        "questions": {"dept": {"type": "choice", "instructions": "which?", "criteria": {"billing": "invoices"}}}
    })
    assert res.status_code == 200
    assert "Server-Timing" in res.headers
    assert res.headers["Server-Timing"].startswith("inference;dur=")
    assert "X-Inference-Time-Ms" in res.headers
    dur = float(res.headers["X-Inference-Time-Ms"])
    assert dur >= 0.0


# ------------------------------------------------------------------ batching
# Batching is opt-in: with the default ServingConfig each request is dispatched
# immediately. These tests raise the window / batch size so concurrent requests share one
# Router.predict_batch forward pass. `httpx.ASGITransport` is used because TestClient
# serializes requests through a portal, so nothing is ever concurrently in flight.

_OK = {
    "model": "laya-rl-agent",
    "answers": {"dept": {"type": "choice", "choice": "billing",
                         "probabilities": {"billing": 0.94, "tech": 0.06}, "confidence": 0.94}},
    "usage": {"input_tokens": 42, "output_tokens": 0},
    "routing": {"model": "english", "reason": "English Latin text"},
}


class BatchRouter:
    """Counts batch forward passes and answers each request; a 'poison' state fails the batch."""

    loaded = ["english"]

    def __init__(self, delay=0.0):
        self.batch_calls = 0
        self.single_calls = 0
        self.seen_models = []
        self._delay = delay

    def predict_batch(self, requests, batch_size=None):
        import time
        self.batch_calls += 1
        if self._delay:
            time.sleep(self._delay)
        self.seen_models.extend(r.get("model") for r in requests)
        if any(r.get("state") == "poison" for r in requests):
            raise RuntimeError("batch poisoned")
        return [_OK for _ in requests]

    def predict(self, state, questions, model=None, **kwargs):
        self.single_calls += 1
        if state == "poison":
            raise ValueError("question 'q': bad state")
        return _OK


def _batching_client(fake, **config):
    """An httpx client driving the app directly, so requests can be concurrent."""
    import httpx

    app = create_app(router=fake, config=ServingConfig(**config))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def test_concurrent_requests_are_batched():
    import asyncio

    fake = BatchRouter(delay=0.05)

    async def scenario():
        async with _batching_client(fake, batch_window_ms=30, batch_max=16) as client:
            return await asyncio.gather(*[client.post("/v1/systemone", json=REQ) for _ in range(8)])

    responses = asyncio.run(scenario())
    assert all(r.status_code == 200 for r in responses)
    assert fake.batch_calls == 1, "8 concurrent requests should share one forward pass"
    assert fake.single_calls == 0


def test_batch_max_caps_the_batch():
    import asyncio

    fake = BatchRouter(delay=0.02)

    async def scenario():
        async with _batching_client(fake, batch_window_ms=50, batch_max=2) as client:
            return await asyncio.gather(*[client.post("/v1/systemone", json=REQ) for _ in range(6)])

    responses = asyncio.run(scenario())
    assert all(r.status_code == 200 for r in responses)
    assert fake.batch_calls >= 3, "batch_max=2 cannot serve 6 requests in one pass"


def test_zero_window_dispatches_immediately():
    import asyncio

    fake = BatchRouter()

    async def scenario():
        async with _batching_client(fake, batch_window_ms=0, batch_max=16) as client:
            return await client.post("/v1/systemone", json=REQ)

    assert asyncio.run(scenario()).status_code == 200
    assert fake.batch_calls == 1


def test_full_queue_returns_503_with_retry_after():
    import asyncio

    fake = BatchRouter(delay=0.5)   # keep the worker busy so the bounded queue fills

    async def scenario():
        async with _batching_client(fake, batch_window_ms=500, batch_max=16,
                                    queue_max=1, request_timeout_s=5) as client:
            return await asyncio.gather(*[client.post("/v1/systemone", json=REQ) for _ in range(6)])

    responses = asyncio.run(scenario())
    rejected = [r for r in responses if r.status_code == 503]
    assert rejected, "a full queue must reject with 503"
    assert all("retry-after" in r.headers for r in rejected)


def test_request_timeout_returns_504():
    import asyncio

    fake = BatchRouter(delay=0.3)

    async def scenario():
        async with _batching_client(fake, batch_window_ms=0, request_timeout_s=0.01) as client:
            return await client.post("/v1/systemone", json=REQ)

    assert asyncio.run(scenario()).status_code == 504


def test_batch_failure_falls_back_per_request():
    import asyncio

    fake = BatchRouter()

    async def scenario():
        async with _batching_client(fake, batch_window_ms=30, batch_max=16) as client:
            return await asyncio.gather(
                client.post("/v1/systemone", json=REQ),
                client.post("/v1/systemone", json={**REQ, "state": "poison"}))

    good, bad = asyncio.run(scenario())
    assert good.status_code == 200
    # the poison request fails alone; the batch fell back to per-request calls
    assert bad.status_code == 422
    assert fake.single_calls == 2


def test_model_override_survives_batching():
    import asyncio

    fake = BatchRouter()

    async def scenario():
        async with _batching_client(fake, batch_window_ms=30, batch_max=16) as client:
            return await client.post("/v1/systemone", json={**REQ, "model": "multilingual"})

    assert asyncio.run(scenario()).status_code == 200
    assert fake.seen_models == ["multilingual"]
