"""Server-shim tests: verify the Jev /v1/systemone surface without a GPU.

A fake Router is injected so nothing loads a checkpoint; we only assert that the
HTTP layer maps requests/responses and enforces auth as hs-jev expects.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from laya.serve import _apply_thread_limit, _env_bool, _resolve_model, create_app  # noqa: E402


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
