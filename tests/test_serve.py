"""Server-shim tests: verify the Jev /v1/systemone surface without a GPU.

A fake Router is injected so nothing loads a checkpoint; we only assert that the
HTTP layer maps requests/responses and enforces auth as hs-jev expects.
"""
import json
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from laya.serve import (  # noqa: E402
    DEFAULT_MAX_TOKEN_BUDGET,
    MAX_BODY_BYTES,
    _apply_thread_limit,
    _env_bool,
    _resolve_max_token_budget,
    _resolve_max_loaded,
    _resolve_model,
    create_app,
)

# Read from the router rather than copied here: the workload below has to reach the
# typed-decisions checkpoint the same way a request does.
from laya.router import _TYPED_DECISION_WORKFLOWS  # noqa: E402


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


class BudgetRouter(FakeRouter):
    """A router whose predict() takes token-budget keywords and records them."""

    def predict(self, state, questions, model=None, **kwargs):
        out = super().predict(state, questions, model=model)
        self.calls[-1].update(kwargs)
        return out


def _client(monkeypatch, api_key=None):
    if api_key is None:
        monkeypatch.delenv("LAYA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("LAYA_API_KEY", api_key)
    fake = FakeRouter()
    return TestClient(create_app(router=fake)), fake


def _budget_client(monkeypatch, api_key=None):
    if api_key is None:
        monkeypatch.delenv("LAYA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("LAYA_API_KEY", api_key)
    fake = BudgetRouter()
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


class DeviceRouter:
    """A Router with resident checkpoints whose real devices are known.

    `Agent.device` is a `torch.device`; `laya.mcp.device.agent_device` also accepts the
    plain string, which keeps this stub free of torch and of checkpoint weights.
    """

    def __init__(self, **devices):
        self._agents = {name: SimpleNamespace(device=device)
                        for name, device in devices.items()}
        self.loaded = list(devices)

    def predict(self, state, questions, model=None):
        return {"model": "laya-rl-agent",
                "answers": {"dept": {"type": "choice", "choice": "billing",
                                     "probabilities": {"billing": 1.0}, "confidence": 1.0}},
                "usage": {"input_tokens": 1, "output_tokens": 0},
                "routing": {"model": (self.loaded or ["english"])[0]}}


def test_health_reports_where_inference_actually_runs(monkeypatch):
    """`LAYA_DEVICE` is a request, not a fact: the Agent falls back to CPU silently."""
    monkeypatch.setenv("LAYA_DEVICE", "cuda")
    fake = DeviceRouter(english="cpu")            # asked for cuda, ended up on cpu
    body = TestClient(create_app(router=fake)).get("/health").json()
    assert body["device"] == "cpu", body
    assert body["device_is_preference"] is False, body
    assert body["checkpoint_devices"] == {"english": "cpu"}, body


def test_health_names_each_checkpoint_device(monkeypatch):
    monkeypatch.setenv("LAYA_DEVICE", "cuda")
    fake = DeviceRouter(english="cpu", multilingual="cuda")
    body = TestClient(create_app(router=fake)).get("/health").json()
    assert body["checkpoint_devices"] == {"english": "cpu", "multilingual": "cuda"}, body
    # The top-level answer is the first resident one, exactly as `laya_status` reports it.
    assert body["device"] == body["checkpoint_devices"][body["loaded"][0]], body


def test_health_without_a_resident_checkpoint_flags_the_preference(monkeypatch):
    monkeypatch.setenv("LAYA_DEVICE", "cuda")
    body = TestClient(create_app(router=FakeRouter())).get("/health").json()
    assert body["device_is_preference"] is True, body
    assert body["checkpoint_devices"] == {}, body
    assert body["device"] == "cuda", body          # what was asked for, labelled as such


def test_health_agrees_with_the_mcp_status_tool(monkeypatch):
    """One fact about the device, reported the same way by both surfaces."""
    pytest.importorskip("mcp")
    from laya.mcp.tools import laya_status

    monkeypatch.setenv("LAYA_DEVICE", "cuda")
    fake = DeviceRouter(english="cpu")
    body = TestClient(create_app(router=fake)).get("/health").json()
    status = laya_status(router=fake, loaded=list(fake.loaded))
    assert body["device"] == status["device"], (body["device"], status["device"])
    assert body["checkpoint_devices"] == status["checkpoint_devices"], body
    assert body["device_is_preference"] == status["device_is_preference"], body


def test_health_needs_neither_the_mcp_extra_nor_torch():
    """`laya.mcp.device` is documented as importable without `mcp`; prove the server agrees."""
    probe = r'''
import sys
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("mcp blocked")
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, %r)
from laya.mcp.device import agent_device, env_device, resolve_device, router_agent
assert agent_device(type("A", (), {"device": "cpu"})()) == "cpu"
assert env_device.__module__ == "laya.mcp.device"
import laya.serve
assert "mcp" not in sys.modules, "laya.mcp.device reached the mcp distribution"
from fastapi.testclient import TestClient
client = TestClient(laya.serve.create_app(router=type("R", (), {"loaded": ["english"],
    "_agents": {"english": type("A", (), {"device": "cpu"})()}})()))
body = client.get("/health").json()
assert body["device"] == "cpu" and body["checkpoint_devices"] == {"english": "cpu"}, body
print("ok")
''' % ROOT
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ok" in out.stdout, out.stdout


def test_build_router_strips_the_device_before_torch_sees_it(monkeypatch):
    """A value pasted from a Dockerfile or a `.env` file carries a trailing newline."""
    import laya.router
    import laya.serve

    seen = {}

    class RecordingRouter:
        def __init__(self, device=None, **kwargs):
            seen["device"] = device

        def preload(self, names=None):
            seen["preloaded"] = names

    monkeypatch.setattr(laya.router, "Router", RecordingRouter)
    monkeypatch.setenv("LAYA_PRELOAD", "0")
    for raw, want in ((" cpu\n", "cpu"), ("cuda ", "cuda"), ("   ", None), ("cpu", "cpu")):
        monkeypatch.setenv("LAYA_DEVICE", raw)
        laya.serve.build_router()
        assert seen["device"] == want, "%r -> %r" % (raw, seen["device"])
    monkeypatch.delenv("LAYA_DEVICE")
    laya.serve.build_router()
    assert seen["device"] is None, repr(seen["device"])      # unset means auto


def test_health_supports_router_without_loaded_revisions(monkeypatch):
    # FakeRouter deliberately has no loaded_revisions attribute. Injected test or
    # embedding routers predating revision reporting must remain health-compatible.
    client, _ = _client(monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["revisions"] == {}
    # Same for the #351 fallback counters: a router with no _agents and no agents
    # with counters still reports a stable, all-zero shape.
    assert r.json()["cpu_fallbacks"] == {"english": {"count": 0, "last_reason": None}}


def test_health_reports_cpu_fallback_counters(monkeypatch):
    """A resident agent that triggered the scoped CPU fallback shows it in /health."""
    from types import SimpleNamespace

    client, fake = _client(monkeypatch)
    fake._agents = {"english": SimpleNamespace(
        cpu_fallback_count=2,
        last_fallback_reason="CUDA out of memory. Tried to allocate 1.00 GiB",
    )}
    r = client.get("/health")
    assert r.status_code == 200
    fb = r.json()["cpu_fallbacks"]
    assert fb["english"]["count"] == 2, fb
    assert "out of memory" in fb["english"]["last_reason"], fb


def test_models_lists_checkpoints(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/v1/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    # exactly the names _resolve_model honours as explicit checkpoints
    assert names == ["english", "multilingual", "typed-decisions"]
    for m in r.json()["models"]:
        assert m["description"]


def test_models_requires_auth_when_key_set(monkeypatch):
    client, _ = _client(monkeypatch, api_key="s3cret")
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


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
    pytest.importorskip("torch")
    monkeypatch.delenv("LAYA_THREADS", raising=False)
    assert _apply_thread_limit() is None  # unset -> no-op, no torch import
    for bad in ("0", "-4", "abc", ""):
        monkeypatch.setenv("LAYA_THREADS", bad)
        assert _apply_thread_limit() is None
    monkeypatch.setenv("LAYA_THREADS", "8")
    assert _apply_thread_limit() == 8
    import torch
    assert torch.get_num_threads() == 8


# README's own answer to a checkpoint-rebuild storm is a constructor argument --
# `Router(max_loaded=3)   # keep all three hot, e.g. with auto_task_detection` -- and #172
# measured what ignoring it costs: 20-23 s per request reloading a checkpoint on CPU against
# 49-136 ms with it resident. The server builds its own Router from the environment and had no
# way to pass it, so the one configuration that needs the knob (auto task routing, which adds a
# third checkpoint reached on demand) could not use it. These drive `build_router()` itself,
# with the loader replaced by a stub, so nothing is downloaded.
class _StubAgent:
    def __init__(self, name):
        self.name = name

    def system_one(self, state, questions):
        return {"model": self.name, "answers": {}, "usage": {}}


def _server_router(monkeypatch, **env):
    """The Router `laya-serve` builds for `env`, with loads recorded instead of performed."""
    from laya.router import normalise_name
    from laya.serve import build_router

    monkeypatch.setenv("LAYA_PRELOAD", "0")       # nothing may download
    monkeypatch.setenv("LAYA_AUTO_TASK", "1")     # the config that puts three checkpoints in play
    monkeypatch.delenv("LAYA_MAX_LOADED", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    router = build_router()
    built = []

    def load(name):
        key = normalise_name(name)
        if key in router._agents:
            router._touch(key)
            return router._agents[key]
        built.append(key)
        router._agents[key] = _StubAgent(key)
        router._order.append(key)
        router._evict()
        return router._agents[key]

    router.load = load
    return router, built


# One request per checkpoint, so a cap of 2 cannot hold them all.
_WORKLOAD = [
    ({"body": "I was charged twice, please refund the duplicate"},
     {"issue": {"type": "choice", "options": ["billing", "other"]}}),
    ({"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung"},
     {"issue": {"type": "choice", "options": ["billing", "other"]}}),
    ({"body": "Invoice 4411 was paid twice. Please refund the duplicate line."},
     {qid: {"type": "choice", "options": ["yes", "no"]}
      for qid in sorted(_TYPED_DECISION_WORKFLOWS["customer_service"])}),
]


def _run_workload(router, cycles):
    for _ in range(cycles):
        for state, questions in _WORKLOAD:
            router.predict(state, questions)


def test_max_loaded_reaches_the_router_the_server_builds(monkeypatch):
    from laya.router import Router

    # Unset has to be reported as "not set", not as a copy of Router's default, or the two
    # numbers drift the day the default moves. Checked at the resolver, because a copy of 2 and
    # the real default are otherwise indistinguishable at the Router.
    monkeypatch.delenv("LAYA_MAX_LOADED", raising=False)
    assert _resolve_max_loaded() is None
    for raw in ("abc", "0", "-2", "2.5", ""):
        monkeypatch.setenv("LAYA_MAX_LOADED", raw)
        assert _resolve_max_loaded() is None, raw
    monkeypatch.setenv("LAYA_MAX_LOADED", "3")
    assert _resolve_max_loaded() == 3

    # And it reaches the Router the server actually builds. The literal below is the value the
    # docs quote, so moving Router's default has to move those too.
    router, _ = _server_router(monkeypatch)
    assert router.max_loaded == Router().max_loaded
    assert router.max_loaded == 2
    for raw, want in (("3", 3), ("1", 1), (" 4 ", 4)):
        router, _ = _server_router(monkeypatch, LAYA_MAX_LOADED=raw)
        assert router.max_loaded == want, raw
    # A bad value falls back the way LAYA_MAX_CONCURRENT's does: it must not stop the server
    # and must not be read as "no limit" or "one".
    for raw in ("abc", "0", "-2", "2.5", ""):
        router, _ = _server_router(monkeypatch, LAYA_MAX_LOADED=raw)
        assert router.max_loaded == 2, raw


def test_raising_the_cap_stops_the_server_rebuilding_a_checkpoint(monkeypatch):
    from laya.router import DEFAULT_MODELS

    cycles = 3
    router, built = _server_router(monkeypatch)
    _run_workload(router, cycles)
    # The instrument has to be the workload the clause is about: three checkpoints in play,
    # and the default cap that cannot hold them.
    assert len(set(built)) == 3, built
    assert router.max_loaded == 2
    assert len(built) == 9, built               # every request after the second rebuilds one

    roomy, built3 = _server_router(monkeypatch, LAYA_MAX_LOADED="3")
    _run_workload(roomy, cycles)
    assert roomy.max_loaded == 3
    assert len(built3) == 3, built3             # each checkpoint once, then they stay resident
    assert sorted(roomy.loaded) == sorted(DEFAULT_MODELS)



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


class ValidatingRouter:
    """The real guard, without a checkpoint: what `Agent.system_one` runs before encoding.

    The app does not validate `criteria` itself -- the agent does -- so the stub calls the
    same guard `system_one` calls, and any `ValueError` it raises is what `serve` has to map
    to 422. `predict` still fails loudly if the guard lets something through.
    """

    loaded = ["english"]

    def predict(self, state, questions, model=None):
        from laya.agent import Agent
        for qid, qdef in questions.items():
            Agent._check_question(qid, qdef)
        raise AssertionError("validation should have rejected this before predict()")


def test_a_nested_choice_label_is_a_caller_error_not_a_server_fault(monkeypatch):
    """A `criteria` list containing a list/dict label is the caller's mistake, so it must be 422.

    It used to raise `TypeError: unhashable type: 'list'` from `_to_internal`, three frames below
    `_check_question`, which names neither the question nor the label -- and `serve` maps only
    `ValueError` to 422, so the caller got a 500 "inference failed" with the reason discarded.
    `ValueError` is what carries the message to the client, so the guard has to raise that type.
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=ValidatingRouter()), raise_server_exceptions=False)

    for label in (["billing"], {"billing": "x"}):
        body = dict(REQ)
        body["questions"] = {"dept": {"type": "choice", "instructions": "Which team?",
                                      "criteria": [label, "tech"]}}
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 422, (label, response.status_code, response.text)
        assert "choice label 0" in response.text, response.text


def test_a_colliding_choice_label_is_a_caller_error_not_a_server_fault(monkeypatch):
    """A `criteria` list that cannot produce one answer key per option must be 422, not 500.

    `_to_internal` normalises the list form to `{label: None}`, so two entries that land on one
    key scored fewer options than the caller wrote. `serve` maps `ValueError` to 422 and anything
    else to a 500 "inference failed", so the guard has to raise `ValueError` and say which labels
    collided. (An unhashable label is the same class of mistake, but JSON has no tuple: a list or
    dict label arrives as one of those and the guard above already names it, which
    `test_a_nested_choice_label_is_a_caller_error_not_a_server_fault` covers.)
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=ValidatingRouter()), raise_server_exceptions=False)

    for criteria, expect in (
        (["billing", "billing", "tech"], "repeats label 0"),
        ([1, 1.0], "repeats label 0"),       # one dict key, two entries
        ([True, 1], "repeats label 0"),      # `True == 1` is one dict key too
    ):
        body = dict(REQ)
        body["questions"] = {"dept": {"type": "choice", "instructions": "Which team?",
                                      "criteria": criteria}}
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 422, (criteria, response.status_code, response.text)
        assert expect in response.text, (criteria, response.text)


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


def test_a_missing_state_is_rejected_rather_than_answered():
    """No `state` key, or `"state": null`, must be a 400 and not a decision about "null".

    `serialize_state(None)` is `json.dumps(None)` -- the four characters `null` -- so the request
    was answered as a decision about that literal text: HTTP 200, byte-identical to sending
    `"state": "null"`, and at ~0.94 confidence on the real checkpoint. The caller gets an answer
    about a state they never supplied, with nothing in the response to say so.
    """
    router = FakeRouter()
    client = TestClient(create_app(router=router))
    questions = {"dept": {"type": "choice", "instructions": "which?",
                          "criteria": {"billing": "invoices"}}}

    for body in ({"questions": questions},                      # no state key
                 {"state": None, "questions": questions}):      # explicit null
        res = client.post("/v1/systemone", json=body)
        assert res.status_code == 400, (body, res.status_code, res.text)
        assert "'state' is required" in res.text, res.text

    # a state that IS a string is the caller's business, including the text "null" and ""
    for state in ("null", "", "0"):
        res = client.post("/v1/systemone", json={"state": state, "questions": questions})
        assert res.status_code == 200, (state, res.status_code, res.text)


class GatedRouter(FakeRouter):
    """Blocks inside predict until released, so a second request arrives while
    the first still holds its admission slot (#330)."""

    def __init__(self):
        super().__init__()
        import threading
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict(self, state, questions, model=None):
        self.entered.set()
        assert self.release.wait(timeout=10), "test did not release the router"
        return super().predict(state, questions, model=model)


def test_admission_bound_refuses_with_503_when_full(monkeypatch):
    """With one admission slot and inference blocked, a second concurrent
    request gets 503 instead of queueing another body in memory."""
    import asyncio

    import httpx

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "1")
    fake = GatedRouter()
    app = create_app(router=fake)
    seen = {}

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            first = asyncio.ensure_future(client.post("/v1/systemone", json=REQ))
            # Poll: a blocking wait here would stall the loop the first
            # request needs to reach inference.
            for _ in range(200):
                if fake.entered.is_set():
                    break
                await asyncio.sleep(0.05)
            assert fake.entered.is_set(), "first request never reached inference"
            # Give the first request a moment to settle past the gate too, so the
            # second request deterministically finds the slot taken.
            await asyncio.sleep(0.2)
            second = await client.post("/v1/systemone", json=REQ)
            seen["second"] = second.status_code
            seen["retry_after"] = second.headers.get("retry-after")
            fake.release.set()
            seen["first"] = (await first).status_code

    asyncio.run(drive())

    assert seen["second"] == 503, seen
    assert seen["retry_after"] == "1", seen
    assert seen["first"] == 200, seen


def test_admission_slot_is_released_after_inference(monkeypatch):
    """Slots are reusable: sequential requests with a bound of one all pass."""
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "1")
    client = TestClient(create_app(router=FakeRouter()))
    assert client.post("/v1/systemone", json=REQ).status_code == 200
    assert client.post("/v1/systemone", json=REQ).status_code == 200


def test_accepted_connections_set_tcp_nodelay(monkeypatch):
    """#620: asyncio skips TCP_NODELAY when an accepted socket reports proto 0, as it
    does on macOS and Windows, so Nagle held back small responses by about 50 ms."""
    import asyncio
    import socket

    import uvicorn

    import laya.serve

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(laya.serve, "create_app", lambda: None)
    laya.serve.main()

    async def drive():
        config = uvicorn.Config(create_app(router=FakeRouter()), host="127.0.0.1", port=0,
                                http=captured["http"], log_level="warning")
        server = uvicorn.Server(config)
        serving = asyncio.ensure_future(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        while not server.server_state.connections:
            await asyncio.sleep(0.01)
        (conn,) = server.server_state.connections
        nodelay = conn.transport.get_extra_info("socket").getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        writer.close()
        server.should_exit = True
        await serving
        return nodelay

    assert asyncio.run(drive())


def test_no_budget_keeps_the_call_unchanged(monkeypatch):
    """An injected router whose predict() takes no kwargs continues to work when body sends no budget."""
    client, fake = _client(monkeypatch)
    for body in (REQ, dict(REQ, max_len=None, head_max_len=None)):
        assert client.post("/v1/systemone", json=body).status_code == 200
    assert len(fake.calls) == 2


def test_token_budget_forwarded(monkeypatch):
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "max_len": 4096, "head_max_len": 256})
    assert r.status_code == 200
    assert fake.calls[0]["max_len"] == 4096
    assert fake.calls[0]["head_max_len"] == 256


@pytest.mark.parametrize("bad_budget", ["fast", True, 3.14])
def test_token_budget_validation_type(monkeypatch, bad_budget):
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "max_len": bad_budget})
    assert r.status_code == 422
    assert "must be an integer" in r.json()["detail"]


@pytest.mark.parametrize("bad_val", [0, -10])
def test_token_budget_validation_positive(monkeypatch, bad_val):
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "max_len": bad_val})
    assert r.status_code == 422
    assert "must be a positive integer" in r.json()["detail"]


def test_token_budget_exceeds_server_cap(monkeypatch):
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "max_len": 9000})
    assert r.status_code == 422
    assert "exceeds server limit" in r.json()["detail"]


def test_token_budget_head_max_len_equal_to_max_len(monkeypatch):
    """Core accepts head_max_len == max_len; serve forwards both without artificial restriction."""
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "max_len": 512, "head_max_len": 512})
    assert r.status_code == 200
    assert fake.calls[0]["max_len"] == 512
    assert fake.calls[0]["head_max_len"] == 512


def test_token_budget_env_cap_override(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_TOKEN_BUDGET", "2048")
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "max_len": 4096})
    assert r.status_code == 422
    assert "exceeds server limit" in r.json()["detail"]

    r2 = client.post("/v1/systemone", json={**REQ, "max_len": 2048})
    assert r2.status_code == 200
    assert fake.calls[0]["max_len"] == 2048


def test_resolve_max_token_budget_fallback(monkeypatch, caplog):
    monkeypatch.delenv("LAYA_MAX_TOKEN_BUDGET", raising=False)
    assert _resolve_max_token_budget() == DEFAULT_MAX_TOKEN_BUDGET
    for bad in ("abc", "-10", "0"):
        monkeypatch.setenv("LAYA_MAX_TOKEN_BUDGET", bad)
        assert _resolve_max_token_budget() == DEFAULT_MAX_TOKEN_BUDGET
    assert "invalid LAYA_MAX_TOKEN_BUDGET" in caplog.text
    assert "LAYA_MAX_TOKEN_BUDGET must be positive" in caplog.text

