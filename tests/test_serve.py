"""Server-shim tests: verify the Jev /v1/systemone surface without a GPU.

A fake Router is injected so nothing loads a checkpoint; we only assert that the
HTTP layer maps requests/responses and enforces auth as hs-jev expects.
"""
import ast
import asyncio
import inspect
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
    BATCH_BODY_CALL_CONTROLS,
    BATCH_BODY_ITEM_CONTROLS,
    BODY_CONTROLS,
    BODY_REFUSALS,
    DEFAULT_MAX_TOKEN_BUDGET,
    MAX_BODY_BYTES,
    MAX_STATE_CHARS,
    _apply_thread_limit,
    _check_request_limits,
    _env_bool,
    _LONE_SURROGATE_DETAIL,
    _resolve_max_token_budget,
    _resolve_idle_unload_seconds,
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


def test_root_path_from_environment_updates_openapi_and_keeps_routes(monkeypatch):
    monkeypatch.setenv("LAYA_ROOT_PATH", "/laya")
    app = create_app(router=FakeRouter())

    assert app.root_path == "/laya"
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").json()["servers"] == [{"url": "/laya"}]


@pytest.mark.parametrize("root_path", [None, ""])
def test_root_path_defaults_to_empty(monkeypatch, root_path):
    if root_path is None:
        monkeypatch.delenv("LAYA_ROOT_PATH", raising=False)
    else:
        monkeypatch.setenv("LAYA_ROOT_PATH", root_path)

    app = create_app(router=FakeRouter())

    assert app.root_path == ""



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


# A router that answers with the full Laya payload: every answer type, the per-answer
# additions and the extended usage report -- what a strict client must be projected off.
FULL_ANSWERS = {
    "queue": {"type": "choice", "choice": "billing",
              "probabilities": {"billing": 0.9519, "tech": 0.0327, "other": 0.0154},
              "confidence": 0.797, "answer_confidence": 0.9519,
              "action": {"act_probability": 1.0}},
    "urgency": {"type": "score", "score": 1.6994,
                "legend": {"0": "calm", "1": "firm", "2": "angry"},
                "probabilities": {"0": 0.0249, "1": 0.65, "2": 0.3251},
                "confidence": 0.1925, "answer_confidence": 0.65,
                "action": {"act_probability": 0.5}},
    "threat": {"type": "noul", "noul": 0.9148, "confidence": 0.9148,
               "answer_confidence": 0.9148, "action": {"act_probability": 1.0}},
}
FULL_USAGE = {"input_tokens": 83, "output_tokens": 0, "state_tokens": 12,
              "state_tokens_dropped": 0, "truncated": False, "truncated_questions": []}


class StrictFullRouter:
    """Returns the full payload, additions and all, whatever the request."""

    loaded = ["english"]

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        return {"model": "laya-rl-agent", "answers": FULL_ANSWERS, "usage": dict(FULL_USAGE),
                "routing": {"model": "english", "reason": "English Latin text"}}

    def predict_batch(self, requests, **kwargs):
        return [self.predict(item["state"], item["questions"], model=item.get("model"))
                for item in requests]


def _strict_client(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("LAYA_JEV_STRICT", raising=False)
    else:
        monkeypatch.setenv("LAYA_JEV_STRICT", value)
    fake = StrictFullRouter()
    return TestClient(create_app(router=fake)), fake


def test_jev_strict_projects_the_full_payload(monkeypatch):
    client, _ = _strict_client(monkeypatch, "1")
    r = client.post("/v1/systemone", json=REQ)
    assert r.status_code == 200
    body = r.json()
    # the strict contract: exactly three top-level fields, the contracted answer keys
    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "laya-rl-agent"
    assert body["answers"]["queue"] == {"type": "choice", "choice": "billing",
                                        "confidence": 0.797,
                                        "probabilities": {"billing": 0.9519, "tech": 0.0327,
                                                          "other": 0.0154}}
    assert body["answers"]["urgency"] == {"type": "score", "score": 1.6994,
                                          "confidence": 0.1925,
                                          "probabilities": {"0": 0.0249, "1": 0.65, "2": 0.3251},
                                          "legend": {"0": "calm", "1": "firm", "2": "angry"}}
    assert body["answers"]["threat"] == {"type": "noul", "noul": 0.9148}
    # usage reduced to the two contracted counts, with the values the result carried
    assert body["usage"] == {"input_tokens": 83, "output_tokens": 0}


def test_jev_strict_is_off_by_default(monkeypatch):
    """No flag: the full payload, additions and all, byte for byte."""
    client, _ = _strict_client(monkeypatch, None)
    body = client.post("/v1/systemone", json=REQ).json()
    assert set(body) == {"model", "answers", "usage", "routing"}
    assert "action" in body["answers"]["queue"]
    assert "confidence" in body["answers"]["threat"]
    assert body["usage"] == FULL_USAGE


def test_jev_strict_rejects_falsy_spellings(monkeypatch):
    client, _ = _strict_client(monkeypatch, "0")
    body = client.post("/v1/systemone", json=REQ).json()
    assert "routing" in body


def test_jev_strict_batch_projects_every_item(monkeypatch):
    client, fake = _strict_client(monkeypatch, "1")
    r = client.post("/v1/systemone/batch", json={"states": ["one", "two"],
                                                 "questions": REQ["questions"]})
    assert r.status_code == 200
    data = r.json()
    assert [set(item) for item in data["results"]] == [
        {"model", "answers", "usage"}, {"model", "answers", "usage"}]
    assert all("routing" not in item for item in data["results"])
    assert all(item["usage"] == {"input_tokens": 83, "output_tokens": 0}
               for item in data["results"])
    # total_usage is the batch envelope's own, untouched by the projection
    assert data["total_usage"] == {"input_tokens": 166, "output_tokens": 0}


def test_jev_strict_leaves_an_unknown_answer_shape_unchanged(monkeypatch):
    """A shape the contract does not name is passed through: the caller still sees it."""
    monkeypatch.setenv("LAYA_JEV_STRICT", "1")
    monkeypatch.delenv("LAYA_API_KEY", raising=False)

    class OddRouter:
        loaded = ["english"]

        def predict(self, state, questions, model=None):
            return {"model": "laya-rl-agent",
                    "answers": {"odd": {"type": "weird", "payload": 1}},
                    "usage": {"input_tokens": 1, "output_tokens": 0}}

    client = TestClient(create_app(router=OddRouter()))
    body = client.post("/v1/systemone", json=REQ).json()
    assert body["answers"]["odd"] == {"type": "weird", "payload": 1}
    assert "routing" not in body


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


def test_resolve_model_follows_the_router_registry(monkeypatch):
    """A checkpoint added to laya.router must become pinnable over HTTP with no edit here.

    serve used to keep two hand-copies of that registry -- a set of accepted names and a map of
    published Hugging Face ids. Either copy left behind means a client that names a real
    checkpoint is auto-routed instead, and the response reports whichever model did answer, so
    the dropped pin is invisible from the outside. Only the name maps are read; nothing loads a
    checkpoint.
    """
    from laya import router as router_mod

    monkeypatch.setitem(router_mod.DEFAULT_MODELS, "spanish", (router_mod.BUNDLE_REPO, "spanish"))
    monkeypatch.setitem(router_mod.STANDALONE_MODELS, "spanish", "convaiinnovations/laya-spanish")
    monkeypatch.setitem(router_mod._ALIASES, "es", "spanish")

    # The engine accepts the new name and its alias, which is the whole contract serve needs.
    assert router_mod.normalise_name("spanish") == "spanish"
    assert router_mod.normalise_name("es") == "spanish"

    for named in ("spanish", "SPANISH", "es", "convaiinnovations/laya-spanish"):
        assert _resolve_model(named) == "spanish", named

    # And the pin reaches inference rather than stopping in the resolver.
    client, fake = _client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "model": "spanish"})
    assert r.status_code == 200, r.status_code
    assert fake.calls[-1]["model"] == "spanish", fake.calls[-1]

    # Every standalone id the router publishes pins its checkpoint here -- except the bundle
    # repo, whose documented meaning stays "let the Router choose".
    for name, repo in router_mod.STANDALONE_MODELS.items():
        want = None if repo == router_mod.BUNDLE_REPO else name
        assert _resolve_model(repo) is want, repo

    # A name the router does not know is still the caller's own model id: ignored, not an error.
    assert _resolve_model("jev-1") is None


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
    monkeypatch.delenv("LAYA_DEFAULT_MODEL", raising=False)
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


# README's other answer written as a constructor argument is `Router(default="multilingual")`,
# given where it explains that very short Latin-script text ("Quero cancelar", "Esqueci minha
# senha") carries nothing identifying its language and so goes to `default`. Both packaged
# servers build their Router from the environment and neither could pass it: the only reader of
# LAYA_DEFAULT_MODEL was examples/server.py, so the demo server honoured the variable while
# laya-serve routed the ambiguous states to the English checkpoint it was told not to use.
#
# Unlike the numeric knobs above, an unresolvable name is fatal here. `LAYA_MAX_LOADED=abc`
# falling back to 2 costs a rebuild; `LAYA_DEFAULT_MODEL=mutli` falling back to english costs the
# wrong checkpoint answering every ambiguous state, which is the failure the operator was trying
# to configure away. So the message is core's and the server refuses to start.
def test_default_model_reaches_the_router_the_server_builds(monkeypatch):
    from laya.router import DEFAULT_MODELS, Router, _ALIASES, normalise_name
    from laya.serve import _default_model_option

    # Unset is "not asked for", not a copy of Router's default -- the same reason
    # `_resolve_max_loaded` returns None, and the same drift it prevents.
    monkeypatch.delenv("LAYA_DEFAULT_MODEL", raising=False)
    assert _default_model_option() == {}
    for blank in ("", "   ", "\n"):
        monkeypatch.setenv("LAYA_DEFAULT_MODEL", blank)
        assert _default_model_option() == {}, repr(blank)

    # The accepted spellings are core's tables read out of core, so a name added to _ALIASES
    # arrives here on its own -- and a value this module resolves differently from the way
    # `Router` itself resolves it fails the same line.
    for name in sorted(set(DEFAULT_MODELS) | set(_ALIASES)):
        monkeypatch.setenv("LAYA_DEFAULT_MODEL", name)
        assert _default_model_option() == {"default": normalise_name(name)}, name

    # And it reaches the Router the server builds. The literal is the one the README and
    # docs/docker.md quote, so moving Router's default has to move those too.
    router, _ = _server_router(monkeypatch)
    assert router.default == Router().default
    assert router.default == "english"
    for raw, want in (("multilingual", "multilingual"), ("ml", "multilingual"),
                      (" MULTI ", "multilingual"), ("typed-decisions", "typed-decisions")):
        router, _ = _server_router(monkeypatch, LAYA_DEFAULT_MODEL=raw)
        assert router.default == want, raw

    # The decision the whole knob exists to change, measured at `route()` rather than at the
    # attribute -- `route` documents that it decides "without loading or running anything", so
    # this costs no weights.
    ambiguous = ("12345 !!!", "Quero cancelar", "Esqueci minha senha")
    stock, _ = _server_router(monkeypatch)
    portuguese, _ = _server_router(monkeypatch, LAYA_DEFAULT_MODEL="multilingual")
    for state in ambiguous:
        assert stock.route(state).model == "english", state
        assert "using default (english)" in stock.route(state).reason, state
        assert portuguese.route(state).model == "multilingual", state
        assert "using default (multilingual)" in portuguese.route(state).reason, state
    # A fallback, not a pin: text the detector can place routes on what it detects.
    assert portuguese.route({"body": "Please refund the duplicate charge"}).model == "english"


def test_an_unresolvable_default_stops_the_server_with_core_s_words(monkeypatch):
    from laya.serve import build_router

    monkeypatch.setenv("LAYA_PRELOAD", "0")
    monkeypatch.delenv("LAYA_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("LAYA_DEFAULT_MODEL", "mutli-lingual")
    with pytest.raises(SystemExit) as raised:
        build_router()
    message = str(raised.value)
    # Names the variable, quotes the value the operator typed, and then says in core's own words
    # what the accepted set is -- so the typo is fixed from the message, not from the source.
    assert message.startswith("invalid LAYA_DEFAULT_MODEL 'mutli-lingual': unknown model"), message
    for name in ("english", "multilingual", "typed-decisions", "ml"):
        assert name in message, message
    # The contrast with the numeric knobs has to stay the contrast: with the name fixed, the bad
    # number below still falls back instead of stopping the server.
    monkeypatch.setenv("LAYA_DEFAULT_MODEL", "english")
    monkeypatch.setenv("LAYA_MAX_LOADED", "abc")
    assert build_router().default == "english"
    assert build_router().max_loaded == 2



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


@pytest.mark.parametrize("bad_type", [[], {}, "bogus", 123])
def test_malformed_question_type_is_a_named_422(monkeypatch, bad_type):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=ValidatingRouter()), raise_server_exceptions=False)
    body = dict(REQ)
    body["questions"] = {"refund": {"type": bad_type, "instructions": "Is a refund requested?"}}

    response = client.post("/v1/systemone", json=body)

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == (
        "question 'refund': unknown type %r; use one of ['choice', 'noul', 'score']" % (bad_type,))


@pytest.mark.parametrize("bad_ins", [None, "", "   ", []])
def test_malformed_instructions_is_a_named_422(monkeypatch, bad_ins):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=ValidatingRouter()), raise_server_exceptions=False)
    body = dict(REQ)
    body["questions"] = {"refund": {"type": "noul", "instructions": bad_ins}}
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422, response.text
    assert "instructions" in response.text, response.text


@pytest.mark.parametrize("bad_qid", ["", "   "])
def test_malformed_question_id_is_a_named_422(monkeypatch, bad_qid):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=ValidatingRouter()), raise_server_exceptions=False)
    body = dict(REQ)
    body["questions"] = {bad_qid: {"type": "noul", "instructions": "Is it urgent?"}}
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422, response.text
    assert "question id" in response.text, response.text


def test_score_null_level_is_a_named_422(monkeypatch):
    """A null score level would come back as `legend: {"<i>": null}`, which Jev clients refuse to
    parse (#302). The request is rejected as a named 422 instead of answered 200."""
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=ValidatingRouter()), raise_server_exceptions=False)
    body = dict(REQ)
    body["questions"] = {"urgency": {"type": "score", "instructions": "How urgent?",
                                     "criteria": ["low", None, "high"]}}
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422, response.text
    assert "null level" in response.text and "urgency" in response.text, response.text


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


_WAIT_TIMEOUT = 5.0  # seconds; a loopback bind that has not landed by now never will (#721)


async def _wait_for_condition(condition, what, timeout=_WAIT_TIMEOUT, interval=0.01):
    """Poll `condition()` until it is truthy, or fail naming `what` rather than hang (#721).

    An unbounded `while not ...: await asyncio.sleep(interval)` looks identical whether the
    condition arrives in a millisecond or never, and prints nothing either way, so a host whose
    loopback networking reports differently from the runners leaves pytest stalled instead of
    failing. Bounding the wait turns that into an ordinary failure that names the step.
    """

    async def poll():
        while not condition():
            await asyncio.sleep(interval)

    try:
        await asyncio.wait_for(poll(), timeout)
    except asyncio.TimeoutError:
        pytest.fail(f"timed out after {timeout}s waiting for {what}")


def test_a_bounded_wait_fails_on_a_condition_that_never_lands():
    """#721: the bound is the whole point, so it is exercised against a condition that never
    becomes true. Unbounded, this is the hang the helper exists to remove, and the same code
    path then cannot be checked by a test that waits for it."""
    async def lands():
        await _wait_for_condition(lambda: True, "an already-true condition")

    asyncio.run(lands())  # the satisfied case returns rather than waiting out the timeout

    async def never():
        await _wait_for_condition(lambda: False, "a condition that never lands", timeout=0.05)

    with pytest.raises(pytest.fail.Exception, match="a condition that never lands"):
        asyncio.run(never())


def test_accepted_connections_set_tcp_nodelay(monkeypatch):
    """#620: asyncio skips TCP_NODELAY when an accepted socket reports proto 0, as it
    does on macOS and Windows, so Nagle held back small responses by about 50 ms."""
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
        writer = None
        try:
            await _wait_for_condition(lambda: server.started, "uvicorn to bind and start serving")
            port = server.servers[0].sockets[0].getsockname()[1]
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            await _wait_for_condition(lambda: server.server_state.connections,
                                      "the loopback connection to be accepted")
            (conn,) = server.server_state.connections
            return conn.transport.get_extra_info("socket").getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        finally:
            # A timed-out wait raises out of `drive()`, and the server task and the client
            # connection would then outlive the test, so tear both down on every path.
            if writer is not None:
                writer.close()
            server.should_exit = True
            await asyncio.wait_for(serving, _WAIT_TIMEOUT)

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

BATCH_REQ = {
    "states": ["first state", "second state"],
    "questions": REQ["questions"],
}


def test_batch_happy_path(monkeypatch):
    client, fake = _client(monkeypatch)
    r = client.post("/v1/systemone/batch", json=BATCH_REQ)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert "total_usage" in data
    assert data["total_usage"]["input_tokens"] == sum(
        res["usage"]["input_tokens"] for res in data["results"]
    )
    assert len(fake.calls) == 2
    assert "Server-Timing" in r.headers
    assert "X-Inference-Time-Ms" in r.headers


def test_batch_sums_output_tokens():
    """total_usage must sum output_tokens, not report a hardcoded 0 (each result carries its own)."""
    class _UsageRouter:
        def predict(self, state, questions, model=None, **kwargs):
            return {"model": "laya-rl-agent",
                    "answers": {"dept": {"type": "choice", "choice": "billing",
                                         "probabilities": {"billing": 1.0}, "confidence": 1.0}},
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                    "routing": {"model": "english"}}

    client = TestClient(create_app(router=_UsageRouter()))
    r = client.post("/v1/systemone/batch", json=BATCH_REQ)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total_usage"]["output_tokens"] == sum(
        res["usage"]["output_tokens"] for res in data["results"])
    assert data["total_usage"]["output_tokens"] == 6   # two states x 3 each
    assert data["total_usage"]["input_tokens"] == 10


def test_batch_missing_state_in_list_returns_400(monkeypatch):
    """A None state inside states list must be rejected with 400 'state' is required."""
    client, _ = _client(monkeypatch)
    bad_req = {"states": [None, "hello"], "questions": REQ["questions"]}
    r = client.post("/v1/systemone/batch", json=bad_req)
    assert r.status_code == 400
    assert "'state' is required" in r.json()["detail"]


def test_batch_too_many_options_returns_413(monkeypatch):
    """Questions with more than MAX_CHOICE_OPTIONS must be rejected with 413."""
    client, _ = _client(monkeypatch)
    bad_questions = {
        "dept": {
            "type": "choice",
            "instructions": "which?",
            "criteria": {f"opt_{i}": f"desc_{i}" for i in range(101)},
        }
    }
    r = client.post("/v1/systemone/batch", json={"states": ["state"], "questions": bad_questions})
    assert r.status_code == 413
    assert "too many choice options" in r.json()["detail"]


def test_batch_empty_states_returns_400(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/systemone/batch", json={"states": [], "questions": REQ["questions"]})
    assert r.status_code == 400
    assert "states" in r.json()["detail"]


def test_batch_missing_states_returns_400(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/systemone/batch", json={"questions": REQ["questions"]})
    assert r.status_code == 400


def test_batch_missing_questions_returns_400(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/systemone/batch", json={"states": ["state 1"]})
    assert r.status_code == 400


def test_batch_too_many_states_returns_413(monkeypatch):
    client, _ = _client(monkeypatch)
    from laya.serve import MAX_BATCH_STATES
    oversized = {"states": ["state"] * (MAX_BATCH_STATES + 1), "questions": REQ["questions"]}
    r = client.post("/v1/systemone/batch", json=oversized)
    assert r.status_code == 413
    assert "too many states" in r.json()["detail"]


def _noul_questions(n):
    """`n` minimal questions, so a batch's row count is `len(states) * n` and nothing else."""
    return {("q%03d" % i): {"type": "noul", "instructions": "is it urgent?"} for i in range(n)}


def _batch_env_client(monkeypatch, **env):
    """A batch client with the batch-token knob explicitly set, or explicitly unset.

    Every test below states the budget it measures against. Without this an exported
    `LAYA_MAX_BATCH_TOKENS` silently changes the chunk and these pass or fail for a reason that is not
    in the file -- there is no `conftest.py` in this repo to isolate the environment, and
    `_batch_client` clears only `LAYA_API_KEY`.
    """
    if "LAYA_MAX_BATCH_TOKENS" in env:
        monkeypatch.setenv("LAYA_MAX_BATCH_TOKENS", env["LAYA_MAX_BATCH_TOKENS"])
    else:
        monkeypatch.delenv("LAYA_MAX_BATCH_TOKENS", raising=False)
    return _batch_client(monkeypatch)


def _sent_batch_size(fake):
    """The `batch_size` the route handed `predict_batch`, or the sentinel when it passed none."""
    _, kwargs = fake.batch_calls[0]
    return kwargs["batch_size"] if "batch_size" in kwargs else "<absent>"


# --------------------------------------------------------------------- the batch token budget
#
# `/v1/systemone/batch` collates `states x questions` rows into one tensor and `batch_size` defaults to
# `None` -- "all in one pass" -- so the two field caps multiply: 64 states of 64 questions is 4096 rows
# from a 4.1 KB body. The budget SPLITS the work rather than refusing it, because a refusal would have
# to be right about hardware this process cannot see: measured on english, peak RSS did not move
# between 32, 64 and 128 rows. So the contract under test is "one pass stays inside the budget, and a
# request that already fits is passed no `batch_size` at all".

def test_batch_within_the_budget_is_passed_no_batch_size(monkeypatch):
    """A request that fits must behave byte-identically, which means not passing `batch_size`.

    `predict_batch` warns that changing batch shapes can move floating-point results, so a request
    that works today must not start answering differently. Not-passing is the only way to guarantee
    that: it leaves `predict_batch` on its own default.
    """
    import laya.serve as serve_mod

    client, fake = _batch_env_client(monkeypatch)
    rows = serve_mod.DEFAULT_MAX_BATCH_TOKENS // serve_mod._BATCH_ROW_TOKENS_ASSUMED
    r = client.post("/v1/systemone/batch",
                    json={"states": ["hi"] * 8, "questions": _noul_questions(rows // 8)})
    assert r.status_code == 200, r.text
    requests, kwargs = fake.batch_calls[0]
    assert len(requests) * len(requests[0]["questions"]) == rows, "exactly at the budget"
    assert "batch_size" not in kwargs, kwargs


def test_batch_over_the_budget_is_split_not_refused(monkeypatch):
    """The 4096-row shape still answers; it is chunked so one pass stays inside the budget."""
    import laya.serve as serve_mod

    client, fake = _batch_env_client(monkeypatch)
    r = client.post("/v1/systemone/batch",
                    json={"states": ["hi"] * serve_mod.MAX_BATCH_STATES,
                          "questions": _noul_questions(serve_mod.MAX_QUESTIONS)})
    assert r.status_code == 200, r.text
    rows_per_pass = serve_mod.DEFAULT_MAX_BATCH_TOKENS // serve_mod._BATCH_ROW_TOKENS_ASSUMED
    expected = max(1, rows_per_pass // serve_mod.MAX_QUESTIONS)
    assert _sent_batch_size(fake) == expected
    requests, _ = fake.batch_calls[0]
    assert len(requests) == serve_mod.MAX_BATCH_STATES, "every state is still sent"
    assert expected * serve_mod.MAX_QUESTIONS * serve_mod._BATCH_ROW_TOKENS_ASSUMED \
        <= serve_mod.DEFAULT_MAX_BATCH_TOKENS, "one pass must fit the budget"


def test_batch_one_row_over_the_budget_is_split(monkeypatch):
    """One row over, with both factors inside their own caps, so the comparison cannot be off by one."""
    import laya.serve as serve_mod

    client, fake = _batch_env_client(monkeypatch)
    rows = serve_mod.DEFAULT_MAX_BATCH_TOKENS // serve_mod._BATCH_ROW_TOKENS_ASSUMED
    states, questions = 9, rows // 8
    assert states <= serve_mod.MAX_BATCH_STATES and questions <= serve_mod.MAX_QUESTIONS
    assert states * questions > rows, "must be over by at least one row"
    r = client.post("/v1/systemone/batch",
                    json={"states": ["hi"] * states, "questions": _noul_questions(questions)})
    assert r.status_code == 200, r.text
    assert _sent_batch_size(fake) == max(1, rows // questions)


def test_batch_a_raised_max_len_chunks_harder(monkeypatch):
    """A row cap would be the wrong bound, because `max_len` is a request field.

    `max_len` is capped only by `DEFAULT_MAX_TOKEN_BUDGET` (8192), 16x the english checkpoint's own
    512, so 256 rows at 8192 is the same token count as 4096 rows at 512 and collates the same tensor.
    Counting tokens means a wider row buys proportionally fewer rows per pass.
    """
    import laya.serve as serve_mod

    wide = serve_mod.DEFAULT_MAX_TOKEN_BUDGET
    assert wide > serve_mod._BATCH_ROW_TOKENS_ASSUMED
    rows = serve_mod.DEFAULT_MAX_BATCH_TOKENS // serve_mod._BATCH_ROW_TOKENS_ASSUMED

    narrow_client, narrow = _batch_env_client(monkeypatch)
    narrow_client.post("/v1/systemone/batch",
                       json={"states": ["hi"] * 8, "questions": _noul_questions(rows // 8)})
    assert "batch_size" not in narrow.batch_calls[0][1], "fits at the default width"

    wide_client, wide_fake = _batch_env_client(monkeypatch)
    r = wide_client.post("/v1/systemone/batch",
                         json={"states": ["hi"] * 8, "questions": _noul_questions(rows // 8),
                               "max_len": wide})
    assert r.status_code == 200, r.text
    rows_per_pass = serve_mod.DEFAULT_MAX_BATCH_TOKENS // wide
    assert _sent_batch_size(wide_fake) == max(1, rows_per_pass // (rows // 8))


def test_batch_an_explicit_batch_size_is_never_overridden(monkeypatch):
    """The caller asked for a shape. Planning one on top would silently change what they requested."""
    import laya.serve as serve_mod

    client, fake = _batch_env_client(monkeypatch)
    r = client.post("/v1/systemone/batch",
                    json={"states": ["hi"] * serve_mod.MAX_BATCH_STATES,
                          "questions": _noul_questions(serve_mod.MAX_QUESTIONS),
                          "batch_size": 7})
    assert r.status_code == 200, r.text
    assert _sent_batch_size(fake) == 7


def test_batch_chunking_follows_a_raised_budget(monkeypatch):
    """A deployment whose hardware can take more stops being chunked."""
    import laya.serve as serve_mod

    client, fake = _batch_env_client(monkeypatch, LAYA_MAX_BATCH_TOKENS="2097152")
    r = client.post("/v1/systemone/batch",
                    json={"states": ["hi"] * serve_mod.MAX_BATCH_STATES,
                          "questions": _noul_questions(serve_mod.MAX_QUESTIONS)})
    assert r.status_code == 200, r.text
    assert "batch_size" not in fake.batch_calls[0][1], "4096 rows x 512 fits a 2 097 152 budget"


def test_batch_chunking_follows_a_lowered_budget(monkeypatch):
    """And a LOWERED one -- the regime an operator on small hardware actually uses."""
    import laya.serve as serve_mod

    assert 2048 < serve_mod.DEFAULT_MAX_BATCH_TOKENS, "2048 must be the lowered regime"
    client, fake = _batch_env_client(monkeypatch, LAYA_MAX_BATCH_TOKENS="2048")
    r = client.post("/v1/systemone/batch",
                    json={"states": ["hi"] * 8, "questions": _noul_questions(2)})
    assert r.status_code == 200, r.text
    # 2048 / 512 = 4 rows per pass, / 2 questions = 2 states per pass
    assert _sent_batch_size(fake) == 2


@pytest.mark.parametrize("raw", ["nonsense", "0", "-5", "3.5"])
def test_batch_budget_warns_and_falls_back_on_an_unusable_value(monkeypatch, caplog, raw):
    """A bad knob must not disable the chunking, and must say so in the log."""
    import laya.serve as serve_mod

    monkeypatch.setenv("LAYA_MAX_BATCH_TOKENS", raw)
    with caplog.at_level(logging.WARNING, logger="laya.serve"):
        assert serve_mod._resolve_max_batch_tokens() == serve_mod.DEFAULT_MAX_BATCH_TOKENS
    assert "LAYA_MAX_BATCH_TOKENS" in caplog.text, caplog.text
    assert ("invalid" in caplog.text or "must be positive" in caplog.text), caplog.text

    client, fake = _batch_client(monkeypatch)
    r = client.post("/v1/systemone/batch",
                    json={"states": ["hi"] * serve_mod.MAX_BATCH_STATES,
                          "questions": _noul_questions(serve_mod.MAX_QUESTIONS)})
    assert r.status_code == 200, r.text
    assert _sent_batch_size(fake) != "<absent>", "the fallback budget must still chunk"


def test_batch_an_empty_environment_value_falls_back_silently(monkeypatch):
    """Unset and empty both mean "not configured", which is not worth a warning."""
    import laya.serve as serve_mod

    monkeypatch.setenv("LAYA_MAX_BATCH_TOKENS", "")
    assert serve_mod._resolve_max_batch_tokens() == serve_mod.DEFAULT_MAX_BATCH_TOKENS


def test_batch_chunk_plan_floors_at_one_state(monkeypatch):
    """A single state cannot be split, so the plan stops at one state per pass and never 0.

    At that floor one pass carries `len(questions)` rows, which is exactly what one `/v1/systemone`
    request can already ask for -- so the batch route's worst pass is the single route's worst pass.
    """
    import laya.serve as serve_mod

    monkeypatch.delenv("LAYA_MAX_BATCH_TOKENS", raising=False)
    plan = serve_mod._batch_chunk_size(serve_mod.MAX_BATCH_STATES, serve_mod.MAX_QUESTIONS,
                                       serve_mod.DEFAULT_MAX_TOKEN_BUDGET)
    assert plan == 1, plan
    assert serve_mod._batch_chunk_size(1, serve_mod.MAX_QUESTIONS,
                                       serve_mod.DEFAULT_MAX_TOKEN_BUDGET) == 1


@pytest.mark.parametrize("states,questions", [(0, 4), (4, 0), (0, 0)])
def test_batch_chunk_plan_is_none_for_a_degenerate_shape(monkeypatch, states, questions):
    """Nothing to split, and no division by the question count."""
    import laya.serve as serve_mod

    monkeypatch.delenv("LAYA_MAX_BATCH_TOKENS", raising=False)
    assert serve_mod._batch_chunk_size(states, questions) is None


def test_batch_an_empty_questions_body_is_not_chunked(monkeypatch):
    """Zero rows: no plan, no crash, and `predict_batch` short-circuits on its own."""
    client, fake = _batch_env_client(monkeypatch)
    r = client.post("/v1/systemone/batch", json={"states": ["hi"] * 8, "questions": {}})
    assert r.status_code == 200, r.text
    assert "batch_size" not in fake.batch_calls[0][1]


def test_batch_a_non_object_questions_body_keeps_its_own_400(monkeypatch):
    """The planner must not turn a 400 into something else by calling len() on a list."""
    client, _ = _batch_env_client(monkeypatch)
    r = client.post("/v1/systemone/batch", json={"states": ["hi"] * 8, "questions": ["nope"] * 300})
    assert r.status_code == 400, r.text
    assert "'questions' must be an object" in r.json()["detail"], r.json()["detail"]


def test_batch_too_many_questions_still_gets_its_own_413(monkeypatch):
    """Chunking does not replace the per-question cap: 300 questions is still refused, by name."""
    import laya.serve as serve_mod

    client, _ = _batch_env_client(monkeypatch)
    r = client.post("/v1/systemone/batch",
                    json={"states": ["hi"], "questions": _noul_questions(serve_mod.MAX_QUESTIONS + 1)})
    assert r.status_code == 413, r.text
    assert "too many questions" in r.json()["detail"], r.json()["detail"]


def test_batch_individual_oversized_state_returns_413(monkeypatch):
    client, _ = _client(monkeypatch)
    from laya.serve import MAX_STATE_CHARS
    bad_req = {
        "states": ["ok", "X" * (MAX_STATE_CHARS + 10)],
        "questions": REQ["questions"],
    }
    r = client.post("/v1/systemone/batch", json=bad_req)
    assert r.status_code == 413
    assert "state too large" in r.json()["detail"]


def test_batch_auth_required_when_key_set(monkeypatch):
    client, _ = _client(monkeypatch, api_key="secret123")
    r = client.post("/v1/systemone/batch", json=BATCH_REQ)
    assert r.status_code == 401
    ok = client.post(
        "/v1/systemone/batch",
        json=BATCH_REQ,
        headers={"Authorization": "Bearer secret123"},
    )
    assert ok.status_code == 200


def test_batch_admission_bound_refuses_with_503(monkeypatch):
    """Batch endpoint must respect the admission bound and return 503 when busy."""
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
            for _ in range(200):
                if fake.entered.is_set():
                    break
                await asyncio.sleep(0.05)
            assert fake.entered.is_set(), "first request never reached inference"
            await asyncio.sleep(0.2)
            seen["batch_second"] = (await client.post("/v1/systemone/batch", json=BATCH_REQ)).status_code
            fake.release.set()
            seen["first"] = (await first).status_code

    asyncio.run(drive())

    assert seen["batch_second"] == 503, seen
    assert seen["first"] == 200, seen


class BatchCapableFakeRouter(FakeRouter):
    """FakeRouter that implements predict_batch."""

    def __init__(self):
        super().__init__()
        self.batch_calls = []

    def predict_batch(self, requests, batch_size=None):
        self.batch_calls.append(requests)
        return [
            {
                "model": "laya-rl-agent",
                "answers": {
                    "dept": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.94, "tech": 0.06},
                        "confidence": 0.94,
                    }
                },
                "usage": {"input_tokens": 42, "output_tokens": 0},
                "routing": {"model": "english", "reason": "English Latin text"},
            }
            for _ in requests
        ]


def test_batch_uses_predict_batch_when_available(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    fake = BatchCapableFakeRouter()
    client = TestClient(create_app(router=fake))
    r = client.post("/v1/systemone/batch", json=BATCH_REQ)
    assert r.status_code == 200, r.text
    assert len(fake.batch_calls) == 1
    assert len(fake.batch_calls[0]) == 2
    assert len(fake.calls) == 0  # Confirms no sequential predict() fallback calls were made
def test_an_unpaired_surrogate_is_a_caller_error_not_a_server_fault():
    r"""A `\udXXX` escape with no pair parses as JSON and then cannot be UTF-8 encoded.

    The tokenizer raised `TypeError` from inside `build_sequence`, and `serve` maps only
    `ValueError` to 422, so a malformed string in the client's own body came back as a 500
    "inference failed" -- a server fault plus an operator traceback for the caller's mistake.
    A *paired* surrogate is an ordinary astral character (an emoji) and must keep working.
    """
    seen = []

    class RecordingRouter:
        loaded = ["english"]

        def predict(self, state, questions, model=None):
            seen.append(state)
            return {"model": "stub", "answers": {}, "usage": {}}

    client = TestClient(create_app(router=RecordingRouter()), raise_server_exceptions=False)

    lone = [
        b'{"state":"\\ud800","questions":{"q":{"type":"noul","instructions":"x"}}}',
        b'{"state":"ok","questions":{"q":{"type":"noul","instructions":"\\udfff"}}}',
        b'{"state":"ok","questions":{"q":{"type":"choice","instructions":"x",'
        b'"criteria":["\\ud800","b"]}}}',
    ]
    for body in lone:
        res = client.post("/v1/systemone", content=body,
                          headers={"content-type": "application/json"})
        assert res.status_code == 400, (body, res.status_code, res.text)
        assert "unpaired surrogate" in res.text, res.text

    # a paired surrogate is one astral character by the time json.loads is done: it must reach
    # the router rather than be rejected
    seen.clear()
    ok = json.dumps({"state": "hi \U0001f600",
                     "questions": {"q": {"type": "noul", "instructions": "x"}}}).encode("utf-8")
    res = client.post("/v1/systemone", content=ok,
                      headers={"content-type": "application/json"})
    assert res.status_code == 200, (res.status_code, res.text)
    assert seen and "\U0001f600" in str(seen[0]), seen


def _lone_surrogate_bodies(batch):
    r"""Every slot 06462bf lists a `\udXXX` escape can hide in, shaped for the route asked about.

    Built from one table rather than copied from the single-route test above, because the point is
    that both routes see the same strings -- and a batch carries more than one state, so an escape
    in the second of them is enough to reach the tokenizer. Each body is otherwise a request that
    answers 200, so a refusal can only be about the escape.
    """
    plain = {"q": {"type": "noul", "instructions": "x"}}
    slots = {
        "state": {"states": ["ok", "\ud800"], "questions": plain} if batch
        else {"state": "\ud800", "questions": plain},
        "instructions": {"questions": {"q": {"type": "noul", "instructions": "\udfff"}}},
        "criteria label": {"questions": {"q": {"type": "choice", "instructions": "x",
                                               "criteria": {"\ud800": "a", "b": "c"}}}},
    }
    for label, body in slots.items():
        if label != "state":
            body["states" if batch else "state"] = ["ok"] if batch else "ok"
    return slots


def _surrogate_client():
    router = BatchCapableFakeRouter()
    return TestClient(create_app(router=router), raise_server_exceptions=False), router


@pytest.mark.parametrize("path,batch",
                         [("/v1/systemone", False), ("/v1/systemone/batch", True)],
                         ids=["single", "batch"])
@pytest.mark.parametrize("slot", sorted(_lone_surrogate_bodies(False)))
def test_both_decision_routes_refuse_the_escape_before_the_router_sees_it(path, batch, slot,
                                                                         monkeypatch):
    r"""`/v1/systemone/batch` tokenizes the same body and had no guard on it.

    06462bf turned a lone `\udXXX` escape into a `400` on the single route and recorded what it had
    not covered: "only `/v1/systemone` is checked". The batch route went on walking no guard, so the
    identical string reached the tokenizer there. Measured on the cached `english` checkpoint, CPU,
    before this change:

    ```
    slot              Router.predict        Router.predict_batch     HTTP single  HTTP batch
    state             TypeError             TypeError                400          200
    instructions      TypeError             TypeError                400          200
    criteria label    TypeError             TypeError                400          200
    paired emoji      ok (1 answers)        ok (1 answers)           200          200
    ```

    None of those batch `200`s mean the escape is harmless. The HTTP columns were measured with a
    stub router that does not tokenize -- the handler builds one request per state and hands them to
    `predict_batch`, which raises the `TypeError` above for all three slots on the real one, so each
    is a `500 inference failed` on a live server plus a `_log.exception` traceback per request. That
    is exactly the shape 06462bf was written to stop. So each slot is its own case, the message is
    read from `laya.serve` rather than retyped here, and the router is asserted to have seen
    nothing: the refusal has to land before the forward pass it exists to prevent.
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client, router = _surrogate_client()
    body = _lone_surrogate_bodies(batch)[slot]
    # `json.dumps` escapes the unpaired surrogate by default, which is what a client sending one of
    # these puts on the wire: pure ASCII that `json.loads` turns back into an unencodable character.
    res = client.post(path, content=json.dumps(body).encode("ascii"),
                      headers={"content-type": "application/json"})
    assert res.status_code == 400, (slot, res.status_code, res.text)
    assert _LONE_SURROGATE_DETAIL in res.text, (slot, res.text)
    assert not router.calls and not router.batch_calls, (
        "%s was refused after it reached inference: %r %r" % (slot, router.calls, router.batch_calls))


@pytest.mark.parametrize("path,batch",
                         [("/v1/systemone", False), ("/v1/systemone/batch", True)],
                         ids=["single", "batch"])
def test_an_emoji_still_reaches_both_decision_routes(path, batch, monkeypatch):
    """A *paired* surrogate is one astral character by the time the parser is done.

    The other half of the guard: a check keyed on surrogate code points rather than on
    unpairedness would reject every emoji a state contains, which is why 06462bf asserts it for the
    single route and it is asserted here for both.
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client, router = _surrogate_client()
    body = {"states": ["hi \U0001f600"]} if batch else {"state": "hi \U0001f600"}
    body["questions"] = {"q": {"type": "noul", "instructions": "x"}}
    res = client.post(path, content=json.dumps(body).encode("utf-8"),
                      headers={"content-type": "application/json"})
    assert res.status_code == 200, (res.status_code, res.text)
    states = ([r["state"] for r in router.batch_calls[-1]] if batch
              else [router.calls[-1]["state"]])
    assert any("\U0001f600" in str(s) for s in states), states


def test_a_deeply_nested_state_is_not_a_recursion_error():
    """`json.loads` accepts nesting far deeper than Python's recursion limit.

    A recursive walk over the parsed body therefore turned a body the parser handles into a
    `RecursionError`, i.e. one 500 replaced by another. The walk uses an explicit stack, so the
    depth the JSON parser accepts is the depth this handles.
    """
    class NestedRouter:
        loaded = ["english"]

        def predict(self, state, questions, model=None):
            return {"model": "stub", "answers": {}, "usage": {}}

    client = TestClient(create_app(router=NestedRouter()), raise_server_exceptions=False)
    questions = {"q": {"type": "noul", "instructions": "x"}}

    for depth in (200, 600, 1200):
        body = json.dumps({"state": ["x" * 3] * depth, "questions": questions}).encode("utf-8")
        res = client.post("/v1/systemone", content=body,
                          headers={"content-type": "application/json"})
        assert res.status_code == 200, (depth, res.status_code, res.text)
def test_inference_timing_headers_absent_on_error():
    """Inference timing headers must not be attached to error responses."""
    router = FakeRouter()
    client = TestClient(create_app(router=router))
    res = client.post("/v1/systemone", json={"state": "missing questions"})
    assert res.status_code == 400
    assert "Server-Timing" not in res.headers
    assert "X-Inference-Time-Ms" not in res.headers


# --------------------------------------------------------------------------- #
# The state-size gate has to measure the text that reaches the tokenizer
# --------------------------------------------------------------------------- #

# Two dict states whose `str()` and whose `serialize_state()` are different lengths, which is
# what the gate measured wrongly. The numbers are exact rather than approximate: if either one
# moves, the divergence these checks pin has moved with it.
_QUOTE_STATE = {"body": '"' * 49988}       # len(str()) == 50000, serialized == 99988
_ZWSP_STATE = {"body": "\u200b" * 8332}    # len(str()) == 50004, serialized == 8344


def _serialized_len(state):
    """The length `laya.common.serialize_state` produces, without importing torch."""
    return len(json.dumps(state, ensure_ascii=False))


def test_the_gate_counts_the_json_the_tokenizer_will_see(monkeypatch):
    """`repr` leaves a `"` inside a value one character; JSON escapes it to two.

    So a state sitting exactly on the 50000-character gate was tokenized as 99988 characters --
    the documented limit admitting very nearly twice what it says.
    """
    assert len(str(_QUOTE_STATE)) == MAX_STATE_CHARS   # exactly at a repr-measured gate
    assert _serialized_len(_QUOTE_STATE) == 99988      # what build_sequence would encode
    client, fake = _client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "state": _QUOTE_STATE})
    assert r.status_code == 413
    assert r.json()["detail"] == "state too large (99988 > %d chars)" % MAX_STATE_CHARS
    assert fake.calls == []                            # refused before inference, as intended


def test_a_state_a_sixth_of_the_limit_is_not_refused_for_its_repr(monkeypatch):
    """The same mismatch the other way: `repr` escapes a zero-width space to six characters.

    `ensure_ascii=False` writes the one character it is, so this state serializes to 8344 --
    and was answered "state too large (50004 > 50000 chars)".
    """
    assert len(str(_ZWSP_STATE)) == 50004              # over a repr-measured gate
    assert _serialized_len(_ZWSP_STATE) == 8344        # a sixth of the limit, once serialized
    client, fake = _client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "state": _ZWSP_STATE})
    assert r.status_code == 200
    assert fake.calls[0]["state"] == _ZWSP_STATE       # reached inference unchanged


# Deliberately states whose `repr` and whose JSON are the same length, so these four answer the
# same before and after the fix. They are here to pin what must NOT move: the limit still admits a
# state that sits exactly on it and still refuses one character more.
# The explicit ids are required, not cosmetic. pytest builds each case's node id out of the
# parameter value, and writes that node id into the PYTEST_CURRENT_TEST environment variable -- a
# 50 000-character state produces a 50 084-character id, and a Windows environment variable cannot
# exceed 32 767 characters, so every case errors at teardown with
# `ValueError: the environment variable is longer than 32767 characters`. It passes on Linux, which
# has no comparable per-variable limit. `b"[" * 100000` in this file carries an id for the same
# reason.
@pytest.mark.parametrize("state, expected", [
    # str() and JSON are both exactly 50000
    pytest.param({"body": "a" * 49988}, 200, id="dict-exactly-at-the-limit"),
    # both exactly one character over
    pytest.param({"body": "a" * 49989}, 413, id="dict-one-character-over"),
    # a string state is its own serialization
    pytest.param("A" * MAX_STATE_CHARS, 200, id="str-exactly-at-the-limit"),
    pytest.param("A" * (MAX_STATE_CHARS + 1), 413, id="str-one-character-over"),
])
def test_the_declared_limit_is_the_limit_for_ordinary_states(monkeypatch, request, state, expected):
    # Asserted here rather than left to the Windows job: pytest writes this node id into
    # PYTEST_CURRENT_TEST, where Windows caps a variable at 32 767 characters, so dropping the
    # explicit ids above turns every case in this test into a teardown error -- on Windows only.
    # This assertion fails on any platform, so the ids cannot be lost without CI saying so.
    assert len(request.node.nodeid) < 32767, \
        "node id is %d characters; give this parameter an explicit short id" % len(request.node.nodeid)
    client, _ = _client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "state": state})
    assert r.status_code == expected


def test_the_gate_agrees_with_the_function_that_feeds_the_tokenizer():
    """Pin serve.py's inline `json.dumps` to `serialize_state` itself.

    serve.py cannot import it -- `laya.common` imports torch at module level and
    `import laya.serve` must not (tests/test_lazy_import.py) -- so the mirror is held here
    instead. Looked up rather than imported at module scope so that a serve.py without the
    helper fails this one check instead of aborting the suite's import.
    """
    import laya.serve as serve_mod

    state_length = getattr(serve_mod, "_state_length", None)
    assert state_length is not None, "laya.serve no longer measures the state through _state_length"
    serialize_state = pytest.importorskip("laya.common").serialize_state
    for state in (_QUOTE_STATE, _ZWSP_STATE, "plain text", {}, [],
                  {"a": {"b": ["c", 1, 1.5, None, True]}},
                  ["\u4e2d\u6587", '"', "'", "\n", "\x00", "\U0001f600"],
                  # A non-BMP format character is the widest divergence a client can send: `repr`
                  # writes it as the ten characters `\U000e0001`, `ensure_ascii=False` as one.
                  {"tag": "\U000e0001" * 10},
                  {"d": "it's", "q": 'say "hi"'}):
        assert state_length(state) == len(serialize_state(state)), state


@pytest.fixture
def default_int_digits():
    """Pin CPython's int-to-str digit guard for the tests that rely on it.

    The guard defaults to 4300 but is operator-configurable (`PYTHONINTMAXSTRDIGITS`,
    `sys.set_int_max_str_digits`). With it switched off, `json.dumps(10 ** 4300)` succeeds and
    `json.loads` accepts a 4301-digit body, so these two tests fail while the code under test stays
    correct -- verified: `PYTHONINTMAXSTRDIGITS=0` turns both of them red on an unmodified tree. A
    test that depends on an operator's setting is measuring the environment, not the change.
    """
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(previous)


def test_an_unserializable_state_is_a_400_not_a_count_nothing_measured(default_int_digits):
    """The old `except Exception` answered 413 "state too large (50001 > 50000 chars)".

    That reported a length nothing had measured, for a state that may be a few bytes. These
    states are unreachable over HTTP -- `json.loads` builds only JSON types -- but they are
    what an in-process caller of this helper can pass, and on the old path a state
    `serialize_state` cannot render passed the gate and failed later as a 500.
    """
    from fastapi import HTTPException

    # 10 ** 4300 has 4301 digits, one past CPython's int-to-str limit, and is built by
    # arithmetic because `int("1" * 4301)` would hit that same limit while parsing.
    for state in ({1, 2}, 10 ** 4300, {"n": 10 ** 4300}, [10 ** 4300]):
        with pytest.raises(HTTPException) as exc:
            _check_request_limits(state, REQ["questions"])
        assert exc.value.status_code == 400
        # The whole detail, so a count nothing measured cannot creep back into it.
        assert exc.value.detail == "'state' must be JSON-serializable"


def test_an_oversized_integer_state_never_reaches_the_gate(monkeypatch, default_int_digits):
    """Why the 400 above is not an HTTP-reachable case: the parser refuses the body first."""
    client, fake = _client(monkeypatch)
    r = client.post("/v1/systemone",
                    content=b'{"state": ' + b"1" * 4301 + b', "questions": {}}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["detail"] == "request body must be valid JSON"
    assert fake.calls == []


# --- the lower-bound probe in front of the exact measurement ---------------------------------

def test_the_probe_refuses_an_oversized_value_without_serializing_the_state(monkeypatch):
    """An oversized state is refused without the 2 MiB dump whose result cannot change the verdict.

    This is the check that goes red if `_state_length_lower_bound_over` is removed: without it the gate
    still answers 413, so the *absence of the dump* is the only observable difference.
    """
    import laya.serve as serve_mod

    calls = []
    real_dumps = json.dumps

    def counting_dumps(obj, **kw):
        calls.append(kw)
        return real_dumps(obj, **kw)

    # Scoped to the module's own binding: `serve_mod.json` IS the stdlib module, so patching an
    # attribute on it would replace `json.dumps` for the whole process for the test's duration.
    monkeypatch.setattr(serve_mod, "json", SimpleNamespace(dumps=counting_dumps))
    # F2: a conversation LIST is this library's dominant state shape, and the probe's list branch
    # can be deleted with every other check still green unless one of these is list-shaped.
    for state in ({"body": "a" * (2 * 1024 * 1024)},
                  ["z" * 60000],
                  [{"role": "user", "content": "x" * 70000}],
                  ({"role": "user", "content": "x" * 70000},)):
        calls.clear()
        assert serve_mod._state_length(state) > serve_mod.MAX_STATE_CHARS, state.__class__.__name__
        assert calls == [], "a %s state was serialized although one value already exceeds the cap" \
            % type(state).__name__


def test_the_probe_verdict_always_matches_the_exact_measurement():
    """The probe may only report "provably over"; its verdict must never differ from the encoder's.

    An unsound probe would be a security bug in the same family as the one this PR fixes, so the two
    are compared across shapes placed deliberately on both sides of the limit -- including the
    escape-heavy values where `repr` and JSON diverge, which is what the gate exists for.
    """
    import laya.serve as serve_mod
    cap = serve_mod.MAX_STATE_CHARS
    shapes = []
    for ch in ("a", '"', "\u200b", "\U000e0001", "\n", "\\", "\u4e2d"):
        for n in (1, 100, cap // 2, cap - 1, cap, cap + 1, cap * 2):
            shapes.append({"body": ch * n})
            shapes.append([ch * n])
            shapes.append({"a": {"b": [ch * n]}})
    shapes += [
        {("k%05d" % i): "v" for i in range(3000)},
        {("k%05d" % i): "v" * 40 for i in range(3000)},
        [{"role": "user", "content": "hi"} for _ in range(4000)],
        {"n": 1, "f": 1.5, "t": True, "z": None},
        {},
        [],
    ]
    assert len(shapes) == 7 * 7 * 3 + 6, len(shapes)
    for state in shapes:
        exact = len(json.dumps(state, ensure_ascii=False))
        probed = serve_mod._state_length(state)
        assert (probed > cap) == (exact > cap), \
            "probe and encoder disagree for a %d-character state" % exact
        if exact <= cap:
            # Stronger than agreeing on the verdict: under the cap the *exact* length is returned,
            # so the 413 message keeps quoting a number something actually measured.
            assert probed == exact


def test_the_early_refusal_reports_a_number_it_measured():
    """The 413 must not invent a count. Reporting `cap + 1` would answer "50001 > 50000" for a
    60 012-character state -- the same fabricated count this gate was written to remove.
    """
    import laya.serve as serve_mod
    cap = serve_mod.MAX_STATE_CHARS
    # `cap + 1` is NOT the tell: {"body": "a" * 50001} legitimately sums to exactly that. The
    # property that distinguishes a measured bound from a fabricated one is `cap < reported <= exact`.
    for state in ({"body": "x" * 60000}, {"body": "y" * (2 * 1024 * 1024)}, {"a": ["z" * 80000]},
                  {"body": "a" * (cap + 1)}, ["w" * 60000]):
        exact = len(json.dumps(state, ensure_ascii=False))
        reported = serve_mod._state_length(state)
        assert reported > cap, reported
        assert reported <= exact, \
            "reported %d for a state of %d characters -- a 413 must never overstate" % (reported, exact)


def test_the_probe_is_bounded_and_falls_through_rather_than_guessing():
    """The probe examines a constant number of values, so it can never become the expensive step.

    A state of 200 000 small values is not provably over within that budget, and must fall through
    to the encoder rather than be refused on a partial sum.
    """
    import laya.serve as serve_mod
    assert serve_mod._STATE_PROBE_VALUES <= 256
    wide = {("k%06d" % i): "x" for i in range(200000)}
    assert serve_mod._state_length_lower_bound_over(wide, serve_mod.MAX_STATE_CHARS) == 0
    assert serve_mod._state_length(wide) == len(json.dumps(wide, ensure_ascii=False))


def test_the_probe_walks_exact_container_types_only():
    """A subclass may override `values()` while `json.dumps` reads the real items, so walking one
    would let the "lower bound" exceed the true length. Measured before this was narrowed: a
    13-character state refused as `60000 > 50000`. No HTTP request can reach it -- `json.loads`
    builds exact types -- but the 400 branch exists for in-process callers.
    """
    import laya.serve as serve_mod

    class Lying(dict):
        def values(self):
            return ["x" * 60000]

    state = Lying(x="tiny")
    exact = len(json.dumps(state, ensure_ascii=False))
    assert serve_mod._state_length_lower_bound_over(state, serve_mod.MAX_STATE_CHARS) == 0
    assert serve_mod._state_length(state) == exact == 13

# --------------------------------------------------------------------- per-call controls
#
# `Router.predict` takes nine arguments beyond `state` and `questions` that a JSON body could
# state. `laya/serve.py` splits them into the two lists the tests below read, and the first test
# pins those lists against core's own signature: a control added to `predict` cannot be silently
# ignored by the HTTP surface, because it has to be placed on one side of that line first.

NEW_CONTROLS = ("task", "lang", "lang_guess", "min_confidence")
NEW_CONTROL_VALUES = {"task": "typed-decisions", "lang": "de", "lang_guess": "de",
                      "min_confidence": 0.9}


def test_every_predict_control_is_forwarded_or_refused():
    """`BODY_CONTROLS` and `BODY_REFUSALS` must cover `Router.predict` exactly, both directions."""
    from laya.router import Router

    taken = set(inspect.signature(Router.predict).parameters) - {"self", "state", "questions"}
    declared = set(BODY_CONTROLS) | set(BODY_REFUSALS)
    assert not set(BODY_CONTROLS) & set(BODY_REFUSALS)
    assert taken == declared, "predict() takes %s; serve declares %s" % (
        sorted(taken), sorted(declared))


@pytest.mark.parametrize("key", NEW_CONTROLS)
def test_each_control_reaches_predict(monkeypatch, key):
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, key: NEW_CONTROL_VALUES[key]})
    assert r.status_code == 200, r.text
    assert fake.calls[0][key] == NEW_CONTROL_VALUES[key]


def test_unset_controls_are_not_sent_as_none(monkeypatch):
    """An absent control must stay absent rather than be forwarded as `None`.

    Core reads `None` as "inherit what the Router was built with", so sending `lang_guess=None`
    would override a deployment's `Router(lang_guess=...)`. It would also 500 on any router whose
    `predict` does not take the keyword -- the `FakeRouter` this suite has always used.
    """
    client, fake = _budget_client(monkeypatch)
    assert client.post("/v1/systemone", json=REQ).status_code == 200
    assert not [key for key in NEW_CONTROLS if key in fake.calls[0]]


@pytest.mark.parametrize("key", NEW_CONTROLS)
def test_explicit_null_is_no_control(monkeypatch, key):
    client, fake = _budget_client(monkeypatch)
    assert client.post("/v1/systemone", json={**REQ, key: None}).status_code == 200
    assert not [name for name in NEW_CONTROLS if name in fake.calls[0]]


def test_min_confidence_zero_is_still_a_threshold(monkeypatch):
    """`0.0` is falsy but is a value the caller chose; `if min_confidence:` would drop it."""
    client, fake = _budget_client(monkeypatch)
    assert client.post("/v1/systemone", json={**REQ, "min_confidence": 0.0}).status_code == 200
    assert fake.calls[0]["min_confidence"] == 0.0


@pytest.mark.parametrize("key", ["lang", "lang_guess"])
@pytest.mark.parametrize("value", [True, 5, ["de"], {"code": "de"}])
def test_a_language_control_must_be_a_code_string(monkeypatch, key, value):
    """`Router` stringifies a language hint, so `true` becomes the code `"true"`.

    That is a real, non-English code, so the JSON boolean would decide the checkpoint. A callable
    hint -- core's other accepted form -- cannot cross an HTTP body either, which leaves the code
    string as the only form to accept here.
    """
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, key: value})
    assert r.status_code == 422, r.text
    assert "must be a language code string" in r.json()["detail"]
    assert not fake.calls, "a refused request must not run inference"


@pytest.mark.parametrize("value", [1.5, -0.1, True, "0.9"])
def test_min_confidence_is_validated_by_core(monkeypatch, value):
    """The bounds belong to `laya.confidence`, so serve must report core's own message.

    Comparing against the `ValueError` core raises for the same value is what keeps the accepted
    range from drifting: a restated `[0.0, 1.0]` check here would pass CI on the day core widened
    it and start rejecting requests the abstention gate would have honoured.
    """
    from laya.confidence import check_min_confidence

    try:
        check_min_confidence(value)
    except ValueError as error:
        expected = str(error)
    else:
        raise AssertionError("core accepted %r; this probe needs a rejecting value" % (value,))

    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "min_confidence": value})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == expected
    assert not fake.calls


def test_an_infinite_threshold_is_refused_by_core(monkeypatch):
    """JSON writes no NaN, but a number literal can parse to infinity.

    `check_min_confidence` tests `math.isfinite` for exactly this. A range check written here would
    happen to reject it too, and then keep passing on the day core's rule changed -- so the request
    is sent as raw bytes: Python cannot serialise an infinity back to JSON, and `1e999` is what a
    client that never round-tripped through Python would actually put on the wire.
    """
    client, fake = _budget_client(monkeypatch)
    raw = (json.dumps(REQ)[:-1] + ', "min_confidence": 1e999}').encode("utf-8")
    r = client.post("/v1/systemone", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    assert "min_confidence must be a float in [0.0, 1.0]" in r.json()["detail"]
    assert not fake.calls


def test_an_unknown_task_gets_cores_answer_not_a_500(monkeypatch):
    """`task` is forwarded verbatim, as the CLI forwards `--task`, and core names the valid set.

    `route()` normalises a task through `normalise_name`, whose `ValueError` the existing mapping
    reports as a 422 carrying its message. Listing the accepted tasks in `serve` instead would be
    a second registry to keep in step -- the failure mode #544 and #638 removed elsewhere.
    """
    from laya.router import normalise_name

    class RoutingRouter(BudgetRouter):
        def predict(self, state, questions, model=None, **kwargs):
            if "task" in kwargs:
                normalise_name(kwargs["task"])  # core's guard, called exactly as route() calls it
            return super().predict(state, questions, model=model, **kwargs)

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=RoutingRouter()), raise_server_exceptions=False)

    r = client.post("/v1/systemone", json={**REQ, "task": "not-a-task"})
    assert r.status_code == 422, r.text
    assert "unknown model" in r.json()["detail"]

    ok = client.post("/v1/systemone", json={**REQ, "task": "typed-decisions"})
    assert ok.status_code == 200, ok.text


@pytest.mark.parametrize("key", BODY_REFUSALS)
def test_a_hook_control_is_refused_not_dropped(monkeypatch, key):
    """All five used to be read into the body and ignored, so a client got a silent no.

    `laya.integrations.langchain::_reject_remote_hooks` already refuses the same five on a node
    with a `base_url`, on the grounds that a hook runs in the server's process. The endpoint now
    says that itself instead of answering as though the control had been honoured.
    """
    value = {"hooks": [], "on_predict_start": "cache", "on_predict_end": "audit",
             "hooks_raise": False, "hooks_timeout": 5.0}[key]
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, key: value})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert key in detail
    assert "cannot be sent to this endpoint" in detail
    assert not fake.calls, "the refusal must come before inference"


def test_a_refusal_names_every_hook_control_sent(monkeypatch):
    client, fake = _budget_client(monkeypatch)
    r = client.post("/v1/systemone", json={**REQ, "hooks_raise": True, "hooks_timeout": 5.0})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "hooks_raise" in detail and "hooks_timeout" in detail
    assert not fake.calls


@pytest.mark.parametrize("key", BODY_REFUSALS)
def test_a_null_hook_control_is_not_a_refusal(monkeypatch, key):
    """`{"hooks": null}` means "no per-call hooks", which is what core reads as inherit.

    A Jev client that serialises its absent fields must keep working; only a value says the caller
    asked for something this endpoint cannot do.
    """
    client, fake = _budget_client(monkeypatch)
    assert client.post("/v1/systemone", json={**REQ, key: None}).status_code == 200


def test_http_api_page_documents_exactly_the_forwarded_controls():
    """The request-body table and the code must not drift apart in either direction.

    Read back out of the markdown rather than substring-matched: a row this parse cannot see is a
    field the documentation stopped describing, and a documented field with no code behind it is
    the same lie in the other direction. The five refusals have to be named where the page says
    why they are refused, so the reason travels with the field list.
    """
    path = os.path.join(ROOT, "docs", "http-api.md")
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    header = lines.index("| field | required | meaning |")
    rows = []
    for line in lines[header + 2:]:
        if not line.strip():
            break
        rows.append(line.split("|")[1].strip().strip("`"))
    documented = set(rows) - {"state", "questions"}
    assert documented == set(BODY_CONTROLS), "table says %s, serve forwards %s" % (
        sorted(documented), sorted(BODY_CONTROLS))

    page = "\n".join(lines)
    prose = page[page.index("`model`, `task`, `lang`, `lang_guess`"):]
    for key in BODY_REFUSALS:
        assert "`%s`" % key in prose, "%s is not named where the refusal is explained" % key


# --------------------------------------------------------------------- batch endpoint per-call controls
#
# `Router.predict_batch` reads two kinds of control: call-level kwargs that apply to the whole
# batch (chunking, padding, abstention) and per-request keys it lifts off each item of ``requests``
# (checkpoint, task, language, token budget). Before this PR, the HTTP batch endpoint read neither
# -- it always synthesized a plain ``{state, questions, model}`` dict and called ``predict_batch``
# with no kwargs, so a caller could not send any of the controls the single endpoint already
# forwards. The suite below pins the two split lists against core's own signature, verifies every
# control reaches the right side of the Router call, and refuses a hook control the same way
# ``/v1/systemone`` already does.


class BatchRecordingRouter(FakeRouter):
    """Records ``predict_batch``'s full call: the requests list and every call kwarg."""

    def __init__(self):
        super().__init__()
        self.batch_calls = []

    def predict_batch(self, requests, **kwargs):
        self.batch_calls.append(([dict(r) for r in requests], dict(kwargs)))
        return [
            {
                "model": "laya-rl-agent",
                "answers": {
                    "dept": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.94, "tech": 0.06},
                        "confidence": 0.94,
                    }
                },
                "usage": {"input_tokens": 42, "output_tokens": 0},
                "routing": {"model": "english", "reason": "English Latin text"},
            }
            for _ in requests
        ]


class StrictBatchRouter(BatchRecordingRouter):
    """``predict_batch(self, requests)`` -- takes no call kwargs.

    Witness against an always-forward mutation: if the handler unconditionally passed ``batch_size``
    or ``min_confidence``, this Router would raise ``TypeError`` and 500 on a bare body.
    """

    def predict_batch(self, requests):  # noqa: D401 -- strict on purpose
        return super().predict_batch(requests)


BATCH_CALL_VALUES = {"batch_size": 4, "min_confidence": 0.9, "sort_by_length": True}
BATCH_ITEM_VALUES = {"max_len": 64, "head_max_len": 32, "task": "typed-decisions",
                     "lang": "de", "lang_guess": "de"}


def _batch_client(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    fake = BatchRecordingRouter()
    return TestClient(create_app(router=fake)), fake


def test_batch_call_controls_are_exactly_predict_batch_kwargs_or_refusal():
    """A control added to ``Router.predict_batch`` must be placed on one side of the line.

    Either serve forwards it (``BATCH_BODY_CALL_CONTROLS``) or it refuses it (``hooks_timeout``,
    the one hook-execution arg that ``predict_batch`` takes). Silently ignoring a new kwarg is the
    failure mode this pin exists to catch: the same reasoning that made ``/v1/systemone`` refuse
    ``hooks_timeout`` in #724 applies to the batch path -- a caller cannot shorten the deadline
    on which the operator's own hooks execute.
    """
    from laya.router import Router

    taken = set(inspect.signature(Router.predict_batch).parameters) - {"self", "requests"}
    declared = set(BATCH_BODY_CALL_CONTROLS) | set(BODY_REFUSALS)
    assert taken == declared, "predict_batch() takes %s; serve declares %s" % (
        sorted(taken), sorted(declared))


def test_batch_controls_cover_the_single_endpoints_set_plus_batch_only():
    """Everything the single endpoint forwards must reach the batch path too, and vice versa.

    The batch forwards the same per-call controls ``/v1/systemone`` does -- split across the item
    dict (``Router.predict_batch`` reads them per item) and the call kwargs (min_confidence) --
    plus the batch-only chunking and padding knobs (``batch_size``, ``sort_by_length``) that
    ``predict_batch`` accepts and ``predict`` does not. ``model`` still goes into every item, as
    it has since the batch endpoint shipped. If a control moves across either boundary, this
    test names it.
    """
    batch_side = set(BATCH_BODY_ITEM_CONTROLS) | set(BATCH_BODY_CALL_CONTROLS)
    single_side = (set(BODY_CONTROLS) | {"batch_size", "sort_by_length"}) - {"model"}
    assert batch_side == single_side, (
        "batch forwards %s; single endpoint + batch-only says %s"
        % (sorted(batch_side), sorted(single_side)))


@pytest.mark.parametrize("key", sorted(BATCH_ITEM_VALUES))
def test_each_batch_item_control_reaches_every_request(monkeypatch, key):
    client, fake = _batch_client(monkeypatch)
    r = client.post("/v1/systemone/batch", json={**BATCH_REQ, key: BATCH_ITEM_VALUES[key]})
    assert r.status_code == 200, r.text
    requests, _ = fake.batch_calls[0]
    assert len(requests) == len(BATCH_REQ["states"])
    for item in requests:
        assert item.get(key) == BATCH_ITEM_VALUES[key], (key, item)


@pytest.mark.parametrize("key", sorted(BATCH_CALL_VALUES))
def test_each_batch_call_control_reaches_predict_batch_kwargs(monkeypatch, key):
    client, fake = _batch_client(monkeypatch)
    r = client.post("/v1/systemone/batch", json={**BATCH_REQ, key: BATCH_CALL_VALUES[key]})
    assert r.status_code == 200, r.text
    _, kwargs = fake.batch_calls[0]
    assert kwargs.get(key) == BATCH_CALL_VALUES[key], (key, kwargs)


def test_batch_unset_controls_are_not_sent_as_none(monkeypatch):
    """An absent control must stay absent in either the item dict or the call kwargs.

    Core reads an absent argument as "inherit what the Router was built with", so sending
    ``min_confidence=None`` would override a deployment's ``Router(min_confidence=...)``, and
    sending a per-item key with value ``None`` would break a Router whose ``predict_batch``
    predates the key (#294 shape).
    """
    client, fake = _batch_client(monkeypatch)
    assert client.post("/v1/systemone/batch", json=BATCH_REQ).status_code == 200
    requests, kwargs = fake.batch_calls[0]
    assert kwargs == {}
    for item in requests:
        for key in BATCH_BODY_ITEM_CONTROLS:
            assert key not in item, (key, item)


@pytest.mark.parametrize("key", sorted(set(BATCH_ITEM_VALUES) | set(BATCH_CALL_VALUES)))
def test_batch_explicit_null_is_no_control(monkeypatch, key):
    """``{"lang_guess": null}`` means "no hint" -- same shape the single endpoint honours.

    A Jev client that serialises its absent fields must keep working; only a value says the
    caller asked for something.
    """
    client, fake = _batch_client(monkeypatch)
    assert client.post("/v1/systemone/batch", json={**BATCH_REQ, key: None}).status_code == 200
    requests, kwargs = fake.batch_calls[0]
    assert "min_confidence" not in kwargs and "batch_size" not in kwargs
    assert "sort_by_length" not in kwargs
    for item in requests:
        assert key not in item


def test_batch_min_confidence_zero_is_still_a_threshold(monkeypatch):
    """``0.0`` is falsy but is a value the caller chose; ``if min_confidence:`` would drop it."""
    client, fake = _batch_client(monkeypatch)
    assert client.post("/v1/systemone/batch",
                       json={**BATCH_REQ, "min_confidence": 0.0}).status_code == 200
    _, kwargs = fake.batch_calls[0]
    assert kwargs["min_confidence"] == 0.0


def test_batch_sort_by_length_false_is_not_forwarded(monkeypatch):
    """``predict_batch``'s own default is ``False``, so ``False`` is what the caller did NOT ask.

    Forwarding ``False`` explicitly would still work on ``Router`` but silently disappear on any
    attached agent whose ``predict_batch`` predates the knob (#294) -- which is why the MCP tool
    makes the same choice. The endpoint does accept the value; it just does not translate it into
    a kwarg.
    """
    client, fake = _batch_client(monkeypatch)
    assert client.post("/v1/systemone/batch",
                       json={**BATCH_REQ, "sort_by_length": False}).status_code == 200
    _, kwargs = fake.batch_calls[0]
    assert "sort_by_length" not in kwargs


def test_batch_strict_predict_batch_router_answers_a_quiet_body(monkeypatch):
    """An attached Router whose ``predict_batch(self, requests)`` takes nothing else must still work.

    The endpoint forwards kwargs only when the caller asks. If the handler always passed
    ``batch_size=None`` (or any call kwarg) unconditionally, this Router would raise ``TypeError``
    and the endpoint would 500 on a body that sent nothing -- which is exactly the mutation this
    witness kills.
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    fake = StrictBatchRouter()
    client = TestClient(create_app(router=fake))
    r = client.post("/v1/systemone/batch", json=BATCH_REQ)
    assert r.status_code == 200, r.text
    assert len(fake.batch_calls) == 1
    _, kwargs = fake.batch_calls[0]
    assert kwargs == {}


@pytest.mark.parametrize("key,value", [
    ("lang", True),
    ("lang", 5),
    ("lang_guess", ["de"]),
    ("lang_guess", {"code": "de"}),
    ("max_len", 0),
    ("max_len", -1),
    ("max_len", True),
    ("max_len", "64"),
    ("head_max_len", "32"),
    ("min_confidence", 1.5),
    ("min_confidence", -0.1),
    ("min_confidence", "0.9"),
    ("batch_size", 0),
    ("batch_size", -3),
    ("batch_size", True),
    ("batch_size", 1.5),
    ("batch_size", "4"),
    ("sort_by_length", "yes"),
    ("sort_by_length", 1),
])
def test_bad_control_on_batch_is_refused_before_inference(monkeypatch, key, value):
    client, fake = _batch_client(monkeypatch)
    r = client.post("/v1/systemone/batch", json={**BATCH_REQ, key: value})
    assert r.status_code == 422, (key, value, r.text)
    assert not fake.batch_calls, "a refused body must not reach predict_batch"


@pytest.mark.parametrize("key", BODY_REFUSALS)
def test_batch_hook_control_is_refused_not_dropped(monkeypatch, key):
    """The batch endpoint must honour the same refusal policy the single endpoint already states.

    Without ``_refuse_body_refusals``, a batch body would silently hand a caller a shorter
    deadline for the operator's hooks -- ``predict_batch`` *does* take ``hooks_timeout`` as a
    call arg, so this is not a theoretical refusal here. The other four are refused for the same
    reason they are on ``/v1/systemone``: they run inside the server process.
    """
    value = {"hooks": [], "on_predict_start": "cache", "on_predict_end": "audit",
             "hooks_raise": False, "hooks_timeout": 5.0}[key]
    client, fake = _batch_client(monkeypatch)
    r = client.post("/v1/systemone/batch", json={**BATCH_REQ, key: value})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert key in detail
    assert "cannot be sent to this endpoint" in detail
    assert not fake.batch_calls, "the refusal must come before inference"


def test_batch_predict_fallback_forwards_every_control(monkeypatch):
    """When the attached Router has no ``predict_batch``, every control must reach ``predict()``.

    The batch path synthesizes per-state ``predict()`` calls with the same kwargs the single
    endpoint uses -- ``max_len``/``head_max_len``/``task``/``lang``/``lang_guess``/``min_confidence``
    all apply per call. A caller that sends ``min_confidence`` on a batch body must see the
    abstention gate apply per item on the fallback path, not silently skip it because
    ``predict_batch`` was unavailable.
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    fake = BudgetRouter()
    client = TestClient(create_app(router=fake))
    body = {**BATCH_REQ, "max_len": 64, "head_max_len": 32, "task": "typed-decisions",
            "lang": "de", "lang_guess": "de", "min_confidence": 0.5}
    r = client.post("/v1/systemone/batch", json=body)
    assert r.status_code == 200, r.text
    assert len(fake.calls) == len(BATCH_REQ["states"])
    for call in fake.calls:
        assert call["max_len"] == 64
        assert call["head_max_len"] == 32
        assert call["task"] == "typed-decisions"
        assert call["lang"] == "de"
        assert call["lang_guess"] == "de"
        assert call["min_confidence"] == 0.5


def test_batch_predict_fallback_sends_no_kwargs_when_body_is_quiet(monkeypatch):
    """A quiet batch body on the fallback path must call ``predict()`` with no kwargs at all.

    An attached Router whose ``predict()`` does not take the new keywords -- the ``FakeRouter``
    class this suite has always used -- must keep answering on a bare batch. Passing ``None``
    would 500 on that shape and would override a deployment's own ``Router(lang_guess=...)``
    on a modern one.
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    fake = FakeRouter()
    client = TestClient(create_app(router=fake))
    r = client.post("/v1/systemone/batch", json=BATCH_REQ)
    assert r.status_code == 200, r.text
    assert len(fake.calls) == len(BATCH_REQ["states"])


def test_batch_controls_combine_on_one_call(monkeypatch):
    """Call kwargs and item overrides land on the same call, not in isolation.

    A caller that sends a token budget and a chunking cap together must see them on the same
    ``predict_batch`` invocation -- not the budget on the item and the chunk cap dropped, and
    not the reverse. This is the whole-point-of-the-PR shape.
    """
    client, fake = _batch_client(monkeypatch)
    r = client.post("/v1/systemone/batch",
                    json={**BATCH_REQ, "batch_size": 2, "sort_by_length": True,
                          "min_confidence": 0.8, "task": "typed-decisions", "max_len": 128})
    assert r.status_code == 200, r.text
    requests, kwargs = fake.batch_calls[0]
    assert kwargs == {"min_confidence": 0.8, "batch_size": 2, "sort_by_length": True}
    for item in requests:
        assert item["task"] == "typed-decisions"
        assert item["max_len"] == 128
def test_health_liveness_is_open_but_the_detail_needs_the_bearer(monkeypatch):
    """#812: `/health` answered deployment internals to an unauthenticated caller.

    `POST /v1/systemone` was gated and `GET /health` was not, so on a server the operator had
    locked down with `LAYA_API_KEY` anyone could read the resident checkpoint names, each one's
    exact Hugging Face revision SHA, the device state, and `last_fallback_reason`, which quotes
    host hardware ("GPU 0 total 8.00 GiB"). Reconnaissance rather than a data path, but it is
    exactly the inventory you would want before targeting a revision.

    Requiring the bearer outright was not the fix: `compose.http.yaml`'s healthcheck, the Docker
    HEALTHCHECK and any k8s liveness probe all read this endpoint with no credential, and
    docs/http-api.md promises it is always open. So liveness stays open and the detail does not.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LAYA_API_KEY", "s3cret")
    client = TestClient(create_app(router=FakeRouter()))

    # a probe with no credential still gets its 200, which is all a healthcheck reads
    anonymous = client.get("/health")
    assert anonymous.status_code == 200
    assert anonymous.json() == {"status": "ok"}

    # and the same for a wrong bearer: still live, still no detail, and not a 401, because a
    # probe that starts failing on a bad credential is a worse outage than the disclosure
    wrong = client.get("/health", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 200
    assert wrong.json() == {"status": "ok"}

    # the detail is the authorized caller's
    full = client.get("/health", headers={"Authorization": "Bearer s3cret"})
    assert full.status_code == 200
    _, returned = _health_return_keys()
    assert sorted(full.json()) == sorted(returned)
    for leaked in ("loaded", "revisions", "cpu_fallbacks", "checkpoint_devices"):
        assert leaked in full.json()
        assert leaked not in anonymous.json()


def test_health_without_an_api_key_is_unchanged(monkeypatch):
    """A deployment that set no key never asked to be gated, so it gets the whole payload."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.delenv("LAYA_IDLE_UNLOAD_SECONDS", raising=False)
    client = TestClient(create_app(router=FakeRouter()))
    _, returned = _health_return_keys()
    assert sorted(client.get("/health").json()) == sorted(returned)


def _health_return_keys():
    """The keys `health()` hands back, read out of its own `return` statement."""
    path = os.path.join(ROOT, "laya", "serve.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "health":
            for stmt in node.body:
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                    return node, [k.value for k in stmt.value.keys
                                  if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    raise AssertionError("no `def health()` returning a dict literal in laya/serve.py")


def test_http_api_page_documents_exactly_the_health_fields():
    """The payload `/health` documents has to be the payload the handler builds.

    The page showed four keys while the handler returned seven: `device_is_preference`,
    `checkpoint_devices` and `cpu_fallbacks` arrived with the device-fact reporting and #574's
    fallback counters, and both were written up in docs/docker.md -- the page a reader opens before
    pointing a probe at a server was the one page nobody updated. Its sample also printed
    `"device": "auto"`, which is what `LAYA_DEVICE` defaults to, not a value the handler can return.

    So the key list is read out of `health()`'s `return` rather than typed here, in both directions
    like the request-body gate above, and the sample is checked for internal coherence too: it is one
    server's answer, so every per-checkpoint block is keyed by the names in `loaded`, and
    `device_is_preference` and `device` follow the two rules the handler states.
    """
    health, returned = _health_return_keys()
    assert returned, "health() returns no literal keys; retarget this"

    page = open(os.path.join(ROOT, "docs", "http-api.md"), encoding="utf-8").read()
    section = page[page.index("### `GET /health`"):page.index("### `POST")]
    sample = json.loads(section.split("```json", 1)[1].split("```", 1)[0])

    assert sorted(sample) == sorted(returned), "the sample says %s, health() returns %s" % (
        sorted(sample), sorted(returned))
    for key in returned:
        assert "\n- `%s` " % key in section, (
            "%s is in the sample with no bullet defining it, so the field is listed and not "
            "explained" % key)

    # The per-checkpoint blocks all key on the same resident checkpoints. A fourth dict-valued key
    # means the handler grew one and this list has to grow with it, deliberately.
    resident = set(sample["loaded"])
    assert resident, "the sample shows no resident checkpoint, so the blocks below prove nothing"
    blocks = [k for k, v in sample.items() if isinstance(v, dict) and v]
    assert len(blocks) == 3, "dict-valued blocks are %s; retarget this if the payload grew" % blocks
    for block in blocks:
        assert set(sample[block]) == resident, "%s is keyed %r, loaded says %r" % (
            block, sorted(sample[block]), sorted(resident))

    # `device` is a device label, not the configuration word: the labels come from the only function
    # that can produce one when nothing is resident.
    dev_path = os.path.join(ROOT, "laya", "mcp", "device.py")
    with open(dev_path, encoding="utf-8") as handle:
        dev_tree = ast.parse(handle.read(), filename=dev_path)
    labels = set()
    for node in ast.walk(dev_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_device":
            labels = {stmt.value.value for stmt in ast.walk(node)
                      if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant)
                      and isinstance(stmt.value.value, str)}
    assert labels, "resolve_device() returns no literal device label; retarget this"
    assert sample["device"] in labels, "%r is not a device label resolve_device can return (%s)" % (
        sample["device"], sorted(labels))

    # The handler's two rules about those fields: the preference flag is true exactly while nothing
    # is resident, and the top-level answer is the first resident checkpoint's own device.
    assert sample["device_is_preference"] == (not sample["checkpoint_devices"]), (
        "device_is_preference says %r with checkpoint_devices %r" % (
            sample["device_is_preference"], sample["checkpoint_devices"]))
    if sample["checkpoint_devices"]:
        assert sample["device"] == sample["checkpoint_devices"][sample["loaded"][0]], (
            "device must be the first resident checkpoint's device, as serve.py computes it")

    # And the shape of a fallback entry, which no page has ever spelled out: read from the dict the
    # handler builds per checkpoint.
    counters = None
    for node in ast.walk(health):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript) \
                and isinstance(node.value, ast.Dict):
            counters = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    assert counters, "health() builds no per-checkpoint dict literal; retarget this"
    for name in resident:
        assert sorted(sample["cpu_fallbacks"][name]) == sorted(counters), (
            "cpu_fallbacks entries say %s, the handler builds %s" % (
                sorted(sample["cpu_fallbacks"][name]), sorted(counters)))


def _decision_response_site(rel):
    """What one Agent builds the decision response out of, read from its own literals.

    Returns the result dict's keys, the keys its `usage` block always carries, the keys it adds to
    `usage` only under a condition, and the `model` constant it stamps. Read out of the source rather
    than transcribed, because the point of this gate is that the page and both agents describe the
    same payload; a hand-copied list would be a third copy to keep in step.
    """
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        keys = usage = head = None
        optional = set()
        for stmt in ast.walk(fn):
            if not isinstance(stmt, ast.Assign) or not stmt.targets:
                continue
            first = stmt.targets[0]
            names = ([k.value for k in stmt.value.keys
                      if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                     if isinstance(stmt.value, ast.Dict) else [])
            if isinstance(first, ast.Name) and first.id == "usage" and "state_tokens" in names:
                usage = names
            elif isinstance(first, ast.Subscript) and isinstance(first.value, ast.Name):
                if first.value.id == "usage" and isinstance(first.slice, ast.Constant):
                    optional.add(first.slice.value)
                elif first.value.id == "window_results" and {"answers", "model", "usage"} <= set(names):
                    keys = names
                    head = next((v.value for k, v in zip(stmt.value.keys, stmt.value.values)
                                 if isinstance(k, ast.Constant) and k.value == "model"
                                 and isinstance(v, ast.Constant)), None)
        if usage:
            assert keys, "%s: %s builds a usage block in no result dict literal" % (rel, fn.name)
            assert head, "%s: %s's result dict has no literal `model` constant" % (rel, fn.name)
            return {"keys": sorted(keys), "usage": sorted(usage),
                    "optional": sorted(optional), "head": head}
    raise AssertionError("%s: no `usage = {...}` literal with a `state_tokens` key" % rel)


def _route_decision_keys():
    """Every keyword some `RouteDecision(...)` is built with, across all of `_route`'s branches."""
    path = os.path.join(ROOT, "laya", "router.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RouteDecision":
            keys.update(kw.arg for kw in node.keywords if kw.arg)
    assert keys, "no RouteDecision(...) call with keywords in laya/router.py; retarget this"
    return sorted(keys)


def _md_table_keys(page, header):
    """The first column of a markdown table, read as rows rather than searched for substrings."""
    lines = page.splitlines()
    start = lines.index(header)
    rows = []
    for line in lines[start + 2:]:
        if not line.strip():
            break
        rows.append(line.split("|")[1].strip().strip("`"))
    return rows


def test_http_api_page_documents_the_decision_response_keys():
    r"""The page documents two of the six `usage` keys the agents actually send.

    `### Response` showed `"usage": {"input_tokens": 74, "output_tokens": 0}`, while
    `Agent.predict_batch` builds six keys and `OnnxAgent._infer_batch` builds the same six -- the
    truncation report #174 asked for, and the only place a caller can see that the state it sent was
    cut before the model read it. It showed three of the four `routing` keys (`workflow` is on every
    branch of `_route`) and four of the eight `laya.lang.analyse()` returns. And it named a
    `lang_guess` key of `routing` that no code path has ever set: `lang_guess` is a *request* control
    (`BODY_CONTROLS`), and the evidence a hint acted on is spelled out in `reason`.

    ```
    usage   documented 2  <-  agent 6 always + options when options collapse, onnx the same
    routing documented 4  <-  RouteDecision 5: model, repo, reason, detection, workflow
    detection documented 4  <-  analyse() 8
    lang_guess   documented 1  <-  set by 0 branches
    ```

    So every key set below is read out of the source -- the result and usage literals, the
    `RouteDecision(...)` keywords, `analyse()` at runtime, which needs no checkpoint -- and held to
    the page in both directions, as the request-body and `/health` gates above do. The sample is the
    response to the request the page itself prints, so the coherence checks at the end can read it as
    one server's answer rather than as six unrelated numbers.
    """
    from laya.lang import analyse

    page = open(os.path.join(ROOT, "docs", "http-api.md"), encoding="utf-8").read()
    section = page[page.index("### Response"):page.index("### Confidence")]
    sample = json.loads(section.split("```json", 1)[1].split("```", 1)[0])

    torch_site = _decision_response_site(os.path.join("laya", "agent.py"))
    onnx_site = _decision_response_site(os.path.join("laya", "onnx_agent.py"))
    top, always, optional = torch_site["keys"], torch_site["usage"], torch_site["optional"]
    assert {k: v for k, v in onnx_site.items() if k != "head"} == \
        {k: v for k, v in torch_site.items() if k != "head"}, (
        "the two agents report different fields, so the page cannot describe both: torch %s, "
        "onnx %s" % (torch_site, onnx_site))

    assert sorted(sample) == sorted(top + ["routing"]), (
        "the sample says %s, an agent result is %s plus the `routing` Router.attach" % (
            sorted(sample), top))
    assert sample["model"] == torch_site["head"], (
        "the sample's head name is %r, `Agent.predict_batch` stamps %r" % (
            sample["model"], torch_site["head"]))

    assert sorted(sample["usage"]) == always, (
        "the sample's usage block says %s, the agents build %s" % (sorted(sample["usage"]), always))
    documented = _md_table_keys(section, "| `usage` key | meaning |")
    assert sorted(documented) == sorted(always + optional), (
        "the usage table says %s, the agents build %s (always) + %s (only when it has to say so)"
        % (sorted(documented), always, optional))

    routing = _route_decision_keys()
    assert sorted(sample["routing"]) == routing, (
        "the sample's routing block says %s, `_route` builds %s" % (sorted(sample["routing"]),
                                                                   routing))
    assert sorted(_md_table_keys(section, "| `routing` key | meaning |")) == routing, (
        "the routing table must name exactly the keys a RouteDecision can carry")

    # The hint the page used to describe as a key is not one, and no branch has ever set it.
    assert "lang_guess" not in routing + always + optional, (
        "`lang_guess` is now a response key; the prose describes it as request-side evidence")
    assert "`lang_guess` evidence" not in page, "the page still calls lang_guess a routing key"

    # `detection` is whatever analyse() returns, so the field names come from calls rather than a
    # list -- over enough states to show the set does not move with the script, which is what lets
    # one row describe it.
    shapes = {tuple(sorted(analyse(state))) for state in
              ("I was charged twice this month, I want my money back",
               "Rechnung \u00fcber zwei Abbuchungen, ich bitte um Erstattung",
               "\u3042\u306e\u8acb\u6c4f\u304c\u91cd\u8907\u3057\u3066\u3044\u307e\u3059", "")}
    assert len(shapes) == 1, "analyse() returns a different key set per script: %s" % (sorted(shapes),)
    detected = sorted(shapes.pop())
    assert sorted(sample["routing"]["detection"]) == detected, (
        "the sample shows %s, analyse() returns %s" % (
            sorted(sample["routing"]["detection"]), detected))
    detection_row = [line for line in section.splitlines() if line.startswith("| `detection` |")]
    assert len(detection_row) == 1, "the routing table has no single `detection` row"
    for key in detected:
        assert "`%s`" % key in detection_row[0], (
            "%s is in the sample's detection block but not named in the row that defines it" % key)

    # And the sample has to be internally coherent, since it is presented as one real answer: the
    # truncation flag is that worst case being non-zero, and the per-question list is empty when
    # nothing was cut.
    usage = sample["usage"]
    assert usage["output_tokens"] == 0, "the head generates nothing, so a sample must not show more"
    assert usage["truncated"] == (usage["state_tokens_dropped"] > 0), usage
    assert (usage["truncated_questions"] == []) == (not usage["truncated"]), usage
    assert 0 < usage["state_tokens"] and usage["state_tokens_dropped"] < usage["state_tokens"], usage


def _answer_literal_keys(rel):
    """What one agent stamps on each answer, read out of its own dict literals.

    The three `answers[qid] = {...}` literals of the decode step, keyed by the `"type"` each one
    writes. From the source rather than transcribed, for the same reason as `_decision_response_site`:
    a hand-copied list would be a third copy of the contract to keep in step.
    """
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                and target.value.id == "answers" and isinstance(node.value, ast.Dict)):
            continue
        names = [k.value for k in node.value.keys
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        stamp = next((v.value for k, v in zip(node.value.keys, node.value.values)
                      if isinstance(k, ast.Constant) and k.value == "type" and isinstance(v, ast.Constant)),
                     None)
        if stamp in ("choice", "score", "noul"):
            found.setdefault(stamp, set()).update(names)
    assert set(found) == {"choice", "score", "noul"}, "%s builds no answer literal for %s" % (
        rel, sorted({"choice", "score", "noul"} - set(found)))
    return {qtype: sorted(keys) for qtype, keys in found.items()}


def _gate_written_keys():
    """Every key the abstention gate writes onto an answer, from `laya/confidence.py` itself."""
    path = os.path.join(ROOT, "laya", "confidence.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    written = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                and target.value.id == "a" and isinstance(target.slice, ast.Constant)):
            written.add(target.slice.value)
    return sorted(written)


def _md_answer_table(section):
    """The `| answer type | keys |` table as rows: which keys the page says an answer carries.

    Only backticks name keys, and only a lowercase identifier among them: prose in a cell is
    description, and `"0".. "k-1"` names the shape of `probabilities` rather than a field. A dotted
    name is one key reached through another -- `action.act_probability` is the `action` dict -- so it
    counts as its head.
    """
    lines = section.splitlines()
    start = lines.index("| answer type | keys |")
    rows = {}
    for line in lines[start + 2:]:
        if not line.strip():
            break
        cells = line.split("|")
        names = set()
        for token in cells[2].split("`")[1::2]:
            head = token.split(".")[0]
            if head.isidentifier() and head == head.lower():
                names.add(head)
        rows[cells[1].strip().strip("`")] = names
    assert set(rows) == {"choice", "score", "noul", "all", "gate"}, (
        "the answer table rows are %s; this gate reads one row per question type, one for the keys "
        "every answer shares, and one for the gate report" % sorted(rows))
    return rows


def test_http_api_page_documents_the_gate_report_on_an_answer():
    r"""The page documents every key an answer can carry, including the three the gate adds (#361).

    `### Response` describes the answer key set in a table and the request controls in another, and the
    usage and routing tables of the same page are already held to the code in both directions. The
    answer table was the one left out, and it had drifted in the way only a table nobody checks drifts:
    it named the three keys an agent builds and the three `apply_confidence_gate` writes onto the very
    same answers in none of them. A caller could set `min_confidence` from the request table and then
    read a response page that said the answer had been "marked" without saying what the mark is called,
    what state the answer is in, or what threshold produced it.

    So the table is compared against both writers: the `answers[qid] = {...}` literals of each agent,
    and the `a[...] = ...` assignments of the gate. Both directions, and against the printed sample too
    -- which must carry no gate report, because the request the page prints sets no threshold.
    """
    page = open(os.path.join(ROOT, "docs", "http-api.md"), encoding="utf-8").read()
    section = page[page.index("### Response"):page.index("### Confidence")]
    rows = _md_answer_table(section)

    torch_site = _answer_literal_keys(os.path.join("laya", "agent.py"))
    onnx_site = _answer_literal_keys(os.path.join("laya", "onnx_agent.py"))
    assert torch_site == onnx_site, (
        "the two agents build different answers, so the page cannot describe both: torch %s, onnx %s"
        % (torch_site, onnx_site))

    gate = _gate_written_keys()
    assert gate, "the gate writes no `a[...] = ...` in laya/confidence.py; retarget this"
    assert sorted(rows["gate"]) == gate, (
        "the gate report says %s, `apply_confidence_gate` and `flag_low_confidence` write %s"
        % (sorted(rows["gate"]), gate))

    # The row's first column is the discriminator every answer stamps, so it is documented by the row
    # that documents the type rather than as a key inside the cell.
    discriminator = {"type"}
    for qtype, built in torch_site.items():
        documented = rows[qtype] | rows["all"] | rows["gate"] | discriminator
        assert documented == set(built) | set(gate), (
            "the %s rows say an answer carries %s, the agents build %s and the gate adds %s" % (
                qtype, sorted(documented), built, gate))

    # And the sample is one answer to the request the page itself prints, which sends no threshold:
    # so its answers show the always-on keys and no gate report at all.
    curl = page[page.index("### `POST /v1/systemone`"):page.index("### Response")]
    assert "min_confidence" not in curl.split("```bash", 1)[1].split("```", 1)[0], (
        "the printed request now sets a threshold, so the sample below it has to show the gate report")
    sample = json.loads(section.split("```json", 1)[1].split("```", 1)[0])
    assert sample["answers"], "the printed sample carries no answers"
    for qid, answer in sample["answers"].items():
        assert not set(gate) & set(answer), (
            "%s is ungated in the printed request but carries the gate key %s" % (
                qid, sorted(set(gate) & set(answer))))
        assert set(answer) == discriminator | rows[answer["type"]] | rows["all"], (
            "the sample's %s answer says %s, its table row plus the shared row say %s" % (
                qid, sorted(answer), sorted(discriminator | rows[answer["type"]] | rows["all"])))


class IdleRouter(FakeRouter):
    def __init__(self):
        import threading
        super().__init__()
        self.loaded = ["english"]
        self.unloaded = threading.Event()
        self.unload_threads = []

    def unload(self):
        import threading
        self.unload_threads.append(threading.current_thread().name)
        self.loaded = []
        self.unloaded.set()

    def predict(self, *args, **kwargs):
        self.loaded = ["english"]
        return super().predict(*args, **kwargs)


@pytest.mark.parametrize("raw, expected", [
    (None, 0.0), ("", 0.0), ("  ", 0.0), ("0", 0.0), ("-5", 0.0), ("abc", 0.0),
    ("nan", 0.0), ("inf", 0.0), ("-inf", 0.0), ("1e999", 0.0),
    ("300", 300.0), (" 0.5 ", 0.5), ("1e2", 100.0),
])
def test_idle_unload_setting_is_finite_and_opt_in(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("LAYA_IDLE_UNLOAD_SECONDS", raising=False)
    else:
        monkeypatch.setenv("LAYA_IDLE_UNLOAD_SECONDS", raw)
    assert _resolve_idle_unload_seconds() == expected


def test_idle_unload_is_off_by_default(monkeypatch):
    monkeypatch.delenv("LAYA_IDLE_UNLOAD_SECONDS", raising=False)
    router = IdleRouter()
    with TestClient(create_app(router)) as client:
        assert not router.unloaded.wait(0.15)
        assert router.loaded == ["english"]
        assert "idle_seconds" not in client.get("/health").json()


def test_idle_unload_reloads_on_the_next_request(monkeypatch):
    monkeypatch.setenv("LAYA_IDLE_UNLOAD_SECONDS", "0.1")
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    router = IdleRouter()
    with TestClient(create_app(router)) as client:
        first = client.post("/v1/systemone", json=REQ).json()
        assert router.unloaded.wait(2.0)
        health = client.get("/health").json()
        assert health["loaded"] == [] and health["idle_unload_seconds"] == 0.1
        assert health["idle_seconds"] >= 0.1
        assert client.post("/v1/systemone", json=REQ).json() == first
        assert router.loaded == ["english"]
    assert all(name.startswith("laya-infer") for name in router.unload_threads)


def test_idle_details_stay_authenticated(monkeypatch):
    monkeypatch.setenv("LAYA_IDLE_UNLOAD_SECONDS", "300")
    monkeypatch.setenv("LAYA_API_KEY", "key")
    with TestClient(create_app(IdleRouter())) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/health", headers={"Authorization": "Bearer key"}).json()["idle_unload_seconds"] == 300


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_idle_window_starts_after_worker_finishes_even_on_cancel_or_failure(monkeypatch, batch, cancel, fail):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("LAYA_IDLE_UNLOAD_SECONDS", "0.2")
    monkeypatch.delenv("LAYA_API_KEY", raising=False)

    class BlockingRouter(IdleRouter):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()
            self.active = False

        def predict(self, *args, **kwargs):
            self.active = True
            self.started.set()
            try:
                assert self.release.wait(3.0)
                if fail:
                    raise ValueError("test failure")
                return super().predict(*args, **kwargs)
            finally:
                self.active = False
                self.finished.set()

        def unload(self):
            assert not self.active, "unload overlapped inference"
            super().unload()

    router = BlockingRouter()
    path = "/v1/systemone/batch" if batch else "/v1/systemone"
    body = {"states": [REQ["state"]], "questions": REQ["questions"]} if batch else REQ
    app = create_app(router)
    if cancel:
        import httpx

        async def exercise():
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    task = asyncio.create_task(client.post(path, json=body))
                    try:
                        assert await asyncio.to_thread(router.started.wait, 2.0)
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                        # A reaper can now queue behind the still-running forward pass.
                        await asyncio.sleep(0.3)
                        assert not router.unloaded.is_set()
                        router.release.set()
                        assert await asyncio.to_thread(router.finished.wait, 2.0)
                        assert not await asyncio.to_thread(router.unloaded.wait, 0.1)
                        assert await asyncio.to_thread(router.unloaded.wait, 2.0)
                    finally:
                        router.release.set()
        asyncio.run(exercise())
    else:
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as callers:
            request = callers.submit(client.post, path, json=body)
            try:
                assert router.started.wait(2.0)
                time.sleep(0.3)
                assert not router.unloaded.is_set()
                router.release.set()
                assert request.result(timeout=2.0).status_code == (422 if fail else 200)
                assert not router.unloaded.wait(0.1)
                assert router.unloaded.wait(2.0)
            finally:
                router.release.set()


def test_idle_unload_retries_failure_and_stops_at_shutdown(monkeypatch):
    monkeypatch.setenv("LAYA_IDLE_UNLOAD_SECONDS", "0.05")

    class RetryRouter(IdleRouter):
        attempts = 0

        def unload(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("test unload failure")
            super().unload()

    router = RetryRouter()
    with TestClient(create_app(router)):
        assert router.unloaded.wait(2.0)
    assert router.attempts == 2
    router.loaded = ["english"]
    router.unloaded.clear()
    assert not router.unloaded.wait(0.15)
