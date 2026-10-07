"""Remote mode of the MCP server (LAYA_BASE_URL): answers come from a laya-serve over HTTP.

A stdlib HTTP server stands in for laya-serve, so these tests need no checkpoint, no torch
and no external network. Run: python -m pytest tests/test_mcp_remote.py
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("mcp")

from laya.mcp import server as mcp_server  # noqa: E402
from laya.mcp.tools import (  # noqa: E402
    ToolError, laya_decide, laya_predict, laya_predict_batch, laya_preset, laya_route, laya_route_batch,
    laya_shortlist, laya_status,
)
from laya.mcp.remote import DEFAULT_TIMEOUT_S, RemoteError, RemoteRouter  # noqa: E402

ANSWER = {
    "model": "laya-rl-agent",
    "answers": {"dept": {"type": "choice", "choice": "billing",
                         "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.9}},
    "usage": {"input_tokens": 40, "output_tokens": 0},
    "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text"},
}
HEALTH = {"status": "ok", "loaded": ["english"], "revisions": {"english": "abc"},
          "device": "mps", "device_is_preference": False, "checkpoint_devices": {"english": "mps"}}


class FakeServe(BaseHTTPRequestHandler):
    """Records every request; answers /health and /v1/systemone like laya-serve."""

    requests: list = []
    status_override: int | None = None
    detail_override: str = "bad request"
    require_bearer: str | None = None

    def log_message(self, *args):  # quiet
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeServe.requests.append(("GET", self.path, None, dict(self.headers)))
        if self.path == "/health":
            if FakeServe.require_bearer and self.headers.get("Authorization") != "Bearer " + FakeServe.require_bearer:
                return self._send(200, {"status": "ok"})
            return self._send(200, HEALTH)
        self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        FakeServe.requests.append(("POST", self.path, body, dict(self.headers)))
        if FakeServe.require_bearer and self.headers.get("Authorization") != "Bearer " + FakeServe.require_bearer:
            return self._send(401, {"detail": "invalid or missing bearer token"})
        if FakeServe.status_override:
            return self._send(FakeServe.status_override, {"detail": FakeServe.detail_override})
        if self.path == "/v1/systemone":
            return self._send(200, ANSWER)
        self._send(404, {"detail": "Not Found"})


@pytest.fixture
def serve():
    FakeServe.requests = []
    FakeServe.status_override = None
    FakeServe.require_bearer = None
    httpd = HTTPServer(("127.0.0.1", 0), FakeServe)
    thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _questions():
    return {"dept": {"type": "choice", "instructions": "which team?", "criteria": {"billing": None, "tech": None}}}


def test_predict_posts_the_serve_body_and_returns_it_unchanged(serve):
    router = RemoteRouter(serve)
    result = router.predict({"text": "billed twice"}, _questions(), lang="en", max_len=512, min_confidence=0.3)
    assert result == ANSWER
    method, path, body, headers = FakeServe.requests[-1]
    assert (method, path) == ("POST", "/v1/systemone")
    assert body == {"state": {"text": "billed twice"}, "questions": _questions(),
                    "lang": "en", "max_len": 512, "min_confidence": 0.3}
    assert "Authorization" not in headers


def test_auto_model_is_not_sent_but_a_pinned_one_is(serve):
    router = RemoteRouter(serve)
    router.predict("x", _questions(), model="auto")
    assert "model" not in FakeServe.requests[-1][2]
    router.predict("x", _questions(), model="multilingual")
    assert FakeServe.requests[-1][2]["model"] == "multilingual"


def test_bearer_token_is_sent_when_configured(serve):
    FakeServe.require_bearer = "s3cret"
    router = RemoteRouter(serve, api_key="s3cret")
    assert router.predict("x", _questions()) == ANSWER
    assert FakeServe.requests[-1][3]["Authorization"] == "Bearer s3cret"
    with pytest.raises(RemoteError) as info:
        RemoteRouter(serve, api_key="wrong").predict("x", _questions())
    assert info.value.code == "unauthorized"


@pytest.mark.parametrize("status, code", [(400, "invalid_request"), (422, "invalid_request"),
                                          (503, "busy"), (500, "upstream_error")])
def test_http_errors_map_to_tool_error_codes(serve, status, code):
    FakeServe.status_override = status
    FakeServe.detail_override = "question 'dept': criteria must be an object"
    with pytest.raises(RemoteError) as info:
        RemoteRouter(serve).predict("x", _questions())
    assert info.value.code == code
    assert "criteria must be an object" in info.value.message


def test_unreachable_server_is_a_clean_error():
    with pytest.raises(RemoteError) as info:
        RemoteRouter("http://127.0.0.1:9", timeout=2).predict("x", _questions())
    assert info.value.code == "unreachable"


def test_loaded_and_revisions_come_from_health(serve):
    router = RemoteRouter(serve)
    assert router.loaded == ["english"]
    assert router.loaded_revisions == {"english": "abc"}
    assert RemoteRouter("http://127.0.0.1:9", timeout=2).loaded == []


def test_routing_stays_local_and_needs_no_server(monkeypatch):
    router = RemoteRouter("http://127.0.0.1:9", timeout=2)
    def no_http(*args, **kwargs):
        raise AssertionError("routing attempted an HTTP request")
    monkeypatch.setattr(router, "_request", no_http)
    decision = router.route({"text": "Mein Konto wurde zweimal belastet"}, _questions())
    assert decision.model == "multilingual"
    assert router.route_batch([{"state": "Mein Konto wurde zweimal belastet", "questions": _questions()}])[0].model == "multilingual"


def test_predict_batch_is_one_call_per_request_in_order(serve):
    router = RemoteRouter(serve)
    items = [{"state": "a", "questions": _questions()},
             {"state": "b", "questions": _questions(), "model": "english", "max_len": 256}]
    results = router.predict_batch(items, batch_size=8, sort_by_length=True, min_confidence=0.5)
    assert results == [ANSWER, ANSWER]
    posts = [r for r in FakeServe.requests if r[0] == "POST"]
    assert [p[2]["state"] for p in posts] == ["a", "b"]
    assert posts[1][2]["model"] == "english" and posts[1][2]["max_len"] == 256
    assert all(p[2]["min_confidence"] == 0.5 for p in posts)


def test_lifecycle_is_server_owned(serve):
    router = RemoteRouter(serve)
    assert router.preload(["english"]) is router
    assert router.unload() is None
    with pytest.raises(RemoteError) as info:
        router.load("english")
    assert info.value.code == "unsupported_remote"


def test_tools_translate_remote_errors(serve):
    FakeServe.status_override = 422
    FakeServe.detail_override = "question 'dept': unknown type"
    with pytest.raises(ToolError) as info:
        laya_predict({"text": "x"}, _questions(), router=RemoteRouter(serve))
    assert info.value.code == "invalid_request" and "unknown type" in info.value.message
    with pytest.raises(ToolError) as info:
        laya_predict_batch([{"state": {"text": "x"}, "questions": _questions()}], router=RemoteRouter(serve))
    assert info.value.code == "invalid_request"


def test_predict_tool_shape_through_a_remote_router(serve):
    out = laya_predict({"text": "billed twice"}, _questions(), router=RemoteRouter(serve))
    assert out["answers"] == ANSWER["answers"]
    assert out["routing"]["model"] == "english"
    assert "device" not in out          # no in-process agent to read a device from
    assert out["latency_ms"] >= 0


def test_status_reports_the_server_not_local_torch(serve):
    out = laya_status(router=RemoteRouter(serve), preload=False)
    assert out["mode"] == "remote" and out["base_url"] == serve
    assert out["loaded"] == ["english"] and out["device"] == "mps"
    assert out["server"]["status"] == "ok"
    assert "torch_version" not in out
    down = laya_status(router=RemoteRouter("http://127.0.0.1:9", timeout=2))
    assert down["mode"] == "remote" and down["loaded"] == [] and "server_error" in down


def test_shortlist_is_refused_in_remote_mode(serve):
    many = {"pick": {"type": "choice", "instructions": "pick", "criteria": {"opt%d" % i: None for i in range(60)}}}
    with pytest.raises(ToolError) as info:
        laya_shortlist({"text": "x"}, many, k=5, router=RemoteRouter(serve))
    assert info.value.code == "unsupported_remote"


def test_server_builds_a_remote_router_from_the_environment(monkeypatch, serve):
    monkeypatch.setenv("LAYA_BASE_URL", serve + "/")
    monkeypatch.setenv("LAYA_API_KEY", "k")
    monkeypatch.setattr(mcp_server, "_ROUTER", None)
    router = mcp_server._ensure_router()
    assert isinstance(router, RemoteRouter)
    assert router.base_url == serve and router.api_key == "k"
    monkeypatch.setattr(mcp_server, "_ROUTER", None)


@pytest.mark.parametrize("raw, expected", [("", None), ("  ", None), ("127.0.0.1:8123", "http://127.0.0.1:8123"),
                                           ("http://h:1/", "http://h:1"), ("https://laya.example", "https://laya.example")])
def test_base_url_normalisation(monkeypatch, raw, expected):
    monkeypatch.setenv("LAYA_BASE_URL", raw)
    assert mcp_server._remote_base_url() == expected


@pytest.mark.parametrize("raw", ["bad", "0", "-1", "nan", "inf", "1e999"])
def test_remote_timeout_rejects_invalid_environment(monkeypatch, raw):
    monkeypatch.setenv("LAYA_REMOTE_TIMEOUT", raw)
    assert RemoteRouter("http://localhost:8000").timeout == DEFAULT_TIMEOUT_S


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_remote_timeout_rejects_invalid_explicit_values(timeout):
    with pytest.raises(ValueError, match="finite positive"):
        RemoteRouter("http://localhost:8000", timeout=timeout)


@pytest.mark.parametrize("key, value", [
    ("hooks", []), ("on_predict_start", lambda ctx: None), ("on_predict_end", lambda ctx: None),
    ("hooks_raise", False), ("hooks_timeout", 1), ("lang_guess", lambda state: "en"),
])
def test_remote_predict_refuses_controls_without_a_wire_form(serve, key, value):
    with pytest.raises(RemoteError) as info:
        RemoteRouter(serve).predict("x", _questions(), **{key: value})
    assert info.value.code == "unsupported_remote"
    assert not FakeServe.requests


def test_remote_batch_refuses_hook_timeouts_and_item_hooks(serve):
    router = RemoteRouter(serve)
    items = [{"state": "x", "questions": _questions()}]
    with pytest.raises(RemoteError, match="hooks_timeout"):
        router.predict_batch(items, hooks_timeout=1)
    items[0]["hooks"] = [lambda ctx: None]
    with pytest.raises(RemoteError, match="per-request hooks"):
        router.predict_batch(items)
    assert not FakeServe.requests


def test_remote_decide_and_preset_use_the_server(serve):
    router = RemoteRouter(serve)
    schema = {"type": "object", "properties": {"dept": {"type": "string", "enum": ["billing", "tech"]}}}
    result = laya_decide({"text": "billed twice"}, schema, router=router)
    assert result["values"] == {"dept": "billing"}
    assert FakeServe.requests[-1][2]["questions"]["dept"]["criteria"] == {"billing": None, "tech": None}
    laya_preset("triage", {"text": "billed twice"}, router=router, preset_builder=mcp_server._preset_builder)
    assert FakeServe.requests[-1][1] == "/v1/systemone"
    FakeServe.status_override = 503
    for fn in (lambda: laya_decide({"text": "x"}, schema, router=router),
               lambda: laya_preset("triage", {"text": "x"}, router=router, preset_builder=mcp_server._preset_builder)):
        with pytest.raises(ToolError) as info:
            fn()
        assert info.value.code == "busy"


def test_remote_route_tools_do_not_contact_the_server(serve):
    router = RemoteRouter(serve)
    state = {"text": "Mein Konto wurde zweimal belastet"}
    assert laya_route(state, _questions(), router=router)["model"] == "multilingual"
    assert laya_route_batch([{"state": state, "questions": _questions()}], router=router)["decisions"][0]["model"] == "multilingual"
    assert not FakeServe.requests


@pytest.mark.parametrize("url", ["file:///tmp/model", "localhost:8000", "http://", "http://host?q=x", "http://host#x"])
def test_remote_router_requires_an_http_server_url(url):
    with pytest.raises(ValueError, match="HTTP"):
        RemoteRouter(url)


def test_remote_mcp_startup_and_every_tool_avoid_torch(serve):
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent('''
        import builtins, json, sys
        original_import = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("remote MCP imported torch")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = no_torch
        from laya.mcp import server
        server.server.run = lambda: None
        server.main()
        state = {"text": "billed twice"}
        questions = {"dept": {"type": "choice", "instructions": "which team?",
                              "criteria": {"billing": "payments", "tech": "bugs"}}}
        schema = {"type": "object", "properties": {"dept": {"type": "string", "enum": ["billing", "tech"]}}}
        items = [{"state": state, "questions": questions}]
        calls = [lambda: server.laya_status_tool(),
                 lambda: server.laya_predict_tool(state, questions),
                 lambda: server.laya_predict_batch_tool(items),
                 lambda: server.laya_decide_tool(state, schema),
                 lambda: server.laya_preset_tool("triage", state),
                 lambda: server.laya_route_tool(state, questions),
                 lambda: server.laya_route_batch_tool(items)]
        for call in calls:
            out = json.loads(call())
            assert "error" not in out, out
        try:
            server.laya_shortlist_tool(state, questions)
        except Exception as exc:
            assert "unsupported_remote" in str(exc), exc
        else:
            raise AssertionError("remote shortlist did not refuse the call")
        assert "torch" not in sys.modules
        assert not server._ROUTER._agents
    ''')
    env = dict(os.environ, LAYA_BASE_URL=serve, LAYA_THREADS="2", LAYA_PRELOAD="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_remote_timeout_is_forwarded_and_http_failures_are_validated(monkeypatch, serve):
    monkeypatch.setenv("LAYA_REMOTE_TIMEOUT", "0.25")
    router = RemoteRouter(serve)
    assert router.timeout == 0.25
    assert RemoteRouter(serve, timeout=3).timeout == 3
    from urllib.error import URLError
    import urllib.request
    def timed_out(*args, **kwargs):
        raise URLError(TimeoutError("test timeout"))
    monkeypatch.setattr(urllib.request, "urlopen", timed_out)
    with pytest.raises(RemoteError) as info:
        router.predict("x", _questions())
    assert info.value.code == "timeout"


def test_remote_transport_preserves_options_and_all_wire_controls(serve):
    questions = {"dept": {"type": "choice", "instructions": "Which team handles `message`?",
                          "criteria": {"billing": "refunds", "tech": "bugs"}}}
    controls = {"task": "typed_decisions", "lang": "de", "lang_guess": "de", "max_len": 512,
                "head_max_len": 128, "min_confidence": 0.7}
    state = {"message": "Rückerstattung bitte"}
    RemoteRouter(serve).predict(state, questions, **controls)
    assert FakeServe.requests[-1][2] == dict(state=state, questions=questions, **controls)


@pytest.mark.parametrize("raw", [b"not JSON", b"\xff", b"[]", b"null"])
def test_remote_transport_reports_malformed_responses(monkeypatch, raw):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import urllib.request

    @contextmanager
    def response(*args, **kwargs):
        yield SimpleNamespace(read=lambda: raw)
    monkeypatch.setattr(urllib.request, "urlopen", response)
    with pytest.raises(RemoteError) as info:
        RemoteRouter("http://localhost:8000").predict("x", _questions())
    assert info.value.code == "upstream_error"
