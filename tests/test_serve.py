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
    DEFAULT_MAX_TOKENS,
    MAX_BODY_BYTES,
    _apply_thread_limit,
    _env_bool,
    _resolve_max_tokens,
    _resolve_model,
    create_app,
)


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


def test_health_supports_router_without_loaded_revisions(monkeypatch):
    # FakeRouter deliberately has no loaded_revisions attribute. Injected test or
    # embedding routers predating revision reporting must remain health-compatible.
    client, _ = _client(monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["revisions"] == {}


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


def test_a_nested_choice_label_is_a_caller_error_not_a_server_fault(monkeypatch):
    """A `criteria` list containing a list/dict label is the caller's mistake, so it must be 422.

    It used to raise `TypeError: unhashable type: 'list'` from `_to_internal`, three frames below
    `_check_question`, which names neither the question nor the label -- and `serve` maps only
    `ValueError` to 422, so the caller got a 500 "inference failed" with the reason discarded.
    `ValueError` is what carries the message to the client, so the guard has to raise that type.
    """
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

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    client = TestClient(create_app(router=ValidatingRouter()), raise_server_exceptions=False)

    for label in (["billing"], {"billing": "x"}):
        body = dict(REQ)
        body["questions"] = {"dept": {"type": "choice", "instructions": "Which team?",
                                      "criteria": [label, "tech"]}}
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 422, (label, response.status_code, response.text)
        assert "choice label 0" in response.text, response.text


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
            seen["second"] = (await client.post("/v1/systemone", json=REQ)).status_code
            fake.release.set()
            seen["first"] = (await first).status_code

    asyncio.run(drive())

    assert seen["second"] == 503, seen
    assert seen["first"] == 200, seen


def test_admission_slot_is_released_after_inference(monkeypatch):
    """Slots are reusable: sequential requests with a bound of one all pass."""
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "1")
    client = TestClient(create_app(router=FakeRouter()))
    assert client.post("/v1/systemone", json=REQ).status_code == 200
    assert client.post("/v1/systemone", json=REQ).status_code == 200


class BudgetRouter(FakeRouter):
    """A router whose predict() takes the token-budget keywords, and records them."""

    def predict(self, state, questions, model=None, **budget):
        out = super().predict(state, questions, model=model)
        self.calls[-1]["budget"] = budget      # one record per call, with what was forwarded
        return out


def test_token_budget_is_forwarded(monkeypatch):
    """`max_len`/`head_max_len` in the body reach Router.predict.

    Without this the README's long-document advice -- raise `max_len` -- had no HTTP
    equivalent: the field was dropped and the state was answered from its first window (#549).
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    router = BudgetRouter()
    client = TestClient(create_app(router=router))

    body = dict(REQ, max_len=8192, head_max_len=256)
    assert client.post("/v1/systemone", json=body).status_code == 200
    assert router.calls[-1]["budget"] == {"max_len": 8192, "head_max_len": 256}

    # one of the two alone is forwarded alone
    assert client.post("/v1/systemone", json=dict(REQ, max_len=2048)).status_code == 200
    assert router.calls[-1]["budget"] == {"max_len": 2048}


def test_no_budget_keeps_the_call_unchanged(monkeypatch):
    """A client that sends neither field, or nulls, calls predict() exactly as before.

    `FakeRouter.predict` takes no budget keyword at all, which is the contract an injected or
    predating router relies on: passing one unconditionally would break it.
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    router = FakeRouter()
    client = TestClient(create_app(router=router))
    for body in (REQ, dict(REQ, max_len=None, head_max_len=None)):
        assert client.post("/v1/systemone", json=body).status_code == 200
    assert len(router.calls) == 2


@pytest.mark.parametrize("field", ["max_len", "head_max_len"])
@pytest.mark.parametrize("value", [0, -1, DEFAULT_MAX_TOKENS + 1, 1000000])
def test_a_budget_outside_the_bound_is_422(monkeypatch, field, value):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.delenv("LAYA_MAX_TOKENS", raising=False)
    router = BudgetRouter()
    client = TestClient(create_app(router=router))
    res = client.post("/v1/systemone", json=dict(REQ, **{field: value}))
    assert res.status_code == 422, (field, value, res.status_code, res.text)
    assert field in res.text and "LAYA_MAX_TOKENS" in res.text, res.text
    assert router.calls == [], "no inference should run for a rejected budget"


@pytest.mark.parametrize("value", ["8192", 8192.0, True, [8192], {"n": 8192}])
def test_a_budget_that_is_not_an_integer_is_422(monkeypatch, value):
    """A string, a float, a bool or a container is a client error, not a silent coercion.

    `True` matters on its own: `isinstance(True, int)` is true in Python, so `"max_len": true`
    would otherwise be read as a one-token budget.
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    router = BudgetRouter()
    client = TestClient(create_app(router=router))
    res = client.post("/v1/systemone", json=dict(REQ, max_len=value))
    assert res.status_code == 422, (value, res.status_code, res.text)
    assert "must be an integer" in res.text, res.text
    assert router.calls == []


def test_the_bound_comes_from_the_env(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_MAX_TOKENS", "1024")
    router = BudgetRouter()
    client = TestClient(create_app(router=router))
    assert client.post("/v1/systemone", json=dict(REQ, max_len=1024)).status_code == 200
    assert router.calls[-1]["budget"] == {"max_len": 1024}
    res = client.post("/v1/systemone", json=dict(REQ, max_len=1025))
    assert res.status_code == 422, res.text
    assert "between 1 and 1024" in res.text, res.text


def test_resolve_max_tokens(monkeypatch):
    """Same tolerance as LAYA_MAX_CONCURRENT: an unreadable value falls back, not crashes."""
    monkeypatch.delenv("LAYA_MAX_TOKENS", raising=False)
    assert _resolve_max_tokens() == DEFAULT_MAX_TOKENS
    for raw in ("", "not-a-number", "0", "-5"):
        monkeypatch.setenv("LAYA_MAX_TOKENS", raw)
        assert _resolve_max_tokens() == DEFAULT_MAX_TOKENS, raw
    monkeypatch.setenv("LAYA_MAX_TOKENS", "512")
    assert _resolve_max_tokens() == 512
