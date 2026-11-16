"""Standalone Monid integration checks; no checkpoints or network required."""
import io
import json
import os
import sys
import urllib.error
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.monid import MonidAgent, attach_monid  # noqa: E402
from laya import Router  # noqa: E402


QUESTIONS = {"urgent": {"type": "noul", "instructions": "Is this urgent?"},
             "billing": {"type": "noul", "instructions": "Is this about billing?"}}
OUTPUT = {"answers": {"urgent": {"noul": 0.9}, "billing": {"noul": 0.8}}}


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setenv("MONID_API_KEY", "fake-monid-key")
    opener = MagicMock()
    monkeypatch.setattr("examples.monid.urllib.request.build_opener", lambda *args: opener)
    return opener


def response(status, run):
    result = MagicMock()
    result.__enter__.return_value.status = status
    result.__enter__.return_value.read.return_value = json.dumps(run).encode("utf-8")
    return result


def test_request_and_router_registration(http, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    http.open.return_value = response(200, {"output": OUTPUT})
    router = attach_monid(Router())
    result = router.predict({"ticket": "退款很急"}, QUESTIONS, model="monid")
    assert result["answers"] == OUTPUT["answers"]
    req = http.open.call_args.args[0]
    assert req.full_url == "https://api.monid.ai/v1/run"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer fake-monid-key"
    assert req.get_header("Content-type") == "application/json; charset=utf-8"
    assert "退款很急".encode("utf-8") in req.data
    assert json.loads(req.data) == {
        "provider": "typesafe", "endpoint": "/systemone",
        "input": {"body": {"state": {"ticket": "退款很急"}, "questions": QUESTIONS, "model": "jev-latest"}}}
    assert 0 < http.open.call_args.kwargs["timeout"] <= 60


def test_without_key_keeps_local_default(monkeypatch):
    monkeypatch.delenv("MONID_API_KEY", raising=False)
    router = Router()
    before = router.route("Hello", QUESTIONS)
    assert attach_monid(router) is router
    assert router.route("Hello", QUESTIONS) == before
    assert "monid" not in router.models
    with pytest.raises(ValueError, match="MONID_API_KEY"):
        MonidAgent()


def test_async_completion(http, monkeypatch):
    monkeypatch.setattr("examples.monid.time.sleep", lambda seconds: None)
    http.open.side_effect = [response(202, {"runId": "run-1"}),
                             response(200, {"status": "RUNNING"}),
                             response(200, {"status": "COMPLETED", "output": OUTPUT})]
    assert MonidAgent().predict("hello", QUESTIONS) == OUTPUT
    for call in http.open.call_args_list[1:]:
        assert call.args[0].full_url == "https://api.monid.ai/v1/runs/run-1"
        assert call.args[0].get_method() == "GET"


@pytest.mark.parametrize("status", ["FAILED", "TIMED_OUT", "STOPPED", "BLOCKED"])
def test_terminal_failure(http, status):
    http.open.side_effect = [response(202, {"runId": "run-1"}), response(200, {"status": status})]
    with pytest.raises(RuntimeError, match=status):
        MonidAgent().predict("hello", QUESTIONS)


def test_provider_error_before_output(http):
    http.open.return_value = response(200, {"status": "COMPLETED",
                                           "providerResponse": {"httpStatus": 429, "error": "rate limited"}})
    with pytest.raises(RuntimeError, match="Laya server error 429: rate limited"):
        MonidAgent().predict("hello", QUESTIONS)


def test_monid_http_error(http):
    http.open.side_effect = urllib.error.HTTPError("https://api.monid.ai/v1/run", 401, "bad key", {},
                                                   io.BytesIO(b'{"message":"bad key"}'))
    with pytest.raises(RuntimeError, match="Monid server error 401"):
        MonidAgent().predict("hello", QUESTIONS)


def test_poll_deadline(http, monkeypatch):
    http.open.return_value = response(202, {"runId": "run-1"})
    ticks = iter([0, 0, 61])
    monkeypatch.setattr("examples.monid.time.monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="Monid run timed out"):
        MonidAgent().predict("hello", QUESTIONS)
