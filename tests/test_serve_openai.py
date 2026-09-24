"""OpenAI-compatible route tests: no GPU, a fake Router is injected.

The translation itself is covered by tests/test_openai.py; these assert the HTTP surface.
"""
import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from laya.serve import create_app  # noqa: E402


class AnsweringRouter:
    """Answers every question with its first option, so the shape is deterministic."""

    loaded = ["english"]

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                key = next(iter(q["criteria"]))
                answers[qid] = {"type": "choice", "choice": key, "confidence": 1.0,
                                "probabilities": {key: 1.0}}
            elif q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.8, "confidence": 0.8}
            else:
                answers[qid] = {"type": "score", "score": 0.0, "confidence": 1.0,
                                "probabilities": {"0": 1.0, "1": 0.0}}
        return {"answers": answers, "usage": {"input_tokens": 4, "output_tokens": 0},
                "routing": {"model": "english"}}


SCHEMA = {"type": "object", "properties": {
    "department": {"type": "string", "enum": ["billing", "support"]},
    "urgent": {"type": "boolean"}}}

TOOLS = [{"type": "function", "function": {
    "name": "route", "description": "Route a ticket",
    "parameters": {"type": "object", "properties": {
        "department": {"type": "string", "enum": ["billing", "support"]}}}}}]


def _client(monkeypatch, api_key=None):
    if api_key is None:
        monkeypatch.delenv("LAYA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("LAYA_API_KEY", api_key)
    fake = AnsweringRouter()
    return TestClient(create_app(router=fake)), fake


def test_models(monkeypatch):
    client, _ = _client(monkeypatch)
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["english", "multilingual", "typed-decisions"]
    assert body["data"][0]["object"] == "model"


def test_chat_json_schema(monkeypatch):
    client, fake = _client(monkeypatch)
    r = client.post("/v1/chat/completions", json={
        "model": "laya", "messages": [{"role": "user", "content": "billed twice"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": SCHEMA}}})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert json.loads(body["choices"][0]["message"]["content"]) == {"department": "billing", "urgent": True}
    assert body["usage"] == {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4}
    assert fake.calls[0]["state"] == "billed twice"


def test_chat_tools(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "tools": TOOLS})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    call = body["choices"][0]["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "route"
    assert json.loads(call["function"]["arguments"]) == {"department": "billing"}


def test_chat_unsupported_is_400(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert "response_format" in r.json()["detail"]


def test_chat_bad_schema_is_422(monkeypatch):
    client, _ = _client(monkeypatch)
    bad = {"type": "object", "properties": {"note": {"type": "string"}}}  # a free string
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": bad}}})
    assert r.status_code == 422
    assert "properties.note" in r.json()["detail"]


def test_responses_json_schema(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/responses", json={
        "input": "hi", "text": {"format": {"type": "json_schema", "schema": SCHEMA}}})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    item = body["output"][0]
    assert item["type"] == "message"
    assert json.loads(item["content"][0]["text"]) == {"department": "billing", "urgent": True}
    assert body["usage"] == {"input_tokens": 4, "output_tokens": 0, "total_tokens": 4}


def test_responses_tools(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/responses", json={"input": "hi", "tools": TOOLS})
    assert r.status_code == 200
    item = r.json()["output"][0]
    assert item["type"] == "function_call"
    assert item["name"] == "route"
    assert json.loads(item["arguments"]) == {"department": "billing"}


def test_moderations(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/moderations", json={"input": "some post"})
    assert r.status_code == 200
    body = r.json()
    assert set(["id", "model", "results"]).issubset(body)
    result = body["results"][0]
    assert "flagged" in result and "categories" in result and "category_scores" in result
    # the fake answers every noul at 0.8, so the noul categories are flagged
    assert result["categories"]["toxic"] is True
    assert result["flagged"] is True


def test_moderations_batch(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/moderations", json={"input": ["a", "b"]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_auth_required(monkeypatch):
    client, _ = _client(monkeypatch, api_key="s3cret")
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_malformed_json_is_400(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/chat/completions", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_deeply_nested_json_is_400(monkeypatch):
    client, _ = _client(monkeypatch)
    deep = "[" * 2000 + "]" * 2000
    r = client.post("/v1/responses", content=deep, headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_moderation_rejects_too_many_inputs(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/moderations", json={"input": ["text"] * 65})
    assert r.status_code == 413
    assert client.post("/v1/moderations", json={"input": ["text"] * 64}).status_code == 200
