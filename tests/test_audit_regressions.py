"""Regression tests for fixes found during the cross-runtime audit."""
import io
import json
from contextlib import redirect_stdout

from laya.cli import show_answers
from laya.integrations.langchain import _SameOriginRedirectHandler
from laya.onnx_agent import ONNXAgent
from laya.serve import MAX_BODY_BYTES, create_app


class FakeRouter:
    loaded = ["english"]

    def predict(self, state, questions, model=None):
        return {
            "model": "test",
            "answers": {},
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


def test_serve_rejects_oversized_stream_without_content_length():
    from fastapi.testclient import TestClient

    app = create_app(router=FakeRouter())
    client = TestClient(app)
    payload = {
        "state": "hello",
        "questions": {"q": {"type": "noul", "instructions": "?"}},
        "padding": "x" * (MAX_BODY_BYTES + 1024),
    }
    encoded = json.dumps(payload).encode()
    # An iterator makes httpx use a streaming body, so this exercises the
    # Content-Length-independent path rather than the existing header guard.
    response = client.post(
        "/v1/systemone",
        content=iter([encoded[:32], encoded[32:]]),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


def test_serve_lifespan_shuts_down_inference_executor():
    from fastapi.testclient import TestClient
    from unittest.mock import patch

    with patch("concurrent.futures.ThreadPoolExecutor.shutdown") as shutdown:
        with TestClient(create_app(router=FakeRouter())):
            pass
    shutdown.assert_called_once_with(wait=True, cancel_futures=True)


def test_serve_non_ascii_auth_is_401_not_500(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LAYA_API_KEY", "s3cret")
    app = create_app(router=FakeRouter())
    client = TestClient(app)
    response = client.post(
        "/v1/systemone",
        content=b'{"state":"hello","questions":{}}',
        headers=[
            ("content-type", "application/json"),
            (b"authorization", b"Bearer caf\xe9"),
        ],
    )
    assert response.status_code == 401


def test_onnx_empty_questions_matches_agent_contract():
    agent = ONNXAgent.__new__(ONNXAgent)
    result = ONNXAgent._infer(agent, "hello", {})
    assert result == {
        "model": "laya-rl-agent-onnx",
        "answers": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_onnx_long_list_state_keeps_newest_turn(monkeypatch):
    import laya.onnx_agent as module

    seen = []
    original = module.build_sequence

    def fake_build(*args, **kwargs):
        seen.append(kwargs.get("truncate_left"))
        raise RuntimeError("stop after encoding")

    monkeypatch.setattr(module, "build_sequence", fake_build)
    agent = ONNXAgent.__new__(ONNXAgent)
    agent.cfg = {"max_len": 32, "head_max_len": 16}
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.tok = object()
    try:
        ONNXAgent._infer(agent, [{"text": "old"}, {"text": "new"}], {
            "q": {"type": "noul", "instructions": "?"}
        })
    except RuntimeError as exc:
        assert str(exc) == "stop after encoding"
    else:
        raise AssertionError("encoding should have been reached")
    assert seen == [True]


def test_cli_renders_choice_probability_from_probabilities_map():
    output = io.StringIO()
    with redirect_stdout(output):
        show_answers({
            "answers": {
                "q": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.91, "no": 0.09}}
            }
        })
    assert "yes (p=0.910)" in output.getvalue()


def test_default_langchain_router_initialization_is_singleton(monkeypatch):
    import threading

    import laya.integrations.langchain as module
    import laya.router as router_module

    created = []
    lock = threading.Lock()

    class FakeRouter:
        def __init__(self):
            with lock:
                created.append(self)

    monkeypatch.setattr(router_module, "Router", FakeRouter)
    monkeypatch.setattr(module, "_DEFAULT_ROUTER", None)
    barrier = threading.Barrier(2)
    results = []

    def get_router():
        barrier.wait()
        results.append(module._get_default_router())

    threads = [threading.Thread(target=get_router) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(created) == 1
    assert results[0] is results[1]


def test_remote_redirect_handler_rejects_cross_origin_credentials():
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        "https://laya.example/v1/systemone",
        headers={"Authorization": "Bearer secret"},
    )
    handler = _SameOriginRedirectHandler()
    same_origin = handler.redirect_request(
        request, None, 302, "Found", {}, "https://laya.example/v1/systemone?retry=1"
    )
    assert same_origin.get_header("Authorization") == "Bearer secret"
    try:
        handler.redirect_request(
            request, None, 302, "Found", {}, "http://attacker.example/collect"
        )
    except urllib.error.URLError as exc:
        assert "cross-origin" in str(exc)
    else:
        raise AssertionError("cross-origin redirect was accepted")


def test_fast_path_rejects_sequences_over_its_capacity():
    from types import SimpleNamespace

    import torch

    from laya.agent import Agent

    agent = Agent.__new__(Agent)
    agent._fast = SimpleNamespace(max_len=16)
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.amp_enabled = False
    agent.mps_amp_min_rows = 5
    batch = {
        "input_ids": torch.zeros((1, 17), dtype=torch.long),
        "attention_mask": torch.ones((1, 17), dtype=torch.long),
        "marker_pos": torch.zeros((1, 2), dtype=torch.long),
        "marker_mask": torch.ones((1, 2), dtype=torch.bool),
        "qtype": torch.zeros((1,), dtype=torch.long),
    }
    try:
        agent._infer(batch)
    except ValueError as exc:
        assert "max_len=16" in str(exc)
    else:
        raise AssertionError("oversized fast-path request was accepted")


def test_fast_top_two_handles_one_option():
    import pytest
    pytest.importorskip("tilelang")
    from laya.fast import _top_two
    import torch

    values = _top_two(torch.tensor([[0.75]]))
    assert values.shape == (1, 2)
    assert torch.allclose(values, torch.tensor([[0.75, 0.0]]))
