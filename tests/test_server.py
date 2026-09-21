"""
Tests for laya.server — validates all endpoints.

Run with:
    pip install "laya[dev]"
    pytest tests/ -v

These tests use FastAPI's TestClient (synchronous ASGI test adapter) so
they do NOT require a running server or a real model download.
The `_init_router` function is patched with a lightweight mock that returns
a predictable response — so tests run in milliseconds, not minutes.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Mock payloads — what Router.predict() would return
# ---------------------------------------------------------------------------

_MOCK_RAW_CHOICE = {
    "model": "laya-rl-agent",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.94, "technical": 0.06},
            "confidence": 0.92,
            "action": {"act_probability": 0.91},
        }
    },
    "usage": {"input_tokens": 64, "output_tokens": 0},
}

_MOCK_RAW_NOUL = {
    "model": "laya-rl-agent",
    "answers": {
        "is_urgent": {
            "type": "noul",
            "noul": 0.87,
            "confidence": 0.87,
            "action": {"act_probability": 0.85},
        }
    },
    "usage": {"input_tokens": 48, "output_tokens": 0},
}

_MOCK_RAW_SCORE = {
    "model": "laya-rl-agent",
    "answers": {
        "frustration": {
            "type": "score",
            "score": 1.84,
            "legend": {"0": "calm", "1": "concerned", "2": "very angry"},
            "probabilities": {"0": 0.05, "1": 0.27, "2": 0.68},
            "confidence": 0.74,
            "action": {"act_probability": 0.73},
        }
    },
    "usage": {"input_tokens": 72, "output_tokens": 0},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_router(raw: Dict[str, Any] = None) -> MagicMock:
    """Return a mock Router whose predict() returns *raw* synchronously."""
    router = MagicMock()
    router.predict.return_value = raw if raw is not None else _MOCK_RAW_CHOICE
    return router


def _make_client(raw: Dict[str, Any] = None) -> TestClient:
    """
    Build a TestClient where _init_router is patched to return a mock.

    This prevents any model download from happening during tests.
    The patch target is "laya.server._init_router" because server.py
    calls _init_router() in the lifespan, and _init_router is a
    module-level function that can be patched before the app starts.
    """
    mock_router = _make_mock_router(raw)
    with patch("laya.server._init_router", return_value=mock_router):
        from laya import server as srv
        # Force re-import in case a previous test cached the module
        import importlib
        importlib.reload(srv)
        app = srv.create_app()
    # TestClient must be created AFTER the patch context because the
    # lifespan runs when entering TestClient.__enter__.
    # We patch _init_router globally so even after the `with` exits the
    # already-created app object holds the right lifespan reference.
    with patch("laya.server._init_router", return_value=mock_router):
        return TestClient(app)


@pytest.fixture()
def client_choice():
    """TestClient returning a choice answer."""
    mock_router = _make_mock_router(_MOCK_RAW_CHOICE)
    with patch("laya.server._init_router", return_value=mock_router):
        from laya.server import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


@pytest.fixture()
def client_noul():
    mock_router = _make_mock_router(_MOCK_RAW_NOUL)
    with patch("laya.server._init_router", return_value=mock_router):
        from laya.server import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


@pytest.fixture()
def client_score():
    mock_router = _make_mock_router(_MOCK_RAW_SCORE)
    with patch("laya.server._init_router", return_value=mock_router):
        from laya.server import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_ok(client_choice):
    resp = client_choice.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "version" in data


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

def test_models_returns_list(client_choice):
    resp = client_choice.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    model = data["data"][0]
    assert "id" in model
    assert model["capabilities"]["question_types"] == ["choice", "score", "noul"]
    assert model["capabilities"]["batch"] is True


# ---------------------------------------------------------------------------
# POST /v1/decide — choice
# ---------------------------------------------------------------------------

def test_decide_choice(client_choice):
    resp = client_choice.post("/v1/decide", json={
        "state": {"subject": "Charged twice!", "body": "Please refund."},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": {"billing": "invoices", "technical": "bugs"},
            }
        },
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answers"]["department"]["type"] == "choice"
    assert data["answers"]["department"]["choice"] == "billing"
    assert data["answers"]["department"]["confidence"] == pytest.approx(0.92)
    assert "billing" in data["answers"]["department"]["probabilities"]


# ---------------------------------------------------------------------------
# POST /v1/decide — noul
# ---------------------------------------------------------------------------

def test_decide_noul(client_noul):
    resp = client_noul.post("/v1/decide", json={
        "state": "Fix this ASAP!",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "Is this urgent?",
            }
        },
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answers"]["is_urgent"]["type"] == "noul"
    assert data["answers"]["is_urgent"]["noul"] == pytest.approx(0.87)


# ---------------------------------------------------------------------------
# POST /v1/decide — score
# ---------------------------------------------------------------------------

def test_decide_score(client_score):
    resp = client_score.post("/v1/decide", json={
        "state": "I am FURIOUS",
        "questions": {
            "frustration": {
                "type": "score",
                "instructions": "How frustrated?",
                "criteria": ["calm", "concerned", "very angry"],
            }
        },
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answers"]["frustration"]["type"] == "score"
    assert data["answers"]["frustration"]["score"] == pytest.approx(1.84)
    assert "0" in data["answers"]["frustration"]["legend"]


# ---------------------------------------------------------------------------
# POST /v1/systemone — Jev wire-compatible alias
# ---------------------------------------------------------------------------

def test_systemone_alias_identical_to_decide(client_choice):
    """POST /v1/systemone must return the same shape as POST /v1/decide."""
    payload = {
        "state": "Charged twice!",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": {"billing": "invoices", "technical": "bugs"},
            }
        },
    }
    resp_decide = client_choice.post("/v1/decide", json=payload)
    resp_systemone = client_choice.post("/v1/systemone", json=payload)

    assert resp_decide.status_code == 200
    assert resp_systemone.status_code == 200
    d1 = resp_decide.json()
    d2 = resp_systemone.json()
    assert d1["answers"].keys() == d2["answers"].keys()
    assert d1["answers"]["department"]["choice"] == d2["answers"]["department"]["choice"]


# ---------------------------------------------------------------------------
# POST /v1/decide/batch
# ---------------------------------------------------------------------------

def test_decide_batch(client_choice):
    resp = client_choice.post("/v1/decide/batch", json={
        "states": ["Charged twice!", "Bug in the app"],
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": {"billing": "invoices", "technical": "bugs"},
            }
        },
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["total_usage"]["input_tokens"] > 0
    for result in data["results"]:
        assert "department" in result["answers"]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_decide_missing_questions_returns_422(client_choice):
    resp = client_choice.post("/v1/decide", json={
        "state": "some text",
        # missing "questions"
    })
    assert resp.status_code == 422


def test_decide_unknown_type_returns_422(client_choice):
    resp = client_choice.post("/v1/decide", json={
        "state": "some text",
        "questions": {
            "q": {"type": "unknown_type", "instructions": "?"}
        },
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# X-Response-Time-Ms header
# ---------------------------------------------------------------------------

def test_response_time_header_present(client_choice):
    resp = client_choice.get("/health")
    assert "x-response-time-ms" in resp.headers

# ---------------------------------------------------------------------------
# Jev / TypeSafe Wire Compatibility
# ---------------------------------------------------------------------------

def test_jev_typesafe_wire_compatibility(client_choice):
    """
    Ensures that the strict Pydantic schemas allow extra fields and accept
    list-based criteria for choice/noul questions, matching the JS Jev SDK payload.
    """
    payload = {
        "model": "english",
        "state": "I need some help",
        "some_extra_metadata": "123",  # Extra top-level field (should be ignored)
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": ["technical", "billing"],  # List instead of dict
                "extra_question_field": True           # Extra question field
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is it urgent?",
                "criteria": ["No", "Yes"],             # List instead of dict
                "context": "Customer email"            # Extra question field
            }
        }
    }
    
    resp = client_choice.post("/v1/systemone", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "department" in data["answers"]
