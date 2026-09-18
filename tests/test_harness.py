"""Tests for Laya server and client harness."""

import pytest
import laya
from laya.server import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "laya"


def test_list_models(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    model_ids = [m["id"] for m in data["data"]]
    assert "laya-latest" in model_ids
    assert "jev-latest" in model_ids


def test_decide_helper():
    ans = laya.decide(
        "I need a refund on invoice #4411",
        choices=["billing", "technical", "sales"],
        instructions="Route ticket",
    )
    assert ans["type"] == "choice"
    assert ans["choice"] == "billing"
    assert "billing" in ans["probabilities"]
    assert ans["confidence"] > 0.0


def test_judge_helper():
    p = laya.judge(
        "Payment was declined by bank due to insufficient funds",
        instructions="Is this a payment failure?",
    )
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    assert p > 0.5


def test_rate_helper():
    ans = laya.rate(
        "The server is completely down and customer data is unreachable",
        criteria=["Cosmetic", "Minor issue", "Severe outage blocking users"],
        instructions="Rate severity",
    )
    assert ans["type"] == "score"
    assert ans["score"] > 1.0
