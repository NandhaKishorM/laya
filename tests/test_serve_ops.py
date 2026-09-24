"""Readiness, metrics, request ids and config validation for laya-serve.

Run: python -m pytest tests/test_serve_ops.py -q
"""
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from laya.serve import create_app  # noqa: E402
from laya.serving import ConfigError, ServingConfig  # noqa: E402

BODY = {"state": "hi", "questions": {"q": {"type": "noul", "instructions": "?"}}}


class FakeRouter:
    loaded = ["english"]

    def predict_batch(self, requests, batch_size=None):
        return [{"model": "laya-rl-agent", "answers": {},
                 "usage": {"input_tokens": 1, "output_tokens": 0},
                 "routing": {"model": "english"}} for _ in requests]

    def predict(self, state, questions, model=None, **kwargs):
        return {"model": "laya-rl-agent", "answers": {},
                "usage": {"input_tokens": 1, "output_tokens": 0}, "routing": {"model": "english"}}


def test_ready_after_startup(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig())) as client:
        body = client.get("/ready").json()
        assert body["status"] == "ready"
        assert body["loaded"] == ["english"]


def test_health(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig())) as client:
        assert client.get("/health").json()["status"] == "ok"


def test_metrics_counts(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig())) as client:
        client.post("/v1/systemone", json=BODY)
        text = client.get("/metrics").text
    assert 'laya_requests_total{model="english",status="ok"}' in text
    assert "laya_batches_total" in text
    assert "laya_batch_size_count" in text
    assert 'laya_state_backend{mode="local"} 1.0' in text
    assert "laya_inflight_total" in text


def test_drain_endpoint_stops_intake_but_keeps_liveness(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig(batch_window_ms=0))) as client:
        assert client.get("/ready").status_code == 200
        assert client.post("/drain").status_code == 200
        assert client.get("/ready").status_code == 503, "readiness must drop so the LB stops routing"
        assert client.get("/health").status_code == 200, "liveness stays up for the preStop window"
        rejected = client.post("/v1/systemone", json=BODY)
        assert rejected.status_code == 503 and rejected.headers.get("retry-after")
        assert client.post("/drain").status_code == 200, "drain must be idempotent"


def test_drain_endpoint_is_bearer_gated(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "s3cret")
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig())) as client:
        assert client.post("/drain").status_code == 401
        assert client.post("/drain", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_metrics_can_be_disabled(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig(metrics_enabled=False))) as client:
        assert client.get("/metrics").status_code == 404


def test_metrics_gated_by_api_key(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "s3cret")
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig())) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_bearer_scheme_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "s3cret")
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig())) as client:
        for header in ("bearer s3cret", "BEARER s3cret", "BeArEr s3cret", "Bearer  s3cret", "Bearer s3cret "):
            status = client.post("/v1/systemone", json=BODY, headers={"authorization": header}).status_code
            assert status == 200, header
        assert client.post("/v1/systemone", json=BODY, headers={"authorization": "Bearer wrong"}).status_code == 401
        assert client.post("/v1/systemone", json=BODY).status_code == 401


def test_preload_runs_in_background(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_PRELOAD", "1")

    class PreloadRouter(FakeRouter):
        def preload(self, names=None):
            time.sleep(0.4)
            self.loaded = ["english"]

    import laya.serve as serve_module

    monkeypatch.setattr(serve_module, "build_router", lambda preload=True: PreloadRouter())
    with TestClient(serve_module.create_app()) as client:
        assert client.get("/health").status_code == 200, "liveness must answer during preload"
        assert client.get("/ready").status_code == 503, "readiness must gate on preload"
        deadline = time.time() + 3
        while client.get("/ready").status_code != 200 and time.time() < deadline:
            time.sleep(0.05)
        assert client.get("/ready").status_code == 200


def test_warmup_runs_after_preload(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_PRELOAD", "1")
    monkeypatch.setenv("LAYA_WARMUP", "1")
    calls = []

    class WarmRouter(FakeRouter):
        def preload(self, names=None):
            self.loaded = ["english"]

        def predict(self, state, questions=None, model=None, **kwargs):
            calls.append(model)
            return super().predict(state, questions, model=model, **kwargs)

    import laya.serve as serve_module

    monkeypatch.setattr(serve_module, "build_router", lambda preload=True: WarmRouter())
    with TestClient(serve_module.create_app()) as client:
        deadline = time.time() + 3
        while client.get("/ready").status_code != 200 and time.time() < deadline:
            time.sleep(0.05)
        assert client.get("/ready").status_code == 200
    assert calls == ["english"], "one warmup pass per resident checkpoint"


def test_warmup_can_be_disabled(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_PRELOAD", "1")
    monkeypatch.setenv("LAYA_WARMUP", "0")
    calls = []

    class WarmRouter(FakeRouter):
        def preload(self, names=None):
            self.loaded = ["english"]

        def predict(self, state, questions=None, model=None, **kwargs):
            calls.append(model)
            return super().predict(state, questions, model=model, **kwargs)

    import laya.serve as serve_module

    monkeypatch.setattr(serve_module, "build_router", lambda preload=True: WarmRouter())
    with TestClient(serve_module.create_app()) as client:
        deadline = time.time() + 3
        while client.get("/ready").status_code != 200 and time.time() < deadline:
            time.sleep(0.05)
    assert calls == []


def test_preload_failure_fails_health(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_PRELOAD", "1")

    class BrokenPreload(FakeRouter):
        def preload(self, names=None):
            raise RuntimeError("weights missing")

    import laya.serve as serve_module

    monkeypatch.setattr(serve_module, "build_router", lambda preload=True: BrokenPreload())
    with TestClient(serve_module.create_app()) as client:
        deadline = time.time() + 3
        while client.get("/health").status_code == 200 and time.time() < deadline:
            time.sleep(0.02)
        assert client.get("/health").status_code == 503
        assert client.get("/ready").status_code == 503


def test_request_id_header(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig())) as client:
        response = client.post("/v1/systemone", json=BODY)
        assert response.headers.get("x-request-id")
        forwarded = client.post("/v1/systemone", json=BODY, headers={"x-request-id": "abc123"})
        assert forwarded.headers["x-request-id"] == "abc123"


def test_config_rejects_bad_values(monkeypatch):
    for name, value in (("LAYA_BATCH_MAX", "abc"), ("LAYA_QUEUE_MAX", "0"),
                        ("LAYA_BATCH_WINDOW_MS", "-1"), ("LAYA_REQUEST_TIMEOUT_S", "soon"),
                        ("LAYA_INFER_WORKERS", "0"), ("LAYA_LOG_FORMAT", "yaml"),
                        ("LAYA_METRICS", "enabled")):
        monkeypatch.setenv(name, value)
        with pytest.raises(ConfigError):
            ServingConfig.from_env()
        monkeypatch.delenv(name)


def test_preload_typo_fails_fast(monkeypatch):
    monkeypatch.setenv("LAYA_PRELOAD", "maybe")

    import laya.serve as serve_module

    monkeypatch.setattr(serve_module, "build_router", lambda preload=True: FakeRouter())
    with pytest.raises(ConfigError):
        serve_module.create_app()


def test_proxy_config(monkeypatch):
    monkeypatch.setenv("LAYA_PROXY_HEADERS", "1")
    monkeypatch.setenv("LAYA_FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    config = ServingConfig.from_env()
    assert config.proxy_headers is True
    assert config.forwarded_allow_ips == "10.0.0.0/8"
    monkeypatch.setenv("LAYA_PROXY_HEADERS", "nope")
    with pytest.raises(ConfigError):
        ServingConfig.from_env()


def test_shared_state_requires_redis_extra(monkeypatch):
    import importlib.util

    from laya.serve import create_app
    from laya.state import build_state

    if importlib.util.find_spec("redis") is not None:
        pytest.skip("redis installed; the missing-extra path is not exercisable")
    config = ServingConfig(state_url="redis://127.0.0.1:6379/0")
    with pytest.raises(ImportError):
        build_state(config)
    with pytest.raises(ConfigError):
        create_app(router=FakeRouter(), config=config)


def test_state_failure_policy_config(monkeypatch):
    monkeypatch.setenv("LAYA_STATE_FAILURE_POLICY", "bogus")
    with pytest.raises(ConfigError):
        ServingConfig.from_env()
    monkeypatch.setenv("LAYA_STATE_FAILURE_POLICY", "open")
    assert ServingConfig.from_env().state_failure_policy == "open"


def test_config_defaults(monkeypatch):
    for name in ("LAYA_BATCH_WINDOW_MS", "LAYA_BATCH_MAX", "LAYA_QUEUE_MAX", "LAYA_REQUEST_TIMEOUT_S",
                 "LAYA_SHUTDOWN_GRACE_S", "LAYA_INFER_WORKERS", "LAYA_METRICS", "LAYA_LOG_FORMAT"):
        monkeypatch.delenv(name, raising=False)
    config = ServingConfig.from_env()
    assert (config.batch_window_ms, config.batch_max, config.queue_max) == (10, 16, 64)
    assert config.infer_workers == 1 and config.metrics_enabled and config.log_format == "text"


def test_registry_isolation(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    # Two apps in one process must not collide on the default Prometheus registry.
    with TestClient(create_app(router=FakeRouter(), config=ServingConfig())) as a:
        with TestClient(create_app(router=FakeRouter(), config=ServingConfig())) as b:
            assert a.get("/metrics").status_code == 200
            assert b.get("/metrics").status_code == 200
