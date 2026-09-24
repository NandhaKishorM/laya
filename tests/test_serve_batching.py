"""Batching, backpressure and timeout behavior of laya-serve. No GPU, a fake Router.

The HTTP surface is driven through ``httpx.ASGITransport`` so many requests can be in flight
at once (``TestClient`` serializes through its portal). Every test wraps its coroutine in
``asyncio.run`` so no pytest plugin is required.

Run: python -m pytest tests/test_serve_batching.py -q
"""
import asyncio
import re
import time

import pytest

httpx = pytest.importorskip("httpx")

from laya.serve import create_app  # noqa: E402
from laya.serving import Batcher, Metrics, RequestQueueFull, ServingConfig  # noqa: E402

BODY = {"state": "hi", "questions": {"q": {"type": "noul", "instructions": "?"}}}


def _ok(routing="english"):
    return {"model": "laya-rl-agent", "answers": {"q": {"type": "noul", "noul": 0.7, "confidence": 0.7}},
            "usage": {"input_tokens": 3, "output_tokens": 0}, "routing": {"model": routing}}


class FakeRouter:
    """Counts forward passes and answers every request."""

    loaded = ["english"]

    def __init__(self, delay=0.0):
        self.batch_calls = 0
        self.single_calls = 0
        self.seen_models = []
        self._delay = delay

    def predict_batch(self, requests, batch_size=None):
        self.batch_calls += 1
        if self._delay:
            time.sleep(self._delay)
        self.seen_models.extend(r.get("model") for r in requests)
        states = [r.get("state") for r in requests]
        if "poison" in states:
            raise RuntimeError("batch poisoned")
        return [_ok() for _ in requests]

    def predict(self, state, questions, model=None, **kwargs):
        self.single_calls += 1
        if state == "poison":
            raise ValueError("question %r: bad state" % "q")
        return _ok()


def _client(router, config):
    transport = httpx.ASGITransport(app=create_app(router=router, config=config))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _run(coro):
    return asyncio.run(coro)


def test_concurrent_requests_are_batched():
    fake = FakeRouter(delay=0.05)

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=30, batch_max=16)) as client:
            return await asyncio.gather(*[client.post("/v1/systemone", json=BODY) for _ in range(8)])

    responses = _run(scenario())
    assert all(r.status_code == 200 for r in responses)
    assert fake.batch_calls == 1, "8 concurrent requests should share one forward pass"
    assert fake.single_calls == 0


def test_batch_max_caps_the_batch():
    fake = FakeRouter(delay=0.05)

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=50, batch_max=2)) as client:
            return await asyncio.gather(*[client.post("/v1/systemone", json=BODY) for _ in range(6)])

    responses = _run(scenario())
    assert all(r.status_code == 200 for r in responses)
    assert fake.batch_calls >= 3, "batch_max=2 cannot serve 6 requests in one pass"


def test_zero_window_dispatches_immediately():
    fake = FakeRouter()

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=0)) as client:
            return await client.post("/v1/systemone", json=BODY)

    assert _run(scenario()).status_code == 200
    assert fake.batch_calls == 1


def test_backpressure_returns_503_with_retry_after():
    fake = FakeRouter(delay=0.5)

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=500, batch_max=16, queue_max=1,
                                               request_timeout_s=5)) as client:
            return await asyncio.gather(*[client.post("/v1/systemone", json=BODY) for _ in range(6)])

    responses = _run(scenario())
    rejected = [r for r in responses if r.status_code == 503]
    assert rejected, "a full queue must reject with 503"
    assert all("retry-after" in r.headers for r in rejected)


def test_timeout_returns_504():
    fake = FakeRouter(delay=1.0)

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=5, request_timeout_s=0.01)) as client:
            return await client.post("/v1/systemone", json=BODY)

    assert _run(scenario()).status_code == 504


def test_batch_failure_falls_back_per_request():
    fake = FakeRouter()

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=30, batch_max=16)) as client:
            return await asyncio.gather(
                client.post("/v1/systemone", json=BODY),
                client.post("/v1/systemone", json={"state": "poison", "questions": BODY["questions"]}))

    good, bad = _run(scenario())
    assert good.status_code == 200
    # the poison request fails alone; the batch fell back to per-request calls
    assert bad.status_code == 422
    assert fake.single_calls == 2


def test_client_limit_returns_429():
    fake = FakeRouter(delay=0.3)

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=0, client_max_inflight=1,
                                               request_timeout_s=10)) as client:
            first, second = await asyncio.gather(
                client.post("/v1/systemone", json=BODY, headers={"x-laya-client": "tenant-a"}),
                client.post("/v1/systemone", json=BODY, headers={"x-laya-client": "tenant-a"}))
            text = (await client.get("/metrics")).text
            return first, second, text

    first, second, text = _run(scenario())
    statuses = sorted([first.status_code, second.status_code])
    assert statuses == [200, 429], statuses
    limited = first if first.status_code == 429 else second
    assert limited.headers.get("retry-after")
    assert 'laya_rejections_total{reason="client_limit"} 1.0' in text


def test_queue_oldest_and_shed_reason_metrics():
    fake = FakeRouter(delay=0.5)

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=0, batch_max=1, queue_max=2,
                                               request_timeout_s=10)) as client:
            tasks = [asyncio.ensure_future(client.post("/v1/systemone", json=BODY)) for _ in range(5)]
            await asyncio.sleep(0.15)          # one dispatching, two queued, two rejected
            text = (await client.get("/metrics")).text
            await asyncio.gather(*tasks, return_exceptions=True)
            return text

    text = _run(scenario())
    assert 'laya_rejections_total{reason="queue_full"}' in text
    oldest = re.search(r"laya_queue_oldest_seconds ([0-9.]+)", text)
    assert oldest and float(oldest.group(1)) > 0, text


def test_circuit_breaker_opens_then_recovers():
    class FlakyRouter(FakeRouter):
        def __init__(self):
            super().__init__()
            self.failures = 0

        def predict_batch(self, requests, batch_size=None):
            raise RuntimeError("engine down")

        def predict(self, state, questions=None, model=None, **kwargs):
            if self.failures < 2:
                self.failures += 1
                raise TypeError("engine down")
            return _ok()

    fake = FlakyRouter()

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=0, batch_max=1, breaker_threshold=2,
                                               breaker_cooldown_s=0.3, request_timeout_s=10)) as client:
            first = (await client.post("/v1/systemone", json=BODY)).status_code
            second = (await client.post("/v1/systemone", json=BODY)).status_code
            open_status = (await client.post("/v1/systemone", json=BODY)).status_code
            text = (await client.get("/metrics")).text
            await asyncio.sleep(0.35)          # cooldown elapses
            probe = (await client.post("/v1/systemone", json=BODY)).status_code
            after = (await client.post("/v1/systemone", json=BODY)).status_code
            return first, second, open_status, probe, after, text

    first, second, open_status, probe, after, text = _run(scenario())
    assert (first, second, open_status) == (500, 500, 503)
    assert 'laya_rejections_total{reason="circuit_open"} 1.0' in text
    assert "laya_circuits_open 1.0" in text, "the open breaker must be visible on /metrics"
    assert (probe, after) == (200, 200), "half-open probes must close the breaker on success"


def test_half_open_admits_one_probe_at_a_time():
    class ProbeRouter(FakeRouter):
        def __init__(self):
            super().__init__(delay=0.25)
            self.healthy = False

        def predict_batch(self, requests, batch_size=None):
            raise RuntimeError("engine down")

        def predict(self, state, questions=None, model=None, **kwargs):
            if not self.healthy:
                raise TypeError("engine down")
            time.sleep(self._delay)
            return _ok()

    fake = ProbeRouter()

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=0, batch_max=1, breaker_threshold=1,
                                               breaker_cooldown_s=0.15, request_timeout_s=10)) as client:
            opened = (await client.post("/v1/systemone", json=BODY)).status_code
            await asyncio.sleep(0.2)
            fake.healthy = True
            first, second = await asyncio.gather(client.post("/v1/systemone", json=BODY),
                                                 client.post("/v1/systemone", json=BODY))
            return opened, sorted([first.status_code, second.status_code])

    opened, pair = _run(scenario())
    assert opened == 500
    assert pair == [200, 503], "half-open must admit exactly one probe, not a flood"


def test_half_open_requires_success_streak():
    class RecoveringRouter(FakeRouter):
        def __init__(self):
            super().__init__()
            self.healthy = False

        def predict_batch(self, requests, batch_size=None):
            raise RuntimeError("engine down")

        def predict(self, state, questions=None, model=None, **kwargs):
            if not self.healthy:
                raise TypeError("engine down")
            return _ok()

    fake = RecoveringRouter()

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=0, batch_max=1, breaker_threshold=1,
                                               breaker_cooldown_s=0.1, breaker_probe_successes=2,
                                               request_timeout_s=10)) as client:
            opened = (await client.post("/v1/systemone", json=BODY)).status_code
            blocked = (await client.post("/v1/systemone", json=BODY)).status_code
            await asyncio.sleep(0.15)
            fake.healthy = True
            probe1 = (await client.post("/v1/systemone", json=BODY)).status_code
            probe2 = (await client.post("/v1/systemone", json=BODY)).status_code
            after = (await client.post("/v1/systemone", json=BODY)).status_code
            return opened, blocked, probe1, probe2, after

    assert _run(scenario()) == (500, 503, 200, 200, 200)


def test_breaker_is_per_client():
    class FlakyRouter(FakeRouter):
        def predict_batch(self, requests, batch_size=None):
            raise RuntimeError("engine down")

        def predict(self, state, questions=None, model=None, **kwargs):
            if state == "unstable":
                raise TypeError("engine down")
            return _ok()

    fake = FlakyRouter()
    poison = {"state": "unstable", "questions": BODY["questions"]}

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=0, batch_max=1, breaker_threshold=1,
                                               breaker_cooldown_s=30, request_timeout_s=10)) as client:
            a1 = (await client.post("/v1/systemone", json=poison,
                                    headers={"x-laya-client": "tenant-a"})).status_code
            a2 = (await client.post("/v1/systemone", json=poison,
                                    headers={"x-laya-client": "tenant-a"})).status_code
            b1 = (await client.post("/v1/systemone", json=BODY,
                                    headers={"x-laya-client": "tenant-b"})).status_code
            return a1, a2, b1

    # tenant-a's poison opens tenant-a's breaker only; tenant-b keeps being served
    assert _run(scenario()) == (500, 503, 200)


def test_drain_fails_queued_requests_fast():
    from concurrent.futures import ThreadPoolExecutor

    from laya.state import LocalState

    async def scenario():
        config = ServingConfig(queue_max=8, shutdown_grace_s=0.01)
        batcher = Batcher(FakeRouter(), config, Metrics(enabled=False),
                          ThreadPoolExecutor(max_workers=1), LocalState(config))
        futures = [await batcher.submit({"state": "x", "questions": {"q": {}}}) for _ in range(3)]
        await batcher.drain()
        return await asyncio.gather(*futures, return_exceptions=True)

    results = _run(scenario())
    assert all(isinstance(result, RequestQueueFull) for result in results)


def test_model_override_survives_batching():
    fake = FakeRouter()

    async def scenario():
        async with _client(fake, ServingConfig(batch_window_ms=30)) as client:
            return await client.post("/v1/systemone", json={**BODY, "model": "multilingual"})

    assert _run(scenario()).status_code == 200
    assert fake.seen_models == ["multilingual"]
