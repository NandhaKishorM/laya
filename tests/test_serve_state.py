"""Unit tests for the pluggable admission/breaker state store (laya.state).

Run: python -m pytest tests/test_serve_state.py -q
"""
import asyncio
import time

import pytest

httpx = pytest.importorskip("httpx")

from laya.serve import create_app  # noqa: E402
from laya.serving import ServingConfig  # noqa: E402
from laya.state import BreakerDecision, LocalState  # noqa: E402

BODY = {"state": "hello", "questions": {"q": {"type": "noul", "instructions": "?"}}}


def _run(coro):
    return asyncio.run(coro)


def test_local_admission_parity():
    async def scenario():
        state = LocalState(ServingConfig(client_max_inflight=2))
        assert await state.acquire("a", 2) is True
        assert await state.acquire("a", 2) is True
        assert await state.acquire("a", 2) is False, "third in-flight is over the cap"
        await state.release("a")
        assert await state.acquire("a", 2) is True
        for _ in range(5):                     # limit <= 0 means unlimited and is not tracked
            assert await state.acquire("b", 0) is True
        assert (await state.snapshot())["clients_active"] == 1  # only the tracked cap
        return True

    assert _run(scenario())


def test_local_breaker_state_machine():
    config = ServingConfig(breaker_threshold=2, breaker_cooldown_s=0.1, breaker_probe_successes=2)

    async def scenario():
        state = LocalState(config)
        assert (await state.breaker_allow("k", 2, 0.1, 2)).allowed
        await state.breaker_failure("k", 2, 0.1)          # 1st failure, still closed
        assert (await state.breaker_allow("k", 2, 0.1, 2)).allowed
        await state.breaker_failure("k", 2, 0.1)          # threshold -> open
        denied = await state.breaker_allow("k", 2, 0.1, 2)
        assert not denied.allowed and denied.retry_after >= 1

        await asyncio.sleep(0.15)                          # cooldown elapses
        probe = await state.breaker_allow("k", 2, 0.1, 2)
        assert probe.allowed and probe.probe
        assert not (await state.breaker_allow("k", 2, 0.1, 2)).allowed, "one probe at a time"
        await state.breaker_success("k", 2)                # 1 of 2 successes; still half-open
        assert (await state.snapshot())["circuits_tracked"] == 1
        probe2 = await state.breaker_allow("k", 2, 0.1, 2)
        assert probe2.allowed and probe2.probe
        await state.breaker_success("k", 2)                # streak met -> closed and reaped
        assert (await state.snapshot())["circuits_tracked"] == 0
        return True

    assert _run(scenario())


def test_breaker_ttl_explicit_and_disabled():
    async def scenario():
        expiring = LocalState(ServingConfig(breaker_threshold=1, breaker_cooldown_s=0.01,
                                            breaker_entry_ttl_s=0.05))
        await expiring.breaker_failure("k", 1, 0.01)
        assert (await expiring.snapshot())["circuits_tracked"] == 1
        time.sleep(0.06)
        expiring._maybe_sweep(time.perf_counter())
        assert (await expiring.snapshot())["circuits_tracked"] == 0, "TTL must reap the entry"

        kept = LocalState(ServingConfig(breaker_threshold=1, breaker_cooldown_s=0.01,
                                        breaker_entry_ttl_s=0))
        await kept.breaker_failure("k", 1, 0.01)
        time.sleep(0.06)
        kept._maybe_sweep(time.perf_counter())
        assert (await kept.snapshot())["circuits_tracked"] == 1, "0 disables the TTL"
        return True

    assert _run(scenario())


def test_effective_ttl_auto_defaults_to_floor():
    assert LocalState(ServingConfig(breaker_cooldown_s=5.0))._effective_ttl() == 60.0
    assert LocalState(ServingConfig(breaker_cooldown_s=120.0))._effective_ttl() == 120.0


class FakeState:
    """Records every call, so a test can prove the app routes through the injected store."""

    def __init__(self):
        self.ops = []

    async def acquire(self, key, limit):
        self.ops.append(("acquire", key))
        return True

    async def release(self, key):
        self.ops.append(("release", key))

    async def breaker_allow(self, key, threshold, cooldown_s, probe_successes):
        self.ops.append(("allow", key))
        return BreakerDecision(True)

    async def breaker_success(self, key, probe_successes):
        self.ops.append(("success", key))

    async def breaker_failure(self, key, threshold, cooldown_s):
        self.ops.append(("failure", key))

    async def snapshot(self):
        return {"clients_active": 0.0, "circuits_open": 0.0, "circuits_tracked": 0.0}

    async def close(self):
        self.ops.append(("close",))


class TinyRouter:
    loaded = ["english"]

    def predict_batch(self, requests, batch_size=None):
        return [{"model": "m", "answers": {}, "usage": {}, "routing": {"model": "english"}}
                for _ in requests]

    def predict(self, state, questions=None, model=None, **kwargs):
        return {"model": "m", "answers": {}, "usage": {}, "routing": {"model": "english"}}


def test_injected_state_is_used_by_serve_and_batcher():
    fake = FakeState()

    async def scenario():
        app = create_app(router=TinyRouter(), config=ServingConfig(batch_window_ms=0), state=fake)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.post("/v1/systemone", json=BODY)).status_code == 200
            await client.get("/metrics")
        return True

    assert _run(scenario())
    kinds = [op[0] for op in fake.ops]
    for expected in ("acquire", "release", "allow", "success"):
        assert expected in kinds, "state store missing %s call: %s" % (expected, kinds)
