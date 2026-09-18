"""P1: async client and retry policy."""
import asyncio

import pytest

from conftest import FakeAgent, noul
from laya.aio import AsyncAgent, RetryPolicy, with_retries
from laya.errors import InferenceError, QuestionError

NOUL_Q = {"type": "noul", "instructions": "is it spam?"}


def run(coro):
    return asyncio.run(coro)


class TestRetryPolicy:
    def test_backoff_grows(self):
        p = RetryPolicy(initial_backoff=0.1, multiplier=2.0, jitter=False)
        assert p.backoff_for(1) == pytest.approx(0.1)
        assert p.backoff_for(2) == pytest.approx(0.2)
        assert p.backoff_for(3) == pytest.approx(0.4)

    def test_backoff_is_capped(self):
        p = RetryPolicy(initial_backoff=1.0, multiplier=10.0, max_backoff=2.0, jitter=False)
        assert p.backoff_for(5) == pytest.approx(2.0)

    def test_jitter_stays_within_bounds(self):
        p = RetryPolicy(initial_backoff=1.0, multiplier=1.0, jitter=True)
        for _ in range(50):
            assert 0.5 <= p.backoff_for(1) <= 1.0

    def test_retries_only_listed_types(self):
        p = RetryPolicy(max_attempts=3)
        assert p.should_retry(InferenceError("x"), 1)
        assert not p.should_retry(QuestionError("x"), 1), "deterministic errors must not retry"

    def test_stops_at_max_attempts(self):
        p = RetryPolicy(max_attempts=2)
        assert p.should_retry(InferenceError("x"), 1)
        assert not p.should_retry(InferenceError("x"), 2)

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)


class TestWithRetries:
    def test_succeeds_after_transient_failure(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise InferenceError("transient")
            return "ok"

        policy = RetryPolicy(max_attempts=5, initial_backoff=0.001, jitter=False)
        assert with_retries(flaky, policy) == "ok"
        assert len(calls) == 3

    def test_gives_up_and_reraises(self):
        def always():
            raise InferenceError("down")

        with pytest.raises(InferenceError):
            with_retries(always, RetryPolicy(max_attempts=2, initial_backoff=0.001, jitter=False))

    def test_does_not_retry_deterministic_errors(self):
        calls = []

        def bad():
            calls.append(1)
            raise QuestionError("malformed")

        with pytest.raises(QuestionError):
            with_retries(bad, RetryPolicy(max_attempts=5, initial_backoff=0.001))
        assert len(calls) == 1, "must fail fast"


class TestAsyncAgent:
    def test_system_one_awaits(self):
        agent = AsyncAgent(FakeAgent(lambda s, q: {"x": noul(0.9)}))
        out = run(agent.system_one("state", {"x": NOUL_Q}))
        assert out["answers"]["x"].noul == 0.9

    def test_batch_covers_every_state(self):
        agent = AsyncAgent(FakeAgent(lambda s, q: {"x": noul(len(s) / 10)}))
        out = run(agent.batch(["a", "bb", "ccc"], {"x": NOUL_Q}))
        assert [r["answers"]["x"].noul for r in out] == [0.1, 0.2, 0.3]

    def test_batch_respects_concurrency_limit(self):
        import threading
        live, peak, lock = [0], [0], threading.Lock()

        def slow(s, q):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            import time as t; t.sleep(0.01)
            with lock:
                live[0] -= 1
            return {"x": noul(0.5)}

        agent = AsyncAgent(FakeAgent(slow))
        run(agent.batch(["s"] * 12, {"x": NOUL_Q}, max_concurrency=3))
        assert peak[0] <= 3

    def test_batch_can_return_exceptions(self):
        def boom(s, q):
            if s == "bad":
                raise QuestionError("nope")
            return {"x": noul(0.5)}

        agent = AsyncAgent(FakeAgent(boom))
        out = run(agent.batch(["ok", "bad"], {"x": NOUL_Q}, return_exceptions=True))
        assert isinstance(out[1], QuestionError)

    def test_batch_rejects_bad_concurrency(self):
        agent = AsyncAgent(FakeAgent())
        with pytest.raises(ValueError):
            run(agent.batch(["a"], {"x": NOUL_Q}, max_concurrency=0))

    def test_retries_through_async_path(self):
        calls = []

        def flaky(s, q):
            calls.append(1)
            if len(calls) < 2:
                raise InferenceError("transient")
            return {"x": noul(0.7)}

        agent = AsyncAgent(
            FakeAgent(flaky), retry=RetryPolicy(max_attempts=3, initial_backoff=0.001, jitter=False)
        )
        out = run(agent.system_one("s", {"x": NOUL_Q}))
        assert out["answers"]["x"].noul == 0.7 and len(calls) == 2

    def test_exposes_wrapped_agent(self):
        inner = FakeAgent()
        assert AsyncAgent(inner).agent is inner

    def test_concurrent_calls_do_not_block_each_other(self):
        agent = AsyncAgent(FakeAgent(lambda s, q: {"x": noul(0.5)}))

        async def main():
            return await asyncio.gather(*(agent.system_one(str(i), {"x": NOUL_Q}) for i in range(5)))

        assert len(run(main())) == 5
