"""Deterministic regression coverage for heterogeneous Router batching."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

import pytest

from laya.router import Router


Q = {"intent": {"type": "noul", "instructions": "Relevant?"}}


def request(state, **overrides):
    return {"state": state, "questions": Q, **overrides}


@pytest.fixture
def fake_agent(monkeypatch):
    import laya.agent

    built = []
    calls = []

    class Agent:
        def __init__(self, repo, *, device, token, subfolder):
            self.checkpoint = subfolder or "english"
            built.append(self.checkpoint)

        def predict_batch(self, states, questions, batch_size=None):
            calls.append((self.checkpoint, list(states), questions, batch_size))
            if "raise" in states:
                raise RuntimeError("inference failed")
            return [
                {"model": "laya-rl-agent", "answers": {"seen": state}, "usage": {}}
                for state in states
            ]

        def system_one(self, state, questions):
            return self.predict_batch([state], questions)[0]

    monkeypatch.setattr(laya.agent, "Agent", Agent)
    return built, calls


@pytest.mark.parametrize("capacity,expected", [
    (1, (["english", "multilingual", "typed-decisions"], ["typed-decisions"])),
    (2, (["english", "multilingual", "typed-decisions"], ["multilingual", "typed-decisions"])),
    (3, (["english", "multilingual", "typed-decisions"],
         ["english", "multilingual", "typed-decisions"])),
])
def test_mixed_groups_keep_order_and_lru(fake_agent, capacity, expected):
    built, calls = fake_agent
    router = Router(max_loaded=capacity, auto_task_detection=True)
    typed = {key: {"type": "noul", "instructions": "?"} for key in
             ("action", "needs_review", "outcome", "risk", "urgency")}
    items = [request("English text one"), request("مرحبا"),
             request("English text two"), {"state": "decision", "questions": typed},
             request("forced", model="ml"), request("forced english", lang="en"),
             request("explicit task", task="typed_decisions")]
    decisions = router.route_batch(items)
    assert router.loaded == []
    results = router.predict_batch(items)
    assert [r["answers"]["seen"] for r in results] == [r["state"] for r in items]
    assert [r["routing"] for r in results] == list(map(dict, decisions))
    assert built == expected[0]
    assert router.loaded == expected[1]
    assert [(c[0], c[1]) for c in calls] == [
        ("english", ["English text one", "English text two", "forced english"]),
        ("multilingual", ["مرحبا", "forced"]),
        ("typed-decisions", ["decision"]),
        ("typed-decisions", ["explicit task"]),
    ]
    assert router.predict_many([]) == []
    assert router.route_batch(()) == []


@pytest.mark.parametrize("items,error,fragment", [
    (None, TypeError, "requests must be a sequence"),
    ({}, TypeError, "requests must be a sequence"),
    ("text", TypeError, "requests must be a sequence"),
    ([None], TypeError, "request 0"),
    ([{"questions": Q}], ValueError, "request 0 is missing required key 'state'"),
    ([{"state": "x"}], ValueError, "request 0 is missing required key 'questions'"),
    ([request("x"), {"state": "y", "questions": None}], TypeError, "request 1 'questions'"),
    ([request("x"), request("y", model="invalid")], ValueError, "unknown model"),
])
def test_invalid_batch_fails_before_loading(fake_agent, items, error, fragment):
    built, _ = fake_agent
    router = Router()
    with pytest.raises(error, match=fragment):
        router.predict_batch(items)
    assert built == []
    assert router.loaded == []


def test_inference_exception_propagates_and_cache_remains_consistent(fake_agent):
    built, calls = fake_agent
    router = Router(max_loaded=1)
    with pytest.raises(RuntimeError, match="inference failed"):
        router.predict_batch([request("first"), request("raise", lang="ar"),
                              request("unreached", lang="ar")])
    assert built == ["english", "multilingual"]
    assert [(c[0], c[1]) for c in calls] == [
        ("english", ["first"]),
        ("multilingual", ["raise", "unreached"]),
    ]
    assert router.loaded == ["multilingual"]
    assert list(router._agents) == router.loaded
    assert router.predict("after failure", Q, lang="ar")["answers"]["seen"] == "after failure"


def test_warm_cache_and_repeated_batches(fake_agent):
    built, _ = fake_agent
    router = Router(max_loaded=2)
    router.load("multilingual")
    result = router.predict_batch([request("en", model="english"),
                                   request("ar", model="multilingual"),
                                   request("en again", model="english")])
    assert [r["answers"]["seen"] for r in result] == ["en", "ar", "en again"]
    assert built == ["multilingual", "english"]
    assert router.loaded == ["english", "multilingual"]
    router.predict_batch([request("ar", model="multilingual"), request("en", model="english")])
    assert built == ["multilingual", "english"]



def test_same_checkpoint_same_questions_uses_one_agent_batch(fake_agent):
    _, calls = fake_agent
    router = Router(max_loaded=2)
    items = [
        request("one", model="english"),
        request("two", model="english"),
        request("three", model="english"),
    ]

    results = router.predict_batch(items, batch_size=2)

    assert [r["answers"]["seen"] for r in results] == ["one", "two", "three"]
    assert len(calls) == 1
    assert calls[0][0] == "english"
    assert calls[0][1] == ["one", "two", "three"]
    assert calls[0][2] == Q
    assert calls[0][3] == 2


def test_same_checkpoint_different_questions_split_agent_batches(fake_agent):
    _, calls = fake_agent
    router = Router(max_loaded=2)
    q2 = {"risk": {"type": "noul", "instructions": "Risky?"}}
    items = [
        {"state": "one", "questions": Q, "model": "english"},
        {"state": "two", "questions": q2, "model": "english"},
        {"state": "three", "questions": Q, "model": "english"},
    ]

    results = router.predict_batch(items)

    assert [r["answers"]["seen"] for r in results] == ["one", "two", "three"]
    assert len(calls) == 2
    assert calls[0][0] == "english"
    assert calls[0][1] == ["one", "three"]
    assert calls[0][2] == Q
    assert calls[1][0] == "english"
    assert calls[1][1] == ["two"]
    assert calls[1][2] == q2


def test_predict_batch_honours_hooks_timeout(fake_agent):
    def slow_hook(ctx):
        sleep(0.3)

    router = Router(hooks_timeout=0.05, on_predict_start=slow_hook)
    with pytest.raises(TimeoutError):
        router.predict_batch([request("one")])

    # A per-call override wins, exactly as it does on `predict`.
    router = Router(hooks_timeout=0.05, on_predict_start=slow_hook)
    results = router.predict_batch([request("one")], hooks_timeout=5.0)
    assert len(results) == 1


def test_route_batch_forwards_lang_guess(fake_agent):
    built, _ = fake_agent
    router = Router()

    decisions = router.route_batch([
        request("hola", lang_guess="es"),
        request("hello", lang_guess="en-US"),
    ])

    assert [decision.model for decision in decisions] == ["multilingual", "english"]
    assert built == []

def test_concurrent_batch_load_deduplicates(monkeypatch):
    import laya.agent

    built = []
    guard = Lock()

    class SlowAgent:
        def __init__(self, repo, *, device, token, subfolder):
            sleep(0.01)
            with guard:
                built.append(subfolder or "english")

        def predict_batch(self, states, questions, batch_size=None):
            return [
                {"model": "stub", "answers": {"seen": state}, "usage": {}}
                for state in states
            ]

        def system_one(self, state, questions):
            return self.predict_batch([state], questions)[0]

    monkeypatch.setattr(laya.agent, "Agent", SlowAgent)
    router = Router(max_loaded=2)
    start = Barrier(8)

    def worker(_):
        start.wait()
        return router.predict_batch([request("hello", model="english"),
                                     request("مرحبا", model="multilingual")])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(8)))
    assert built == ["english", "multilingual"]
    assert router.loaded == ["english", "multilingual"]
    assert all([r["answers"]["seen"] for r in result] == ["hello", "مرحبا"]
               for result in results)


def test_equal_questions_with_different_option_order_score_separately(fake_agent):
    """A question schema that arrives with a different key order must be scored with
    its own order. Dict equality ignores insertion order, but options are positional
    in the rendered sequence, so grouping reordered-but-equal schemas would make the
    second request's batched answers differ from its single-request answers."""
    built, calls = fake_agent
    ordered = {"intent": {"type": "choice", "instructions": "Pick one",
                          "criteria": {"zulu": "last", "alpha": "first"}}}
    reordered = {"intent": {"type": "choice", "instructions": "Pick one",
                            "criteria": {"alpha": "first", "zulu": "last"}}}
    requests = [{"state": "one", "questions": ordered},
                {"state": "two", "questions": reordered}]
    results = Router(max_loaded=1, default="english").predict_batch(requests)
    assert len(results) == 2
    # separate agent calls, each carrying its own caller's option order
    orders = [list(call[2]["intent"]["criteria"]) for call in calls]
    assert orders == [["zulu", "alpha"], ["alpha", "zulu"]]


def _lang_recording_router(monkeypatch, lang_temperatures):
    """A Router over a fake agent that records the `lang` each batch call received."""
    import laya.agent

    calls = []

    class Agent:
        def __init__(self, repo, *, device, token, subfolder):
            self.checkpoint = subfolder or "english"
            if lang_temperatures is not None:
                self.lang_temperatures = lang_temperatures

        def predict_batch(self, states, questions, batch_size=None, **overrides):
            calls.append({"n": len(states), "lang": overrides.get("lang")})
            return [{"model": "fake", "answers": {}} for _ in states]

        def system_one(self, state, questions, **overrides):
            calls.append({"n": 1, "lang": overrides.get("lang")})
            return {"model": "fake", "answers": {}}

    monkeypatch.setattr(laya.agent, "Agent", Agent)
    return Router(max_loaded=1, default="english"), calls


def test_lang_reaches_the_agent_when_it_carries_lang_temperatures(monkeypatch):
    """`predict` forwards the request's language so per-language temperatures apply; the batch
    path forwarded only the token budgets, so the same request scored differently depending on
    which entry point served it. The request is routed as German either way, which is what made
    the difference hard to see."""
    router, calls = _lang_recording_router(monkeypatch, {"de": [1.5, 1.5, 1.5]})
    router.predict_batch([request("a", lang="de"), request("b", lang="de"), request("c", lang="fr")])
    assert sorted(c["lang"] for c in calls) == ["de", "fr"]
    # requests sharing a language still share one forward pass
    assert {"n": 2, "lang": "de"} in calls
    assert {"n": 1, "lang": "fr"} in calls


def test_lang_is_not_added_to_the_group_key_without_lang_temperatures(monkeypatch):
    """An agent with no per-language temperatures does not use `lang`, so naming it must not
    split a group that shares one forward pass today. The explicit `model` keeps every request
    on one checkpoint, so the only thing that could split the group is the lang key."""
    router, calls = _lang_recording_router(monkeypatch, None)
    router.predict_batch([request("a", model="english", lang="de"),
                          request("b", model="english", lang="fr"),
                          request("c", model="english")])
    assert calls == [{"n": 3, "lang": None}]


def test_predict_and_predict_batch_pass_the_same_lang(monkeypatch):
    """The invariant that was violated: one request, either entry point, same language."""
    router, calls = _lang_recording_router(monkeypatch, {"de": [1.5, 1.5, 1.5]})
    router.predict("a", Q, model="english", lang="de")
    via_predict = calls[-1]["lang"]
    calls.clear()
    router.predict_batch([request("a", model="english", lang="de")])
    assert calls[-1]["lang"] == via_predict == "de"
