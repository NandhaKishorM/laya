"""Native auto/review routing contracts, without model weights or required extras."""
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.integrations.langchain import LayaRouter  # noqa: E402


CRITERIA = {"billing": "invoices", "tech": "bugs"}


def decision(**answer):
    return {"answers": {"route": answer}}


class Runner:
    def __init__(self, result=None, responses=None):
        self.result = result
        self.responses = responses or {}
        self.calls = []
        self.batch_calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return self.responses.get(state, self.result)

    def predict_batch(self, states, questions, **kwargs):
        self.batch_calls.append((states, questions, kwargs))
        return [self.responses.get(state, self.result) for state in states]


def native(runner, **kwargs):
    return LayaRouter(CRITERIA, agent=runner, fallback="human", decision_policy="native", **kwargs)


class NativePolicyTests(unittest.TestCase):
    def test_confidence_default_preserves_legacy_choices_and_gating(self):
        runner = Runner(decision(choice="billing", route="review", answer_confidence=0.99))
        router = LayaRouter(CRITERIA, agent=runner, fallback="human", confidence_threshold=0.8)
        self.assertEqual(router.invoke("x"), "billing")
        runner.result = decision(choice="tech", answer_confidence=0.2)
        self.assertEqual(router.invoke("x"), "human")
        runner.result = decision(choice="tech", confidence=0.9)
        self.assertEqual(router.invoke("x"), "tech")

    def test_review_precedes_choice_and_confidence(self):
        for extra in ({}, {"choice": None}, {"choice": []}, {"choice": "unmapped"},
                      {"choice": "billing", "answer_confidence": 0.99},
                      {"choice": "billing", "confidence": "unreadable"}):
            with self.subTest(extra=extra):
                result = decision(route="review", **extra)
                router = native(Runner(result))
                self.assertEqual(router.invoke("x"), "human")
                self.assertIs(router.last_decision, result)

    def test_auto_uses_the_final_backend_policy_without_a_second_gate(self):
        result = decision(route="auto", choice="tech", answer_confidence=0.0,
                          abstention="abstained", prediction_set=["billing", "tech"])
        self.assertEqual(native(Runner(result)).invoke("x"), "tech")

    def test_invalid_auto_destinations_fail_closed(self):
        for answer in ({"route": "auto"}, *[
                {"route": "auto", "choice": choice} for choice in (None, True, 3, [], {}, "other", "")]):
            with self.subTest(answer=answer), self.assertRaisesRegex(RuntimeError, "Malformed decision response"):
                native(Runner(decision(**answer))).invoke("x")

    def test_missing_or_invalid_native_routes_are_not_implicit_auto(self):
        for answer in ({"choice": "billing"}, *[
                {"route": route, "choice": "billing"} for route in (None, True, 1, [], {}, "AUTO", "unknown", "")]):
            with self.subTest(answer=answer), self.assertRaisesRegex(RuntimeError, "Malformed decision response"):
                native(Runner(decision(**answer))).invoke("x")

    def test_malformed_answers_use_the_adapter_error(self):
        for result in (None, [], True, {}, {"answers": None}, {"answers": []}, {"answers": {}},
                       {"answers": {"route": None}}, {"answers": {"route": "review"}}):
            with self.subTest(result=result), self.assertRaisesRegex(RuntimeError, "Malformed decision response"):
                native(Runner(result)).invoke("x")

    def test_native_requires_an_explicit_nonempty_fallback(self):
        for fallback in (None, "", " \t\n", False, [], {}):
            with self.subTest(fallback=fallback), self.assertRaises(ValueError):
                LayaRouter(CRITERIA, agent=Runner(), decision_policy="native", fallback=fallback)

    def test_unknown_policies_are_configuration_errors(self):
        for policy in (None, False, [], {}, "", "AUTO", "native "):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                LayaRouter(CRITERIA, agent=Runner(), decision_policy=policy, fallback="human")

    def test_native_rejects_a_nonzero_wrapper_threshold(self):
        for threshold in (-0.1, 0.8, float("inf"), float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                native(Runner(), confidence_threshold=threshold)

    def test_configuration_changes_are_rechecked_before_inference(self):
        for field, value in (("decision_policy", "typo"), ("fallback", " "), ("confidence_threshold", 0.5)):
            with self.subTest(field=field):
                runner = Runner(decision(route="auto", choice="billing"))
                router = native(runner)
                setattr(router, field, value)
                with self.assertRaises(ValueError):
                    router.invoke("x")
                with self.assertRaises(ValueError):
                    router.batch(["x"])
                self.assertEqual(runner.calls, [])
                self.assertEqual(runner.batch_calls, [])

    def test_backend_controls_still_reach_single_and_batch_calls(self):
        runner = Runner(decision(route="review"))
        router = native(runner, min_confidence=0.4, max_len=128, head_max_len=64, lang="en")
        self.assertEqual(router.invoke("x"), "human")
        self.assertEqual(router.batch(["x", "y"]), ["human", "human"])
        for _, questions, controls in runner.calls + runner.batch_calls:
            self.assertEqual(questions["route"]["criteria"], CRITERIA)
            self.assertEqual(controls, {"min_confidence": 0.4, "max_len": 128, "head_max_len": 64, "lang": "en"})

    def test_shared_batch_preserves_order_and_each_native_decision(self):
        runner = Runner(responses={
            "review": decision(route="review", choice="billing", answer_confidence=0.99),
            "auto": decision(route="auto", choice="tech"),
            "broken": decision(choice="billing"),
        })
        router = native(runner)
        self.assertEqual(router.batch(["review", "auto", "review"]), ["human", "tech", "human"])
        self.assertEqual(len(runner.batch_calls), 1)
        self.assertEqual(runner.calls, [])
        self.assertEqual(router.batch([]), [])
        with self.assertRaisesRegex(RuntimeError, "Malformed decision response"):
            router.batch(["auto", "broken", "review"])
        outcomes = router.batch(["auto", "broken", "review"], return_exceptions=True)
        self.assertEqual(outcomes[0], "tech")
        self.assertIsInstance(outcomes[1], RuntimeError)
        self.assertEqual(outcomes[2], "human")

    def test_backend_errors_are_not_disguised_as_review(self):
        failure = RuntimeError("backend failed")

        class Broken:
            def predict(self, *args, **kwargs):
                raise failure

        with self.assertRaises(RuntimeError) as caught:
            native(Broken()).invoke("x")
        self.assertIs(caught.exception, failure)

    def test_vega_policy_survives_the_adapter_without_model_weights(self):
        from laya.vega import VegaAgent

        class Engine:
            mode = "auto"

            def decide(self, state, questions, **kwargs):
                answer = {"choice": "billing", "probs": {"billing": 0.99, "tech": 0.01},
                          "confidence": 0.99, "prediction_set": ["billing"],
                          "abstain": False, "unbound": False, "readout": "engine"}
                if state == "ambiguous":
                    answer["prediction_set"] = ["billing", "tech"]
                elif state == "abstain":
                    answer["abstain"] = True
                elif state == "unbound":
                    answer["unbound"] = True
                elif state == "band_review":
                    answer.update(route="review", route_reason="fitted band requires review")
                return {"answers": {qid: dict(answer) for qid in questions}, "n_tokens": 8}

        agent = object.__new__(VegaAgent)
        agent.vega, agent.calibration = Engine(), None
        agent.alpha = agent.risk = "0.1"
        agent._lock = threading.RLock()
        agent.model_id, agent.hooks = "weight-free-test", []
        agent.hooks_raise, agent.hooks_timeout = True, None
        states = ["auto", "ambiguous", "abstain", "unbound", "band_review"]
        router = native(agent)
        # Current Vega reports unbound but does not use that raw flag alone as a review gate.
        expected = ["billing", "human", "human", "billing", "human"]
        self.assertEqual([router.invoke(state) for state in states], expected)
        self.assertEqual(router.batch(states), expected)
        self.assertEqual(native(agent, min_confidence=1.0).invoke("auto"), "human")
        legacy = LayaRouter(CRITERIA, agent=agent, fallback="human", confidence_threshold=0.8)
        self.assertEqual(legacy.batch(states), ["billing"] * 5)


if __name__ == "__main__":
    unittest.main()
