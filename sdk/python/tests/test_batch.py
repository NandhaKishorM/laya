import copy
import json
import unittest

from laya_client import (
    LayaAPIError, LayaClient, LayaConnectionError, LayaResponseError,
    LayaTimeoutError, LayaValidationError,
)
from _support import PREDICTION, QUESTIONS, Server


def batch_payload(count=2):
    results = []
    for i in range(count):
        result = copy.deepcopy(PREDICTION)
        result["usage"]["input_tokens"] += i
        result["answers"]["refund"]["noul"] = (i + 1) / (count + 1)
        results.append(result)
    return {
        "results": results,
        "total_usage": {"input_tokens": sum(r["usage"]["input_tokens"] for r in results), "output_tokens": 0},
    }


class BatchTests(unittest.TestCase):
    def test_batch_request_and_order(self):
        states = ["hello नमस्ते", {"text": "billing", "flag": False}, ["technical", {"count": 0}]]
        payload = batch_payload(3)
        with Server(payload) as server:
            result = LayaClient(base_url=server.url + "/proxy/", api_key="batch-token").predict_batch(
                states, QUESTIONS)
        self.assertEqual(result, payload)
        self.assertEqual(len(server.requests), 1)
        method, path, headers, body = server.requests[0]
        self.assertEqual((method, path), ("POST", "/proxy/v1/systemone/batch"))
        self.assertEqual(headers["Authorization"], "Bearer batch-token")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(json.loads(body), {"states": states, "questions": QUESTIONS})
        self.assertEqual([r["usage"]["input_tokens"] for r in result["results"]], [123, 124, 125])

    def test_singleton_and_current_server_boundary(self):
        # The server owns its limit. The client sends each list once, without chunking.
        for count in (1, 64, 65):
            with self.subTest(count=count), Server(batch_payload(count)) as server:
                result = LayaClient(base_url=server.url).predict_batch(["hello"] * count, QUESTIONS)
                self.assertEqual(len(result["results"]), count)
                self.assertEqual(len(server.requests), 1)
                self.assertEqual(len(json.loads(server.requests[0][3])["states"]), count)

    def test_controls_are_forwarded_without_reordering_states(self):
        controls = {"model": "custom", "task": "email", "lang": "en", "lang_guess": "de",
                    "max_len": 128, "head_max_len": 64, "min_confidence": {"default": 0.0},
                    "batch_size": 1, "sort_by_length": True}
        states = ["a longer state", "short"]
        with Server(batch_payload()) as server:
            result = LayaClient(base_url=server.url).predict_batch(states, QUESTIONS, **controls, timeout=2)
        self.assertEqual(result, batch_payload())
        self.assertEqual(json.loads(server.requests[0][3]), {"states": states, "questions": QUESTIONS, **controls})
        with Server(batch_payload()) as server:
            LayaClient(base_url=server.url).predict_batch(states, QUESTIONS, sort_by_length=False, min_confidence=0)
        self.assertIs(json.loads(server.requests[0][3])["sort_by_length"], False)

    def test_invalid_states_never_send(self):
        cyclic = []
        cyclic.append(cyclic)
        cases = [None, [], "hello", {}, ("hello",), [None], [True], [1], [1.5], [object()],
                 ["valid", None], [{"nested": float("nan")}], [{"nested": float("inf")}],
                 [{1: "key"}], [{"nested": (1, 2)}], [cyclic], ["\ud800"]]
        with Server() as server:
            client = LayaClient(base_url=server.url)
            for i, states in enumerate(cases):
                with self.subTest(case=i), self.assertRaises(LayaValidationError):
                    client.predict_batch(states, QUESTIONS)
            self.assertEqual(server.requests, [])

    def test_invalid_questions_and_controls_never_send(self):
        with Server() as server:
            client = LayaClient(base_url=server.url)
            for questions in (None, [], {}, {"x": None}, {"x": {"type": "choice"}}):
                with self.subTest(questions=questions), self.assertRaises(LayaValidationError):
                    client.predict_batch(["hello"], questions)
            cases = [{"model": ""}, {"task": 1}, {"lang": False}, {"lang_guess": ""},
                     {"max_len": True}, {"head_max_len": 0}, {"min_confidence": True},
                     {"min_confidence": {}}, {"min_confidence": {"default": 1.1}},
                     *({"batch_size": v} for v in (True, False, 0, -1, 1.5, "2")),
                     *({"sort_by_length": v} for v in (0, 1, "false", [])),
                     *({"timeout": v} for v in (False, 0, -1, float("inf"), float("nan")))]
            for controls in cases:
                with self.subTest(controls=controls), self.assertRaises(LayaValidationError):
                    client.predict_batch(["hello"], QUESTIONS, **controls)
            with self.assertRaises(TypeError):
                client.predict_batch(["hello"], QUESTIONS, hooks=[])
            self.assertEqual(server.requests, [])

    def test_response_envelope_and_count_are_validated(self):
        cases = [[], {}, {"count": 2, "results": []}]
        for key, value in (("results", None), ("results", {}), ("results", []),
                           ("results", [PREDICTION]), ("results", [PREDICTION] * 3),
                           ("total_usage", None), ("total_usage", []), ("total_usage", {})):
            payload = batch_payload()
            payload[key] = value
            cases.append(payload)
        for payload in cases:
            with self.subTest(payload=payload), Server(payload) as server:
                with self.assertRaises(LayaResponseError):
                    LayaClient(base_url=server.url).predict_batch(["first", "second"], QUESTIONS)
                self.assertEqual(len(server.requests), 1)

    def test_total_usage_is_validated(self):
        for key in ("input_tokens", "output_tokens"):
            for value in (None, True, -1, 1.5, "1"):
                payload = batch_payload()
                payload["total_usage"][key] = value
                with self.subTest(key=key, value=value), Server(payload) as server:
                    with self.assertRaises(LayaResponseError):
                        LayaClient(base_url=server.url).predict_batch(["first", "second"], QUESTIONS)

    def test_invalid_item_is_not_dropped_or_returned_as_partial_success(self):
        bad_answer = copy.deepcopy(PREDICTION)
        bad_answer["answers"]["refund"]["noul"] = True
        bad_usage = copy.deepcopy(PREDICTION)
        bad_usage["usage"]["input_tokens"] = -1
        for item in (None, [], {}, {"error": "inference failed"}, bad_answer, bad_usage):
            payload = batch_payload()
            payload["results"][1] = item
            with self.subTest(item=item), Server(payload) as server:
                with self.assertRaisesRegex(LayaResponseError, r"results\[1\]"):
                    LayaClient(base_url=server.url).predict_batch(["first", "second"], QUESTIONS)
                self.assertEqual(len(server.requests), 1)

    def test_strict_and_extended_results_are_preserved(self):
        payload = batch_payload()
        payload["future"] = {"value": None}
        payload["total_usage"]["new_measurement"] = 7
        extended = payload["results"][1]
        extended["routing"] = {"model": "custom", "future": [1, False]}
        extended["answers"]["team"].update(
            probabilities={"billing": 1.0, "technical": 0.0}, confidence=0.0, answer_confidence=1.0,
            x_tournament={"finalists": ["billing"]}, abstention="passed", abstention_threshold=0.8)
        with Server(payload) as server:
            result = LayaClient(base_url=server.url).predict_batch(["first", "second"], QUESTIONS, min_confidence=0.8)
        self.assertEqual(result, payload)
        self.assertNotIn("abstention", result["results"][0]["answers"]["team"])

    def test_invalid_json_is_a_response_error(self):
        for raw in (b"not json", b"null", b'{"results": NaN}', b"\xff"):
            with self.subTest(raw=raw), Server(raw=raw) as server:
                with self.assertRaises(LayaResponseError):
                    LayaClient(base_url=server.url).predict_batch(["hello"], QUESTIONS)

    def test_http_errors_and_unsupported_endpoints_do_not_retry_or_fall_back(self):
        for status in (400, 401, 404, 405, 413, 422, 500, 503):
            detail = {"message": "batch refused", "status": status}
            with self.subTest(status=status), Server({"detail": detail}, status=status) as server:
                with self.assertRaises(LayaAPIError) as caught:
                    LayaClient(base_url=server.url).predict_batch(["hello"], QUESTIONS)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(caught.exception.detail, detail)
                self.assertEqual([r[1] for r in server.requests], ["/v1/systemone/batch"])

    def test_redirect_does_not_replay_batch_or_forward_credentials(self):
        with Server(batch_payload(1)) as destination:
            with Server({}, status=307, headers={"Location": destination.url}) as source:
                with self.assertRaises(LayaAPIError) as caught:
                    LayaClient(base_url=source.url, api_key="batch-token").predict_batch(["hello"], QUESTIONS)
                self.assertEqual(caught.exception.status_code, 307)
                self.assertEqual(len(source.requests), 1)
            self.assertEqual(destination.requests, [])

    def test_timeouts_do_not_retry_or_fall_back(self):
        for overrides in ({}, {"timeout": 0.05}):
            with self.subTest(overrides=overrides), Server(batch_payload(1), slow=True) as server:
                client = LayaClient(base_url=server.url, timeout=0.05 if not overrides else 2)
                with self.assertRaises(LayaTimeoutError):
                    client.predict_batch(["hello"], QUESTIONS, **overrides)
                self.assertEqual(len(server.requests), 1)

    def test_interrupted_response_is_a_connection_error(self):
        with Server(batch_payload(1), truncated=True) as server:
            with self.assertRaises(LayaConnectionError):
                LayaClient(base_url=server.url).predict_batch(["hello"], QUESTIONS)
            self.assertEqual(len(server.requests), 1)


if __name__ == "__main__":
    unittest.main()
