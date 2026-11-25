import copy
import io
import json
import socket
import unittest
import urllib.error
from unittest.mock import patch

from laya_client import (
    LayaAPIError, LayaClient, LayaConnectionError, LayaResponseError,
    LayaTimeoutError, LayaValidationError,
)
from _support import PREDICTION, QUESTIONS, Server


class ClientTests(unittest.TestCase):
    def test_predict_preserves_payload_and_server_default(self):
        state = {"message": "退款 🐍", "items": [None, False, 0, ""], "nested": {"a": 1}}
        questions = copy.deepcopy(QUESTIONS)
        questions["team"]["option_order"] = ["technical", "billing"]
        questions["team"]["future_field"] = {"enabled": True}
        original = copy.deepcopy((state, questions))
        with Server() as server:
            result = LayaClient(base_url=server.url + "/proxy/laya/", api_key="secret").predict(state, questions)
        method, path, headers, raw = server.requests[0]
        self.assertEqual((method, path), ("POST", "/proxy/laya/v1/systemone"))
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(json.loads(raw), {"state": state, "questions": questions})
        self.assertEqual(result, PREDICTION)
        self.assertEqual((state, questions), original)

    def test_controls_and_zero_threshold_are_forwarded(self):
        controls = dict(model="registered-checkpoint", task="helpdesk", lang="de",
                        lang_guess="de-AT", max_len=128, head_max_len=64, min_confidence=0.0)
        with Server() as server:
            LayaClient(base_url=server.url).predict("hello", QUESTIONS, timeout=2, **controls)
        body = json.loads(server.requests[0][3])
        self.assertEqual(body, {"state": "hello", "questions": QUESTIONS, **controls})
        self.assertNotIn("Authorization", server.requests[0][2])

    def test_threshold_names_are_server_owned(self):
        thresholds = {"default": 0.9, "future:bucket": 0.0}
        with Server() as server:
            LayaClient(base_url=server.url).predict("hello", QUESTIONS, min_confidence=thresholds)
        self.assertEqual(json.loads(server.requests[0][3])["min_confidence"], thresholds)

    def test_shared_references_are_not_cycles(self):
        shared = {"flag": False}
        with Server() as server:
            LayaClient(base_url=server.url).predict([shared, shared], QUESTIONS)
        self.assertEqual(json.loads(server.requests[0][3])["state"], [shared, shared])

    def test_constructor_does_not_connect(self):
        with patch("urllib.request.build_opener", side_effect=AssertionError("unexpected connection")):
            client = LayaClient(api_key="private-token")
        self.assertNotIn("private-token", repr(client))

    def test_invalid_urls(self):
        for url in (None, 42, "", "localhost:8000", "file:///tmp/x", "ftp://example.com",
                    "http://", "http://user:pass@localhost", "http://@localhost", "http://localhost?q=1",
                    "http://localhost#x", " http://localhost", "http://localhost/a b",
                    "http://localhost:99999", "http://localhost:abc", "http://[broken", "http://本地",
                    "http://localhost/?", "http://localhost/#", "http://local\x00host"):
            with self.subTest(url=url), self.assertRaises(LayaValidationError):
                LayaClient(base_url=url)

    def test_invalid_tokens_do_not_leak(self):
        for token in ("", 123, " secret", "secret\nheader", "secret\x7f", "密钥"):
            with self.subTest(token=token), self.assertRaises(LayaValidationError) as failure:
                LayaClient(api_key=token)
            self.assertNotIn("secret", str(failure.exception))

    def test_invalid_timeout_before_network(self):
        for timeout in (0, -1, True, "2", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(timeout=repr(timeout)[:30]), self.assertRaises(LayaValidationError):
                LayaClient(timeout=timeout)
        with Server() as server:
            client = LayaClient(base_url=server.url)
            for timeout in (0, -1, True, "2", float("nan"), float("inf")):
                with self.subTest(timeout=timeout), self.assertRaises(LayaValidationError):
                    client.predict("hello", QUESTIONS, timeout=timeout)
            self.assertEqual(server.requests, [])

    def test_invalid_state_never_sends(self):
        cyclic = []
        cyclic.append(cyclic)
        deep = []
        for _ in range(1100):
            deep = [deep]
        states = [None, True, 2, 1.5, (), {"x": object()}, {"x": (1, 2)}, {1: "rewritten"},
                  {"x": float("nan")}, {"x": float("inf")}, cyclic, deep, "\ud800"]
        with Server() as server:
            client = LayaClient(base_url=server.url)
            for i, state in enumerate(states):
                with self.subTest(case=i), self.assertRaises(LayaValidationError):
                    client.predict(state, QUESTIONS)
            self.assertEqual(server.requests, [])

    def test_invalid_questions_never_send(self):
        cases = [None, [], {}, {"x": None}, {"x": {"type": "unknown", "instructions": ""}},
                 {"x": {"type": "choice"}}, {42: QUESTIONS["refund"]}]
        with Server() as server:
            client = LayaClient(base_url=server.url)
            for questions in cases:
                with self.subTest(questions=questions), self.assertRaises(LayaValidationError):
                    client.predict("hello", questions)
            self.assertEqual(server.requests, [])

    def test_invalid_controls_never_send(self):
        cases = [{"model": ""}, {"task": 1}, {"lang": False}, {"lang_guess": lambda x: x},
                 {"max_len": True}, {"head_max_len": 0}, {"max_len": 1.5},
                 {"min_confidence": True}, {"min_confidence": -0.1}, {"min_confidence": 1.01},
                 {"min_confidence": {}}, {"min_confidence": {"default": False}},
                 {"min_confidence": {"default": float("nan")}}, {"min_confidence": {1: 0.5}}]
        with Server() as server:
            client = LayaClient(base_url=server.url)
            for controls in cases:
                with self.subTest(controls=controls), self.assertRaises(LayaValidationError):
                    client.predict("hello", QUESTIONS, **controls)
            with self.assertRaises(TypeError):
                client.predict("hello", QUESTIONS, hooks=object())
            self.assertEqual(server.requests, [])

    def test_health_minimal_and_full(self):
        for payload in ({"status": "ok"}, {"status": "ok", "loaded": ["new-model"], "device": "cpu",
                                         "new_metadata": {"capability": True}}):
            with self.subTest(payload=payload), Server(payload) as server:
                self.assertEqual(LayaClient(base_url=server.url).health(), payload)
            self.assertEqual(server.requests[0][0], "GET")
            self.assertEqual(server.requests[0][3], b"")

    def test_health_does_not_invent_authentication(self):
        with Server({"status": "ok"}) as server:
            result = LayaClient(base_url=server.url, api_key="wrong").health()
        self.assertEqual(result, {"status": "ok"})
        self.assertNotIn("authenticated", result)

    def test_strict_response_does_not_invent_abstention(self):
        with Server() as server:
            result = LayaClient(base_url=server.url).predict("hello", QUESTIONS, min_confidence=0.99)
        self.assertNotIn("routing", result)
        for answer in result["answers"].values():
            self.assertNotIn("answer_confidence", answer)
            self.assertNotIn("abstention", answer)
            self.assertNotIn("low_confidence", answer)
        self.assertEqual(result["answers"]["refund"], {"type": "noul", "noul": 0.7})

    def test_extensions_and_tournament_values_are_preserved(self):
        payload = copy.deepcopy(PREDICTION)
        payload["routing"] = {"model": "future-model", "future": [1, False]}
        payload["future"] = {"value": None}
        payload["usage"]["new_measurement"] = 3
        team = payload["answers"]["team"]
        team.update(probabilities={"billing": 1.0, "technical": 0.0}, confidence=0.0,
                    answer_confidence=1.0, x_tournament={"finalists": ["billing"], "rounds": []},
                    abstention="passed", abstention_threshold=0.8, new_answer_field=[False])
        with Server(payload) as server:
            result = LayaClient(base_url=server.url).predict("hello", QUESTIONS)
        self.assertEqual(result, payload)

    def test_invalid_responses(self):
        cases = []
        for key, value in (("model", None), ("answers", []), ("usage", None), ("routing", [])):
            p = copy.deepcopy(PREDICTION)
            p[key] = value
            cases.append(p)
        for key, value in (("confidence", True), ("confidence", -1), ("confidence", 1.01),
                           ("probabilities", {}), ("probabilities", {"billing": 1}),
                           ("choice", "missing"), ("type", "score"), ("low_confidence", "true"),
                           ("abstention", "invented"), ("abstention", "passed"),
                           ("abstention_threshold", 0.8), ("action", False)):
            p = copy.deepcopy(PREDICTION)
            p["answers"]["team"][key] = value
            cases.append(p)
        for key, value in (("score", True), ("score", 3), ("legend", {})):
            p = copy.deepcopy(PREDICTION)
            p["answers"]["urgency"][key] = value
            cases.append(p)
        p = copy.deepcopy(PREDICTION)
        p["usage"]["input_tokens"] = True
        cases.append(p)
        p = copy.deepcopy(PREDICTION)
        del p["answers"]["refund"]
        cases.append(p)
        for i, payload in enumerate(cases):
            with self.subTest(case=i), Server(payload) as server, self.assertRaises(LayaResponseError):
                LayaClient(base_url=server.url).predict("hello", QUESTIONS)

    def test_invalid_json_successes(self):
        for raw in (b"<html>wrong upstream</html>", b"null", b"[]", b"{}",
                    b'{"x": NaN}', b"\xff", b'{"x": Infinity}'):
            with self.subTest(raw=raw), Server(raw=raw) as server, self.assertRaises(LayaResponseError):
                LayaClient(base_url=server.url).predict("hello", QUESTIONS)

    def test_invalid_health(self):
        for raw in (b"[]", b"null", b'{"status":"starting"}', b'{"loaded":[]}'):
            with self.subTest(raw=raw), Server(raw=raw) as server, self.assertRaises(LayaResponseError):
                LayaClient(base_url=server.url).health()

    def test_http_status_and_structured_details(self):
        for status in (400, 401, 413, 422, 500, 503):
            detail = [{"loc": ["body", "model"], "msg": "invalid"}]
            with self.subTest(status=status), Server({"detail": detail}, status=status) as server:
                with self.assertRaises(LayaAPIError) as failure:
                    LayaClient(base_url=server.url).predict("hello", QUESTIONS)
            self.assertEqual(failure.exception.status_code, status)
            self.assertEqual(failure.exception.detail, detail)
            self.assertEqual(len(server.requests), 1)

    def test_non_json_error_keeps_http_status(self):
        with Server(raw=b"<html>proxy unavailable</html>", status=503) as server:
            with self.assertRaises(LayaAPIError) as failure:
                LayaClient(base_url=server.url).health()
        self.assertEqual(failure.exception.status_code, 503)
        self.assertIn("proxy unavailable", failure.exception.detail)

    def test_error_detail_shapes(self):
        for detail in ("bad request", {"reason": "limit"}, None):
            with self.subTest(detail=detail), Server({"detail": detail}, status=422) as server:
                with self.assertRaises(LayaAPIError) as failure:
                    LayaClient(base_url=server.url).predict("hello", QUESTIONS)
            self.assertEqual(failure.exception.detail, detail)

    def test_deep_error_json_keeps_http_status(self):
        raw = b"[" * 2000 + b"0" + b"]" * 2000
        with Server(raw=raw, status=500) as server:
            with self.assertRaises(LayaAPIError) as failure:
                LayaClient(base_url=server.url).health()
        self.assertEqual(failure.exception.status_code, 500)

    def test_redirect_never_forwards_token_or_replays(self):
        with Server() as other:
            for status in (301, 302, 303, 307, 308):
                with self.subTest(status=status), Server(status=status, headers={"Location": other.url}) as server:
                    with self.assertRaises(LayaAPIError) as failure:
                        LayaClient(base_url=server.url, api_key="secret").predict("hello", QUESTIONS)
                self.assertEqual(failure.exception.status_code, status)
                self.assertEqual(len(server.requests), 1)
            self.assertEqual(other.requests, [])

    def test_timeout_does_not_retry(self):
        with Server(slow=True) as server:
            with self.assertRaises(LayaTimeoutError):
                LayaClient(base_url=server.url, timeout=2).predict("hello", QUESTIONS, timeout=0.1)
            self.assertEqual(len(server.requests), 1)

    def test_connection_refused(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            url = "http://127.0.0.1:%d" % listener.getsockname()[1]
            with self.assertRaises(LayaConnectionError):
                LayaClient(base_url=url, timeout=0.5).health()

    def test_interrupted_response(self):
        with Server(truncated=True) as server:
            with self.assertRaises(LayaConnectionError):
                LayaClient(base_url=server.url).predict("hello", QUESTIONS)

    def test_response_is_closed_on_decode_error(self):
        stream = io.BytesIO(b"invalid")
        with patch("urllib.request.OpenerDirector.open", return_value=stream):
            with self.assertRaises(LayaResponseError):
                LayaClient().health()
        self.assertTrue(stream.closed)

    def test_http_error_stream_is_closed(self):
        stream = io.BytesIO(b'{"detail":"no"}')
        error = urllib.error.HTTPError("http://localhost", 401, "Unauthorized", {}, stream)
        with patch("urllib.request.OpenerDirector.open", side_effect=error):
            with self.assertRaises(LayaAPIError):
                LayaClient().health()
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
