"""HTTP contract tests without downloading checkpoints: python tests/test_server.py."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from laya.router import Router  # noqa: E402
from laya.server import create_app  # noqa: E402

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "support"]},
    "urgency": {"type": "score", "instructions": {"rubric": "urgent?"}, "criteria": ["low", {"level": "high"}]},
    "refund": {"type": "noul", "instructions": "Refund requested?", "criteria": {"true": {"meaning": "yes"}}},
}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.router = Router()
        self.agent = Mock()
        self.agent.system_one.return_value = {"model": "laya-rl-agent", "answers": {},
                                             "usage": {"input_tokens": 12, "output_tokens": 0}}
        self.router.attach("english", self.agent)
        self.client = TestClient(create_app(self.router), raise_server_exceptions=False)

    def test_health_does_not_load_checkpoints(self):
        router = Router()
        client = TestClient(create_app(router))
        self.assertEqual(client.get("/health").json()["loaded_models"], [])
        self.assertEqual(router.loaded, [])

    def test_route_detects_language_without_loading(self):
        response = self.client.post("/v1/route", json={"state": {"body": "मुझे पैसे वापस चाहिए"}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "multilingual")
        self.assertEqual(self.router.loaded, ["english"])
        self.agent.system_one.assert_not_called()

    def test_predict_preserves_state_question_shapes_and_result(self):
        state = [{"role": "user", "content": "Please refund the duplicate charge"}]
        response = self.client.post("/v1/predict", json={"state": state, "questions": QUESTIONS, "model": "en"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["usage"], self.agent.system_one.return_value["usage"])
        self.assertEqual(response.json()["routing"]["model"], "english")
        self.agent.system_one.assert_called_once_with(state, QUESTIONS)

    def test_workflow_repo_is_always_a_string(self):
        router = Router(auto_task_detection=True)
        questions = {key: {"type": "noul", "instructions": "test"}
                     for key in ["action", "category", "churn_risk", "needs_human", "urgency"]}
        response = TestClient(create_app(router)).post("/v1/route", json={"state": "hello", "questions": questions})
        self.assertEqual(response.json()["repo"], "convaiinnovations/laya/typed-decisions")
        self.assertEqual(response.json()["workflow"], "customer_service")
        self.assertEqual(router.loaded, [])

    def test_invalid_questions_rejected_before_inference(self):
        invalid = [
            {}, {"q": {"type": "unknown", "instructions": "test"}},
            {"q": {"type": "noul"}}, {"q": {"type": "noul", "instructions": "test", "criteria": []}},
            {"q": {"type": "noul", "instructions": "test", "criteria": {"yes": "yes"}}},
            {"q": {"type": "choice", "instructions": "test", "criteria": []}},
            {"q": {"type": "choice", "instructions": "test", "criteria": ["a", "a"]}},
            {"q": {"type": "score", "instructions": "test", "criteria": {"0": "low"}}},
            {"q": {"type": "score", "instructions": "test", "criteria": []}},
        ]
        for questions in invalid:
            with self.subTest(questions=questions):
                response = self.client.post("/v1/predict", json={"state": "secret input", "questions": questions})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["error"]["code"], "validation_error")
                self.assertNotIn("secret input", response.text)
        self.agent.system_one.assert_not_called()

    def test_missing_state_and_unknown_fields_are_rejected(self):
        for body in [{"questions": QUESTIONS}, {"state": "hello", "questions": QUESTIONS, "typo": True}]:
            self.assertEqual(self.client.post("/v1/predict", json=body).status_code, 422)

    def test_null_and_structured_criteria_are_preserved(self):
        questions = copy.deepcopy(QUESTIONS)
        questions["department"]["criteria"] = {"a": None, "b": 0, "c": False}
        response = self.client.post("/v1/predict", json={"state": None, "questions": questions, "model": "en"})
        self.assertEqual(response.status_code, 200)
        self.agent.system_one.assert_called_once_with(None, questions)

    def test_unknown_model_is_a_client_error(self):
        response = self.client.post("/v1/predict", json={"state": "hello", "questions": QUESTIONS, "model": "missing"})
        self.assertEqual(response.status_code, 422)
        self.agent.system_one.assert_not_called()

    def test_auth_on_all_endpoints(self):
        client = TestClient(create_app(self.router, api_key="test-key"))
        for path in ["/health", "/v1/route", "/v1/predict"]:
            for header in [{}, {"Authorization": "Bearer wrong"}]:
                response = client.get(path, headers=header) if path == "/health" else client.post(
                    path, json={"state": "hello", "questions": QUESTIONS}, headers=header)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json()["error"]["code"], "unauthorized")
        self.assertEqual(client.get("/health", headers={"Authorization": "Bearer test-key"}).status_code, 200)

    def test_model_failure_does_not_expose_internal_details(self):
        self.agent.system_one.side_effect = RuntimeError("private/path/checkpoint")
        with self.assertLogs("laya.server", level="ERROR"):
            response = self.client.post("/v1/predict", json={"state": "hello", "questions": QUESTIONS})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "inference_error")
        self.assertNotIn("private/path", response.text)

    def test_option_budget_failure_is_a_client_error(self):
        self.agent.system_one.side_effect = ValueError("question 'q' options exceed head_max_len=32")
        response = self.client.post("/v1/predict", json={"state": "hello", "questions": QUESTIONS})
        self.assertEqual(response.status_code, 422)

    def test_cors_is_explicit_and_supports_auth_preflight(self):
        headers = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
                   "Access-Control-Request-Headers": "authorization,content-type"}
        client = TestClient(create_app(self.router, api_key="test", cors_origins=["http://localhost:3000"]))
        response = client.options("/v1/predict", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")
        headers["Origin"] = "http://untrusted.example"
        self.assertNotIn("access-control-allow-origin", client.options("/v1/predict", headers=headers).headers)

    def test_openapi_includes_discriminated_questions(self):
        schema = self.client.get("/openapi.json").json()
        questions = schema["components"]["schemas"]["PredictRequest"]["properties"]["questions"]
        self.assertEqual(questions["additionalProperties"]["discriminator"]["propertyName"], "type")


if __name__ == "__main__":
    unittest.main()
