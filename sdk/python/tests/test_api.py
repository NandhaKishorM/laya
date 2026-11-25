import importlib.metadata
import inspect
from pathlib import Path
import sys
import unittest

import laya_client
from laya_client import LayaClient, LayaConnectionError, LayaError, LayaTimeoutError


class APITests(unittest.TestCase):
    def test_public_exports(self):
        self.assertEqual(set(laya_client.__all__), {
            "LayaClient", "LayaError", "LayaValidationError", "LayaAPIError",
            "LayaConnectionError", "LayaTimeoutError", "LayaResponseError",
        })
        for name in laya_client.__all__:
            if name.endswith("Error"):
                self.assertTrue(issubclass(getattr(laya_client, name), LayaError))
        self.assertTrue(issubclass(LayaTimeoutError, LayaConnectionError))

    def test_predict_parameters(self):
        parameters = inspect.signature(LayaClient.predict).parameters
        self.assertEqual(list(parameters), ["self", "state", "questions", "model", "task", "lang",
                                           "lang_guess", "max_len", "head_max_len", "min_confidence", "timeout"])
        for name in list(parameters)[3:]:
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameters[name].default)
        self.assertEqual(list(inspect.signature(LayaClient.health).parameters), ["self", "timeout"])

    def test_predict_batch_parameters(self):
        from laya_client.types import BatchPrediction
        signature = inspect.signature(LayaClient.predict_batch)
        parameters = signature.parameters
        self.assertEqual(list(parameters), ["self", "states", "questions", "model", "task", "lang",
                                           "lang_guess", "max_len", "head_max_len", "min_confidence",
                                           "batch_size", "sort_by_length", "timeout"])
        for name in list(parameters)[3:]:
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameters[name].default)
        self.assertIs(signature.return_annotation, BatchPrediction)
        self.assertEqual(BatchPrediction.__required_keys__, {"results", "total_usage"})

    def test_constructor_defaults(self):
        parameters = inspect.signature(LayaClient).parameters
        self.assertEqual({k: p.default for k, p in parameters.items()},
                         {"base_url": "http://127.0.0.1:8000", "api_key": None, "timeout": 120.0})
        self.assertTrue(all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in parameters.values()))

    def test_types_match_strict_response_requirements(self):
        from laya_client.types import ChoiceAnswer, NoulAnswer, Prediction, ScoreAnswer
        self.assertIn("confidence", ChoiceAnswer.__required_keys__)
        self.assertIn("confidence", ScoreAnswer.__required_keys__)
        self.assertIn("confidence", NoulAnswer.__optional_keys__)
        self.assertIn("abstention", ChoiceAnswer.__optional_keys__)
        self.assertIn("routing", Prediction.__optional_keys__)

    def test_installed_package_has_no_runtime_requirements(self):
        distribution = importlib.metadata.distribution("laya-client")
        self.assertFalse(distribution.requires)
        installed = Path(distribution.locate_file("laya_client/__init__.py")).resolve()
        self.assertEqual(Path(laya_client.__file__).resolve(), installed)
        self.assertNotIn("src", installed.parts)
        self.assertTrue(Path(distribution.locate_file("laya_client/py.typed")).is_file())
        self.assertTrue(any("LICENSE" in str(p) for p in distribution.files))

    def test_no_inference_or_http_dependencies_in_client_environment(self):
        names = {d.metadata["Name"].lower().replace("_", "-") for d in importlib.metadata.distributions()}
        forbidden = {"laya", "torch", "transformers", "numpy", "safetensors",
                     "huggingface-hub", "requests", "httpx"}
        self.assertFalse(names & forbidden, names & forbidden)
        self.assertFalse(set(sys.modules) & {"laya", "torch", "transformers", "numpy"})


if __name__ == "__main__":
    unittest.main()
