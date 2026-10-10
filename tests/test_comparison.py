"""Offline comparison contracts; no downloaded weights or external services."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya.comparison import ComparisonAgent, comparison_questions  # noqa: E402
from laya._comparison_data import comparison_table, quantity, validate_state  # noqa: E402


class Recorder:
    def __init__(self):
        self.calls = []
        self.response = {"sentinel": object()}

    def predict(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response

    def predict_batch(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [self.response]


def state():
    return {"task": "numeric_comparison", "assertion": "The quantity named reading exceeds the quantity named limit.",
            "fields": [{"name": "reading", "value": "1", "unit": "kg"},
                       {"name": "limit", "value": "900", "unit": "g"}]}


class ComparisonContracts(unittest.TestCase):
    def test_default_delegates_without_modification(self):
        base, expert = Recorder(), Recorder()
        agent = ComparisonAgent(base, expert)
        questions, request, hook = {"unrelated": {}}, state(), object()
        self.assertIs(agent.predict(request, questions, hooks=hook), base.response)
        self.assertEqual(base.calls, [((request, questions), {"hooks": hook})])
        self.assertEqual(expert.calls, [])
        self.assertIs(agent.predict_batch([request], questions, lang="en")[0], base.response)
        self.assertEqual(base.calls[-1], (([request], questions), {"batch_size": None, "lang": "en"}))

    def test_enabled_dispatch_and_question_isolation(self):
        base, expert = Recorder(), Recorder()
        agent = ComparisonAgent(base, expert)
        self.assertIs(agent.predict(state(), comparison=True), expert.response)
        self.assertEqual(base.calls, [])
        self.assertEqual(expert.calls[-1][1], {"batch_size": 1})
        questions = comparison_questions()
        questions["verdict"]["criteria"].clear()
        self.assertEqual(len(comparison_questions()["verdict"]["criteria"]), 3)
        with self.assertRaises(ValueError):
            agent.predict(state(), questions, comparison=True)
        with self.assertRaises(TypeError):
            agent.predict(state(), comparison=True, hooks=object())

    def test_invalid_flags_and_batches(self):
        agent = ComparisonAgent(Recorder(), Recorder())
        for flag in (1, None, "auto", {}, []):
            with self.subTest(flag=flag), self.assertRaises(TypeError):
                agent.predict(state(), comparison=flag)
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                agent.predict_batch([], comparison=True, batch_size=size)

    def test_state_validation_never_mutates(self):
        request = state()
        saved = copy.deepcopy(request)
        normalized = validate_state(request)
        self.assertEqual(request, saved)
        self.assertIn("semantics", normalized)
        normalized["fields"][0]["value"] = 5
        self.assertEqual(request, saved)

    def test_unsupported_inputs(self):
        cases = [{"task": "general"}, {"operator_probabilities": [1, 0, 0, 0, 0, 0]},
                 {"assertion": " "}, {"assertion": "x" * 16385}, {"fields": []},
                 {"semantics": "Assume missing values are zero"}]
        # Empty fields are legitimate missing context; other changes are errors.
        for changes in cases:
            if changes == {"fields": []}:
                self.assertEqual(validate_state({**state(), **changes})["fields"], [])
            else:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_state({**state(), **changes})
        for fields in ([state()["fields"][0]] * 2, [{"name": "x", "value": True}],
                       [{"name": "x", "unit": []}], [{"name": "x", "gold": "supported"}],
                       [{"name": str(i)} for i in range(8)]):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_state({**state(), "fields": fields})

    def test_exact_units_and_null_candidates(self):
        table = comparison_table(state()["fields"])
        self.assertEqual(table.shape, (8, 8, 6, 3))
        self.assertEqual(table[0, 1, 2].argmax().item(), 0)  # 1 kg > 900 g
        self.assertEqual(table[0, 1, 0].argmax().item(), 1)
        self.assertTrue(bool((table[7, :, :, 2] == 1).all()))
        for value in (None, True, "NaN", "Infinity", "1e1000", "bad"):
            self.assertIsNone(quantity({"value": value, "unit": "g"}))
        self.assertIsNone(quantity({"value": 1, "unit": "unknown"}))
        self.assertEqual(quantity({"value": "0.1", "unit": "kg"}), quantity({"value": 100, "unit": "g"}))
        self.assertNotEqual(quantity({"value": 1, "unit": "kB"}), quantity({"value": 1, "unit": "KiB"}))

    def test_public_wrapper_import_stays_lightweight(self):
        code = "import sys, json; from laya import ComparisonAgent, comparison_questions; print(json.dumps('torch' in sys.modules))"
        result = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertFalse(json.loads(result))


if __name__ == "__main__":
    unittest.main()
