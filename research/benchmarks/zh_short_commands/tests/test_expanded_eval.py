"""Offline integrity checks for the additional diagnostic; no model download."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import expanded_eval as evaluation


class TestExpandedAudit(unittest.TestCase):
    def setUp(self):
        manifest = evaluation.read(BENCH / "expanded_manifest.json")
        self.doc = {"manifest": manifest,
                    "config": {"runner_sha256": evaluation.digest(evaluation.__file__),
                               "calibration": {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}},
                    "suites": {}}
        for suite, cases in evaluation.datasets().items():
            self.doc["suites"][suite] = {}
            for config in evaluation.prompts.CONFIGS:
                pairs = evaluation.prompts.cases_for(config, [c["text"] for c in cases])
                raw = evaluation.upstream.stub_logits(pairs)
                result = evaluation.make_records(config, cases, raw, None)
                result.update(repeat_logits=[[float(v) for v in row] for row in raw],
                              repeat_decision_changes=0, repeat_max_logit_delta=0.0)
                self.doc["suites"][suite][config] = result
        self.manifest = manifest

    def check(self, doc):
        with patch.object(evaluation, "read", side_effect=lambda p: self.manifest if Path(p).name == "expanded_manifest.json" else doc):
            evaluation.audit("unused.json")

    def test_stub_archive_passes(self):
        self.check(self.doc)

    def test_tampered_request_gold_and_metrics_fail(self):
        for key, value in (("text", "替换输入"), ("gold_label", "none"), ("confidence", 0.01)):
            doc = copy.deepcopy(self.doc)
            doc["suites"]["expanded96"]["choice_criteria"]["cases"][0][key] = value
            with self.assertRaises(AssertionError):
                self.check(doc)
        doc = copy.deepcopy(self.doc)
        doc["suites"]["expanded96"]["choice_criteria"]["report"]["accuracy"] = 1.0
        with self.assertRaises(AssertionError):
            self.check(doc)

    def test_dropped_and_duplicate_records_fail(self):
        for duplicate in (False, True):
            doc = copy.deepcopy(self.doc)
            rows = doc["suites"]["expanded96"]["choice_criteria"]["cases"]
            rows.append(copy.deepcopy(rows[0])) if duplicate else rows.pop()
            with self.assertRaises((AssertionError, IndexError)):
                self.check(doc)

    def test_nonfinite_logits_fail(self):
        doc = copy.deepcopy(self.doc)
        doc["suites"]["expanded96"]["choice_criteria"]["cases"][0]["logits"][0] = float("nan")
        with self.assertRaises(AssertionError):
            self.check(doc)

    def test_changed_repeat_is_detected(self):
        doc = copy.deepcopy(self.doc)
        doc["suites"]["expanded96"]["choice_criteria"]["repeat_logits"][0][0] += 100
        with self.assertRaises(AssertionError):
            self.check(doc)


if __name__ == "__main__":
    unittest.main()
