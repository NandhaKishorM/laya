"""Saved summaries tolerate float roundoff without accepting changed evidence."""
import copy
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audit  # noqa: E402

METRICS = ("accuracy", "macro_f1", "mean_answer_confidence", "ece",
           "latency_ms_p50", "latency_ms_p90", "repeat_agreement")


class SavedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "english"
        shutil.copytree(ROOT / "results/v1/english", self.out)
        self.saved = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))

    def result(self, saved):
        (self.out / "summary.json").write_text(json.dumps(saved), encoding="utf-8")
        with redirect_stdout(io.StringIO()) as output:
            code = audit.main(["--run-dir", str(self.out)])
        if code:
            self.assertIn("saved summary differs", output.getvalue())
        return code

    def test_one_ulp_in_each_metric_passes_the_cli_audit(self):
        for metric in METRICS:
            with self.subTest(metric=metric):
                saved = copy.deepcopy(self.saved)
                value = saved["modes"]["choice_fa"][metric]
                saved["modes"]["choice_fa"][metric] = math.nextafter(value, math.inf)
                self.assertEqual(self.result(saved), 0)

    def test_changed_metric_and_nonfinite_values_fail_the_cli_audit(self):
        for metric in METRICS:
            value = self.saved["modes"]["choice_fa"][metric]
            for changed in (value + 1e-6 * max(1, abs(value)), float("nan"), float("inf"),
                            -float("inf"), True, str(value), None):
                with self.subTest(metric=metric, changed=changed):
                    saved = copy.deepcopy(self.saved)
                    saved["modes"]["choice_fa"][metric] = changed
                    self.assertEqual(self.result(saved), 1)

    def test_counts_flags_and_metadata_stay_exact(self):
        for path, value in ((["modes", "choice_fa", "correct"], 22),
                            (["modes", "choice_fa", "correct"], 21.0),
                            (["modes", "choice_fa", "repeat"], False),
                            (["modes", "choice_fa", "by_label", "cancel", "n"], 16.0),
                            (["modes", "choice_fa", "confusion", "cancel", "cancel"], True),
                            (["normalize"], 0),
                            (["backend", "checkpoint"], "changed")):
            with self.subTest(path=path, value=value):
                saved = copy.deepcopy(self.saved)
                target = saved
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertEqual(self.result(saved), 1)

    def test_missing_and_extra_fields_are_not_ignored(self):
        for path in ([], ["modes"], ["modes", "choice_fa"],
                     ["modes", "choice_fa", "by_label", "cancel"]):
            for extra in (False, True):
                with self.subTest(path=path, extra=extra):
                    saved = copy.deepcopy(self.saved)
                    target = saved
                    for key in path:
                        target = target[key]
                    if extra:
                        target["unexpected"] = 0
                    else:
                        del target[next(iter(target))]
                    self.assertEqual(self.result(saved), 1)

    def test_json_object_key_order_is_irrelevant(self):
        def reverse(value):
            if isinstance(value, dict):
                return {key: reverse(value[key]) for key in reversed(value)}
            return value
        self.assertEqual(self.result(reverse(self.saved)), 0)


class SummaryComparisonTests(unittest.TestCase):
    def metric(self, saved, expected):
        return audit._summary_matches({"modes": {"choice_fa": {"ece": saved}}},
                                      {"modes": {"choice_fa": {"ece": expected}}})

    def test_relative_and_absolute_tolerance(self):
        # Absolute tolerance handles zero; relative tolerance scales latency-sized values.
        self.assertTrue(self.metric(5e-13, 0.0))
        self.assertFalse(self.metric(2e-12, 0.0))
        self.assertTrue(self.metric(1000.0 + 5e-10, 1000.0))
        self.assertFalse(self.metric(1000.0 + 2e-9, 1000.0))

    def test_nullable_metrics_and_numeric_json_representations(self):
        self.assertTrue(self.metric(None, None))
        self.assertFalse(self.metric(None, 0.0))
        self.assertFalse(self.metric(0.0, None))
        self.assertTrue(self.metric(1, 1.0))
        self.assertTrue(self.metric(1.0, 1))
        for value in (True, "1", float("nan"), float("inf"), -float("inf"), 10**400):
            self.assertFalse(self.metric(value, 1.0))
            self.assertFalse(self.metric(1.0, value))
            self.assertFalse(self.metric(value, value))

    def test_numeric_metadata_does_not_inherit_metric_tolerance(self):
        for path in (("backend", "ece"), ("modes", "choice_fa", "unrecognized_metric")):
            def nested(value):
                for key in reversed(path):
                    value = {key: value}
                return value
            self.assertFalse(audit._summary_matches(nested(0.5 + 1e-15), nested(0.5)))

    def test_nested_list_order_length_and_scalar_types_are_exact(self):
        expected = {"backend": {"items": ["revision", 1, True, None]}}
        self.assertTrue(audit._summary_matches(copy.deepcopy(expected), expected))
        for items in (["revision", 1.0, True, None], ["revision", 1, 1, None],
                      ["revision", True, 1, None], ["revision", 1, True],
                      ["revision", 1, True, None, None], None):
            self.assertFalse(audit._summary_matches({"backend": {"items": items}}, expected))


if __name__ == "__main__":
    unittest.main()
