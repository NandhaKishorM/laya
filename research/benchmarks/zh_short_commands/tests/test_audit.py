"""Offline tests for audit.py: the archive must stay re-derivable and tamper-evident.

Run from the repository root:

    python -m unittest discover -s research/benchmarks/zh_short_commands/tests -v

Half of these deliberately corrupt a copy of the committed archive and require the
audit to refuse it. An audit nobody has seen fail is not evidence of anything.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BENCH)))
for path in (BENCH, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import audit  # noqa: E402
import prompts  # noqa: E402
import run as runner  # noqa: E402

ARCHIVE = os.path.join(BENCH, "results", "v1", "laya-multilingual")


class TestFrozenInputs(unittest.TestCase):
    def setUp(self):
        self.cases, self.manifest = audit.load_frozen()

    def test_manifest_hashes_pin_the_current_files(self):
        self.assertEqual(len(self.cases), 18)
        self.assertEqual(audit.sha256_file(os.path.join(BENCH, "data", "cases.jsonl")),
                         self.manifest["cases_sha256"])
        self.assertEqual(audit.sha256_file(os.path.join(BENCH, "prompts.py")),
                         self.manifest["prompts_sha256"])

    def test_manifest_counts_match_the_cases(self):
        labels = {}
        families = {}
        for case in self.cases:
            labels[case["gold"]] = labels.get(case["gold"], 0) + 1
            families[case["family"]] = families.get(case["family"], 0) + 1
        self.assertEqual(labels, self.manifest["labels"])
        self.assertEqual(families, self.manifest["families"])

    def test_manifest_freezes_the_ladder_this_file_builds(self):
        self.assertEqual(self.manifest["ladder"]["choice"], list(prompts.CHOICE_CONFIGS))
        self.assertEqual(self.manifest["ladder"]["noul"], list(prompts.NOUL_CONFIGS))


class TestCommittedArchive(unittest.TestCase):
    def test_archive_audits_clean(self):
        result = audit.audit_run(ARCHIVE)
        self.assertEqual(result["cases"], 18)
        self.assertEqual(len(result["configs"]), 7)
        self.assertEqual(result["records"], 342)

    def test_archive_pins_the_weights(self):
        doc = audit.load_json(os.path.join(ARCHIVE, "report.json"))
        self.assertIn("sha256", audit.audit_run(ARCHIVE)["provenance"])
        self.assertEqual(doc["config"]["checkpoint"]["weight_bytes"], 643835514)
        self.assertFalse(doc["config"]["checkpoint"]["stub"])

    def test_stub_archive_has_no_weight_to_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main(["--stub", "--out", tmp])
            # The stub archive has to pass the whole audit, provenance included.
            self.assertIn("stub run", audit.audit_run(tmp)["provenance"])


class TestTamperingIsRefused(unittest.TestCase):
    """Each case mutates one thing in a real archive; the audit must raise."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "report.json")
        with open(os.path.join(ARCHIVE, "report.json"), encoding="utf-8") as f:
            self.doc = json.load(f)

    def rewrite(self, mutate):
        doc = json.loads(json.dumps(self.doc))
        mutate(doc)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        return self.tmp.name

    def assert_refused(self, mutate, needle):
        folder = self.rewrite(mutate)
        with self.assertRaises(ValueError) as caught:
            audit.audit_run(folder)
        self.assertIn(needle, str(caught.exception))

    def first(self, doc, config):
        return next(r for r in doc["cases"] if r["config"] == config)

    def test_flipped_correctness(self):
        def mutate(doc):
            record = self.first(doc, "choice_criteria")
            record["correct"] = 1 - record["correct"]
        self.assert_refused(mutate, "contradicts the prediction")

    def test_swapped_state(self):
        self.assert_refused(
            lambda doc: self.first(doc, "choice_json_state").update(state="停下"),
            "state is not what this rung sends")

    def test_swapped_instructions(self):
        def mutate(doc):
            self.first(doc, "noul_scenario")["instructions"] = "这条指令是否要求加快速度？"
        self.assert_refused(mutate, "instructions are not what this rung sends")

    def test_rewritten_gold_label(self):
        self.assert_refused(
            lambda doc: self.first(doc, "choice_criteria").update(gold_label="none"),
            "frozen text or gold label was altered")

    def test_dropped_record(self):
        def mutate(doc):
            doc["cases"] = [r for r in doc["cases"] if r["case_id"] != "zh-cmd-007"]
        self.assert_refused(mutate, "incomplete archive")

    def test_duplicated_record(self):
        def mutate(doc):
            doc["cases"].append(dict(self.first(doc, "noul_plain")))
            doc["cases"][-1]["case_id"] = "zh-cmd-001"
        self.assert_refused(mutate, "duplicate record")

    def test_probability_out_of_range(self):
        def mutate(doc):
            self.first(doc, "noul_criteria")["p_true"] = 1.5
        self.assert_refused(mutate, "p_true is not a probability")

    def test_prediction_that_does_not_follow_from_the_probability(self):
        def mutate(doc):
            record = self.first(doc, "noul_plain")
            record["pred_bool"] = not record["pred_bool"]
        self.assert_refused(mutate, "pred_bool does not follow")

    def test_noul_confidence_that_is_not_the_maximum(self):
        def mutate(doc):
            self.first(doc, "noul_json_state")["confidence"] = 0.5
        self.assert_refused(mutate, "confidence is not the two-way max")

    def test_protocol_from_a_different_prompt_revision(self):
        self.assert_refused(
            lambda doc: doc["config"].update(prompts_sha256="0" * 64),
            "different frozen protocol")

    def test_relabelled_accuracy(self):
        def mutate(doc):
            doc["report"]["choice_scenario"]["accuracy"] = 0.95
        self.assert_refused(mutate, "choice_scenario/accuracy")

    def test_relabelled_macro_f1(self):
        def mutate(doc):
            doc["report"]["choice_criteria"]["macro_f1"] = 0.5
        self.assert_refused(mutate, "choice_criteria/macro_f1")

    def test_relabelled_ece(self):
        def mutate(doc):
            doc["report"]["choice_json_state"]["ece"] = 0.01
        self.assert_refused(mutate, "choice_json_state/ece")

    def test_relabelled_dimension(self):
        def mutate(doc):
            doc["report"]["noul_scenario"]["accuracy_by_dimension"]["wants_stop"] = 0.9
        self.assert_refused(mutate, "noul_scenario/wants_stop")

    def test_summary_that_disagrees_with_its_report(self):
        def mutate(doc):
            doc["summary"]["noul_criteria"]["accuracy"] = 0.9
        self.assert_refused(mutate, "summary/noul_criteria disagrees")

    def test_unexplained_report_field(self):
        self.assert_refused(
            lambda doc: doc["report"]["choice_criteria"].update(score=1.0),
            "unexplained field")

    def test_a_ladder_other_than_the_frozen_one(self):
        def mutate(doc):
            doc["config"]["configs"] = ["choice_criteria"]
        self.assert_refused(mutate, "different ladder")

    def test_missing_weights_for_a_real_run(self):
        self.assert_refused(
            lambda doc: doc["config"]["checkpoint"].pop("weight_sha256"),
            "no weight_sha256")


class TestMetricsMatchTheHarness(unittest.TestCase):
    """audit.py re-implements the metrics without numpy; pin them to laya_eval's."""

    def setUp(self):
        try:
            import numpy as np
        except ImportError:                        # pragma: no cover - numpy ships with laya
            self.skipTest("numpy is not installed")
        from research.eval import laya_eval
        self.np = np
        self.harness = laya_eval

    def sample(self, n, seed):
        rng = self.np.random.RandomState(seed)
        logits = rng.randn(n, 6) * 3.0
        probs = self.np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        gold = rng.randint(0, 6, n)
        pred = probs.argmax(axis=1)
        return [float(p) for p in probs.max(axis=1)], [int(p == g) for p, g in zip(pred, gold)], \
            [int(g) for g in gold], [int(p) for p in pred]

    def test_macro_f1_matches(self):
        for seed in range(5):
            confidences, corrects, gold, pred = self.sample(60, seed)
            self.assertAlmostEqual(audit.macro_f1(gold, pred),
                                   self.harness.macro_f1(gold, pred), places=12)

    def test_ece_matches(self):
        for seed in range(5):
            confidences, corrects, _, _ = self.sample(60, seed)
            self.assertAlmostEqual(audit.ece(confidences, corrects),
                                   self.harness.ece(confidences, corrects), places=12)

    def test_a_confidence_of_exactly_one_lands_in_the_last_bin(self):
        self.assertAlmostEqual(audit.ece([1.0, 1.0], [1, 1]), 0.0, places=12)
        self.assertAlmostEqual(audit.ece([1.0, 1.0], [0, 1]), 0.5, places=12)

    def test_a_confidence_of_exactly_zero_is_counted(self):
        self.assertAlmostEqual(audit.ece([0.0], [0]), 0.0, places=12)
        self.assertAlmostEqual(audit.ece([0.0], [1]), 1.0, places=12)

    def test_summarise_rounding_matches(self):
        confidences, corrects, gold, pred = self.sample(18, 7)
        block = self.harness.summarise(confidences, corrects, gold, pred)
        self.assertEqual(round(audit.mean(corrects), 4), block["accuracy"])
        self.assertEqual(round(audit.macro_f1(gold, pred), 4), block["macro_f1"])
        self.assertEqual(round(audit.ece(confidences, corrects), 4), block["ece"])
        self.assertEqual(round(audit.mean(confidences), 6),
                         round(float(self.np.mean(confidences)), 6))

    def test_std_matches_numpy_population_std(self):
        for seed in range(5):
            confidences, _, _, _ = self.sample(40, seed)
            self.assertAlmostEqual(audit.std(confidences),
                                   float(self.np.std(confidences)), places=12)


class TestReadmeQuotesTheArchive(unittest.TestCase):
    """A wrong number in the prose is the failure this benchmark exists to catch.

    Every value is checked for presence, not for position, so reformatting either table is
    fine but editing a number without regenerating the archive is not. It is a net, not a
    proof: a value that also occurs in another row satisfies it.
    """

    def setUp(self):
        self.doc = audit.load_json(os.path.join(ARCHIVE, "report.json"))
        self.pages = {}
        for name in ("README.md", "README.zh-CN.md"):
            with open(os.path.join(BENCH, name), encoding="utf-8") as f:
                self.pages[name] = f.read()

    def test_accuracy_and_confusion_fractions_are_quoted(self):
        for config, block in self.doc["report"].items():
            records = [r for r in self.doc["cases"] if r["config"] == config]
            fraction = "%d/%d" % (sum(r["correct"] for r in records), len(records))
            for name, text in self.pages.items():
                self.assertIn(fraction, text, "%s %s" % (name, config))
                self.assertIn("%.4f" % block["accuracy"], text, "%s %s" % (name, config))

    def test_dimension_accuracies_are_quoted(self):
        for config in prompts.NOUL_CONFIGS:
            for dim, value in self.doc["report"][config]["accuracy_by_dimension"].items():
                for name, text in self.pages.items():
                    self.assertIn("%.3f" % value, text, "%s %s/%s" % (name, config, dim))

    def test_choice_macro_f1_and_ece_are_quoted(self):
        for config in prompts.CHOICE_CONFIGS:
            block = self.doc["report"][config]
            for key in ("macro_f1", "ece"):
                for name, text in self.pages.items():
                    self.assertIn("%.4f" % block[key], text, "%s %s/%s" % (name, config, key))

    def test_the_weight_hash_is_quoted_whole(self):
        digest = self.doc["config"]["checkpoint"]["weight_sha256"]
        for name, text in self.pages.items():
            self.assertIn(digest, text, name)


class TestArchiveFilesArePresent(unittest.TestCase):
    def test_expected_files(self):
        for name in ("audit.py", "prompts.py", "run.py", "README.md", "README.zh-CN.md",
                     "data/cases.jsonl", "data/manifest.json",
                     "results/v1/laya-multilingual/report.json"):
            self.assertTrue(os.path.exists(os.path.join(BENCH, name)), name)

    def test_no_stray_output_committed(self):
        for _dirpath, _dirnames, filenames in os.walk(BENCH):
            _dirnames[:] = [d for d in _dirnames if d != "__pycache__"]
            for name in filenames:
                self.assertFalse(name.endswith((".pyc", ".tmp")), name)

    def test_every_file_is_lf(self):
        """`.gitattributes` keeps these LF, because audit.py pins the sha256 of the raw
        bytes of two of them. This catches a file written on Windows before that, or a
        report.json regenerated without newline="\\n"."""
        for name in ("data/cases.jsonl", "data/manifest.json", "prompts.py", "run.py",
                     "audit.py", "README.md", "README.zh-CN.md",
                     "results/v1/laya-multilingual/report.json"):
            with open(os.path.join(BENCH, name), "rb") as f:
                self.assertNotIn(b"\r\n", f.read(), name)

    def test_readme_links_the_language_switch(self):
        for name, other in (("README.md", "README.zh-CN.md"),
                            ("README.zh-CN.md", "README.md")):
            with open(os.path.join(BENCH, name), encoding="utf-8") as f:
                self.assertIn("(%s)" % other, f.read(), name)


if __name__ == "__main__":
    unittest.main()
