"""Offline tests: python -m unittest research.eval.test_metamorphic -v."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research.eval import metamorphic as m

CASE = ({"utterance": "I was charged twice"}, {"intent": {
    "type": "choice", "instructions": "Choose the request category.",
    "criteria": {"billing": "payment issues", "technical": "software help", "sales": "new purchases"},
}})


def semantic_scorer(cases):
    weights = {"payment issues": 0.7, "software help": 0.2, "new purchases": 0.1}
    return [[weights[v] for v in q["intent"]["criteria"].values()] for _, q in cases]


class TransformTests(unittest.TestCase):
    def test_mapping_and_unchanged_inputs(self):
        original = deepcopy(CASE)
        baseline, order, rename = m.make_variants(CASE, random.Random(13))
        self.assertEqual(CASE, original)
        self.assertEqual(baseline["case"], CASE)
        self.assertNotEqual(order["canonical_indices"], [0, 1, 2])
        self.assertEqual(rename["kind"], "label_rename")
        for variant in (baseline, order, rename):
            criteria = variant["case"][1]["intent"]["criteria"]
            descriptions = list(CASE[1]["intent"]["criteria"].values())
            self.assertEqual(list(criteria.values()),
                             [descriptions[i] for i in variant["canonical_indices"]])
        order["case"][0]["utterance"] = "changed"
        self.assertEqual(CASE, original)

    def test_rename_labels_preserves_order_and_descriptions(self):
        case = m.MetamorphicCase(*CASE)
        variant = m.rename_labels(case)
        self.assertEqual(variant.kind, "label_rename")
        self.assertEqual(variant.canonical_to_transformed,
                         {"billing": "A", "technical": "B", "sales": "C"})
        criteria = variant.case.questions["intent"]["criteria"]
        self.assertEqual(list(criteria), ["A", "B", "C"])
        self.assertEqual(list(criteria.values()),
                         list(CASE[1]["intent"]["criteria"].values()))
        self.assertEqual(variant.as_record()["canonical_indices"], [0, 1, 2])
        again = m.rename_labels(case)
        self.assertEqual(variant, again)

    def test_rename_labels_fallback_beyond_alphabet(self):
        case = deepcopy(CASE)
        case[1]["intent"]["criteria"] = {str(i): f"description {i}" for i in range(30)}
        variant = m.rename_labels(m.MetamorphicCase(*case))
        self.assertEqual(variant.canonical_to_transformed["0"], "key_0")
        self.assertEqual(variant.canonical_to_transformed["29"], "key_29")
        self.assertEqual(len(set(variant.canonical_to_transformed.values())), 30)

    def test_rename_labels_exposes_lexical_sensitivity(self):
        # A scorer keyed on the label tokens (not the descriptions) is
        # invariant under permutation but drifts under renaming.
        def label_scorer(cases):
            # Opaque labels pull the probability mass flat AND move the winner:
            # baseline picks billing (0.7), the renamed input picks B (0.4),
            # which canonicalizes back to technical.
            weights = {"billing": 0.7, "technical": 0.2, "sales": 0.1,
                       "A": 0.3, "B": 0.4, "C": 0.3}
            return [[weights[v] for v in q["intent"]["criteria"]] for _, q in cases]
        case = m.MetamorphicCase(*CASE, gold_index=0)
        result = m.evaluate_variants(None, case,
                                     [m.permute_options(case, seed=42), m.rename_labels(case)],
                                     score=label_scorer)
        self.assertEqual(result.variants[0]["probabilities"], [0.7, 0.2, 0.1])
        self.assertEqual(result.variants[1]["probabilities"], [0.3, 0.4, 0.3])
        report = m.compare_predictions(baseline=result.baseline, variants=result.variants)
        self.assertEqual(report["option_order"]["semantic_agreement_rate"], 1)
        self.assertEqual(report["option_order"]["max_probability_drift"], 0)
        self.assertEqual(report["label_rename"]["semantic_agreement_rate"], 0)
        self.assertGreater(report["label_rename"]["max_probability_drift"], 0.3)

    def test_identity_shuffle_fallback(self):
        with patch("random.Random.shuffle", return_value=None):
            variant = m.permute_options(m.MetamorphicCase(*CASE), seed=42)
        self.assertEqual(variant.as_record()["canonical_indices"], [1, 2, 0])

    def test_explicit_api_mapping_and_gold(self):
        case = m.MetamorphicCase(*CASE, gold_index=0)
        permuted = m.permute_options(case, seed=42)
        self.assertEqual(permuted, m.permute_options(case, seed=42))
        self.assertEqual(permuted.canonical_to_transformed,
                         {"billing": "billing", "technical": "technical", "sales": "sales"})
        self.assertEqual(permuted.transformed_to_canonical["billing"], "billing")
        self.assertEqual(permuted.case.option_keys[permuted.case.gold_index], "billing")
        result = m.evaluate_variants(None, case, [permuted], score=semantic_scorer)
        report = m.compare_predictions(baseline=result.baseline, variants=result.variants)
        self.assertEqual(report["overall"]["semantic_agreement_rate"], 1)
        self.assertEqual(report["overall"]["max_probability_drift"], 0)

    def test_many_options_and_nested_descriptions(self):
        case = deepcopy(CASE)
        case[1]["intent"]["criteria"] = {str(i): {"description": [str(i)]} for i in range(28)}
        order = m.make_variants(case, random.Random(0))[1]
        criteria = order["case"][1]["intent"]["criteria"]
        self.assertEqual(len(criteria), 28)

    def test_none_or_blank_descriptions_accepted(self):
        for criteria in ({"a": None, "b": "two"}, {"a": " ", "b": ""}, {"a": None, "b": None}):
            case = deepcopy(CASE)
            case[1]["intent"]["criteria"] = criteria
            variants = m.make_variants(case, random.Random(0))
            self.assertEqual(len(variants), 3)

    def test_invalid_cases(self):
        for criteria in ({}, {"a": "one"}, {1: "one", "b": "two"}, {" ": "one", "b": "two"}, {"": "one", "b": "two"}):
            case = deepcopy(CASE)
            case[1]["intent"]["criteria"] = criteria
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                m.make_variants(case, random.Random(0))
        for questions in ({}, {"a": CASE[1]["intent"], "b": CASE[1]["intent"]},
                          {"a": {"type": "noul", "criteria": {"false": "no", "true": "yes"}}}):
            with self.assertRaises(ValueError):
                m.make_variants(({}, questions), random.Random(0))


class MetricTests(unittest.TestCase):
    def test_inverse_mapping(self):
        self.assertEqual(m.canonicalize([0.1, 0.7, 0.2], [2, 0, 1]), [0.7, 0.2, 0.1])

    def test_invalid_distributions(self):
        for vector, mapping in (([], []), ([0.2, 0.8], [0, 0]),
                                ([0.2, 0.8], [0]), ([0.2, 0.8], [0, 2]),
                                ([float("nan"), 0.5], [0, 1]),
                                ([float("inf"), 0], [0, 1]),
                                ([-0.1, 1.1], [0, 1]), ([0.2, 0.2], [0, 1])):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                m.canonicalize(vector, mapping)

    def test_known_divergence_and_drift(self):
        result = m.distribution_metrics([1, 0], [0, 1])
        self.assertFalse(result["semantic_agreement"])
        self.assertAlmostEqual(result["js_divergence"], math.log(2))
        self.assertEqual(result["mean_probability_drift"], 1)
        self.assertEqual(result["confidence_drift"], 0)
        identical = m.distribution_metrics([0.5, 0.5, 0], [0.5, 0.5, 0])
        self.assertTrue(identical["semantic_agreement"])
        self.assertEqual(identical["js_divergence"], 0)

    def test_agreement_is_distinct_from_stability(self):
        same_winner = m.distribution_metrics([0.91, 0.06, 0.03], [0.88, 0.08, 0.04])
        self.assertTrue(same_winner["semantic_agreement"])
        self.assertGreater(same_winner["mean_probability_drift"], 0)
        self.assertGreater(same_winner["js_divergence"], 0)
        changed_winner = m.distribution_metrics([0.91, 0.06, 0.03], [0.08, 0.86, 0.06])
        self.assertFalse(changed_winner["semantic_agreement"])
        self.assertLess(changed_winner["confidence_drift"], 0)

    def test_summary_and_confidence_increase(self):
        pairs = [m.distribution_metrics([0.6, 0.4], [0.1, 0.9]), m.distribution_metrics([0.6, 0.4], [0.8, 0.2])]
        summary = m.summarise_pairs(pairs)
        self.assertEqual(summary["semantic_agreement_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_probability_drift"], 0.35)
        self.assertAlmostEqual(summary["max_probability_drift"], 0.5)
        self.assertAlmostEqual(summary["mean_confidence_drift"], 0.25)
        self.assertAlmostEqual(summary["worst_confidence_increase_on_disagreement"], 0.3)
        self.assertEqual(m.summarise_pairs([]), {"n": 0})
        decreasing = m.summarise_pairs([m.distribution_metrics([0.9, 0.1], [0.4, 0.6])])
        self.assertEqual(decreasing["worst_confidence_increase_on_disagreement"], 0)


class EvaluationTests(unittest.TestCase):
    def test_invariance_batching_seed_gold_and_json(self):
        outputs = []
        for batch_size in (1, 2, 16):
            sizes = []
            def score(cases):
                sizes.append(len(cases))
                return semantic_scorer(cases)
            result = m.evaluate([CASE, CASE], score, [0, None], batch_size=batch_size)
            self.assertLessEqual(max(sizes), batch_size)
            self.assertEqual(result["report"]["overall"]["semantic_agreement_rate"], 1)
            self.assertEqual(result["report"]["overall"]["max_probability_drift"], 0)
            self.assertEqual(result["report"]["quality"]["baseline"]["accuracy"], 1)
            self.assertEqual(result["report"]["quality"]["baseline"]["n_labelled"], 1)
            self.assertAlmostEqual(result["report"]["quality"]["baseline"]["ece"], 0.3)
            outputs.append(result)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])
        self.assertEqual(json.loads(json.dumps(outputs[0], allow_nan=False)), outputs[0])

    def test_position_and_label_bias_detected(self):
        report = m.evaluate([CASE], lambda cases: [[1, 0, 0] for _ in cases])["report"]
        self.assertEqual(report["option_order"]["semantic_agreement_rate"], 0)
        # A pure position scorer is untouched by renaming: order is preserved,
        # so any drift in the label_rename group can only come from labels.
        self.assertEqual(report["label_rename"]["semantic_agreement_rate"], 1)
        self.assertEqual(report["label_rename"]["max_probability_drift"], 0)
        self.assertEqual(report["quality"]["baseline"], {"n_labelled": 0})

    def test_ties_use_canonical_order(self):
        result = m.evaluate([CASE], lambda cases: [[1/3] * 3 for _ in cases])
        self.assertEqual(result["report"]["overall"]["semantic_agreement_rate"], 1)

    def test_empty_bad_scorers_and_gold(self):
        self.assertEqual(m.evaluate([], lambda _: self.fail("should not score"))["report"]["overall"], {"n": 0})
        for score in (lambda cases: [], lambda cases: [[1, 0] for _ in cases]):
            with self.assertRaises(ValueError):
                m.evaluate([CASE], score)
        for gold in ([], [3], [-1], [True], [0.5]):
            with self.assertRaises(ValueError):
                m.evaluate([CASE], semantic_scorer, gold)
        with self.assertRaises(ValueError):
            m.evaluate([CASE], semantic_scorer, batch_size=0)

    def test_model_adapter_temperature(self):
        class Agent:
            temperature = [1.0] * 3
            temperature_raw = [0.5] * 3
            temperature_by_options = {}
            temperature_by_options_raw = {}
        with patch.object(m.harness, "score_cases", return_value=[[0, math.log(3)]]):
            self.assertAlmostEqual(m.model_scorer(Agent())([CASE])[0][1], 0.75)
            self.assertAlmostEqual(m.model_scorer(Agent(), True)([CASE])[0][1], 0.9)

    def test_cli_partial_failure_writes_report_and_fails(self):
        agent = MagicMock()
        agent.device = "cpu"
        agent.cfg = {"max_len": 512, "head_max_len": 192}
        agent.temperature = [1.0] * 3
        agent.temperature_by_options = {}
        rows = [{"text": "x", "label_text": "a"}, {"text": "y", "label_text": "b"}]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "report.json"
            with redirect_stdout(StringIO()), patch("laya.load", return_value=agent), \
                 patch.object(m.harness, "load_language", side_effect=[rows, RuntimeError("offline")]), \
                 patch.object(m, "model_scorer", return_value=lambda cases: [[0.6, 0.4] for _ in cases]):
                code = m.main(["--langs", "en,de", "--n-opts", "2", "--out", str(out)])
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(payload["report"]["de"], {"error": "offline"})
        self.assertEqual(len(payload["cases"]), 2)
        self.assertEqual(payload["config"]["temperature"], [1.0] * 3)


if __name__ == "__main__":
    unittest.main()
