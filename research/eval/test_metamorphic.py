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


class SelectivePredictionTests(unittest.TestCase):
    def test_auroc_known_values(self):
        self.assertEqual(m.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1)
        self.assertEqual(m.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]), 0)
        self.assertEqual(m.auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]), 0.5)
        self.assertAlmostEqual(m.auroc([0.9, 0.4, 0.6, 0.1], [True, True, False, False]), 0.75)
        self.assertIsNone(m.auroc([0.9, 0.8], [True, True]))
        self.assertIsNone(m.auroc([], []))

    def test_disagreement_flags_confident_errors(self):
        # "stable" cases follow the descriptions and are right at 0.7. "fragile"
        # cases follow the label tokens and are wrong at 0.9; renaming moves their
        # winner. Confidence ranks the errors on top; agreement under renaming
        # ranks them at the bottom. Reordering keeps keys with their descriptions,
        # so it cannot tell the two apart here.
        stable = ({"utterance": "stable"}, deepcopy(CASE[1]))
        fragile = ({"utterance": "fragile"}, deepcopy(CASE[1]))
        label_weights = {"billing": 0.9, "technical": 0.05, "sales": 0.05, "A": 0.2, "B": 0.6, "C": 0.2}

        def score(cases):
            out = []
            for state, questions in cases:
                if state["utterance"] == "stable":
                    out.extend(semantic_scorer([(state, questions)]))
                else:
                    out.append([label_weights[k] for k in questions["intent"]["criteria"]])
            return out

        report = m.evaluate([stable, stable, fragile, fragile], score, [0, 0, 1, 1])["report"]
        sp = report["selective_prediction"]
        self.assertEqual((sp["n_labelled"], sp["n_wrong"]), (4, 2))
        self.assertEqual(sp["auroc"]["confidence"], 0)
        self.assertEqual(sp["auroc"]["agreement_label_rename"], 1)
        self.assertEqual(sp["auroc"]["support_label_rename"], 1)
        self.assertEqual(sp["auroc"]["agreement_option_order"], 0.5)
        self.assertEqual(sp["auroc"]["support_all"], 1)
        self.assertEqual(sp["accuracy_at_coverage"]["confidence"]["0.5"], 0)
        self.assertEqual(sp["accuracy_at_coverage"]["support_all"]["0.5"], 1)
        self.assertNotIn("agreement_label_rename", sp["accuracy_at_coverage"])

    def test_unlabelled_or_one_class(self):
        report = m.evaluate([CASE], semantic_scorer)["report"]
        self.assertEqual(report["selective_prediction"], {"n_labelled": 0})
        sp = m.evaluate([CASE, CASE], semantic_scorer, [0, 0])["report"]["selective_prediction"]
        self.assertEqual((sp["n_labelled"], sp["n_wrong"]), (2, 0))
        self.assertIsNone(sp["auroc"]["confidence"])
        self.assertEqual(sp["accuracy_at_coverage"]["confidence"]["0.5"], 1)
class CharTok:
    """One id per character, so option spans are predictable without model weights."""

    cls_token_id = 2
    sep_token_id = 3
    mask_token = "[MASK]"
    mask_token_id = 1

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (ord(char) % 90) for char in text]
        return {"input_ids": ids[:max_length] if truncation and max_length is not None else ids}


class DriftingTok(CharTok):
    """A truncating tokenizer whose capped output is not a slice of the uncapped one."""

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        result = super().__call__(text, add_special_tokens, truncation, max_length)
        if truncation and max_length is not None and len(text) > max_length:
            result["input_ids"][-1] = 99
        return result


class DroppingTok(CharTok):
    """One id per character, but the tokenizer discards "~", so some prefixes
    encode to the same span as their shorter neighbours."""

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) for char in text if char != "~"]
        return {"input_ids": ids[:max_length] if truncation and max_length is not None else ids}


def lamp_case(labels, descriptions, instructions="Choose the lamp command."):
    return ({"utterance": "the room is dark"},
            {"intent": {"type": "choice", "instructions": instructions,
                        "criteria": dict(zip(labels, descriptions))}})


def uniform_scorer(cases):
    return [[1.0 / len(q["intent"]["criteria"])] * len(q["intent"]["criteria"]) for _, q in cases]


def tight_case():
    """Labels that share a four-character prefix, so a tight budget merges their spans."""
    return lamp_case(["iot_hue_light%d" % i for i in range(8)],
                     ["turn off lamp %d" % i for i in range(8)])


class BudgetTests(unittest.TestCase):
    def test_roomy_case_is_not_budget_confounded(self):
        case = m.MetamorphicCase(*lamp_case(
            ["billing", "technical", "sales"],
            ["payment issues", "software help", "new purchases"]))
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=192)
        result = m.evaluate([case.as_pair()], uniform_scorer, budget=probe)
        variants = result["cases"][0]["variants"]
        for variant in variants:
            self.assertEqual(variant["budget"]["distinct_spans"], 3)
            self.assertIsNone(variant["budget"]["tokens_per_option"])
        self.assertEqual(variants[2]["budget"]["retained_descriptions"],
                         ["payment issues", "software help", "new purchases"])
        self.assertIs(variants[2]["budget_comparison"]["budget_confounded"], False)
        self.assertEqual(variants[2]["budget_comparison"]["reasons"], [{"code": "clean"}])
        self.assertEqual(result["report"]["budget"]["label_rename"],
                         {"n": 1, "clean": 1, "confounded": 0, "unknown": 0, "confounded_rate": 0.0})
        self.assertEqual(result["report"]["budget"]["overall"]["n"], 2)

    def test_tight_budget_exposes_collapsed_spans_and_content(self):
        # At head_max_len=64 the per-option re-cap keeps four characters, so the baseline
        # labels collapse to "iot_" and the renamed option keeps "t" of its description
        # instead (#543/#569).
        case = m.MetamorphicCase(*tight_case())
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=64)
        result = m.evaluate([case.as_pair()], uniform_scorer, budget=probe)
        variants = result["cases"][0]["variants"]
        baseline, rename = variants[0], variants[2]
        self.assertEqual(baseline["budget"]["distinct_spans"], 1)
        self.assertEqual(baseline["budget"]["retained_text"], ["iot_"] * 8)
        self.assertEqual(baseline["budget"]["retained_descriptions"], [""] * 8)
        self.assertEqual(rename["budget"]["distinct_spans"], 8)
        self.assertEqual(rename["budget"]["retained_text"], ["%s: t" % label for label in "ABCDEFGH"])
        self.assertEqual(rename["budget"]["retained_descriptions"], ["t"] * 8)
        comparison = rename["budget_comparison"]
        self.assertIs(comparison["budget_confounded"], True)
        self.assertEqual([reason["code"] for reason in comparison["reasons"]],
                         ["distinct_spans", "collision_partition", "retained_description"])
        self.assertEqual(comparison["reasons"][-1]["slots"], list(range(8)))
        # Permuting the same collapsing options still moves every key/value pair together.
        self.assertIs(variants[1]["budget_comparison"]["budget_confounded"], False)
        self.assertEqual(result["report"]["budget"]["label_rename"],
                         {"n": 1, "clean": 0, "confounded": 1, "unknown": 0, "confounded_rate": 1.0})
        self.assertEqual(result["report"]["budget"]["overall"]["confounded_rate"], 0.5)

    def test_permutation_alone_is_never_information_loss(self):
        case = m.MetamorphicCase(*lamp_case(["o%03d" % i for i in range(28)],
                                            ["description %d" % i for i in range(28)]))
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=184)
        permuted = m.permute_options(case, seed=7)
        result = m.evaluate_variants(None, case, [permuted], score=uniform_scorer, budget=probe)
        self.assertEqual(result.baseline["budget"]["distinct_spans"], 28)
        self.assertEqual(result.variants[0]["budget"]["distinct_spans"], 28)
        self.assertEqual(result.variants[0]["budget_comparison"]["reasons"], [{"code": "clean"}])

    def test_both_sides_collapse_differently(self):
        # Equal collision counts, different collided slots: a count-only diagnostic
        # would call this clean, so the partition comparison has to catch it.
        case = m.MetamorphicCase(*lamp_case(
            ["aaaa1", "aaaa2", "cccc", "dddd"], ["one", "two", "three", "four"]))
        labels = ["bbbb", "aaaa3", "aaaa4", "dddd"]
        variant_case = m.MetamorphicCase(*lamp_case(labels, ["one", "two", "three", "four"]))
        mapping = dict(zip(case.option_keys, labels))
        variant = m.MetamorphicVariant("label_rename", variant_case, mapping)
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=38)
        result = m.evaluate_variants(None, case, [variant], score=uniform_scorer, budget=probe)
        baseline, variant_record = result.baseline, result.variants[0]
        self.assertEqual(baseline["budget"]["distinct_spans"], 3)
        self.assertEqual(variant_record["budget"]["distinct_spans"], 3)
        self.assertEqual(baseline["budget"]["span_classes"], [0, 0, 1, 2])
        self.assertEqual(variant_record["budget"]["span_classes"], [0, 1, 1, 2])
        comparison = variant_record["budget_comparison"]
        self.assertIs(comparison["budget_confounded"], True)
        self.assertEqual([reason["code"] for reason in comparison["reasons"]],
                         ["collision_partition"])

    def test_only_the_variant_collapses(self):
        # The mirror of the tight case: the reverse direction, where the baseline
        # labels stay distinct and the longer renamed labels push the differing
        # character past the re-cap, has to be reported the same way.
        baseline_case = m.MetamorphicCase(*lamp_case(["a", "b", "c", "d"], ["turn off lamp"] * 4))
        variant_case = m.MetamorphicCase(*lamp_case(
            ["iot_hue_light%d" % i for i in range(4)], ["turn off lamp"] * 4))
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=40)
        baseline = probe.measure(*baseline_case.as_pair())
        variant = probe.measure(*variant_case.as_pair())
        self.assertEqual(baseline["retained_text"], ["%s: t" % label for label in "abcd"])
        self.assertEqual(baseline["distinct_spans"], 4)
        self.assertEqual(variant["retained_text"], ["iot_"] * 4)
        self.assertEqual(variant["distinct_spans"], 1)
        comparison = m.compare_budgets(baseline, variant, [0, 1, 2, 3], kind="label_rename")
        self.assertIs(comparison["budget_confounded"], True)
        self.assertEqual([reason["code"] for reason in comparison["reasons"]],
                         ["distinct_spans", "collision_partition", "retained_description"])

    def test_non_involutive_permutation_keeps_descriptions_aligned(self):
        # canonical A B C presented B C A: [1, 2, 0] is a 3-cycle, not its own inverse,
        # so indexing the variant by it scrambles the slots instead of restoring them.
        case = m.MetamorphicCase(*lamp_case(
            ["A", "B", "C"], ["alpha description", "beta description", "gamma description"]))
        variant = m.MetamorphicVariant(
            "option_order",
            m.MetamorphicCase(*lamp_case(["B", "C", "A"],
                                         ["beta description", "gamma description", "alpha description"])),
            {"A": "A", "B": "B", "C": "C"})
        self.assertEqual(variant.as_record()["canonical_indices"], [1, 2, 0])
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=192)
        result = m.evaluate_variants(None, case, [variant], score=uniform_scorer, budget=probe)
        record = result.variants[0]
        self.assertEqual(record["budget"]["retained_descriptions"],
                         ["beta description", "gamma description", "alpha description"])
        self.assertIs(record["budget_comparison"]["budget_confounded"], False)
        self.assertEqual(record["budget_comparison"]["reasons"], [{"code": "clean"}])

    def test_non_involutive_permutation_keeps_collapsed_partitions_aligned(self):
        # The same 3-cycle with labels that collide under the re-cap: the span classes
        # have to be restored to canonical order before the partitions are compared.
        case = m.MetamorphicCase(*lamp_case(
            ["aaaa1", "aaaa2", "cccc"], ["one", "two", "three"]))
        variant = m.MetamorphicVariant(
            "option_order",
            m.MetamorphicCase(*lamp_case(["aaaa2", "cccc", "aaaa1"], ["two", "three", "one"])),
            {"aaaa1": "aaaa1", "aaaa2": "aaaa2", "cccc": "cccc"})
        self.assertEqual(variant.as_record()["canonical_indices"], [1, 2, 0])
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=35)
        result = m.evaluate_variants(None, case, [variant], score=uniform_scorer, budget=probe)
        self.assertEqual(result.baseline["budget"]["span_classes"], [0, 0, 1])
        record = result.variants[0]
        self.assertEqual(record["budget"]["span_classes"], [0, 1, 0])
        self.assertIs(record["budget_comparison"]["budget_confounded"], False)
        self.assertEqual(record["budget_comparison"]["reasons"], [{"code": "clean"}])

    def test_ambiguous_prefix_attribution_is_unknown(self):
        # DroppingTok discards "~", so "aaaa: b" and "aaaa: b~" encode to the same span:
        # which characters the model kept is undecidable, and neither may be asserted.
        case = m.MetamorphicCase(*lamp_case(["aaaa", "cccc"], ["b~zzz", "d~zzz"]))
        probe = m.BudgetProbe(DroppingTok(), max_len=512, head_max_len=34)
        result = m.evaluate([case.as_pair()], uniform_scorer, budget=probe)
        variants = result["cases"][0]["variants"]
        self.assertTrue(variants[0]["budget"]["truncated"])
        self.assertEqual(variants[0]["budget"]["retained_text"], [None, None])
        self.assertEqual(variants[1]["budget"]["retained_text"], [None, None])
        for variant in variants[1:]:
            comparison = variant["budget_comparison"]
            self.assertIsNone(comparison["budget_confounded"])
            self.assertEqual(comparison["reasons"],
                             [{"code": "unverified_retained_text", "slots": [0, 1]}])

    def test_attribution_requires_a_globally_unique_witness(self):
        # Two prefixes encode to the same span 15 characters apart, and the length
        # profile dips back to the span's length only at the second one: a search
        # bounded to the neighbourhood of the bisection point sees a single witness
        # and would attribute the span to it.
        text, kept = "x" * 40, [9, 9]
        table = {size: [1] for size in range(5)}
        table.update({size: [0, 0, 0] for size in range(5, 40)})
        table[5], table[20] = list(kept), list(kept)
        table[40] = kept + [0, 0]

        def encode(part):
            return table[len(part)]

        self.assertEqual(encode(text[:5]), kept)
        self.assertEqual(encode(text[:20]), kept)
        self.assertIsNone(m._retained_prefix(encode, text, kept, encode(text)))

    def test_overlong_options_are_not_attributed(self):
        # Above the documented length bound the prefix search is skipped rather than
        # guessed, even though the unique witness here would be trivially findable.
        text = "y" * (m._ATTRIBUTION_LIMIT + 1)

        def encode(part):
            return [7] * len(part)

        self.assertIsNone(m._retained_prefix(encode, text, [7], encode(text)))

    def test_renaming_without_descriptions_is_unknown_not_clean(self):
        # Descriptions may legitimately be absent; then the label is the only semantic
        # content, and renaming it makes the experiment meaningless, not clean.
        case = m.MetamorphicCase(*lamp_case(["billing", "technical", "sales"], [None, None, None]))
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=192)
        result = m.evaluate([case.as_pair()], uniform_scorer, budget=probe)
        variants = result["cases"][0]["variants"]
        baseline, renamed = variants[0], variants[2]
        self.assertEqual(baseline["budget"]["description_present"], [False] * 3)
        self.assertEqual(baseline["budget"]["retained_descriptions"], [""] * 3)
        self.assertEqual(renamed["presented_options"], ["A", "B", "C"])
        comparison = renamed["budget_comparison"]
        self.assertIsNone(comparison["budget_confounded"])
        self.assertEqual(comparison["reasons"],
                         [{"code": "missing_semantic_description", "slots": [0, 1, 2]}])
        self.assertEqual(comparison["retained_descriptions"],
                         {"baseline": [""] * 3, "variant": [""] * 3})
        # The model observation is untouched: the stub sees the same distribution.
        self.assertTrue(renamed["comparison"]["semantic_agreement"])
        self.assertEqual(renamed["probabilities"], [1 / 3] * 3)
        # Permuting label-only options moves every key/value pair together and stays valid.
        self.assertIs(variants[1]["budget_comparison"]["budget_confounded"], False)
        self.assertEqual(result["report"]["budget"]["label_rename"],
                         {"n": 1, "clean": 0, "confounded": 0, "unknown": 1, "confounded_rate": None})

    def test_blank_descriptions_are_missing_too(self):
        for blank in ("", "   ", "\t"):
            with self.subTest(description=repr(blank)):
                case = m.MetamorphicCase(*lamp_case(["billing", "technical"], [blank, "software help"]))
                probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=192)
                result = m.evaluate([case.as_pair()], uniform_scorer, budget=probe)
                comparison = result["cases"][0]["variants"][2]["budget_comparison"]
                self.assertIsNone(comparison["budget_confounded"])
                self.assertEqual(comparison["reasons"],
                                 [{"code": "missing_semantic_description", "slots": [0]}])

    def test_structured_descriptions_are_not_missing(self):
        case = m.MetamorphicCase(*lamp_case(
            ["billing", "technical", "sales"],
            [{"description": ["payment issues"]}, 0, False]))
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=192)
        result = m.evaluate([case.as_pair()], uniform_scorer, budget=probe)
        renamed = result["cases"][0]["variants"][2]
        self.assertEqual(renamed["budget"]["description_present"], [True, True, True])
        self.assertIs(renamed["budget_comparison"]["budget_confounded"], False)
        self.assertEqual(renamed["budget_comparison"]["reasons"], [{"code": "clean"}])

    def test_description_presence_classification(self):
        for value, expected in ((None, False), ("", False), ("   ", False), ("\t", False),
                                ("payment issues", True), ("  padded  ", True),
                                (0, True), (False, True), ({}, True), ([], True)):
            with self.subTest(value=value):
                self.assertIs(m._description_present(value), expected)

    def test_unverifiable_rendering_stays_conservative(self):
        # Two long options whose spans the tokenizer cap cuts mid-option: the probe cannot
        # attribute the surviving span to a character boundary, so it reports unknown.
        case = m.MetamorphicCase(*lamp_case(["alpha", "beta"],
                                            ["description " * 8, "description " * 8]))
        probe = m.BudgetProbe(DriftingTok(), max_len=512, head_max_len=192)
        result = m.evaluate([case.as_pair()], uniform_scorer, budget=probe)
        for variant in result["cases"][0]["variants"]:
            self.assertTrue(variant["budget"]["truncated"])
            self.assertIsNone(variant["budget"]["retained_text"][0])
        for variant in result["cases"][0]["variants"][1:]:
            comparison = variant["budget_comparison"]
            self.assertIsNone(comparison["budget_confounded"])
            self.assertEqual(comparison["reasons"][0]["code"], "unverified_retained_text")
        self.assertEqual(result["report"]["budget"]["overall"],
                         {"n": 2, "clean": 0, "confounded": 0, "unknown": 2, "confounded_rate": None})

    def test_broken_tokenizer_is_reported_not_raised(self):
        case = m.MetamorphicCase(*CASE)
        probe = m.BudgetProbe(object(), max_len=512, head_max_len=192)
        result = m.evaluate([case.as_pair()], semantic_scorer, budget=probe)
        comparison = result["cases"][0]["variants"][1]["budget_comparison"]
        self.assertIsNone(comparison["budget_confounded"])
        self.assertEqual(comparison["reasons"][0]["code"], "measurement_error")
        self.assertIsNone(result["report"]["budget"]["overall"]["confounded_rate"])

    def test_without_a_probe_nothing_changes(self):
        case = m.MetamorphicCase(*CASE)
        result = m.evaluate([case.as_pair()], semantic_scorer)
        self.assertNotIn("budget", result["report"])
        for variant in result["cases"][0]["variants"]:
            self.assertNotIn("budget", variant)
            self.assertNotIn("budget_comparison", variant)

    def test_probe_is_deterministic_and_serialisable(self):
        case = m.MetamorphicCase(*tight_case())
        first = m.evaluate([case.as_pair()], uniform_scorer,
                           budget=m.BudgetProbe(CharTok(), max_len=512, head_max_len=64))
        second = m.evaluate([case.as_pair()], uniform_scorer,
                            budget=m.BudgetProbe(CharTok(), max_len=512, head_max_len=64))
        self.assertEqual(first, second)
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)
        probe = m.BudgetProbe(CharTok(), max_len=512, head_max_len=64)
        self.assertEqual(probe.measure(*case.as_pair()), probe.measure(*case.as_pair()))
        self.assertEqual(probe.measure(*case.as_pair()), m.BudgetProbe(CharTok(), 512, 64).measure(*case.as_pair()))


if __name__ == "__main__":
    unittest.main()
