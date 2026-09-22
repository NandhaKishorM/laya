"""Offline Arabic preset, corpus, routing and benchmark-metric regressions."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laya

spec = importlib.util.spec_from_file_location("bench_arabic", ROOT / "research/scripts/bench_arabic.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
from research.scripts import bench_arabic_workflows as workflows


class ArabicTests(unittest.TestCase):
    def setUp(self):
        self.data = bench.load_cases(bench.DEFAULT_DATA)

    def test_parallel_corpus_coverage(self):
        self.assertEqual(len(self.data["cases"]), 36)
        for group in bench.GROUPS:
            cases = [c for c in self.data["cases"] if c["group"] == group]
            self.assertEqual(len(cases), 12)
            self.assertEqual({c["expected"]["category"] for c in cases}, set(bench.CATEGORIES))
            self.assertEqual({c["expected"]["refund_requested"] for c in cases}, {False, True})
            self.assertEqual({c["expected"]["urgency"] for c in cases}, {0, 1, 2})
        for key in {c["parallel_id"] for c in self.data["cases"]}:
            cases = [c for c in self.data["cases"] if c["parallel_id"] == key]
            self.assertEqual({c["group"] for c in cases}, set(bench.GROUPS))
            self.assertTrue(all(c["expected"] == cases[0]["expected"] for c in cases))

    def test_presets_keep_the_same_output_contract(self):
        ar, en = bench.questions("ar"), bench.questions("en")
        self.assertEqual(set(ar), set(en))
        self.assertEqual(list(ar["category"]["criteria"]), list(en["category"]["criteria"]))
        for key in ar:
            self.assertEqual(ar[key]["type"], en[key]["type"])
            self.assertTrue(any("\u0600" <= c <= "\u06ff" for c in ar[key]["instructions"]))
            q = laya.Agent._to_internal(ar[key])
            self.assertEqual(len(laya.render_options(q)), {"category": 4, "refund_requested": 2, "urgency": 3}[key])
        ar["category"]["criteria"].clear()
        self.assertEqual(len(laya.arabic_triage_questions()["category"]["criteria"]), 4)

    def test_routing_only_never_loads_weights(self):
        router = laya.Router()
        router.load = lambda *a, **k: self.fail("routing-only tried loading weights")
        report = bench.run(self.data, router, ["ar", "en"], routing_only=True)
        self.assertEqual(len(report["cases"]), 72)
        self.assertEqual(len(report["summaries"]), 8)
        self.assertTrue(all(s["routing_accuracy"] == 1 for s in report["summaries"].values()))
        self.assertTrue(all("category_accuracy" not in s for s in report["summaries"].values()))

    def test_arabic_script_variants(self):
        for text in ["أَبَا فُلُوسِي", "ﺃﺑﺎ ﻓﻠﻮﺳﻲ", "Please refund this payment، أبا فلوسي", "أبا المبلغ ١٢٣ د.إ"]:
            self.assertEqual(laya.Router().route(text).model, "multilingual", text)

    def test_structured_instructions_preserve_arabic(self):
        instructions = {"task": "صنف الرسالة", "rules": ["اقرأ النص كاملا"]}
        q = laya.Agent._to_internal({"type": "noul", "instructions": instructions})
        self.assertIn("صنف الرسالة", q["ins"])
        self.assertNotIn("\\u", q["ins"])
        self.assertEqual(json.loads(q["ins"]), instructions)

    def test_arabic_questions_with_english_state(self):
        router = laya.Router()
        state = "Please help with my account."
        for qs in [laya.arabic_triage_questions(),
                   {"topic": {"type": "choice", "instructions": "Which topic?",
                              "criteria": {"حساب": None, "توصيل": None}}}]:
            decision = router.route(state, qs)
            self.assertEqual(decision.model, "multilingual")
            self.assertIn("question text", decision.reason)
            self.assertTrue(decision["detection"]["is_english"])
            self.assertGreater(decision["question_detection"]["script_profile"]["arabic"], 0)
            self.assertEqual(router.route(state, qs, model="english").model, "english")
            self.assertEqual(router.route(state, qs, lang="en").model, "english")
        self.assertEqual(router.route(state, {"سؤال": {"type": "noul", "instructions": "Is this urgent?"}}).model,
                         "english")

    def test_metrics_use_probability_not_entropy_confidence(self):
        row = copy.deepcopy(self.data["cases"][0])
        row.update(route={"model": "multilingual"}, latency_ms=10, answers={
            "category": {"choice": "billing", "confidence": 0.123,
                         "probabilities": {"billing": 0.7, "technical": 0.1, "delivery": 0.1, "general": 0.1}},
            "refund_requested": {"noul": 0.8}, "urgency": {"score": 0.25},
        })
        metrics = bench.summarize([row])
        self.assertEqual(metrics["category_accuracy"], 1)
        self.assertAlmostEqual(metrics["category_ece"], 0.3)
        self.assertAlmostEqual(metrics["refund_brier"], 0.04)
        self.assertAlmostEqual(metrics["refund_ece"], 0.2)
        self.assertEqual(metrics["urgency_mae"], 0.25)
        self.assertEqual(metrics["latency_p95_ms"], 10)
        row["answers"]["refund_requested"]["noul"] = 0.2
        row["answers"]["urgency"]["score"] = 1.5
        wrong = bench.summarize([row])
        self.assertEqual(wrong["refund_accuracy"], 0)
        self.assertAlmostEqual(wrong["refund_brier"], 0.64)
        self.assertEqual(wrong["urgency_mae"], 1.5)
        row["answers"]["refund_requested"]["noul"] = float("nan")
        with self.assertRaises(ValueError):
            bench.summarize([row])

    def test_ece_bin_endpoints(self):
        self.assertEqual(bench.ece([0, 1], [0, 1]), 0)
        self.assertEqual(bench.ece([0, 1], [1, 0]), 1)
        self.assertEqual(bench.percentile([0, 10, 20], 0.95), 19)

    def test_inference_path_records_actual_outputs(self):
        class StubRouter:
            def route(self, *args):
                return {"model": "multilingual"}

            def predict(self, *args):
                return {"routing": self.route(), "usage": {"input_tokens": 5, "output_tokens": 0}, "answers": {
                    "category": {"choice": "general", "probabilities": dict(zip(bench.CATEGORIES, [0, 0, 0, 1]))},
                    "refund_requested": {"noul": 0}, "urgency": {"score": 1},
                }}
        result = bench.run(self.data, StubRouter(), ["ar"])
        self.assertEqual(result["summaries"]["ar/all"]["category_accuracy"], 0.25)
        self.assertEqual(result["summaries"]["ar/all"]["refund_accuracy"], 0.75)
        self.assertAlmostEqual(result["summaries"]["ar/all"]["urgency_mae"], 2 / 3)
        self.assertEqual(result["cases"][0]["usage"]["input_tokens"], 5)

    def test_invalid_dataset_rejected(self):
        for change in (lambda d: d["cases"].append(d["cases"][0]),
                       lambda d: d["cases"][0]["expected"].update(refund_requested="true"),
                       lambda d: d["cases"][0]["expected"].update(urgency=True)):
            data = copy.deepcopy(self.data)
            change(data)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "invalid.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    bench.load_cases(path)

    def test_workflow_routing_coverage(self):
        data = workflows.load_data(workflows.DATA)
        self.assertEqual(len(data["cases"]), 80)
        self.assertEqual(set(data["workflows"]), {"intent", "sentiment", "moderation", "relevance", "urgency"})
        router = laya.Router()
        router.load = lambda *a, **k: self.fail("routing-only loaded a model")
        report = workflows.evaluate(data, router, routing_only=True)
        self.assertEqual(len(report["cases"]), 160)
        for row in report["cases"]:
            self.assertEqual(row["current"]["model"], row["expected_route"], row["id"])
            self.assertNotIn("answer", row["current"])
        self.assertTrue(any(r["original"]["model"] != r["expected_route"] for r in report["cases"]))
        for name in data["workflows"]:
            for group in workflows.GROUPS:
                self.assertEqual(sum(c["workflow"] == name and c["group"] == group for c in data["cases"]), 4)

    def test_workflow_metrics_and_baselines(self):
        rows = [{"type": "noul", "expected": True, "expected_route": "multilingual",
                 "current": {"model": "multilingual", "answer": {"noul": 0.8}}},
                {"type": "noul", "expected": False, "expected_route": "multilingual",
                 "current": {"model": "multilingual", "answer": {"noul": 0.8}}}]
        result = workflows.summarize(rows, "current")
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["brier"], 0.34)
        self.assertAlmostEqual(result["ece"], 0.3)
        self.assertEqual(result["majority_accuracy"], 0.5)
        score = {"type": "score", "max_score": 2, "expected": 2, "expected_route": "multilingual",
                 "current": {"model": "multilingual", "answer": {"score": 1.5}}}
        result = workflows.summarize([score], "current")
        self.assertEqual(result["mae"], 0.5)
        self.assertEqual(result["nearest_level_accuracy"], 1)
        rows[0]["current"]["answer"]["noul"] = float("nan")
        with self.assertRaises(ValueError):
            workflows.summarize(rows, "current")


if __name__ == "__main__":
    unittest.main()
