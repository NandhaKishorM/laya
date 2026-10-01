"""Weight-free checks: no model, no network. Run directly or through unittest.

    python -m unittest discover -s research/benchmarks/es_phone_turns/tests -v
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "train"))

import audit  # noqa: E402
import build_train  # noqa: E402
import domains  # noqa: E402
import metrics  # noqa: E402
import prompts  # noqa: E402


def record(gold, predicted, confidence, tags=("direct",)):
    return {"id": "x", "text": "t", "gold": gold, "predicted": predicted,
            "confidence": confidence, "tags": list(tags), "latency_ms": 1.0}


class Metrics(unittest.TestCase):
    def test_answering_keep_talking_by_mistake_is_not_a_wrong_action(self):
        # It costs a slower turn, not a call sent to the wrong place.
        records = [record("ventas", "ninguno", 0.9)]
        self.assertEqual(metrics.wrong_actions(records), [])

    def test_acting_when_it_should_not_is_a_wrong_action(self):
        records = [record("ninguno", "ventas", 0.9), record("ventas", "soporte", 0.9),
                   record("ventas", "ventas", 0.9)]
        self.assertEqual(len(metrics.wrong_actions(records)), 2)

    def test_the_gate_only_lets_actions_through(self):
        records = [record("ninguno", "ninguno", 0.99), record("ventas", "ventas", 0.95),
                   record("ninguno", "soporte", 0.97), record("colgar", "colgar", 0.4)]
        row = metrics.fast_path(records, thresholds=(0.9,))[0]
        self.assertEqual(row["fast"], 2)
        self.assertEqual(row["wrong"], 1)
        self.assertAlmostEqual(row["precision"], 0.5)
        # Two turns truly were actions; one of them went through the gate.
        self.assertAlmostEqual(row["action_recall"], 0.5)

    def test_borderline_cases_stay_out_of_the_headline(self):
        records = [record("ventas", "ventas", 0.9),
                   record("ninguno", "ventas", 0.9, tags=("borderline",))]
        summary = metrics.summarize(records, ["ventas", "ninguno"])
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(len(summary["borderline"]), 1)


class Prompts(unittest.TestCase):
    def test_every_rung_asks_the_same_question_with_the_same_options(self):
        for dataset in prompts.MENUS:
            expected = prompts.labels(dataset)
            for rung in prompts.RUNGS:
                _, questions = prompts.build(rung, "hola", dataset)
                self.assertEqual(list(questions["accion"]["criteria"]), expected)

    def test_every_case_has_a_label_of_its_menu(self):
        for dataset in prompts.MENUS:
            labels = set(prompts.labels(dataset))
            path = HERE / "data" / f"{dataset}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                self.assertIn(json.loads(line)["gold"], labels)


class Training(unittest.TestCase):
    def test_no_training_destination_is_a_test_destination(self):
        tested = {label for dataset in prompts.MENUS for label in prompts.labels(dataset)}
        trained = {label for domain in domains.DOMAINS.values()
                   for label in domain["departments"]}
        trained |= {label for label, _ in
                    domains.HUMAN + domains.HANG_UP + domains.KEEP_TALKING}
        self.assertEqual(tested & trained, set())

    def test_a_test_sentence_with_one_word_changed_is_refused(self):
        references = [set(build_train.normalize(
            "No me pase con ventas, solo quiero saber el horario."))]
        near = build_train.normalize("No me pase con tarjetas, solo quiero saber el horario.")
        far = build_train.normalize("Se me perdió la tarjeta y necesito bloquearla ya.")
        self.assertTrue(build_train.too_close(near, references))
        self.assertFalse(build_train.too_close(far, references))

    def test_a_case_always_offers_its_own_answer(self):
        import random
        rng = random.Random(1)
        for kind, department, must in (
                ("department", "tarjetas", ("tarjetas",)), ("human", None, ()),
                ("hang_up", None, ()), ("keep", None, ("tarjetas",))):
            for _ in range(200):
                item = build_train.case(rng, "hola", "banco", kind, department, must, 0.94)
                question = item["questions"]["accion"]
                gold = item["gold"]["accion"]
                self.assertIn(gold["label"], question["criteria"])
                self.assertEqual(list(gold["probabilities"]), list(question["criteria"]))
                self.assertAlmostEqual(sum(gold["probabilities"].values()), 1.0)
                for name in must:
                    self.assertIn(name, question["criteria"])

    def test_the_committed_training_set_holds_no_test_sentence(self):
        path = HERE / "train" / "train.jsonl"
        if not path.exists():
            self.skipTest("train.jsonl has not been built")
        references = []
        for dataset in prompts.MENUS:
            for line in (HERE / "data" / f"{dataset}.jsonl").read_text("utf-8").splitlines():
                references.append(set(build_train.normalize(json.loads(line)["text"])))
        tested = {label for dataset in prompts.MENUS for label in prompts.labels(dataset)}
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            # The recogniser-style rewrite can add a filler or repeat a word; the sentence
            # underneath was checked before that, so the bar here is the exact sentence.
            self.assertNotIn(set(build_train.normalize(row["state"])), references)
            self.assertEqual(set(row["questions"]["accion"]["criteria"]) & tested, set())


class Audit(unittest.TestCase):
    """The audit has to be able to fail. Each test breaks a copy of the archive one way."""

    def setUp(self):
        self.reports = sorted((HERE / "results").glob("*.json"))
        if not self.reports:
            self.skipTest("no results to audit")
        self.dir = Path(tempfile.mkdtemp())
        shutil.copy(self.reports[0], self.dir / self.reports[0].name)
        self.path = self.dir / self.reports[0].name

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def break_with(self, change):
        report = json.loads(self.path.read_text(encoding="utf-8"))
        change(report)
        self.path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        return audit.audit(self.dir)[1]

    def test_the_untouched_archive_passes(self):
        self.assertEqual(audit.audit(self.dir)[1], [])

    def test_a_better_accuracy_than_the_records_give_is_caught(self):
        def change(report):
            report["summary"]["accuracy"] = 0.99
        self.assertTrue(self.break_with(change))

    def test_a_flipped_decision_is_caught(self):
        def change(report):
            first = report["records"][0]
            first["predicted"] = first["gold"] if first["predicted"] != first["gold"] else "colgar"
        self.assertTrue(self.break_with(change))

    def test_a_rewritten_gold_is_caught(self):
        # The cheapest way to look better: turn a miss into a hit by editing the answer key.
        def change(report):
            missed = [r for r in report["records"] if r["predicted"] != r["gold"]]
            target = missed[0] if missed else report["records"][0]
            target["gold"] = target["predicted"] if missed else "colgar"
        self.assertTrue(self.break_with(change))

    def test_a_dropped_case_is_caught(self):
        def change(report):
            report["records"].pop()
        self.assertTrue(self.break_with(change))

    def test_a_hidden_wrong_action_is_caught(self):
        def change(report):
            report["summary"]["wrong_actions"] = 0
            report["summary"]["fast_path"] = []
        self.assertTrue(self.break_with(change))


if __name__ == "__main__":
    unittest.main()
