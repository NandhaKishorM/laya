"""Offline tests for the Chinese short-command benchmark: no checkpoint, no network.

Run from the repository root:

    python -m unittest discover -s research/benchmarks/zh_short_commands/tests -v

The pipeline tests drive ``run.py --stub``, whose scorer is a fixed pseudo-random
vector, so they assert the plumbing (record shapes, counts, re-derivation) rather
than any model behaviour.
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

import prompts  # noqa: E402
import run as runner  # noqa: E402

JSON_STATE_CONFIGS = ("choice_json_state", "noul_json_state")
SCENARIO_CONFIGS = ("choice_scenario", "noul_scenario")


def load_cases():
    cases = []
    with open(os.path.join(BENCH, "data", "cases.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


class TestData(unittest.TestCase):
    def setUp(self):
        self.cases = load_cases()

    def test_cases_parse_and_are_unique(self):
        self.assertTrue(self.cases, "cases.jsonl is empty")
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)), "duplicate case ids")

    def test_every_case_has_a_known_gold_label(self):
        for case in self.cases:
            self.assertIn(case["gold"], prompts.LABELS, case["id"])
            self.assertTrue(case["text"].strip(), case["id"])
            self.assertTrue(case["family"].strip(), case["id"])

    def test_every_label_is_covered(self):
        covered = {c["gold"] for c in self.cases}
        self.assertEqual(covered, set(prompts.LABELS), "a label has no case")


class TestPolicy(unittest.TestCase):
    def test_choice_options_are_the_label_policy(self):
        self.assertEqual(tuple(prompts.CHOICE_CRITERIA), tuple(prompts.LABELS))

    def test_four_noul_dimensions(self):
        self.assertEqual(len(prompts.NOUL_DIMENSIONS), 4)
        self.assertEqual(set(prompts.NOUL_INSTRUCTIONS), set(prompts.NOUL_DIMENSIONS))
        self.assertEqual(set(prompts.NOUL_CRITERIA), set(prompts.NOUL_DIMENSIONS))

    def test_noul_criteria_use_the_two_documented_keys(self):
        """`criteria` on a noul is keyed true/false: those keys ARE the option text."""
        for dim, criteria in prompts.NOUL_CRITERIA.items():
            self.assertEqual(sorted(criteria), ["false", "true"], dim)
            self.assertTrue(criteria["true"].strip() and criteria["false"].strip(), dim)
            self.assertNotEqual(criteria["true"], criteria["false"], dim)

    def test_noul_gold_policy(self):
        """Which dimensions a gold label should fire. `none` fires nothing: it is not
        a command, so all four dimensions are negative for it. Direction labels fire
        `is_command` only — the four dimensions cover speed and stop, not direction,
        which is why task A (six-way choice) exists alongside task B."""
        expectations = {
            "faster": ["wants_faster", "is_command"],
            "slower": ["wants_slower", "is_command"],
            "stop": ["wants_stop", "is_command"],
            "left": ["is_command"],
            "right": ["is_command"],
            "none": [],
        }
        self.assertEqual(sorted(expectations), sorted(prompts.LABELS))
        for gold, expected in expectations.items():
            fired = [d for d in prompts.NOUL_DIMENSIONS if prompts.NOUL_GOLD[d](gold)]
            self.assertEqual(fired, expected, gold)

    def test_only_the_none_label_is_not_a_command(self):
        for gold in prompts.LABELS:
            self.assertEqual(prompts.NOUL_GOLD["is_command"](gold), gold != "none", gold)


class TestLadder(unittest.TestCase):
    """The ablation rungs must actually differ in the request they build."""

    def setUp(self):
        self.texts = ["快一点", "停下"]

    def test_every_config_builds_one_pair_per_text(self):
        for config in prompts.CONFIGS:
            self.assertEqual(len(prompts.cases_for(config, self.texts)), len(self.texts), config)

    def test_ladder_shape(self):
        self.assertEqual(list(prompts.CHOICE_CONFIGS),
                         ["choice_criteria", "choice_scenario", "choice_json_state"])
        self.assertEqual(list(prompts.NOUL_CONFIGS),
                         ["noul_plain", "noul_criteria", "noul_scenario", "noul_json_state"])
        self.assertEqual(list(prompts.CONFIGS),
                         list(prompts.CHOICE_CONFIGS) + list(prompts.NOUL_CONFIGS))

    def test_noul_plain_omits_criteria_and_the_higher_rungs_add_them(self):
        plain = prompts.noul_questions("noul_plain")
        for dim, qdef in plain.items():
            self.assertNotIn("criteria", qdef, dim)
        for config in ("noul_criteria", "noul_scenario", "noul_json_state"):
            for dim, qdef in prompts.noul_questions(config).items():
                self.assertEqual(sorted(qdef["criteria"]), ["false", "true"], (config, dim))

    def test_scenario_rungs_prepend_the_scenario(self):
        for base_config, scenario_config in (("choice_criteria", "choice_scenario"),
                                            ("noul_criteria", "noul_scenario")):
            if base_config in prompts.CHOICE_CONFIGS:
                before = prompts.choice_question(base_config)["intent"]["instructions"]
                after = prompts.choice_question(scenario_config)["intent"]["instructions"]
            else:
                before = prompts.noul_questions(base_config)["wants_stop"]["instructions"]
                after = prompts.noul_questions(scenario_config)["wants_stop"]["instructions"]
            self.assertNotEqual(before, after)
            self.assertTrue(after.startswith(prompts.SCENARIO_PREFIX))
            self.assertEqual(after, prompts.SCENARIO_PREFIX + before)

    def test_json_state_is_the_only_wrapped_state(self):
        for config in prompts.CONFIGS:
            state = prompts.state_for(config, "停下")
            if config in JSON_STATE_CONFIGS:
                self.assertIsInstance(state, dict, config)
                self.assertEqual(state["text"], "停下")
            else:
                self.assertEqual(state, "停下", config)

    def test_noul_rungs_ask_the_four_dimensions_in_order(self):
        for config in prompts.NOUL_CONFIGS:
            questions = prompts.cases_for(config, self.texts)[0][1]
            self.assertEqual(tuple(questions), prompts.NOUL_DIMENSIONS, config)
            for qdef in questions.values():
                self.assertEqual(qdef["type"], "noul", config)

    def test_choice_rungs_ask_one_six_option_question(self):
        for config in prompts.CHOICE_CONFIGS:
            questions = prompts.cases_for(config, self.texts)[0][1]
            self.assertEqual(len(questions), 1, config)
            qdef = next(iter(questions.values()))
            self.assertEqual(qdef["type"], "choice")
            self.assertEqual(len(qdef["criteria"]), len(prompts.LABELS))

    def test_unknown_config_is_rejected(self):
        with self.assertRaises(ValueError):
            prompts.cases_for("nope", self.texts)


class TestStubRun(unittest.TestCase):
    """Drive the whole runner offline and check what it wrote."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        # The runner prints its own table; keep the test log clean.
        with contextlib.redirect_stdout(io.StringIO()):
            rc = runner.main(["--stub", "--out", cls._tmp.name])
        if rc != 0:
            raise AssertionError("stub run exited %d" % rc)
        with open(os.path.join(cls._tmp.name, "report.json"), encoding="utf-8") as f:
            cls.doc = json.load(f)
        cls.cases = load_cases()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_document_parts(self):
        self.assertEqual(sorted(self.doc), ["cases", "config", "report", "summary"])
        for config in prompts.CONFIGS:
            self.assertIn(config, self.doc["report"])
            self.assertIn(config, self.doc["summary"])

    def test_record_counts(self):
        n = len(self.cases)
        per_config = {}
        for record in self.doc["cases"]:
            per_config[record["config"]] = per_config.get(record["config"], 0) + 1
        for config in prompts.NOUL_CONFIGS:
            self.assertEqual(per_config[config], n * len(prompts.NOUL_DIMENSIONS), config)
        for config in prompts.CHOICE_CONFIGS:
            self.assertEqual(per_config[config], n, config)

    def test_every_case_is_recorded_once_per_role(self):
        choice = [c for c in self.doc["cases"] if c["config"] == "choice_criteria"]
        self.assertEqual([c["case_id"] for c in choice], [c["id"] for c in self.cases])

    def test_probabilities_are_finite_and_normalised(self):
        for record in self.doc["cases"]:
            if "p_gold" in record:
                for key in ("probability", "p_gold", "confidence"):
                    self.assertGreaterEqual(record[key], 0.0, record["case_id"])
                    self.assertLessEqual(record[key], 1.0, record["case_id"])
                self.assertEqual(record["pred_label"], prompts.LABELS[record["pred_index"]])
            else:
                self.assertGreaterEqual(record["p_true"], 0.0)
                self.assertLessEqual(record["p_true"], 1.0)
                self.assertEqual(record["pred_bool"], record["p_true"] >= 0.5)

    def test_every_record_carries_the_request_it_was_asked(self):
        """The record must say what was actually sent, not just what was scored."""
        by_id = {c["id"]: c for c in self.cases}
        for record in self.doc["cases"]:
            case = by_id[record["case_id"]]
            self.assertEqual(record["text"], case["text"], record["case_id"])
            self.assertEqual(record["state"],
                             prompts.state_for(record["config"], case["text"]),
                             record["case_id"])
            self.assertEqual(record["gold_label"], case["gold"], record["case_id"])
            if "dimension" in record:
                expected = prompts.noul_questions(record["config"])[record["dimension"]]["instructions"]
            else:
                expected = prompts.choice_question(record["config"])["intent"]["instructions"]
            self.assertEqual(record["instructions"], expected, record["case_id"])

    def test_accuracy_re_derives_from_the_cases(self):
        """The laya_eval property: report can be recomputed from cases alone."""
        for config in prompts.CHOICE_CONFIGS:
            records = [c for c in self.doc["cases"] if c["config"] == config]
            acc = sum(r["correct"] for r in records) / len(records)
            self.assertAlmostEqual(
                round(acc, 4), round(self.doc["report"][config]["accuracy"], 4), places=4)

    def test_noul_accuracy_re_derives_per_dimension(self):
        for config in prompts.NOUL_CONFIGS:
            records = [c for c in self.doc["cases"] if c["config"] == config]
            for dim, value in self.doc["report"][config]["accuracy_by_dimension"].items():
                subset = [r for r in records if r["dimension"] == dim]
                self.assertAlmostEqual(
                    round(sum(r["correct"] for r in subset) / len(subset), 4),
                    round(value, 4), places=4, msg="%s/%s" % (config, dim))

    def test_hashes_of_the_frozen_inputs_are_recorded(self):
        self.assertEqual(self.doc["config"]["cases_sha256"],
                         runner.sha256_file(os.path.join(BENCH, "data", "cases.jsonl")))
        self.assertEqual(self.doc["config"]["prompts_sha256"],
                         runner.sha256_file(os.path.join(BENCH, "prompts.py")))

    def test_stub_run_records_no_checkpoint(self):
        self.assertTrue(self.doc["config"]["checkpoint"]["stub"])


if __name__ == "__main__":
    unittest.main()
