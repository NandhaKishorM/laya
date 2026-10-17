"""Offline contracts: python -m unittest research.eval.test_context_robustness -v."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from collections import Counter
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from laya.confidence import apply_confidence_gate
from research.eval import context_robustness as c


def case(cid="one"):
    return {"schema": c.CASE_SCHEMA, "id": cid, "state": cid + ":clean",
            "questions": {"route": {"type": "choice", "instructions": "Choose the request category.",
                                     "criteria": {"A": "billing", "B": None, "C": ""}}},
            "expected": "A", "gold_rationale": "The request concerns a payment.",
            "variants": [
                {"id": "footer", "kind": "benign", "state": cid + ":footer",
                 "rationale": "A receipt footer does not change the request."},
                {"id": "policy", "kind": "adversarial", "state": cid + ":policy",
                 "control_id": "footer", "rationale": "Quoted routing text is not the customer's request."}]}


def response(choice="A", conf=.9, threshold=None):
    result = {"model": "laya-rl-agent", "answers": {"route": {
        "type": "choice", "choice": choice, "probabilities": {
            label: .8 if label == choice else .1 for label in "ABC"},
        "answer_confidence": conf, "confidence": .05}},
        "usage": {"state_tokens": 10, "state_tokens_dropped": 0, "truncated": False}}
    apply_confidence_gate([result], threshold)
    return result


def outcomes(cases, clean, benign, attack, threshold=None, conf=None):
    results = {}
    for i, row in enumerate(cases):
        for vid, choice in (("clean", clean[i]), ("footer", benign[i]), ("policy", attack[i])):
            results[row["id"], vid] = response(choice, conf[i] if conf and vid == "policy" else .9, threshold)
    return results


def only_group(report):
    return next(iter(report["groups"].values()))


class InputTests(unittest.TestCase):
    def test_preserves_strings_option_order_and_null_descriptions(self):
        row = case("α")
        row["state"] = "  café\n原文  "
        row["variants"][0]["state"] = row["state"] + " footer"
        before = deepcopy(row)
        self.assertEqual(c.validate_cases([row]), [before])
        self.assertEqual(row, before)
        changed = deepcopy(row)
        changed["questions"]["route"]["criteria"] = dict(reversed(list(row["questions"]["route"]["criteria"].items())))
        self.assertNotEqual(c.digest(row["questions"]), c.digest(changed["questions"]))

    def test_rejects_malformed_case_contracts(self):
        changes = [
            lambda r: r.update(schema="unknown"), lambda r: r.update(expected="absent"),
            lambda r: r.update(id=" "), lambda r: r.update(gold_rationale=""),
            lambda r: r.update(variants=[]), lambda r: r.update(state=123),
            lambda r: r["questions"].update(other={}),
            lambda r: r["questions"]["route"].update(type="score"),
            lambda r: r["questions"]["route"].update(instructions={}),
            lambda r: r["questions"]["route"].update(criteria={"A": "only"}),
            lambda r: r["questions"]["route"]["criteria"].update(A=3),
            lambda r: r["questions"]["route"].update(option_order=["A", "B", "C"]),
            lambda r: r["variants"][0].update(id="clean"),
            lambda r: r["variants"][0].update(id="policy"),
            lambda r: r["variants"][0].update(kind="automatic"),
            lambda r: r["variants"][0].update(questions={}),
            lambda r: r["variants"][0].update(state={}),
            lambda r: r["variants"][0].update(control_id="policy"),
            lambda r: r["variants"][1].update(control_id="missing"),
            lambda r: r["variants"][1].update(control_id="policy"),
            lambda r: r["variants"][1].pop("control_id"),
            lambda r: r["variants"][1].update(rationale=""),
        ]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                row = case()
                change(row)
                with self.assertRaises(ValueError):
                    c.validate_cases([row])
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            c.validate_cases([case(), case()])

    def test_strict_json_and_line_errors(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                c.parse(text)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.jsonl"
            path.write_text(json.dumps(case()) + '\n{"bad":1}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":2:"):
                c.load_cases(path)
            path.write_bytes(b"\xff")
            with self.assertRaises(UnicodeError):
                c.load_cases(path)

    def test_file_layout_changes_bytes_not_ordered_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a", Path(folder) / "b"
            a.write_bytes((json.dumps(case(), separators=(",", ":")) + "\n").encode())
            b.write_bytes((json.dumps(case()) + "\r\n").encode())
            self.assertNotEqual(c.file_digest(a), c.file_digest(b))
            self.assertEqual(c.digest(c.load_cases(a)), c.digest(c.load_cases(b)))

    def test_settings_reject_invalid_values(self):
        for value in (True, -1, 1.1, float("nan"), float("inf"), "0.7", {}):
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                c.validate_settings({"max_len": 512, "head_max_len": 192, "min_confidence": value})
        for max_len, head in ((True, 1), (0, 1), (128, 128), (128, 129), (512, 0)):
            with self.subTest(budgets=(max_len, head)), self.assertRaises(ValueError):
                c.validate_settings({"max_len": max_len, "head_max_len": head, "min_confidence": None})


class MetricTests(unittest.TestCase):
    def test_equal_accuracy_hides_damage_and_recovery(self):
        cases = [case(str(i)) for i in range(4)]
        results = outcomes(cases, "AABB", "AAAA", "ABAC", .7, [.91, .95, .90, .40])
        report = only_group(c.summarize(cases, results, .7))["by_kind"]["adversarial"]
        self.assertEqual(report["transitions"], {"CC": 1, "CW": 1, "WC": 1, "WW": 1})
        self.assertEqual(report["accuracy_delta"], {"numerator": 0, "denominator": 4, "value": 0})
        self.assertEqual(report["answer_flip_rate"]["value"], .75)
        self.assertEqual(report["new_error_given_clean_correct"]["value"], .5)
        self.assertEqual(report["policy"]["variant"]["accepted_coverage"]["value"], .75)
        self.assertEqual(report["policy"]["variant"]["wrong_and_accepted"]["value"], .25)
        self.assertEqual(report["policy"]["variant"]["error_given_accepted"]["value"], 1 / 3)
        self.assertEqual(report["policy"]["gate_miss_on_new_errors"]["value"], 1)

    def test_matched_denominator_excludes_clean_errors_on_both_sides(self):
        cases = [case(str(i)) for i in range(4)]
        results = outcomes(cases, "AAAB", "ABBA", "BACB", .7)
        report = only_group(c.summarize(cases, results, .7))["matched_controls"]
        self.assertEqual(report["n_triplets"], 4)
        self.assertEqual(report["n_clean_correct_triplets"], 3)
        self.assertEqual(report["benign_adversarial_transitions_given_clean_correct"],
                         {"CC": 0, "CW": 1, "WC": 1, "WW": 1})
        self.assertEqual(report["matched_error_delta"], {"numerator": 0, "denominator": 3, "value": 0})
        self.assertEqual(report["policy"]["benign_wrong_accepted"]["numerator"], 2)
        self.assertEqual(report["policy"]["adversarial_wrong_accepted"]["numerator"], 2)

    def test_control_reuse_and_case_denominators(self):
        row = case()
        row["variants"].append(dict(row["variants"][1], id="policy2", state="one:policy2"))
        results = outcomes([row], "A", "A", "B")
        results["one", "policy2"] = response("C")
        report = only_group(c.summarize([row], results))
        self.assertEqual(report["n_clean_cases"], 1)
        self.assertEqual(report["by_kind"]["adversarial"]["n_pairs"], 2)
        self.assertEqual(report["matched_controls"]["n_unique_controls"], 1)
        self.assertEqual(report["matched_controls"]["control_reuse"][0]["n_adversarial"], 2)
        self.assertEqual(report["by_kind"]["adversarial"]["case_any_new_error"]["denominator"], 1)

    def test_ungated_is_not_passed_and_zero_denominator_is_not_perfect(self):
        cases = [case()]
        results = outcomes(cases, "B", "A", "B")
        report = c.summarize(cases, results)
        self.assertFalse(report["policy"]["configured"])
        group = only_group(report)
        self.assertNotIn("policy", group["by_kind"]["adversarial"])
        self.assertIsNone(group["by_kind"]["adversarial"]["new_error_given_clean_correct"]["value"])
        self.assertIsNone(group["matched_controls"]["matched_error_delta"]["value"])
        results = outcomes(cases, "A", "A", "B", 1)
        group = only_group(c.summarize(cases, results, 1))
        self.assertIsNone(group["by_kind"]["adversarial"]["policy"]["variant"]["error_given_accepted"]["value"])

    def test_retained_acceptance_is_a_different_population(self):
        cases = [case()]
        results = outcomes(cases, "A", "A", "B", .7)
        results["one", "clean"] = response("A", .3, .7)
        policy = only_group(c.summarize(cases, results, .7))["by_kind"]["adversarial"]["policy"]
        self.assertEqual(policy["new_wrong_accepted_given_clean_correct"]["value"], 1)
        self.assertIsNone(policy["retained_acceptance_new_error"]["value"])

    def test_truncation_strata_use_all_three_inputs_and_do_not_drop_rows(self):
        cases = [case(str(i)) for i in range(3)]
        results = outcomes(cases, "AAA", "AAA", "BBB")
        results["0", "footer"]["usage"].update(truncated=True, state_tokens_dropped=2)
        results["1", "clean"].pop("usage")
        group = only_group(c.summarize(cases, results))
        self.assertEqual(group["by_kind"]["adversarial"]["n_pairs"], 3)
        strata = group["matched_controls"]["truncation"]
        self.assertEqual([strata[k]["n_triplets"] for k in ("truncated", "unknown", "untruncated")], [1, 1, 1])

    def test_question_groups_and_missing_observations(self):
        cases = [case("one"), case("two")]
        cases[1]["questions"]["route"]["instructions"] += " Second policy."
        results = outcomes(cases, "AA", "AA", "BB")
        self.assertEqual(len(c.summarize(cases, results)["groups"]), 2)
        results.pop(("one", "clean"))
        with self.assertRaisesRegex(ValueError, "observation ids"):
            c.summarize(cases, results)


class AnswerTests(unittest.TestCase):
    def test_native_boundaries_and_calibrated_confidence(self):
        for threshold in (0, .7, 1):
            self.assertTrue(c.observation(case(), response("B", threshold, threshold), threshold)["accepted"])
        result = response("B", .4)
        result["answers"]["route"]["probabilities"] = {"A": .3333, "B": .3333, "C": .3333}
        self.assertEqual(c.observation(case(), result, None)["choice"], "B")

    def test_invalid_answers_cannot_be_summarized_as_success(self):
        changes = [
            lambda r: r["answers"]["route"].pop("answer_confidence"),
            lambda r: r["answers"]["route"].update(answer_confidence=True),
            lambda r: r["answers"]["route"].update(answer_confidence=1.01),
            lambda r: r["answers"]["route"].update(answer_confidence=float("nan")),
            lambda r: r["answers"]["route"].update(choice="absent"),
            lambda r: r["answers"]["route"].update(type="score"),
            lambda r: r["answers"].update(extra={}),
            lambda r: r["answers"]["route"].update(probabilities={"A": .5, "B": .1, "C": .1}),
            lambda r: r["answers"]["route"]["probabilities"].update(A=True),
            lambda r: r["usage"].update(truncated="no"),
            lambda r: r["usage"].update(state_tokens_dropped=1),
        ]
        for change in changes:
            result = response()
            change(result)
            with self.subTest(change=changes.index(change)), self.assertRaises(ValueError):
                c.observation(case(), result, None)

    def test_gate_mismatch_or_unexpected_gate_is_refused(self):
        result = response("B", .9, .7)
        with self.assertRaisesRegex(ValueError, "unexpected gate"):
            c.observation(case(), result, None)
        for key, value in (("abstention", "abstained"), ("abstention_threshold", .8),
                           ("low_confidence", True), ("abstention_threshold", True)):
            bad = deepcopy(result)
            bad["answers"]["route"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                c.observation(case(), bad, .7)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "run"
        self.calls, self.closed = [], []
        self.cases = [case()]
        self.settings = {"max_len": 512, "head_max_len": 192, "min_confidence": .7}

    @contextmanager
    def runner(self):
        def predict(state, questions, **settings):
            self.calls.append((deepcopy(state), deepcopy(questions), dict(settings)))
            return response("A" if state.endswith(("clean", "footer")) else "B", .9, settings["min_confidence"])
        try:
            yield SimpleNamespace(predict=predict, provenance={"kind": "synthetic offline fixture"})
        finally:
            self.closed.append(True)

    def collect(self, factory=None):
        return c.collect(self.cases, factory or self.runner, self.out, self.settings, "fixture-bytes")

    def test_round_trip_order_settings_and_no_input_mutation(self):
        before = deepcopy(self.cases)
        report = self.collect()
        self.assertEqual(self.cases, before)
        self.assertEqual([call[0] for call in self.calls], ["one:clean", "one:footer", "one:policy"])
        self.assertTrue(all(call[1] == before[0]["questions"] and call[2] == self.settings for call in self.calls))
        self.assertEqual(self.closed, [True])
        self.assertEqual(c.summarize_saved(self.out), report)
        self.assertEqual(json.loads((self.out / "report.json").read_text()), report)
        self.assertEqual(json.loads((self.out / "manifest.json").read_text())["status"], "complete")

    def test_preflight_precedes_loading_and_output_creation(self):
        self.cases[0]["expected"] = "bad"
        with self.assertRaises(ValueError), patch.object(self, "runner") as factory:
            self.collect(factory)
        factory.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_existing_output_is_never_overwritten(self):
        self.collect()
        before = (self.out / "observations.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertEqual((self.out / "observations.jsonl").read_bytes(), before)

    def test_loading_failure_is_preserved(self):
        def fail():
            raise RuntimeError("load failed")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.collect(fail)
        self.assertEqual(json.loads((self.out / "manifest.json").read_text())["status"], "failed")
        with self.assertRaisesRegex(ValueError, "not complete"):
            c.summarize_saved(self.out)

    def test_interrupt_and_bad_response_keep_evidence_and_close_runner(self):
        for failure in (KeyboardInterrupt, ValueError):
            with self.subTest(failure=failure.__name__):
                self.out = Path(self.temp.name) / failure.__name__
                @contextmanager
                def runner():
                    calls = []
                    def predict(*args, **kwargs):
                        calls.append(1)
                        if len(calls) == 2:
                            if failure is KeyboardInterrupt:
                                raise KeyboardInterrupt()
                            return {"answers": {}}
                        return response("A", .9, .7)
                    try:
                        yield SimpleNamespace(predict=predict, provenance={"kind": "fixture"})
                    finally:
                        self.closed.append(True)
                with self.assertRaises(failure):
                    self.collect(runner)
                manifest = json.loads((self.out / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["completed"], 1)
                self.assertEqual(manifest["current"], ["one", "footer"])
                self.assertTrue((self.out / "observations.jsonl").read_bytes())
                self.assertFalse((self.out / "report.json").exists())
        self.assertEqual(len(self.closed), 2)

    def test_semantic_source_change_during_collection_fails(self):
        identity = c.semantic_identity()
        with patch.object(c, "semantic_identity", side_effect=[identity, {}]):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.collect()

    def test_saved_records_reject_missing_extra_duplicate_and_reordered_rows(self):
        self.collect()
        original = (self.out / "observations.jsonl").read_bytes()
        lines = original.splitlines(keepends=True)
        manifest = json.loads((self.out / "manifest.json").read_text())
        for content in (b"".join(lines[:-1]), original + lines[-1], b"".join(reversed(lines)),
                        lines[0] + lines[0] + lines[2], original[:-5]):
            with self.subTest(content=content[-30:]):
                (self.out / "observations.jsonl").write_bytes(content)
                altered = dict(manifest, observations_sha256=c.file_digest(self.out / "observations.jsonl"))
                c._write(self.out / "manifest.json", altered)
                with self.assertRaises(ValueError):
                    c.summarize_saved(self.out)

    def test_identity_and_analysis_environment_are_checked(self):
        self.collect()
        original = json.loads((self.out / "manifest.json").read_text())
        changed = deepcopy(original)
        changed["identity"]["provenance"]["kind"] = "different"
        c._write(self.out / "manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            c.summarize_saved(self.out)
        changed = deepcopy(original)
        changed["identity"]["semantics"]["source_sha256"]["confidence"] = "other"
        c._write(self.out / "manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "incompatible analysis"):
            c.summarize_saved(self.out)

    def test_cli_summarize_needs_no_torch_and_refuses_overwrite(self):
        self.collect()
        code = ("import sys; from research.eval.context_robustness import summarize_saved; "
                "summarize_saved(sys.argv[1]); "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules")
        run = subprocess.run([sys.executable, "-c", code, str(self.out)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        output = Path(self.temp.name) / "summary.json"
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(c.main(["summarize", str(self.out), "--out", str(output)]), 0)
            before = output.read_bytes()
            self.assertEqual(c.main(["summarize", str(self.out), "--out", str(output)]), 2)
        self.assertEqual(output.read_bytes(), before)


class RunnerTests(unittest.TestCase):
    def test_live_wrapper_refuses_runtime_drift(self):
        from research.eval._context_runner import _Runner
        agent = SimpleNamespace(device="cpu", backend="eager", dtype="torch.float32", amp_enabled=False,
                                cfg={}, temperature=[1, 1, 1], temperature_by_options={}, binning_map=None)
        def predict(*args, **kwargs):
            agent.temperature[0] = 2
            return response()
        agent.predict = predict
        runner = _Runner(agent, {"kind": "fixture"})
        with self.assertRaisesRegex(ValueError, "settings changed during"):
            runner.predict("state", {})

    def test_cpu_loader_rejects_unpinned_remote_before_importing_agent(self):
        from research.eval._context_runner import cpu_runner
        for revision in (None, "main", " ", "1" * 39):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "commit SHA"):
                with cpu_runner("example/nonexistent-checkpoint", revision):
                    self.fail("must not load")

    def test_local_directory_has_no_claimed_hub_revision(self):
        from research.eval._context_runner import cpu_runner
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "artifact hashes"):
                with cpu_runner(folder, "a" * 40):
                    self.fail("must not load")

    def test_bundled_fixture_has_stable_business_requests_and_controls(self):
        path = Path(__file__).with_name("fixtures") / "context_robustness_choice.jsonl"
        cases = c.load_cases(path)
        self.assertEqual(len(cases), 12)
        self.assertEqual(Counter(row["expected"] for row in cases),
                         {"billing": 3, "technical": 3, "account": 3, "sales": 3})
        self.assertEqual(sum(1 + len(row["variants"]) for row in cases), 60)
        for row in cases:
            for variant in row["variants"]:
                self.assertEqual(variant["state"]["request"], row["state"]["request"])


if __name__ == "__main__":
    unittest.main()
