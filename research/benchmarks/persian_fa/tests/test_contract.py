"""Exercise artifact failures independently of any model or installed packages."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audit  # noqa: E402
import compare  # noqa: E402
import run  # noqa: E402
from common import load_cases, request_for, sha256_file  # noqa: E402
from normalize import normalize  # noqa: E402


def quiet(fn, *args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args)


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "run"
        self.args = ["--backend", "keyword", "--limit", "4", "--repeats", "2", "--output", str(self.out)]
        self.assertEqual(quiet(run.main, self.args), 0)
        self.meta = json.loads((self.out / "metadata.json").read_text(encoding="utf-8"))
        self.rows = [json.loads(s) for s in (self.out / "responses.jsonl").read_text(encoding="utf-8").splitlines()]

    def write(self, meta, rows):
        (self.out / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        (self.out / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def result(self):
        return quiet(audit.main, ["--run-dir", str(self.out)])

    def test_record_keeps_only_message_in_request_and_complete_response(self):
        row = self.rows[0]
        self.assertEqual(row["request"]["state"], {"message": load_cases()[0]["text"]})
        self.assertEqual(row["response"]["answers"]["department"]["choice"], row["prediction"])
        self.assertTrue(self.meta["smoke_test"])
        self.assertTrue(self.meta["not_a_model_result"])
        self.assertEqual(self.result(), 0)

    def test_metadata_cannot_hide_or_duplicate_measurements(self):
        for field, value in (("modes", []), ("modes", ["choice_fa", "choice_fa"]),
                             ("modes", ["unknown"]), ("modes", [{}]), ("repeats", 0),
                             ("repeats", True), ("repeats", 10**20), ("case_ids", []),
                             ("case_ids", [self.meta["case_ids"][0]] * 4), ("case_ids", [None]),
                             ("n_cases", 64), ("n_cases", True), ("normalize", "false"),
                             ("smoke_test", False), ("not_a_model_result", False),
                             ("errors", 1), ("errors", True), ("completed_requests", 1),
                             ("record_format", 99), ("record_format", None), ("backend", {})):
            with self.subTest(field=field, value=value):
                self.write(dict(self.meta, **{field: value}), self.rows)
                self.assertEqual(self.result(), 1)

    def test_incomplete_duplicate_extra_or_bad_identity_rows_are_rejected(self):
        variants = [self.rows[:-1], self.rows + [self.rows[0]],
                    [self.rows[0], self.rows[0]] + self.rows[2:],
                    [dict(self.rows[0], repeat=True)] + self.rows[1:],
                    [dict(self.rows[0], case_id="unknown")] + self.rows[1:],
                    [dict(self.rows[0], mode="unknown")] + self.rows[1:],
                    [dict(self.rows[0], case_id=[])] + self.rows[1:], [None] + self.rows[1:]]
        for rows in variants:
            with self.subTest(first=rows[0]):
                self.write(self.meta, rows)
                self.assertEqual(self.result(), 1)

    def test_confidence_latency_and_probabilities_are_checked(self):
        for field, values in (("answer_confidence", [None, -0.1, 1.1, float("nan"), float("inf"), True, 0.6]),
                              ("latency_ms", [None, -1, float("nan"), float("inf"), True, 10**400])):
            for value in values:
                with self.subTest(field=field, value=value):
                    self.write(self.meta, [dict(self.rows[0], **{field: value})] + self.rows[1:])
                    self.assertEqual(self.result(), 1)
        for value in (None, "0.7", True, -0.1, 1.1, float("nan"), float("inf"), 0.1):
            with self.subTest(probability=value):
                row = dict(self.rows[0])
                row["probabilities"] = dict(row["probabilities"], **{row["prediction"]: value})
                self.write(self.meta, [row] + self.rows[1:])
                self.assertEqual(self.result(), 1)

    def test_raw_request_and_raw_answer_are_not_optional(self):
        for field, value in (("request", None), ("request", {"state": {"message": "changed"}, "questions": {}}),
                             ("response", None), ("response", {"answers": {}})):
            with self.subTest(field=field, value=value):
                self.write(self.meta, [dict(self.rows[0], **{field: value})] + self.rows[1:])
                self.assertEqual(self.result(), 1)
        row = json.loads(json.dumps(self.rows[0]))
        row["response"]["answers"]["department"]["answer_confidence"] = 0.5
        self.write(self.meta, [row] + self.rows[1:])
        self.assertEqual(self.result(), 1)

    def test_calibrated_confidence_can_differ_from_winning_probability(self):
        # Laya's optional binning calibration changes answer_confidence, not the vector.
        row = json.loads(json.dumps(self.rows[0]))
        row["answer_confidence"] = 0.25
        row["response"]["answers"]["department"]["answer_confidence"] = 0.25
        self.write(self.meta, [row] + self.rows[1:])
        self.assertEqual(self.result(), 0)

    def test_normalization_flag_must_match_the_recorded_request(self):
        case = next(c for c in load_cases() if normalize(c["text"]) != c["text"])
        raw, normalized = request_for(case, "choice_fa"), request_for(case, "choice_fa", True)
        self.assertNotEqual(raw["state"], normalized["state"])
        self.assertEqual(normalized["state"], {"message": normalize(case["text"])})
        self.assertEqual(raw["questions"], normalized["questions"])
        with patch.object(run.KeywordBackend, "predict", wraps=None) as predict:
            # A recording backend checks the warm-up as well as the measured request.
            requests = []
            backend = run.KeywordBackend()
            original = run.KeywordBackend.KEYWORDS
            def answer(state, questions):
                requests.append(state)
                label = next(iter(original))
                return {"answers": {"department": {"type": "choice", "choice": label,
                        "probabilities": {k: 0.7 if k == label else 0.1 for k in run.LABELS},
                        "answer_confidence": 0.7}}}
            predict.side_effect = answer
            out = Path(self.tmp.name) / "norm"
            self.assertEqual(quiet(run.main, ["--backend", backend.name, "--normalize", "--limit", "1",
                                             "--repeats", "1", "--modes", "choice_fa", "--output", str(out)]), 0)
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(quiet(audit.main, ["--run-dir", str(out)]), 0)

    def test_malformed_json_and_non_object_metadata_return_failure(self):
        for value in ([], None, True):
            self.write(value, self.rows)
            self.assertEqual(self.result(), 1)
        self.write(self.meta, self.rows)
        (self.out / "responses.jsonl").write_text("{broken\n", encoding="utf-8")
        self.assertEqual(self.result(), 1)

    def test_recomputed_summary_is_verified(self):
        self.assertEqual(quiet(audit.main, ["--run-dir", str(self.out), "--write"]), 0)
        path = self.out / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["modes"]["choice_fa"]["false_cancel"] = 100
        path.write_text(json.dumps(summary), encoding="utf-8")
        self.assertEqual(self.result(), 1)

    def test_comparison_refuses_keyword_baseline_and_broken_inputs(self):
        with self.assertRaises(SystemExit):
            quiet(compare.main, [str(self.out)])
        self.write(self.meta, self.rows[:-1])
        with self.assertRaises(SystemExit):
            quiet(compare.main, [str(ROOT / "results/v1/multilingual"), str(self.out)])


class FailureAndSourceTests(unittest.TestCase):
    def test_failed_request_stays_in_denominator_and_returns_nonzero(self):
        original = run.KeywordBackend.predict
        calls = 0
        def predict(backend, state, questions):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("private exception details")
            return original(backend, state, questions)
        with tempfile.TemporaryDirectory() as tmp, patch.object(run.KeywordBackend, "predict", predict):
            out = Path(tmp) / "failed"
            self.assertEqual(quiet(run.main, ["--backend", "keyword", "--limit", "4", "--repeats", "1",
                                             "--modes", "choice_fa", "--output", str(out)]), 1)
            self.assertEqual(quiet(audit.main, ["--run-dir", str(out), "--write"]), 0)
            summary = json.loads((out / "summary.json").read_text())["modes"]["choice_fa"]
            self.assertEqual((summary["n"], summary["failures"]), (4, 1))
            self.assertNotIn("private exception details", (out / "responses.jsonl").read_text(encoding="utf-8"))

    def test_interrupted_run_retains_metadata_and_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "interrupted"
            with patch.object(run.KeywordBackend, "predict", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    quiet(run.main, ["--backend", "keyword", "--output", str(out)])
            meta = json.loads((out / "metadata.json").read_text())
            self.assertEqual(meta["completed_requests"], 0)
            self.assertEqual(quiet(audit.main, ["--run-dir", str(out)]), 1)

    def test_invalid_cli_never_starts_a_model(self):
        for opts in (["--limit", "0"], ["--limit", "65"], ["--repeats", "0"],
                     ["--modes", "choice_fa", "choice_fa"], ["--revision", "main"]):
            with self.subTest(opts=opts), patch.object(run, "LayaBackend") as load:
                with self.assertRaises(SystemExit):
                    quiet(run.main, ["--output", "unused", *opts])
                load.assert_not_called()

    def test_source_pin_detects_line_ending_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").write_bytes(b"frozen\n")
            (root / "SOURCE.json").write_text(json.dumps({"files": [
                {"path": "data", "sha256": sha256_file(root / "data")}
            ]}))
            good = audit.Audit()
            audit.check_source(good, root)
            self.assertFalse(good.problems)
            (root / "data").write_bytes(b"frozen\r\n")
            bad = audit.Audit()
            audit.check_source(bad, root)
            self.assertTrue(bad.problems)

    def test_frozen_archives_cannot_be_overwritten(self):
        with self.assertRaises(SystemExit):
            quiet(audit.main, ["--run-dir", str(ROOT / "results/v1/english"), "--write"])
        with self.assertRaises(SystemExit):
            quiet(compare.main, [str(ROOT / "results/v1/english"), "--out",
                                 str(ROOT / "results/new-comparison.md")])


if __name__ == "__main__":
    unittest.main()
