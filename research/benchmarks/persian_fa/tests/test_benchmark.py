"""Model-free tests: no torch, no downloads, no network.

    python -m unittest discover -s tests -v
"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audit  # noqa: E402
import run  # noqa: E402
from common import FAMILIES, load_cases  # noqa: E402
from metrics import expected_calibration_error, score  # noqa: E402
from normalize import normalize  # noqa: E402
from prompts import LABELS, MODES  # noqa: E402


def quiet(fn, *args):
    with redirect_stdout(io.StringIO()):
        return fn(*args)


class DatasetTests(unittest.TestCase):
    def test_dataset_and_manifest_pass_audit(self):
        self.assertEqual(quiet(audit.main, []), 0)

    def test_balanced_grid(self):
        cases = load_cases()
        self.assertEqual(len(cases), 64)
        for fam in FAMILIES:
            for lab in LABELS:
                n = sum(1 for c in cases if c["family"] == fam and c["label"] == lab)
                self.assertEqual(n, 2, f"{fam}/{lab}")

    def test_prompt_modes_share_keys(self):
        keys = [tuple(m["department"]["criteria"]) for m in MODES.values()]
        self.assertTrue(all(k == LABELS for k in keys))

    def test_model_never_sees_label(self):
        from prompts import build_state
        case = load_cases()[0]
        self.assertEqual(build_state(case["text"]), {"message": case["text"]})


class NormalizeTests(unittest.TestCase):
    def test_arabic_letters_and_digits(self):
        self.assertEqual(normalize("كسر ۱۲۹ ي"), "کسر 129 ی")

    def test_zwnj(self):
        self.assertEqual(normalize("میخوام"), "می‌خوام")
        self.assertEqual(normalize("هزینه ی تمدید"), "هزینه‌ی تمدید")

    def test_finglish_untouched(self):
        self.assertEqual(normalize("mikham laghv konam"), "mikham laghv konam")


class MetricTests(unittest.TestCase):
    CASES = [
        {"id": "a", "family": "formal", "label": "billing", "trap": None},
        {"id": "b", "family": "formal", "label": "cancel", "trap": None},
        {"id": "c", "family": "negation", "label": "other", "trap": "negated_cancel"},
        {"id": "d", "family": "negation", "label": "technical", "trap": None},
    ]

    def row(self, cid, pred, conf=0.9, error=None):
        return {"case_id": cid, "mode": "m", "repeat": 0, "prediction": pred,
                "answer_confidence": conf, "latency_ms": 10.0, "error": error}

    def test_counts_false_cancel_and_failures(self):
        rows = [self.row("a", "billing"), self.row("b", "cancel"),
                self.row("c", "cancel"), self.row("d", None, error="RuntimeError")]
        s = score(self.CASES, rows, "m")
        self.assertEqual(s["correct"], 2)
        self.assertEqual(s["failures"], 1)
        self.assertEqual(s["false_cancel"], 1)
        self.assertEqual(s["by_trap"]["negated_cancel"], {"correct": 0, "n": 1})
        self.assertEqual(s["confusion"]["technical"]["FAILED"], 1)

    def test_missing_row_counts_as_wrong(self):
        s = score(self.CASES, [self.row("a", "billing")], "m")
        self.assertEqual((s["correct"], s["failures"]), (1, 3))

    def test_ece(self):
        self.assertAlmostEqual(expected_calibration_error([1.0, 1.0], [True, True]), 0.0)
        self.assertAlmostEqual(expected_calibration_error([0.9, 0.9], [False, False]), 0.9)


class PipelineTests(unittest.TestCase):
    def run_keyword(self, out):
        quiet(run.main, ["--backend", "keyword", "--repeats", "2", "--output", str(out)])

    def test_keyword_run_passes_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "kw"
            self.run_keyword(out)
            self.assertEqual(quiet(audit.main, ["--run-dir", str(out), "--write"]), 0)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["modes"]["choice_fa"]["n"], 64)

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "kw"
            self.run_keyword(out)
            with self.assertRaises(SystemExit):
                self.run_keyword(out)

    def test_audit_catches_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "kw"
            self.run_keyword(out)
            path = out / "responses.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()

            bad = json.loads(lines[0])
            bad["prediction"] = "sales"  # not a label
            path.write_text("\n".join([json.dumps(bad)] + lines[1:]) + "\n", encoding="utf-8")
            self.assertEqual(quiet(audit.main, ["--run-dir", str(out)]), 1)

            path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")  # one row missing
            self.assertEqual(quiet(audit.main, ["--run-dir", str(out)]), 1)

    def test_audit_catches_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "kw"
            self.run_keyword(out)
            meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
            meta["sha256"]["prompts.py"] = "0" * 64
            (out / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            self.assertEqual(quiet(audit.main, ["--run-dir", str(out)]), 1)


class LayaBackendWiringTest(unittest.TestCase):
    """Exercises the laya code path with a stand-in module shaped like laya's real API
    (laya.load(...) -> agent.predict(state, questions) -> {"answers": {...}})."""

    def test_laya_backend_with_stub(self):
        import types
        calls = {}

        class Agent:
            device = "cpu"

            def predict(self, state, questions):
                keys = list(questions["department"]["criteria"])
                return {"answers": {"department": {
                    "type": "choice", "choice": keys[0],
                    "probabilities": {k: (0.7 if i == 0 else 0.1) for i, k in enumerate(keys)},
                    "answer_confidence": 0.7}}}

        def load(source, device=None, subfolder=None, revision=None):
            calls.update(source=source, subfolder=subfolder, revision=revision)
            return Agent()

        stub = types.ModuleType("laya")
        stub.load, stub.__version__ = load, "stub"
        sys.modules["laya"] = stub
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "stub"
                quiet(run.main, ["--backend", "laya", "--checkpoint", "multilingual",
                                 "--revision", "a" * 40, "--repeats", "1", "--output", str(out)])
                self.assertEqual(calls, {"source": "convaiinnovations/laya",
                                         "subfolder": "multilingual", "revision": "a" * 40})
                self.assertEqual(quiet(audit.main, ["--run-dir", str(out)]), 0)
        finally:
            del sys.modules["laya"]


if __name__ == "__main__":
    unittest.main()
