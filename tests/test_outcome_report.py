"""Outcome report: static HTML over a decision log, built by a script, checked here.

Weight-free throughout: the database is a tmp SQLite file filled through the real
`OutcomeLogHook`, and the script under test runs in a subprocess exactly as an
operator runs it, so the report is verified as produced rather than as assembled.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "outcome_report.py")

sys.path.insert(0, ROOT)  # noqa: E402

from laya import OutcomeLogHook  # noqa: E402
from laya.hooks import PredictContext  # noqa: E402

QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": None, "tech": None}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["calm", "firm", "angry"]},
}


def _ctx(state="billed twice", run_id="r1", questions=None, error=None):
    answers = {"dept": {"type": "choice", "choice": "billing",
                        "probabilities": {"billing": 0.94, "tech": 0.06},
                        "confidence": 0.94},
               "urgency": {"type": "score", "score": 1.0,
                           "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
                           "confidence": 0.5, "legend": {"0": "low", "1": "high"}}}
    return PredictContext(states=[state], questions=dict(QUESTIONS if questions is None
                                                         else questions),
                          results=None if error is not None else [{"model": "m",
                                                                  "answers": answers,
                                                                  "usage": {}}],
                          decision={"model": "english"}, model="english",
                          elapsed_ms=1.0, error=error, run_id=run_id)


def _logged_db(path, states=("billed twice", "refund please")):
    hook = OutcomeLogHook(path)
    try:
        for i, state in enumerate(states):
            hook.on_predict_end(_ctx(state=state, run_id="r%d" % i))
        hook.record_outcome("r0", "dept", "billing", source="review")
        return hook
    except BaseException:
        hook.close()
        raise


def _run(*args):
    proc = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True,
                          cwd=ROOT)
    return proc


def test_report_renders_volume_and_coverage(tmp_path):
    db, out = str(tmp_path / "d.db"), str(tmp_path / "r.html")
    _logged_db(db).close()
    proc = _run("--db", db, "--out", out)
    assert proc.returncode == 0, proc.stderr
    assert "wrote %s" % out in proc.stdout
    page = open(out, encoding="utf-8").read()
    assert "4 logged answers, 1 labels" in page
    assert "dept" in page and "urgency" in page
    assert "review" in page                       # the label's provenance is shown
    assert "<script" not in page and "<svg" in page  # static, no JavaScript


def test_report_never_shows_raw_state_text(tmp_path):
    db, out = str(tmp_path / "d.db"), str(tmp_path / "r.html")
    hook = OutcomeLogHook(db)
    try:
        hook.on_predict_end(_ctx(state="<script>alert(1)</script>", run_id="r9"))
    finally:
        hook.close()
    assert _run("--db", db, "--out", out).returncode == 0
    page = open(out, encoding="utf-8").read()
    assert "alert(1)" not in page  # hashing, not redacting: the text is never stored


def test_report_lists_the_review_queue_and_payload(tmp_path):
    db, out = str(tmp_path / "d.db"), str(tmp_path / "r.html")
    hook = OutcomeLogHook(db)
    try:
        hook.on_predict_end(_ctx(run_id="r1"))
        answers = {"dept": {"type": "noul", "noul": 0.4, "confidence": 0.2,
                            "abstention": "abstained", "abstention_threshold": 0.9}}
        gated = _ctx(run_id="r2")
        gated.results = [{"model": "m", "answers": answers, "usage": {}}]
        gated.questions = {"dept": {"type": "noul", "instructions": "?"}}
        hook.on_predict_end(gated)
        payload = {"version": 2, "temperature": [0.9473, 1.0, 1.0],
                   "temperature_by_options": {"choice:2": 0.9},
                   "model_id_or_path": "x", "subfolder": None, "config": None}
        payload_path = str(tmp_path / "cal.json")
        json.dump(payload, open(payload_path, "w"))
    finally:
        hook.close()
    assert _run("--db", db, "--out", out, "--payload", payload_path).returncode == 0
    page = open(out, encoding="utf-8").read()
    assert "abstained" in page and "0.9473" in page and "Version: 2" in page
    assert "choice:2" in page and "r2" in page


def test_report_refuses_gracefully(tmp_path):
    missing = str(tmp_path / "absent" / "d.db")
    proc = _run("--db", missing, "--out", str(tmp_path / "r.html"))
    assert proc.returncode == 1
    assert "cannot read" in proc.stderr
    assert not os.path.exists(str(tmp_path / "r.html"))
