"""Decision outcome log: the raw material for offline per-question calibration.

An `OutcomeLogHook` installed on a Router or Agent records every answered call to a local
SQLite file -- one row per question -- so a later job can join outcomes, fit per-question
temperatures and thresholds with `laya.calibrate`, and ship the result as a calibration
payload. The serving path is untouched: the hook only observes `on_predict_end`, and it
never raises, so a full disk or a closed database costs a warning, never a decision.

Privacy is structural, not promised: raw state text is never stored unless the deployment
opts in with `include_state=True`. What is always stored is a SHA-256 over the canonical
JSON of the state, which is enough to join replays and to deduplicate, and nothing that
reads the state back out.
"""

import hashlib
import json
import math
import sqlite3
import threading
import time
import warnings

from .hooks import BaseHook

#: Schema written by `_init_schema`. Bumped only by a migration this module ships.
SCHEMA_VERSION = 1


def _canonical_state_hash(state: object) -> str:
    """SHA-256 over the canonical JSON of `state`, or over its `str` when it has no JSON form."""
    try:
        text = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(state)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _answer_summary(answer: object) -> tuple:
    """`(value_text, value_real, confidence)` for one answer, `None` where it has none.

    `choice` carries text, `score` and `noul` carry numbers; every type may carry a
    `confidence`. Anything else -- an unknown shape, a non-dict -- summarizes to nothing
    rather than raising, because the hook must not fail a request a shape it never heard
    of.
    """
    if not isinstance(answer, dict):
        return None, None, None
    kind = answer.get("type")
    value_text = value_real = None
    if kind == "choice":
        value_text = answer.get("choice")
        if not isinstance(value_text, str):
            value_text = None
    elif kind == "score":
        value_real = answer.get("score")
    elif kind == "noul":
        value_real = answer.get("noul")
    if isinstance(value_real, bool) or not isinstance(value_real, (int, float)):
        value_real = None
    confidence = answer.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = None
    return value_text, value_real, confidence


class OutcomeLogHook(BaseHook):
    """Record every decision to SQLite on `on_predict_end`.

    One row per answered question in `decisions`, plus an `outcomes` table `record_outcome`
    fills later with labels. Install it where the deployment already installs hooks --
    `Router(hooks=[...])`, `Agent(hooks=[...])`, or `set_default_hooks` -- and it starts
    logging; remove it and logging stops. There is no other switch.
    """

    def __init__(self, path: str, *, include_state: bool = False) -> None:
        """Open (creating) the SQLite file at `path`; `:memory:` stays process-local.

        `include_state` stores the canonical JSON of each state beside its hash. Off by
        default: states routinely hold user content, and the hash is all the join and
        replay bookkeeping needs.
        """
        self.path = path
        self.include_state = include_state
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._init_schema()

    def _init_schema(self) -> None:
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS decisions ("
            "run_id TEXT NOT NULL, ts REAL NOT NULL, model TEXT, "
            "qid TEXT, qtype TEXT, state_hash TEXT, state_json TEXT, "
            "answer_json TEXT, value_text TEXT, value_real REAL, "
            "confidence REAL, abstention TEXT, elapsed_ms REAL, error TEXT)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions (run_id)")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS outcomes ("
            "run_id TEXT NOT NULL, qid TEXT NOT NULL, label_json TEXT NOT NULL, "
            "source TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0, ts REAL NOT NULL, "
            "PRIMARY KEY (run_id, qid))"
        )
        self._db.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)
        self._db.commit()

    def close(self) -> None:
        """Release the database handle; later calls log a warning instead of failing."""
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def __enter__(self) -> "OutcomeLogHook":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def on_predict_end(self, ctx: object) -> None:
        """Append one row per answered question. Never raises: a logging failure warns."""
        try:
            self._log(getattr(ctx, "run_id", None), ctx)
        except Exception as exc:  # noqa: BLE001 -- telemetry must not fail a request
            warnings.warn("laya: OutcomeLogHook failed: %s" % exc, RuntimeWarning, stacklevel=2)

    def _log(self, run_id: object, ctx: object) -> None:
        states = getattr(ctx, "states", None) or []
        results = getattr(ctx, "results", None)
        questions = getattr(ctx, "questions", None) or {}
        model = getattr(ctx, "model", None)
        decision = getattr(ctx, "decision", None) or {}
        model = model or decision.get("model")
        elapsed_ms = getattr(ctx, "elapsed_ms", None)
        error = getattr(ctx, "error", None)
        rows = []
        ts = time.time()
        if not isinstance(results, list) or error is not None:
            # A failed or short-circuited call answers nothing: one row names the failure
            # so error analysis can count it, instead of the call vanishing from the log.
            rows.append((run_id, ts, model, None, None, None, None, None, None, None, None,
                         None, elapsed_ms, str(error) if error is not None else None))
        else:
            for state, result in zip(states, results):
                if not isinstance(result, dict):
                    continue
                answers = result.get("answers") or {}
                state_hash = _canonical_state_hash(state)
                state_json = None
                if self.include_state:
                    try:
                        state_json = json.dumps(state, sort_keys=True, ensure_ascii=False,
                                                default=str)
                    except (TypeError, ValueError):
                        state_json = str(state)
                for qid, qdef in questions.items():
                    answer = answers.get(qid) if isinstance(answers, dict) else None
                    value_text, value_real, confidence = _answer_summary(answer)
                    qtype = qdef.get("type") if isinstance(qdef, dict) else None
                    rows.append((run_id, ts, model, qid, qtype, state_hash, state_json,
                                 json.dumps(answer, sort_keys=True, ensure_ascii=False,
                                            default=str) if answer is not None else None,
                                 value_text, value_real, confidence,
                                 answer.get("abstention") if isinstance(answer, dict) else None,
                                 elapsed_ms, None))
        with self._lock:
            if self._db is None:
                raise RuntimeError("OutcomeLogHook is closed")
            self._db.executemany(
                "INSERT INTO decisions (run_id, ts, model, qid, qtype, state_hash, state_json,"
                "answer_json, value_text, value_real, confidence, abstention, elapsed_ms, error)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._db.commit()

    def record_outcome(self, run_id: str, qid: str, label: object, source: str,
                         weight: float = 1.0) -> None:
        """Store one label for a logged decision; a second call for the same cell replaces it.

        `label` is whatever the source asserts -- a choice string, a score level, a
        `{"correct": ...}` judgement -- kept as JSON so weak signals and human review share
        the table. `source` names where it came from (`stability`, `review`, `shadow-eval`)
        so the fitter can weight or filter by provenance. `weight` scales the label in the
        fit (a stability score for weak labels, 1.0 for human ones); it must be finite.
        """
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                or not math.isfinite(weight):
            raise ValueError("weight must be a finite number, got %r" % (weight,))
        with self._lock:
            if self._db is None:
                raise RuntimeError("OutcomeLogHook is closed")
            self._db.execute(
                "INSERT OR REPLACE INTO outcomes (run_id, qid, label_json, source, weight, ts)"
                " VALUES (?,?,?,?,?,?)",
                (run_id, qid, json.dumps(label, sort_keys=True, ensure_ascii=False,
                                         default=str), source, float(weight), time.time()))
            self._db.commit()

    def decisions(self) -> list:
        """Every logged decision row, oldest first, as plain dicts (for export and tests)."""
        with self._lock:
            return [dict(row) for row in
                    self._db.execute("SELECT * FROM decisions ORDER BY ts, rowid")]

    def outcomes(self) -> list:
        """Every recorded outcome, as plain dicts."""
        with self._lock:
            return [dict(row) for row in
                    self._db.execute("SELECT * FROM outcomes ORDER BY ts, run_id, qid")]
