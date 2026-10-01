"""Labelled evaluation harness: score a runner on a dataset and gate a build on it.

Pure Python plus numpy, and free of torch at import time, so the metric math and the dataset
parsing can be unit tested with no weights. `evaluate` only needs a runner with a
``predict(state, questions, model=...)`` method, plus -- for ``batch_size`` above 1 -- a
``predict_batch`` in either of the two shapes the repo ships: the positional one, or the
per-request-dict one `Router.predict_batch` documents. A fixture runner stands in for a
checkpoint in tests.

The report is deterministic for a fixed runner: the same dataset produces the same numbers, and
``EvalReport.compare`` turns a baseline into a pass/fail with the per-metric deltas, which is what
the CI gate consumes.

`compare` reads ``overall`` and nothing else, so a run that says nothing about *what it measured*
makes a gate that can only compare arithmetic. Every report therefore carries a run identity in
``config`` -- ``schema``, ``questions_sha256``, ``laya_version`` and (from the CLI)
``dataset_sha256`` and the ``thresholds`` actually applied -- and
``EvalReport.comparable_to`` refuses a comparison between two runs that are not the same
measurement. The identity is deliberately free of anything time-bearing, so a report stays
byte-reproducible for a fixed runner.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .confidence import check_min_confidence

#: The shape of the report this module writes, so a consumer can refuse one it cannot read.
#: ``research/evals/act_head_eval.py`` publishes a report under its own tag off this prefix.
REPORT_SCHEMA = "laya-evals-report/1"

#: The keys that decide whether two reports are the same measurement. A key missing
#: on either side is unknown rather than a conflict, so a baseline committed before these existed
#: keeps comparing exactly as it did. ``schema`` is also read at the top level, which is where
#: ``research/evals/act_head_eval.py`` puts its own tag.
_IDENTITY_KEYS = ("schema", "dataset_sha256", "questions_sha256")

_IDENTITY_LABELS = {
    "schema": "report schema",
    "dataset_sha256": "dataset bytes",
    "questions_sha256": "question schema",
}


def _identity_of(document: Any) -> Dict[str, Any]:
    """Pull the identity keys out of a report, from ``config`` or from the top level."""
    if not isinstance(document, dict):
        return {}
    config = document.get("config")
    found = dict(config) if isinstance(config, dict) else {}
    for key in _IDENTITY_KEYS:
        if found.get(key) is None and document.get(key) is not None:
            found[key] = document[key]
    return found


class EvalError(ValueError):
    """A dataset or report is malformed; the message names the row or field."""


@dataclass
class Example:
    """One labelled decision: a state, its questions, and the expected answer per question id."""

    state: Any
    questions: Dict[str, Any]
    expected: Dict[str, Any]
    tags: Tuple[str, ...] = ()
    language: Optional[str] = None
    model: Optional[str] = None    # explicit checkpoint, forwarded to the runner

    @classmethod
    def from_dict(cls, row: Any, where: str = "row") -> "Example":
        if not isinstance(row, dict):
            raise EvalError("%s must be an object, got %s" % (where, type(row).__name__))
        for key in ("state", "questions", "expected"):
            if key not in row:
                raise EvalError("%s is missing %r" % (where, key))
        questions, expected = row["questions"], row["expected"]
        if not isinstance(questions, dict):
            raise EvalError("%s 'questions' must be an object" % where)
        if not isinstance(expected, dict):
            raise EvalError("%s 'expected' must be an object keyed by question id" % where)
        unknown = sorted(set(expected) - set(questions))
        if unknown:
            raise EvalError("%s 'expected' names unknown question(s): %s" % (where, ", ".join(unknown)))
        tags = row.get("tags") or ()
        if not isinstance(tags, (list, tuple)):
            raise EvalError("%s 'tags' must be a list" % where)
        return cls(state=row["state"], questions=questions, expected=expected,
                   tags=tuple(str(t) for t in tags), language=row.get("language"),
                   model=row.get("model"))


class Dataset:
    """An ordered collection of labelled examples."""

    def __init__(self, examples: Iterable[Example]):
        self.examples: List[Example] = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def from_jsonl(cls, path: str) -> "Dataset":
        examples: List[Example] = []
        with open(path, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise EvalError("%s:%d is not valid JSON: %s" % (path, lineno, exc))
                examples.append(Example.from_dict(row, where="%s:%d" % (path, lineno)))
        if not examples:
            raise EvalError("%s contains no examples" % path)
        return cls(examples)


def _canonical(value: Any) -> str:
    # `sort_keys=False` on purpose. `examples/hooks/cache.py:22-25` states the rule this mirrors:
    # "Deliberately not `sort_keys=True`: a choice question's criteria order is positional, so two
    # orders are two questions, and `_question_schema` in `laya/router.py` keeps them apart for the
    # same reason." `sort_keys` reorders *dict* keys and leaves *lists* alone, so with a list-valued
    # `criteria` -- the shape a dataset row actually has -- this was already order-preserving; the
    # flag only ever mattered for a dict-valued one, where folding is the wrong direction.
    return json.dumps(value, sort_keys=False, separators=(",", ":"), ensure_ascii=False,
                      default=str)


def _as_tokenized_instructions(body: Dict[str, Any]) -> Any:
    """`instructions` exactly as `Agent._to_internal` (`laya/agent.py:736-748`) hands it over.

    A non-string becomes ``json.dumps(ins, ensure_ascii=False)``, and that step is not cosmetic:
    ``tests/test_criteria.py:229-251`` pins that the default ``ensure_ascii=True`` escaped
    non-ASCII to literal ``\\uXXXX``, the tokenizer read it as escape text, and one German
    question answered noul=0.1652 as a dict against 0.2650 as the identical plain string. So the
    fingerprint normalizes the same way, or it hashes the input's JSON shape instead of the text
    the model reads.

    Not mirrored: the tokenizer's mask-token strip in `common.build_sequence`, which needs
    `tok.mask_token` and would mean guessing which tokenizer a run loads. Instructions carrying
    a mask token therefore hash apart even though they render alike -- a false refusal, which is
    the safe direction to be wrong in.
    """
    ins = body.get("instructions")
    return ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)


def questions_fingerprint(dataset: "Dataset") -> str:
    """A stable hash of *what was asked*, over every question in `dataset`.

    This is the question-schema identity `docs/staged-adoption.md` asks an operator to record
    with the policy, and the part of the run identity a report can compute for itself: the
    dataset file hash is the CLI's, but the questions are parsed here.

    It covers every field the answer depends on, which is the same rule `examples/hooks/cache.py`
    states for a cache key ("has to cover everything the answer depends on") and the same one
    Laya already applies to its own question identity: `Router._question_schema`
    (`laya/router.py:141`) and this module's batch grouping (`laya/evals.py`) both hash the whole
    questions dict. `tests/test_router_batch.py:543` pins that rewording `instructions` alone
    moves a row into its own batch group.

    It is a function of the decision space, not of the rows, so scoring more states on the same
    questions leaves it unchanged.

    `criteria` is hashed raw, without `Agent._to_internal`'s list-to-dict and lowercase-key
    rewrites. That is deliberate and asymmetric: two `noul` questions that render alike but are
    written differently would then be refused, which is the safe direction to be wrong in.
    Normalizing them would widen the change past what the identity needs.

    `labels` is hashed for the same reason `instructions` is: it is validated
    (`laya/agent.py:722-726` -> `_resolve_noul_labels`) and carried into the internal question
    (`laya/agent.py:748-749`), and `_resolve_noul_labels` (`laya/common.py:92`) turns it into the
    option text the model reads. Leaving it out let two question sets with different rendered
    options hash alike, which is the one thing this function exists to prevent.
    """
    schemas = set()
    for example in dataset.examples:
        for qid, question in example.questions.items():
            body = question if isinstance(question, dict) else {}
            schemas.add(_canonical({"qid": qid, "type": body.get("type"),
                                    "instructions": _as_tokenized_instructions(body),
                                    "labels": body.get("labels"),
                                    "criteria": body.get("criteria")}))
    digest = hashlib.sha256()
    for schema in sorted(schemas):
        digest.update(schema.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_fingerprint(path: str) -> str:
    """The sha256 of a file's bytes -- the dataset identity, where the path is only a name.

    `config.dataset` is the path as typed, and two datasets share a path across a rebase, a CI
    cache or a colleague's checkout. Raises `EvalError` when the file cannot be read, so a report
    never carries a hash of something other than the bytes that were parsed.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
    except OSError as exc:
        raise EvalError("%s cannot be read for a fingerprint: %s" % (path, exc)) from exc
    return digest.hexdigest()


def _library_version() -> Optional[str]:
    """`laya.__version__`, read lazily so this module stays importable on its own.

    `laya/__init__.py` defines the string at import time, so this is a module lookup rather
    than a load of anything torch-backed -- but it is imported inside the call to keep
    `laya.evals` free of an import cycle, and a failure here must not fail a run that has
    already scored every row.
    """
    try:
        import laya
    except Exception:  # pragma: no cover - identity only
        return None
    return getattr(laya, "__version__", None)


def _run_identity(dataset: "Dataset") -> Dict[str, Any]:
    """The `config` keys this module can fill in on its own, with no knowledge of the CLI."""
    identity: Dict[str, Any] = {"schema": REPORT_SCHEMA,
                                "questions_sha256": questions_fingerprint(dataset)}
    version = _library_version()
    if version is not None:
        identity["laya_version"] = version
    return identity


# --------------------------------------------------------------------------- evaluators
class Evaluator:
    """Per-answer metric. `score` returns a number, or None when it does not apply."""

    name = "evaluator"

    def score(self, answer: Dict[str, Any], expected: Any) -> Optional[float]:
        raise NotImplementedError


class ChoiceAccuracy(Evaluator):
    name = "choice_accuracy"

    def score(self, answer, expected):
        if answer.get("type") != "choice" or not isinstance(expected, str):
            return None
        return 1.0 if answer.get("choice") == expected else 0.0


class NoulAccuracy(Evaluator):
    name = "noul_accuracy"

    def score(self, answer, expected):
        if answer.get("type") != "noul" or not isinstance(expected, bool):
            return None
        return 1.0 if bool(answer.get("noul", 0.0) >= 0.5) == expected else 0.0


class ScoreMAE(Evaluator):
    name = "score_mae"

    def score(self, answer, expected):
        if answer.get("type") != "score" or not isinstance(expected, (int, float)) or isinstance(expected, bool):
            return None
        return abs(float(answer.get("score", 0.0)) - float(expected))


class ScoreWithin(Evaluator):
    name = "score_within"

    def __init__(self, tolerance: float = 0.1):
        self.tolerance = float(tolerance)
        self.name = "score_within_%g" % self.tolerance

    def score(self, answer, expected):
        if answer.get("type") != "score" or not isinstance(expected, (int, float)) or isinstance(expected, bool):
            return None
        return 1.0 if abs(float(answer.get("score", 0.0)) - float(expected)) <= self.tolerance else 0.0


class MeanConfidence(Evaluator):
    name = "mean_confidence"

    def score(self, answer, expected):
        return _answer_confidence(answer)


DEFAULT_EVALUATORS = (ChoiceAccuracy, NoulAccuracy, ScoreMAE, MeanConfidence)


def default_evaluators() -> List[Evaluator]:
    return [factory() for factory in DEFAULT_EVALUATORS]


def _answer_confidence(answer: Dict[str, Any]) -> Optional[float]:
    """The calibrated confidence Laya reports: `answer_confidence`, not the entropy score.

    `answer["confidence"]` is entropy-based for choice and score, so calibration metrics must use
    `answer_confidence`, which Laya reports on every answer type. The other keys are fallbacks for
    a stripped-down result.
    """
    confidence = answer.get("answer_confidence")
    if isinstance(confidence, (int, float)):
        return float(confidence)
    confidence = answer.get("confidence")
    if isinstance(confidence, (int, float)):
        return float(confidence)
    if answer.get("type") == "noul":
        p = float(answer.get("noul", 0.0))
        return max(p, 1.0 - p)
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        return max(float(v) for v in probabilities.values())
    return None


def _correct(answer: Dict[str, Any], expected: Any) -> Optional[bool]:
    if answer.get("type") == "choice" and isinstance(expected, str):
        return answer.get("choice") == expected
    if answer.get("type") == "noul" and isinstance(expected, bool):
        return bool(answer.get("noul", 0.0) >= 0.5) == expected
    return None


def ece(confidences: Sequence[float], corrects: Sequence[bool], bins: int = 15) -> Optional[float]:
    """Expected Calibration Error, reusing the repository's own `common.ece_score`."""
    if not confidences:
        return None
    from .common import ece_score     # lazy: keeps `import laya.evals` torch-free

    return float(ece_score(np.asarray(confidences, dtype=float),
                           np.asarray(corrects, dtype=bool), bins=bins))


def _answered_model(result: Any) -> Optional[Any]:
    """Which checkpoint produced `result`, read the way each runner records it.

    A `Router` puts its choice under `routing["model"]` and leaves no checkpoint name at the top
    level: `result["model"]` is the payload's family tag, `"laya-rl-agent"`, on every Agent and
    ONNXAgent result too. So a routed run has to read `routing`, or its `by model` slice reports
    one family for every checkpoint that answered.
    """
    result = result or {}
    routing = result.get("routing")
    if isinstance(routing, dict):
        chosen = routing.get("model")
        if isinstance(chosen, str) and chosen:
            return chosen
    return result.get("model")


# --------------------------------------------------------------------------- evaluation
@dataclass
class EvalReport:
    """Overall and per-slice metrics, with the per-case records they were derived from."""

    config: Dict[str, Any] = field(default_factory=dict)
    overall: Dict[str, float] = field(default_factory=dict)
    slices: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {"config": self.config, "overall": self.overall, "slices": self.slices,
                "cases": self.cases}

    def to_markdown(self) -> str:
        lines = ["| metric | value |", "|---|---|"]
        for name in sorted(self.overall):
            lines.append("| %s | %.4f |" % (name, self.overall[name]))
        for dimension in sorted(self.slices):
            lines.append("")
            lines.append("### by %s" % dimension)
            lines.append("")
            metrics = sorted({m for group in self.slices[dimension].values() for m in group})
            lines.append("| %s | %s |" % (dimension, " | ".join(metrics)))
            lines.append("|---|%s" % ("---|" * len(metrics)))
            for value in sorted(self.slices[dimension]):
                cells = ["%.4f" % self.slices[dimension][value].get(m, float("nan"))
                         if m in self.slices[dimension][value] else "" for m in metrics]
                lines.append("| %s | %s |" % (value, " | ".join(cells)))
        return "\n".join(lines) + "\n"

    def comparable_to(self, baseline: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Is `baseline` the same measurement as this report? Returns (ok, reasons).

        `compare` reads `overall` and only `overall`, so two reports of different datasets,
        different question schemas or different report shapes produce identical arithmetic and
        an identical pass. That is the failure `docs/staged-adoption.md` runs at: a gate that
        cannot tell which experiment produced a number cannot support a promotion decision.

        A key absent on either side is *unknown*, not a conflict, so every baseline committed
        before the identity existed -- including the scheduled gate's
        `research/results/eval_english_51_languages.json`, which comes from `research/eval/` and
        has no `config.schema` -- keeps comparing exactly as it did. Only a key present on both
        sides with different values refuses the comparison, and each reason names the key and
        both values so the failure is actionable from the console alone.

        `laya_version` and `thresholds` are recorded in `config` but deliberately not compared
        here: a patch release must not invalidate a committed baseline.
        """
        other = _identity_of(baseline)
        reasons: List[str] = []
        for key in _IDENTITY_KEYS:
            here, there = self.config.get(key), other.get(key)
            if here is None or there is None or here == there:
                continue
            reasons.append("%s (%s): baseline is %s, this run is %s"
                           % (key, _IDENTITY_LABELS[key], there, here))
        return (not reasons), reasons

    def compare(self, baseline: Dict[str, Any], tolerances: Optional[Dict[str, float]] = None,
                ) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        """Compare `overall` to a baseline report's `overall`. Returns (ok, deltas).

        With no `tolerances`, every shared metric must match exactly; a tolerance is the maximum
        absolute difference allowed for that metric. A baseline metric the report no longer has
        fails the comparison (its delta carries ``missing: True`` and a NaN value): a run whose
        every example errored under ``on_error="skip"`` has an empty ``overall``, and must not
        pass the gate by having nothing left to compare.
        """
        base = (baseline or {}).get("overall", baseline or {})
        tolerances = tolerances or {}
        deltas: Dict[str, Dict[str, float]] = {}
        ok = True
        for metric, base_value in base.items():
            # Latency is informational; a re-run differs by timing noise, not quality, so it is
            # compared only when a tolerance explicitly names it.
            if metric.endswith("_ms") and metric not in tolerances:
                continue
            allowed = float(tolerances.get(metric, 0.0))
            if metric not in self.overall:
                deltas[metric] = {"baseline": float(base_value), "value": float("nan"),
                                  "diff": float("nan"), "tolerance": allowed, "missing": True}
                ok = False
                continue
            value = self.overall[metric]
            diff = value - float(base_value)
            deltas[metric] = {"baseline": float(base_value), "value": value,
                              "diff": diff, "tolerance": allowed}
            if abs(diff) > allowed:
                ok = False
        return ok, deltas


def _group_cases(cases: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        if key == "tag":
            values = case.get("tags") or []
        else:
            value = case.get(key)
            values = [] if value is None else [value]
        for value in values:
            groups.setdefault(str(value), []).append(case)
    return groups


def _percentiles(values: Sequence[float]) -> Tuple[float, float]:
    """(median, 95th) using the nearest-rank rule for latency."""
    ordered = sorted(values)
    n = len(ordered)
    # Nearest-rank 95th percentile: the ceil(0.95 * n)-th smallest value (1-indexed).
    # `int(n * 0.95)` truncated where it needed to round up, returning the
    # (0.95n + 1)-th value whenever n is a multiple of 20. Integer ceil fixes that.
    rank = (95 * n + 99) // 100
    return (float(statistics.median(ordered)), float(ordered[rank - 1]))


def _aggregate(cases: Sequence[Dict[str, Any]], evaluators: Sequence[Evaluator]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for evaluator in evaluators:
        values = [c["scores"][evaluator.name] for c in cases
                  if c.get("scores", {}).get(evaluator.name) is not None]
        if values:
            out[evaluator.name] = float(statistics.fmean(values))
    paired = [(c["confidence"], c["correct"]) for c in cases
              if c.get("confidence") is not None and c.get("correct") is not None]
    if paired:
        value = ece([p[0] for p in paired], [p[1] for p in paired])
        if value is not None and not np.isnan(value):
            out["ece"] = value
    return out


# The two batch call shapes the repo ships. `RouterRunner.predict_batch(states, questions,
# model=..., batch_size=...)` is the positional one; `Router.predict_batch(requests, ...)` takes
# one dict per request, the form `route_batch` and the `laya-evals` CLI document.
_BATCH_STATES = "states"
_BATCH_REQUESTS = "requests"


def _batch_form(runner: Any) -> Optional[str]:
    """Return which batch call `runner` accepts, or ``None`` to score it one ``predict`` at a time.

    Deliberately not ``hasattr(runner, "predict_batch")``: having a batch entry point and being
    callable the way this harness calls it are different claims, and the attribute test returned
    true for ``Router`` -- the one real runner whose ``predict(state, questions, model=...)``
    matches the contract `evaluate` states. Every chunk of more than one example then raised
    ``TypeError: ... unexpected keyword argument 'model'``, which under ``on_error="skip"`` became
    a report with zero cases and an empty ``overall``.
    """
    fn = getattr(runner, "predict_batch", None)
    if fn is None:
        return None
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        # Not introspectable (a C-level or hand-rolled ``__call__``): keep the call this harness
        # made before the shape was checked, rather than silently dropping to single predicts.
        return _BATCH_STATES
    if any(p.name == "model" or p.kind is p.VAR_KEYWORD for p in params):
        return _BATCH_STATES
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    # A bound method, so `positional[0]` is the batch argument itself.
    if positional and positional[0].name == "requests":
        return _BATCH_REQUESTS
    return None


def _takes_sort_by_length(runner: Any) -> bool:
    """Whether `runner`'s batch entry point can be given the length-grouping knob.

    The same signature check `_batch_form` makes, for the one optional argument this harness
    forwards: `sort_by_length` is an optimisation, so a runner whose ``predict_batch`` predates it
    (#294) is scored unsorted rather than raising ``TypeError`` at the first chunk of a long run.
    A ``**kwargs`` forwarder counts, because whatever it wraps is a real runner.
    """
    fn = getattr(runner, "predict_batch", None)
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "sort_by_length" or p.kind is p.VAR_KEYWORD for p in params)


def _takes_min_confidence(runner: Any, fn_name: str = "predict_batch") -> bool:
    """Whether `runner`'s entry point accepts an abstention threshold on the call.

    The same signature check `_takes_sort_by_length` makes, applied to whichever entry point the
    harness is about to call: `min_confidence` changes the answer (an abstention overwrites a
    low-confidence choice), so it is a scoring control, not an optimisation. A runner that predates
    the gate (#361) must still be scoreable -- silently dropping the threshold and reporting the
    same run would give a `precision@coverage` number for a policy that never ran -- so when the
    guard is false the harness raises rather than lies. The CLI catches the raise into a
    pre-flight message before any checkpoint loads.
    """
    fn = getattr(runner, fn_name, None)
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "min_confidence" or p.kind is p.VAR_KEYWORD for p in params)


def evaluate(runner: Any, dataset: Dataset, evaluators: Optional[Sequence[Evaluator]] = None,
             batch_size: Optional[int] = None, on_error: str = "fail",
             config: Optional[Dict[str, Any]] = None, sort_by_length: bool = False,
             min_confidence: Optional[float] = None) -> EvalReport:
    """Run `runner` over `dataset`, aggregating per-answer metrics overall and per slice.

    `runner` needs a ``predict(state, questions, model=...)`` method, and for `batch_size` above 1
    a ``predict_batch`` in either shape the repo ships: the positional
    ``predict_batch(states, questions, model=..., batch_size=...)``, or the per-request-dict form
    `Router.predict_batch` takes -- ``predict_batch([{"state": ..., "questions": ...,
    "model": ...}, ...], batch_size=...)``. A runner that offers neither is scored one ``predict``
    at a time, which is slower but not wrong.

    `sort_by_length` is forwarded to a chunked runner so similarly sized examples share a forward
    pass and pad to a shorter maximum. It reaches the checkpoint's own batching only when the
    runner's ``predict_batch`` takes it, and only when it is on: an unset control is not sent, so a
    runner that predates the knob is unaffected by a run that does not ask. Nothing about the
    scored answers changes -- the results come back in chunk order either way.

    `min_confidence` is the abstention threshold `Router` and `ONNXAgent` apply to
    `answer_confidence` (#361): answers below it come back abstained, so the run scores the
    policy at that threshold, not the raw argmax. Unlike `sort_by_length` this changes the
    answers, so a runner whose batch entry point (or whose single ``predict``, on the fallback
    path) predates the gate is refused rather than silently scored without it -- the report
    would otherwise publish a `precision@coverage` figure for an abstention policy that never
    ran.

    `on_error` is ``"fail"`` (re-raise a runner error) or ``"skip"`` (record it and continue),
    the latter for evaluating a flaky fleet without aborting the whole run.

    A batched run has two honest timing answers, so the report gives both. ``latency_p50_ms`` /
    ``latency_p95_ms`` are per request: every row of a chunk returns from the same call, so each
    one waited that whole call. ``cost_per_decision_p50_ms`` / ``cost_per_decision_p95_ms``
    divide a call by its own chunk size, which is the throughput figure. ``--batch-size`` therefore
    lowers the second and raises the first. What the harness really did -- the runner shape it
    resolved to and how many rows shared a call -- lands in ``report.config["timing"]``, because the
    requested flag alone does not say whether anything was batched.
    """
    if on_error not in ("fail", "skip"):
        raise EvalError("on_error must be 'fail' or 'skip', got %r" % on_error)
    # `min_confidence` is validated here -- and, unlike `sort_by_length`, a run that asks for it
    # on a runner that does not accept it is refused -- because an abstention threshold changes
    # which answers score as correct. Silently dropping it would publish a `precision@coverage`
    # number for a policy that never ran, which is exactly the class of lie a baseline report is
    # supposed to prevent. `laya.confidence.check_min_confidence` is the same validator the Router
    # uses, so the accepted range cannot drift from what the gate itself enforces.
    try:
        mc = check_min_confidence(min_confidence) if min_confidence is not None else None
    except ValueError as exc:
        # Core's validator raises a bare ValueError; the CLI's usage-error handler catches EvalError
        # and turns it into exit 2 with a printed message. Re-raising here keeps a mistyped
        # `--min-confidence 1.5` on the same path as a bad tolerance, instead of a traceback.
        raise EvalError(str(exc)) from exc
    if mc is not None:
        # Whichever entry points this run can actually reach all have to take the threshold. A
        # batched run still sends a chunk of one through `predict`, so requiring only
        # ``predict_batch`` would let the runner silently score those rows without the gate --
        # the exact class of lie the guard exists to prevent.
        targets = ["predict"]
        if batch_size is not None and batch_size > 1 and _batch_form(runner):
            targets.append("predict_batch")
        for target in targets:
            if not _takes_min_confidence(runner, target):
                raise EvalError(
                    "evaluate(min_confidence=%r) refused: this runner's %s does not accept the "
                    "abstention threshold. Either use a Router/ONNXAgent that gates on "
                    "answer_confidence (#361), or drop the threshold -- the report would "
                    "otherwise score a policy that never ran." % (min_confidence, target))
    evaluators = list(evaluators) if evaluators is not None else default_evaluators()
    cases: List[Dict[str, Any]] = []
    waits: List[float] = []          # what each request actually waited: its chunk's whole call
    shares: List[float] = []         # that call split across its chunk: throughput per decision
    chunks = rows_grouped = rows_alone = max_chunk = 0
    errors: List[Dict[str, Any]] = []
    examples = dataset.examples
    # Only worth grouping if the runner can be handed the group in one call at all.
    batch_form = _batch_form(runner) if batch_size is not None and batch_size > 1 else None
    # The chunk shape travels with the calls that have one: a chunk of a single example is a plain
    # `predict`, which has no batch to reorder, and an off control is not sent at all.
    shape = ({"sort_by_length": True}
             if sort_by_length and batch_form and _takes_sort_by_length(runner) else {})
    # `min_confidence` joins the batch shape whenever a chunk can go through `predict_batch`, and
    # the single-`predict` path picks it up below. The pre-flight refusal above has already
    # proven every entry point this run can reach accepts it, so no re-check sits here: a run
    # that reaches this line either asked for the gate and will get it on every call, or asked
    # for nothing.
    if mc is not None and batch_form:
        shape = dict(shape, min_confidence=mc)

    index = 0
    while index < len(examples):
        # Batches only when the runner can share a forward pass: consecutive examples with the
        # same checkpoint and identical questions. Otherwise every example is one predict.
        chunk = [examples[index]]
        if batch_form:
            signature = (examples[index].model,
                         json.dumps(examples[index].questions, sort_keys=False, default=str))
            while (index + len(chunk) < len(examples) and len(chunk) < batch_size
                   and (examples[index + len(chunk)].model,
                        json.dumps(examples[index + len(chunk)].questions, sort_keys=False, default=str)) == signature):
                chunk.append(examples[index + len(chunk)])
        # Why these sit above the call and `waits`/`shares` below: the shape counters answer "what
        # did the harness issue", which is settled the moment the chunk is grouped, while the
        # metric lists answer "what did a request wait", which needs a call that returned. Counting
        # only completed calls made a run that issued one shared forward and lost it to an error
        # under `on_error="skip"` report the same `rows_grouped` and `max_chunk` as a runner with
        # no batching at all (#592 review).
        chunks += 1
        if len(chunk) > 1:
            rows_grouped += len(chunk)
        else:
            rows_alone += 1
        if len(chunk) > max_chunk:
            max_chunk = len(chunk)
        started = time.perf_counter()
        try:
            if len(chunk) > 1:
                if batch_form == _BATCH_REQUESTS:
                    # The chunk already shares one checkpoint and one question schema, so the
                    # per-request dicts carry exactly what the positional call would pass.
                    results = runner.predict_batch(
                        [{"state": e.state, "questions": e.questions, "model": e.model}
                         for e in chunk], batch_size=batch_size, **shape)
                else:
                    results = runner.predict_batch([e.state for e in chunk], chunk[0].questions,
                                                   model=chunk[0].model, batch_size=batch_size,
                                                   **shape)
            else:
                # A chunk of one still sees the gate: the pre-flight refusal proved `predict`
                # accepts it whenever `mc` is set, so nothing here has to re-check the signature.
                single_kwargs = {"min_confidence": mc} if mc is not None else {}
                results = [runner.predict(chunk[0].state, chunk[0].questions,
                                          model=chunk[0].model, **single_kwargs)]
        except Exception as exc:  # noqa: BLE001 -- honoured by on_error
            if on_error == "fail":
                raise
            errors.extend({"index": index + offset, "error": "%s: %s" % (type(exc).__name__, exc)}
                          for offset in range(len(chunk)))
            index += len(chunk)
            continue
        elapsed = (time.perf_counter() - started) * 1000.0
        # A chunk is one call, so each of its rows waited all of `elapsed`; the split figure is a
        # throughput share, not a latency, and putting the two in one list made a batched run
        # publish a per-decision cost under the name of the per-request one (#585).
        waits.extend([elapsed] * len(chunk))
        shares.extend([elapsed / len(chunk)] * len(chunk))
        for offset, (example, result) in enumerate(zip(chunk, results)):
            answers = (result or {}).get("answers") or {}
            for qid, expected in example.expected.items():
                answer = answers.get(qid)
                if not isinstance(answer, dict):
                    if on_error == "fail":
                        raise EvalError("runner returned no answer for question %r" % qid)
                    errors.append({"index": index + offset, "question": qid, "error": "missing answer"})
                    continue
                case = {
                    "qid": qid,
                    "language": example.language,
                    "model": example.model or _answered_model(result),
                    "tags": list(example.tags),
                    "expected": expected,
                    "answer": answer,
                    "confidence": _answer_confidence(answer),
                    "correct": _correct(answer, expected),
                    "scores": {evaluator.name: evaluator.score(answer, expected)
                               for evaluator in evaluators},
                }
                # Shortlist metadata describes the retrieval stage, not the answer. Keep it
                # beside the case so opt-in evaluations can attribute the source of an error.
                shortlist = (result or {}).get("shortlist")
                if isinstance(shortlist, dict) and qid in shortlist:
                    case["shortlist"] = shortlist[qid]
                cases.append(case)
        index += len(chunk)

    report = EvalReport(config=dict(config or {}), cases=cases)
    report.overall = _aggregate(cases, evaluators)
    if waits:
        report.overall["latency_p50_ms"], report.overall["latency_p95_ms"] = _percentiles(waits)
        (report.overall["cost_per_decision_p50_ms"],
         report.overall["cost_per_decision_p95_ms"]) = _percentiles(shares)
    # The run identity, stamped here rather than left to the caller: everything below is
    # computable from `dataset`, so a programmatic `evaluate` is as identifiable as a CLI run.
    # The CLI adds `dataset_sha256` and `thresholds`, which need the file path and the gate.
    report.config = dict(report.config, **_run_identity(dataset))
    report.config = dict(report.config, timing={
        "latency_metric": "per request: the wall time of the call that returned it, unsplit",
        "cost_metric": "per decision: that call divided by its own chunk size",
        "batch_size": batch_size,
        "batch_form": batch_form,
        "sort_by_length": sort_by_length,
        # Requested and sent are different claims: `sort_by_length` needs a chunk to reorder, so a
        # run with no batch form asked for something this harness cannot do.
        "sort_by_length_sent": bool(shape.get("sort_by_length")),
        # Same shape for `min_confidence`: a run that asked for a threshold and landed on a
        # chunked path where the runner accepted it gets a different report from one that was
        # refused at the guard, and the report should say which. The `_sent` value reflects
        # whether at least one call actually carried the threshold, not whether it was requested.
        "min_confidence": min_confidence,
        "min_confidence_sent": (
            # The pre-flight refusal proved the entry points this run reaches accept the
            # threshold, so a non-None `mc` on a run that made at least one call is exactly the
            # claim "the threshold was on every call". An empty dataset sends nothing.
            mc is not None and chunks > 0
        ),
        "chunks": chunks,
        "rows_grouped": rows_grouped,
        "rows_alone": rows_alone,
        "max_chunk": max_chunk,
    })
    for dimension in ("language", "model", "qid", "tag"):
        groups = _group_cases(cases, dimension)
        if groups:
            report.slices[dimension] = {value: _aggregate(group, evaluators)
                                        for value, group in groups.items()}
    if errors:
        report.config = dict(report.config, errored=errors)
    return report


def assert_regression(report: EvalReport, baseline: Dict[str, Any],
                      tolerances: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    """Raise AssertionError when `report` drifts from `baseline` beyond `tolerances`."""
    ok, deltas = report.compare(baseline, tolerances)
    if not ok:
        failed = {m: d for m, d in deltas.items()
                  if d.get("missing") or abs(d["diff"]) > d["tolerance"]}
        raise AssertionError("evaluation regressed: %s" % json.dumps(failed, sort_keys=True))
    return deltas
