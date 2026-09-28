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
"""
from __future__ import annotations

import inspect
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


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
    """(median, 95th) using the nearest-rank rule the report has always used for latency."""
    ordered = sorted(values)
    # Nearest-rank: the ceil(0.95 * n)-th smallest. int(n * 0.95) would land one rank too high
    # whenever 0.95 * n is whole -- every sample size that is a multiple of 20 (#680).
    return (float(statistics.median(ordered)),
            float(ordered[math.ceil(len(ordered) * 0.95) - 1]))


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


def evaluate(runner: Any, dataset: Dataset, evaluators: Optional[Sequence[Evaluator]] = None,
             batch_size: Optional[int] = None, on_error: str = "fail",
             config: Optional[Dict[str, Any]] = None) -> EvalReport:
    """Run `runner` over `dataset`, aggregating per-answer metrics overall and per slice.

    `runner` needs a ``predict(state, questions, model=...)`` method, and for `batch_size` above 1
    a ``predict_batch`` in either shape the repo ships: the positional
    ``predict_batch(states, questions, model=..., batch_size=...)``, or the per-request-dict form
    `Router.predict_batch` takes -- ``predict_batch([{"state": ..., "questions": ...,
    "model": ...}, ...], batch_size=...)``. A runner that offers neither is scored one ``predict``
    at a time, which is slower but not wrong.

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
    evaluators = list(evaluators) if evaluators is not None else default_evaluators()
    cases: List[Dict[str, Any]] = []
    waits: List[float] = []          # what each request actually waited: its chunk's whole call
    shares: List[float] = []         # that call split across its chunk: throughput per decision
    chunks = rows_grouped = rows_alone = max_chunk = 0
    errors: List[Dict[str, Any]] = []
    examples = dataset.examples
    # Only worth grouping if the runner can be handed the group in one call at all.
    batch_form = _batch_form(runner) if batch_size is not None and batch_size > 1 else None

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
                         for e in chunk], batch_size=batch_size)
                else:
                    results = runner.predict_batch([e.state for e in chunk], chunk[0].questions,
                                                   model=chunk[0].model, batch_size=batch_size)
            else:
                results = [runner.predict(chunk[0].state, chunk[0].questions, model=chunk[0].model)]
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
                cases.append({
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
                })
        index += len(chunk)

    report = EvalReport(config=dict(config or {}), cases=cases)
    report.overall = _aggregate(cases, evaluators)
    if waits:
        report.overall["latency_p50_ms"], report.overall["latency_p95_ms"] = _percentiles(waits)
        (report.overall["cost_per_decision_p50_ms"],
         report.overall["cost_per_decision_p95_ms"]) = _percentiles(shares)
    report.config = dict(report.config, timing={
        "latency_metric": "per request: the wall time of the call that returned it, unsplit",
        "cost_metric": "per decision: that call divided by its own chunk size",
        "batch_size": batch_size,
        "batch_form": batch_form,
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
