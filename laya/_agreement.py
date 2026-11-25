"""Offline, paired teacher-repeat and student agreement on choice decisions.

Teacher observations are comparison targets, not ground truth. Nothing here computes accuracy
or calibration, calls a teacher service, or estimates a ceiling on student performance.
"""
from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np

from .confidence import GATE_ABSTAINED
from .evals import EvalError


RUN_SCHEMA = "laya-agreement-run/1"
REPORT_SCHEMA = "laya-agreement-report/1"


def _text(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvalError("%s must be a non-empty string" % where)


def _json(value: Any, where: str) -> str:
    """Strict JSON, preserving object order because it can change model inputs.

    Containers are matched by `isinstance`, scalars by exact type. `json.dumps` writes a `dict` or
    `list` subclass as the very object or array it wraps -- an `OrderedDict`, a `defaultdict`, a
    `Counter`, or anything read back with `object_pairs_hook=OrderedDict`, the way a caller keeps the
    key order this module fingerprints -- encodes byte-for-byte as the mapping it equals and hashes to
    the same fingerprint, so matching the encoder here refuses nothing the encoder would have taken.
    Scalars are matched by exact type: `numpy.float64` and `numpy.str_` do encode, but as a type the
    writer did not declare, and `numpy.int64` and `numpy.bool_` raise `TypeError` in the encoder.
    """
    def check(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("object keys must be strings")
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("unsupported value of type %s" % type(item).__name__)

    try:
        check(value)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        encoded.encode("utf-8")
        return encoded
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise EvalError("%s must be strict JSON: %s" % (where, exc)) from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json(value, "fingerprint input").encode("utf-8")).hexdigest()


def _controls(bootstrap_samples: int, seed: int) -> None:
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 100000:
        raise EvalError("bootstrap_samples must be an integer from 100 to 100000")
    if type(seed) is not int or seed < 0:
        raise EvalError("seed must be a non-negative integer")


def _provenance(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise EvalError("%s must be an object" % where)
    _text(value.get("model"), where + ".model")
    if "revision" not in value:
        raise EvalError("%s needs revision (use null when unknown)" % where)
    if value["revision"] is not None:
        _text(value["revision"], where + ".revision")
    if not isinstance(value.get("settings"), dict):
        raise EvalError("%s.settings must be an object" % where)
    _json(value, where)


def _labels(question: Any, where: str) -> List[str]:
    if not isinstance(question, dict) or question.get("type") != "choice":
        raise EvalError("%s must be a choice question" % where)
    instructions = question.get("instructions")
    if (instructions is None or not isinstance(instructions, (str, dict, list, int, float))
            or (isinstance(instructions, str) and not instructions.strip())
            or (isinstance(instructions, (dict, list)) and not instructions)):
        raise EvalError("%s needs non-empty instructions" % where)
    criteria = question.get("criteria")
    if not isinstance(criteria, (dict, list)) or not criteria:
        raise EvalError("%s.criteria must be a non-empty object or list of string labels" % where)
    # Agent._to_internal preserves dict keys and expands a list to {label: None}.
    # This protocol accepts string labels only, so saved JSON and runtime answers agree.
    labels = list(criteria)
    if any(not isinstance(label, str) for label in labels):
        raise EvalError("%s.criteria labels must be strings" % where)
    if len(set(labels)) != len(labels):
        raise EvalError("%s.criteria contains duplicate labels" % where)
    if "labels" in question:
        raise EvalError("%s: labels is only supported for noul questions" % where)
    if "option_order" in question:
        order = question["option_order"]
        if (not isinstance(order, list) or any(type(i) is not int for i in order)
                or sorted(order) != list(range(len(labels)))):
            raise EvalError("%s.option_order must be a permutation of the option indices" % where)
    return labels


def _run(value: Any, role: str, student: bool = False) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != RUN_SCHEMA:
        raise EvalError("%s.schema must be %r" % (role, RUN_SCHEMA))
    _json(value, role)
    _text(value.get("run_id"), role + ".run_id")
    _provenance(value.get("provenance"), role + ".provenance")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvalError("%s.cases must be a non-empty list" % role)
    by_id = {}
    for index, case in enumerate(cases):
        where = "%s.cases[%d]" % (role, index)
        if not isinstance(case, dict):
            raise EvalError("%s must be an object" % where)
        _text(case.get("id"), where + ".id")
        case_id = case["id"]
        if case_id in by_id:
            raise EvalError("%s: duplicate case id %r" % (role, case_id))
        if case.get("state") is None:
            raise EvalError("%s needs a non-null state" % where)
        questions, answers = case.get("questions"), case.get("answers")
        if not isinstance(questions, dict) or not questions:
            raise EvalError("%s.questions must be a non-empty object" % where)
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise EvalError("%s.answers must have exactly the question ids" % where)
        for qid, question in questions.items():
            _text(qid, where + ".questions key")
            labels = _labels(question, "%s.questions[%r]" % (where, qid))
            answer = answers[qid]
            if answer is None and student:
                continue
            if not isinstance(answer, str) or answer not in labels:
                raise EvalError("%s.answers[%r] must be a choice label%s"
                                % (where, qid, " or null" if student else ""))
        if case.get("language") is not None and not isinstance(case["language"], str):
            raise EvalError("%s.language must be a string or null" % where)
        tags = case.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise EvalError("%s.tags must be a list of strings" % where)
        by_id[case_id] = case
    # Sorting cases makes the artifact identity independent of JSONL/array row order.
    normalized = dict(value, cases=[by_id[key] for key in sorted(by_id)])
    return {"run": value, "by_id": by_id, "fingerprint": _fingerprint(normalized)}


def _align(reference: Dict, other: Dict, role: str) -> None:
    if set(reference["by_id"]) != set(other["by_id"]):
        raise EvalError("%s case ids must exactly match reference case ids" % role)
    for case_id, case in reference["by_id"].items():
        candidate = other["by_id"][case_id]
        for key in ("state", "questions"):
            if _json(case[key], key) != _json(candidate[key], key):
                raise EvalError("%s case %r has different %s (including option order)"
                                % (role, case_id, key))
        # Slice labels belong to the inputs, not to the model being compared.
        if (case.get("language") != candidate.get("language")
                or set(case.get("tags", [])) != set(candidate.get("tags", []))):
            raise EvalError("%s case %r has different language or tags" % (role, case_id))


def _teachers(reference: Any, repeat: Any) -> tuple:
    ref, rep = _run(reference, "reference"), _run(repeat, "repeat")
    if reference["run_id"] == repeat["run_id"]:
        raise EvalError("reference and repeat need distinct run_id values")
    for key in ("model", "revision", "settings"):
        if _json(reference["provenance"][key], key) != _json(repeat["provenance"][key], key):
            raise EvalError("teacher provenance.%s must match between reference and repeat" % key)
    _align(ref, rep, "repeat")
    return ref, rep


def _aggregate(records: List[Dict]) -> Dict[str, Any]:
    n = len(records)
    teacher = sum(r["teacher_repeat_agreement"] for r in records)
    student = sum(r["student_teacher_agreement"] for r in records)
    return {
        "n_cases": len({r["case_id"] for r in records}), "n_decisions": n,
        "teacher_repeat_agreement": teacher / n,
        "student_teacher_agreement": student / n,
        "student_coverage": sum(r["accepted"] for r in records) / n,
        "paired_agreement_delta": (student - teacher) / n,
    }


def _interval(records: List[Dict], samples: int, seed: int) -> Optional[List[float]]:
    totals = {}
    for record in records:
        cluster = totals.setdefault(record["case_id"], [0, 0])
        cluster[0] += record["delta"]
        cluster[1] += 1
    if len(totals) < 2:
        return None
    clusters = np.asarray([totals[key] for key in sorted(totals)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    # Each replicate resamples whole cases. Ratio of sums keeps the estimand per decision
    # when cases have different question counts; batching bounds the index matrix memory.
    batch = max(1, min(samples, 131072 // len(clusters)))
    for start in range(0, samples, batch):
        stop = min(samples, start + batch)
        indices = rng.integers(len(clusters), size=(stop - start, len(clusters)))
        sums = clusters[indices].sum(axis=1)
        draws[start:stop] = sums[:, 0] / sums[:, 1]
    return np.percentile(draws, [2.5, 97.5]).tolist()


def _report(ref: Dict, rep: Dict, stu: Dict, bootstrap_samples: int, seed: int) -> Dict[str, Any]:
    records = []
    for case_id in sorted(ref["by_id"]):
        case = ref["by_id"][case_id]
        input_fingerprint = _fingerprint({"state": case["state"], "questions": case["questions"]})
        for qid in sorted(case["questions"]):
            first = case["answers"][qid]
            second = rep["by_id"][case_id]["answers"][qid]
            student = stu["by_id"][case_id]["answers"][qid]
            teacher_agreement, student_agreement = second == first, student == first
            records.append({
                "case_id": case_id, "qid": qid, "teacher_reference": first,
                "teacher_repeat": second, "student": student,
                "teacher_repeat_agreement": teacher_agreement,
                "student_teacher_agreement": student_agreement, "accepted": student is not None,
                "delta": int(student_agreement) - int(teacher_agreement),
                "input_fingerprint": input_fingerprint, "language": case.get("language"),
                "tags": sorted(set(case.get("tags", []))),
            })
    overall = _aggregate(records)
    overall["paired_delta_ci95"] = _interval(records, bootstrap_samples, seed)
    slices = {}
    for dimension in ("qid", "language", "tag"):
        groups = {}
        for record in records:
            values = record["tags"] if dimension == "tag" else [record[dimension]]
            for value in values:
                if value is not None:
                    groups.setdefault(value, []).append(record)
        slices[dimension] = {key: _aggregate(groups[key]) for key in sorted(groups)}
    runs = {"reference": ref, "repeat": rep, "student": stu}
    limitations = [
        "Agreement is not accuracy, a student performance ceiling, or evidence that a teacher can be replaced.",
        "Teacher independence and provenance are declared by the caller, not verified by this report.",
        "The paired interval resamples original cases with their saved observations; it does not resample teacher calls.",
    ]
    unknown = [role for role, data in runs.items() if data["run"]["provenance"]["revision"] is None]
    if unknown:
        limitations.append("Unknown model revision for %s; the observations cannot be fully regenerated from this provenance."
                           % ", ".join(unknown))
    if overall["n_cases"] < 2:
        limitations.append("A paired interval needs at least two original cases; paired_delta_ci95 is null.")
    return {
        "schema": REPORT_SCHEMA,
        "config": {
            "protocol": "teacher-reference-paired/1", "bootstrap_samples": bootstrap_samples,
            "seed": seed, "cluster_unit": "case_id", "confidence_level": 0.95,
            "interval_method": "paired case bootstrap, percentile, ratio of decision sums",
            "provenance": {role: deepcopy(data["run"]["provenance"]) for role, data in runs.items()},
            "run_ids": {role: data["run"]["run_id"] for role, data in runs.items()},
            "input_fingerprints": {role: data["fingerprint"] for role, data in runs.items()},
        },
        "overall": overall, "slices": slices, "cases": records, "limitations": limitations,
    }


def compare_agreement(reference: Dict, repeat: Dict, student: Dict, *,
                      bootstrap_samples: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Compare three saved choice runs, paired by case id and question id.

    A student null is an abstention: it contributes zero agreement and zero coverage on the
    same denominator as answered decisions. Missing cases or answers are errors.
    """
    _controls(bootstrap_samples, seed)
    ref, rep = _teachers(reference, repeat)
    stu = _run(student, "student", student=True)
    if student["run_id"] in (reference["run_id"], repeat["run_id"]):
        raise EvalError("student needs a run_id distinct from both teacher runs")
    _align(ref, stu, "student")
    return _report(ref, rep, stu, bootstrap_samples, seed)


def evaluate_agreement(runner: Any, reference: Dict, repeat: Dict, *,
                       student_provenance: Dict, model: Optional[str] = None,
                       bootstrap_samples: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Run a local student once per case after validating both teacher observations.

    Provenance describes the caller's runner; no checkpoint revision is inferred or fabricated.
    The runner receives copied inputs and its exceptions propagate without dropping cases.

    An abstention reaches this through the gate's own vocabulary: an answer carrying
    ``abstention`` equal to ``laya.confidence.GATE_ABSTAINED`` becomes a null, which is what
    `student_coverage` then counts as uncovered. The gate's two other states do not. `passed` is
    a clearance, and `unevaluated` means the gate had no usable confidence to compare -- the
    student still answered, so the choice is kept.
    """
    _controls(bootstrap_samples, seed)
    ref, rep = _teachers(reference, repeat)
    _provenance(student_provenance, "student_provenance")
    if model is not None:
        _text(model, "model")
    cases = []
    for case_id in sorted(ref["by_id"]):
        case = ref["by_id"][case_id]
        result = runner.predict(deepcopy(case["state"]), deepcopy(case["questions"]), model=model)
        answers = result.get("answers") if isinstance(result, dict) else None
        if not isinstance(answers, dict) or set(answers) != set(case["questions"]):
            raise EvalError("student case %r must return exactly the question ids" % case_id)
        choices = {}
        for qid, question in case["questions"].items():
            answer = answers[qid]
            where = "student case %r question %r" % (case_id, qid)
            if not isinstance(answer, dict) or answer.get("type") != "choice":
                raise EvalError("%s must return a choice answer" % where)
            choice = answer.get("choice")
            if not isinstance(choice, str) or choice not in _labels(question, where):
                raise EvalError("%s returned an invalid choice label" % where)
            # The marker belongs to the gate's vocabulary, not to this module: of the three states
            # `laya.confidence` can write, only GATE_ABSTAINED means the student declined. A caller
            # that read the field instead of the value ("if answer.get(...)") would drop the
            # `unevaluated` answers -- gated with no usable confidence, choice intact -- from coverage.
            choices[qid] = None if answer.get("abstention") == GATE_ABSTAINED else choice
        cases.append(dict(deepcopy(case), answers=choices))
    run_id, suffix = "student", 0
    while run_id in (reference["run_id"], repeat["run_id"]):
        suffix += 1
        run_id = "student-%d" % suffix
    student = {"schema": RUN_SCHEMA, "run_id": run_id,
               "provenance": deepcopy(student_provenance), "cases": cases}
    stu = _run(student, "student", student=True)
    return _report(ref, rep, stu, bootstrap_samples, seed)
