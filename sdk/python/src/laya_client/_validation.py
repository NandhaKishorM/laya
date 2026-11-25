"""Transport checks, deliberately separate from the server's domain validation."""
import math
import threading
from typing import Any

from .errors import LayaResponseError, LayaValidationError


def finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def probability(value: Any) -> bool:
    return finite(value) and 0 <= value <= 1


def timeout_value(value: Any) -> float:
    if not finite(value) or not 0 < value <= threading.TIMEOUT_MAX:
        raise LayaValidationError("timeout must be positive, finite and within the platform timeout limit")
    return float(value)


def json_value(value: Any, ancestors=None) -> None:
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, str):
        value.encode("utf-8")  # Reject unpaired surrogates before sending.
        return
    if type(value) is float and math.isfinite(value):
        return
    if not isinstance(value, (list, dict)):
        raise LayaValidationError("request must contain only finite JSON values")
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        raise LayaValidationError("request contains a circular reference")
    ancestors.add(id(value))
    try:
        if isinstance(value, dict):
            for key in value:
                if not isinstance(key, str):
                    raise LayaValidationError("JSON object keys must be strings")
                key.encode("utf-8")
            values = value.values()
        else:
            values = value
        for item in values:
            json_value(item, ancestors)
    finally:
        ancestors.remove(id(value))


def questions_value(value: Any) -> None:
    if not isinstance(value, dict) or not value:
        raise LayaValidationError("questions must be a nonempty object")
    for question in value.values():
        if not isinstance(question, dict) or question.get("type") not in ("choice", "score", "noul"):
            raise LayaValidationError("each question must have type choice, score or noul")
        if "instructions" not in question:
            raise LayaValidationError("each question requires instructions")


def expect(condition: bool, field: str) -> None:
    if not condition:
        raise LayaResponseError("Invalid Laya response: " + field)


def batch_prediction(value: Any, questions: dict, count: int) -> None:
    expect(isinstance(value, dict), "batch prediction object")
    results = value.get("results")
    expect(isinstance(results, list), "results")
    expect(len(results) == count, "results count does not match states")
    usage = value.get("total_usage")
    expect(isinstance(usage, dict), "total_usage")
    for key in ("input_tokens", "output_tokens"):
        expect(type(usage.get(key)) is int and usage[key] >= 0, "total_usage." + key)
    for i, result in enumerate(results):
        try:
            prediction(result, questions)
        except LayaResponseError as exc:
            raise LayaResponseError("Invalid Laya response: results[%d]: %s" % (i, exc)) from exc


def prediction(value: Any, questions: dict) -> None:
    expect(isinstance(value, dict), "prediction object")
    expect(isinstance(value.get("model"), str), "model")
    answers, usage = value.get("answers"), value.get("usage")
    expect(isinstance(answers, dict), "answers")
    expect(isinstance(usage, dict), "usage")
    for key in ("input_tokens", "output_tokens"):
        expect(type(usage.get(key)) is int and usage[key] >= 0, "usage." + key)
    if "routing" in value:
        expect(isinstance(value["routing"], dict), "routing")
    for qid, question in questions.items():
        answer = answers.get(qid)
        field = "answers." + qid
        expect(isinstance(answer, dict) and answer.get("type") == question["type"], field + ".type")
        kind = answer["type"]
        for key in ("confidence", "answer_confidence", "x_jev_confidence", "abstention_threshold"):
            if key in answer or (key == "confidence" and kind != "noul"):
                expect(probability(answer.get(key)), field + "." + key)
        if "low_confidence" in answer:
            expect(type(answer["low_confidence"]) is bool, field + ".low_confidence")
        if "abstention" in answer or "abstention_threshold" in answer:
            # The gate reports a state and the threshold that produced it together.
            expect(answer.get("abstention") in ("passed", "abstained", "unevaluated"), field + ".abstention")
            expect(probability(answer.get("abstention_threshold")), field + ".abstention_threshold")
        for key in ("action", "x_tournament"):
            if key in answer:
                expect(isinstance(answer[key], dict), field + "." + key)
        if kind == "noul":
            expect(probability(answer.get("noul")), field + ".noul")
            continue
        probs = answer.get("probabilities")
        expect(isinstance(probs, dict) and bool(probs) and all(probability(p) for p in probs.values()),
               field + ".probabilities")
        criteria = question.get("criteria")
        if kind == "choice":
            expect(isinstance(answer.get("choice"), str) and answer["choice"] in probs, field + ".choice")
            labels = list(criteria) if isinstance(criteria, dict) else criteria
            if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
                expect(answer["choice"] in labels and all(label in probs for label in labels),
                       field + ".choice labels")
        else:
            expect(finite(answer.get("score")) and answer["score"] >= 0, field + ".score")
            expect(isinstance(answer.get("legend"), dict), field + ".legend")
            if isinstance(criteria, list):
                expect(answer["score"] <= len(criteria) - 1, field + ".score")
                expect(all(str(i) in probs and str(i) in answer["legend"] for i in range(len(criteria))),
                       field + ".rubric levels")
