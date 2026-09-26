"""Schema-driven decisions: turn a JSON schema (or a pydantic model) into Laya questions.

Pure Python and torch-free, so `import laya` stays light. The mapping is a documented subset:
an object of properties, each an enum choice, a boolean, or a bounded integer scale. Anything
that cannot be answered from a fixed option set (free strings, arrays, nested objects) is
rejected with an error that names the path.

    import laya

    class Ticket(BaseModel):
        department: Literal["billing", "support", "sales"]
        urgency: Literal[0, 1, 2]
        needs_human: bool

    values = laya.decide(agent, state, schema=Ticket)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

MAX_PROPERTIES = 32
MAX_OPTIONS = 32
MAX_SCORE_LEVELS = 10


class SchemaError(ValueError):
    """A schema cannot be expressed as Laya questions; the message names the path."""


@dataclass
class DecisionResult:
    """The detailed result of `decide(..., return_details=True)`.

    `values` is the schema-shaped output. `confidence` and `probabilities` are keyed by field,
    and `answers` is Laya's raw answer per field.
    """

    values: Dict[str, Any]
    confidence: Dict[str, float]
    probabilities: Dict[str, Dict[str, Any]]
    answers: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None
    routing: Optional[Dict[str, Any]] = None


@dataclass
class _Field:
    name: str
    kind: str                       # "choice" | "score" | "noul"
    question: Dict[str, Any]
    options: List[Tuple[str, Any]] = field(default_factory=list)   # choice: (label, value)
    minimum: Optional[int] = None                                  # score: level 0 value


def _require_pydantic():
    try:
        import pydantic  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "pydantic is required for the typed-model helpers; install 'laya[structured]'"
        ) from exc


def _schema_of(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_json_schema"):        # pydantic v2
        return model.model_json_schema()
    if hasattr(model, "schema"):                   # pydantic v1
        return model.schema()
    if isinstance(model, dict):
        return model
    raise SchemaError("expected a JSON schema dict or a pydantic model, got %s" % type(model).__name__)


def _enum_field(path: str, name: str, values: Sequence[Any], description: Optional[str]) -> _Field:
    if len(values) > MAX_OPTIONS:
        raise SchemaError("%s: %d options exceeds MAX_OPTIONS=%d" % (path, len(values), MAX_OPTIONS))
    if not values:
        raise SchemaError("%s: 'enum' must not be empty" % path)
    if all(isinstance(v, bool) for v in values):
        return _noul_field(path, name, description)
    options = [(("null" if v is None else str(v)), v) for v in values]
    if len({label for label, _ in options}) != len(options):
        raise SchemaError("%s: enum values produce duplicate choice labels" % path)
    criteria = {label: None for label, _ in options}
    question = {
        "type": "choice",
        "instructions": description or ("What is `%s`?" % name),
        "criteria": criteria,
    }
    return _Field(name=name, kind="choice", question=question, options=options)


def _noul_field(path: str, name: str, description: Optional[str]) -> _Field:
    question = {
        "type": "noul",
        "instructions": description or ("Is `%s` true?" % name),
    }
    return _Field(name=name, kind="noul", question=question)


def _score_field(path: str, name: str, prop: Dict[str, Any],
                 description: Optional[str]) -> _Field:
    lo, hi = prop.get("minimum"), prop.get("maximum")
    if not isinstance(lo, int) or not isinstance(hi, int):
        raise SchemaError(
            "%s: a numeric field needs integer 'minimum' and 'maximum' to become a score" % path)
    if hi < lo:
        raise SchemaError("%s: 'maximum' %d is below 'minimum' %d" % (path, hi, lo))
    span = hi - lo + 1
    if span > MAX_SCORE_LEVELS:
        raise SchemaError(
            "%s: %d levels exceeds MAX_SCORE_LEVELS=%d; narrow the range or use an enum"
            % (path, span, MAX_SCORE_LEVELS))
    question = {
        "type": "score",
        "instructions": description or ("Score `%s` from %d to %d" % (name, lo, hi)),
        "criteria": [str(v) for v in range(lo, hi + 1)],
    }
    return _Field(name=name, kind="score", question=question, minimum=lo)


def _field(path: str, name: str, prop: Dict[str, Any]) -> _Field:
    if not isinstance(prop, dict):
        raise SchemaError("%s: property must be an object, got %s" % (path, type(prop).__name__))
    description = prop.get("description")
    # Pydantic v2 renders `Optional[X]` as `{"anyOf": [<X>, {"type": "null"}]}` with no
    # top-level type/enum/const, the same nullable shape the list form `type: ["string", "null"]`
    # already handles below. Unwrap the single non-null branch (carrying the outer description)
    # so `Optional[Literal[...]]`, `Optional[int]` and friends map instead of raising. A union of
    # two real types is genuinely ambiguous and still rejected.
    if not ({"const", "enum", "type"} & set(prop)):
        union = prop.get("anyOf") or prop.get("oneOf")
        if union is not None:
            branches = [b for b in union if isinstance(b, dict) and b.get("type") != "null"]
            if len(branches) != 1:
                raise SchemaError(
                    "%s: only 'Optional[...]' unions (one non-null branch) are supported, got %d"
                    % (path, len(branches)))
            branch = dict(branches[0])
            branch.setdefault("description", description)
            return _field(path, name, branch)
    if "const" in prop:
        return _enum_field(path, name, [prop["const"]], description)
    if "enum" in prop:
        return _enum_field(path, name, prop["enum"], description)

    jtype = prop.get("type")
    if isinstance(jtype, list):                # nullable: ["string", "null"]
        non_null_types = [t for t in jtype if t != "null"]
        if len(non_null_types) > 1:
            raise SchemaError("%s: 'type' has multiple non-null types; unions are not supported" % path)
        jtype = non_null_types[0] if non_null_types else None
    if jtype == "boolean":
        return _noul_field(path, name, description)
    if jtype == "string":
        raise SchemaError(
            "%s: a free string cannot be a fixed option set; use 'enum' or a boolean" % path)
    if jtype in ("integer", "number"):
        return _score_field(path, name, prop, description)
    if jtype == "array":
        raise SchemaError("%s: arrays are not supported; ask one field per element" % path)
    if jtype == "object":
        raise SchemaError("%s: nested objects are not supported; flatten the schema" % path)
    if "$ref" in prop:
        raise SchemaError("%s: $ref/recursion is not supported; flatten the schema" % path)
    raise SchemaError("%s: unsupported schema %r" % (path, prop))


def plan_from_json_schema(schema: Dict[str, Any]) -> List[_Field]:
    """Validate a JSON schema and return one planned field per property."""
    if not isinstance(schema, dict):
        raise SchemaError("expected a JSON schema object, got %s" % type(schema).__name__)
    if schema.get("type") not in (None, "object") or "properties" not in schema:
        raise SchemaError("the top level must be an object with 'properties'")
    properties = schema["properties"]
    if not isinstance(properties, dict) or not properties:
        raise SchemaError("'properties' must be a non-empty object")
    if len(properties) > MAX_PROPERTIES:
        raise SchemaError("%d properties exceeds MAX_PROPERTIES=%d" % (len(properties), MAX_PROPERTIES))
    return [_field("properties.%s" % name, name, prop) for name, prop in properties.items()]


def questions_from_json_schema(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Turn a JSON schema into Laya questions (a documented subset; see the module docstring)."""
    return {f.name: f.question for f in plan_from_json_schema(schema)}


def questions_from_pydantic(model: Any) -> Dict[str, Dict[str, Any]]:
    """Turn a pydantic model into Laya questions. Requires pydantic (the `structured` extra)."""
    _require_pydantic()
    return questions_from_json_schema(_schema_of(model))


def _project(answers: Dict[str, Any], fields: Sequence[_Field]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for f in fields:
        answer = answers.get(f.name)
        if answer is None:
            continue
        if f.kind == "noul":
            values[f.name] = bool(float(answer.get("noul", 0.0)) >= 0.5)
        elif f.kind == "score":
            probs = answer.get("probabilities") or {}
            if probs:
                idx = max(range(len(probs)), key=lambda i: float(probs.get(str(i), probs.get(i, 0.0))))
            else:
                idx = int(round(float(answer.get("score", 0.0)))) - int(f.minimum or 0)
            values[f.name] = int(f.minimum or 0) + idx
        else:  # choice
            label = str(answer.get("choice"))
            values[f.name] = next((value for lbl, value in f.options if lbl == label), label)
    return values


def answers_to_json(answers: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Project Laya answers onto the schema values (choice value, integer level, boolean)."""
    return _project(answers, plan_from_json_schema(schema))


def answer_to_pydantic(model: Any, answers: Dict[str, Any]) -> Any:
    """Project Laya answers into a pydantic model instance. Requires pydantic."""
    _require_pydantic()
    return model(**answers_to_json(answers, _schema_of(model)))


def _details(values: Dict[str, Any], answers: Dict[str, Any], result: Dict[str, Any]) -> DecisionResult:
    confidence: Dict[str, float] = {}
    probabilities: Dict[str, Dict[str, Any]] = {}
    for name, answer in answers.items():
        confidence[name] = float(answer.get("confidence", 0.0))
        if answer.get("type") == "noul":
            p = float(answer.get("noul", 0.0))
            probabilities[name] = {"false": round(1.0 - p, 4), "true": round(p, 4)}
        else:
            probabilities[name] = dict(answer.get("probabilities") or {})
    return DecisionResult(
        values=values,
        confidence=confidence,
        probabilities=probabilities,
        answers=dict(answers),
        usage=result.get("usage"),
        routing=result.get("routing"),
    )


def decide(runner, state: Any, schema: Any = None, *, questions: Optional[Dict[str, Any]] = None,
           return_details: bool = False, **predict_kwargs) -> Any:
    """Answer `state` against a schema (or explicit questions) and return the decided values.

    Pass exactly one of `schema` or `questions`. With `schema`, the values follow the schema
    (choice values, integer levels, booleans). With `questions`, the raw answers are returned.
    `return_details=True` returns a `DecisionResult` instead of the plain values. Extra keyword
    arguments are forwarded to `runner.predict`.
    """
    if (schema is None) == (questions is None):
        raise ValueError("pass exactly one of schema= or questions=")

    fields: Optional[List[_Field]] = None
    if schema is not None:
        fields = plan_from_json_schema(_schema_of(schema))
        questions = {f.name: f.question for f in fields}

    result = runner.predict(state, questions, **predict_kwargs)
    answers = result.get("answers", {}) or {}
    values = _project(answers, fields) if fields is not None else dict(answers)
    if return_details:
        return _details(values, answers, result)
    return values


__all__ = [
    "DecisionResult",
    "SchemaError",
    "answers_to_json",
    "answer_to_pydantic",
    "decide",
    "plan_from_json_schema",
    "questions_from_json_schema",
    "questions_from_pydantic",
]
