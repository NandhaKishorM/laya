"""
Pydantic v2 schemas for Laya's decision primitives.

These models cover both the API request / response layer and can be used
as typed return values when calling Agent.system_one() directly in Python.

Usage (direct)::

    import laya
    from laya.schemas import DecisionResponse

    agent = laya.load("convaiinnovations/laya")
    raw    = agent.predict(state, questions)
    result = DecisionResponse.model_validate(raw)

    dept   = result.answers["department"]          # typed as ChoiceAnswer
    conf   = result.answers["department"].confidence

Usage (PydanticAI-style)::

    from laya.schemas import DecideRequest, DecisionResponse

    req  = DecideRequest(state=email_dict, questions=laya.email_questions())
    resp = DecisionResponse.model_validate(agent.predict(req.state, req.questions))
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Question-definition models (request side)
# ---------------------------------------------------------------------------


class ChoiceQuestion(BaseModel):
    """Categorical selection question — returns the best label + full probability distribution."""

    type: Literal["choice"]
    instructions: str = Field(..., description="Natural-language question to ask about the state.")
    criteria: Dict[str, Optional[str]] = Field(
        ...,
        min_length=1,
        description=(
            "Mapping of label → optional description. "
            "Example: {'billing': 'invoices and refunds', 'technical': 'bugs and outages'}"
        ),
    )
    model_config = {"extra": "ignore"}

    @field_validator("criteria", mode="before")
    @classmethod
    def _normalize_criteria_list(cls, v: Any) -> Any:
        if isinstance(v, list):
            return {str(item): None for item in v}
        return v

class ScoreQuestion(BaseModel):
    """Ordinal scoring question — returns an expected value on a rubric scale."""

    type: Literal["score"]
    instructions: str = Field(..., description="Natural-language question to ask about the state.")
    criteria: List[str] = Field(
        ...,
        min_length=2,
        description=(
            "Ordered rubric levels from lowest (index 0) to highest. "
            "Example: ['calm', 'concerned', 'very angry']"
        ),
    )

    model_config = {"extra": "ignore"}


class NoulCriteria(BaseModel):
    """Optional custom labels for the false/true poles of a noul question."""

    false: Optional[str] = Field(None, description="Label for the 'false' outcome.")
    true: Optional[str] = Field(None, description="Label for the 'true' outcome.")


class NoulQuestion(BaseModel):
    """Boolean probability question — returns calibrated P(true) ∈ [0, 1]."""

    type: Literal["noul"]
    instructions: str = Field(..., description="Yes/no statement to evaluate against the state.")
    criteria: Optional[NoulCriteria] = Field(
        None,
        description="Optional custom pole labels. Defaults to 'no, the statement does not hold' / 'yes, the statement holds'.",
    )
    model_config = {"extra": "ignore"}

    @field_validator("criteria", mode="before")
    @classmethod
    def _normalize_criteria_list(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) == 2:
            return {"false": str(v[0]), "true": str(v[1])}
        return v

# Discriminated union — Pydantic uses the `type` field to pick the right model.
AnyQuestion = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, NoulQuestion],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Answer models (response side)
# ---------------------------------------------------------------------------


class ActionResult(BaseModel):
    """Meta-signal from the act_head: probability that an automated action is safe to take."""

    act_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P(act) — confidence that the model's answer warrants an automated downstream action.",
    )


class ChoiceAnswer(BaseModel):
    """Result for a `choice` question."""

    type: Literal["choice"]
    choice: str = Field(..., description="The winning label.")
    probabilities: Dict[str, float] = Field(..., description="Softmax probability for every candidate label.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized Shannon-entropy confidence: 1 − H(p)/log(K). Calibrated — treat 0.85 as ~85% accurate.",
    )
    action: ActionResult

    model_config = {"extra": "ignore"}


class ScoreAnswer(BaseModel):
    """Result for a `score` question."""

    type: Literal["score"]
    score: float = Field(..., description="Expected ordinal level E[i·p_i]. Range: [0, K−1].")
    legend: Dict[str, str] = Field(..., description="Mapping of level index (as string) → rubric label.")
    probabilities: Dict[str, float] = Field(..., description="Softmax probability for every level index.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Normalized entropy confidence.")
    action: ActionResult

    model_config = {"extra": "ignore"}


class NoulAnswer(BaseModel):
    """Result for a `noul` (boolean probability) question."""

    type: Literal["noul"]
    noul: float = Field(..., ge=0.0, le=1.0, description="Calibrated P(true) ∈ [0, 1].")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="max(P(true), 1−P(true)) — how far the model is from the 0.5 decision boundary.",
    )
    action: ActionResult

    model_config = {"extra": "ignore"}


# Discriminated union for answers.
AnyAnswer = Annotated[
    Union[ChoiceAnswer, ScoreAnswer, NoulAnswer],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Usage / token info
# ---------------------------------------------------------------------------


class UsageInfo(BaseModel):
    """Token usage statistics for the forward pass."""

    input_tokens: int = Field(..., ge=0, description="Number of non-padding input tokens processed.")
    output_tokens: int = Field(0, description="Always 0 for Laya (non-autoregressive, no generated tokens).")


# ---------------------------------------------------------------------------
# Top-level request / response
# ---------------------------------------------------------------------------


class DecideRequest(BaseModel):
    """
    Request body for ``POST /v1/decide``.

    Examples
    --------
    .. code-block:: python

        DecideRequest(
            state={"from": "user@acme.com", "body": "I was charged twice!"},
            questions={
                "department": ChoiceQuestion(
                    type="choice",
                    instructions="Which department should handle this?",
                    criteria={"billing": "invoices/refunds", "technical": "bugs"},
                ),
                "is_urgent": NoulQuestion(
                    type="noul",
                    instructions="Does the message communicate time pressure?",
                ),
            },
        )
    """

    model: str = Field(
        "english",
        description="HuggingFace model ID or model alias (e.g. 'english', 'multilingual', 'convaiinnovations/laya').",
    )
    state: Union[str, Dict[str, Any], List[Any]] = Field(
        ...,
        description="The input to evaluate — any text string, JSON dict, or conversation turn list.",
    )
    questions: Dict[str, AnyQuestion] = Field(
        ...,
        min_length=1,
        description="Named questions to evaluate in a single parallel forward pass.",
    )

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _check_questions_not_empty(self) -> "DecideRequest":
        if not self.questions:
            raise ValueError("`questions` must contain at least one entry.")
        return self


class DecisionResponse(BaseModel):
    """
    Response body for ``POST /v1/decide``.

    The ``answers`` dict maps each question ID to a typed answer.
    Use ``.model_dump()`` to serialize, or access fields directly::

        resp.answers["department"].choice        # "billing"
        resp.answers["department"].confidence    # 0.94
        resp.answers["is_urgent"].noul           # 0.87
    """

    model: str = Field(..., description="Model identifier that produced these answers.")
    answers: Dict[str, AnyAnswer] = Field(..., description="Per-question typed answers.")
    usage: UsageInfo

    model_config = {"extra": "ignore"}

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "DecisionResponse":
        """
        Convenience constructor: validate raw dict returned by ``Agent.system_one()``.

        .. code-block:: python

            raw    = agent.predict(state, questions)
            result = DecisionResponse.from_raw(raw)
        """
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Structured error detail returned by the server on 4xx / 5xx responses."""

    code: str = Field(..., description="Machine-readable error code, e.g. 'invalid_question_type'.")
    message: str = Field(..., description="Human-readable error message.")
    param: Optional[str] = Field(None, description="The request parameter that caused the error, if applicable.")


class ErrorResponse(BaseModel):
    """Top-level error envelope (mirrors OpenAI error shape)."""

    error: ErrorDetail

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Batch variants
# ---------------------------------------------------------------------------


class BatchDecideRequest(BaseModel):
    """
    Request body for ``POST /v1/decide/batch``.

    Evaluates the *same* question set over multiple states in one call.
    Each state gets its own entry in the response ``results`` list.
    """

    model: str = Field("english")
    states: List[Union[str, Dict[str, Any], List[Any]]] = Field(
        ...,
        min_length=1,
        max_length=256,
        description="List of states to evaluate. Max 256 per request.",
    )
    questions: Dict[str, AnyQuestion] = Field(..., min_length=1)

    model_config = {"extra": "ignore"}


class BatchDecisionResponse(BaseModel):
    """Response body for ``POST /v1/decide/batch``."""

    model: str
    results: List[DecisionResponse] = Field(..., description="One DecisionResponse per input state, in order.")
    total_usage: UsageInfo = Field(..., description="Aggregated token usage across all states.")

    model_config = {"extra": "ignore"}
