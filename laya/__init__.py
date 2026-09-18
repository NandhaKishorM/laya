"""Laya: Fast, non-autoregressive System 1 decision engine with calibrated probabilities."""

from .agent import Agent, RLAgent, load
from .common import (
    QTYPES,
    QTYPE_NAMES,
    confidence_from_probs,
    ece_score,
    proper_reward,
    render_options,
    td_lambda_targets,
)
from .email import clean_email_body, email_questions, email_state
from .presets import guard_questions, moderation_questions, router_questions, triage_questions
from .schemas import (
    AnyAnswer,
    AnyQuestion,
    BatchDecideRequest,
    BatchDecisionResponse,
    ChoiceAnswer,
    ChoiceQuestion,
    DecideRequest,
    DecisionResponse,
    ErrorDetail,
    ErrorResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    UsageInfo,
)

__version__ = "0.1.7"
__all__ = [
    # Core agent
    "Agent",
    "RLAgent",
    "load",
    # Email helpers
    "clean_email_body",
    "email_questions",
    "email_state",
    # Presets
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    # Training utilities
    "proper_reward",
    "td_lambda_targets",
    "ece_score",
    "confidence_from_probs",
    "render_options",
    "QTYPES",
    "QTYPE_NAMES",
    # Pydantic schemas (request)
    "DecideRequest",
    "BatchDecideRequest",
    "ChoiceQuestion",
    "ScoreQuestion",
    "NoulQuestion",
    "AnyQuestion",
    # Pydantic schemas (response)
    "DecisionResponse",
    "BatchDecisionResponse",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NoulAnswer",
    "AnyAnswer",
    "UsageInfo",
    # Error models
    "ErrorDetail",
    "ErrorResponse",
    # Version
    "__version__",
]

