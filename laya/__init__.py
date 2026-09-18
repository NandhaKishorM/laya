"""Laya: Fast, non-autoregressive System 1 decision engine with calibrated probabilities."""

from . import patterns
from .agent import Agent, RLAgent, load
from .aio import AsyncAgent, RetryPolicy
from .common import (
    QTYPES,
    QTYPE_NAMES,
    confidence_from_probs,
    ece_score,
    proper_reward,
    render_criterion,
    render_options,
    td_lambda_targets,
)
from .email import clean_email_body, email_questions, email_state
from .errors import (
    ConfigurationError,
    DeviceError,
    IncompatibleModelError,
    InferenceError,
    InvalidCriteriaError,
    LayaError,
    ModelNotFoundError,
    OptionsTooLongError,
    QuestionError,
    UnknownQuestionTypeError,
)
from .presets import guard_questions, moderation_questions, router_questions, triage_questions
from .types import (
    Action,
    ChoiceAnswer,
    NoulAnswer,
    Response,
    ScoreAnswer,
    Truncation,
    Usage,
)

__version__ = "0.2.0"
__all__ = [
    "Agent",
    "AsyncAgent",
    "RetryPolicy",
    "RLAgent",
    "load",
    "patterns",
    "Action",
    "ChoiceAnswer",
    "NoulAnswer",
    "ScoreAnswer",
    "Response",
    "Truncation",
    "Usage",
    "LayaError",
    "ConfigurationError",
    "ModelNotFoundError",
    "IncompatibleModelError",
    "QuestionError",
    "UnknownQuestionTypeError",
    "InvalidCriteriaError",
    "OptionsTooLongError",
    "InferenceError",
    "DeviceError",
    "clean_email_body",
    "email_questions",
    "email_state",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "proper_reward",
    "td_lambda_targets",
    "ece_score",
    "confidence_from_probs",
    "render_options",
    "render_criterion",
    "QTYPES",
    "QTYPE_NAMES",
    "__version__",
]
