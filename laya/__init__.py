"""Laya: Fast, non-autoregressive System 1 decision engine with calibrated probabilities."""

from .agent import Agent, RLAgent, load
from .client import (
    AsyncLayaClient,
    LayaClient,
    decide,
    judge,
    rate,
    system_one,
)
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

__version__ = "0.1.6"
__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "LayaClient",
    "AsyncLayaClient",
    "system_one",
    "decide",
    "judge",
    "rate",
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
    "QTYPES",
    "QTYPE_NAMES",
    "__version__",
]
