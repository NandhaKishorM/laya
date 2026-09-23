"""Laya: Fast, non-autoregressive System 1 decision engine with calibrated probabilities."""

from .email import clean_email_body, email_state
from .hooks import Hook, PredictContext, PredictHook
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .router import DEFAULT_MODELS, RouteDecision, Router

__version__ = "0.3.10"

# Routing, language detection and email cleaning are pure Python. The torch-backed names are
# resolved lazily so that `import laya` -- and therefore `from laya import Router` or
# `from laya.lang import detect_script` -- does not pay torch's import time and memory.
_LAZY_ATTRS = {
    "Agent": (".agent", "Agent"),
    "RLAgent": (".agent", "RLAgent"),
    "load": (".agent", "load"),
    "proper_reward": (".common", "proper_reward"),
    "td_lambda_targets": (".common", "td_lambda_targets"),
    "ece_score": (".common", "ece_score"),
    "confidence_from_probs": (".common", "confidence_from_probs"),
    "render_options": (".common", "render_options"),
    "QTYPES": (".common", "QTYPES"),
    "QTYPE_NAMES": (".common", "QTYPE_NAMES"),
    "shortlist_choice": (".shortlist", "shortlist_choice"),
    "predict_shortlist": (".shortlist", "predict_shortlist"),
    "embed_fn_from_agent": (".shortlist", "embed_fn_from_agent"),
    "LayaRouter": (".integrations", "LayaRouter"),
    "LayaGuardrail": (".integrations", "LayaGuardrail"),
    "LayaGuardrailError": (".integrations", "LayaGuardrailError"),
    "LayaTriage": (".integrations", "LayaTriage"),
    "LayaEvaluator": (".integrations", "LayaEvaluator"),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name)) from None
    import importlib

    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value      # cache: __getattr__ runs at most once per name
    return value


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))


__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    "shortlist_choice",
    "predict_shortlist",
    "embed_fn_from_agent",
    "detect_language",
    "detect_script",
    "is_english",
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
    "LayaRouter",
    "LayaGuardrail",
    "LayaGuardrailError",
    "LayaTriage",
    "LayaEvaluator",
    "PredictContext",
    "PredictHook",
    "Hook",
    "__version__",
]
