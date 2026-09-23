"""Redact PII from the state before the model ever sees it.

A start hook can rewrite `ctx.states`; the rewrite is what gets tokenized.

    python examples/hooks/redact.py
"""
import re

import laya

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"\+?\d[\d ()-]{7,}\d")


def scrub(value):
    if isinstance(value, str):
        return PHONE.sub("[phone]", EMAIL.sub("[email]", value))
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def redact(ctx):
    ctx.states = [scrub(state) for state in ctx.states]


agent = laya.load("convaiinnovations/laya", on_predict_start=redact)
result = agent.system_one(
    "Email jane@example.com or call +1 555 010 9999 about invoice 42.",
    {"urgent": {"type": "noul", "instructions": "Is this urgent?"}},
)
print(result["answers"])
