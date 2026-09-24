"""Audit every decision: log it, and optionally ship it to an external service.

Run from the repository root:

    python examples/hooks/audit.py
"""
import json

import laya


def on_predict_end(ctx):
    record = {
        "model": ctx.model,
        "answers": ctx.results[0]["answers"] if ctx.results else None,
        "routing": ctx.results[0].get("routing") if ctx.results else None,
        "usage": ctx.usage,
        "elapsed_ms": round(ctx.elapsed_ms or 0.0, 2),
    }
    print(json.dumps(record, indent=2))
    # Ship it to an external service if you want:
    #   import requests
    #   requests.post("https://example.invalid/decisions", json=record, timeout=2)


QUESTIONS = {
    "dept": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "invoices and payments", "support": "product help"},
    }
}


# Direct Agent use.
agent = laya.load("convaiinnovations/laya", on_predict_end=on_predict_end)
agent.system_one("I was charged twice for the same invoice.", QUESTIONS)

# Router use: the hook also sees ctx.decision (which checkpoint was chosen).
from laya import Router  # noqa: E402

router = Router(on_predict_end=on_predict_end)
router.predict("I was charged twice for the same invoice.", QUESTIONS)
