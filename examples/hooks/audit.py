"""Audit every decision: log it, and optionally ship it to an external service.

A hook fires once per call, and one call can carry many states, so the audit trail is written per
decision: `ctx.states` and `ctx.results` are aligned by index.

Run from the repository root:

    python examples/hooks/audit.py
"""
import json

import laya


def on_predict_end(ctx):
    for state, result in zip(ctx.states, ctx.results or []):
        record = {
            "run_id": ctx.run_id,
            "model": ctx.model,
            "state": state,
            "answers": result["answers"],
            "routing": result.get("routing"),
            "usage": result.get("usage"),
            "call_usage": ctx.usage,
            "call_elapsed_ms": round(ctx.elapsed_ms or 0.0, 2),
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

# One call, several states: the hook still fires once, and writes one record per decision.
agent.predict_batch(
    [
        "The app crashes every time I open the export screen.",
        "Where do I change my notification settings?",
    ],
    QUESTIONS,
)

# Router use: the hook also sees ctx.decision (which checkpoint was chosen).
from laya import Router  # noqa: E402

router = Router(on_predict_end=on_predict_end)
router.predict("I was charged twice for the same invoice.", QUESTIONS)
