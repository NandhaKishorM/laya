"""Example 32 -- the typed-decisions checkpoint on a real workflow.

`laya-typed-decisions` is fine-tuned on four workflows with fixed question-id signatures. This
example runs the `invoice_processing` workflow end to end, then lets `Router` pick that
checkpoint from the question-id signature alone.
"""
import json

from _common import STATE_EN, banner, describe, device_line, load, router

banner("32", "Typed-decisions workflow", """
    Three checkpoints ship with Laya. The English and multilingual ones are general; the third,
    `laya-typed-decisions`, is fine-tuned on four synthetic workflows whose question ids are
    fixed at training time:

      invoice_processing      {discrepancy_severity, disposition, duplicate, matches_order, urgency}
      customer_service        {action, category, churn_risk, needs_human, urgency}
      security_incidents      {credential_compromise, disposition, severity, true_positive, urgency}
      agent_trace_observability  {action, needs_review, outcome, risk, urgency}

    The question schemas below are the real ones the checkpoint was trained on (dataset
    `LocalLLaMA/typed-decisions`, config `invoice_processing`). One state, five typed decisions,
    one forward pass -- and because the ids are an exact signature, `Router` can recognise the
    workflow without being told which model to use.
    """)

# --- the real invoice_processing question schema -------------------------------------------
INVOICE_QUESTIONS = {
    "disposition": {
        "type": "choice",
        "instructions": "How should this vendor invoice be dispositioned?",
        "criteria": {
            "approve": "Matches the order and delivery; approve for payment.",
            "hold": "Something needs confirming before payment; hold pending clarification.",
            "manual_review": "A human in finance must review the discrepancy.",
            "reject": "Should not be paid: duplicate, unauthorised or materially wrong.",
        },
    },
    "matches_order": {
        "type": "noul",
        "instructions": "The invoice reconciles with the purchase order and the recorded delivery.",
        "criteria": {
            "false": "There is a discrepancy against the order or the delivery.",
            "true": "Line items, quantities and amounts reconcile.",
        },
    },
    "duplicate": {
        "type": "noul",
        "instructions": "This invoice appears to duplicate an invoice already submitted.",
    },
    "discrepancy_severity": {
        "type": "score",
        "instructions": "How material is any discrepancy between the invoice, the order "
                        "and the delivery?",
        "criteria": [
            "None: everything reconciles.",
            "Trivial: rounding or a cosmetic difference.",
            "Moderate: a real difference worth confirming.",
            "Material: a large or unexplained difference.",
        ],
    },
    "urgency": {
        "type": "score",
        "instructions": "How time-sensitive is processing this invoice?",
        "criteria": [
            "No time pressure; can wait indefinitely.",
            "Routine; handle within the normal queue.",
            "Elevated; should be handled within the same week.",
            "Critical; requires action within the same day.",
        ],
    },
}

# --- two real states: one with a 40% price variance, one that reconciles -------------------
VARIANCE = {
    "invoice": {
        "id": "INV-2026-8842", "vendor": "Northwind Castings", "currency": "USD",
        "total_usd": 300020.0,
        "lines": [{"sku": "SKU-940", "qty": 1000, "unit_usd": 300.02, "total_usd": 300020.0}],
    },
    "purchase_order": {
        "id": "PO-3392", "total_usd": 214300.0,
        "lines": [{"qty": 1000, "unit_usd": 214.3}],
        "freight_terms": "freight prepaid by vendor, not separately billable",
    },
    "delivery": {"date": "2026-03-18", "received_qty": 1000, "condition": "accepted with exceptions"},
    "payment": {"terms": "net 45", "status": "overdue", "days_until_due": -3},
    "vendor_history": {
        "invoices_12m": 40, "disputes_12m": 0,
        "prior_invoice_ids": ["INV-2026-6633", "INV-2026-9563", "INV-2026-7939"],
    },
}

CLEAN = {
    "invoice": {
        "id": "INV-2026-9107", "vendor": "Cedar Freight", "currency": "USD",
        "total_usd": 18450.0,
        "lines": [{"sku": "SKU-221", "qty": 300, "unit_usd": 61.5, "total_usd": 18450.0}],
    },
    "purchase_order": {
        "id": "PO-3477", "total_usd": 18450.0,
        "lines": [{"qty": 300, "unit_usd": 61.5}],
        "freight_terms": "freight prepaid by vendor, not separately billable",
    },
    "delivery": {"date": "2026-03-20", "received_qty": 300, "condition": "accepted in full"},
    "payment": {"terms": "net 30", "status": "current", "days_until_due": 18},
    "vendor_history": {"invoices_12m": 12, "disputes_12m": 0,
                       "prior_invoice_ids": ["INV-2026-7781"]},
}

agent = load("typed-decisions")
device_line(agent)
print("   cfg max_len=%d  head_max_len=%d  model_name=%r"
      % (agent.cfg["max_len"], agent.cfg["head_max_len"], agent.cfg["model_name"]))

for name, state in (("VARIANCE (unit price 300.02 vs PO 214.30)", VARIANCE),
                    ("CLEAN (invoice == PO == delivery)", CLEAN)):
    print("\n   == %s ==" % name)
    result = agent.predict(state, INVOICE_QUESTIONS)
    describe(result["answers"])
    disp = result["answers"]["disposition"]
    print("   disposition distribution: %s" % "  ".join(
        "%s=%.3f" % (k, v) for k, v in sorted(disp["probabilities"].items(), key=lambda kv: -kv[1])))
    print("   tokens used: %d in, %d out" % (result["usage"]["input_tokens"],
                                             result["usage"]["output_tokens"]))

# --- Router recognises the workflow from the question-id signature alone -------------------
print("\n   == Router with auto_task_detection=True ==")
r = router(auto_task_detection=True)
decision = r.route(VARIANCE, INVOICE_QUESTIONS)
print("   workflow : %s" % decision["workflow"])
print("   model    : %s" % decision["model"])
print("   reason   : %s" % decision["reason"])

# The agent is already resident, so attach it instead of paying for a duplicate 421M load.
r.attach("typed-decisions", agent)
routed = r.predict(VARIANCE, INVOICE_QUESTIONS)
print("   routed predict used %r; disposition=%s (p=%.3f)"
      % (routed["routing"]["model"], routed["answers"]["disposition"]["choice"],
         max(routed["answers"]["disposition"]["probabilities"].values())))

# A question set that is *not* a workflow signature must not be captured by auto-detection.
other = r.route(STATE_EN, {"department": {"type": "choice", "instructions": "Which team?",
                                          "criteria": {"billing": None, "other": None}}})
print("   a non-workflow id set routes to %r (%s)" % (other["model"], other["reason"]))
print("\n   structured record:\n%s" % json.dumps(
    {"routing": dict(routed["routing"]),
     "disposition": routed["answers"]["disposition"]["choice"],
     "matches_order": routed["answers"]["matches_order"]["noul"],
     "duplicate": routed["answers"]["duplicate"]["noul"]}, indent=2)[:900])
