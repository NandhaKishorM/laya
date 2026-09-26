"""Example 38 -- composing three checkpoints in one pipeline.

Routes each incoming item, answers a general triage set on the language-appropriate
checkpoint, then hands the same state to `typed-decisions` for a specialist action.
"""

from _common import laya, banner, heading, router

banner("38", "A multi-checkpoint pipeline", """
    One `predict()` is one checkpoint answering one question set. Real traffic often needs
    two decisions in sequence, and the right checkpoint changes between them.

    This pipeline has three stages:

      1. route       -- `Router` inspects the state and the question ids, cheaply;
      2. general     -- the `triage_questions()` preset runs on `english` or
                        `multilingual`, whichever the item's language selects;
      3. specialist  -- the same state goes to `typed-decisions`, which `Router` picks
                        because the question-id set is exactly the `customer_service`
                        workflow signature from `laya/router.py`.

    Stage 2 answers "what is this and how bad is it"; stage 3 answers "what should the agent
    do". Two items, one English and one German, so the general stage switches checkpoints
    while the specialist stage does not.
    """)

GENERAL = laya.triage_questions()

# These five ids are the real `customer_service` signature from `laya/router.py`. Router
# matches on the id set alone, which is what selects the fine-tuned checkpoint; the wording
# below is ours.
SPECIALIST = {
    "action": {
        "type": "choice",
        "instructions": "What action should the support team take on this ticket?",
        "criteria": {
            "auto_resolve": "Answer directly; no human needs to touch it.",
            "route_to_agent": "Send it to a support agent in the right queue.",
            "escalate": "Escalate now: urgent, at risk, or beyond frontline policy.",
        },
    },
    "category": {
        "type": "choice",
        "instructions": "Which category best describes the ticket?",
        "criteria": {
            "billing": "charges, invoices, refunds, payments",
            "technical": "bugs, outages, integrations, login failures",
            "account": "profile, seats, permissions, cancellation",
            "other": "none of the other options fits",
        },
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the ticket suggest the customer may leave or cancel?",
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Does the ticket need a human agent rather than an automated reply?",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is the ticket?",
        "criteria": ["no time pressure", "routine", "elevated, same week", "critical, same day"],
    },
}

ITEMS = [
    ("TCK-3001", "en", {
        "ticket_id": "TCK-3001",
        "channel": "email",
        "message": "Our production database keeps dropping connections since this morning and "
                   "the whole team is blocked. We were also billed twice this month. Someone "
                   "needs to fix this today or we will cancel.",
    }),
    ("TCK-3002", "de", {
        "ticket_id": "TCK-3002",
        "channel": "email",
        "message": "Der Kunde wurde zweimal belastet und moechte eine Rueckzahlung des "
                   "doppelten Betrags. Zusaetzlich funktioniert die Anmeldung seit gestern "
                   "nicht mehr.",
    }),
]

# max_loaded=3 keeps exactly the three checkpoints this pipeline needs resident at once.
r = router(max_loaded=3, auto_task_detection=True)
print("   router: %r" % r)


def show(answers, indent="   "):
    """One compact line per answer -- top choice only, so the row stays narrow."""
    for qid, answer in answers.items():
        if answer["type"] == "choice":
            best = max(answer["probabilities"].values())
            detail = "%s (p=%.3f, conf=%.3f)" % (answer["choice"], best, answer["confidence"])
        elif answer["type"] == "score":
            detail = "%.2f / %d (conf=%.3f)" % (answer["score"],
                                                len(answer["probabilities"]) - 1,
                                                answer["confidence"])
        else:
            detail = "%.3f (%s)  conf=%.3f" % (answer["noul"],
                                               "true" if answer["noul"] > 0.5 else "false",
                                               answer["confidence"])
        print("%s%-18s %-7s %s" % (indent, qid, answer["type"], detail))


for ticket_id, lang, state in ITEMS:
    heading("%s [%s]" % (ticket_id, lang))

    general = r.predict(state, GENERAL)
    print("   general  -> %-14s reason: %s"
          % (general["routing"]["model"], general["routing"]["reason"]))
    show(general["answers"])

    specialist = r.predict(state, SPECIALIST)
    print("   specialist -> %-12s workflow=%s" % (specialist["routing"]["model"],
                                                  specialist["routing"]["workflow"]))
    print("                reason: %s" % specialist["routing"]["reason"])
    show(specialist["answers"])
    answers = specialist["answers"]
    print("   DECISION: action=%s  category=%s  urgency=%.2f  needs_human=%.3f"
          % (answers["action"]["choice"], answers["category"]["choice"],
             answers["urgency"]["score"], answers["needs_human"]["noul"]))

print("\n   checkpoints resident: %s" % r.loaded)
print("""
   Three checkpoints, two stages, one state per item. `english` and `multilingual` split
   the general stage by language, and `typed-decisions` answers the specialist stage for
   both. The routing reasons above are the whole argument for the split: the general stage
   is chosen by the item's language, the specialist stage by the question-id signature, and
   neither decision costs a forward pass.

   One honest limit: the specialist stage ran on `typed-decisions`, not on the multilingual
   checkpoint, so for the German item it answered across languages -- and the soft stage-3
   confidence above is the model saying so. In a real service, translate non-English items
   or hand them to a human before this stage.

   `max_loaded=3` is exactly what this pipeline needs. A service that only sees English
   could drop `multilingual`; one that alternates languages at `max_loaded=1` would reload
   on every request (example 24 shows the trade).
   """)
