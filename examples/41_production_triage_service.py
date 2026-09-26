"""Example 41 -- a small production triage service built from Laya.

Routes each incoming ticket with `Router`, answers a triage question set in one forward pass, and
turns the probabilities into an operational decision with an explicit application policy. Four
tickets (English, German, Hindi) so the policy visibly branches.
"""
import json

from _common import laya, banner, describe, router

banner("41", "Production triage service", """
    This is the capstone: everything the earlier examples showed, wired into one call path.

      1. `Router` picks the checkpoint from the ticket's script/language -- English Latin text
         goes to `english`, Devanagari and non-English Latin go to `multilingual`;
      2. one `predict()` answers a triage question set (intent, urgency, frustration, refund,
         churn) in a single forward pass;
      3. an application policy maps probabilities to an action: auto-respond, route to a team,
         or escalate to a human.

    The split matters. The model returns probabilities; the **policy** is ours -- the thresholds,
    the team mapping and the decision to escalate are application code, not model output. Read
    the JSON record as: `routing` and `answers` come from Laya, `decision` comes from the policy.
    """)

# --- application policy: thresholds and routing rules (ours, not the model's) --------------
CHURN_ESCALATE = 0.50
URGENT_ESCALATE = 0.50
INTENT_CONFIDENT = 0.35          # below this the intent itself is too soft to auto-answer

def decide(answers):
    """Map triage answers to an action. Everything here is application policy."""
    intent = answers["intent"]["choice"]
    intent_conf = answers["intent"]["answer_confidence"]
    churn = answers["churn_risk"]["noul"]
    urgent = answers["is_urgent"]["noul"]
    frustration = answers["frustration"]["score"]
    refund = answers["refund_requested"]["noul"]
    reasons = []

    if churn >= CHURN_ESCALATE:
        reasons.append("churn_risk=%.2f >= %.2f" % (churn, CHURN_ESCALATE))
        return "escalate_to_human", "retention", True, reasons
    if urgent >= URGENT_ESCALATE and intent in ("technical_help", "cancellation"):
        reasons.append("is_urgent=%.2f on %s" % (urgent, intent))
        return "escalate_to_human", "engineering_oncall", True, reasons
    if intent == "cancellation":
        reasons.append("intent=cancellation")
        return "escalate_to_human", "retention", True, reasons
    if intent in ("refund", "billing_question") or refund >= 0.5:
        reasons.append("intent=%s refund=%.2f" % (intent, refund))
        return "route_to_team", "billing", False, reasons
    if intent == "technical_help":
        reasons.append("intent=technical_help")
        return "route_to_team", "engineering", False, reasons
    if intent == "information" and intent_conf >= INTENT_CONFIDENT:
        reasons.append("intent=information answer_confidence=%.2f >= %.2f"
                       % (intent_conf, INTENT_CONFIDENT))
        return "auto_respond", "self_service", False, reasons
    if frustration >= 2.5:
        reasons.append("frustration=%.2f" % frustration)
        return "route_to_team", "senior_support", False, reasons
    reasons.append("fallback: intent=%s answer_confidence=%.2f" % (intent, intent_conf))
    return "route_to_team", "general_support", False, reasons

# --- four tickets, one per branch -----------------------------------------------------------
TICKETS = [
    {"ticket_id": "TCK-1001", "language": "en",
     "message": "We were billed twice and nobody has replied in three days. We have decided to "
                "cancel our plan and move to a competitor unless this is refunded today."},
    {"ticket_id": "TCK-1002", "language": "en",
     "message": "Can you explain the difference between the Pro and Enterprise plans? "
                "We are comparing options for next quarter."},
    {"ticket_id": "TCK-1003", "language": "de",
     "message": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung des "
                "doppelten Betrags auf die urspruengliche Zahlungsmethode."},
    {"ticket_id": "TCK-1004", "language": "hi",
     "message": "हमारा डैशबोर्ड कल रिलीज़ के बाद से लोड नहीं हो रहा है और पूरी टीम ब्लॉक है, "
                "कृपया आज ही ठीक करें।"},
]

QUESTIONS = laya.triage_questions()
r = router(max_loaded=2)          # english + multilingual stay resident after first use

records = []
for ticket in TICKETS:
    state = {"ticket_id": ticket["ticket_id"], "language": ticket["language"],
             "message": ticket["message"]}
    result = r.predict(state, QUESTIONS)
    answers = result["answers"]
    action, team, human, reasons = decide(answers)
    record = {
        "ticket_id": ticket["ticket_id"],
        "language": ticket["language"],
        "routing": {"model": result["routing"]["model"], "reason": result["routing"]["reason"]},
        "answers": {
            "intent": {"choice": answers["intent"]["choice"],
                       "answer_confidence": answers["intent"]["answer_confidence"]},
            "is_urgent": {"noul": answers["is_urgent"]["noul"]},
            "frustration": {"score": answers["frustration"]["score"]},
            "refund_requested": {"noul": answers["refund_requested"]["noul"]},
            "churn_risk": {"noul": answers["churn_risk"]["noul"]},
        },
        "decision": {"action": action, "team": team, "human_review": human, "reasons": reasons},
    }
    records.append(record)
    print("\n   == %s [%s] -> %s ==" % (ticket["ticket_id"], ticket["language"],
                                        result["routing"]["model"]))
    print("   routing reason: %s" % result["routing"]["reason"])
    describe(answers)
    print("   DECISION: %s -> %s (human_review=%s)   because %s"
          % (action, team, human, "; ".join(reasons)))

print("\n   == structured decision records ==")
for record in records:
    print(json.dumps(record, indent=2, ensure_ascii=False))

print("\n   == action mix ==")
for record in records:
    print("   %s  %-18s %-18s human_review=%s"
          % (record["ticket_id"], record["answers"]["intent"]["choice"],
             record["decision"]["action"], record["decision"]["human_review"]))
print("   both checkpoints resident: %s" % r.loaded)
print("\n   `routing`, `answers` and every probability are model output.")
print("   `action`, `team`, `human_review` and the thresholds are application policy --")
print("   swap the policy without retraining, and refit temperatures before trusting the")
print("   probabilities (see example 34).")
