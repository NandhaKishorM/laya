"""Example 12 -- the same facts, three ways.

A state is read as text, so one complaint written as a plain ticket, as an email with a
greeting and signature, and as a JSON record yields different probabilities -- and one
question changes its answer outright.
"""
from _common import banner, device_line, heading, load

banner("12", "The same facts, three ways", """
    Example 11 showed that a dict is serialised to JSON and read as text. This one shows
    what that means in practice: there is no schema engine and no field lookup, so the
    phrasing and the structure of the state decide what the model sees.

    One duplicate-charge complaint appears three times below -- as a one-line ticket, as
    an email with a greeting and a signature, and as a structured JSON record -- and the
    same three questions are asked of each. `department` and `refund_requested` turn out
    to be robust to the rewrite; `churn_risk` is not, and the reason is worth reading.
    """)

agent = load("english")
device_line(agent)

PLAIN = ("Support ticket: invoice #4411 was billed twice for March. The customer asks for "
         "the duplicate charge to be refunded today and says the plan will be cancelled if "
         "it is not resolved.")

EMAIL = {
    "from": "Dana Reyes <dana.reyes@acme.example>",
    "to": "support@vendor.example",
    "subject": "Duplicate charge on invoice #4411",
    "body": ("Hi support team,\n\nI noticed that invoice #4411 was billed twice for March. "
             "Please refund the duplicate charge today, otherwise we will cancel our plan.\n\n"
             "Best regards,\nDana Reyes\nFinance, Acme GmbH\ndana.reyes@acme.example\n"
             "+49 30 1234 5678"),
}

RECORD = {
    "invoice_id": "4411",
    "month": "March",
    "duplicate_charge": True,
    "line_items": [{"description": "plan subscription", "amount": 129.00, "occurrences": 2}],
    "refund_requested": True,
    "refund_deadline": "today",
    "account_action_if_unresolved": "cancel plan",
    "customer_note": "The same invoice appears twice on the March statement.",
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts",
                     "other": "everything else"},
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly request a refund?",
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the customer threaten to cancel or leave?",
    },
}

STATES = [
    ("plain ticket", PLAIN),
    ("email + signature", EMAIL),
    ("JSON record", RECORD),
]


def cell(answer):
    """The headline of one answer, short enough to sit in a column."""
    if answer["type"] == "choice":
        return "%s p=%.3f" % (answer["choice"], answer["probabilities"][answer["choice"]])
    if answer["type"] == "score":
        return "%.2f/%d" % (answer["score"], len(answer["probabilities"]) - 1)
    return "%s P=%.3f" % ("true" if answer["noul"] > 0.5 else "false", answer["noul"])


results = {}
for label, state in STATES:
    results[label] = agent.predict(state, QUESTIONS)["answers"]

heading("the same three questions, one column per phrasing")
print("   %-18s %-19s %-19s %s" % ("question", STATES[0][0], STATES[1][0], STATES[2][0]))
for qid in QUESTIONS:
    print("   %-18s %-19s %-19s %s" % (qid, cell(results[STATES[0][0]][qid]),
                                       cell(results[STATES[1][0]][qid]),
                                       cell(results[STATES[2][0]][qid])))

heading("what moved, and what did not")
for qid, spec in QUESTIONS.items():
    answers = [results[label][qid] for label, _ in STATES]
    if spec["type"] == "choice":
        labels = [a["choice"] for a in answers]
        probs = [a["probabilities"][a["choice"]] for a in answers]
        if len(set(labels)) == 1:
            print("   %-18s STABLE   all three -> %s (p %.3f - %.3f)"
                  % (qid, labels[0], min(probs), max(probs)))
        else:
            print("   %-18s MOVED    %s" % (qid, "  ->  ".join(
                "%s p=%.3f" % (lab, p) for lab, p in zip(labels, probs))))
    elif spec["type"] == "score":
        scores = [a["score"] for a in answers]
        spread = max(scores) - min(scores)
        print("   %-18s %-7s %.2f  ->  %.2f  ->  %.2f  (spread %.2f)"
              % (qid, "STABLE" if spread < 0.5 else "MOVED", scores[0], scores[1], scores[2],
                 spread))
    else:
        vals = [a["noul"] for a in answers]
        flags = [v > 0.5 for v in vals]
        if len(set(flags)) == 1:
            print("   %-18s STABLE   %s in all three (P true %.3f - %.3f)"
                  % (qid, "true" if flags[0] else "false", min(vals), max(vals)))
        else:
            print("   %-18s MOVED    %s  ->  %s  ->  %s  (P true %.3f -> %.3f -> %.3f)"
                  % (qid, "true" if flags[0] else "false", "true" if flags[1] else "false",
                     "true" if flags[2] else "false", vals[0], vals[1], vals[2]))

churn = [results[label]["churn_risk"]["noul"] for label, _ in STATES]
dept = [results[label]["department"] for label, _ in STATES]
refund = [results[label]["refund_requested"]["noul"] for label, _ in STATES]
print("""
   All three states carry the same facts, and no question definition changed. Yet the
   probability of `%s` moves from %.3f to %.3f to %.3f, crossing the 0.5 line, while
   `department` stays `%s` (p %.3f - %.3f) and a refund stays requested throughout
   (P true %.3f - %.3f).

   The reason is that the record does not say "we will cancel" as a sentence; it says
   `account_action_if_unresolved: "cancel plan"`. To the encoder that reads like a
   conditional field, not a threat, so `churn_risk` drops. The lesson is not "JSON is
   worse" -- it is that a field name is text too, and structure silently changes the
   decision. If an answer matters, ask it explicitly and check the confidence.
   """ % ("churn_risk", churn[0], churn[1], churn[2], dept[0]["choice"],
          min(a["probabilities"][a["choice"]] for a in dept),
          max(a["probabilities"][a["choice"]] for a in dept),
          min(refund), max(refund)))
