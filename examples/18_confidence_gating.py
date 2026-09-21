"""Example 18 -- confidence gating for automated routing.

Answers a batch of support emails and branches on Laya's calibrated confidence: act
automatically at 0.85 or above, hand the rest to a human.
"""
from _common import banner, describe, device_line, heading, load

banner("18", "Confidence gating", """
    The production pattern from the README. Laya's probabilities are trained with strictly
    proper scoring rules, so `confidence` is a real probability, not a vibe: it is the
    probability mass Laya puts on its own top answer.

    We route at >= 0.85 automatically and escalate everything below that to a human. Watch
    the second email: its label is right, but Laya is not confident enough to act on it
    alone -- which is exactly what the gate is for.
    """)

THRESHOLD = 0.85

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email in `body`?",
        "criteria": {
            "billing": "invoices, payments, refunds, duplicate charges",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts, upgrades",
            "other": "everything else",
        },
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the sender ask for a refund?",
    },
}

EMAILS = [
    ("duplicate charge",
     "We were billed twice for March, invoice #4411. Please reverse the duplicate charge today."),
    ("api outage",
     "Your API has been returning 502 errors for our whole team since 9am. "
     "Production is down, please fix ASAP."),
    ("invoice download",
     "Where can I download the PDF invoice for February? I cannot find it in the billing portal."),
    ("enterprise pricing",
     "We are a 400-person company evaluating your enterprise tier. "
     "Can you send pricing and a security questionnaire?"),
    ("vague request",
     "Hi, I need some help with my account. Thanks."),
    ("feature idea",
     "It would be great if the export button supported CSV as well as PDF. No rush."),
]

agent = load("english")
device_line(agent)

heading("the full typed answer for the first email")
first = agent.predict({"body": EMAILS[0][1]}, QUESTIONS)
describe(first["answers"])

heading("gating decision for every email")
auto, escalated, seen = [], [], {}
for name, body in EMAILS:
    answers = agent.predict({"body": body}, QUESTIONS)["answers"]
    dept = answers["department"]["choice"]
    conf = answers["department"]["confidence"]
    seen[name] = answers["department"]
    if conf >= THRESHOLD:
        decision = "AUTOMATE -> %s queue" % dept
        auto.append(name)
    else:
        decision = "ESCALATE -> human triage"
        escalated.append(name)
    print("   %-18s %-8s conf=%.3f  %s" % (name, dept, conf, decision))

heading("summary")
print("   threshold            %.2f" % THRESHOLD)
print("   automated (%d)        %s" % (len(auto), ", ".join(auto)))
print("   escalated (%d)        %s" % (len(escalated), ", ".join(escalated)))
outage = seen["api outage"]
outage_p = outage["probabilities"][outage["choice"]]
print("""
   The gate is about certainty, not correctness. "api outage" is labelled `%s`
   with p=%.3f -- the right answer -- yet its confidence is %.3f, below the bar, so it
   still goes to a human. That is the trade you tune with THRESHOLD: raise it and fewer
   mistakes reach production, but more tickets cost a human first.
   """ % (outage["choice"], outage_p, outage["confidence"]))
