"""Example 18 -- confidence gating for automated routing.

Answers a batch of support emails and branches on Laya's calibrated confidence: act
automatically at 0.85 or above, hand the rest to a human.
"""
from _common import banner, describe, device_line, heading, load

banner("18", "Confidence gating", """
    The production pattern from the README. Gate on `answer_confidence`: the calibrated
    probability Laya puts on the answer it reports, defined the same way on every question
    type. `confidence` is a different number on `choice` and `score` -- 1 minus normalised
    entropy, a measure of how concentrated the distribution is -- and the README warns against
    carrying a threshold over from it.

    We route at >= 0.85 automatically and escalate everything below that to a human. The gate
    is about certainty, not correctness: a ticket can come back with the label you expect and
    still go to a human because `answer_confidence` sits below the bar.
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
    answer_conf = answers["department"]["answer_confidence"]
    seen[name] = answers["department"]
    if answer_conf >= THRESHOLD:
        decision = "AUTOMATE -> %s queue" % dept
        auto.append(name)
    else:
        decision = "ESCALATE -> human triage"
        escalated.append(name)
    print("   %-18s %-8s answer_conf=%.3f  %s" % (name, dept, answer_conf, decision))

heading("summary")
print("   threshold            %.2f" % THRESHOLD)
print("   automated (%d)        %s" % (len(auto), ", ".join(auto)))
print("   escalated (%d)        %s" % (len(escalated), ", ".join(escalated)))
if escalated:
    soft = seen[escalated[0]]
    print("""
   The gate is about certainty, not correctness. "%s" comes back labelled `%s` with
   answer_confidence %.3f, below the %.2f bar, so it goes to a human -- even though that label
   may well be the right one. That is the trade you tune with THRESHOLD: raise it and fewer
   uncertain answers reach production, but more tickets cost a human first.
   """ % (escalated[0], soft["choice"], soft["answer_confidence"], THRESHOLD))
else:
    print("""
   Nothing fell below the %.2f bar on this batch. Lower THRESHOLD to make the gate stricter,
   or lengthen EMAILS until something uncertain shows up.
   """ % THRESHOLD)
