"""Example 18 -- confidence gating for automated routing.

Answers a batch of support emails and branches on the probability Laya reports for its answer:
act automatically at 0.85 or above, hand the rest to a human -- reading back, per question, the
temperature that probability was actually scaled by.
"""
from _common import banner, describe, device_line, heading, load
from laya.common import QTYPES, temp_bucket

banner("18", "Confidence gating", """
    The production pattern from the README. Gate on `answer_confidence`: the probability Laya
    puts on the answer it reports, `max(p)`, defined the same way on every question type.
    `confidence` is a different number on `choice` and `score` -- 1 minus normalised entropy, a
    measure of how concentrated the distribution is -- and the README warns against carrying a
    threshold over from it.

    Whether `answer_confidence` is a *calibrated* probability is a claim about the temperatures
    rather than about the field. The README's Calibration section puts mean ECE on `laya` at
    0.466 as shipped, 0.081 after refitting one temperature per (question type, option-count)
    bucket on held-out data, and warns that both checkpoints are over-confident out of the box.
    The 0.85 threshold below is our policy on the shipped distribution, not a cutoff a
    measurement on this checkpoint's held-out data earned.

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


def option_count(question):
    """k: the number of options the answer is scored over, the other half of the bucket key."""
    if question["type"] == "noul":
        return 2
    return len(question["criteria"])


def scale_for(agent, qtype, k):
    """(bucket, temperature really applied, from the bucket map) -- core's own lookup, replayed.

    `Agent.predict_batch` reads `temperature_by_options.get(temp_bucket(...), temperature[QTYPES[...]])`;
    this is that line run against the loaded agent so the page cannot describe a scaling the
    checkpoint does not perform.
    """
    name = temp_bucket(QTYPES[qtype], k)
    if name in agent.temperature_by_options:
        return name, agent.temperature_by_options[name], True
    return name, agent.temperature[QTYPES[qtype]], False


heading("what each question's number was actually scaled by")
print("   %-22s %-7s %-3s %-12s %-10s %s"
      % ("question", "type", "k", "bucket", "T applied", "source"))
for qid, q in QUESTIONS.items():
    k = option_count(q)
    name, applied, in_map = scale_for(agent, q["type"], k)
    print("   %-22s %-7s %-3d %-12s %-10.4f %s"
          % (qid, q["type"], k, name, applied, "bucket map" if in_map else "per-type default"))
print("   a temperature above 1.0 softens the logits before the softmax, which is what a fitted")
print("   scale is for; 1.0 leaves the number the raw softmax. Neither by itself makes the shipped")
print("   probability a calibrated one -- the README's Calibration loop is what closes that gap,")
print("   and until it has run on your held-out data the 0.85 cutoff above is a policy on the")
print("   shipped distribution rather than a measured accuracy.")

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
