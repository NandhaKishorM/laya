"""Example 15 -- triage preset on one realistic support ticket.

Runs `laya.triage_questions()` over an enterprise billing complaint and reads the five
typed answers that make up a triage decision.
"""

from _common import laya, banner, describe, device_line, heading, load

banner("15", "Preset: support triage", """
    `triage_questions()` is a ready-made schema for inbound support: five questions, one
    forward pass. It mixes all three primitives -- a multi-way choice, two yes/no
    probabilities, and an ordinal score -- so a single call fills a whole triage record.

    The ticket below is an enterprise customer who has been double charged for three
    months, had tickets ignored, and threatens to leave. Every field except `is_urgent`
    comes back decisive.
    """)

TICKET = {
    "message": "I have been charged twice for the same seat for three months in a row and support has "
               "ignored every ticket I opened. I am the admin for our enterprise account and this is "
               "unacceptable. Refund the duplicate charges immediately or we will cancel and switch "
               "to a competitor.",
}

agent = load("english")
device_line(agent)

heading("the five preset answers")
answers = agent.predict(TICKET, laya.triage_questions())["answers"]
describe(answers)

heading("the shape of a triage record")
a = answers
print("   intent             choice: %s (p=%.3f, conf=%.3f) -- wants money back, not a bug fix." % (
    a["intent"]["choice"], a["intent"]["probabilities"][a["intent"]["choice"]],
    a["intent"]["confidence"]))
print("   is_urgent          noul: %.3f -- angry and says \"immediately\", but names no deadline." %
      a["is_urgent"]["noul"])
print("   frustration        score: %.2f / 3 -- \"clearly annoyed\", short of \"very angry\"." %
      a["frustration"]["score"])
print("   refund_requested   noul: %.3f -- \"refund the duplicate charges\" is explicit." %
      a["refund_requested"]["noul"])
print("   churn_risk         noul: %.3f -- \"cancel and switch to a competitor\"." %
      a["churn_risk"]["noul"])
print("   The contrast below shows what does move the one soft field, `is_urgent`.")

heading("contrast: a ticket with an actual deadline")
DEADLINE_TICKET = {
    "message": "Our renewal is due tomorrow and the card on file keeps failing. "
               "Please help us before 5pm or the account lapses."
}
deadline = agent.predict(DEADLINE_TICKET, laya.triage_questions())["answers"]
describe(deadline)
print("""
   Same schema, same checkpoint, and now `is_urgent` fires at %.3f because the ticket
   states a deadline ("due tomorrow ... before 5pm"). The preset is a signal, not a
   verdict: `intent` and the two `noul` fields are the dependable ones, while `is_urgent`
   is worth reading alongside the raw text.
   """ % deadline["is_urgent"]["noul"])
