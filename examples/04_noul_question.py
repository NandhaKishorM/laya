"""Example 04 -- a calibrated yes/no with a noul question.

`noul` is a two-option question that returns P(true) directly. The probability is
calibrated, so the decision rule is yours: pick the threshold your application can
afford, rather than accepting a built-in 0.5.
"""
from _common import banner, describe, device_line, heading, load

banner("04", "Calibrated yes/no (noul)", """
    A `noul` question answers one statement with a probability instead of a label.
    The answer has `noul` -- P(true), in 0..1, the probability the statement holds --
    and `confidence`, which is max(p, 1-p) by the model's design.

    Note that `confidence` says nothing about which side won -- it is symmetric. The
    `noul` value is what carries the direction, so read it directly and apply your own
    threshold. A support queue might act at 0.9, while a cheap pre-filter might act at
    0.5; both are legitimate uses of the same calibrated number.

    The two states below are a clear yes and a clear no. Watch the probability sit near
    the ends of the range, and watch confidence mirror it around 0.5 only when the
    answer is genuinely uncertain.
    """)

agent = load("english")
device_line(agent)

REFUND = {
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

CASES = [
    ("true case: a refund demand",
     {"from": "user@acme.com",
      "subject": "Duplicate charge on invoice #4411",
      "body": "Hi, we were billed twice for March. Please refund the duplicate today "
              "or we will cancel our plan."}),
    ("false case: a thank-you note",
     {"body": "Just wanted to say the new dashboard looks great and the export works now. "
              "Thanks for shipping it!"}),
]

for label, state in CASES:
    heading(label)
    answer = agent.predict(state, REFUND)["answers"]["refund_requested"]
    p = answer["noul"]
    print("   P(statement true) : %.4f" % p)
    print("   confidence        : %.4f   (max(p, 1-p), so it is %.4f either way)"
          % (answer["confidence"], answer["confidence"]))
    for threshold in (0.5, 0.8, 0.9):
        print("     decision at threshold %.1f : %s"
              % (threshold, "true" if p >= threshold else "false"))

print("""
   The model never rounds this to a label for you -- the raw probability is the answer.
   Choose the threshold from the cost of a false positive against a false negative.
   """)

heading("same primitive via the shared describe() helper")
describe(agent.predict(CASES[0][1], REFUND)["answers"])
