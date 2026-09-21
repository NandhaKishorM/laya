"""Example 03 -- an ordinal rubric with a score question.

A `score` question is a choice question whose labels are ordered. Laya returns an
expected level, the full distribution over levels, and the legend that maps each level
index back to its text, so the number never floats free of its rubric.
"""
from _common import banner, device_line, heading, load

banner("03", "Ordinal urgency (score)", """
    A `score` question asks for an ordering, not a category. `criteria` is an ordered
    list from lowest to highest, and the answer reports `score` (the expected level,
    sum(level * probability), 0-based, so a value between levels is normal and
    meaningful), `probabilities` over every level keyed "0", "1", ..., `legend` (the
    criteria list in words), and `confidence`, the normalised entropy of the
    distribution, exactly as for choice.

    The same question is asked about a relaxed request and an outage. The expected
    level moves from below 1 to near 2, and the distribution sharpens.
    """)

agent = load("english")
device_line(agent)

URGENCY = {
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": [                          # ordered: level 0 is the lowest
            "not urgent",
            "soon",
            "critical deadline or blocking issue",
        ],
    },
}

CASES = [
    ("relaxed request",
     {"body": "Hi, when someone gets a chance, could you update the billing address on our "
              "account? No rush at all, this can wait until next week."}),
    ("production outage",
     {"body": "Our production checkout is down and we are losing orders every minute. "
              "This is blocking all sales and needs to be fixed right now."}),
]

for label, state in CASES:
    heading(label)
    answer = agent.predict(state, URGENCY)["answers"]["urgency"]
    print("   expected level : %.3f  (of %d)" % (answer["score"], len(answer["legend"]) - 1))
    print("   confidence     : %.3f" % answer["confidence"])
    print("   distribution over the legend:")
    for index, text in answer["legend"].items():
        prob = answer["probabilities"][index]
        bar = "#" * int(round(prob * 32))
        print("     level %s  %6.3f  %-38s %s" % (index, prob, text, bar))

print("""
   Read the distribution, not just the mean. Both of these pick a sensible level, but
   the relaxed request spreads 30% of its mass onto "soon" -- a good sign not to treat
   the score as a hard class. Your application sets the threshold that matters.
   """)
