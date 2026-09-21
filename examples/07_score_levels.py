"""Example 07 -- choosing a score rubric's granularity.

The same ordinal judgement at 3, 5 and 7 levels, over a mild and a severe state: the
expected level, the full distribution and the confidence that comes with each rubric.
"""
from _common import banner, device_line, heading, load

banner("07", "How many levels a score rubric needs", """
    A `score` question turns an ordering into a number, and you choose how fine that
    ordering is. The examples below ask the same urgency question with 3, 5 and 7 levels,
    anchored at "no rush" and "blocking", and change only the number of steps between them.

    Two things are worth reading in the output:

      * `score` is the expected level on the rubric's own 0-based scale, so it is only
        comparable across rubrics after you divide by (levels - 1). A score of 0.04/2 and
        one of 0.33/6 both mean "at the bottom of this rubric".
      * more levels give the distribution more room to spread, and the confidence is
        normalised entropy, so it notices when the mass is not concentrated.

    A coarse rubric is often the honest choice: extra levels do not create resolution the
    model does not have, they only let the same uncertainty occupy more bins.
    """)

agent = load("english")
device_line(agent)

RUBRICS = {
    3: ["no rush", "normal", "blocking"],
    5: ["no rush", "low", "normal", "high", "blocking"],
    7: ["no rush", "very low", "low", "normal", "high", "very high", "blocking"],
}

STATES = [
    ("mild request",
     {"body": "Hi, when someone gets a chance, could you update the billing address on our "
              "account? No rush at all, this can wait until next week."}),
    ("severe request",
     {"body": "Our production checkout is down and we are losing orders every minute. This is "
              "blocking all sales and needs to be fixed right now."}),
]

for state_name, state in STATES:
    heading(state_name)
    for levels, criteria in RUBRICS.items():
        question = {"urgency": {"type": "score",
                                "instructions": "How urgent is this request?",
                                "criteria": criteria}}
        answer = agent.predict(state, question)["answers"]["urgency"]
        probs = answer["probabilities"]
        spread = 1.0 - max(probs.values())
        up_scale = 100.0 * answer["score"] / (levels - 1)
        print("   %d levels   score %5.3f / %d  (%4.1f%% up the scale)   conf %.3f   "
              "mass off top %.3f"
              % (levels, answer["score"], levels - 1, up_scale, answer["confidence"], spread))
        for index, text in answer["legend"].items():
            prob = probs[index]
            bar = "#" * int(round(prob * 32))
            print("       %s  %5.3f  %-32s %s" % (index, prob, text, bar))
        print()

print("""
   What to take away:

   * The expected value is on each rubric's own scale. The mild state sits near the bottom
     of every rubric (0.04/2, 0.19/4, 0.33/6) and the severe state near the top of every
     rubric, so the judgement is stable even though the raw number is not.
   * More levels widen the distribution: the mass not on the single most likely level grows
     with the level count for both states, and confidence falls as the mass spreads.
   * The extra levels are not resolving anything. On the mild state the 7-level rubric puts
     real mass on "low" and even "normal" -- uncertainty about distinctions the message does
     not support. The 3-level rubric says "no rush" and means it.

   Pick the coarsest rubric that answers your question. You can always threshold the expected
   score in application code; you cannot put resolution back into the model.
   """)
