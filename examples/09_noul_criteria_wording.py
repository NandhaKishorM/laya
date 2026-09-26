"""Example 09 -- wording the criteria of a noul question.

A `noul` question is always [false, true]; `criteria` can spell out what each side means.
No criteria, symmetric criteria and criteria loaded towards one side are compared over a
clear true state and a clear false state.
"""
from _common import banner, device_line, heading, laya, load

banner("09", "Wording the two sides of a noul question", """
    `noul` is a two-option question with fixed labels, `false` and `true`. Its `criteria`
    map is therefore not a label set -- it is a place to define what each side means:

        "criteria": {"false": "<what false looks like>", "true": "<what true looks like>"}

    Strings are enough here; dict criteria are covered later, in example 31. Omit a side and
    the model falls back to the generic phrasing "no, the statement does not hold" / "yes,
    the statement holds".

    Compare the three wordings below. The question is identical, so any movement in
    P(true) comes from the definitions the encoder reads alongside the state.
    """)

agent = load("english")
device_line(agent)

INSTRUCTIONS = "Does the user want a refund?"

STATES = [
    ("clear true",
     {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
      "body": "Hi, we were billed twice for March. Please refund the duplicate today or we "
              "will cancel our plan."}),
    ("clear false",
     {"body": "Just wanted to say the new dashboard looks great and the export works now. "
              "Thanks for shipping it!"}),
]

VARIANTS = [
    ("no criteria", None),
    ("symmetric", {
        "false": "the user is not asking for money back",
        "true": "the user is asking for money back",
    }),
    # deliberately loaded: the false side demands a flawless message, the true side only a hint
    ("loaded towards true", {
        "false": "no problem at all, the message is purely positive",
        "true": "any hint of trouble, dissatisfaction or a request for money back",
    }),
]

for name, criteria in VARIANTS:
    heading(name)
    question = {"refund": {"type": "noul", "instructions": INSTRUCTIONS}}
    if criteria is not None:
        question["refund"]["criteria"] = criteria
    for option in laya.render_options({"t": "noul", "ins": INSTRUCTIONS, "crit": criteria}):
        print("     %s" % option)
    for state_name, state in STATES:
        answer = agent.predict(state, question)["answers"]["refund"]
        label = "true" if answer["noul"] > 0.5 else "false"
        print("       %-11s noul %.4f   conf %.4f   -> %s" % (state_name, answer["noul"],
                                                              answer["confidence"], label))
    print()

print("""
   What actually changed:

   * The clear-true state is close to saturated, so the wording barely moves it: no criteria
     gives P(true) = 0.869, symmetric 0.849, loaded 0.884. A shift of ~0.02 on an answer the
     model already gets right is not a signal to build on.
   * The clear-false state is where the wording shows: P(true) rises from 0.032 with no
     criteria, to 0.050 symmetric, to 0.077 with the side loaded towards true -- the loading
     more than doubles it, and confidence falls from 0.969 to 0.923 to match.
   * Both edits move in the direction you would predict from the text. Calling "any hint of
     trouble" a true case genuinely makes the model read benign messages as more suspicious.
   * Symmetric wording is not a no-op either: defining both sides pulls the false case up
     slightly, because a concrete definition is easier to match than the generic default.

   The wording is a policy decision, not a formatting detail. Write both sides explicitly,
   keep them symmetric unless you mean to bias the boundary, and calibrate the threshold in
   your own application afterwards.
   """)
