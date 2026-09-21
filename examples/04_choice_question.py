"""Example 04 -- routing with a choice question.

A `choice` question offers a closed set of labels; Laya returns the winner plus a
probability for every label, all from one forward pass. Nothing is generated, so there
is no text to parse and no label it can invent outside the `criteria` map.
"""
from _common import STATE_EN, banner, device_line, heading, load

banner("04", "Routing a support email (choice)", """
    A `choice` question gives the model a fixed set of labels. `criteria` is a
    label -> description map: each key becomes one answer option and its description
    tells the model what that label means. The answer then carries the chosen label, a
    probability for every label, and a calibrated confidence.

    `confidence` is not the top probability: it is the normalised entropy of the whole
    distribution, 1 - H(p) / log(k), so it rewards one option clearly beating the rest.

    Watch `department` follow the description map. The duplicate charge lands in
    billing; a checkout returning 500s lands in technical.
    """)

agent = load("english")                  # device=None -> CUDA, else MPS, else CPU
device_line(agent)

DEPARTMENT = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {                    # label -> description; keys are the answer options
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
}

CASES = [
    ("duplicate charge", STATE_EN),
    ("production outage",
     {"body": "Our production checkout has been returning 500 errors since this morning "
              "and no orders are going through."}),
]

for label, state in CASES:
    heading(label)
    answer = agent.predict(state, DEPARTMENT)["answers"]["department"]
    print("   chosen label : %s" % answer["choice"])
    print("   confidence   : %.3f" % answer["confidence"])
    print("   all label probabilities:")
    for option, prob in answer["probabilities"].items():
        bar = "#" * int(round(prob * 32))
        print("     %-10s %6.3f  %s" % (option, prob, bar))

print("""
   The map is the contract: add a label and it becomes a legal answer, remove one and
   the model can no longer pick it. To get different routing, edit the descriptions
   rather than the question text.
   """)
