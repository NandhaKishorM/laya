"""Example 05 -- designing the criteria of a choice question.

The criteria descriptions are part of the prompt, so writing them well is part of the job.
One state and one label set are run three times: vague descriptions, specific descriptions
that share the state's vocabulary, and a deliberately confusable pair.
"""
from _common import banner, device_line, heading, load, top

banner("05", "Designing choice criteria", """
    A `choice` question is only as good as its description map. The keys define the legal
    answers, but the values are text the encoder reads alongside the state, so they steer
    which key wins. This example holds the state and the label set fixed and changes only
    the descriptions.

    The state is ambiguous on purpose: a checkout that returns 500 errors, so customers
    cannot pay. It is a payment problem, a system problem, or both, depending on how the
    options are worded.

    Watch the chosen label and the whole distribution, not just the winner: a map that
    makes the model torn is visible as probability mass sitting on the runner-up.
    """)

agent = load("english")
device_line(agent)

STATE = {
    "body": "Our production checkout has been returning 500 errors since this morning and no "
            "orders are going through. Customers see a failed payment page and we are losing "
            "revenue every minute.",
}

VARIANTS = [
    ("vague", {
        "billing": "money",
        "technical": "tech",
        "sales": "sales",
        "other": "other",
    }),
    ("specific", {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else",
    }),
    ("confusable pair", {
        "billing": "errors and failures that affect payments and orders",
        "technical": "errors and failures in the product and its systems",
        "sales": "pricing, quotes and new contracts",
        "other": "everything else",
    }),
]

for name, criteria in VARIANTS:
    heading(name)
    question = {"department": {"type": "choice",
                               "instructions": "Which department should handle this request?",
                               "criteria": criteria}}
    answer = agent.predict(STATE, question)["answers"]["department"]
    print("   chosen label : %s" % answer["choice"])
    print("   confidence   : %.3f" % answer["confidence"])
    for label, prob in answer["probabilities"].items():
        bar = "#" * int(round(prob * 32))
        print("     %-10s %6.3f  %s" % (label, prob, bar))
    print("   top two      : %s" % top(answer["probabilities"], 2))

print("""
   What changed, and why:

   * "vague" descriptions carry no vocabulary from the state, so the model falls back on the
     surface word "payment" and picks billing.
   * "specific" descriptions reuse the state's own vocabulary -- "outages", "system errors"
     line up with "500 errors" while the billing description talks about invoices and
     refunds, which this message never mentions. The choice flips to technical.
   * the "confusable pair" gives billing and technical near-identical openings ("errors and
     failures ..."). The model stays torn between them instead of splitting the two concepts;
     the runner-up keeps a third of the mass, and confidence moves to reflect that.

   The lesson is not that one map is correct. It is that the map is prompt engineering: if a
   label keeps losing, say so in its description rather than rephrasing the question.
   """)
