"""Example 18 -- router preset turns a request into a model tier.

Runs `laya.router_questions()` over a trivial lookup, a hard refactor and a sensitive
legal question, then derives a "small vs frontier model" policy from the answers.
"""

from _common import laya, banner, describe, heading, load

banner("18", "Preset: small vs frontier routing", """
    `router_questions()` scores an arbitrary request on four axes -- difficulty (0-3),
    domain, whether it needs tools, and whether it is sensitive -- so a model gateway can
    pick a cheap or an expensive model per request instead of sending everything to the
    frontier.

    The three requests are chosen to land in different buckets. After the predictions we
    turn the answers into a small policy function and print the tier each request takes.
    """)

REQUESTS = [
    ("trivial lookup", "What is the capital of Portugal?"),
    ("hard refactor", "Refactor this 900-line Django view into service classes with dependency "
                      "injection, keeping all existing tests green and preserving transaction "
                      "boundaries."),
    ("sensitive legal", "My employer is withholding my final paycheck after I reported a safety "
                        "violation. Should I sue, and can I claim damages under California law?"),
]

agent = load("english")

results = {}
for label, request in REQUESTS:
    heading(label)
    answers = agent.predict({"request": request}, laya.router_questions())["answers"]
    results[label] = answers
    describe(answers)

heading("side by side")
print("   %-17s %-10s %-16s %-11s %s" %
      ("request", "difficulty", "domain", "needs_tools", "is_sensitive"))
for label, _ in REQUESTS:
    a = results[label]
    domain = max(a["domain"]["probabilities"].items(), key=lambda kv: kv[1])
    print("   %-17s %.2f / 3   %-16s %-11.3f %.3f" % (
        label, a["difficulty"]["score"], "%s (%.2f)" % (domain[0], domain[1]),
        a["needs_tools"]["noul"], a["is_sensitive"]["noul"]))

def route_tier(answers):
    """A gateway policy: cheap model unless the request is hard, sensitive or needs tools."""
    difficulty = answers["difficulty"]["score"]
    sensitive = answers["is_sensitive"]["noul"]
    tools = answers["needs_tools"]["noul"]
    if sensitive >= 0.5:
        return "frontier", "sensitive: money/legal/medical/safety (p=%.2f)" % sensitive
    if tools >= 0.5:
        return "frontier+tools", "needs external data (p=%.2f)" % tools
    if difficulty >= 1.5:
        return "frontier", "difficulty %.2f >= 1.5" % difficulty
    return "small", "difficulty %.2f < 1.5, no tools, not sensitive" % difficulty

heading("policy output")
for label, _ in REQUESTS:
    tier, why = route_tier(results[label])
    print("   %-17s -> %-13s %s" % (label, tier, why))

triv = results["trivial lookup"]
sens = results["sensitive legal"]
print("   The threshold 1.5 sits between the lookup cluster and the reasoning end of the scale.")
print("   Two honest notes. First, `difficulty` is an expected level over four buckets, so even")
print("   a one-line lookup scores %.2f -- the zero point is \"trivial\", not \"easy\"." %
      triv["difficulty"]["score"])
print("   Second, the sensitive request scores only %.2f on difficulty, so difficulty alone" %
      sens["difficulty"]["score"])
print("   would send it to the cheap model; `is_sensitive` at %.3f is what promotes it. A" %
      sens["is_sensitive"]["noul"])
print("   gateway should always let sensitivity and tool needs override the difficulty score,")
print("   never the other way round.")
