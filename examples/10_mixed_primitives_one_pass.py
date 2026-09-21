"""Example 10 -- every primitive in one call.

`choice`, `score` and `noul` are different heads over the same encoder pass, so mixing
them costs no extra forward passes. The returned `usage` makes that literal: one input
token count for the batch, and zero output tokens because nothing is generated.
"""
from _common import QUESTIONS, STATE_EN, banner, describe, device_line, load, timed

banner("10", "Mixed primitives, one forward pass", """
    `_common.QUESTIONS` combines all three primitives: `department` (choice), `urgency`
    (score), `churn_risk` and `refund_requested` (noul). They are answered together in
    a single `predict()` call.

    The point is in `usage`: `input_tokens` is the batch total for all four questions,
    and `output_tokens` is 0, because Laya is non-autoregressive. There is no decoding
    loop, no new tokens, and therefore nothing to parse or hallucinate.

    `describe()` prints a compact line per question, which is the shape most services
    log.
    """)

agent = load("english")
device_line(agent)

result, median_ms = timed(lambda: agent.predict(STATE_EN, QUESTIONS))

print("   %d questions, one call:" % len(QUESTIONS))
describe(result["answers"])
print("\n   usage   : %d input tokens, %d output tokens"
      % (result["usage"]["input_tokens"], result["usage"]["output_tokens"]))
print("   latency : %.0f ms per call (median of 3, after warm-up)" % median_ms)
