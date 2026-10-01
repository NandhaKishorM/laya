"""Example 19 -- many states is a loop, not a batch.

Answers twenty short tickets the way a beginner naturally writes it -- one `predict()` per
ticket -- then measures a single call, and states the rule: questions batch, states loop.
"""
import time

from _common import banner, device_line, heading, load, timed

banner("19", "Twenty tickets is a loop, not a batch", """
    `predict(state, questions)` takes exactly one state. There is no batch-of-states
    argument, so the natural way to triage a queue is a `for` loop with one call per
    ticket. That is the right shape -- but it is worth seeing what it costs.

    Below, twenty short tickets each get the same three questions in one call per ticket.
    Then the same three questions are timed on a single ticket, so the loop's per-ticket
    cost can be compared with the cost of one call. The point is not that the loop is
    wrong; it is that the forward pass is per state, and questions ride along for almost
    free, so you should ask everything you need about a state in that state's one call.
    """)

agent = load("english")
device_line(agent)

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts",
                     "other": "everything else"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly ask for a refund?",
    },
}

BODIES = [
    "I was charged twice this month and want the duplicate refunded today.",
    "The API returns 502 errors for our whole team; production is down.",
    "Can you send enterprise pricing and a security questionnaire?",
    "How do I download the PDF invoice for February?",
    "My password reset email never arrives, so I cannot sign in.",
    "Please cancel my subscription at the end of the month.",
    "The export button only offers PDF; CSV would help a lot.",
    "We need two more seats on our plan and a new quote.",
    "The tracking page has not updated in four days.",
    "A refund for order 5521 has not appeared on my statement.",
    "The dashboard is slow since the last release.",
    "Do you offer a discount for annual billing?",
    "I received someone else's invoice in my inbox.",
    "Two-factor authentication codes are rejected on my phone.",
    "Please upgrade us to the business tier this week.",
    "The mobile app crashes whenever I open the reports tab.",
    "Our card was declined even though the bank approved it.",
    "Where can I change the billing contact on the account?",
    "The nightly sync job has failed for three nights running.",
    "I want a full refund and I am disputing the charge with my bank.",
]
TICKETS = [{"ticket": "T-%02d" % (i + 1), "body": body} for i, body in enumerate(BODIES)]

agent.predict(TICKETS[0], QUESTIONS)        # warm up the weights and the first shape

heading("the natural loop: one predict() call per ticket")
t0 = time.perf_counter()
for state in TICKETS:
    agent.predict(state, QUESTIONS)
loop_s = time.perf_counter() - t0
n_answers = len(TICKETS) * len(QUESTIONS)
print("   %d calls, %d answers: %.2f s total" % (len(TICKETS), n_answers, loop_s))
print("   per ticket : %.1f ms" % (1000 * loop_s / len(TICKETS)))
print("   throughput : %.1f questions/second" % (n_answers / loop_s))

heading("one call, timed properly")
_, one_q_ms = timed(lambda: agent.predict(TICKETS[0], {"department": QUESTIONS["department"]}),
                    repeat=3)
_, all_q_ms = timed(lambda: agent.predict(TICKETS[0], QUESTIONS), repeat=3)
print("   one ticket, 1 question : %.1f ms" % one_q_ms)
print("   one ticket, 3 questions: %.1f ms  (%.2fx for 3x the questions)"
      % (all_q_ms, all_q_ms / one_q_ms))
print("   the loop's per-ticket cost was %.1f ms. Every ticket is a fresh sequence length,"
      % (1000 * loop_s / len(TICKETS)))
print("   so the loop pays per-shape warm-up as it goes (on MPS, Metal kernel compilation);")
print("   `timed()` reuses one warm shape and is the cheaper figure. Either way the forward")
print("   pass is the unit, paid once per ticket.")

print("""
   There is no way to fold twenty states into one call, and there does not need to be:
   `predict` is already a batch -- a batch of questions about one state. Batching
   *questions* is what the model is built for, so ask every question you have in the same
   call. Batching *states* is your loop, and its cost scales with the number of tickets.
   For a queue, that means a loop sized to your throughput budget, not one call with
   everything; for one ticket, it means one call with every question in it.
   """)
