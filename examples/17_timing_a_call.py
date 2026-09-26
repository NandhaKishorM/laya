"""Example 17 -- timing a call properly.

Separates checkpoint load time from inference, shows the one-off cost of the first call,
and uses `timed()` medians to measure steady-state cost and how it scales with the number
of questions.
"""
import time

from _common import STATE_EN, banner, device_line, heading, load, timed

banner("17", "Timing a call", """
    A single `perf_counter()` around `agent.predict()` measures three different things
    depending on when you run it: the checkpoint build, the first forward pass (which on
    MPS pays one-off Metal kernel compilation), and ordinary inference. Timing them
    together is how people convince themselves Laya takes ten seconds.

    So: time the load on its own, time the cold first call on its own, then let `timed()`
    discard a warm-up and report a median over several steady-state calls. The bottom line
    is ms per call and ms per question -- and the medians move with whatever else the
    machine is doing, so treat them as a shape, not a benchmark.
    """)

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts",
                                "other": "everything else"}},
    "issue_type": {"type": "choice", "instructions": "What is the underlying issue?",
                   "criteria": {"duplicate_charge": "the same charge appears more than once",
                                "late_delivery": "an order has not arrived on time",
                                "outage": "a service is down or erroring",
                                "login_problem": "the user cannot sign in",
                                "other": "none of the above"}},
    "severity": {"type": "choice", "instructions": "How serious is the problem?",
                 "criteria": {"cosmetic": "annoying but no real impact",
                              "costly": "costs money or time",
                              "blocking": "stops work or business entirely"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "tone": {"type": "score", "instructions": "How negative is the tone?",
             "criteria": ["positive", "neutral", "negative", "very negative"]},
    "refund_requested": {"type": "noul",
                         "instructions": "Does the sender explicitly ask for a refund?"},
    "churn_risk": {"type": "noul",
                   "instructions": "Does the sender threaten to cancel or leave?"},
    "contains_pii": {"type": "noul",
                     "instructions": "Does the message contain personal or financial data?"},
}

heading("1. checkpoint load (weights read, model built, moved to device)")
t0 = time.perf_counter()
agent = load("english")
load_s = time.perf_counter() - t0
device_line(agent)
print("   load: %.2f s  (this happens once per process, not once per call)" % load_s)

heading("2. the cold first call")
t0 = time.perf_counter()
agent.predict(STATE_EN, {"department": QUESTIONS["department"]})
cold_ms = (time.perf_counter() - t0) * 1000
if agent.device.type == "mps":
    why = "one-off Metal kernel compilation for the shapes it just met"
elif agent.device.type == "cuda":
    why = "CUDA context and kernel setup"
else:
    why = "cache warm-up; CPU has no kernel-compilation step"
print("   first predict(): %.0f ms  -- includes %s." % (cold_ms, why))
print("   This number is not the cost of a decision. It is a setup cost.")

heading("3. steady state: timed() warms up once, then reports the median")
sizes = [1, 3, len(QUESTIONS)]
rows = []
for n in sizes:
    subset = dict(list(QUESTIONS.items())[:n])
    _, median_ms = timed(lambda: agent.predict(STATE_EN, subset), repeat=5)
    rows.append((n, median_ms))
    print("   %2d question%s   %8.1f ms/call   %6.2f ms/question"
          % (n, " " if n == 1 else "s", median_ms, median_ms / n))

print("\n   from 1 to %d questions (%.0fx the work): %.1f -> %.1f ms (%.2fx)"
      % (rows[-1][0], float(rows[-1][0]), rows[0][1], rows[-1][1], rows[-1][1] / rows[0][1]))
_, again_ms = timed(lambda: agent.predict(STATE_EN, QUESTIONS), repeat=5)
print("   the full set measured again: %.1f ms -- medians drift with machine load, so read"
      % again_ms)
print("   the ratios, not the absolute milliseconds.")

print("""
   Load time and inference time are different budgets: seconds per process versus
   milliseconds per call. Within inference, adding questions is cheap relative to the
   forward pass, so `input_tokens` grows while latency barely does -- which is why examples
   16 and 36 batch questions and why a per-state loop (example 19) is the thing to avoid.
   """)
