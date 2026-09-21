"""Example 25 -- throughput: one forward pass per call, many questions per call.

`predict(state, questions)` runs every question for one state in a single batched forward pass.
This example measures the same typed answers on the multilingual checkpoint two ways -- one
question per call, then a three-question set per call -- to show where the throughput comes from.
"""
import time

from _common import banner, describe, device_line, load

banner("25", "Batch throughput", """
    One `predict()` call is one state plus its questions, and those questions become the rows of
    a single batched forward pass. There is no batch-of-states argument: `collate_items` wraps
    the question list of *one* state in a single group. So the way to raise throughput is to ask
    more questions per call, not to queue more states into one call.

    Each question is its own row and repeats the state, so the win is not a shared encoding --
    it is one set of kernel launches and a fuller matrix instead of many tiny calls. Phase A
    answers 90 questions in 90 calls; phase B answers the *same* 90 questions in 30 calls.

    The 30 load-test tickets use four distinct bodies (with distinct ids) so the numbers are not
    dominated by MPS compiling a fresh kernel for every new sequence shape; a real deployment
    pays that once per shape and then amortises it, exactly as measured here.
    """)

BODIES = [
    "I was charged twice for March and I want the duplicate refunded today.",
    "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung.",
    "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
    "The dashboard stopped loading after yesterday's release; our team is blocked.",
]
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
        "instructions": "Does the user explicitly request a refund?",
    },
}

agent = load("multilingual")
device_line(agent)
STATES = [{"ticket": "T-%03d" % i, "body": BODIES[i % len(BODIES)]} for i in range(1, 31)]
NS, NQ = len(STATES), len(QUESTIONS)
print("   %d states x %d questions = %d typed answers in each phase" % (NS, NQ, NS * NQ))

agent.predict(STATES[0], QUESTIONS)                     # warm up the load and the common shapes
for qid, question in QUESTIONS.items():
    agent.predict(STATES[1], {qid: question})


def report(label, calls, answers, seconds):
    print("   %-30s %3d calls  %6.2f s  %6.2f answers/s  %6.1f ms/call  %5.1f ms/answer"
          % (label, calls, seconds, answers / seconds, 1000 * seconds / calls,
             1000 * seconds / answers))


print("\n   == phase A: one question per call ==")
t0 = time.perf_counter()
first_a = None
for state in STATES:
    for qid, question in QUESTIONS.items():
        answer = agent.predict(state, {qid: question})["answers"][qid]
        if first_a is None:
            first_a = answer
seconds_a = time.perf_counter() - t0
report("A: 1 question per call", NS * NQ, NS * NQ, seconds_a)

print("\n   == phase B: the whole question set per call ==")
t0 = time.perf_counter()
first_b = None
for state in STATES:
    answers = agent.predict(state, QUESTIONS)["answers"]
    if first_b is None:
        first_b = answers["department"]
seconds_b = time.perf_counter() - t0
report("B: 3 questions per call", NS, NS * NQ, seconds_b)

same = (first_a["choice"] == first_b["choice"]
        and first_a["confidence"] == first_b["confidence"])
print("\n   the same first answer either way: %s (%s)" % (same, first_b["choice"]))
print("   per-answer cost fell from %.1f to %.1f ms (%.1fx); calls fell from %d to %d for the"
      % (1000 * seconds_a / (NS * NQ), 1000 * seconds_b / (NS * NQ), seconds_a / seconds_b,
         NS * NQ, NS))
print("   same %d answers. More questions per call is the lever; `predict` still takes one state."
      % (NS * NQ))
print("\n   one state, three decisions, one forward pass -- that is the batching Laya exposes:")
describe(agent.predict(STATES[0], QUESTIONS)["answers"])
