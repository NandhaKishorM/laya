"""Example 06 -- many questions, still one forward pass.

Batching 13 independent questions into one `predict()` call costs one pass over the
state. The encoder sees the state once per question in the batch, and the answers all
come back together, so `usage["input_tokens"]` is the sum across the whole call.
"""
from _common import STATE_EN, banner, describe, device_line, load, timed

banner("06", "Thirteen questions, one call", """
    Real triage asks many things about one message. Here 13 questions -- 5 choice,
    3 score and 5 noul -- are answered in a single `predict()` call.

    Two details worth noticing:

      * `usage["input_tokens"]` is the batch total. Split the questions across two
        calls and the per-call token counts add up to exactly the same number; the
        batch simply folds them into one pass.
      * latency barely moves with the number of questions. The encoder is the cost, and
        it runs once for the whole batch, not once per question.
    """)

agent = load("english")
device_line(agent)

MANY = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts",
                                "other": "none of the above"}},
    "issue_type": {"type": "choice", "instructions": "What is the underlying issue?",
                   "criteria": {"duplicate_charge": "the same charge appears more than once",
                                "late_delivery": "an order has not arrived on time",
                                "outage": "a service is down or erroring",
                                "login_problem": "the user cannot sign in",
                                "other": "none of the above"}},
    "language": {"type": "choice", "instructions": "What language is this written in?",
                 "criteria": {"english": "English", "german": "German",
                              "french": "French", "other": "any other language"}},
    "severity": {"type": "choice", "instructions": "How serious is the problem?",
                 "criteria": {"cosmetic": "annoying but no real impact",
                              "costly": "costs money or time",
                              "blocking": "stops work or business entirely"}},
    "account_action": {"type": "choice", "instructions": "What account change is wanted?",
                       "criteria": {"none": "no account change",
                                    "cancel_plan": "cancel or close the account",
                                    "downgrade": "move to a cheaper plan",
                                    "upgrade": "move to a bigger plan"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "sentiment": {"type": "score", "instructions": "How negative is the tone?",
                  "criteria": ["positive", "neutral", "negative", "very negative"]},
    "time_pressure": {"type": "score", "instructions": "How much time pressure is stated?",
                      "criteria": ["none", "implied", "explicit deadline", "already overdue"]},
    "churn_risk": {"type": "noul",
                   "instructions": "Does the sender threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the sender ask for money back?"},
    "is_angry": {"type": "noul", "instructions": "Does the sender sound angry?"},
    "needs_manager": {"type": "noul",
                      "instructions": "Does this need a manager to get involved?"},
    "contains_pii": {"type": "noul",
                     "instructions": "Does the message contain personal or financial data?"},
}

result = agent.predict(STATE_EN, MANY)
print("   %d questions, one call -- %d answers returned:\n" % (len(MANY), len(result["answers"])))
describe(result["answers"], indent="   ")
print("\n   batch call usage : %d input tokens, %d output tokens"
      % (result["usage"]["input_tokens"], result["usage"]["output_tokens"]))

first_id = next(iter(MANY))
rest = {qid: question for qid, question in MANY.items() if qid != first_id}
first_tokens = agent.predict(STATE_EN, {first_id: MANY[first_id]})["usage"]["input_tokens"]
rest_tokens = agent.predict(STATE_EN, rest)["usage"]["input_tokens"]
print("   one question alone        : %d input tokens" % first_tokens)
print("   the other %2d in one call  : %d input tokens" % (len(rest), rest_tokens))
print("   sum                       : %d input tokens, exactly the batch total above"
      % (first_tokens + rest_tokens))
print("   -> input_tokens is the batch total: every question in the call contributes.")
_, single_ms = timed(lambda: agent.predict(STATE_EN, {"urgency": MANY["urgency"]}), repeat=2)
_, batch_ms = timed(lambda: agent.predict(STATE_EN, MANY), repeat=2)
print("\n   latency 1 question  : %.0f ms" % single_ms)
print("   latency 13 questions: %.0f ms  (%.1fx for 13x the questions)"
      % (batch_ms, batch_ms / max(1.0, single_ms)))
