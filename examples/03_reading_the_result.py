"""Example 03 -- reading the result of a predict() call.

A guided tour of what `predict()` returns: the three top-level keys, the `answers` dict
keyed by question id, the fields of each of the three answer types, and what `usage` means.
This is the example to come back to when writing your own glue code.
"""
import json

from _common import banner, device_line, heading, load

banner("03", "Reading the result", """
    Every `predict()` call returns one plain dict with three keys:

        {"model": "laya-rl-agent",
         "answers": {<question id>: <answer>, ...},
         "usage": {"input_tokens": <batch total>, "output_tokens": 0}}

    `answers` is keyed by the ids you chose, so glue code can index it directly instead of
    zipping a list back onto the questions. Each answer carries a `type` matching its
    question, the typed decision fields, a `confidence`, and an `action` payload.

    `usage` is the part readers misread: `input_tokens` is the whole batch, with the state
    counted once per question, so it grows with the number of questions rather than being a
    context length. `output_tokens` is always 0 -- Laya never generates text.
    """)

agent = load("english")
device_line(agent)

STATE = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "We were billed twice for March. Please refund the duplicate."}

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
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}

result = agent.predict(STATE, QUESTIONS)
answers = result["answers"]

# ---------------------------------------------------------------- 1. the raw dict
heading("the raw dict for a three-question call")
print(json.dumps(result, indent=2))

# ---------------------------------------------------------------- 2. top-level keys
heading("top level")
for key, value in result.items():
    print("   %-8s %s" % (key, type(value).__name__))

# ---------------------------------------------------------------- 3. parsed fields
heading("the same answers, parsed into a table")
print("   %-12s %-7s %-34s %-11s %s" % ("question id", "type", "primary answer",
                                         "confidence", "action"))
print("   " + "-" * 82)
for qid, a in answers.items():
    if a["type"] == "choice":
        primary = "%s (p=%.3f)" % (a["choice"], a["probabilities"][a["choice"]])
    elif a["type"] == "score":
        primary = "%.3f of %d  (expected, 0-based)" % (a["score"], len(a["legend"]) - 1)
    else:
        primary = "%.4f -> %s" % (a["noul"], "true" if a["noul"] > 0.5 else "false")
    print("   %-12s %-7s %-34s %-11.3f %.3f"
          % (qid, a["type"], primary, a["confidence"], a["action"]["act_probability"]))

heading("the fields each type carries")
print("   choice : type, choice, probabilities {label: p}, confidence, action")
print("   score  : type, score, legend {index: text}, probabilities {index: p},")
print("            confidence, action     (indexes and legend keys are strings: \"0\", \"1\", ...)")
print("   noul   : type, noul (P(true)), confidence, action")
print("   every answer also carries action['act_probability'], a separate two-way head over")
print("   the same encoder (see agent.cfg['act_costs'] / ['cost_wrong_act']). The typed")
print("   fields above are the decision; action is an auxiliary signal your policy may use.")

# ---------------------------------------------------------------- 4. usage
heading("usage: a batch total, not a context length")
one_question = {"churn_risk": QUESTIONS["churn_risk"]}
single = agent.predict(STATE, one_question)["usage"]
print("   all three questions : input=%d  output=%d"
      % (result["usage"]["input_tokens"], result["usage"]["output_tokens"]))
print("   churn_risk alone    : input=%d  output=%d"
      % (single["input_tokens"], single["output_tokens"]))
print("   The state is encoded once per question, so the batch total is the sum over the")
print("   questions in the call -- not the length of a shared context.")

print("""
   Two keys you supply decide the shape of the output: the state and the questions dict.
   The ids in that dict become the keys of `answers`, and nothing else is needed to consume
   the result -- there is no text, so there is nothing to parse.
   """)
