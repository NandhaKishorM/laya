"""Example 34 -- calibration: where the probabilities come from and what temperature does.

Laya's probabilities are trained against strictly proper scoring rules, so they are meant to be
read as probabilities. A per-(question type, option count) temperature then rescales the logits.
This example prints the shipped temperatures, shows which bucket a question selects, and
demonstrates the sharpening honestly on a real prediction.
"""

from _common import banner, describe, device_line, load

from laya.common import temp_bucket
from laya import QTYPE_NAMES, QTYPES, render_options

banner("34", "Calibration and temperature", """
    Laya is trained with a strictly proper scoring rule -- `laya.common.proper_reward` combines
    a log score, a spherical score and (for ordinal `score` questions) a ranked probability
    score. A proper rule is maximised in expectation only by reporting your honest beliefs, so
    the training signal pushes the head toward calibrated distributions rather than confident
    labels.

    The shipped checkpoint adds a temperature, one per (question type, option-count) bucket:

      z = logits / temperature_by_options[bucket]     # e.g. choice:11+ -> 0.1006
      p = softmax(z)

    `temperature` is a list of three floats indexed by question type
    (choice=0, score=1, noul=2); `temperature_by_options` holds the finer buckets and wins when
    its key exists. A temperature below 1 divides by less than one, i.e. multiplies the logits
    and sharpens the distribution. The English checkpoint's `choice:11+` bucket is 0.1006 --
    about a 9.9x logit multiplier -- so large option sets come back almost one-hot.

    The effect below is measured, not asserted: the same question is answered at the shipped
    temperature and again after setting that bucket to 1.0 in `agent.cfg`.
    """)

# --- a 12-option question lands in choice:11+ ----------------------------------------------
# A deliberately split request: it is both a billing problem and an account problem, so the
# honest distribution is not one-hot.
STATE = {"body": "I was charged twice and also my login is broken. Can someone look into it?"}
CATEGORY = {
    "type": "choice",
    "instructions": "Which department should handle this request?",
    "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "account": "login, seats, profile changes",
        "shipping": "delivery or fulfilment of a physical item",
        "security": "phishing, compromise, abuse reports",
        "legal": "contracts, compliance, data requests",
        "hr": "hiring, payroll, leave",
        "partnerships": "resellers, integrations, co-marketing",
        "analytics": "reports, dashboards, metrics",
        "training": "onboarding, documentation, how-to",
        "other": "none of the above",
    },
}
QUESTIONS = {
    "category": CATEGORY,
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

def bucket_for(question):
    internal = {"t": question["type"], "ins": question["instructions"],
                "crit": question.get("criteria")}
    k = len(render_options(internal))
    return temp_bucket(QTYPES[question["type"]], k), QTYPES[question["type"]], k

def top_line(probabilities, n=5):
    ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])[:n]
    return "  ".join("%s=%.4f" % kv for kv in ranked)

agent = load("english")
device_line(agent)
print("   temperature (by type)      : %s"
      % {QTYPE_NAMES[i]: round(t, 4) for i, t in enumerate(agent.cfg["temperature"])})
print("   temperature_by_options     : %s"
      % {k: round(v, 4) for k, v in sorted(agent.cfg["temperature_by_options"].items())})

print("\n   bucket selected per question:")
for qid, question in QUESTIONS.items():
    bucket, qt, k = bucket_for(question)
    scale = agent.cfg["temperature_by_options"].get(bucket, agent.cfg["temperature"][qt])
    print("   %-12s %-16s (%d options) -> t_scale=%.4f  (logit multiplier %.2fx)"
          % (qid, bucket, k, scale, 1.0 / scale))

print("\n   == the shipped temperature on the 12-option question ==")
shipped = agent.predict(STATE, QUESTIONS)["answers"]
print("   category: %s" % top_line(shipped["category"]["probabilities"]))
print("   top choice %r with confidence %.4f"
      % (shipped["category"]["choice"], shipped["category"]["confidence"]))

# --- neutralise exactly that bucket and ask the same question again ------------------------
BUCKET = "choice:11+"
original = agent.cfg["temperature_by_options"][BUCKET]
agent.cfg["temperature_by_options"][BUCKET] = 1.0
neutral = agent.predict(STATE, QUESTIONS)["answers"]
agent.cfg["temperature_by_options"][BUCKET] = original        # put the checkpoint back

print("\n   == the same call with %s set to 1.0 (logits not divided) ==" % BUCKET)
print("   category: %s" % top_line(neutral["category"]["probabilities"]))
print("   top choice %r with confidence %.4f"
      % (neutral["category"]["choice"], neutral["category"]["confidence"]))

print("\n   shipped t=%.4f vs neutral t=1.0: top probability %.4f -> %.4f, confidence %.4f -> %.4f"
      % (original, max(shipped["category"]["probabilities"].values()),
         max(neutral["category"]["probabilities"].values()),
         shipped["category"]["confidence"], neutral["category"]["confidence"]))
print("   the scores above are real: same weights, same state, only the divisor changed.")
print("   the other questions in the set:")
describe(shipped)
