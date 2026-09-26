"""Example 34 -- calibration: where the probabilities come from and what temperature does.

Laya's probabilities are trained against strictly proper scoring rules, so they are meant to be
read as probabilities. A per-(question type, option count) temperature then rescales the logits.
This example prints the shipped temperatures, shows which bucket a question selects, and
demonstrates the sharpening honestly on a real prediction.
"""

from _common import banner, describe, device_line, load

from laya.common import TEMP_MAX, TEMP_MIN, temp_bucket
from laya import QTYPE_NAMES, QTYPES, render_options

banner("34", "Calibration and temperature", """
    Laya is trained with a strictly proper scoring rule -- `laya.common.proper_reward` combines
    a log score, a spherical score and (for ordinal `score` questions) a ranked probability
    score. A proper rule is maximised in expectation only by reporting your honest beliefs, so
    the training signal pushes the head toward calibrated distributions rather than confident
    labels.

    The shipped checkpoint adds a temperature, one per (question type, option-count) bucket:

      z = logits / temperature_by_options[bucket]     # e.g. choice:11+ -> 0.1006 shipped
      p = softmax(z)

    `temperature` is a list of three floats indexed by question type
    (choice=0, score=1, noul=2); `temperature_by_options` holds the finer buckets and wins when
    its key exists. A temperature below 1 divides by less than one, i.e. multiplies the logits
    and sharpens the distribution.

    One thing to know before reading a checkpoint's numbers: the agent clamps every temperature
    into [0.5, 5] when it loads a checkpoint and applies the clamped copy. The English
    checkpoint ships `choice:11+` at 0.1006, which is below the floor, so what actually runs is
    0.5 -- about a 2x logit multiplier, not the 9.9x the raw value suggests. This example prints
    both and measures the sharpening on a real prediction.
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
raw = agent.temperature_by_options_raw
applied = agent.temperature_by_options
print("   temperature (by type)      : %s"
      % {QTYPE_NAMES[i]: round(t, 4) for i, t in enumerate(agent.temperature)})
print("   temperature_by_options raw : %s"
      % {k: round(v, 4) for k, v in sorted(raw.items())})
print("   temperature_by_options used: %s"
      % {k: round(v, 4) for k, v in sorted(applied.items())})
print("   the agent clamps into [%.1f, %.1f]; `predict()` reads the clamped map." % (TEMP_MIN, TEMP_MAX))

print("\n   bucket selected per question:")
for qid, question in QUESTIONS.items():
    bucket, qt, k = bucket_for(question)
    raw_scale = float(raw.get(bucket, agent.temperature_raw[qt]))
    scale = applied.get(bucket, agent.temperature[qt])
    clamped = "" if raw_scale == scale else "   (raw %.4f clamped to the floor)" % raw_scale
    print("   %-12s %-16s (%d options) -> t_scale=%.4f  (logit multiplier %.2fx)%s"
          % (qid, bucket, k, scale, 1.0 / scale, clamped))

print("\n   == the applied temperature on the 12-option question ==")
shipped = agent.predict(STATE, QUESTIONS)["answers"]
print("   category: %s" % top_line(shipped["category"]["probabilities"]))
print("   top choice %r with answer_confidence %.4f"
      % (shipped["category"]["choice"], shipped["category"]["answer_confidence"]))

# --- neutralise exactly that bucket and ask the same question again ------------------------
# `agent.temperature_by_options` is the clamped copy predict() applies, so this is the map to
# edit. `agent.cfg` holds the checkpoint's raw values and is not read per call.
BUCKET = "choice:11+"
original = agent.temperature_by_options[BUCKET]
agent.temperature_by_options[BUCKET] = 1.0
neutral = agent.predict(STATE, QUESTIONS)["answers"]
agent.temperature_by_options[BUCKET] = original        # put the applied temperature back

print("\n   == the same call with %s set to 1.0 (logits not divided) ==" % BUCKET)
print("   category: %s" % top_line(neutral["category"]["probabilities"]))
print("   top choice %r with answer_confidence %.4f"
      % (neutral["category"]["choice"], neutral["category"]["answer_confidence"]))

print("\n   applied t=%.4f vs neutral t=1.0: top probability %.4f -> %.4f, answer_confidence %.4f -> %.4f"
      % (applied[BUCKET], max(shipped["category"]["probabilities"].values()),
         max(neutral["category"]["probabilities"].values()),
         shipped["category"]["answer_confidence"], neutral["category"]["answer_confidence"]))
print("   the scores above are real: same weights, same state, only the divisor changed.")
print("   the other questions in the set:")
describe(shipped)
