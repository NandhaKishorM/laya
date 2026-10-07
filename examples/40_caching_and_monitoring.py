"""Example 40 -- caching answers and monitoring confidence.

Two operational patterns: an in-memory answer cache keyed by the state and the question
definitions, and a confidence monitor that buckets a batch of answers into auto, review and
escalate -- reading back the temperature each of those numbers was actually scaled by.
"""
import hashlib
import json
import math
import time

from _common import laya, banner, device_line, heading, load
from laya.common import QTYPES, temp_bucket

banner("40", "Caching and monitoring", """
    The model is stateless and deterministic, so the same state and the same question
    definitions always return the same answer. That makes the call trivially cacheable: hash
    the inputs, keep the result, and pay for the forward pass once.

    Part A measures a real forward pass against a cache hit. Part B runs 15 tickets through
    the triage preset and gates on `answer_confidence`, the probability of the answer Laya
    reports, to sort every answer into auto-action, human review or escalation -- the
    operational shape example 41 builds on. Part C then prints the temperature this
    checkpoint actually applies to each of those five questions, because that is what decides
    whether the number is a calibrated one or the raw softmax the thresholds below are
    measuring.
    """)

agent = load("english")
device_line(agent)
QUESTIONS = laya.triage_questions()

# --- Part A: an in-memory answer cache -----------------------------------------------------
def cache_key(state, questions):
    """Hash exactly the inputs that determine the answer: the state and the question definitions."""
    payload = json.dumps({"state": state, "questions": questions}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


CACHE = {}


def cached_predict(state, questions):
    """Return (result, key, served_from_cache)."""
    key = cache_key(state, questions)
    if key in CACHE:
        return CACHE[key], key, True
    result = agent.predict(state, questions)
    CACHE[key] = result
    return result, key, False


STATE = {"ticket_id": "TCK-4001",
         "message": "I was charged twice for March and need the duplicate refunded today."}
OTHER = {"ticket_id": "TCK-4002",
         "message": "Where can I download the February invoice PDF?"}

agent.predict(STATE, QUESTIONS)                      # warm up the kernels for this shape
CACHE.clear()                                        # the timed call below must be a real miss
t0 = time.perf_counter()
miss_result, miss_key, miss_hit = cached_predict(STATE, QUESTIONS)
miss_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
hit_result, hit_key, hit = cached_predict(STATE, QUESTIONS)
hit_once_us = (time.perf_counter() - t0) * 1e6

REPEATS = 1000
t0 = time.perf_counter()
for _ in range(REPEATS):
    cached_predict(STATE, QUESTIONS)
hit_us = (time.perf_counter() - t0) * 1e6 / REPEATS

heading("part A: cache miss vs cache hit")
print("   key (state + questions) : %s" % miss_key)
print("   first call  : served_from_cache=%-5s  %7.2f ms   one forward pass" % (miss_hit, miss_ms))
print("   repeat call : served_from_cache=%-5s  %7.3f us   once, then %7.3f us averaged"
      % (hit, hit_once_us, hit_us))
print("   speed-up    : %.0fx" % (miss_ms * 1000 / hit_us))
print("   answers identical: %s" % (miss_result == hit_result))
print("   keys match       : %s" % (miss_key == hit_key))
print("   different state -> a different key: %s"
      % (cache_key(OTHER, QUESTIONS) != miss_key))

# Editing a question while keeping its id must invalidate the entry: the key hashes the
# definitions, not the ids.
EDITED = {qid: dict(q) for qid, q in QUESTIONS.items()}
EDITED["intent"]["criteria"] = dict(EDITED["intent"]["criteria"], other="anything else")
print("   same id, edited criteria -> a different key: %s" % (cache_key(STATE, EDITED) != miss_key))
print("   cache size after the miss: %d entry (the cached state)" % len(CACHE))
print("   A hit re-hashes the state and the whole question set, which is where the microseconds")
print("   go; the forward pass it avoids is the milliseconds. Hashing the definitions rather")
print("   than their ids means an edited criterion or instruction invalidates the entry instead")
print("   of quietly serving a stale answer. A cache shared across checkpoints should still be")
print("   salted with the checkpoint name.")

# --- Part B: confidence monitor over a batch ----------------------------------------------
AUTO_CONFIDENCE = 0.80            # at or above: act without a human
REVIEW_CONFIDENCE = 0.55          # between review and auto: a human checks it
DECISION_IDS = ["intent", "is_urgent", "refund_requested", "churn_risk"]

TICKETS = [
    ("dup charge", "I was charged twice for March and need the duplicate refunded today."),
    ("api outage", "Your API returns 502 errors for our whole team since 9am; production is down."),
    ("invoice pdf", "Where can I download the February invoice PDF?"),
    ("enterprise pricing", "We are evaluating the enterprise tier; can you send pricing?"),
    ("cancel now", "I want to cancel my subscription immediately."),
    ("vague", "Hi, I need some help with my account."),
    ("feature idea", "It would be great if the export button supported CSV. No rush."),
    ("slow dash", "The dashboard is slow but usable, and the colours look a bit off."),
    ("mixed", "Please refund me and also fix the login bug we discussed."),
    ("renewal deadline", "Our renewal is due tomorrow and the card keeps failing; help before 5pm."),
    ("angry churn", "Nobody has replied in three days. This is unacceptable. We will cancel."),
    ("seat count", "How do I change my seat count?"),
    ("all broken", "Everything is broken!!! Nothing works!!!"),
    ("plan diff", "Can you explain the difference between the Pro and Enterprise plans?"),
    ("vat number", "The invoice shows a different VAT number than our registration."),
]


def bucket(confidence):
    """Our thresholds, not the model's: application policy over the reported answer probability."""
    if confidence >= AUTO_CONFIDENCE:
        return "auto"
    if confidence >= REVIEW_CONFIDENCE:
        return "review"
    return "escalate"


def option_count(question):
    """k: the number of options this answer was scored over, the other half of the bucket key."""
    if question["type"] == "noul":
        return 2
    return len(question["criteria"])


def scale_for(agent, qtype, k):
    """(bucket, temperature really applied, whether the bucket map carried it) for one shape.

    A replay of `Agent.predict_batch`'s own lookup -- `temperature_by_options` keyed by
    `temp_bucket`, falling back to the per-type `temperature` -- read off the loaded agent rather
    than restated here, so this page cannot describe a scaling the code does not perform. A
    temperature of 1.0 leaves the logits unscaled, which is the difference between a calibrated
    probability and the raw softmax.
    """
    name = temp_bucket(QTYPES[qtype], k)
    if name in agent.temperature_by_options:
        return name, agent.temperature_by_options[name], True
    return name, agent.temperature[QTYPES[qtype]], False


def clamped_buckets(agent):
    """Buckets whose shipped temperature the runtime refused, as `name=shipped -> applied`.

    This is the condition behind the loader's own warning: the checkpoint ships a value outside
    the range `common.clamp_temperature` accepts, so the scaling published for it is not the one
    recorded on the Hub.
    """
    return ["%s=%.4f -> %.4f" % (name, agent.temperature_by_options_raw[name], applied)
            for name, applied in agent.temperature_by_options.items()
            if name in agent.temperature_by_options_raw
            and agent.temperature_by_options_raw[name] != applied]


def entropy_confidence(probs):
    """1 - H/log(k) over a reported distribution: `confidence`'s definition, recomputed by hand.

    The probabilities a page can read back are rounded to 4 dp while the agent computed the field
    from unrounded ones, so this is compared against the reported value rather than substituted
    for it.
    """
    k = len(probs)
    if k < 2:
        return 1.0
    ent = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return 1.0 - ent / math.log(k)


answer_buckets = {"auto": 0, "review": 0, "escalate": 0}
per_question = {}
rows = []

heading("part B: confidence monitor over %d tickets" % len(TICKETS))
print("   thresholds: auto >= %.2f, review >= %.2f, escalate below"
      % (AUTO_CONFIDENCE, REVIEW_CONFIDENCE))
print("   %-19s %-17s %-7s %-24s %s"
      % ("ticket", "intent", "p", "weakest", "action"))

for name, message in TICKETS:
    answers = agent.predict({"message": message}, QUESTIONS)["answers"]
    for qid, answer in answers.items():
        b = bucket(answer["answer_confidence"])
        answer_buckets[b] += 1
        per_question.setdefault(qid, {"auto": 0, "review": 0, "escalate": 0})[b] += 1
    weakest_id = min(DECISION_IDS, key=lambda q: answers[q]["answer_confidence"])
    weakest_conf = answers[weakest_id]["answer_confidence"]
    top_intent = max(answers["intent"]["probabilities"].values())
    rows.append((name, answers, weakest_id, weakest_conf, bucket(weakest_conf), top_intent))
    print("   %-19s %-17s %-7.2f %-24s %s"
          % (name, answers["intent"]["choice"], top_intent,
             "%s=%.4f" % (weakest_id, weakest_conf), bucket(weakest_conf).upper()))

heading("every answer, bucketed")
print("   all %d answers : auto=%d  review=%d  escalate=%d"
      % (len(TICKETS) * len(QUESTIONS), answer_buckets["auto"], answer_buckets["review"],
         answer_buckets["escalate"]))
print("   %-19s %-7s %-6s %-7s %s" % ("question", "type", "auto", "review", "escalate"))
for qid, counts in per_question.items():
    print("   %-19s %-7s %-6d %-7d %d"
          % (qid, QUESTIONS[qid]["type"], counts["auto"], counts["review"], counts["escalate"]))
frustration = per_question["frustration"]
print("   `frustration` is a %d-level score, so the probability it puts on the reported level"
      % len(QUESTIONS["frustration"]["criteria"]))
print("   is modest even when the answer is informative: %d of %d landed in escalate. Gate on"
      % (frustration["escalate"], len(TICKETS)))
print("   the choice and noul answers you actually act on, not on the score.")

heading("states that look unusual")
print("   %-19s %-17s %-7s %-24s %s"
      % ("ticket", "intent", "p(intent)", "weakest", "why"))
for name, answers, weakest_id, weakest_conf, action, top_intent in rows:
    if action != "escalate":
        continue
    why = "intent split: p=%.2f" % top_intent if top_intent < 0.5 else "%s is soft" % weakest_id
    print("   %-19s %-17s %-7.2f %-24s %s"
          % (name, answers["intent"]["choice"], top_intent,
             "%s=%.4f" % (weakest_id, weakest_conf), why))

heading("part C: what each answer's number was actually scaled by")
print("   %-19s %-7s %-4s %-13s %-9s %s"
      % ("question", "type", "k", "bucket", "applied", "source"))
unscaled = []
for qid, q in QUESTIONS.items():
    k = option_count(q)
    name, applied, in_map = scale_for(agent, q["type"], k)
    if abs(applied - 1.0) < 1e-3:
        unscaled.append(qid)
    print("   %-19s %-7s %-4d %-13s %-9.4f %s"
          % (qid, q["type"], k, name, applied,
             "bucket map" if in_map else "per-type default"))
clamped = clamped_buckets(agent)
if clamped:
    print("   shipped but refused by the runtime: %s" % ", ".join(clamped))
print("   at a temperature indistinguishable from 1.0: %s (%d of %d question shapes). For those"
      % (", ".join("`%s`" % q for q in unscaled) or "none", len(unscaled), len(QUESTIONS)))
print("   the published number is the raw softmax, so the split above sorts an uncalibrated")
print("   answer by a cutoff nothing has measured.")

vague = next(answers for name, answers, *_ in rows if name == "vague")
p = list(vague["intent"]["probabilities"].values())
recomputed = entropy_confidence(p)
reported = vague["intent"]["confidence"]
print("""
   The monitor gates on `answer_confidence`: the probability of the reported answer, `max(p)`,
   defined the same way on every question type. Whether that probability is a calibrated one is a
   claim about the temperatures rather than about the field -- it is the quantity temperature
   scaling fits and the number `min_confidence` reads, so it carries the meaning a threshold
   assumes only where a fitted temperature was really applied. Part C prints the value this
   checkpoint applies to each shape this page asks for, the bucket it refused to apply, and the
   answers whose value sits at 1.0. The README's Calibration section is the loop that closes it.

   `confidence` is a different measure -- on `choice` and `score` it is 1 - H/log(k), how
   concentrated the whole distribution is -- and the README warns against carrying a threshold
   over from it. For the "vague" ticket, recomputing it by hand from the published probabilities
   gives %.4f against the %.4f the agent reported for the same answer, %.6f apart; any gap is
   rounding, because `probabilities` is published at 4 dp while the agent scaled unrounded logits.
   `answer_confidence` for that answer is %.2f. The two entropy figures agree because they are the
   same formula, which is why this page prints the reported field beside its own arithmetic
   instead of substituting the arithmetic for the field. Read together: the model is stating that
   the message does not identify one intent, and the monitor escalates it.

   The thresholds, the 0.5 spam-style gates and the auto/review/escalate split are ours. The
   probabilities are the model's, and Part C is what separates the share of their meaning the
   checkpoint carries from the share that is still yours to measure. Example 41 turns exactly this
   monitor into a service policy.
   """ % (recomputed, reported, abs(recomputed - reported),
          vague["intent"]["answer_confidence"]))
