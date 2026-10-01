"""Cache decisions, and skip inference on a hit.

A start hook calls `ctx.skip(results)`; the engine skips the forward pass and still runs
the end hooks. The key has to cover everything the answer depends on, and a hook has to cover
everything the call carries: hooks fire once per call, and `predict_batch` calls them with
every state of the batch at once.

    python examples/hooks/cache.py
"""
import hashlib
import json

import laya

CACHE = {}
SKIPS = []            # one entry per call `cache_read` served, so the prints below are checkable


def cache_key(ctx, index):
    """One entry per question the model is actually asked, for one state of the call.

    Deliberately not `sort_keys=True`: a choice question's criteria order is positional, so two
    orders are two questions, and `_question_schema` in `laya/router.py` keeps them apart for the
    same reason. Sorting the keys folds them into one entry, and the second caller gets the first
    caller's numbers. `ctx.model` and the token budget belong in the key for the same reason: on
    the Router one hook set serves three checkpoints, and a smaller `max_len` truncates the state.
    """
    payload = json.dumps([ctx.states[index], ctx.questions, ctx.model,
                          ctx.max_len, ctx.head_max_len], default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_read(ctx):
    hits = [CACHE.get(cache_key(ctx, i)) for i in range(len(ctx.states))]
    if all(hit is not None for hit in hits):
        SKIPS.append(len(hits))
        ctx.skip(hits)   # one per state: `skip` replaces the whole call, not its first result


def cache_write(ctx):
    for i, result in enumerate(ctx.results or []):
        CACHE[cache_key(ctx, i)] = result


agent = laya.load("convaiinnovations/laya",
                  on_predict_start=cache_read, on_predict_end=cache_write)

STATE = "I was charged twice for the same invoice."
OTHER = "Where do I change my notification settings?"
CRITERIA = {"refund": "give me money back", "cancel": "stop the service",
            "other": "anything else"}
QUESTIONS = {"ask": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": dict(CRITERIA)}}
# Same labels and descriptions, criteria written in the other order.
REORDERED = {"ask": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": {"other": CRITERIA["other"], "refund": CRITERIA["refund"],
                                  "cancel": CRITERIA["cancel"]}}}

first = agent.system_one(STATE, QUESTIONS)     # runs the model, fills the cache
again = agent.system_one(STATE, QUESTIONS)     # served from CACHE, no forward pass
agent.system_one(STATE, REORDERED)             # runs too: a reordered question is a new entry
print("cache entries:", len(CACHE))            # 2, not 1
print("same answer:", first["answers"] == again["answers"])

# Warm the second state before the batch. `cache_read` skips only when *every* state of the call
# has an entry, so a batch that mixes one warm state with one cold state runs a full forward.
agent.system_one(OTHER, QUESTIONS)
# Batched, in the other order: both states are cached, so the whole call is served.
served = agent.predict_batch([OTHER, STATE], QUESTIONS)
print("results for a 2-state batch:", len(served))
print("one answer per state, in the caller's order:",
      [r["answers"]["ask"]["choice"] for r in served])
print("calls served without a forward pass:", len(SKIPS))    # 2: `again`, and this batch
