"""Cache decisions by state + questions, and skip inference on a hit.

A start hook calls `ctx.skip(results)`; the engine skips the forward pass and still runs
the end hooks.

    python examples/hooks/cache.py
"""
import hashlib
import json

import laya

CACHE = {}


def cache_key(state, questions):
    payload = json.dumps({"state": state, "questions": questions}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_read(ctx):
    hit = CACHE.get(cache_key(ctx.states[0], ctx.questions))
    if hit is not None:
        ctx.skip([hit])


def cache_write(ctx):
    if ctx.results:
        CACHE[cache_key(ctx.states[0], ctx.questions)] = ctx.results[0]


agent = laya.load("convaiinnovations/laya",
                  on_predict_start=cache_read, on_predict_end=cache_write)

STATE = "I was charged twice for the same invoice."
QUESTIONS = {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}

first = agent.system_one(STATE, QUESTIONS)     # runs the model, fills the cache
second = agent.system_one(STATE, QUESTIONS)    # served from CACHE, no forward pass
print("cache entries:", len(CACHE))
print("same answer:", first["answers"] == second["answers"])
