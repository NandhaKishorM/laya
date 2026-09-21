"""Example 14 -- preload, LRU residency, attach and unload.

Shows what stays hot in a `Router`: lazy loading with `max_loaded=1` evicting on a language
switch, the reload cost that `preload` removes, `max_loaded=2` LRU behaviour, `attach()`
adopting an already-built agent, and `unload()`.
"""
import time

from _common import STATE_EN, STATE_HI, banner, heading, router

banner("14", "Preload and memory", """
    A cold checkpoint build costs seconds; language detection costs microseconds. At the
    default `max_loaded=1`, traffic that alternates languages rebuilds a model on every
    switch. `router.loaded` is the residency view, least-recently-used first, so you can
    watch models enter and leave as traffic moves between languages.

    A server preloads instead. Here we also show `attach()`, which registers an Agent the
    process already built rather than paying for -- and holding -- a duplicate copy.
    """)

QUESTION = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts",
                     "other": "everything else"},
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

r = router(max_loaded=1)

heading("1. lazy load: one checkpoint hot")
en_agent = r.load("english")            # cold build #1
out = r.predict(STATE_EN, QUESTION)     # first forward also pays MPS kernel compilation
print("   routed to %r via %r" % (out["routing"]["model"], out["routing"]["reason"]))
print("   loaded: %r   max_loaded=%d" % (r.loaded, r.max_loaded))

heading("2. a language switch evicts")
ml_agent = r.load("multilingual")       # cold build #2 -> evicts english (LRU)
decision = r.route(STATE_HI, QUESTION)
print("   routed to %r" % decision["model"])
print("   reason: %s" % decision["reason"])
print("   loaded: %r   (english is gone)" % r.loaded)

heading("3. the reload cost that preload avoids")
t0 = time.perf_counter()
r.load("english")                       # cold build #3 -> evicts multilingual
reload_s = time.perf_counter() - t0
print("   reloading the 421M-parameter english checkpoint after eviction")
print("   measured cold build: %.1f s" % reload_s)
print("   loaded: %r" % r.loaded)
print("   With max_loaded=1 and alternating languages, a server pays that on every switch.")

heading("4. max_loaded=2: least-recently-used is dropped")
lru = router(max_loaded=2)
lru.attach("english", en_agent)         # adopt built agents: no duplicate loads
lru.attach("multilingual", ml_agent)
print("   after attaching two: loaded=%r" % lru.loaded)
td_agent = lru.load("typed-decisions")  # cold build #4 -> evicts the LRU entry, english
print("   after loading a third: loaded=%r" % lru.loaded)
print("   english was evicted, not multilingual: english was the least recently used.")

heading("5. preload: all three resident")
hot = router()
hot.attach("english", en_agent)
hot.attach("multilingual", ml_agent)
hot.attach("typed-decisions", td_agent)
hot.preload()                           # raises max_loaded to fit everything preloaded
print("   loaded: %r" % hot.loaded)
print("   max_loaded=%d   (in a server you would just write Router(preload=True))" % hot.max_loaded)
decision = hot.route(STATE_HI, QUESTION)
print("   a Hindi route now costs detection only: %r" % decision.reason)

heading("6. attach(): adopt an Agent built elsewhere")
adopted = router(max_loaded=1)
adopted.attach("english", en_agent)
print("   loaded: %r   max_loaded=%d" % (adopted.loaded, adopted.max_loaded))
print("   no cold build: the router registered the 421M-parameter Agent already in memory.")

heading("7. unload(): free resident memory")
adopted.unload("english")
print("   adopted.unload('english') -> %r" % adopted.loaded)
hot.unload()
print("   hot.unload()              -> %r" % hot.loaded)
lru.unload()
print("   lru.unload()              -> %r" % lru.loaded)
print("""
   Rule of thumb: `Router(max_loaded=1)` is fine for a single-language service; anything
   that alternates languages should `preload` (or raise `max_loaded`), because the reload
   measured above is thousands of times more expensive than the routing itself.
   """)
