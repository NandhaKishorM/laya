"""Example 12 -- one ticket, three languages, one call per language.

`router.predict(...)` routes and answers in a single call: the English checkpoint for
English, the multilingual checkpoint for Hindi and German, same questions throughout.
"""
from _common import STATE_EN, STATE_HI, banner, describe, heading, router

banner("12", "Route and predict, multilingually", """
    `router.predict()` is `route()` followed by one forward pass on the checkpoint the
    router picked. Nothing about the question schema changes between languages, and the
    result carries a `routing` block recording why the checkpoint was chosen.

    The same duplicate-charge complaint arrives in English, Hindi and German. English is
    read by `laya`; the other two are read by `laya-multilingual`. Watch `router.loaded`:
    the router builds the multilingual checkpoint only when the Hindi ticket shows up.
    """)

# The same three questions are asked of every language.
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
}

STATE_DE_TICKET = {
    "subject": "Doppelte Belastung auf Rechnung #4411",
    "body": "Wir wurden für März zweimal belastet und fordern eine Rückerstattung. "
            "Bitte erstatten Sie den doppelten Betrag heute.",
}

TICKETS = [
    ("English (en)", STATE_EN),
    ("Hindi (hi)", STATE_HI),
    ("German (de)", STATE_DE_TICKET),
]

r = router(max_loaded=1)       # keep one checkpoint hot; language switches evict
for label, state in TICKETS:
    out = r.predict(state, QUESTIONS)
    routing = out["routing"]
    heading("%s -> routed to %r" % (label, routing["model"]))
    print("   reason: %s" % routing["reason"])
    print("   source: %s" % routing["repo"])
    describe(out["answers"])
    print("   loaded: %r" % (r.loaded))

print("""
   The router switched checkpoints without being told which language it was reading, and
   the answers stay directly comparable: all three tickets land on `billing` and agree a
   refund is wanted. The Hindi ticket is the clearest -- the English checkpoint would be
   near-random on Devanagari, and would still report high confidence while failing.
   """)
