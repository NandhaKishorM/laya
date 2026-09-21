"""Example 21 -- long documents, context budget and truncation.

Laya gives every request a fixed window: the option head takes `head_max_len` and the state gets
the rest of `max_len`. A state larger than that is truncated. This example measures how much of
a long multilingual document survives at the default, then raises the window at runtime.
"""
from _common import banner, device_line, load
from laya.common import build_sequence, serialize_state

banner("21", "Long documents and truncation", """
    `laya-multilingual` defaults to max_len=1024 (head_max_len=256). The configured split
    reserves up to 256 tokens for the option head, leaving 768 for the state in the worst case
    (a small question uses less, so the measured room is a little larger). `build_sequence`
    fills that window with `state[:room]` -- the *head* of the document is kept and the tail is
    dropped. `build_sequence` has a `truncate_left=True` switch to keep the tail instead, but
    `predict`/`system_one` does not expose it. Anything longer than the window is invisible to
    the model, and nothing warns you.

    mmBERT is a RoPE encoder and supports up to 8192 tokens, so the window can be widened at
    runtime: `agent.cfg` is a plain dict and `system_one` reads it on every call. We raise
    `head_max_len` to 512 and `max_len` to 8192, then show the surviving document grow.

    `usage["input_tokens"]` is the batch total for the whole call -- every question's sequence
    summed together -- not a per-question count.
    """)

# --- a long, realistic multilingual state: one month of support-ticket exports --------------
BODIES = [
    "Customer says the invoice total does not match the purchase order and asks for a "
    "corrected statement before the end of the month.",
    "Der Kunde wurde zweimal belastet und bittet um eine Rueckerstattung des doppelten "
    "Betrages auf die urspruengliche Zahlungsmethode.",
    "ग्राहक का कहना है कि लॉगिन काम नहीं कर रहा और उसे आज ही सहायता चाहिए।",
    "El usuario no puede restablecer su contrasena y ya lo ha intentado varias veces sin exito.",
    "Le client signale que sa carte a ete bloquee apres trois tentatives de paiement refusees.",
    "O cliente relata que o pedido chegou danificado e pede a substituicao imediata do produto.",
    "The account owner wants to add two more seats to the enterprise plan and needs a quote.",
    "Der Kunde moechte sein Abonnement kuendigen und fragt nach der Frist fuer die Kündigung.",
    "The integration stopped syncing after the last deployment and the nightly job now fails.",
    "Il cliente chiede se il rimborso e gia stato elaborato e quando apparira sull estratto.",
]
PRIORITIES = ["low", "normal", "high", "critical"]
LANGUAGES = ["en", "de", "hi", "es", "fr", "pt"]


def build_document(records=64):
    """A JSON export of many support tickets -- a realistic long, structured state."""
    rows = []
    for i in range(1, records + 1):
        rows.append({
            "id": "TCK-%04d" % i,
            "language": LANGUAGES[i % len(LANGUAGES)],
            "priority": PRIORITIES[i % len(PRIORITIES)],
            "channel": "email" if i % 2 else "chat",
            "body": BODIES[i % len(BODIES)],
            "meta": {"received": "2026-03-%02dT%02d:15:00Z" % (1 + i % 28, i % 24),
                     "sla_hours": 24 if i % 3 else 4},
        })
    return {"export": "support-tickets-2026-03", "record_count": len(rows), "records": rows}


QUESTIONS = {
    "needs_human": {
        "type": "noul",
        "instructions": "Does any ticket in this export need immediate human attention?",
    },
    "volume": {
        "type": "score",
        "instructions": "How large is the support load in this export?",
        "criteria": ["a few tickets", "a normal week", "a heavy week", "an exceptional backlog"],
    },
}


def head_and_room(agent, question, max_len, head_max_len):
    """Sequence length with an empty state, which gives the space left for the real state."""
    q = {"t": question["type"], "ins": question["instructions"],
         "crit": question.get("criteria")}
    ids_empty, _ = build_sequence(agent.tok, "", q, max_len, head_max_len)
    head = len(ids_empty) - 1                    # [CLS] + instructions + options + [SEP]
    return head, max(0, max_len - len(ids_empty))


agent = load("multilingual")
device_line(agent)
doc = build_document()
doc_tokens = len(agent.tok(serialize_state(doc), add_special_tokens=False)["input_ids"])
print("   document: %d ticket records, %d characters, %d tokens (multilingual tokenizer)"
      % (doc["record_count"], len(serialize_state(doc)), doc_tokens))

visible_by_config = {}
for head_max_len, max_len in ((256, 1024), (512, 8192)):
    agent.cfg["head_max_len"], agent.cfg["max_len"] = head_max_len, max_len
    print("\n   == cfg max_len=%d, head_max_len=%d ==" % (max_len, head_max_len))
    for qid, question in QUESTIONS.items():
        head, room = head_and_room(agent, question, max_len, head_max_len)
        visible = min(doc_tokens, room)
        visible_by_config.setdefault(qid, []).append(visible)
        print("   %-12s head=%3d tokens, state room=%4d -> %4d/%d document tokens visible (%d%%)%s"
              % (qid, head, room, visible, doc_tokens, 100 * visible // doc_tokens,
                 "  TAIL DROPPED" if visible < doc_tokens else "  full document"))
    result = agent.predict(doc, QUESTIONS)
    print("   usage: %d input tokens across %d questions (batch total), %d output tokens"
          % (result["usage"]["input_tokens"], len(QUESTIONS), result["usage"]["output_tokens"]))
    for qid, a in result["answers"].items():
        if a["type"] == "noul":
            print("   %-12s noul=%.3f  conf=%.3f" % (qid, a["noul"], a["confidence"]))
        else:
            print("   %-12s score=%.2f/3  conf=%.3f" % (qid, a["score"], a["confidence"]))

first, second = visible_by_config["needs_human"]
print("\n   widening the window moved the visible document from %d to %d tokens: the tail is"
      % (first, second))
print("   only recoverable by setting the budget *before* the forward pass, not after.")
print("   the head stayed 36/48 tokens even at head_max_len=512 -- it is a ceiling, not a")
print("   reservation, and the unused head room flows back to the state.")

