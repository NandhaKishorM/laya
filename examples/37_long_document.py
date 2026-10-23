"""Example 37 -- long documents, context budget and truncation.

Laya gives every request a fixed window: the option head takes `head_max_len` and the state gets
the rest of `max_len`. A state larger than that is truncated. This example measures how much of
a long multilingual document survives at the default, raises the window at runtime, and then
shows that which *end* of the state survives is decided by the state's own type.
"""
from _common import banner, device_line, heading, load
from laya.common import build_sequence, serialize_state, state_room

banner("37", "Long documents and truncation", """
    `laya-multilingual` defaults to max_len=1024 (head_max_len=256). The configured split
    reserves up to 256 tokens for the option head, leaving 768 for the state in the worst case
    (a small question uses less, so the measured room is a little larger). `build_sequence`
    fills that window with `state[:room]` -- the *head* of the document is kept and the tail is
    dropped. Which end that is comes from the state's own type, not from a parameter: `predict`
    passes `truncate_left=isinstance(state, list)`, so a *list* of turns is clamped from the other
    side and keeps its tail, while a string or dict keeps its head. `state_room` documents the same
    split. And the cut is not invisible to the caller: every call reports it in `usage` --
    `truncated`, `state_tokens`, `state_tokens_dropped`, `truncated_questions` -- which this
    example prints below.

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
default_cfg = dict(agent.cfg)
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
    # The two counts are opt-in, because counting the whole state means tokenizing the whole
    # state; `truncated` is reported either way. See examples/15_truncation_basics.py.
    result = agent.predict(doc, QUESTIONS, state_token_counts=True)
    usage = result["usage"]
    print("   usage: %d input tokens across %d questions (batch total), %d output tokens"
          % (usage["input_tokens"], len(QUESTIONS), usage["output_tokens"]))
    for qid, a in result["answers"].items():
        if a["type"] == "noul":
            print("   %-12s noul=%.3f  conf=%.3f" % (qid, a["noul"], a["confidence"]))
        else:
            print("   %-12s score=%.2f/3  conf=%.3f" % (qid, a["score"], a["confidence"]))
    # The answers above do not look cut off -- `truncated` is the only thing that says so.
    print("   the call reports the clamp: `truncated`=%s, `state_tokens`=%d,"
          " `state_tokens_dropped`=%d, `truncated_questions`=%s"
          % (usage["truncated"], usage["state_tokens"], usage["state_tokens_dropped"],
             ", ".join("`%s`" % qid for qid in usage["truncated_questions"]) or "none"))

first, second = visible_by_config["needs_human"]
print("\n   widening the window moved the visible document from %d to %d tokens: the tail is"
      % (first, second))
print("   recoverable by setting the budget *before* the forward pass, not after -- and by")
print("   handing the state over as a list, which is the section below.")
print("   the head stayed 36/48 tokens even at head_max_len=512 -- it is a ceiling, not a")
print("   reservation, and the unused head room flows back to the state.")

# --- which end of the state survives: the same tokens, two different types -------------------
heading("which end of the state survives")
agent.cfg["head_max_len"], agent.cfg["max_len"] = default_cfg["head_max_len"], default_cfg["max_len"]
max_len, head_max_len = agent.cfg["max_len"], agent.cfg["head_max_len"]

CLOSING = "CLOSING NOTE: the last record of this export mentions the auditors."
turns = [rec["body"] for rec in doc["records"]] + [CLOSING]
turns_text = serialize_state(turns)          # the exact characters a list state serializes to
TAIL_Q = {"asked_last": {"type": "noul",
                         "instructions": "Does the closing note of this export mention the auditors?"}}
# `predict` picks the clamp side with `truncate_left = isinstance(state, list)`, on both agents.
# Rebuild that slice here so the two words that decide it -- first and last -- can be looked at.
q = {"t": "noul", "ins": TAIL_Q["asked_last"]["instructions"], "crit": None}
room = state_room(agent.tok, q, max_len, head_max_len)
print("   %d state tokens, %d of them fit (`state_room`), so %d are dropped either way."
      % (len(agent.tok(turns_text, add_special_tokens=False)["input_ids"]), room,
         len(agent.tok(turns_text, add_special_tokens=False)["input_ids"]) - room))
for label, state in (("a string", turns_text), ("a list", turns)):
    tokens = agent.tok(serialize_state(state), add_special_tokens=False)["input_ids"]
    left = isinstance(state, list)
    kept = tokens[max(0, len(tokens) - room):] if left else tokens[:room]
    answer = agent.predict(state, TAIL_Q, state_token_counts=True)
    usage = answer["usage"]
    print("   the same %d tokens as %-9s: `truncate_left` = isinstance(state, list) -> %s,"
          " so `state_tokens_dropped`=%d (my slice: %d)"
          % (len(tokens), label, left, usage["state_tokens_dropped"], len(tokens) - len(kept)))
    print("   %swhat the model actually read starts %r and ends %r"
          % (" " * 24, agent.tok.decode(kept)[:44], agent.tok.decode(kept)[-44:]))
    print("   %sthe closing note is in it: %s   noul=%.3f  conf=%.3f"
          % (" " * 24, "auditors" in agent.tok.decode(kept),
             answer["answers"]["asked_last"]["noul"], answer["answers"]["asked_last"]["confidence"]))

print("""
   Same characters, same `state_tokens`, same `state_tokens_dropped` -- the type alone decided
   which end of it the model read. The string kept its head, so the auditors sentence never
   reached the model; the list was clamped from the left, so it did. `truncate_left` is not a
   parameter a caller passes: `predict` takes it from `isinstance(state, list)`, and `state_room`
   is what tells you how much of either end fits.

   And the answers disagree with that: both say yes, because a wall of angry tickets is enough to
   make the head fire whatever survived in it. That is the reason the report exists. `truncated`
   and `truncated_questions` are the only part of this result that says the evidence was cut.""")

