"""Example 15 -- what happens when the state does not fit.

Builds a document whose key facts sit at the start and at the end, prints the checkpoint's
window, and shows that truncation keeps the head: the start question is answered and the
end question is not. Then reads the `usage` report that says so, because the answers do not.
"""
from _common import banner, device_line, heading, load

banner("15", "When the state does not fit", """
    Every call gets a fixed window: `max_len` tokens for the whole sequence, of which the
    question text and its options may take up to `head_max_len`. The state gets what is
    left, and a state longer than that is cut down to its head. The answers do not say so;
    the usage report does.

    The document below is a long case file with one key fact in its first sentence and a
    different key fact in its closing line. Two questions are asked about it -- one whose
    evidence is at the very start, one whose evidence is at the very end -- and the second
    one has nothing to read.
    """)

agent = load("english")
device_line(agent)

HEAD = ("CASE HEADER: invoice #4411 was charged twice for order ORD-7781 in March; "
        "the duplicate charge on that invoice is the issue. ")
FILLER = ("Routine scan at the regional depot: the parcel was transferred to the next "
          "carrier without incident and the tracking record was updated automatically. ")
TAIL = ("CLOSING NOTE: the customer said they will cancel their account unless the "
        "duplicate charge is refunded today.")
document = HEAD + FILLER * 40 + TAIL

max_len = agent.cfg["max_len"]
head_max_len = agent.cfg["head_max_len"]
doc_tokens = len(agent.tok(document, add_special_tokens=False)["input_ids"])
floor = max_len - head_max_len

heading("the state and the window")
print("   document        : %d characters, about %d tokens in this checkpoint's tokenizer"
      % (len(document), doc_tokens))
print("   agent.cfg       : max_len=%d, head_max_len=%d" % (max_len, head_max_len))
print("   the question head and options may take up to %d tokens, so the state gets at" % head_max_len)
print("   least max_len - head_max_len = %d tokens. That is %.0f%% of this document; the rest"
      % (floor, 100.0 * floor / doc_tokens))
print("   cannot reach the model, however the questions are worded.")

heading("what the model can see")
print("   the document starts with:")
print("   | %s..." % document[:126].rstrip())
print("   ...and its only mention of cancellation is here, at character %d:"
      % document.index("CLOSING NOTE"))
print("   | %s" % document[document.index("CLOSING NOTE"):][:120].rstrip())

QUESTIONS = {
    "duplicate_invoice": {
        "type": "noul",
        "instructions": "Is the duplicate charge on invoice #4411?",
    },
    "cancel_threat": {
        "type": "noul",
        "instructions": "Does the customer threaten to cancel their account?",
    },
}

# `state_tokens` / `state_tokens_dropped` are opt-in: they count the WHOLE serialized state,
# which means tokenizing all of it, while a prediction only ever reads as much as a question's
# window can hold. `truncated` / `truncated_questions` need no flag and are exact without it,
# so ask for the counts only when you want to know how much was dropped.
result = agent.predict(document, QUESTIONS, state_token_counts=True)
heading("one fact from the head, one from the tail")
for qid in QUESTIONS:
    answer = result["answers"][qid]
    where = ("evidence in the first sentence" if qid == "duplicate_invoice"
             else "evidence only in the closing line")
    print("   %-18s (%-32s)  noul=%.3f  conf=%.3f"
          % (qid, where, answer["noul"], answer["confidence"]))

usage = result["usage"]
print("\n   usage: %d input tokens for %d questions in this call" % (usage["input_tokens"],
                                                                     len(QUESTIONS)))
alone = agent.predict(document, {"cancel_threat": QUESTIONS["cancel_threat"]},
                      state_token_counts=True)
print("   the same document with one question: %d input tokens -- exactly max_len=%d"
      % (alone["usage"]["input_tokens"], max_len))
print("   so usage[\"input_tokens\"] is the batch total across the questions in the call:")
print("   each question contributes its own (capped) sequence.")
print("   and the token counts say nothing about the cut. The keys after them are the report:")
print("   `truncated`             : %s" % usage["truncated"])
print("   `state_tokens`          : %d tokens of state were tokenized" % usage["state_tokens"])
print("   `state_tokens_dropped`  : %d of them never reached the model"
      % usage["state_tokens_dropped"])
print("   `truncated_questions`   : %s"
      % (", ".join("`%s`" % qid for qid in usage["truncated_questions"]) or "none"))
fits = agent.predict("CASE HEADER: invoice #4411 was charged twice. Please refund it.", QUESTIONS,
                     state_token_counts=True)
print("   a state that fits the window: `truncated`=%s, `state_tokens_dropped`=%d, "
      "`truncated_questions`=%s" % (fits["usage"]["truncated"], fits["usage"]["state_tokens_dropped"],
                                    fits["usage"]["truncated_questions"]))

print("""
   `duplicate_invoice` reads the head and answers %.3f. `cancel_threat` answers %.3f with
   confidence %.3f -- the model is sure no cancellation is mentioned, because the only
   mention was cut off. Neither answer shows it: a truncated answer looks exactly as
   certain as a well-founded one. `truncated` is what says so, and `truncated_questions`
   names which of them were built on the cut, so a caller reads the report instead of
   estimating from the length of what it sent. Raising the window at runtime is example 37.
   """ % (result["answers"]["duplicate_invoice"]["noul"], result["answers"]["cancel_threat"]["noul"],
          result["answers"]["cancel_threat"]["confidence"]))
