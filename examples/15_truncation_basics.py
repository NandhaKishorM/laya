"""Example 15 -- what happens when the state does not fit.

Builds a document whose key facts sit at the start and at the end, prints the checkpoint's
window, and shows that truncation keeps the head: the start question is answered and the
end question is not.
"""
from _common import banner, device_line, heading, load

banner("15", "When the state does not fit", """
    Every call gets a fixed window: `max_len` tokens for the whole sequence, of which the
    question text and its options may take up to `head_max_len`. The state gets what is
    left, and a state longer than that is cut down to its head with no warning.

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

result = agent.predict(document, QUESTIONS)
heading("one fact from the head, one from the tail")
for qid in QUESTIONS:
    answer = result["answers"][qid]
    where = ("evidence in the first sentence" if qid == "duplicate_invoice"
             else "evidence only in the closing line")
    print("   %-18s (%-32s)  noul=%.3f  conf=%.3f"
          % (qid, where, answer["noul"], answer["confidence"]))

print("\n   usage: %d input tokens for %d questions in this call" % (result["usage"]["input_tokens"],
                                                                     len(QUESTIONS)))
alone = agent.predict(document, {"cancel_threat": QUESTIONS["cancel_threat"]})
print("   the same document with one question: %d input tokens -- exactly max_len=%d"
      % (alone["usage"]["input_tokens"], max_len))
print("   so usage[\"input_tokens\"] is the batch total across the questions in the call:")
print("   each question contributes its own (capped) sequence.")

print("""
   `duplicate_invoice` reads the head and answers %.3f. `cancel_threat` answers %.3f with
   confidence %.3f -- the model is sure no cancellation is mentioned, because the only
   mention was cut off. Truncation is silent in both directions: nothing warns you, and
   the answer looks as certain as any other. Raising the window at runtime is example 37.
   """ % (result["answers"]["duplicate_invoice"]["noul"], result["answers"]["cancel_threat"]["noul"],
          result["answers"]["cancel_threat"]["confidence"]))
