"""Example 09 -- a conversation as the state.

`state` may be a list of `{"role", "content"}` turns instead of a single document, so
you can ask about an exchange -- who is upset, what they asked for -- without
summarising it into a string first.
"""
import json

from _common import banner, describe, device_line, heading, load

banner("09", "Conversation turns as the state", """
    A list of turns is serialised to JSON and becomes the state, exactly like a dict.
    That means the model sees the whole exchange, not just the last message, and
    questions can be about the arc of the conversation.

    Two exchanges are compared below. One is a calm tracking question that support
    resolves; the other is the same issue after two failed contacts, ending in a refund
    demand and a threat to cancel. `describe()` prints the four questions for each.
    """)

agent = load("english")
device_line(agent)

CALM = [
    {"role": "user", "content": "Hi, could you check on order 5521 for me? The tracking "
                                "has not updated since Friday."},
    {"role": "agent", "content": "Happy to help. The carrier shows a short delay in "
                                 "Cologne; it should move tomorrow."},
    {"role": "user", "content": "That is helpful, thank you. I will keep an eye on it."},
]
ESCALATING = [
    {"role": "user", "content": "This is the third time I have contacted you about order "
                                "5521 and nobody has fixed it."},
    {"role": "agent", "content": "I am sorry for the delay. I can see the order is stuck."},
    {"role": "user", "content": "Sorry is not good enough. I want a full refund today or I "
                                "am disputing the charge with my bank and cancelling my "
                                "account."},
]

TURNS_QUESTIONS = {
    "upset": {"type": "noul", "instructions": "Is the customer upset or frustrated?"},
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "cancel": {"type": "noul", "instructions": "Does the customer threaten to cancel or leave?"},
    "escalation": {"type": "score", "instructions": "How heated is this exchange?",
                   "criteria": ["calm and cooperative", "mild friction",
                                "clearly frustrated", "angry or threatening"]},
}

print("\n   the escalating exchange, as the model receives it:")
print("   %s" % json.dumps(ESCALATING[2], ensure_ascii=False))

for label, turns in (("calm exchange", CALM), ("escalating exchange", ESCALATING)):
    heading(label)
    for turn in turns:
        text = turn["content"]
        print("     %-5s | %s" % (turn["role"], text[:78] + ("..." if len(text) > 78 else "")))
    describe(agent.predict(turns, TURNS_QUESTIONS)["answers"], indent="   ")

print("""
   The calm exchange reads as not upset, no refund and no cancellation risk, with the
   escalation score between "calm" and "mild friction". The escalating one flips every
   noul and pushes the score to about 2, between "clearly frustrated" and "angry".

   Because the answer is computed from the whole turn list in one pass, this is a
   per-conversation decision you can rerun on every new message -- no summarisation
   step, no growing prompt of generated text.
   """)
