"""Example 20 -- the three checkpoints, side by side.

Loads `english`, `multilingual` and `typed-decisions` in turn and asks the same small
question set of an English ticket and a Hindi one, so the choice of checkpoint is visible
before routing is introduced.
"""
import torch

from _common import STATE_EN, STATE_HI, banner, device_line, heading, load

banner("20", "Which checkpoint?", """
    Laya ships three checkpoints with different encoders, windows and training. The names
    do not tell you when to use which, so this example runs the same three questions over
    two tickets: one in English, one in Devanagari Hindi.

      * `english`          421M, ModernBERT-large, 512 tokens -- trained on English.
      * `multilingual`     322M, mmBERT-base, 1024 tokens -- one model for 100+ languages.
      * `typed-decisions`  421M, ModernBERT-large, 1024 tokens -- the English checkpoint
                           fine-tuned on four specific workflows, not a general upgrade.

    Only one checkpoint is resident at a time: each is loaded, used for both tickets, then
    released, because all three together are over a billion parameters.
    """)

TICKETS = [("English ticket", STATE_EN), ("Hindi ticket", STATE_HI)]

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts",
                     "other": "everything else"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

NAMES = ("english", "multilingual", "typed-decisions")
results = {}
specs = {}

for name in NAMES:
    agent = load(name)
    specs[name] = (agent.cfg["encoder"], agent.cfg["max_len"],
                   sum(p.numel() for p in agent.model.parameters()))
    for label, state in TICKETS:
        results[(label, name)] = agent.predict(state, QUESTIONS)["answers"]
    if name == NAMES[0]:
        device_line(agent)
    del agent
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()          # release the checkpoint before building the next

heading("checkpoints")
for name in NAMES:
    encoder, window, params = specs[name]
    print("   %-16s %-28s %5d tokens   %.0fM params" % (name, encoder, window, params / 1e6))


def summary(answers, qid):
    """One compact field per question: the chosen label and its confidence."""
    answer = answers[qid]
    if answer["type"] == "choice":
        return "%-12s p=%.3f conf=%.3f" % (answer["choice"],
                                           answer["probabilities"][answer["choice"]],
                                           answer["confidence"])
    if answer["type"] == "score":
        return "%.2f/%d       conf=%.3f" % (answer["score"], len(answer["probabilities"]) - 1,
                                            answer["confidence"])
    return "%-5s P=%.3f conf=%.3f" % ("true" if answer["noul"] > 0.5 else "false",
                                      answer["noul"], answer["confidence"])


heading("the same questions, per checkpoint")
for label, _ in TICKETS:
    print("   %s" % label)
    for name in NAMES:
        answers = results[(label, name)]
        print("     %-16s department  %s" % (name, summary(answers, "department")))
        print("     %-16s urgency     %s" % ("", summary(answers, "urgency")))
        print("     %-16s refund_req  %s" % ("", summary(answers, "refund_requested")))

en_en = results[("English ticket", "english")]["department"]
ml_en = results[("English ticket", "multilingual")]["department"]
td_en = results[("English ticket", "typed-decisions")]["department"]
en_hi = results[("Hindi ticket", "english")]["department"]
ml_hi = results[("Hindi ticket", "multilingual")]["department"]
td_hi = results[("Hindi ticket", "typed-decisions")]["department"]
en_hi_refund = results[("Hindi ticket", "english")]["refund_requested"]

print("""
   On the English ticket all three agree on `%s`, which is reassuring but not equal: the
   specialised checkpoint is less sure (conf %.3f) than `multilingual` (%.3f) and `english`
   (%.3f). Fine-tuning on four workflows does not make a general-purpose model stronger.

   The Hindi ticket separates them. `english` still returns a label -- `%s` at p=%.3f,
   confidence %.3f -- which is a guess dressed as an answer, and its refund verdict is
   `%s` with confidence %.3f. That is the failure mode routing exists to prevent.
   `multilingual` reads it as `%s` (conf %.3f), and `typed-decisions` -- the same English
   encoder family -- also misses it (`%s`, conf %.3f).

   Reach for `english` on English text; reach for `multilingual` on anything else, Latin
   script or not. `typed-decisions` is a specialist for its own workflow schemas: it is
   never a silent default, and you opt into it explicitly. Examples 21 and 22 turn this
   table into a router.
   """ % (en_en["choice"], td_en["confidence"], ml_en["confidence"], en_en["confidence"],
          en_hi["choice"], en_hi["probabilities"][en_hi["choice"]], en_hi["confidence"],
          "true" if en_hi_refund["noul"] > 0.5 else "false", en_hi_refund["confidence"],
          ml_hi["choice"], ml_hi["confidence"], td_hi["choice"], td_hi["confidence"]))
