"""Example 25 -- the email preset, from raw mail to verdict.

`laya.email_state()` turns a raw message into a state, and `laya.email_questions()` asks
the five typed questions that make up an email triage decision.
"""

from _common import laya, banner, describe, device_line, heading, load

banner("25", "Preset: email phishing and triage", """
    The email preset is two helpers that fit together:

      email_state(subject, body, sender)  ->  a state dict for the checkpoint
      email_questions()                   ->  five typed questions, one forward pass

    `email_state` strips quoted history, signatures and disclaimers before it builds the
    state (example 13 shows exactly what it removes). `email_questions` then asks which
    team should handle the mail, whether it is spam, whether it is phishing, how urgent
    it is, and whether the sender expects a reply.

    Three real messages go through the same call: a credential-phishing lure, a legitimate
    billing request and a subscribed newsletter. The preset is a triage signal, not an
    oracle, so the numbers are printed rather than hidden behind a boolean.
    """)

# --- raw mail in: label, subject, body, sender --------------------------------------------
MESSAGES = [
    ("phishing lure",
     "Urgent: your account has been locked",
     "Dear customer,\n\nWe detected unusual activity on your account and locked it for "
     "security reasons.\n\nVerify immediately at http://wellsfargo--verify.tj49.wsipv6.com "
     "or the account will be closed permanently.\n\nBest regards,\nAccount Security Team",
     "security@wells-fargo-verify.com"),
    ("billing request",
     "Invoice 4411 duplicate charge",
     "Hi billing team,\n\nOur finance system shows two separate captures for the same order, "
     "both on the same day. Could you please reverse the duplicate charge on invoice 4411 and "
     "confirm by email once it is done?\n\nBest regards,\nDana Reyes",
     "dana.reyes@acme.example"),
    ("subscribed newsletter",
     "Acme Weekly: what shipped in March",
     "Here is what is new this month: a faster export, a new dashboard widget and two "
     "integration fixes.\n\nYou are receiving this because you subscribed to product updates. "
     "Unsubscribe at any time.",
     "newsletter@acme.example"),
]

agent = load("english")
device_line(agent)
EMAIL = laya.email_questions()
print("   question ids: %s" % ", ".join(EMAIL))


def verdict(answers):
    """Our delivery policy. The thresholds and routing here are application code."""
    if answers["is_phishing"]["noul"] >= 0.5:
        return "block as phishing and warn the recipient"
    if answers["is_spam"]["noul"] >= 0.5:
        return "quarantine as spam"
    return "deliver to the %s queue" % answers["category"]["choice"]


records = []
for label, subject, body, sender in MESSAGES:
    state = laya.email_state(subject, body, sender)
    result = agent.predict(state, EMAIL)
    answers = result["answers"]
    records.append((label, answers, result))
    heading(label)
    print("   subject: %s" % subject)
    print("   from   : %s" % sender)
    print("   state  : %d body chars after cleaning, %d input tokens for %d questions"
          % (len(state["body"]), result["usage"]["input_tokens"], len(EMAIL)))
    describe(answers)
    category = max(answers["category"]["probabilities"].items(), key=lambda kv: kv[1])
    levels = len(answers["urgency"]["probabilities"]) - 1
    print("   phishing=%.3f  spam=%.3f  category=%s (p=%.3f)  urgency=%.2f / %d"
          % (answers["is_phishing"]["noul"], answers["is_spam"]["noul"],
             answers["category"]["choice"], category[1], answers["urgency"]["score"], levels))
    print("   VERDICT: %s" % verdict(answers))

heading("side by side")
print("   %-21s %-9s %-8s %-10s %-9s %s"
      % ("message", "phishing", "spam", "category", "urgency", "verdict"))
for label, answers, _ in records:
    print("   %-21s %-9.3f %-8.3f %-10s %-9.2f %s"
          % (label, answers["is_phishing"]["noul"], answers["is_spam"]["noul"],
             answers["category"]["choice"], answers["urgency"]["score"], verdict(answers)))

phish = records[0][1]
billing = records[1][1]
news = records[2][1]
news_ranked = sorted(news["category"]["probabilities"].items(), key=lambda kv: -kv[1])
news_top, news_next = news_ranked[0], news_ranked[1]
print("""
   What separates the lure from the legitimate mail is `is_phishing` and `category`, both at
   high confidence; what does not is `is_spam`, where the subscribed newsletter -- bulk
   marketing, but not a threat -- lands either side of the 0.5 gate. The newsletter is the
   honest hard case: its own `category` is a near-tie, so the team label is not dependable and
   the delivery policy is left leaning on `is_spam` alone. When two fields are both sitting on
   their boundary, that is a message for a human rather than a threshold. `needs_reply` does
   not separate the lure from the billing mail either.
 
   `urgency` is the softest field on all three messages: an expected level over three buckets,
   with normalised-entropy confidence, so it is a ranking signal rather than a label. Read the
   score; do not threshold the confidence. The numbers are above, per message and side by side.
   """)
