"""Example 13 -- email states and stripping quoted cruft.

Real mail arrives with quoted replies, signatures and legal footers attached.
`email_state()` builds a clean state dict for you, and `email_state(..., clean=False)`
or `clean_email_body()` lets you see exactly what was thrown away.
"""

from _common import laya, banner, describe, device_line, heading, load

banner("13", "Email cleaning, and why it matters", """
    `laya.email_state(subject, body, sender)` returns a state dict and, by default,
    runs `clean_email_body()` over the body first. That helper drops quoted history
    (everything from an "On ... wrote:" header), lines starting with ">", trailing
    signature blocks, and confidentiality disclaimers.

    Cleaning is worth watching rather than trusting. Below, the same questions run
    against the cleaned state and the raw body of the same message. The first message
    is legitimate and both agree; the second is a phishing lure wrapped in a harmless
    quoted reply, and the raw body pulls the answer to the wrong team.
    """)

agent = load("english")
device_line(agent)
EMAIL_QUESTIONS = laya.email_questions()

LEGIT_BODY = """Hi billing team,

I need help with invoice 4411. Our finance system shows two separate captures for the
same order, and the card statement confirms both of them landed on the same day.

The order was placed for a single keyboard and a couple of cables, so the second charge
looks like a duplicate rather than a legitimate renewal.

Could you please reverse the duplicate charge and confirm by email once the refund has
been issued? I have attached the statement excerpt below for reference.

Best regards,
Dana Reyes
Finance, Acme GmbH
dana.reyes@acme.example
+49 30 1234 5678

CONFIDENTIALITY NOTICE: This message and any attachments are confidential and intended
solely for the use of the named addressee. If you have received this email in error,
please notify the sender and delete it immediately.

On Mon, 11 Mar 2024 at 09:12, Billing Support <billing@vendor.example> wrote:
> Hello Dana,
> We can see two charges on the account and are looking into it.
> We will get back to you shortly.
> Regards,
> Billing Support
"""

heading("the same email, raw and cleaned")
cleaned_body = laya.clean_email_body(LEGIT_BODY)
print("   raw body     : %d lines, %d chars" % (len(LEGIT_BODY.splitlines()), len(LEGIT_BODY)))
print("   cleaned body : %d lines, %d chars" % (len(cleaned_body.splitlines()),
                                                 len(cleaned_body)))
print("   quote header removed: %s   disclaimer removed: %s"
      % ("On Mon, 11 Mar 2024" not in cleaned_body,
         "CONFIDENTIALITY NOTICE" not in cleaned_body))

for name, body in (("RAW BODY", LEGIT_BODY), ("CLEANED BODY", cleaned_body)):
    print("\n   %s ---" % name)
    for line in body.splitlines():
        print("   | %s" % line)

heading("triage of the legitimate email: cleaned vs raw")
cleaned_state = laya.email_state("Invoice 4411 duplicate charge", LEGIT_BODY, "dana@acme.example")
raw_state = laya.email_state("Invoice 4411 duplicate charge", LEGIT_BODY, "dana@acme.example",
                             clean=False)
clean_result = agent.predict(cleaned_state, EMAIL_QUESTIONS)
raw_result = agent.predict(raw_state, EMAIL_QUESTIONS)
print("   cleaned: %d input tokens for %d questions"
      % (clean_result["usage"]["input_tokens"], len(EMAIL_QUESTIONS)))
describe(clean_result["answers"], indent="   ")
print("   raw    : %d input tokens for %d questions"
      % (raw_result["usage"]["input_tokens"], len(EMAIL_QUESTIONS)))
describe(raw_result["answers"], indent="   ")
print("   -> same verdict, %d fewer input tokens: cleaning buys headroom, not accuracy."
      % (raw_result["usage"]["input_tokens"] - clean_result["usage"]["input_tokens"]))

PHISH_BODY = """URGENT: Your account has been locked for security reasons.

Verify immediately at http://wellsfargo--verify.tj49.wsipv6.com or it will be closed
permanently.

Best regards,
Account Security Team

CONFIDENTIALITY NOTICE: This message is confidential and intended solely for the use of
the named addressee. If you have received this email in error, please delete it.

On Mon, 11 Mar 2024 at 09:12, Billing Support <billing@vendor.example> wrote:
> Hello, your invoice is attached.
> Regards,
> Billing Support
"""

heading("a phishing lure with an innocent quoted tail")
phish_clean = laya.email_state("Urgent: your account is locked", PHISH_BODY,
                               "security@wells-fargo-verify.com")
phish_raw = laya.email_state("Urgent: your account is locked", PHISH_BODY,
                             "security@wells-fargo-verify.com", clean=False)
print("   cleaned state:")
describe(agent.predict(phish_clean, EMAIL_QUESTIONS)["answers"], indent="   ")
print("   raw state:")
describe(agent.predict(phish_raw, EMAIL_QUESTIONS)["answers"], indent="   ")
print("""
   The quoted invoice reply and the confidentiality notice are the loudest text in the
   raw body, so it is routed to billing even though it is phishing. Once they are
   stripped, the lure itself dominates and `category` moves to security with a much
   higher phishing probability -- the same message, a materially different decision.
   """)
