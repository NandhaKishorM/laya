"""Example 22 -- criteria as structured data, not just strings.

Criteria can be dicts or lists; `render_criterion` serialises them to compact JSON so the model
sees real JSON in the prompt instead of a Python repr. This example renders all three question
types with structured criteria, then runs the prediction.
"""
import json
import textwrap

from _common import laya, banner, describe, device_line, load

banner("22", "Structured criteria", """
    Every option carries a description, and those descriptions are part of the input -- strip
    them to bare labels and the task changes. Descriptions do not have to be strings: `criteria`
    values may be dicts or lists, which are rendered as compact JSON.

    The contract (guarded by tests/test_criteria.py):
      * a string passes through unchanged;
      * anything structured becomes compact JSON, not a Python repr;
      * for `choice`, None or "" means "no description" and the option is the bare key, but
        0 and False are real values and are kept;
      * for `noul`, a missing `false`/`true` criterion falls back to the default phrasing.

    Before that fix, dict-valued `noul` criteria raised
    `TypeError: can only concatenate str (not "dict") to str`, and `choice`/`score` leaked
    `{'desc': ...}` into the prompt. Calling `laya.render_options` is the cheapest way to see
    exactly what the model reads.
    """)

# --- three structured questions over one phishing-looking email -----------------------------
STATE = {
    "from": "security@acme-support.co",
    "subject": "Verify your account within 24 hours",
    "body": "We detected unusual activity. Confirm your password at http://acme-support.co/login "
            "or your account will be suspended today.",
}

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "What kind of email is this?",
        "criteria": {
            "phishing": {"desc": "credential theft or fraud",
                         "signals": ["lookalike domain", "urgency", "login link"],
                         "team": "security"},
            "billing": {"desc": "invoices, payments or refunds", "signals": [], "team": "finance"},
            "technical": {"desc": "bugs, outages or integrations", "signals": [], "team": "support"},
            "newsletter": {"desc": "bulk marketing the user opted into", "signals": [],
                           "team": "none"},
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How dangerous is this email if a user acts on it?",
        "criteria": [
            {"level": 0, "desc": "benign", "sla_hours": 72},
            {"level": 1, "desc": "suspicious", "sla_hours": 24},
            {"level": 2, "desc": "harmful", "sla_hours": 4},
            {"level": 3, "desc": "critical: credentials or money at risk", "sla_hours": 1},
        ],
    },
    "is_phishing": {
        "type": "noul",
        "instructions": "Is this email a phishing attempt?",
        "criteria": {
            "true": {"desc": "phishing, scam or fraud",
                     "signals": ["lookalike domain", "credential request"]},
            "false": {"desc": "a legitimate email", "signals": []},
        },
    },
}

def internal(question):
    return {"t": question["type"], "ins": question["instructions"], "crit": question["criteria"]}

agent = load("english")
device_line(agent)
print("   Laya %s; the option text below is exactly what the encoder sees (rendered as JSON)."
      % laya.__version__)

# --- what the model actually reads ----------------------------------------------------------
def show_option(index, option):
    lines = textwrap.wrap(option, width=94) or [""]
    print("   option %d: %s" % (index, lines[0]))
    for cont in lines[1:]:
        print("             %s" % cont)

for qid, question in QUESTIONS.items():
    print("\n   == %s (%s) ==" % (qid, question["type"]))
    for i, option in enumerate(laya.render_options(internal(question))):
        show_option(i, option)

# --- the same questions through the real API ------------------------------------------------
print("\n   == prediction ==")
result = agent.predict(STATE, QUESTIONS)
describe(result["answers"])
answer = result["answers"]["is_phishing"]
print("   is_phishing noul=%.4f -> %s (confidence %.4f)"
      % (answer["noul"], "phishing" if answer["noul"] > 0.5 else "legitimate", answer["confidence"]))
print("   severity legend from the dict levels: %s"
      % json.dumps(result["answers"]["severity"]["legend"]))
print("   JSON round-trips: the `true` criterion parses back to %s"
      % json.dumps(json.loads(
          laya.render_options(internal(QUESTIONS["is_phishing"]))[1].split("true: ", 1)[1])))
print("   tokens used: %d in, %d out" % (result["usage"]["input_tokens"],
                                         result["usage"]["output_tokens"]))
