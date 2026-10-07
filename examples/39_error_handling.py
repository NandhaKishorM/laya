"""Example 39 -- the failure modes a caller should handle.

Every case below is executed and its real behaviour reported -- including the two that do *not*
raise when you might expect them to. Nothing here is inferred from the documentation.
"""
import os

from _common import LOCAL_MODELS, banner, device_line, heading, laya, load

from laya.router import Router

banner("39", "Error handling", """
    Five things a caller should be ready for:

      1. a typo in `model=` on a `Router`           -> ValueError, before anything loads;
      2. `laya.load` on a missing local path       -> FileNotFoundError;
      3. a `choice` question with empty criteria   -> ValueError naming it, before the model runs;
      4. a question whose options do not fit       -> ValueError, but only once they overflow;
      5. a `noul` whose probability is near 0.5    -> a perfectly ordinary answer, not an error.

    Cases 1 and 2 are cheap and run first. Everything after that uses the English checkpoint.
    Where the code does not raise, the output says so explicitly.
    """)

TICKET = {"body": "I was charged twice and also my login is broken. Can someone look into it?"}

heading("1. A typo in `model=` on a Router")
# A name `Router(models=...)` does not know is not a typo: it registers a checkpoint of your own
# under that name -- a fine-tune in a local directory, a repo you pushed -- served beside the
# built-ins and loaded only when a request routes to it. The typo is in the request.
router = Router(models={"bogus": "/nowhere/at/all"})
print("   Router(models={'bogus': ...}) registered it beside the built-ins: %s" % sorted(router.registered))
try:
    router.route(TICKET, {"q": {"type": "noul", "instructions": "Is this about billing?"}}, model="bogsu")
    print("   no exception raised")
except ValueError as e:
    print("   raised ValueError: %s" % e)
print("   -> `model=` is resolved against the built-ins plus what you registered, so a typo fails")
print("      before any weights load; nothing is downloaded until a request routes to a name.")

heading("2. laya.load on a path that does not exist")
# `LOCAL_MODELS` is the directory each checkpoint would live in; `MODELS` is the `Router` spec,
# which `_common.checkpoint` returns as `(repo, subfolder)` when `models/` is not in the checkout.
# A path has to come from the first one.
missing = os.path.join(LOCAL_MODELS["english"], "does-not-exist")
try:
    laya.load(missing)
    print("   no exception raised")
except FileNotFoundError as e:
    print("   raised FileNotFoundError: %s" % e)
print("   -> an absolute or ./-relative path is treated as local and never hits the network.")

agent = load("english")
device_line(agent)

heading("3. a choice question with an empty criteria dict")
for criteria in ({}, []):
    try:
        answer = agent.predict(TICKET, {"q": {"type": "choice", "instructions": "Pick one",
                                              "criteria": criteria}})
        print("   criteria=%r -> no exception, answer=%r" % (criteria, answer["answers"]["q"]))
    except Exception as e:
        print("   criteria=%r -> %s: %s" % (criteria, type(e).__name__, e))
print("   -> neither form reaches the decision head. `Agent._check_question` runs inside")
print("      `predict_batch`, before the state is encoded, and raises ValueError naming the")
print("      question and what to add -- so an empty schema is a caller bug with a message that")
print("      says which of twenty questions to fix, not a low-level crash to pattern-match.")
two = agent.predict(TICKET, {"q": {"type": "choice", "instructions": "Pick one",
                                   "criteria": {"a": None, "b": None}}})["answers"]["q"]
print("   (a two-option question with None descriptions is fine: a=%.3f b=%.3f)"
      % (two["probabilities"]["a"], two["probabilities"]["b"]))

heading("4. options that do not fit the sequence")
LARGE = {"opt_%03d" % i: "label number %d" % i for i in range(160)}
VERY_LARGE = {"opt_%03d" % i: "label number %d" % i for i in range(120)}
for name, criteria in (("120 options", VERY_LARGE), ("160 options", LARGE)):
    try:
        answer = agent.predict(TICKET, {"q": {"type": "choice", "instructions": "Pick a label",
                                              "criteria": criteria}})["answers"]["q"]
        print("   %s -> NO exception; squeezed to %r (confidence %.4f)"
              % (name, answer["choice"], answer["confidence"]))
    except ValueError as e:
        print("   %s -> raised ValueError: %s" % (name, e))
print("   -> the guard is `len(markers) != len(render_options(q))`: markers past max_len are")
print("      dropped, so it fires only when whole option blocks fall out of the window. 120")
print("      options slip through with a tiny per-label budget -- low accuracy, no warning.")

heading("5. a noul answer near 0.5")
CANDIDATES = {
    "single_problem": "Is the message about a single problem rather than two?",
    "first_contact": "Is this the customer's first contact with support?",
    "wants_phone_call": "Does the customer prefer a phone call over email?",
}
answers = agent.predict(TICKET, {k: {"type": "noul", "instructions": v}
                                 for k, v in CANDIDATES.items()})["answers"]
for qid, a in sorted(answers.items(), key=lambda kv: abs(kv[1]["noul"] - 0.5)):
    print("   %-16s noul=%.4f  confidence=%.4f" % (qid, a["noul"], a["confidence"]))

def policy(noul, low=0.4, high=0.6):
    """What a caller should do with an uncertain noul -- abstain instead of rounding."""
    if noul < low:
        return "no"
    if noul > high:
        return "yes"
    return "ABSTAIN (needs more context or a human)"

closest = min(answers.values(), key=lambda a: abs(a["noul"] - 0.5))
print("   closest to 0.5: noul=%.4f -> policy says %s" % (closest["noul"], policy(closest["noul"])))
print("   -> a probability near 0.5 is not an error; it is the model honestly saying it cannot")
print("      tell. Thresholding at 0.5 would silently turn that into a confident yes/no.")
