"""Regression tests for the caller-supplied language hint (`lang_guess`).

A caller who already runs a language-identification model can hand routing the answer instead
of being silently misrouted by the built-in stopword heuristic:

    analyse("Care este ora in Tokyo?")
    # {'script': 'latin', 'language': 'en', 'is_english': True}   -> routes to ENGLISH

The hint answers one question -- can the English checkpoint read this state -- so it is checked
after an explicit `lang` and before detection, and a hint that resolves to nothing falls
through to detection so a LID model can abstain.

Run: python tests/test_lang_guess.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.router import Router, _english_from_code  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


# The state the maintainer used on #35: a short Romanian request the heuristic cannot place.
ROMANIAN = "Care este ora in Tokyo?"
GENERIC = {"intent": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}

# ------------------------------------------------------------------ the baseline
r0 = Router()
check("baseline/Romanian is not identified by the heuristic",
      r0.route(ROMANIAN, GENERIC)["detection"]["language"], "en")
# pinned, not "either checkpoint": this is the defect the hint exists to fix, so the
# assertion has to distinguish english from the correct answer to mean anything.
check("baseline/Romanian therefore reaches english",
      r0.route(ROMANIAN, GENERIC)["model"], "english")


# ------------------------------------------------------------------ codes
check("code/Romanian routes multilingual", r0.route(ROMANIAN, GENERIC, lang_guess="ro")["model"],
      "multilingual")
check("code/English routes english", r0.route(ROMANIAN, GENERIC, lang_guess="en")["model"], "english")
check("code/POSIX underscore is read", r0.route(ROMANIAN, GENERIC, lang_guess="en_US")["model"], "english")
check("code/POSIX with encoding is read",
      r0.route(ROMANIAN, GENERIC, lang_guess="en_US.UTF-8")["model"], "english")
check("code/hyphen subtag is read", r0.route(ROMANIAN, GENERIC, lang_guess="de-DE")["model"], "multilingual")
check("code/case is ignored", r0.route(ROMANIAN, GENERIC, lang_guess="RO")["model"], "multilingual")
check("code/whitespace is ignored", r0.route(ROMANIAN, GENERIC, lang_guess="  en  ")["model"], "english")
check("code/unknown code still means non-English",
      r0.route(ROMANIAN, GENERIC, lang_guess="qq")["model"], "multilingual")

# an abstaining hint must fall through to detection, not force a checkpoint
for empty in (None, "", "   "):
    d = r0.route(ROMANIAN, GENERIC, lang_guess=empty)
    check("abstain/%r falls through to detection" % (empty,), d["detection"] is not None, True)

# ------------------------------------------------------------------ callables
check("callable/code is used",
      r0.route(ROMANIAN, GENERIC, lang_guess=lambda s: "ro")["model"], "multilingual")
check("callable/receives the state",
      r0.route(ROMANIAN, GENERIC, lang_guess=lambda s: "en" if "Tokyo" in str(s) else "ro")["model"],
      "english")
check("callable/None falls through to detection",
      r0.route(ROMANIAN, GENERIC, lang_guess=lambda s: None)["detection"] is not None, True)
check("callable/empty string falls through",
      r0.route(ROMANIAN, GENERIC, lang_guess=lambda s: "")["detection"] is not None, True)
check("callable/Romanian model that returns None does not change the default route",
      r0.route(ROMANIAN, GENERIC, lang_guess=lambda s: None)["model"],
      r0.route(ROMANIAN, GENERIC)["model"])

# ------------------------------------------------------------------ installed on the Router
r_inst = Router(lang_guess="ro")
check("installed/applies without a per-call hint", r_inst.route(ROMANIAN, GENERIC)["model"], "multilingual")
check("installed/per-call overrides the installed one",
      r_inst.route(ROMANIAN, GENERIC, lang_guess="en")["model"], "english")
r_fn = Router(lang_guess=lambda s: "ro")
check("installed/callable works too", r_fn.route(ROMANIAN, GENERIC)["model"], "multilingual")
check("installed/absent by default", r0.lang_guess, None)
check("installed/a hint that abstains leaves detection intact",
      Router(lang_guess=lambda s: None).route(ROMANIAN, GENERIC)["detection"] is not None, True)

# ------------------------------------------------------------------ precedence
check("precedence/explicit model beats the hint",
      r0.route(ROMANIAN, GENERIC, model="english", lang_guess="ro")["model"], "english")
check("precedence/explicit task beats the hint",
      r0.route(ROMANIAN, GENERIC, task="typed_decisions", lang_guess="ro")["model"], "typed-decisions")
check("precedence/explicit lang beats the hint",
      r0.route(ROMANIAN, GENERIC, lang="en", lang_guess="ro")["model"], "english")
check_true("precedence/an explicit lang is still reported as explicit",
           "explicit lang" in r0.route(ROMANIAN, GENERIC, lang="en", lang_guess="ro")["reason"])

# ------------------------------------------------------------------ the decision payload
d = r0.route(ROMANIAN, GENERIC, lang_guess="ro")
check("payload/model", d["model"], "multilingual")
check("payload/repo is a string", isinstance(d["repo"], str), True)
check("payload/reason records the caller hint", "lang_guess" in d["reason"], True)
check_true("payload/detection is None when the hint decided it", d["detection"] is None)
check("payload/installed hint names its source",
      "Router(lang_guess=...)" in Router(lang_guess="ro").route(ROMANIAN, GENERIC)["reason"], True)
check_true("payload/repo points at the bundle", "convaiinnovations/laya" in d["repo"])

# ------------------------------------------------------------------ predict forwards it
class _FakeAgent:
    """Stands in for a loaded checkpoint so this suite stays offline and fast."""

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


def _router_with_stub(**kwargs):
    """A Router whose `load` is stubbed, so `predict` exercises routing without weights."""
    r = Router(**kwargs)
    stub = _FakeAgent()

    def _load(name):
        r._agents[name] = stub
        return stub

    r.load = _load
    return r, stub


r_p, stub = _router_with_stub(lang_guess="ro")
out = r_p.predict(ROMANIAN, GENERIC)
check("predict/uses the hint", out["routing"]["model"], "multilingual")
check("predict/forwards a per-call hint",
      r_p.predict(ROMANIAN, GENERIC, lang_guess="en")["routing"]["model"], "english")
check("predict/reaches the model", len(stub.calls), 2)
check_true("predict/passes the state through unchanged", stub.calls[0][0] == ROMANIAN)
check_true("predict/keeps the routing block",
           set(out["routing"]) >= {"model", "repo", "reason", "detection", "workflow"})

# ------------------------------------------------------------------ the helper itself
check("helper/None", _english_from_code(None), None)
check("helper/empty", _english_from_code(""), None)
check("helper/spaces", _english_from_code("   "), None)
check("helper/en", _english_from_code("en"), True)
check("helper/en_US", _english_from_code("en_US"), True)
check("helper/zh_CN", _english_from_code("zh_CN"), False)
check("helper/strips after the dot", _english_from_code("en.UTF-8"), True)
check("helper/only a dot", _english_from_code("."), None)

# the standalone-repo mapping is untouched
r_alone = Router(standalone_repos=True, lang_guess="ro")
check("standalone/hint still uses the standalone repo",
      r_alone.route(ROMANIAN, GENERIC)["repo"], "convaiinnovations/laya-multilingual")

# nothing without a hint moves
# The expected model is pinned per case. Comparing `r0.route(...)` against a fresh
# `Router().route(...)` cannot fail, because both sides are the same pure call on an
# equally configured router -- a regression would move both together.
BEFORE = [("plain english", "I was charged twice and want a refund", "english"),
          ("German with umlauts", "Mein Konto wurde zweimal belastet, bitte erstatten Sie", "multilingual"),
          ("Hindi", "यह एक हिंदी वाक्य है", "multilingual"),
          ("empty", "", "english"),
          ("digits", "12345", "english")]
for label, s, want in BEFORE:
    check("unchanged/" + label, r0.route(s, GENERIC)["model"], want)


# ------------------------------------------------------------------ Hindi (Devanagari script)
# Hindi uses Devanagari (U+0900-U+097F), which is in _SCRIPT_RANGES. The English
# checkpoint (ModernBERT-large, 50k English BPE) has no Devanagari tokens, so any
# state written in Hindi must reach the multilingual checkpoint. These cases cover
# the four main customer-support scenarios: billing, technical, account and cancellation.

HINDI_CASES = [
    # billing & payment
    ("hindi/billing duplicate charge",
     "मुझसे मार्च महीने में दो बार शुल्क लिया गया है, कृपया डुप्लिकेट राशि वापस करें।",
     "multilingual"),
    ("hindi/payment failed refund",
     "मेरा भुगतान विफल हो गया लेकिन पैसे कट गए, मुझे तुरंत वापसी चाहिए।",
     "multilingual"),
    ("hindi/invoice not received",
     "मुझे अप्रैल माह का इनवॉइस अभी तक नहीं मिला है, कृपया भेजें।",
     "multilingual"),
    # technical support
    ("hindi/app crash on settings",
     "एप्लिकेशन हर बार सेटिंग खोलने पर बंद हो जाती है, कृपया जल्दी ठीक करें।",
     "multilingual"),
    ("hindi/login failure",
     "मैं अपने खाते में लॉग इन नहीं कर पा रहा हूँ, पासवर्ड सही है फिर भी एरर आ रहा है।",
     "multilingual"),
    ("hindi/OTP not received",
     "OTP मेरे मोबाइल पर नहीं आ रहा, मैं सत्यापन पूरा नहीं कर पा रहा।",
     "multilingual"),
    # account & cancellation
    ("hindi/cancel threat",
     "अगर यह समस्या जल्द हल नहीं हुई तो मैं अपनी सदस्यता रद्द कर दूँगा।",
     "multilingual"),
    ("hindi/plan upgrade enquiry",
     "मैं अपना प्लान अपग्रेड करना चाहता हूँ, प्रीमियम के क्या फायदे हैं?",
     "multilingual"),
    # short utterances (still Devanagari -- script detection is exact, not length-dependent)
    ("hindi/short help request",
     "नमस्ते, मुझे मदद चाहिए।",
     "multilingual"),
    ("hindi/single word",
     "धन्यवाद",
     "multilingual"),
]

for label, text, want in HINDI_CASES:
    check(label, r0.route(text, GENERIC)["model"], want)

# lang_guess codes for Hindi
check("hindi/lang_guess hi routes multilingual",
      r0.route("मुझे मदद चाहिए।", GENERIC, lang_guess="hi")["model"], "multilingual")
check("hindi/lang_guess hi-IN routes multilingual",
      r0.route("मुझे मदद चाहिए।", GENERIC, lang_guess="hi-IN")["model"], "multilingual")

# script is reported correctly
check("hindi/script detected as devanagari",
      r0.route("यह एक हिंदी वाक्य है।", GENERIC)["detection"]["script"], "devanagari")

# one Devanagari field in a mixed dict is enough to reach multilingual
check("hindi/mixed Hindi-English dict reaches multilingual",
      r0.route({"subject": "Payment issue", "body": "मेरा भुगतान विफल हो गया।"}, GENERIC)["model"],
      "multilingual")


# ------------------------------------------------------------------ Marathi (Devanagari script)
# Marathi shares Devanagari with Hindi (same Unicode block, U+0900-U+097F) but is a
# distinct language spoken by ~90 million people. Routing must reach multilingual for
# both -- any regression sending Devanagari to the English checkpoint collapses
# accuracy to near-random (measured at 0.100 on Hindi on MASSIVE at 20 options).

MARATHI_CASES = [
    # billing & payment
    ("marathi/billing duplicate charge",
     "मला मार्च महिन्यात दोनदा शुल्क आकारले गेले आहे, कृपया अतिरिक्त रक्कम परत करा.",
     "multilingual"),
    ("marathi/payment failed refund",
     "माझे पेमेंट अयशस्वी झाले पण पैसे कापले गेले, कृपया परतावा द्या.",
     "multilingual"),
    ("marathi/invoice not received",
     "मला एप्रिल महिन्याचे बिल अजून मिळाले नाही, कृपया पाठवा.",
     "multilingual"),
    # technical support
    ("marathi/app crash on open",
     "अॅप्लिकेशन उघडत नाही, प्रत्येक वेळी बंद होते, कृपया लवकर सोडवा.",
     "multilingual"),
    ("marathi/login error",
     "मी माझ्या खात्यात लॉग इन करू शकत नाही, पासवर्ड बरोबर असूनही चूक येते.",
     "multilingual"),
    ("marathi/OTP not received",
     "OTP माझ्या मोबाईलवर येत नाही, मी पडताळणी पूर्ण करू शकत नाही.",
     "multilingual"),
    # account & cancellation
    ("marathi/cancel threat",
     "जर ही समस्या लवकर सुटली नाही तर मी माझी सदस्यता रद्द करेन.",
     "multilingual"),
    ("marathi/plan upgrade enquiry",
     "मला माझा प्लान अपग्रेड करायचा आहे, प्रीमियमचे काय फायदे आहेत?",
     "multilingual"),
    # short utterances
    ("marathi/short help request",
     "नमस्कार, मला मदत हवी आहे.",
     "multilingual"),
    ("marathi/single word",
     "धन्यवाद",
     "multilingual"),
]

for label, text, want in MARATHI_CASES:
    check(label, r0.route(text, GENERIC)["model"], want)

# lang_guess codes for Marathi
check("marathi/lang_guess mr routes multilingual",
      r0.route("मला मदत हवी आहे.", GENERIC, lang_guess="mr")["model"], "multilingual")
check("marathi/lang_guess mr-IN routes multilingual",
      r0.route("मला मदत हवी आहे.", GENERIC, lang_guess="mr-IN")["model"], "multilingual")

# script is reported correctly for Marathi too
check("marathi/script detected as devanagari",
      r0.route("माझे पेमेंट अयशस्वी झाले.", GENERIC)["detection"]["script"], "devanagari")

# one Marathi field in a mixed dict reaches multilingual
check("marathi/mixed Marathi-English dict reaches multilingual",
      r0.route({"subject": "Billing problem", "body": "मला दोनदा शुल्क आकारले गेले."}, GENERIC)["model"],
      "multilingual")


# ------------------------------------------------------------------ Hindi vs Marathi consistency
# Both languages use Devanagari. The router decides on script, not language identity,
# so routing must be identical for both.
check("hindi_marathi/same checkpoint regardless of language",
      r0.route("मुझे मदद चाहिए।", GENERIC)["model"],
      r0.route("मला मदत हवी आहे.", GENERIC)["model"])


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
