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
from laya.lang import analyse  # noqa: E402

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
#
# NOTE on future disambiguation: both Hindi and Marathi use Devanagari (U+0900-U+097F).
# Script detection routes by Unicode block, so it cannot distinguish between languages
# sharing the same script. Both currently route to `multilingual`. If a future language-specific
# checkpoint is added (e.g. a dedicated Hindi or Marathi model), these tests would need to
# assert the specific checkpoint name rather than the generic `multilingual` bucket.

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

# lang_guess codes for Hindi: assert model and that detection is None (showing the hint
# was decisive and bypassed built-in script detection).
# We test with English text where lang_guess="hi" forces multilingual routing;
# if lang_guess were ignored, plain English would route to english.
d_hi_dev = r0.route("मुझे मदद चाहिए।", GENERIC, lang_guess="hi")
check("hindi/lang_guess hi routes multilingual on devanagari", d_hi_dev["model"], "multilingual")
check_true("hindi/lang_guess hi bypasses detection (detection is None)", d_hi_dev["detection"] is None)
check_true("hindi/lang_guess hi recorded in reason", "lang_guess" in d_hi_dev["reason"])

d_hi_en = r0.route("I need help with my account billing.", GENERIC, lang_guess="hi")
check("hindi/lang_guess hi overrides English text to multilingual", d_hi_en["model"], "multilingual")
check_true("hindi/lang_guess hi on English sets detection to None", d_hi_en["detection"] is None)

d_hi_in = r0.route("I need help with my account billing.", GENERIC, lang_guess="hi-IN")
check("hindi/lang_guess hi-IN routes multilingual", d_hi_in["model"], "multilingual")
check_true("hindi/lang_guess hi-IN sets detection to None", d_hi_in["detection"] is None)

# script is reported correctly for native Devanagari text (no lang_guess needed)
check("hindi/script detected as devanagari",
      r0.route("यह एक हिंदी वाक्य है।", GENERIC)["detection"]["script"], "devanagari")

# one Devanagari field in a mixed dict is enough to reach multilingual
check("hindi/mixed Hindi-English dict reaches multilingual",
      r0.route({"subject": "Payment issue", "body": "मेरा भुगतान विफल हो गया।"}, GENERIC)["model"],
      "multilingual")

# script detection is not length-weighted: a long English subject must not outvote
# a short Devanagari body field.
check("hindi/long English field does not outvote short Hindi field",
      r0.route(
          {"subject": ("We have been experiencing persistent difficulties with our account billing "
                       "over the past several months and need urgent support from customer care."),
           "body": "कृपया मेरा भुगतान वापस करें।"},
          GENERIC)["model"],
      "multilingual")


# ------------------------------------------------------------------ Marathi (Devanagari script)
# Marathi shares Devanagari with Hindi (same Unicode block, U+0900-U+097F) but is a
# distinct language spoken by ~90 million people. Routing must reach multilingual for
# both -- any regression sending Devanagari to the English checkpoint collapses
# accuracy to near-random (measured at 0.100 on Hindi on MASSIVE at 20 options).
#
# NOTE on future disambiguation: both Hindi and Marathi use Devanagari (U+0900-U+097F).
# Script detection routes by Unicode block, so it cannot distinguish between languages
# sharing the same script. Both currently route to `multilingual`. If a future language-specific
# checkpoint is added (e.g. a dedicated Hindi or Marathi model), these tests would need to
# assert the specific checkpoint name rather than the generic `multilingual` bucket.

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
    # technical support (scenarios mirror Hindi exactly)
    ("marathi/app crash on settings",
     "अॅप्लिकेशन सेटिंग उघडताना प्रत्येक वेळी बंद होते, कृपया लवकर सोडवा.",
     "multilingual"),
    ("marathi/login failure",
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

# lang_guess codes for Marathi: assert model and detection is None
d_mr_dev = r0.route("मला मदत हवी आहे.", GENERIC, lang_guess="mr")
check("marathi/lang_guess mr routes multilingual on devanagari", d_mr_dev["model"], "multilingual")
check_true("marathi/lang_guess mr bypasses detection (detection is None)", d_mr_dev["detection"] is None)
check_true("marathi/lang_guess mr recorded in reason", "lang_guess" in d_mr_dev["reason"])

d_mr_en = r0.route("I need help with my account billing.", GENERIC, lang_guess="mr")
check("marathi/lang_guess mr overrides English text to multilingual", d_mr_en["model"], "multilingual")
check_true("marathi/lang_guess mr on English sets detection to None", d_mr_en["detection"] is None)

d_mr_in = r0.route("I need help with my account billing.", GENERIC, lang_guess="mr-IN")
check("marathi/lang_guess mr-IN routes multilingual", d_mr_in["model"], "multilingual")
check_true("marathi/lang_guess mr-IN sets detection to None", d_mr_in["detection"] is None)

# script is reported correctly for Marathi too
check("marathi/script detected as devanagari",
      r0.route("माझे पेमेंट अयशस्वी झाले.", GENERIC)["detection"]["script"], "devanagari")

# one Marathi field in a mixed dict reaches multilingual
check("marathi/mixed Marathi-English dict reaches multilingual",
      r0.route({"subject": "Billing problem", "body": "मला दोनदा शुल्क आकारले गेले."}, GENERIC)["model"],
      "multilingual")

# long English field must not outvote short Marathi field (mirrors the Hindi case above)
check("marathi/long English field does not outvote short Marathi field",
      r0.route(
          {"subject": ("We have been experiencing persistent difficulties with our account billing "
                       "over the past several months and need urgent support from customer care."),
           "body": "कृपया माझे पैसे परत करा."},
          GENERIC)["model"],
      "multilingual")


# ------------------------------------------------------------------ Hindi vs Marathi consistency
# Both languages use Devanagari. The router decides on script, not language identity,
# so routing must be identical for both.
check("hindi_marathi/same checkpoint regardless of language",
      r0.route("मुझे मदद चाहिए।", GENERIC)["model"],
      r0.route("मला मदत हवी आहे.", GENERIC)["model"])
# A foreign function word that is also an ordinary English word (`come` it, `son` es, `do` pt,
# `care` ro, `todo`/`im`/`per`/`plus`) used to name that language when it merely appeared
# twice, because the score counted occurrences and `best >= 2` was reached by one repeated word.
# Ordinary English then routed to the multilingual checkpoint. Such a word now counts once however
# often it repeats, so one collision no longer clears the bar.
ENGLISH_ON_A_REPEAT = [
    "Come one, come all",                      # come (it) x2
    "My son, your son",                        # son (es) x2
    "Do more, do less",                        # do (pt) x2
    "Care more, care less",                    # care (ro) x2
    "Add a todo, then another todo item",      # todo (es) x2
    "im not able to log in, im stuck",         # im (de) x2
    "add 45 to 87 plus 54 plus 43 plus 22",    # plus (de, fr) x3 -- a CLINC150 test row
]
for s in ENGLISH_ON_A_REPEAT:
    check("repeat/plain english stays english: %r" % s, analyse(s)["is_english"], True)

# The dedupe is limited to those words. A function word that is nobody's English -- `der`, `des`,
# `di`, `sa` -- still counts every occurrence, so a real request whose only evidence is one such
# word repeated stays foreign. Rows from the MASSIVE test splits, which is where deduping every
# word instead of the collisions cost 791 of 148,700 non-English texts their checkpoint.
REPEAT_STAYS_FOREIGN = [
    ("de", "reduzieren der helligkeit der lichter"),
    ("es", "enumerar todos los horarios de los tren a nueva york"),
    ("fr", "jouer des chansons des beatles"),
    ("it", "numero di telefono di giacomo"),
    ("pt", "pede um pacote de massa chinesa faz um pedido takeaway"),
    ("ro", "ar trebui sa port o pelerina de ploaie inainte sa ies afara"),
    ("nl", "herinner me eraan dat ik dat liedje leuk vind"),
    # `van` is left out of the collision list for exactly this row: it is ordinary Dutch
    ("nl", "hey olly ik hou van muziek van frans bauer"),
]
for want, s in REPEAT_STAYS_FOREIGN:
    check("repeat/%s on a repeated word is still foreign" % want, analyse(s)["is_english"], False)

# and real foreign requests -- two different function words each -- are still detected
STILL_FOREIGN = [
    ("es", "Hola, necesito cancelar mi pedido por favor ahora mismo gracias"),
    ("pt", "Eu quero cancelar o meu pedido por favor agora mesmo obrigado"),
    ("it", "Vorrei annullare il mio ordine per favore adesso grazie mille"),
]
for want, s in STILL_FOREIGN:
    check("repeat/genuine %s still foreign" % want, analyse(s)["is_english"], False)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
