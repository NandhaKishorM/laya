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
from laya.lang import analyse, _named_prose_language  # noqa: E402

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
          # letterless, so these follow `default`, which is multilingual since 0.4.0
          ("empty", "", "multilingual"),
          ("digits", "12345", "multilingual")]
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


# An all-caps line is exempt from the acronym blanking -- a customer shouting in Portuguese is
# still Portuguese -- but a line of bare acronyms has no lowercase either, so it took the same
# exemption and was read as prose. `MON DES EST LA` is a hockey team, a state, a time zone and an
# airport, and it was named French.
#
# The bar is on the tokens that actually scored, not on the line: one long word anywhere is no
# evidence of anything, because `ANGELES` and `VEGAS` are long and `LOS`/`LAS`/`EL` are what name
# Spanish. An all-caps line must hold a *matched stopword* of `_SHOUTED_MIN_STOPWORD` letters, or
# non-English diacritics, and one word of `_SHOUTED_MIN_WORD` letters.
ACRONYM_RUNS_NAME_NOTHING = [
    "MON DES EST LA",                               # hockey/state/time-zone/airport -> was 'fr'
    "QUE MON LES DES",                              # four French stopwords, none a word here
    "COM DOS LAN WAN VPN",                          # networking
    "UNO DOS TRES LAS",
    "ESA UN NASA ISS",
    # a long token in the run is not evidence: it is not what named the language
    "COM DOS LAN WAN VPN ROUTER",                   # 'pt' on com/dos, ROUTER is incidental
    "SERVER LOG COM DOS LAN WAN VPN TIMEOUT",       # 'pt' on com/dos
    # all-caps US address and signage blocks, the commonest real instance of this shape
    "STORES LOS ANGELES LAS VEGAS EL PASO CLOSED",  # 'es' on los/las/el
    "SHIP TO EL SEGUNDO LA HABRA LOS BANOS CA",
    "WAREHOUSE LA PORTE SAN DIEGO EL PASO TX",
]
for line in ACRONYM_RUNS_NAME_NOTHING:
    check("shouted/acronym run names nothing: %s" % line, _named_prose_language(line), None)

# the mixed-case path is untouched: acronyms inside ordinary prose were already blanked. These
# two pass with the guard reverted as well -- they pin the pre-existing blanking, not this change.
MIXED_CASE_STILL_NONE = [
    "The MON DES EST LA codes were sent today",
    "Please confirm the COM DOS LAN WAN VPN settings before Friday",
]
for line in MIXED_CASE_STILL_NONE:
    check("shouted/mixed case still names nothing: %s" % line[:28], _named_prose_language(line), None)

# and the case the exemption exists for keeps working: genuine prose, shouted, is still named,
# whether by a long matched stopword or by its diacritics.
SHOUTED_PROSE_STILL_NAMED = [
    ("pt", "ESTA MENSAGEM E CONFIDENCIAL E NAO DEVE SER COMPARTILHADA"),   # esta, deve
    ("pt", "QUERO MEU DINHEIRO DE VOLTA AGORA"),                           # quero, agora
    ("pt", "COMO ESTA O TEMPO HOJE"),                                      # como, esta, hoje
    ("es", "NECESITO CANCELAR MI PEDIDO POR FAVOR AHORA MISMO"),           # necesito
    ("de", "DIE VERBINDUNG ZUM SERVER WURDE UNTERBROCHEN BITTE VERSUCHEN SIE ES SPAETER"),
    ("de", "WIE SPÄT IST ES IN KÖLN HEUTE ABEND"),                         # named on diacritics
]
for want, line in SHOUTED_PROSE_STILL_NAMED:
    check("shouted/%s prose still named: %s" % (want, line[:24]), _named_prose_language(line), want)

# Routing, which is the thing the caller sees. `_named_prose_language` is reached only from the
# mixed-segment scan, and that scan runs only once a state already reads English overall, so a
# unit assertion on it proves nothing about a *short* state: `MON DES EST LA` on its own, and
# under one or two lines of English, tipped the whole-state verdict before the scan could run and
# routed multilingual. Only at three English lines did the scan take over. Assert every dilution.
_COMPLAINT = ("I ordered a blender on the 3rd of March and it arrived broken.\n"
              "I asked for a refund the same week and nobody has replied to me since.\n"
              "%s\n"
              "Please tell me when the money will be back on my card.")
_route = Router(preload=False, default="english")   # see the note below
# `default="english"` deliberately: a bare acronym line is language-undecided, so under the
# stock 0.4.0 default it routes multilingual for want of evidence rather than because it was
# named French. Pinning the default keeps these rows testing #1013 (the acronym must not be
# named as foreign prose) instead of re-testing the default, which test_router.py covers.
check("shouted/bare acronym state routes english",
      _route.route("MON DES EST LA")["model"], "english")
for _n in (1, 2, 3):
    _state = "Refund please.\n" * _n + "MON DES EST LA"
    check("shouted/acronym line under %d english line(s) routes english" % _n,
          _route.route(_state)["model"], "english")
check("shouted/english doc with an acronym line stays english",
      _route.route(_COMPLAINT % "MON DES EST LA")["model"], "english")
check("shouted/english doc with an acronym line reports no segment",
      analyse(_COMPLAINT % "MON DES EST LA")["mixed_segment"], None)
check("shouted/english doc with an address block stays english",
      _route.route(_COMPLAINT % "STORES LOS ANGELES LAS VEGAS EL PASO CLOSED")["model"], "english")
check("shouted/address block as a field stays english",
      _route.route({"subject": "Please check this ticket for the customer today",
                    "note": "STORES LOS ANGELES LAS VEGAS EL PASO CLOSED"})["model"], "english")
# the other direction must not move
check("shouted/english doc with shouted portuguese still routes multilingual",
      _route.route(_COMPLAINT % "QUERO MEU DINHEIRO DE VOLTA AGORA")["model"], "multilingual")
check("shouted/bare shouted portuguese still routes multilingual",
      _route.route("QUERO MEU DINHEIRO DE VOLTA AGORA")["model"], "multilingual")

# A vetoed line is not a verdict of English. `_leaf_non_english` dropped it outright, which
# skipped the diacritic branch that exists for exactly this: text carrying non-English letters
# that no stopword list can name. These three are real MASSIVE rows. Each is named by
# `_analyse_text` but vetoed by `_named_prose_language` for holding only one stopword of that
# language where two are required, and each carries diacritics above NON_EN_DIACRITIC_RATE --
# so each was thrown away, and the field routed english. 114 MASSIVE rows move on this.
_VETOED_BUT_ACCENTED = [
    ("af", "vertel my van my vergaderings van môre oggend"),
    ("af", "sê hardop die skedules van die lys"),
    ("af", "wat is die koördinate van die ewenaar"),
]
for _loc, _line in _VETOED_BUT_ACCENTED:
    check("veto/%s accented field falls through to the diacritic branch: %s" % (_loc, _line[:26]),
          _route.route({"subject": "Please take a look at this ticket for the customer today",
                        "note": _line})["model"], "multilingual")
    check("veto/%s accented line is not named outright: %s" % (_loc, _line[:26]),
          _named_prose_language(_line), None)
# a shouted line with umlauts needs no fall-through -- its diacritics clear the bar directly
check("veto/shouted german with umlauts routes multilingual",
      _route.route("WIE SPÄT IST ES IN KÖLN")["model"], "multilingual")


# Emphasis capitals, which the mixed-case half of the guard got wrong. The whole-state verdict
# re-takes itself with the all-caps runs blanked, so an acronym run cannot outvote the English
# prose around it -- but blanking *every* all-caps run deleted ordinary emphasis too, and
# emphasis falls on the words that carry the sentence, so the words that named the language were
# exactly the ones removed. No test covered this branch for genuine foreign prose, and all of
# these routed english. A run of `_SHOUTED_MIN_WORD` letters is a word, not an acronym, and is
# not blanked out of its own sentence. Eight of the ten are real MASSIVE test rows, shouted
# where a person would shout them; the other two are hand-written, one of them the Spanish case
# the review reported.
EMPHASIS_STILL_FOREIGN = [
    ("de", "sag mir das HEUTIGE DATUM"),
    ("de", "welche wecker habe ICH GESTELLT"),
    ("es", "quiero cancelar mi PEDIDO POR FAVOR"),
    ("es", "QUIERO EL ESTADO DEL brillo de mi pantalla"),
    ("pt", "quero o meu DINHEIRO DE VOLTA"),
    ("pt", "DIZ ME O TEMPO EM barcelona daqui a dois dias"),
    ("nl", "VERTEL ME HET weer deze week"),
    ("fr", "passer l'aspirateur dans LE COULOIR"),
    ("it", "CANCELLA LA MIA sveglia delle sette"),
    ("ro", "SPUNE-MI VREMEA pentru saptamana aceasta"),
]
for _want, _line in EMPHASIS_STILL_FOREIGN:
    check("emphasis/%s prose routes multilingual: %s" % (_want, _line[:30]),
          _route.route(_line)["model"], "multilingual")
    check("emphasis/%s prose keeps its language: %s" % (_want, _line[:30]),
          analyse(_line)["language"], _want)

# Monotonicity: adding one lowercase English token must not make foreign detection worse. The
# fully upper-cased form is held to the shouted bar and clears it; the same text with `ok ` in
# front used to fall into the mixed-case branch instead, which had no bar at all, and routed
# english. 4,285 MASSIVE rows moved that way; 316 still do, against 279 at the parent.
_SHOUT = "QUIERO CANCELAR MI PEDIDO POR FAVOR"
check("emphasis/upper-cased spanish routes multilingual",
      _route.route(_SHOUT)["model"], "multilingual")
check("emphasis/one lowercase token does not undo it",
      _route.route("ok " + _SHOUT)["model"], "multilingual")

# and the acronym side the blanking exists for is still closed, which is what keeps the bar on
# the length of the run rather than on nothing at all.
check("emphasis/acronym run inside english prose still routes english",
      _route.route("The MON DES EST LA codes were sent today")["model"], "english")
check("emphasis/acronym settings line inside english prose still routes english",
      _route.route("Please confirm the COM DOS LAN WAN VPN settings before Friday")["model"],
      "english")
check("emphasis/acronym run after one lowercase token still routes english",
      _route.route("ok MON DES EST LA")["model"], "english")

# Mixed cased and caseless script. Both caps bars answer by taking evidence *away* from the
# Latin letters, and `looks_non_english`, which decides what is left, reads Latin diacritics
# only -- so on a state that is 35% Cyrillic letters there was nothing left and an acronym run
# with a Russian greeting in front of it went to the English checkpoint. The script promotion
# cannot catch that fall: it needs `_non_latin_words`, which drops any run starting with a
# capital, and in a shouted line every run does. Neither bar applies once the caseless letters
# carry the state -- a wrong language name costs nothing there, because the checkpoint is
# chosen on `is_english`.
MIXED_SCRIPT_NOT_ENGLISH = [
    "\u041f\u0420\u0418\u0412\u0415\u0422 MON DES EST LA",   # all caps: the shouted bar
    "\u041f\u0440\u0438\u0432\u0435\u0442 MON DES EST LA",   # mixed case: the blanking
]
for _line in MIXED_SCRIPT_NOT_ENGLISH:
    check("mixed-script/routes multilingual: %s" % _line[:12],
          _route.route(_line)["model"], "multilingual")
    check("mixed-script/is not english: %s" % _line[:12],
          analyse(_line)["is_english"], False)
# with no caseless letters to carry it, the same run is vetoed as before
check("mixed-script/latin-only acronym run is still english",
      _route.route("MON DES EST LA")["model"], "english")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
