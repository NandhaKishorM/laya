"""Routing and language-detection tests. No model weights are loaded: `Router.route` is pure."""
import sys
import os
import threading
import time as _time
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import QTYPES, TEMP_MAX, TEMP_MIN, clamp_temperature, temp_bucket  # noqa: E402
from laya.lang import analyse, detect_script, guess_latin_language, is_english, state_text  # noqa: E402
from laya.router import (  # noqa: E402
    BUNDLE_REPO,
    DEFAULT_MODELS,
    STANDALONE_MODELS,
    _english_from_code,
    _repo_str,
    _split,
    Router,
    match_typed_decisions_workflow,
    normalise_name,
    resolve_model_spec,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


# --------------------------------------------------------------------- script detection
SCRIPTS = [
    ("english", "The customer was charged twice and wants a refund.", "latin"),
    ("armenian", "Հայերեն", "armenian"),
    ("armenian uppercase", "ՀԱՅԵՐԵՆ", "armenian"),
    ("armenian punctuation only", "։֊", "unknown"),
    ("azerbaijani lone schwa", "ə", "latin"),
    ("azerbaijani uppercase", "MÜŞTƏRİ İLƏ ƏLAQƏ SAXLAYIN", "latin"),
    ("french", "Le client a été facturé deux fois et demande un remboursement.", "latin"),
    ("hindi", "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"),
    ("japanese", "お客様は二重に請求されたため返金を希望しています。", "kana"),
    ("chinese", "客户被重复扣款要求退款", "han"),
    ("korean", "고객이 두 번 청구되어 환불을 원합니다", "hangul"),
    ("arabic", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"),
    ("tamil", "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"),
    ("russian", "С клиента дважды сняли деньги и он хочет возврат", "cyrillic"),
    ("thai", "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"),
    ("greek", "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"),
    ("hebrew", "הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"),
    ("empty", "", "unknown"),
    ("digits only", "12345 6789", "unknown"),
]
for label, text, want in SCRIPTS:
    check("script/" + label, detect_script(text), want)


# --------------------------------------------------------------------- english vs not
for label, text, want in [
    ("plain english", "Please refund the duplicate charge on invoice 4411 today.", True),
    ("armenian", "Հայերեն", False),
    ("azerbaijani no diacritics", "Sifarisim gelmedi ve pulum geri qaytarilmadi, zehmet olmasa yoxlayin", False),
    ("azerbaijani few diacritics", "Mən sizin xidmətinizdən razı deyiləm və pulumu geri istəyirəm", False),
    ("english short", "refund me", True),
    ("hindi", "ग्राहक से दो बार शुल्क लिया गया", False),
    ("japanese", "お客様は二重に請求されました", False),
    ("russian", "С клиента дважды сняли деньги", False),
    ("french long", "Le client a été facturé deux fois et il demande un remboursement pour la "
                    "facture qui a été payée le mois dernier avec la carte de crédit", False),
    ("german long", "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die "
                    "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde", False),
    # Latin-script languages with no stopword list of their own: reported in #35, where Romanian
    # states were handed to the English checkpoint (0.330 accuracy, 0.658 ECE on `ro`) instead of
    # the multilingual one. An unidentified language must never be assumed English.
    ("romanian", "Gătește-mi o rețetă de sarmale de post pentru mâine.", False),
    ("romanian invoice", "Am fost taxat de două ori pentru factura din luna martie și vreau banii", False),
    ("polish", "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę", False),
    ("czech", "Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz", False),
    ("turkish", "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım", False),
    ("vietnamese", "Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay", False),
    # English with the odd loanword must not tip over into the multilingual checkpoint
    ("english with loanwords", "We visited a cafe in Zurich and the naive assumption about the "
                               "invoice was wrong, so please refund the duplicate charge", True),
]:
    check("is_english/" + label, is_english(text), want)

# Undecided is reported as undecided rather than dressed up as a detection: a single shared
# function word used to name a language ("para" in Turkish text was called Spanish).
check("latin/undecided is flagged", analyse("Müşteriden iki kez ücret alındı ve para iadesi istiyor")["language_undecided"], True)
check("latin/undecided names no language", analyse("Müşteriden iki kez ücret alındı ve para iadesi istiyor")["language"], None)
check("latin/english is not undecided", analyse("Please refund the duplicate charge on the invoice")["language_undecided"], False)
check("latin/diacritic rate reported", analyse("Gătește-mi o rețetă de sarmale")["diacritic_rate"] > 0.02, True)
check("latin/english has no diacritics", analyse("Please refund the duplicate charge today")["diacritic_rate"], 0.0)
# every branch of analyse() reports the same keys, so a caller can read one without guarding
_KEYS = {"script", "script_profile", "language", "is_english", "language_undecided",
         "diacritic_rate", "non_latin_fraction", "mixed_segment"}
for label, text in [("english", "Please refund the duplicate charge"), ("hindi", "ग्राहक से दो बार"),
                    ("romanian", "Gătește-mi o rețetă de sarmale"), ("no letters", "12345 ???")]:
    check("analyse/keys " + label, set(analyse(text)), _KEYS)
# A 0-0 tie between non-English stopword lists is no evidence for any of them
check("latin_lang/zero tie invents nothing", guess_latin_language("Cât e ora acum la Tokyo"), None)

# Known limitation, kept visible on purpose: Romanian short enough to carry no diacritics and an
# English function word ("in") still reads as English. A real LID model is the fix, not more
# stopwords -- see the discussion in #35.
check("latin/KNOWN GAP romanian without diacritics", is_english("Care este ora in Tokyo?"), True)


# --------------------------------------------------------------------- Latin language guess
for label, text, want in [
    ("english", "The customer was charged twice and wants a refund for this invoice", "en"),
    ("french", "Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr"),
    ("german", "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung", "de"),
    ("spanish", "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura", "es"),
    ("azerbaijani", "Zəhmət olmasa, sifarişim üçün pulu geri qaytarın, çünki məhsul gəlmədi", "az"),
    # one shared function word is not enough to name a language
    ("azerbaijani single hit stays undecided",
     "Müştəridən iki dəfə pul alınıb və o, geri qaytarılmasını istəyir", None),
    ("azerbaijani uppercase dotted I", "MÜŞTƏRİ İLƏ ƏLAQƏ SAXLAYIN VƏ PULU GERİ QAYTARIN", "az"),
    ("too short", "refund", None),
]:
    check("latin_lang/" + label, guess_latin_language(text), want)
# a non-English guess must never fire on ordinary English
check("latin_lang/long english stays en",
      guess_latin_language("Please refund the duplicate charge on invoice 4411 today because "
                           "we have been waiting for three days and nobody has replied to us"), "en")


# --------------------------------------------------------------------- dotted tokens (#177)
# `com` is Portuguese ("with") and `o` is its article, and `_WORD` splits `github.com` into
# `github` + `com`, so every domain in a state scored a Portuguese hit: two of them crossed the
# margin and sent an English state with links in it to the multilingual checkpoint, reported as
# Portuguese. A token whose dot or @ joins word characters is an identifier, not prose.
# `default="english"` because this block is about identifiers not reading as foreign prose, not
# about what the stock default is: a state of bare identifiers is language-undecided either way.
_r_dotted = Router(default="english")
for label, state in [
    ("url and email fields", {"url": "github.com", "email": "user@acme.com"}),
    ("two bare domains", "github.com acme.com"),
    ("link list", {"links": ["example.com", "example.co.uk", "docs.readthedocs.io"]}),
]:
    check("latin_lang/dotted " + label, analyse(state)["language"], None)
    check("is_english/dotted " + label, is_english(state), True)
    check("route/dotted " + label, _r_dotted.route(state).model, "english")
# versions, decimals and dotted abbreviations are identifiers too, and were never prose
for text, label in [("build 1.2.3 on 12.30 with ratio 0.5", "version and decimal"),
                    ("Report by Smith et al., e.g. the U.S.A. office", "dotted abbreviation")]:
    check("is_english/dotted " + label, is_english(text), True)
    check("route/dotted " + label, _r_dotted.route(text).model, "english")
# masking identifiers must not cost the prose around them its language, and a full stop ends a
# sentence rather than joining an identifier: the word before it keeps its letters.
check("latin_lang/portuguese prose with a link",
      guess_latin_language("O cliente nao recebeu o produto, mas quer o dinheiro para a conta, "
                           "veja example.com"), "pt")
check("latin_lang/portuguese sentence with a full stop",
      guess_latin_language("O cliente nao recebeu o produto, mas quer o dinheiro para a conta."), "pt")
check("route/english with a link stays english",
      _r_dotted.route({"body": "Please check example.com and acme.com for the invoice"}).model, "english")


# --------------------------------------------------------------------- state flattening
check("state_text/dict", "charged twice" in state_text({"body": "charged twice", "n": 3}), True)
check("state_text/nested", "deep" in state_text({"a": {"b": ["deep"]}}), True)
check("state_text/list", "x" in state_text(["x", {"y": "z"}]), True)
check("state_text/none", state_text(None), "")
# keys must not drive detection: English keys around Hindi content stay non-English
check("state_text/keys ignored",
      analyse({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})["is_english"], False)

# --------------------------------------------------------------------- dict-state language (#384)
# The same German sentence must route the same way as a string and as a dict value.
# A mapping that is not a builtin dict, a bytes value, and English sibling fields used to
# hide that value: detection then reported language_undecided (or English) and the router
# fell through to the English checkpoint.
_DE = "Mein Konto wurde zweimal belastet"
_r_de = Router()
_de_str = _r_de.route(_DE, {})
_de_dict = _r_de.route({"message": _DE}, {})
check("route/dict german matches string", _de_dict.model, _de_str.model)
check("route/dict german is multilingual", _de_dict.model, "multilingual")
check("route/dict german names de", _de_dict["detection"]["language"], "de")
check("route/dict german is not undecided", _de_dict["detection"]["language_undecided"], False)
check("analyse/dict german matches string", analyse({"message": _DE})["language"], analyse(_DE)["language"])
# English notes must not outvote the message, and a long note must not push it out of the window.
_de_ticket = {
    "ticket_id": "TCK-88213",
    "channel": "web chat",
    "agent_notes": "Please check the shipping status and refund the customer if the charge was duplicated.",
    "message": _DE,
}
check("route/dict german beside english notes", _r_de.route(_de_ticket, {}).model, "multilingual")
check("route/dict german beside english notes is not undecided",
      _r_de.route(_de_ticket, {})["detection"]["language_undecided"], False)
_de_long = {
    "agent_notes": "Please check the shipping status and tell the customer about the refund. " * 80,
    "message": _DE,
}
check("route/dict german after long english note", _r_de.route(_de_long, {}).model, "multilingual")
check("route/dict german after long english note names de",
      _r_de.route(_de_long, {})["detection"]["language"], "de")
# The leaf scan #384 added runs a full `_analyse_text` pass on every line of every value. A line
# too short to be selected is now skipped before that happens. The #384 checks above pin the
# answers; these add the cost side, the constant itself, and the invariant a cheaper scan must
# not trade away.
#
# Ratios to prose of the same size, never wall-clock, so a loaded runner inflates both sides.
# Warm-up then best of nine: the numerators are ~2 ms windows against a ~24 ms denominator, and
# at five reps a single descheduled run moved the 10 000-field ratio to 0.385. Nine reps holds it.
# Previous / ceiling / current, each ceiling the geometric mean of the pair it separates:
#   one-char lines, 50 000 chars       2.85 / 0.80 / 0.23
#   six-char lines, 50 000 chars       1.24 / 0.35 / 0.10
#   10 000 six-character fields        1.85 / 0.68 / 0.25
# Worst current ratio over six trials was 0.232 / 0.098 / 0.249 idle and 0.231 / 0.097 / 0.246
# with twelve busy processes on the box, so every ceiling keeps at least 2.7x of room under load;
# reverting gives 3.19 / 1.25 / 2.27 under that same load, 3.3x or more the other way. The
# measured ratio goes in the check name, so a red CI log says what it was, not only "got False".
def _ms(state, reps=9):
    analyse(state)
    analyse(state)
    best = float("inf")
    for _ in range(reps):
        t = _time.perf_counter()
        analyse(state)
        best = min(best, _time.perf_counter() - t)
    return best * 1000


# Prose LINES, not one long line: `_leaf_non_english` slices each line to 4000 characters, so a
# 50 000-character one-liner does a fraction of the work and would be a meaningless denominator
# (2.2 ms against 24.1 ms for the same characters as lines). Every state here is a dict on
# purpose: `analyse` returns a plain string's verdict before the leaf scan runs.
_PROSE_50K = {"body": "\n".join(["The customer was billed twice and wants a refund."] * 1020)[:50_000]}
_prose_ms = _ms(_PROSE_50K)
for _name, _state, _ceiling in (
    ("one-char lines", {"body": ("a\n" * 25_000)[:50_000]}, 0.80),
    ("six-char lines", {"body": ("abcdef\n" * 7_142)[:50_000]}, 0.35),
    ("10k six-char fields", {"f%d" % i: "abcdef" for i in range(10_000)}, 0.68),
):
    _ratio = _ms(_state) / _prose_ms
    check("leaf scan/%s vs prose = %.2f, ceiling %.2f" % (_name, _ratio, _ceiling),
          _ratio < _ceiling, True)

# The constant. Seven is the largest sound threshold: each branch of `_leaf_non_english` needs
# four `_WORD` tokens -- maximal runs of letters and letter-like numerals, so four need three
# separators between them -- or ten letters. "é à ü ö" is exactly seven characters and four
# tokens, and must still be read: this is the check that pins the threshold, and raising it to 8
# turns that check red. "é à üö" is six characters and three tokens; no six-character line can be
# selected at any threshold, so that check is a behaviour pin rather than a second bound -- it
# passes on unguarded code too, and is here to catch a future change that makes short lines
# selectable. Both values sit past 4000 characters of English, which is what makes the leaf scan
# the code under test: before that the segment scan has already answered.
_EN_PAST_SEGMENT_CAP = "The customer was billed twice and wants a refund. " * 120
check("leaf scan/a 7-character foreign line is still read",
      analyse({"note": _EN_PAST_SEGMENT_CAP, "msg": "é à ü ö"})["is_english"], False)
check("leaf scan/a 6-character line stays English",
      analyse({"note": _EN_PAST_SEGMENT_CAP, "msg": "é à üö"})["is_english"], True)

# #384's invariant, which a budget shared across the state would break: a long earlier value must
# not stop a later one from being read. Cheap to keep, and it is the one way a future attempt to
# bound this scan by total characters would go wrong silently -- the state below would route to
# the English checkpoint, which BENCHMARKS.md shows collapsing off English.
check("leaf scan/a later value is still read after a 50k earlier one",
      analyse({"pad": "x " * 25_000, "msg": "我们三月份被重复收费了两次。"})["is_english"], False)

from collections import UserDict  # noqa: E402
from types import MappingProxyType  # noqa: E402
check("route/userdict german", _r_de.route(UserDict({"message": _DE}), {}).model, "multilingual")
check("route/userdict german is not undecided",
      _r_de.route(UserDict({"message": _DE}), {})["detection"]["language_undecided"], False)
check("route/mappingproxy german",
      _r_de.route(MappingProxyType({"message": _DE}), {}).model, "multilingual")
check("route/bytes german value",
      _r_de.route({"message": _DE.encode("utf-8")}, {}).model, "multilingual")
# A name beside an English request is not a second message.
check("route/dict cyrillic name stays english",
      _r_de.route({"name": "Антон Павлович Чехов",
                   "body": "Please refund the duplicate charge on invoice 4411 today."}, {}).model,
      "english")
check("route/dict jose stays english",
      _r_de.route({"name": "José",
                   "body": "Please refund the duplicate charge on invoice 4411 today."}, {}).model,
      "english")
check("route/dict english body stays english",
      _r_de.route({"body": "Please refund the duplicate charge on invoice 4411 today."}, {}).model,
      "english")


# --------------------------------------------------------------------- workflow signatures
check("profile/armenian", analyse("Հայերեն")["script_profile"], {"armenian": 1.0})
check("profile/armenian mixed with Latin", analyse("Հայերեն abc")["non_latin_fraction"], 0.7)
check("profile/azerbaijani schwa is latin", analyse("ələ")["script_profile"], {"latin": 1.0})

TD = {
    "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
    "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
    "invoice_processing": ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    "security_incidents": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
}
for wf, ids in TD.items():
    check("workflow/" + wf, match_typed_decisions_workflow({i: {} for i in ids}), wf)
check("workflow/partial overlap", match_typed_decisions_workflow({"urgency": {}, "category": {}}), None)
check("workflow/superset", match_typed_decisions_workflow({i: {} for i in TD["customer_service"] + ["extra"]}), None)
check("workflow/empty", match_typed_decisions_workflow({}), None)


# --------------------------------------------------------------------- name normalisation
for alias, want in [("en", "english"), ("laya", "english"), ("multi", "multilingual"),
                    ("ML", "multilingual"), ("typed", "typed-decisions"),
                    ("typed_decisions", "typed-decisions"), ("English", "english"),
                    ("convaiinnovations/laya".split("/")[-1], "english")]:
    check("alias/" + alias, normalise_name(alias), want)
try:
    normalise_name("nope")
    FAIL.append("alias/unknown: should have raised")
except ValueError:
    PASS.append("alias/unknown raises")


# --------------------------------------------------------------------- registry spec resolution (#780)
# `load()` resolves a name or alias through this table instead of forwarding it to the Hub as a
# repo id, so both entry points read one registry. Anything the registry does not know -- a repo
# id, a local path, an ONNX export -- resolves to None and is left alone.
for name, want in [("english", ("convaiinnovations/laya", None)),
                   ("laya", ("convaiinnovations/laya", None)),
                   ("typed-decisions", ("convaiinnovations/laya", "typed-decisions")),
                   ("typed", ("convaiinnovations/laya", "typed-decisions")),
                   ("ml", ("convaiinnovations/laya", "multilingual")),
                   ("MULTI", ("convaiinnovations/laya", "multilingual")),
                   (" typed-decisions ", ("convaiinnovations/laya", "typed-decisions"))]:
    check("spec/" + name, resolve_model_spec(name), want)
check("spec/agrees with normalise_name",
      resolve_model_spec("decisions"), tuple(_split(DEFAULT_MODELS[normalise_name("decisions")])))
for unknown in ("convaiinnovations/laya", "test/custom-model", "/tmp/checkpoint", "./local",
                "nope", "", "convaiinnovations/laya-typed-decisions"):
    check("spec/not a name: " + (unknown or "<empty>"), resolve_model_spec(unknown), None)


# --------------------------------------------------------------------- routing decisions
r = Router()
Q_GENERIC = {"dept": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": None, "tech": None}}}
Q_TD = {i: {"type": "noul", "instructions": "x"} for i in TD["customer_service"]}

cases = [
    ("english text", {"body": "I was charged twice, please refund."}, Q_GENERIC, {}, "english"),
    ("armenian text", {"body": "Հայերեն"}, Q_GENERIC, {}, "multilingual"),
    ("armenian explicit override", {"body": "Հայերեն"}, Q_GENERIC, {"model": "english"}, "english"),
    ("azerbaijani no diacritics", {"body": "Sifarisim gelmedi ve pulum geri qaytarilmadi, zehmet olmasa yoxlayin"},
     Q_GENERIC, {}, "multilingual"),
    ("azerbaijani explicit override", {"body": "Müştəridən iki dəfə pul alınıb"}, Q_GENERIC,
     {"model": "english"}, "english"),
    ("hindi text", {"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC, {}, "multilingual"),
    ("japanese text", {"body": "二重に請求されました"}, Q_GENERIC, {}, "multilingual"),
    ("korean text", {"body": "두 번 청구되었습니다"}, Q_GENERIC, {}, "multilingual"),
    ("arabic text", {"body": "تم خصم المبلغ مرتين"}, Q_GENERIC, {}, "multilingual"),
    # Latin brand names are the letter plurality here, but the request itself is CJK
    ("chinese with a brand", {"body": "我的 iPhone 15 Pro Max 订单还没到"}, Q_GENERIC, {}, "multilingual"),
    ("japanese with brands", {"body": "Amazonで買ったiPhoneが届かない"}, Q_GENERIC, {}, "multilingual"),
    ("korean with a brand", {"body": "Samsung Galaxy 주문이 아직 안 왔어요"}, Q_GENERIC, {}, "multilingual"),
    ("english with a han name", {"body": "My name is 王小明 and my order is late"}, Q_GENERIC, {}, "english"),
    # an English wrapper dilutes the share, but the request is still CJK
    ("chinese in a ticket", {"ticket_id": "TCK-88213", "channel": "web chat",
                             "agent_notes": "Customer asked about a delayed order. Please check shipping status.",
                             "message": "我的订单已经两个星期了还没有到"}, Q_GENERIC, {}, "multilingual"),
    ("korean after english turns", [{"role": "agent", "text": "Hello! Thanks for contacting support."},
                                    {"role": "agent", "text": "Could you share your order number please?"},
                                    {"role": "user", "text": "주문번호는 5521이고 아직 배송이 안 됐어요"}],
     Q_GENERIC, {}, "multilingual"),
    ("english with greek symbols", {"request": "Compute the mean μ and variance σ of X, then P(|X-μ| > 2σ)."},
     Q_GENERIC, {}, "english"),
    # English prose that names someone in their own script: the name is not the request, and a
    # capitalised run, a lone symbol and a pronunciation are all annotation rather than content.
    ("english prose, russian name",
     {"body": "Anton Pavlovich Chekhov (Russian: Антон Павлович Чехов) was a playwright."},
     Q_GENERIC, {}, "english"),
    ("english prose, name with IPA",
     {"body": "Vladimir Nabokov (Russian: Влади́мир Набо́ков [vlɐˈdʲimʲɪr nɐˈbokəf]) wrote Lolita "
              "and taught literature at Cornell for more than a decade."},
     Q_GENERIC, {}, "english"),
    ("english prose, greek name",
     {"body": "Eleftherios Venizelos (Greek: Ελευθέριος Βενιζέλος) served as prime minister."},
     Q_GENERIC, {}, "english"),
    ("english prose, hebrew name",
     {"body": "Amos Oz (Hebrew: עמוס עוז), born Amos Klausner, was an Israeli writer and professor "
              "of literature at Ben-Gurion University of the Negev in Beersheba."},
     Q_GENERIC, {}, "english"),
    ("german text", {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung "
                             "fuer die Rechnung die nicht korrekt ist"}, Q_GENERIC, {}, "multilingual"),
    ("explicit model", {"body": "anything"}, Q_GENERIC, {"model": "multilingual"}, "multilingual"),
    ("explicit model overrides script", {"body": "मुझसे दो बार"}, Q_GENERIC,
     {"model": "english"}, "english"),
    ("explicit task", {"body": "x"}, Q_GENERIC, {"task": "typed_decisions"}, "typed-decisions"),
    ("explicit lang en", {"body": "मुझसे दो बार"}, Q_GENERIC, {"lang": "en"}, "english"),
    ("explicit lang de", {"body": "hello there"}, Q_GENERIC, {"lang": "de"}, "multilingual"),
    ("td workflow, auto OFF", {"body": "I was charged twice"}, Q_TD, {}, "english"),
    # letterless, so these follow `default`, which is multilingual since 0.4.0
    ("empty state", {}, Q_GENERIC, {}, "multilingual"),
    ("none state", None, Q_GENERIC, {}, "multilingual"),
]
for label, state, qs, kw, want in cases:
    check("route/" + label, r.route(state, qs, **kw)["model"], want)

# auto task detection is opt-in
r_auto = Router(auto_task_detection=True)
check("route/td workflow, auto ON",
      r_auto.route({"body": "I was charged twice"}, Q_TD)["model"], "typed-decisions")
check("route/auto ON but generic questions",
      r_auto.route({"body": "I was charged twice"}, Q_GENERIC)["model"], "english")
# explicit model still beats auto-detected workflow
check("route/explicit beats workflow",
      r_auto.route({"body": "x"}, Q_TD, model="multilingual")["model"], "multilingual")
# An auto-detected workflow reports `repo` like every other branch does. It used to hand back the raw
# (repo, subfolder) spec, which serialises to a JSON list instead of the "repo/subfolder" string.
check("route/auto workflow repo is a string",
      r_auto.route({"body": "I was charged twice"}, Q_TD)["repo"], "convaiinnovations/laya/typed-decisions")
check("route/auto workflow repo matches explicit task",
      r_auto.route({"body": "I was charged twice"}, Q_TD)["repo"],
      r_auto.route({"body": "I was charged twice"}, Q_TD, task="typed_decisions")["repo"])
check("route/auto workflow repo standalone",
      Router(auto_task_detection=True, standalone_repos=True).route({"body": "x"}, Q_TD)["repo"],
      "convaiinnovations/laya-typed-decisions")

# decision payload shape
d = r.route({"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC)
check("decision/has repo", d["repo"], "convaiinnovations/laya/multilingual")
check("decision/has reason", isinstance(d["reason"], str) and len(d["reason"]) > 0, True)
check("decision/detection script", d["detection"]["script"], "devanagari")
check("decision/.model property", d.model, "multilingual")
check("decision/is dict", isinstance(d, dict), True)

# default override
check("route/custom default", Router(default="multilingual").route("12345", Q_GENERIC)["model"],
      "multilingual")


# --------------------------------------------------------------------- unknown-Latin routing (#35)
_r_lat = Router()
for label, text in [
    ("romanian", "Gătește-mi o rețetă de sarmale de post pentru mâine."),
    ("romanian agent request", "Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez."),
    ("polish", "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę"),
    ("turkish", "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım"),
]:
    check("route/unknown latin " + label, _r_lat.route(text).model, "multilingual")
# and the reason must say what it actually routed on, not report a language it did not identify
check("route/undecided reason mentions letters",
      "not identified" in _r_lat.route("Müşteriden iki kez ücret alındı ve para iadesi istiyor").reason, True)
check("route/english still english",
      _r_lat.route("Please refund the duplicate charge on invoice 4411 today.").model, "english")
# "refund me" names no language, so it follows `default` like any other undecided text rather
# than being hard-coded English. With an English default it still reaches the English checkpoint.
check("route/short english follows default",
      Router(default="english").route("refund me").model, "english")

# Undecided Latin text follows `default`, as a state with no letters already did. Short messages made
# only of content words carry nothing that names their language, and hard-coding English for them
# sent every short Portuguese message to the checkpoint that is 0.97 confident at 0.47 accuracy on
# `pt`, whatever the router was configured with. Since 0.4.0 the stock default is `multilingual`,
# so both arms below are exercised: the stock one and an explicit English override.
_r_ml = Router(default="multilingual")
_r_en = Router(default="english")
for text in ["Quero cancelar", "Esqueci minha senha", "Fui cobrado duas vezes",
             "Produto veio quebrado, quero trocar", "refund me"]:
    check("route/undecided follows default " + text[:24], _r_ml.route(text).model, "multilingual")
    check("route/undecided stock default " + text[:24], _r_lat.route(text).model, "multilingual")
    check("route/undecided english override " + text[:24], _r_en.route(text).model, "english")
check("route/undecided reason names the default",
      "using default (multilingual)" in _r_ml.route("Esqueci minha senha").reason, True)
# identified English is not undecided, so a non-English default leaves it alone
for text in ["Please refund the duplicate charge", "Please refund the duplicate charge on invoice 4411 today."]:
    check("route/identified english ignores default " + text[:24], _r_ml.route(text).model, "english")
check("route/english reason unchanged",
      _r_lat.route("Please refund the duplicate charge on invoice 4411 today.").reason, "English Latin text")


# --------------------------------------------------------------------- mixed states
# A Portuguese ticket carrying an English stack trace, error payload or form template read as English
# as a whole -- the English part is longer -- and went to the checkpoint that cannot read the
# customer's own words. Any line or field that on its own is named a non-English language now wins.
_TRACE = ("O sistema caiu de novo hoje de manhã, segue o log:\n"
          "Traceback (most recent call last):\n"
          "  File \"/app/main.py\", line 42, in handler\n"
          "    return self.process(request)\n"
          "ConnectionError: the connection to the database was refused because the pool is "
          "exhausted and there is no available slot for this request")
for label, state, segment in [
    ("portuguese ticket + english traceback", _TRACE, "O sistema caiu de novo hoje de manhã, segue o log:"),
    ("portuguese field + english error payload",
     {"descricao": "O pagamento não foi processado",
      "error": {"code": "card_declined", "message": "Your card was declined. Please try again with a "
                "different card or contact your bank for more information."}},
     "O pagamento não foi processado"),
    ("english form template + portuguese body",
     {"subject": "New ticket from the web form", "body": "Quero cancelar meu plano"},
     "Quero cancelar meu plano"),
    ("parenthesis in prose is not code",
     {"subject": "Urgent: production is down for all customers since the last deploy and the "
                 "status page is red for the whole region",
      "body": "Deu erro (500) no login, alguém pode ver isso agora?"},
     "Deu erro (500) no login, alguém pode ver isso agora?"),
    # the rule runs both ways: an English ticket that pastes a foreign log goes to multilingual too
    ("english ticket + portuguese error log",
     "Our Brazilian branch cannot issue invoices since this morning. The system shows this message:\n"
     "ERRO: Não foi possível emitir a nota fiscal, o certificado digital está vencido\n"
     "Can you help us before the end of the day?",
     "ERRO: Não foi possível emitir a nota fiscal, o certificado digital está vencido"),
    ("english ticket + german error log",
     "The nightly sync to the Munich server keeps failing and we lose the whole batch.\n"
     "Fehler: Die Verbindung zum Server wurde unterbrochen, bitte versuchen Sie es spaeter noch einmal\n"
     "Please check the firewall rules on your side.",
     "Fehler: Die Verbindung zum Server wurde unterbrochen, bitte versuchen Sie es spaeter noch einmal"),
    ("english ticket + spanish error payload",
     {"subject": "Payment failed for a customer in Madrid",
      "description": "The customer tried three times with the same card and each attempt was declined by "
                     "the gateway, so we would like to know whether the problem is on our side or with the bank.",
      "error": {"code": "card_declined",
                "message": "La tarjeta fue rechazada por el banco emisor, contacte con su banco"}},
     "La tarjeta fue rechazada por el banco emisor, contacte con su banco"),
    # acronyms are dropped only from mixed-case text: a line written all in capitals keeps its words
    ("all-caps portuguese line",
     "This is the fourth email I have sent about the same order and nobody has answered any of them.\n"
     "The customer wrote this in the chat and then closed the window:\n"
     "QUERO MEU DINHEIRO DE VOLTA AGORA\n"
     "Could someone from the billing team look at order 5512 today?",
     "QUERO MEU DINHEIRO DE VOLTA AGORA"),
]:
    check("mixed/is not english: " + label, is_english(state), False)
    check("mixed/segment reported: " + label, analyse(state)["mixed_segment"], segment)
    check("mixed/routes multilingual: " + label, _r_lat.route(state).model, "multilingual")
check("mixed/reason names the segment",
      "a line or field reads as 'pt'" in _r_lat.route(_TRACE).reason, True)
# English stays English: several English lines, a short foreign sign-off, and code pasted into a request
# (`os.path` reads as Portuguese, `round(el, 2)` as Spanish, `np.mean(na)` as Portuguese)
for label, state in [
    ("multi-line english", "Hi team,\nThe export failed again last night.\nCan you check the logs?\nThanks"),
    ("short portuguese sign-off", "Please resend the invoice for March, the amount is wrong.\nAtenciosamente, Joao"),
    ("os.path", "The build broke after the refactor.\nREPO = os.path.dirname(os.path.dirname(__file__))\n"
                "Please take a look at the import paths when you can."),
    ("round(el)", "The latency script crashes on large runs.\nmix[key] = {\"total_s\": round(el, 2)}\n"
                  "Can you check why the stream is empty?"),
    ("np.mean(na)", "The summary is wrong for empty suites.\nif na: non[m] = round(float(np.mean(na)), 4)\n"
                    "Please guard the empty case."),
    ("english json", {"status": "open", "priority": "high",
                      "message": "The customer was charged twice and wants a refund"}),
    # a line carries far less text than a state, so its evidence must be two different words and no
    # acronyms or slash compounds. A ham-radio listing on 20 Newsgroups (misc.forsale/76512) went to
    # multilingual on `COM ... COM` alone; hockey picks on the team codes, OS/2 on `os` and `dos`.
    ("same word twice (Nav/Com, COM)",
     "I'm looking for good deals on the following (used or new):\nAviation Headsets (with mic).\n"
     "Handheld Nav/Com tranciever (may consider COM only).\nPortable GPS or Loran Navigator."),
    ("team codes", "Round two predictions for the pool, as promised.\nQUE  vs MON:  MON  in 7.\n"
                   "PIT  vs NYI:  PIT  in 5."),
    ("slash compound", "I need a converter for these image formats.\n"
                       "DOS, OS/2 or platform independent programs if possible.\nThanks in advance."),
    ("backslash path", "My modem stopped answering after the upgrade.\nC:\\DOS\\mode COM1:9600,n,8,1,p\n"
                       "Is that the right line for a 9600 baud connection?"),
]:
    check("mixed/english stays english: " + label, is_english(state), True)
    check("mixed/no segment: " + label, analyse(state)["mixed_segment"], None)
# a state is user input: one long line with no joiner took 43 s at 40,000 characters when compounds
# were stripped with an open-ended regex; the segment check now reads at most the 4,000-character cap
import time as _time
_t0 = _time.perf_counter()
analyse({"subject": "The export failed again last night for the whole region", "body": "a" * 200_000})
check("mixed/long single-line field stays fast", _time.perf_counter() - _t0 < 5.0, True)
check("mixed/segment check reads at most the cap",
      analyse({"log": "The export failed again last night for the whole region. " * 80,
               "body": "Quero cancelar meu plano agora mesmo"})["mixed_segment"], None)


# --------------------------------------------------------------------- plain-ASCII Romance (#172)
# A state that lost its accents carries no diacritic rate for the non-English signal to read, and
# mail clients and ticket systems strip them routinely, so the function-word lists are the only
# evidence left. Spanish and Italian complaints were reported `is_english=True` and handed to the
# English checkpoint -- the one that collapses off English -- while the accented spelling of the
# same text reached multilingual. The lists for fr/es/pt/it held mostly accented words (`está`,
# `más`, `è`, `être`, `não`) plus a handful of unaccented ones, so a stripped state matched one
# word or none and fell under the two-hit margin below.
for lang, text in [
    ("es", "El pedido llego roto y nadie responde cuando escribo al soporte"),
    ("es", "Quiero cancelar mi plan y pedir un reembolso"),
    ("es", "La factura tiene un error en el importe total"),
    ("es", "Necesito que me devuelvan el dinero de la compra duplicada"),
    ("it", "Il cliente e stato addebitato due volte e vuole un rimborso"),
    ("it", "Voglio cancellare il mio abbonamento e chiedere un rimborso"),
    ("it", "La fattura contiene un errore nell importo totale"),
    ("pt", "O cliente foi cobrado duas vezes e quer o dinheiro de volta"),
    ("fr", "Le client a ete facture deux fois et demande un remboursement"),
    ("fr", "Je ne peux pas acceder a mon compte et j ai besoin d aide"),
]:
    check("latin_lang/plain ascii " + lang + " " + text[:32], guess_latin_language(text), lang)
    check("is_english/plain ascii " + lang + " " + text[:32], is_english(text), False)
    check("route/plain ascii " + lang + " " + text[:32], _r_lat.route(text).model, "multilingual")
# the accented spellings must keep working: those route on the diacritic rate
for lang, text in [
    ("es", "La facturación tiene un error y necesito una corrección urgente"),
    ("it", "La fattura è sbagliata, devo avere un rimborso per il pagamento"),
    ("fr", "La commande est arrivée cassée et personne ne répond au support"),
]:
    check("route/accented " + lang, _r_lat.route(text).model, "multilingual")

# English must not move for this. The added words are ordinary English tokens as well -- `de facto`,
# `et al.`, `e.g.`, `la carte`, `UN`, `MI5`, `DOS` -- and a state carrying none of them is the case
# that has to keep routing to English.
for text in [
    "The customer was charged twice and wants a refund for this invoice",
    "Please cancel my subscription and refund the duplicate charge today",
    "The report by Smith et al. shows the de facto standard, e.g. the LA office and Rio",
    "Our MI5 and UN contacts discussed the DOS attack in LA last month",
    "No refund was issued, so I am writing to you again about invoice 4411",
    "no refund no reply",
    "The son of the director filed a complaint about the duplicate invoice",
]:
    check("is_english/romance control " + text[:32], is_english(text), True)
    check("route/romance control " + text[:32],
          Router(default="english").route(text).model, "english")

# A word several lists claim (`la`, `e`, `o`) says "not English" without saying *which* language, so
# it may not name one on its own -- the same rule as the 0-0 tie above, which is why the sample
# below still names nothing even though `la` and `e` now count for Italian: it routes to multilingual
# on the Romanian diacritic, not on a guessed language.
check("latin_lang/shared words alone name nothing",
      analyse("Cât e ora acum la Tokyo")["language"], None)
check("route/shared words still multilingual",
      _r_lat.route("Cât e ora acum la Tokyo").model, "multilingual")
# ...but a distinctive word in the same state is enough to name the language it belongs to.
check("latin_lang/distinctive word names the language",
      guess_latin_language("La fattura contiene un errore nell importo totale"), "it")
check("latin_lang/shared hits still count toward a named language",
      guess_latin_language("La factura tiene un error en el importe total"), "es")


# --------------------------------------------------------------------- Brazilian support text
# Short Brazilian messages lean on `você`/`vc`, the unaccented `nao`/`voce` and `gostaria`, none of
# which the `pt` list held, so each matched one word, fell under the two-hit margin and went to the
# English checkpoint -- which on `pt` reports 0.97 mean confidence at 0.47 accuracy (ECE 0.51).
for text in [
    "Boa tarde, gostaria de cancelar o plano",
    "Voce pode me mandar a nota fiscal?",
    "Você pode me mandar a nota fiscal?",
    "Nao consigo fazer login no app",
    "Pix nao caiu na conta",
    "Gostaria de saber o prazo de entrega",
    "Estou esperando faz uma semana",
    "Vc pode cancelar pra mim?",
    # a bug report whose jargon is English keeps only these words to say it is Portuguese
    "Deu erro 500 no endpoint de login depois do update",
    "Depois da atualizacao ninguem consegue logar",
    "Antes funcionava, agora deu pau",
    "Estava tudo certo ate a migracao",
    "Entao o sistema travou de novo",
]:
    check("latin_lang/pt-br " + text[:32], guess_latin_language(text), "pt")
    check("route/pt-br " + text[:32], _r_lat.route(text).model, "multilingual")
# each added word that is also an English token must not move English text
for text in [
    "Our Sao Paulo office still has not received the invoice",
    "My VC asked for the cap table and the invoice",
    "Nossa Cafe charged my card twice this month",
    "The Boa Vista branch reported an outage this morning",
    "The pra team will review the claim tomorrow",
]:
    check("route/pt-br control " + text[:32], _r_lat.route(text).model, "english")


# --------------------------------------------------------------------- romanized Bangla
# Bangla is often typed in Latin letters ("Banglish") when no Bengali keyboard is at hand. It has no
# diacritics and matched no stopword list, so it was reported `is_english=True` and handed to the
# English checkpoint, which scores 0.08 on Bangla MASSIVE at 0.94 confidence. The Bengali-script
# spelling of the same text already routed on script alone.
for text in [
    "amar kach theke duibar taka kata hoyeche, doya kore ferot din",
    "ami invoice er jonno duibar charge peyechi, refund chai",
    "Ami ei product ta niye khub hotash, ekhon e cancel korte chai",
    "apnara keno amar call dhorchen na? ajke kichu ekta korun",
    "bhai amar account e login korte parchi na",
    "taka ekhono ferot paini, kobe pabo?",
    "order ta kobe asbe bolte parben?",
]:
    check("latin_lang/banglish " + text[:32], guess_latin_language(text), "bn")
    check("is_english/banglish " + text[:32], is_english(text), False)
    check("route/banglish " + text[:32], _r_lat.route(text).model, "multilingual")
check("route/bengali script", _r_lat.route("আমার কাছ থেকে দুইবার টাকা কাটা হয়েছে").model, "multilingual")

# English must not move. `chai`, `ar`, `ami`, `koto`, `kore`, `oi` are Bangla words that also turn
# up in English text as a drink, an acronym or a name; one of them next to English function words
# stays English.
for text in [
    "Chai latte order was charged twice, please refund the extra amount",
    "The AR team says the ETA for the fix is Friday",
    "Ami Patel from the Koto office sent the invoice to Kore Ltd",
    "Our AR and VR demo in Oi Bahia went well, the client wants a quote",
    "Take the age of the account into account before you refund",
]:
    check("is_english/banglish control " + text[:32], is_english(text), True)
    check("route/banglish control " + text[:32], _r_lat.route(text).model, "english")
# ...and no other language may move either: the `bn` list claims no word another list holds, and
# leaves out Romance words such as `ora`, `nei`, `vai`.
from laya.lang import _STOP  # noqa: E402
check("latin_lang/bn list shares no word with another list",
      sorted(w for w in _STOP.get("bn", ()) for lg, words in _STOP.items() if lg != "bn" and w in words), [])
check("latin_lang/romanian with ei stays romanian",
      guess_latin_language("Ei nu sunt de acord cu factura, vreau o corecție"), "ro")


# --------------------------------------------------------------------- plain-ASCII German (#54)
# No umlaut for the diacritic rate to catch, and `in`/`was` counted for English alone, so these were
# labelled English and handed to the checkpoint that cannot read them.
for text in ["trage diesen termin in meinen kalender ein",
             "wie lautet die temperatur in fulda in hessen",
             "schalte das licht im wohnzimmer aus",
             "was ist die aktuelle zeit"]:
    check("latin_lang/ascii german " + text, guess_latin_language(text), "de")
    check("route/ascii german " + text, _r_lat.route(text).model, "multilingual")
check("route/english control for #54",
      _r_lat.route("I would like to book a flight to Berlin tomorrow").model, "english")
# English that shares words with the German list stays English. `in` and `den` leave the first, a real
# en-US MASSIVE utterance, one German hit short of flipping; the chat line carries `im` and flips if
# any one of the English words `am`, `an` or `so` joins the German list.
for text in ["turn off smart lamp in den", "im so sorry, am an hour late, stuck in traffic"]:
    check("route/english sharing german words " + text, _r_lat.route(text).model, "english")
# German words that Spanish (`es`) or French (`du`) also claim would stop naming those languages
check("latin_lang/spanish es stays evidence", guess_latin_language("que hora es en australia"), "es")
check("latin_lang/french du stays evidence", guess_latin_language("baisse le volume du haut-parleur"), "fr")

# ------------------------------------------------------------------ accented loanwords in English (#337)
# The diacritic rate is measured over every character, so one `é` in a short English sentence
# clears the 0.02 floor and used to veto the English resolution outright: plain English with a
# loanword or foreign name (`café`, `résumé`, `José`, `Zürich`) went to the checkpoint the
# README says collapses on English-heavy Latin text. English wins the veto back only through the
# word rescue: at least two distinct function words no other list holds, and at most one word
# carrying a non-English letter.
for text in ["Please send me the café menu today please",
             "Could you email me your résumé before the meeting",
             "Send the invoice to José before Friday",
             "We visited Zürich last summer and loved it"]:
    check("latin_lang/loanword english stays english " + text, guess_latin_language(text), "en")
    check("route/loanword english stays english " + text, _r_lat.route(text).model, "english")
# Genuinely non-English accented text keeps its multilingual routing: a German sentence with
# umlauts and no English function word is not rescued.
check("latin_lang/accented german stays non-english",
      is_english("Grüße aus Köln, wir melden uns wegen der Rechnung"), False)
check("route/accented german stays multilingual",
      _r_lat.route("Grüße aus Köln, wir melden uns wegen der Rechnung").model, "multilingual")
# Danish still has no list here, and its accented sentences pick up just one or two English-shaped
# words (`i`, `at`, `for`, `have`), which is not the two-distinct-word English the rescue requires.
for text in ["sluk lyset i soveværelset",                          # da
             "kan jeg få en refundering for det dobbelte beløb",   # da
             "stäng av ljuset i sovrummet"]:                        # sv
    check("latin_lang/nordic accented stays non-english " + text, is_english(text), False)
    check("route/nordic accented stays multilingual " + text, _r_lat.route(text).model, "multilingual")

# Swedish-specific words now identify both normal text and ASCII-normalised support prose. The
# second sample is the real false-negative shape: `få` alone was too weak to stop its two English-
# shaped words (`i`, `kan`) from pulling the request onto the English checkpoint.
for text in ["Om ni inte kan få tillbaka de raderade filerna i dag avslutar jag mitt abonnemang.",
             "Om ni inte kan fa tillbaka de raderade filerna i dag avslutar jag mitt abonnemang.",
             "jag vill att ni hjalper mig med detta"]:
    check("latin_lang/swedish is named " + text, guess_latin_language(text), "sv")
    check("route/swedish uses multilingual " + text, _r_lat.route(text).model, "multilingual")
    check("route/swedish detection reports sv " + text, _r_lat.route(text)["detection"]["language"], "sv")
# Common Swedish support phrasing should identify the language across billing, account, technical
# and delivery messages, both with and without Swedish diacritics.
for text in ["Kan ni hjälpa mig?", "Min faktura är fel", "Jag behöver hjälp med betalningen",
             "Kan ni skicka kvittot igen?", "Jag vill byta lösenord",
             "Appen kraschar när jag öppnar inställningarna",
             "Var hittar jag inställningen för tvåfaktorsinloggning?",
             "Vi har debiterats två gånger för mars",
             "Var är mitt paket? Spårningen har inte uppdaterats"]:
    check("latin_lang/swedish support is named " + text, guess_latin_language(text), "sv")
    check("route/swedish support uses multilingual " + text,
          _r_lat.route(text).model, "multilingual")
for text in ["min faktura ar fel", "jag behover hjalp med betalningen",
             "kan ni skicka kvittot igen", "jag vill byta losenord",
             "appen kraschar nar jag oppnar installningarna",
             "var hittar jag installningen for tva faktorsinloggning",
             "vi har blivit debiterade tva ganger for mars",
             "var ar mitt paket sparningen har inte uppdaterats"]:
    check("latin_lang/ascii swedish support is named " + text, guess_latin_language(text), "sv")
    check("route/ascii swedish support uses multilingual " + text,
          _r_lat.route(text).model, "multilingual")
# Short login failures often contain English technical vocabulary and can arrive without Swedish
# diacritics. Swedish `kan` + `inte` must outweigh the incidental English token `in`.
for text in ["Kan inte logga in", "kan inte logga in", "Jag kan inte logga in", "Vi kan inte logga in"]:
    check("latin_lang/short swedish login is named " + text, guess_latin_language(text), "sv")
    check("route/short swedish login uses multilingual " + text,
          _r_lat.route(text).model, "multilingual")
    check("route/short swedish login detection reports sv " + text,
          _r_lat.route(text)["detection"]["language"], "sv")
# Two- and three-word Swedish support fragments do not reach the general four-word evidence
# threshold. Only distinctly Swedish terms should name them; generic words and Nordic controls
# remain undecided rather than being guessed as Swedish.
for text in ["Ingen åtkomst", "Ingen atkomst", "Fakturan är fel", "Betalningen nekades",
             "Behöver hjälp", "Behover hjalp", "Glömt lösenord", "Glomt losenord",
             "Felmeddelande igen", "Kvitto saknas", "Inloggningen fungerar"]:
    check("latin_lang/short Swedish support is named " + text, guess_latin_language(text), "sv")
    check("route/short Swedish support uses multilingual " + text,
          _r_lat.route(text).model, "multilingual")
    check("route/short Swedish support detection reports sv " + text,
          _r_lat.route(text)["detection"]["language"], "sv")
for text in ["Ingen adgang", "Fakturaen feil", "Glemt passord", "Pakken forsinket",
             "No account access", "Password forgotten"]:
    check("latin_lang/short non-Swedish stays undecided " + text,
          guess_latin_language(text), None)
check("latin_lang/english login stays english",
      guess_latin_language("I cannot login to my account"), "en")
check("route/english login stays english",
      _r_lat.route("I cannot login to my account").model, "english")
check("latin_lang/danish login is not called swedish",
      guess_latin_language("Jeg kan ikke logge inn"), None)
check("latin_lang/danish account login is not called swedish",
      guess_latin_language("Jeg kan ikke logge ind på min konto"), None)
# Two non-English-letter words is a running non-English vocabulary, not one loanword: the rescue
# does not fire even with English function words present.
check("latin_lang/two diacritic words are not one loanword",
      is_english("The naïve façade needs a fresh coat of paint"), False)


# --------------------------------------------------------------------- temperature clamp (#35)
# A fitted temperature below 1 sharpens logits. The shipped `choice:11+` bucket is 0.1006, which
# turned a 0.24 top probability into 0.99 confidence on 13-option skill routing.
check("clamp/pathological sharpening", clamp_temperature(0.1006), 0.5)
check("clamp/shipped choice:11+ is rejected", clamp_temperature(0.10058280825614929), TEMP_MIN)
check("clamp/legitimate value untouched", clamp_temperature(1.7601518630981445), 1.7601518630981445)
check("clamp/neutral untouched", clamp_temperature(1.0), 1.0)
check("clamp/upper bound", clamp_temperature(9.0), TEMP_MAX)
check("clamp/zero", clamp_temperature(0.0), TEMP_MIN)
check("clamp/negative", clamp_temperature(-3.0), TEMP_MIN)
check("clamp/none falls back to neutral", clamp_temperature(None), 1.0)
check("clamp/garbage falls back to neutral", clamp_temperature("x"), 1.0)
check("clamp/nan falls back to neutral", clamp_temperature(float("nan")), 1.0)
check("clamp/inf falls back to neutral", clamp_temperature(float("inf")), 1.0)
check("clamp/bools are not temperatures", clamp_temperature(True), 1.0)
check("clamp/False is not a sharpening zero", clamp_temperature(False), 1.0)
check("clamp/bounds are sane", TEMP_MIN <= 1.0 <= TEMP_MAX, True)
# 13 options is the bucket the reported skill-router landed in
check("clamp/13 options is the 11+ bucket", temp_bucket(QTYPES["choice"], 13), "choice:11+")


# --------------------------------------------------------------------- LRU bookkeeping
class _Stub:
    def __init__(self, name):
        self.name = name

    def system_one(self, state, questions):
        return {"model": self.name, "answers": {}, "usage": {}}


def stubbed_router(max_loaded):
    rr = Router(max_loaded=max_loaded)
    rr.load = lambda n, _r=rr: _load_stub(_r, n)
    return rr


def _load_stub(rr, name):
    key = normalise_name(name)
    if key in rr._agents:
        rr._touch(key)
        return rr._agents[key]
    rr._agents[key] = _Stub(key)
    rr._order.append(key)
    rr._evict()
    return rr._agents[key]


rr = stubbed_router(1)
rr.load("english"); rr.load("multilingual")
check("lru/cap 1 keeps newest", rr.loaded, ["multilingual"])
check("lru/cap 1 agents match order", sorted(rr._agents), ["multilingual"])

rr = stubbed_router(2)
rr.load("english"); rr.load("multilingual"); rr.load("typed-decisions")
check("lru/cap 2 evicts oldest", rr.loaded, ["multilingual", "typed-decisions"])

rr = stubbed_router(2)
rr.load("english"); rr.load("multilingual"); rr.load("english")   # touch english
rr.load("typed-decisions")
check("lru/touch protects", sorted(rr.loaded), ["english", "typed-decisions"])

rr.unload("english")
check("lru/unload one", "english" in rr.loaded, False)
rr.unload()
check("lru/unload all", rr.loaded, [])


# --------------------------------------------------------------------- default cap (#172)
# #172 measured a workload that alternates languages at 20-23 s per request on CPU (reloading a
# checkpoint every request) against 49-136 ms with both resident. Automatic routing only ever
# chooses between `english` and `multilingual`, so the default holds both, and a deployment that
# never alternates never builds the second.
def counting_router(cap=None):
    """Router whose loader records which checkpoints it had to build."""
    rr = Router() if cap is None else Router(max_loaded=cap)
    built = []

    def load(name, _rr=rr, _built=built):
        key = normalise_name(name)
        if key in _rr._agents:
            _rr._touch(key)
            return _rr._agents[key]
        _built.append(key)
        _rr._agents[key] = _Stub(key)
        _rr._order.append(key)
        _rr._evict()
        return _rr._agents[key]

    rr.load = load
    return rr, built


check("lru/default is two", Router().max_loaded, 2)
_en = {"body": "I was charged twice for invoice 4411, please refund."}
_ml = {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung"}
for cap, want_built in ((1, 20), (2, 2)):
    cr, built = counting_router(cap)
    for _ in range(10):                          # the reported alternating workload
        cr.predict(_en, Q_GENERIC)
        cr.predict(_ml, Q_GENERIC)
    check("lru/alternating traffic, cap=%d builds" % cap, len(built), want_built)
# and the default costs a single-language deployment nothing at all
cr, built = counting_router()
for _ in range(5):
    cr.predict(_en, Q_GENERIC)
check("lru/single-language traffic builds one checkpoint", built, ["english"])


# --------------------------------------------------------------------- bundle vs standalone
check("bundle/english is repo root", DEFAULT_MODELS["english"], (BUNDLE_REPO, None))
check("bundle/multilingual subfolder", DEFAULT_MODELS["multilingual"], (BUNDLE_REPO, "multilingual"))
check("bundle/typed subfolder", DEFAULT_MODELS["typed-decisions"], (BUNDLE_REPO, "typed-decisions"))
check("repo_str/root", _repo_str((BUNDLE_REPO, None)), "convaiinnovations/laya")
check("repo_str/sub", _repo_str((BUNDLE_REPO, "multilingual")), "convaiinnovations/laya/multilingual")
check("repo_str/plain string", _repo_str("some/repo"), "some/repo")

r_bundle = Router()
r_alone = Router(standalone_repos=True)
check("bundle/default router uses bundle",
      r_bundle.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"], "convaiinnovations/laya/multilingual")
check("standalone/opt-in uses own repo",
      r_alone.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"], "convaiinnovations/laya-multilingual")
check("standalone/english unchanged",
      r_alone.route({"m": "I was charged twice"}, Q_GENERIC)["repo"], "convaiinnovations/laya")
check("standalone map complete", sorted(STANDALONE_MODELS), sorted(DEFAULT_MODELS))
# a local-path override must still work (the Space and tests rely on it)
r_local = Router(models={"english": "/tmp/en", "multilingual": "/tmp/ml"})
check("override/local path kept", r_local.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"], "/tmp/ml")


# --------------------------------------------------------------------- preload
# Exercise the real preload/load/LRU paths; only checkpoint construction is stubbed.
with patch("laya.agent.Agent", side_effect=lambda repo, **kw: _Stub(repo)) as build:
    rp = Router(preload=True)
    check("preload/all three stay resident", sorted(rp.loaded),
          ["english", "multilingual", "typed-decisions"])
    check("preload/max_loaded raised", rp.max_loaded, 3)
    check("preload/builds each model once", build.call_count, 3)
    check("preload/returns router", rp.preload() is rp, True)
    check("preload/repeated call reuses models", build.call_count, 3)

    empty = Router()
    empty.preload([])
    check("preload/empty selection leaves models unloaded", empty.loaded, [])
    check("preload/empty selection builds nothing", build.call_count, 3)

    rp2 = Router()
    rp2.preload(["english", "multilingual"])
    check("preload/subset stays resident", sorted(rp2.loaded), ["english", "multilingual"])
    check("preload/subset capacity", rp2.max_loaded, 2)
    rp2.load("english")
    check("preload/touch does not evict", sorted(rp2.loaded), ["english", "multilingual"])
    check("preload/touch does not rebuild", build.call_count, 5)

    incremental = Router()
    incremental.preload(["english"])
    english = incremental.load("english")
    incremental.preload(["multilingual"])
    check("preload/incremental keeps both models", incremental.loaded, ["english", "multilingual"])
    check("preload/incremental capacity", incremental.max_loaded, 2)
    check("preload/incremental reuses original", incremental.load("english") is english, True)
    check("preload/incremental avoids rebuilds", build.call_count, 7)

    incremental.preload(["en", "english", "multi", "ml"])
    check("preload/aliases do not inflate capacity", incremental.max_loaded, 2)
    check("preload/aliases reuse models", build.call_count, 7)
    incremental.preload(["english", "typed-decisions"])
    check("preload/overlap preserves unrequested models", sorted(incremental.loaded),
          ["english", "multilingual", "typed-decisions"])
    check("preload/overlap capacity", incremental.max_loaded, 3)
    for name in DEFAULT_MODELS:
        incremental.predict("hello", Q_GENERIC, model=name)
    check("preload/predictions never rebuild", build.call_count, 8)

    attached = Router()
    original = _Stub("already-built")
    attached.attach("english", original)
    attached.preload(["multilingual"])
    check("preload/keeps attached model", attached.load("english") is original, True)
    check("preload/attached and new stay resident", sorted(attached.loaded), ["english", "multilingual"])
    check("preload/attached capacity", attached.max_loaded, 2)
    check("preload/attached model not rebuilt", build.call_count, 9)

    roomy = Router(max_loaded=5)
    roomy.preload(["en", "english", "multi"])
    check("preload/larger capacity is preserved", roomy.max_loaded, 5)
    check("preload/duplicates build once", build.call_count, 11)


# --------------------------------------------------------------------- attach
ra = stubbed_router(1)
sentinel = _Stub("already-built")
ra.attach("english", sentinel)
check("attach/registers under the name", ra._agents["english"], sentinel)
check("attach/counts as resident", "english" in ra.loaded, True)
# `max_loaded` starts at `max(1, max_loaded)`, so one attach to a cap-1 router cannot
# move it and `ra.max_loaded >= 1` held before `attach` was ever called. The cap only
# rises on the attach that would not otherwise fit.
check("attach/first attach leaves the cap alone", ra.max_loaded, 1)
ra.attach("multilingual", _Stub("second"))
check("attach/raises max_loaded to hold the extra one", ra.max_loaded, 2)
ra.unload("multilingual")
# attaching then loading another must not evict the attached one
_load_stub(ra, "multilingual")
check("attach/survives a later load", sorted(ra.loaded), ["english", "multilingual"])
check("attach/still the same object", ra._agents["english"] is sentinel, True)
check("attach/accepts aliases", stubbed_router(1).attach("en", _Stub("x")) is not None, True)


# --------------------------------------------------------------------- thread safety (issue #95)

def _concurrent_load_dedup():
    """Concurrent load() of the same checkpoint must build one Agent, shared by all callers."""
    import laya.agent as _agent_mod
    constructions = []
    cl = threading.Lock()

    class _SlowAgent:
        def __init__(self, *args, **kwargs):
            _time.sleep(0.05)  # widen the check-then-build window
            with cl:
                constructions.append(1)

        def system_one(self, state, questions):
            return {"model": "fake", "answers": {}, "usage": {}}

    old = _agent_mod.Agent
    _agent_mod.Agent = _SlowAgent
    try:
        r = Router()
        got = []

        def _worker():
            got.append(r.load("english"))

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return len({id(x) for x in got}), len(constructions), len(r._order), sorted(r._agents)
    finally:
        _agent_mod.Agent = old

unique, built, order_len, agents = _concurrent_load_dedup()
check("threads/8 concurrent loads share one Agent", unique, 1)
check("threads/Agent constructed exactly once", built, 1)
check("threads/LRU views stay consistent", (order_len == 1 and agents == ["english"]), True)


def _concurrent_hotpath():
    """Concurrent hot-path loads of an already-cached model must keep _order/_agents consistent."""
    import laya.agent as _agent_mod

    class _Agent:
        def __init__(self, *args, **kwargs):
            pass

        def system_one(self, state, questions):
            return {"model": "fake", "answers": {}, "usage": {}}

    old = _agent_mod.Agent
    _agent_mod.Agent = _Agent
    try:
        r = Router(max_loaded=3)
        r.load("english")  # warm the cache

        def _worker():
            r.load("english")

        threads = [threading.Thread(target=_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return len(r._order), len(r._agents), r._order
    finally:
        _agent_mod.Agent = old

order_len, agents_len, order = _concurrent_hotpath()
check("threads/hot-path loads keep one entry", order_len, 1)
check("threads/hot-path loads keep agents consistent", agents_len, 1)
check("threads/hot-path order intact", order, ["english"])


def _cold_build_does_not_hold_lifecycle_lock():
    """A cold build must not stall `loaded` or a resident checkpoint for its whole duration.

    The build runs a download plus construction, seconds to minutes. Held under `_lock`, it made
    `GET /health` (which reads `loaded`) and every request for an already-resident checkpoint wait
    for it, so a liveness probe timed out during a lazy load.
    """
    import laya.agent as _agent_mod
    building, release = threading.Event(), threading.Event()
    constructions = []

    class _BlockingAgent:
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            building.set()
            release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _BlockingAgent
    try:
        r = Router(max_loaded=3)
        resident = _Stub("english")
        r.attach("english", resident)
        loaders = [threading.Thread(target=r.load, args=("multilingual",)) for _ in range(2)]
        for t in loaders:
            t.start()
        building.wait(5)
        timings = {}
        for label, call in (("loaded", lambda: r.loaded),
                            ("resident load", lambda: r.load("english"))):
            done = threading.Event()
            threading.Thread(target=lambda c=call, d=done: (c(), d.set()), daemon=True).start()
            timings[label] = done.wait(1.0)
        release.set()
        for t in loaders:
            t.join(5)
        loaded_after, built = sorted(r.loaded), len(constructions)
        # An unload issued while a build is in flight waits for it, so the build cannot land after it.
        building.clear()
        release.clear()
        threading.Thread(target=r.load, args=("typed-decisions",), daemon=True).start()
        building.wait(5)
        unloader = threading.Thread(target=r.unload, args=("typed-decisions",))
        unloader.start()
        _time.sleep(0.1)
        release.set()
        unloader.join(5)
        timings["unload waits for the build"] = "typed-decisions" not in r.loaded
        return timings, built, loaded_after
    finally:
        release.set()
        _agent_mod.Agent = old

timings, built, resident_after = _cold_build_does_not_hold_lifecycle_lock()
check("threads/loaded answers during a cold build", timings["loaded"], True)
check("threads/resident checkpoint answers during a cold build", timings["resident load"], True)
check("threads/concurrent cold loads still build once", built, 1)
check("threads/cold build lands in the LRU", resident_after, ["english", "multilingual"])
check("threads/unload during a cold build frees it", timings["unload waits for the build"], True)


def _on_load_reenters_router(target, hooks_concurrent):
    """An `on_load` hook may call `router.load()` again without deadlocking.

    `_build_lock` is not re-entrant, so this holds only while `load()` dispatches `on_load`
    after releasing it. Returns whether the outer load finished, the agent the hook got back,
    the outer agent, and what ended up loaded.
    """
    import laya.agent as _agent_mod

    class _FastAgent:
        def __init__(self, *args, **kwargs):
            pass

    got = []

    class _ReentrantHook:
        def on_load(self, ctx):
            if ctx.model == "multilingual" and not got:
                got.append(ctx.router.load(target))

    old = _agent_mod.Agent
    _agent_mod.Agent = _FastAgent
    try:
        r = Router(max_loaded=3, hooks=[_ReentrantHook()], hooks_concurrent=hooks_concurrent)
        outer = []
        t = threading.Thread(target=lambda: outer.append(r.load("multilingual")), daemon=True)
        t.start()
        t.join(5)
        return not t.is_alive(), got[0] if got else None, outer[0] if outer else None, sorted(r.loaded)
    finally:
        _agent_mod.Agent = old

for concurrent in (True, False):
    tag = "concurrent" if concurrent else "serialised"
    finished, inner, outer, loaded = _on_load_reenters_router("multilingual", concurrent)
    check(f"threads/on_load reloading the same checkpoint does not deadlock ({tag})", finished, True)
    check(f"threads/on_load reload returns the resident agent ({tag})",
          inner is not None and inner is outer, True)
    finished, inner, outer, loaded = _on_load_reenters_router("typed-decisions", concurrent)
    check(f"threads/on_load loading another checkpoint does not deadlock ({tag})", finished, True)
    check(f"threads/on_load can load another checkpoint ({tag})", loaded, ["multilingual", "typed-decisions"])


def _test_per_checkpoint_unload_granularity():
    """unload("english") must not wait for an in-flight build of multilingual.

    unload("multilingual") must wait for its own in-flight build, and resident access
    (r.loaded and r.load("english")) must remain non-blocking while multilingual builds.
    """
    import laya.agent as _agent_mod
    multi_building = threading.Event()
    multi_release = threading.Event()

    class _ControllableAgent:
        def __init__(self, repo, *args, **kwargs):
            if "multilingual" in repo or kwargs.get("subfolder") == "multilingual":
                multi_building.set()
                multi_release.wait(10)

        def system_one(self, state, questions):
            return {"model": "fake", "answers": {}, "usage": {}}

    old = _agent_mod.Agent
    _agent_mod.Agent = _ControllableAgent
    try:
        r = Router(max_loaded=3)
        resident = _Stub("english")
        r.attach("english", resident)

        multi_loader = threading.Thread(target=r.load, args=("multilingual",), daemon=True)
        multi_loader.start()
        multi_building.wait(5)

        # 1. unload("english") must complete immediately without waiting for multilingual build
        english_unload_done = threading.Event()
        threading.Thread(target=lambda: (r.unload("english"), english_unload_done.set()), daemon=True).start()
        unloaded_english_fast = english_unload_done.wait(1.0)
        english_resident = "english" in r.loaded

        # 7. loaded and resident access remain non-blocking
        r.attach("english", resident)
        loaded_done = threading.Event()
        threading.Thread(target=lambda: (r.loaded, loaded_done.set()), daemon=True).start()
        loaded_fast = loaded_done.wait(1.0)

        resident_load_done = threading.Event()
        threading.Thread(target=lambda: (r.load("english"), resident_load_done.set()), daemon=True).start()
        resident_load_fast = resident_load_done.wait(1.0)

        # 2. unload("multilingual") waits for multilingual build
        multi_unload_done = threading.Event()
        multi_unloader = threading.Thread(
            target=lambda: (r.unload("multilingual"), multi_unload_done.set()), daemon=True
        )
        multi_unloader.start()
        unloaded_multi_early = multi_unload_done.wait(0.1)

        multi_release.set()
        multi_loader.join(5)
        multi_unloader.join(5)

        unloaded_multi_finished = multi_unload_done.wait(1.0)
        multi_resident_after = "multilingual" in r.loaded

        return (
            unloaded_english_fast,
            english_resident,
            loaded_fast,
            resident_load_fast,
            unloaded_multi_early,
            unloaded_multi_finished,
            multi_resident_after,
        )
    finally:
        multi_release.set()
        _agent_mod.Agent = old

(
    unloaded_en_fast,
    en_resident,
    loaded_fast,
    resident_load_fast,
    unloaded_multi_early,
    unloaded_multi_finished,
    multi_resident_after,
) = _test_per_checkpoint_unload_granularity()
check("threads/unload english does not wait for multilingual build", unloaded_en_fast, True)
check("threads/english is evicted after unload", en_resident, False)
check("threads/loaded property is non-blocking during build", loaded_fast, True)
check("threads/resident load is non-blocking during build", resident_load_fast, True)
check("threads/unload multilingual waits for in-flight build", unloaded_multi_early, False)
check("threads/unload multilingual completes after build", unloaded_multi_finished, True)
check("threads/multilingual not resident after unload", multi_resident_after, False)


def _test_concurrent_load_same_checkpoint_dedup():
    """Concurrent loads for the same checkpoint must construct exactly one Agent."""
    import laya.agent as _agent_mod
    build_started = threading.Event()
    build_release = threading.Event()
    constructions = []

    class _SlowAgent:
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            build_started.set()
            build_release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _SlowAgent
    try:
        r = Router(max_loaded=3)
        got = []
        threads = [
            threading.Thread(target=lambda: got.append(r.load("multilingual")), daemon=True)
            for _ in range(5)
        ]
        for t in threads:
            t.start()

        build_started.wait(5)
        build_release.set()
        for t in threads:
            t.join(5)

        return len(constructions), len({id(a) for a in got}), len(got)
    finally:
        build_release.set()
        _agent_mod.Agent = old

built_count, unique_agents, total_callers = _test_concurrent_load_same_checkpoint_dedup()
check("threads/concurrent loads construct exactly one Agent", built_count, 1)
check("threads/concurrent callers receive identical Agent instance", unique_agents, 1)
check("threads/all concurrent callers complete", total_callers, 5)


def _test_attach_during_build_wins():
    """attach() called while a build is in flight must win without being overwritten."""
    import laya.agent as _agent_mod
    build_started = threading.Event()
    build_release = threading.Event()

    class _ControllableAgent:
        def __init__(self, *args, **kwargs):
            build_started.set()
            build_release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _ControllableAgent
    try:
        r = Router(max_loaded=3)
        attached_agent = _Stub("english")
        loader_result = []

        loader = threading.Thread(target=lambda: loader_result.append(r.load("english")), daemon=True)
        loader.start()

        build_started.wait(5)
        r.attach("english", attached_agent)

        build_release.set()
        loader.join(5)

        subsequent = r.load("english")
        return (
            loader_result[0] is attached_agent,
            subsequent is attached_agent,
            r._agents.get("english") is attached_agent,
        )
    finally:
        build_release.set()
        _agent_mod.Agent = old

loader_won, subsequent_won, resident_won = _test_attach_during_build_wins()
check("threads/attach during build wins for in-flight loader", loader_won, True)
check("threads/attach during build wins for subsequent load", subsequent_won, True)
check("threads/attach during build remains resident", resident_won, True)


def _test_failed_build_cleans_inflight_and_allows_retry():
    """Failed builds clean up in-flight state, propagate exceptions, and allow retry."""
    import laya.agent as _agent_mod
    fail_first = [True]

    class _FailingAgent:
        def __init__(self, *args, **kwargs):
            if fail_first[0]:
                raise ValueError("corrupted weights download")
            self.model = "ok"

    old = _agent_mod.Agent
    _agent_mod.Agent = _FailingAgent
    try:
        r = Router(max_loaded=3)
        errors = []

        def _loader(err_list):
            try:
                r.load("english")
            except ValueError as e:
                err_list.append(str(e))

        t1 = threading.Thread(target=_loader, args=(errors,), daemon=True)
        t2 = threading.Thread(target=_loader, args=(errors,), daemon=True)
        t1.start()
        t2.start()
        t1.join(5)
        t2.join(5)

        inflight_cleared = "english" not in r._loading
        not_resident = "english" not in r._agents

        fail_first[0] = False
        retried_agent = r.load("english")
        retried_resident = "english" in r.loaded

        return (
            len(errors),
            errors[0] if errors else None,
            inflight_cleared,
            not_resident,
            retried_agent.model,
            retried_resident,
        )
    finally:
        _agent_mod.Agent = old

err_count, first_err, inflight_cleared, not_res, retried_model, retried_res = (
    _test_failed_build_cleans_inflight_and_allows_retry()
)
check("threads/failed build raises to concurrent loaders", err_count, 2)
check("threads/failed build error message preserved", first_err, "corrupted weights download")
check("threads/failed build cleans _loading registry", inflight_cleared, True)
check("threads/failed build does not leave model resident", not_res, True)
check("threads/subsequent load retries and succeeds", retried_model, "ok")
check("threads/retried load lands in resident set", retried_res, True)


def _test_waiting_callers_no_deadlock():
    """Concurrent mix of loads and unloads on same and different models must not deadlock."""
    import laya.agent as _agent_mod

    class _FastAgent:
        def __init__(self, *args, **kwargs):
            _time.sleep(0.005)

    old = _agent_mod.Agent
    _agent_mod.Agent = _FastAgent
    try:
        r = Router(max_loaded=3)
        models = ["english", "multilingual", "typed-decisions"]
        threads = []
        for _ in range(6):
            for m in models:
                threads.append(threading.Thread(target=r.load, args=(m,)))
                threads.append(threading.Thread(target=r.unload, args=(m,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        all_finished = not any(t.is_alive() for t in threads)
        loading_clean = len(r._loading) == 0
        return all_finished, loading_clean
    finally:
        _agent_mod.Agent = old

no_hang, loading_empty = _test_waiting_callers_no_deadlock()
check("threads/concurrent load and unload callers do not deadlock", no_hang, True)
check("threads/all in-flight markers cleared after completion", loading_empty, True)


def _test_unload_reload_sequence_no_resurrection():
    """An older in-flight build cannot resurrect the model after an unload/reload sequence."""
    import laya.agent as _agent_mod
    first_build_started = threading.Event()
    first_build_release = threading.Event()
    agent_instances = []

    class _SequencedAgent:
        def __init__(self, *args, **kwargs):
            agent_instances.append(self)
            if len(agent_instances) == 1:
                first_build_started.set()
                first_build_release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _SequencedAgent
    try:
        r = Router(max_loaded=3)
        t_load = threading.Thread(target=r.load, args=("english",), daemon=True)
        t_load.start()
        first_build_started.wait(5)

        unload_done = threading.Event()
        t_unload = threading.Thread(target=lambda: (r.unload("english"), unload_done.set()), daemon=True)
        t_unload.start()

        first_build_release.set()
        t_load.join(5)
        t_unload.join(5)

        unloaded_ok = unload_done.is_set()
        not_resident_after_unload = "english" not in r.loaded

        c_agent = r.load("english")
        is_new_agent = c_agent is agent_instances[1] and c_agent is not agent_instances[0]
        resident_final = "english" in r.loaded

        return unloaded_ok, not_resident_after_unload, is_new_agent, resident_final
    finally:
        first_build_release.set()
        _agent_mod.Agent = old

unloaded_ok, not_res_after, is_new_agent, res_final = _test_unload_reload_sequence_no_resurrection()
check("threads/unload in-flight build finishes successfully", unloaded_ok, True)
check("threads/model not resident after unload completes", not_res_after, True)
check("threads/reload builds new instance without resurrection", is_new_agent, True)
check("threads/reload lands in resident set", res_final, True)



# --------------------------------------------------------------------- unlisted scripts
# `detect_script` counts an alphabetic character only when one of `_SCRIPT_RANGES` claims
# it. Those ranges cover the scripts the checkpoints were measured on, and most of Unicode
# is outside them -- 68% of alphabetic codepoints, including the CJK extensions, the kana
# supplements, bopomofo, halfwidth katakana and dozens of smaller scripts. An unclaimed
# character used to be counted nowhere, so text written only in such a script produced a
# total of 0, was reported as "unknown", and `analyse` treats "unknown" as English. It was
# therefore routed to the English checkpoint, which has no tokens for it at all, with the
# reason "no letters detected in state" -- for text that plainly has letters.
#
# Every case below reported unknown / is_english=True / model "english" before this change.
for label, text in (
    ("halfwidth katakana", "ｱﾘｶﾞﾄｳ"),
    ("bopomofo", "ㄆㄇㄈㄉ"),
    ("kana supplement", "\U0001B000\U0001B001"),
    ("CJK Ext-B", "\U00020000\U00020001"),
    ("hangul jamo ext-A", "\ua960\ua961"),
    ("Cherokee", "ᏣᎳᎩ"),
    ("Mongolian", "ᠮᠣᠩᠭᠣᠯ"),
    ("Syriac", "ܫܠܡܐ"),
    ("Thaana", "ދިވެހި"),
    ("Tifinagh", "ⵜⴰⵎⴰⵣⵉⵖⵜ"),
    ("Yi", "ꆈꌠ"),
):
    check("unlisted/" + label + " is not called English", analyse(text)["is_english"], False)
    check("unlisted/" + label + " routes to multilingual",
          Router().route(text)["model"], "multilingual")

# Fullwidth Latin is Latin, not an unlisted script.
check("unlisted/fullwidth latin is latin", detect_script("ＨＥＬＬＯ"), "latin")

# A state with no letters at all takes `default`, whatever that is, which is the behaviour that
# must not change. Both arms are checked so this cannot pass by agreeing with one constant.
for label, text in (("empty", ""), ("digits only", "12345 67890"), ("emoji only", "😀😀😀")):
    check("unlisted/letterless " + label + " is still unknown",
          analyse(text)["script"], "unknown")
    check("unlisted/letterless " + label + " keeps the stock default",
          Router().route(text)["model"], "multilingual")
    check("unlisted/letterless " + label + " keeps an english default",
          Router(default="english").route(text)["model"], "english")

# The scripts the table does name must be untouched.
for label, text, script in (
    ("english", "please cancel my subscription", "latin"),
    ("german", "Mein Konto wurde zweimal belastet, bitte erstatten Sie den Betrag", "latin"),
    ("hindi", "यह एक हिंदी वाक्य है", "devanagari"),
    ("chinese", "请取消我的订阅", "han"),
    ("japanese", "ありがとう", "kana"),
    ("korean", "감사합니다", "hangul"),
    ("russian", "Мой аккаунт был списан дважды", "cyrillic"),
    ("arabic", "تم خصم حسابي مرتين", "arabic"),
    ("greek", "Η χρέωση έγινε δύο φορές", "greek"),
    ("armenian", "Իմ հաշիվը գանձվել է երկու անգամ", "armenian"),
):
    check("unlisted/regression " + label + " script", detect_script(text), script)

# ------------------------------------------------- lang codes that name no language (#359)
# `C`, `POSIX` and `C.UTF-8` are valid `$LANG` values that identify nothing, and `C.UTF-8` is
# the default in the official Python image -- which is where `laya-serve` runs. Passing one to
# `Router(lang=...)` used to resolve to "not English" and pin every request to the multilingual
# checkpoint, before detection ever ran. The blank-string case below already abstained; these
# codes are the same kind of non-answer, so they abstain too. The ISO 639-2 special codes say
# the same thing in the standard's vocabulary.
_ENGLISH_STATE = "Please refund the duplicate charge on invoice 4411"
for code in ("C", "POSIX", "C.UTF-8", "c.utf8", "c", "posix",
             "und", "zxx", "mul", "UND", "Zxx", " und "):
    check("lang-code/%r abstains at the code level" % code, _english_from_code(code), None)
    decision = Router().route(_ENGLISH_STATE, lang=code)
    check("lang-code/%r lets detection name the checkpoint" % code, decision.model, "english")
    check("lang-code/%r says the hint was not used" % code,
          "explicit" in str(decision.reason), False)

# An empty hint is the case this mirrors, so it must still behave the same way.
check("lang-code/empty string still abstains", _english_from_code(""), None)
check("lang-code/None still abstains", _english_from_code(None), None)
check("lang-code/whitespace still abstains", _english_from_code("   "), None)

# The change must not touch codes that do name a language: English still routes now, and a
# non-English code still forces the multilingual checkpoint rather than being second-guessed.
for code in ("en", "eng", "english", "EN", "en-US", "en_US.UTF-8"):
    check("lang-code/%r is still decisive English" % code, _english_from_code(code), True)
    check("lang-code/%r routes without detection" % code,
          Router().route(_ENGLISH_STATE, lang=code).reason.count("explicit"), 1)
for code in ("de", "fr", "zh", "ja", "pt-BR", "de_DE.UTF-8"):
    check("lang-code/%r is still decisive non-English" % code, _english_from_code(code), False)
    check("lang-code/%r routes to the multilingual checkpoint" % code,
          Router().route(_ENGLISH_STATE, lang=code).model, "multilingual")

# A subtag after an agnostic primary is not itself agnostic: `C` names nothing, but the primary
# subtag is what is compared, so a hypothetical `C-something` also abstains and is not treated
# as a language by accident.
check("lang-code/agnostic primary wins over its subtag", _english_from_code("C.UTF-8"), None)
check("lang-code/posix with a modifier abstains", _english_from_code("POSIX-1"), None)


# --------------------------------------------------------------------- registered checkpoints
# A Router serves the checkpoints a caller registers beside the built-in three: named in `model=`
# or `task=` like them, and loaded, evicted and unloaded like them.
from laya.router import canonical_name  # noqa: E402

check("registry/canonical lowercases and aliases", canonical_name(" ML "), "multilingual")
check("registry/canonical leaves an unknown name as typed", canonical_name("Papers"), "papers")

_SECTION_Q = {"section": {"type": "choice", "instructions": "Which section?", "criteria": {"a": "A", "b": "B"}}}
_CLAIM = "Transparency improved operator performance in 11 of 17 studies."
r = Router(models={"papers": "/tmp/laya-papers", "tone": ("acme/laya-tone", None)})
check("registry/registered names resolve", r.resolve("papers"), "papers")
check("registry/registered names resolve case-insensitively", r.resolve(" PAPERS "), "papers")
check("registry/built-ins still resolve", r.resolve("ml"), "multilingual")
check("registry/models holds built-ins plus registered", sorted(r.models),
      ["english", "multilingual", "papers", "tone", "typed-decisions"])
try:
    r.resolve("nope")
    check("registry/unknown name raises", False, True)
except ValueError as e:
    check("registry/unknown name lists the registered ones too", "'papers'" in str(e), True)
check("registry/normalise_name still knows only the built-ins", resolve_model_spec("papers"), None)

d = r.route(_CLAIM, _SECTION_Q, model="papers")
check("route/model= names a registered checkpoint", d.model, "papers")
check("route/repo is the registered source", d["repo"], "/tmp/laya-papers")
check("route/model= accepts a registered (repo, subfolder) pair", r.route(_CLAIM, _SECTION_Q, model="tone")["repo"],
      "acme/laya-tone")
check("route/task= names a registered checkpoint", r.route(_CLAIM, _SECTION_Q, task="tone").model, "tone")
check("route/a registered checkpoint is never chosen automatically", r.route(_CLAIM, _SECTION_Q).model, "english")
check("route/route_batch keeps the mix in order",
      [x.model for x in r.route_batch([{"state": _CLAIM, "questions": _SECTION_Q, "model": "papers"},
                                       {"state": _CLAIM, "questions": _SECTION_Q},
                                       {"state": _CLAIM, "questions": _SECTION_Q, "model": "tone"}])],
      ["papers", "english", "tone"])
check("registered/reports source and description", Router(models={"papers": "/tmp/laya-papers"}).registered,
      {"papers": {"source": "/tmp/laya-papers", "description": None}})
check("registered/built-ins are not listed", Router().registered, {})

# The built-in typed-decisions workflows are unchanged: opt-in, and still routed to the built-in.
_CS = {q: _SECTION_Q["section"] for q in ("action", "category", "churn_risk", "needs_human", "urgency")}
check("workflow/still needs auto_task_detection", r.route(_CLAIM, _CS).model, "english")
check("workflow/with auto_task_detection",
      Router(auto_task_detection=True).route(_CLAIM, _CS)["workflow"], "customer_service")

# a checkpoint registered after construction is not read as an artifact map of a nested env.
_old_env = os.environ.get("LAYA_SHA256_DIGESTS")
os.environ["LAYA_SHA256_DIGESTS"] = '{"english": {"model.safetensors": "%s"}}' % ("c" * 64)
try:
    _r_late = Router()
    _r_late.register("mine", "/tmp/laya-mine")
    check("register/nested LAYA_SHA256_DIGESTS gives a later checkpoint the empty placeholder",
          _r_late.sha256_digests["mine"], {})
finally:
    if _old_env is None:
        os.environ.pop("LAYA_SHA256_DIGESTS", None)
    else:
        os.environ["LAYA_SHA256_DIGESTS"] = _old_env

# a source starting with ~ is a local path.
check("register/~ in a source is expanded", Router(models={"mine": "~/laya-mine"}).models["mine"],
      os.path.expanduser("~/laya-mine"))

# `auto` is the routing word everywhere, so it cannot name a checkpoint.
try:
    Router().register("auto", "/tmp/laya-auto")
    check("register/auto is refused", True, False)
except ValueError as _e:
    check("register/auto is refused", "auto" in str(_e), True)

# an attach()ed agent with no source is not part of a whole-router preload.
with patch("laya.agent.Agent", side_effect=lambda repo, **kw: _Stub(repo)) as _build_att:
    _r_att = Router()
    _r_att.attach("mine", _Stub("mine"))
    _r_att.unload("mine")
    _r_att.preload()
    check("preload/skips a name with no source", sorted(_r_att.loaded),
          ["english", "multilingual", "typed-decisions"])

# register(): the same thing after construction.
r2 = Router()
check("register/returns the canonical name", r2.register("Tone", "acme/laya-tone", description="tone"), "tone")
check("register/description is reported", r2.registered["tone"], {"source": "acme/laya-tone", "description": "tone"})
check("register/a second call replaces the source", (r2.register("tone", "/tmp/tone-v2"), r2.models["tone"])[1], "/tmp/tone-v2")
check("register/replacing the source keeps the description", r2.registered["tone"]["description"], "tone")
check("register/a built-in name re-points that checkpoint",
      Router(models={"english": "/tmp/my-english"}).models["english"], "/tmp/my-english")
check("register/an alias re-points its built-in", Router(models={"en": "/tmp/mine"}).models["english"], "/tmp/mine")
for bad, why in (("Bad/Name", "slash"), ("", "empty"), ("-dash", "leading dash"), ("a b", "space")):
    try:
        Router().register(bad, "/tmp/x")
        check("register/refuses %s" % why, False, True)
    except ValueError:
        check("register/refuses %s" % why, True, True)
try:
    Router(models={"papers": 42})
    check("register/a source must be a path, repo or pair", False, True)
except TypeError:
    check("register/a source must be a path, repo or pair", True, True)

# Every name-taking option accepts a registered name, the way it accepts a built-in one.
r3 = Router(models={"papers": "/tmp/laya-papers"}, revisions={"papers": "abc123"}, default="papers",
            sha256_digests={"papers": {"model.safetensors": "a" * 64}})
check("register/default may be a registered name", r3.default, "papers")
check("register/revisions keyed by a registered name", r3.revisions, {"papers": "abc123"})
check("register/sha256_digests keyed by a registered name", r3.sha256_digests["papers"], {"model.safetensors": "a" * 64})
check("register/undecided text falls back to the registered default",
      r3.route("Quero cancelar", {"q": _SECTION_Q["section"]}).model, "papers")
_old_env = os.environ.get("LAYA_SHA256_DIGESTS")
os.environ["LAYA_SHA256_DIGESTS"] = '{"papers": {"model.safetensors": "%s"}}' % ("b" * 64)
try:
    check("register/LAYA_SHA256_DIGESTS may name a registered checkpoint",
          Router(models={"papers": "/tmp/laya-papers"}).sha256_digests["papers"], {"model.safetensors": "b" * 64})
finally:
    if _old_env is None:
        os.environ.pop("LAYA_SHA256_DIGESTS", None)
    else:
        os.environ["LAYA_SHA256_DIGESTS"] = _old_env

# attach() under a new name registers it with no source: resident now, not reloadable later.
r4 = Router()
r4.attach("adhoc", _Stub("adhoc"))
check("attach/new name becomes resident", "adhoc" in r4.loaded, True)
check("attach/new name is routable", r4.route(_CLAIM, _SECTION_Q, model="adhoc").model, "adhoc")
check("attach/new name has no source", r4.models["adhoc"], None)
r4.unload("adhoc")
try:
    r4.load("adhoc")
    check("attach/reload without a source is refused", False, True)
except ValueError as e:
    check("attach/reload without a source is refused", "no source" in str(e), True)

# load() hands a registered local path or repo to Agent as a source, not as a registry name.
import laya.agent as _agent_for_registry
_built = []


class _RecordingAgent:
    def __init__(self, repo, **kwargs):
        _built.append((repo, kwargs.get("subfolder")))


_prev_agent = _agent_for_registry.Agent
_agent_for_registry.Agent = _RecordingAgent
try:
    r5 = Router(models={"papers": "/tmp/laya-papers", "tone": ("acme/laya-tone", "v2")}, max_loaded=3)
    r5.load("papers")
    r5.load("tone")
    check("load/a registered path reaches Agent unchanged", _built[0], ("/tmp/laya-papers", None))
    check("load/a registered (repo, subfolder) pair reaches Agent", _built[1], ("acme/laya-tone", "v2"))
    check("load/registered checkpoints are resident like built-ins", r5.loaded, ["papers", "tone"])
    r5.preload(["papers", "english"])
    check("load/preload accepts registered names", "english" in r5.loaded and "papers" in r5.loaded, True)
    r5.unload("papers")
    check("load/unload accepts registered names", "papers" in r5.loaded, False)
finally:
    _agent_for_registry.Agent = _prev_agent


# A refused register() leaves the router as it was.
_at = Router(models={"a": "/a", "b": "/b"})
try:
    _at.register("Bad/Name", "/x", description="x")
    _at_raised = False
except ValueError:
    _at_raised = True
check("register/refused call raises", _at_raised, True)
check("register/refused call leaves no checkpoint", ("bad/name" in _at.models, "bad/name" in _at.descriptions), (False, False))
_at2 = Router(models={"a": "/a"})
try:
    _at2.register("a", 42)
except TypeError:
    pass
check("register/refused call keeps the old source", _at2.registered["a"]["source"], "/a")

# An explicit `sha256_digests[name] = None` is a present entry that opts that checkpoint out of
# digest checks; a refused register() must restore it, not drop it (dropping it would hand the
# checkpoint back to the environment's digest map).
_at3 = Router(models={"a": "/a"}, sha256_digests={"a": None})
try:
    _at3.register("a", 42)
    check("register/refused call raises on a bad source", False, True)
except TypeError:
    check("register/refused call raises on a bad source", True, True)
check("register/refused call keeps an opted-out digest entry", ("a" in _at3.sha256_digests, _at3.sha256_digests.get("a", "gone")),
      (True, None))
check("register/refused call keeps the old source after a digest opt-out", _at3.models["a"], "/a")

# unregister() removes a registered name everywhere.
_rp = Router(models={"a": "/a", "b": "/b"})
_rp.descriptions["a"] = "A"
_rp.sha256_digests["a"] = None
_rp.revisions["a"] = "main"
_rp.attach("a", object())
_rp.unregister("a")
check("unregister/removes the name everywhere",
      ("a" in _rp.models, "a" in _rp.descriptions, "a" in _rp.sha256_digests,
       "a" in _rp.revisions, "a" in _rp.loaded, "a" in _rp.registered), (False,) * 6)
check("unregister/others are untouched", sorted(_rp.registered), ["b"])
for _bad in ("english", "ml", "never-registered"):
    try:
        _rp.unregister(_bad)
        check("unregister/%s refused" % _bad, False, True)
    except ValueError:
        check("unregister/%s refused" % _bad, True, True)
_rp.register("a", "/a")
check("unregister/the name can be registered again", "a" in _rp.registered, True)

# Routing while another thread re-registers and unregisters never fails or answers from a torn state.
_rc = Router(models={"a": "/a"})
_bad = []


def _churn():
    for _ in range(40):
        _rc.register("a", "/a")
        _rc.register("c", "/c")
        _rc.unregister("c")


def _route():
    for _ in range(200):
        try:
            _got = _rc.route(_CLAIM, model="a").model
            if _got != "a":
                _bad.append(_got)
            _rc.registered
            try:
                _rc.route(_CLAIM, model="c")
            except ValueError:      # unregistered at that instant
                pass
        except Exception as exc:  # noqa: BLE001
            _bad.append(exc)


_th = [threading.Thread(target=_churn)] + [threading.Thread(target=_route) for _ in range(3)]
for _t in _th:
    _t.start()
for _t in _th[1:]:
    _t.join()
_th[0].join()
check("register+unregister while routing stay consistent", _bad[:3], [])


def _swallow(fn, *args):
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        pass


def _test_unregister_during_build_leaves_no_zombie():
    """unregister waits for a build already in flight and leaves nothing under the dropped name."""
    import laya.agent as _agent_mod
    started, release = threading.Event(), threading.Event()

    class _Slow:
        def __init__(self, *args, **kwargs):
            started.set()
            release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _Slow
    try:
        r = Router(models={"tone": "/tone"})
        outcome = []

        def _load():
            try:
                outcome.append(r.load("tone"))
            except ValueError as e:
                outcome.append(e)

        loader = threading.Thread(target=_load, daemon=True)
        loader.start()
        started.wait(2)
        _timer = threading.Timer(0.3, release.set)
        _timer.start()
        t0 = _time.perf_counter()
        r.unregister("tone")
        waited = _time.perf_counter() - t0
        loader.join(2)
        _timer.cancel()
        return (waited >= 0.2, loader.is_alive(), len(outcome) == 1,
                "tone" in r.loaded, "tone" in r.models, "tone" in r._agents, "tone" in r._order,
                "tone" in r._loading)
    finally:
        release.set()
        _agent_mod.Agent = old


check("unregister/a build in flight is waited for and leaves nothing behind",
      _test_unregister_during_build_leaves_no_zombie(),
      (True, False, True, False, False, False, False, False))

# The default checkpoint cannot be unregistered, and the router keeps routing.
_rd = Router(models={"papers": "/x"}, default="papers")
try:
    _rd.unregister("papers")
    check("unregister/the default is refused", False, True)
except ValueError as _e:
    check("unregister/the default is refused", "default" in str(_e), True)
check("unregister/a refused default still routes", _rd.route("12345").model, "papers")


# Re-registering a name with a new source unloads the resident Agent; the same source does not.
class _EvictLog:
    def __init__(self):
        self.evicts = []

    def on_evict(self, ctx):
        self.evicts.append(ctx.model)


_log = _EvictLog()
_rr = Router(hooks=[_log])
_rr.attach("papers", _Stub("papers"))
_rr.register("papers", "/y")
check("register/a new source unloads the resident agent", "papers" in _rr.loaded, False)
check("register/a new source fires on_evict", _log.evicts, ["papers"])
_rr.attach("papers", _Stub("papers"))
_rr.register("papers", "/y")
check("register/the same source keeps the resident agent", "papers" in _rr.loaded, True)
check("register/the same source fires no on_evict", _log.evicts, ["papers"])
_rr.register("fresh", "/z")
check("register/a new name unloads nothing", _log.evicts, ["papers"])


# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all routing tests passed")
sys.exit(1 if FAIL else 0)
