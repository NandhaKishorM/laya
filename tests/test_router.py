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
    _repo_str,
    Router,
    match_typed_decisions_workflow,
    normalise_name,
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
         "diacritic_rate", "non_latin_fraction"}
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
_r_dotted = Router()
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
    ("empty state", {}, Q_GENERIC, {}, "english"),
    ("none state", None, Q_GENERIC, {}, "english"),
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
check("route/short english still english", _r_lat.route("refund me").model, "english")

# Undecided Latin text follows `default`, as a state with no letters already did. Short messages made
# only of content words carry nothing that names their language, and hard-coding English for them
# sent every short Portuguese message to the checkpoint that is 0.97 confident at 0.47 accuracy on
# `pt`, whatever the router was configured with.
_r_ml = Router(default="multilingual")
for text in ["Quero cancelar", "Esqueci minha senha", "Fui cobrado duas vezes",
             "Produto veio quebrado, quero trocar", "refund me"]:
    check("route/undecided follows default " + text[:24], _r_ml.route(text).model, "multilingual")
    check("route/undecided stock default " + text[:24], _r_lat.route(text).model, "english")
check("route/undecided reason names the default",
      "using default (multilingual)" in _r_ml.route("Esqueci minha senha").reason, True)
# identified English is not undecided, so a non-English default leaves it alone
for text in ["Please refund the duplicate charge", "Please refund the duplicate charge on invoice 4411 today."]:
    check("route/identified english ignores default " + text[:24], _r_ml.route(text).model, "english")
check("route/english reason unchanged",
      _r_lat.route("Please refund the duplicate charge on invoice 4411 today.").reason, "English Latin text")


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
    check("route/romance control " + text[:32], _r_lat.route(text).model, "english")

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

# A state with no letters at all must keep behaving exactly as before.
for label, text in (("empty", ""), ("digits only", "12345 67890"), ("emoji only", "😀😀😀")):
    check("unlisted/letterless " + label + " is still unknown",
          analyse(text)["script"], "unknown")
    check("unlisted/letterless " + label + " keeps the default",
          Router().route(text)["model"], "english")

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

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all routing tests passed")
sys.exit(1 if FAIL else 0)
