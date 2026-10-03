"""Email cleaning: a disclaimer footer must not delete the sender's actual request.

Regression tests for `laya.email.clean_email_body`. `_DISCLAIMER` used to be applied to whole
paragraphs, so any paragraph that merely *mentioned* boilerplate was deleted outright. When the
footer ran on without a blank line, the request went with it:

    clean_email_body("My account is locked.\\nThis email is confidential...\\nPlease unlock it.")
    # before: ''      <- the whole body, request included, was deleted
    # after:  'My account is locked. Please unlock it.'

Dropping the request is silent and severe; leaving one boilerplate line behind is neither, so the
cleaning errs towards keeping text.

The same asymmetry drives the sign-off tests below. `_SIGNATURE_MARKERS` allowed a whole
sentence behind the closing word, so an ordinary body line beginning with one of them was read
as the start of a signature and everything after it was cut:

    clean_email_body("Hi,\\n\\nThanks for the quick reply.\\nCould you refund invoice 4411?")
    # before: 'Hi,'    <- the request was cut away

The last block guards a different kind of silence. `email_questions` was defined twice, here and
in `laya/presets.py`, and `laya/__init__.py` re-exports the `presets` one. Editing the copy in
`laya/email.py` moved `laya.email.email_questions` and left `laya.email_questions` where it was,
with no test and no lint failing.

The block at the end is the same shape again: `clean_email_body` takes a `max_chars` budget and
`email_state` cut every body to its default, so a request in the last paragraphs of a long email was
deleted before the model saw it -- while the keyword a caller would use to stop that,
`email_state(..., max_chars=8000)`, fell into `**extra` and became a field of the state instead.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya  # noqa: E402
from laya import email as email_module  # noqa: E402
from laya import presets  # noqa: E402
from laya.email import clean_email_body, email_state  # noqa: E402

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


DISCLAIMER = "This email is confidential and intended solely for the named addressee."

# --------------------------------------------------------------- the request survives the footer
check(
    "inline footer/no blank line keeps the request",
    clean_email_body("My account is locked.\n%s\nPlease unlock it." % DISCLAIMER),
    "My account is locked. Please unlock it.",
)
check(
    "inline footer/unpunctuated request line is fully recovered",
    clean_email_body("My account is locked\n%s\nPlease unlock it." % DISCLAIMER),
    "My account is locked Please unlock it.",
)
check(
    "inline footer/fused request without a trailing sentence",
    clean_email_body("Please unlock my account\n%s" % DISCLAIMER),
    "Please unlock my account",
)
check(
    "inline footer/fused order reference is fully recovered",
    clean_email_body("RMA 5521 is still pending\n%s\nPlease advise." % DISCLAIMER),
    "RMA 5521 is still pending Please advise.",
)
check(
    # Keep-ward trade-off, documented on purpose: a capitalised continuation of a
    # boilerplate sentence can be a real fragment ("...in error,\nPlease delete it."),
    # so it stays. Leaving one boilerplate line behind is harmless; dropping the
    # request is not.
    "inline footer/capitalised continuation stays",
    clean_email_body("If you have received this message in error,\nPlease delete it."),
    "Please delete it.",
)
check(
    # Documented boundary: an all-lowercase fused request ("locked\nthis email ...")
    # is indistinguishable from a wrapped boilerplate footer, so the first line is
    # still lost. Only a newline followed by an uppercase letter splits.
    "inline footer/lowercase fusion still recovers the tail",
    clean_email_body(
        "my account is locked\n"
        "this email is confidential and intended solely for the named addressee.\n"
        "please unlock it."
    ),
    "please unlock it.",
)
check(
    "inline footer/body is never emptied",
    clean_email_body("My account is locked. %s" % DISCLAIMER),
    "My account is locked.",
)
check_true(
    "inline footer/is not empty",
    clean_email_body("My account is locked. %s" % DISCLAIMER).strip() != "",
)
check(
    "email_state/body keeps the request",
    email_state("Locked out", "My account is locked. %s" % DISCLAIMER)["body"],
    "My account is locked.",
)

# --------------------------------------------------------------- a pure footer is still removed
check(
    "standalone footer paragraph is still dropped",
    clean_email_body("My account is locked.\n\n%s" % DISCLAIMER),
    "My account is locked.",
)
check(
    "wrapped standalone footer is still dropped",
    clean_email_body(
        "My account is locked.\n\nThis email and any files transmitted with it are\n"
        "confidential and intended solely for the named addressee."
    ),
    "My account is locked.",
)
check(
    "received-in-error footer is still dropped",
    clean_email_body(
        "Please reopen ticket 4411.\n\nIf you have received this message in error, delete it."
    ),
    "Please reopen ticket 4411.",
)

# --------------------------------------------------------------- unrelated cleaning is unchanged
check(
    "quoted history is still removed",
    clean_email_body("Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text"),
    "Thanks for the update.",
)
check(
    "signature block is still removed",
    clean_email_body("Hi team,\nCan you confirm the refund?\nRegards,\nAlice"),
    "Hi team,\nCan you confirm the refund?",
)
check("empty body stays empty", clean_email_body(""), "")

# --------------------------------------------------------------- Portuguese and Spanish mail
# With English-only markers none of this was removed, and the quoted history below (a cancellation)
# reached the model next to the new message (a refund request).
PT_EMAIL = """Olá equipe,

Fomos cobrados duas vezes na fatura de março. Por favor, estornem a cobrança duplicada hoje.

Atenciosamente,
João Silva
Financeiro - ACME Ltda

Enviado do meu iPhone

Esta mensagem pode conter informações confidenciais. Se você recebeu esta mensagem por engano, favor apagá-la.

Em seg., 22 de set. de 2026 às 10:14, Suporte <suporte@x.com> escreveu:
> Olá João, recebemos seu chamado de cancelamento do plano Enterprise.
"""
check(
    "pt/full reply keeps only the request",
    clean_email_body(PT_EMAIL),
    "Olá equipe,\n\nFomos cobrados duas vezes na fatura de março. "
    "Por favor, estornem a cobrança duplicada hoje.",
)
check(
    "pt/outlook original-message block is cut",
    clean_email_body(
        "Segue o comprovante do pagamento.\n\n-----Mensagem original-----\n"
        "De: Maria <maria@acme.com>\nAssunto: cancelar contrato\nQueremos cancelar o contrato."
    ),
    "Segue o comprovante do pagamento.",
)
check(
    "pt/outlook header without separator is cut",
    clean_email_body(
        "Segue o comprovante.\n\nDe: Maria <maria@acme.com>\nEnviado: segunda-feira\n"
        "Assunto: cancelar contrato\nQueremos cancelar o contrato."
    ),
    "Segue o comprovante.",
)
check(
    "pt/gmail attribution wrapped over two lines is cut whole",
    clean_email_body(
        "O acesso voltou, obrigado.\n\nEm seg., 22 de set. de 2026 às 10:14, Suporte Técnico <\n"
        "suporte@acme.com> escreveu:\n> texto antigo"
    ),
    "O acesso voltou, obrigado.",
)
check(
    "pt/short sign-off is removed",
    clean_email_body("Bom dia,\nO boleto de março não chegou.\nObrigado,\nAna"),
    "Bom dia,\nO boleto de março não chegou.",
)
check(
    "es/reply keeps only the request",
    clean_email_body(
        "Hola,\nNo puedo acceder a mi cuenta desde ayer.\nSaludos,\nCarlos\n\n"
        "El lun, 22 sept 2026 a las 10:14, Soporte <soporte@x.com> escribió:\n> texto anterior"
    ),
    "Hola,\nNo puedo acceder a mi cuenta desde ayer.",
)
check(
    "es/disclaimer footer is dropped",
    clean_email_body(
        "Necesito la factura de marzo.\n\nSi usted ha recibido este mensaje por error, bórrelo."
    ),
    "Necesito la factura de marzo.",
)

# --------------------------------------------------------------- ...without eating the request
check(
    "pt/request mentioning `confidencial` is kept",
    clean_email_body("Preciso do contrato confidencial assinado até sexta."),
    "Preciso do contrato confidencial assinado até sexta.",
)
check(
    "pt/`Obrigado` opening a sentence is not a signature",
    clean_email_body("Oi,\nRecebi a resposta.\nObrigado pelo retorno, mas continua\nsem funcionar."),
    "Oi,\nRecebi a resposta.\nObrigado pelo retorno, mas continua\nsem funcionar.",
)
check(
    "pt/`caso tenha recebido` footer is dropped",
    clean_email_body(
        "Favor reenviar a nota fiscal.\n\nCaso tenha recebido esta mensagem por engano, "
        "notifique o remetente."
    ),
    "Favor reenviar a nota fiscal.",
)
check(
    "pt/gmail attribution wrapped inside the name is cut whole",
    clean_email_body(
        "Resolvido, pode fechar.\n\nEm qua., 24 de set. de 2026 às 09:02, Suporte\n"
        "Técnico <suporte@acme.com> escreveu:\n> texto antigo"
    ),
    "Resolvido, pode fechar.",
)
check(
    "en/wrapped attribution is cut whole too",
    clean_email_body(
        "Fixed, thanks.\n\nOn Wed, Sep 24, 2026 at 9:02 AM Support Team <\n"
        "support@acme.com> wrote:\n> old text"
    ),
    "Fixed, thanks.",
)
check(
    "pt/`Em ... escreveu:` without a date is body text",
    clean_email_body("Oi,\nEm resposta ao que você escreveu:\no pedido 4411 ainda não chegou."),
    "Oi,\nEm resposta ao que você escreveu:\no pedido 4411 ainda não chegou.",
)
check(
    "pt/`destinado exclusivamente` in a request is kept",
    clean_email_body("O valor é destinado exclusivamente ao pagamento do boleto. Podem confirmar?"),
    "O valor é destinado exclusivamente ao pagamento do boleto. Podem confirmar?",
)
check(
    "pt/`De:` without an address is body text",
    clean_email_body("Preciso das férias.\nDe: 10/09 a 15/09\nPode aprovar?"),
    "Preciso das férias.\nDe: 10/09 a 15/09\nPode aprovar?",
)

# --------------------------------------------------------------- French mail
# With English-only markers none of this was removed, and the quoted history below (a cancellation)
# reached the model next to the new message (a refund request).
FR_EMAIL = """Bonjour,

J'ai été facturé deux fois sur la facture de mars. Merci de rembourser le double paiement aujourd'hui.

Cordialement,
Jean Dupont

Envoyé depuis mon iPhone

Ce message peut contenir des informations confidentielles. Si vous avez reçu ce message par erreur, merci de le supprimer.

Le lun. 22 sept. 2026 à 10:14, Support <support@x.com> a écrit :
> Bonjour Jean, nous avons reçu votre demande d'annulation du contrat Enterprise.
"""
check(
    "fr/full reply keeps only the request",
    clean_email_body(FR_EMAIL),
    "Bonjour,\n\nJ'ai été facturé deux fois sur la facture de mars. "
    "Merci de rembourser le double paiement aujourd'hui.",
)
for footer in [
    "Envoyé depuis mon iPhone",
    "Envoyé de mon iPad.",
    "Envoyé depuis iPhone",
    "Envoyé de iPad",
    "ENVOYÉ DEPUIS MON IPHONE",
]:
    body = "Merci de rembourser la facture."
    check("fr/device footer cut/" + footer, clean_email_body(body + "\n\n" + footer), body)

for sentence in [
    "Envoyé depuis mon iPhone par erreur.",
    "Envoyé de mon iPad hier.",
]:
    body = "Bonjour,\n%s\nMerci de rembourser la facture." % sentence
    check("fr/device sentence kept/" + sentence, clean_email_body(body), body)

check(
    "fr/outlook original-message block is cut",
    clean_email_body(
        "Voici le justificatif de paiement.\n\n-----Message d'origine-----\n"
        "De : Marie <marie@acme.com>\nObjet : résilier le contrat\nNous voulons résilier le contrat."
    ),
    "Voici le justificatif de paiement.",
)
check(
    "fr/outlook header without separator is cut",
    clean_email_body(
        "Voici le justificatif.\n\nDe : Marie Dupont\nEnvoyé : lundi 22 septembre 2026\n"
        "Objet : résilier le contrat\nNous voulons résilier le contrat."
    ),
    "Voici le justificatif.",
)
check(
    "fr/gmail attribution wrapped over two lines is cut whole",
    clean_email_body(
        "L'accès est rétabli, merci.\n\nLe lun. 22 sept. 2026 à 10:14, Support Technique <\n"
        "support@acme.com> a écrit :\n> ancien texte"
    ),
    "L'accès est rétabli, merci.",
)
check(
    "fr/short sign-off is removed",
    clean_email_body("Bonjour,\nLa facture de mars n'est pas arrivée.\nMerci,\nJean"),
    "Bonjour,\nLa facture de mars n'est pas arrivée.",
)
check(
    "fr/`Bien à vous` sign-off is removed",
    clean_email_body("Bonjour,\nLa facture de mars n'est pas arrivée.\nBien à vous,\nMarie"),
    "Bonjour,\nLa facture de mars n'est pas arrivée.",
)
check(
    "fr/disclaimer footer is dropped",
    clean_email_body(
        "J'ai besoin de la facture de mars.\n\nSi vous avez reçu ce message par erreur, supprimez-le."
    ),
    "J'ai besoin de la facture de mars.",
)
check(
    "fr/exclusive-use footer is dropped",
    clean_email_body(
        "Voici le devis demandé.\n\nCe document est à l'usage exclusif du destinataire."
    ),
    "Voici le devis demandé.",
)

# --------------------------------------------------------------- ...without eating the request
check(
    "fr/request mentioning `confidentiel` is kept",
    clean_email_body("Le contrat confidentiel doit être signé avant vendredi."),
    "Le contrat confidentiel doit être signé avant vendredi.",
)
check(
    "fr/`Merci` opening a sentence is not a signature",
    clean_email_body("Bonjour,\nMerci pour votre aide.\nRappelez-moi."),
    "Bonjour,\nMerci pour votre aide.\nRappelez-moi.",
)
check(
    "fr/`Le ... a écrit :` without a date is body text",
    clean_email_body("Bonjour,\nLe rapport que vous avez écrit :\nla commande 4411 n'est pas arrivée."),
    "Bonjour,\nLe rapport que vous avez écrit :\nla commande 4411 n'est pas arrivée.",
)
check(
    "fr/`exclusivement` in a request is kept",
    clean_email_body("Le montant est destiné exclusivement au paiement de la facture. Pouvez-vous confirmer ?"),
    "Le montant est destiné exclusivement au paiement de la facture. Pouvez-vous confirmer ?",
)
check(
    "fr/`De :` without an address is body text",
    clean_email_body("J'ai besoin de congés.\nDe : 10/09 à 15/09\nC'est possible ?"),
    "J'ai besoin de congés.\nDe : 10/09 à 15/09\nC'est possible ?",
)

# ------------------------------------------- `From:` starts prose, not only a quote header (#338)
# The marker used to be `^\s*From:\s.+$`, which matched any line beginning "From: ". English
# prose opens that way ("From: my side the integration works, but please refund ..."), so the
# request was deleted and the model answered an empty body. A reply header always carries the
# sender, so the marker is gated on an address -- the rule `De:` above already uses, for the
# same reason. The suite had no `From:` case at all before this, which is why it stayed green.
for label, body in [
    ("the reported case",
     "Hi team,\nFrom: my side the integration works, but please refund the duplicate charge today.\nThanks"),
    ("lowercase", "Hi,\nfrom: my side the integration works, but please refund the charge.\nThanks"),
    ("uppercase", "Hi,\nFROM: my side the integration works, but please refund the charge.\nThanks"),
    ("extra spacing", "Hi,\nFrom:   my side the integration works, but please refund the charge.\nThanks"),
    ("mid-body", "Hello,\nFrom: what I can see the charge was taken twice, please refund it.\nRegards"),
]:
    check_true("en/`From:` prose keeps the request: " + label,
               "refund" in clean_email_body(body), clean_email_body(body))

# The positive controls: a real quoted header block still goes, and takes its quoted request
# with it, so the fix does not simply stop cutting on `From:`.
for label, body in [
    ("address then Sent:",
     "Hi,\nPlease look at this.\n\nFrom: Alice <alice@example.com>\nSent: Monday\n"
     "To: Bob\nSubject: Refund\n\nPlease refund the duplicate charge to my card."),
    ("address only", "Hi,\nSee below.\n\nFrom: alice@example.com\nPlease refund the duplicate charge."),
    ("angle address", "Hi,\nSee below.\n\nFrom: <alice@example.com>\nPlease refund the duplicate charge."),
    ("name and address",
     "Hi,\nSee below.\n\nFrom: Alice Smith <alice@example.com>\nPlease refund the duplicate charge."),
]:
    check("en/quoted block still cut: " + label, clean_email_body(body), "Hi,\nPlease look at this."
          if label == "address then Sent:" else "Hi,\nSee below.")

# ...and the request in the new part is kept even when a quoted block follows it.
check(
    "en/`From:` quote header keeps the new request",
    clean_email_body("Hi,\nPlease refund the duplicate charge today.\n\n"
                     "From: Alice <alice@example.com>\nSent: Monday\n\nOld thread: refund the first charge."),
    "Hi,\nPlease refund the duplicate charge today.",
)

# A bare `From: Name` header, with no address, is the English counterpart of `De: Maria Souza`:
# the marker above cannot cut it (that is the prose case), so it is recognised by its neighbours
# instead. Before this, only the Portuguese/Spanish `Enviado:`/`Data:` pair was a neighbour, so an
# English quote header without an address leaked its whole quoted block into the body.
for label, body in [
    ("then Sent:", "Hi,\nSee below.\n\nFrom: Alice Smith\nSent: Monday, 21 Sep 2026\n\nPlease refund."),
    ("then Date:", "Hi,\nSee below.\n\nFrom: Alice Smith\nDate: 21/09/2026\n\nPlease refund."),
    ("one-word name, then Sent:", "Hi,\nSee below.\n\nFrom: Alice\nSent: Monday\n\nPlease refund."),
]:
    check("en/bare `From:` header still cut: " + label, clean_email_body(body), "Hi,\nSee below.")

# A bare `From:` with no header neighbour is not distinguishable from prose, and a dropped
# request is the worse error, so it is kept.
for label, body in [
    ("From: Name then To:", "Hi,\nSee below.\n\nFrom: Alice Smith\nTo: Bob\n\nPlease refund the charge."),
    ("From: Name alone", "Hi,\nSee below.\n\nFrom: Alice Smith\nPlease refund the charge."),
]:
    check_true("en/bare `From:` with no header neighbour is kept: " + label,
               "refund" in clean_email_body(body), clean_email_body(body))

# --------------------------------------------------------------- Brazilian clients and footers
BOLETO = "Preciso da segunda via do boleto."
for label, footer in [
    ("galaxy", "Enviado do meu Galaxy"),
    ("samsung default, 41 characters", "Enviado do meu smartphone Samsung Galaxy."),
    ("outlook ios", "Enviado do Outlook para iOS"),
    ("outlook android download line", "Obter o Outlook para Android"),
    ("windows mail", "Enviado do Email para Windows"),
    ("yahoo", "Enviado do Yahoo Mail no Android"),
    ("eco footer", "Antes de imprimir, pense em sua responsabilidade e compromisso com o MEIO AMBIENTE."),
    ("eco footer, reversed", "Pense no meio ambiente antes de imprimir este e-mail."),
]:
    check("pt/footer removed: " + label, clean_email_body(BOLETO + "\n\n" + footer), BOLETO)
check(
    "en/outlook download line removed",
    clean_email_body("Please resend the invoice.\n\nGet Outlook for iOS"),
    "Please resend the invoice.",
)
# Exchange leaves the address out of the header; the `Enviado:`/dated `Data:` line under it still marks it
for label, second in [("enviado", "Enviado: sexta-feira, 19 de setembro de 2026 10:02"),
                      ("data", "Data: sexta-feira, 19 de setembro de 2026 10:02")]:
    check(
        "pt/outlook header without address is cut: " + label,
        clean_email_body("Segue o comprovante.\n\nDe: Maria Souza\n%s\nPara: Suporte\n"
                         "Assunto: cancelar contrato\n\nQueremos cancelar o contrato." % second),
        "Segue o comprovante.",
    )
# ...and none of them may take the request with it
for label, body in [
    ("device words inside a request", "Oi,\nSegue o pedido.\nEnviado do meu celular o comprovante ontem."),
    ("`De:`/`Para:` date range", "Preciso das férias.\nDe: 10/09\nPara: 15/09\nPode aprovar?"),
    ("`De:` + undated `Data:`", "Relatório do evento.\nDe: João\nData: amanhã cedo\nPode confirmar?"),
    ("`antes de imprimir` in a request", "Antes de imprimir o boleto, confira o valor. Está errado."),
    ("`get` + device word", "Hi,\nThe box is at the front desk.\nGet mail"),
]:
    check("pt/kept: " + label, clean_email_body(body), body)

# ------------------------------------------- a sign-off word inside the body is not a sign-off
SHORT = "Hi,\n\nThanks for the quick reply.\nCould you refund invoice 4411 as well?"
check("signoff word/short mail keeps the request", clean_email_body(SHORT), SHORT)
check(
    "signoff word/thanks mid-body keeps what follows",
    clean_email_body(
        "Hello,\n\nWe were billed twice in March.\nThanks for looking into it.\n"
        "The duplicate is 49 EUR on invoice 4411."
    ),
    "Hello,\n\nWe were billed twice in March.\nThanks for looking into it.\n"
    "The duplicate is 49 EUR on invoice 4411.",
)
check(
    "signoff word/best mid-body keeps what follows",
    clean_email_body(
        "Hi team,\n\nOur account is locked.\nBest practice would be a manual unlock.\n"
        "Please unlock account 88213 today."
    ),
    "Hi team,\n\nOur account is locked.\nBest practice would be a manual unlock.\n"
    "Please unlock account 88213 today.",
)
check(
    "signoff word/email_state keeps the request",
    email_state("Duplicate charge", SHORT)["body"],
    SHORT,
)

# ------------------------------------------- real sign-offs are still cut (positive controls)
BODY = "Hi,\n\nPlease refund invoice 4411."
for label, tail in [
    ("thanks comma", "Thanks,\nAnna"),
    ("thanks bang", "Thanks!"),
    ("best regards", "Best regards,\nAnna"),
    ("kind regards", "Kind regards"),
    ("cheers name", "Cheers, Anna"),
    ("many thanks", "Many thanks,\nAnna Meier"),
    ("thank you", "Thank you,"),
    ("thanks in advance", "Thanks in advance,"),
    ("sincerely", "Sincerely,\nA. Meier"),
    ("sent from phone", "Sent from my iPhone"),
    ("sent from tablet", "Sent from my iPad."),
    ("sent from android", "Sent from my Android"),
    ("sent from mobile", "Sent from my mobile"),
    ("sent from iphone model", "Sent from my iPhone 15 Pro"),
    ("sent from android phone", "Sent from my Android phone"),
    ("sent from ipad model", "Sent from my iPad Pro"),
    ("sent from iphone app", "Sent from my iPhone using Tapatalk"),
    ("dash delimiter", "--\nAnna Meier\nSupport"),
]:
    check("signoff cut/" + label, clean_email_body("%s\n\n%s" % (BODY, tail)), BODY)

for suffix in ["device", "Max", "mini", "Plus"]:
    tail = "Sent from my iPhone %s" % suffix
    check("device footer suffix/" + suffix, clean_email_body("%s\n\n%s" % (BODY, tail)), BODY)

# A device mention can describe the request rather than close the message.
for sentence in [
    "Sent from my iPhone by mistake.",
    "Sent from my iPad yesterday.",
    "Sent from my Android by mistake.",
    "Sent from my mobile yesterday.",
    "Sent from my iPhone using Tapatalk to report a problem.",
]:
    body = "Hi support,\n%s\nPlease cancel the duplicate order." % sentence
    check("device sentence kept/" + sentence, clean_email_body(body), body)

# ------------------------------------------- closings the case rule did not reach (#132 follow-up)
# These were cut before #132 and are not now: `warmest` is not in the alternation, `and regards`
# is not one of its continuations, and `[A-Z]` is ASCII, so a name in any other script reads as a
# sentence. Each leaves the signature block in the body that the sign-off rule exists to remove.
for label, tail in [
    ("thanks and regards", "Thanks and regards,\nAnna"),
    ("thanks & regards", "Thanks & Regards,\nAnna"),
    ("warmest regards", "Warmest regards,\nAnna"),
    ("warmest wishes", "Warmest wishes,"),
    ("non-ascii name", "Regards, Łukasz"),
    ("non-ascii name, accented", "Thanks, José"),
    ("cyrillic name", "Regards, Дмитрий"),
]:
    check("signoff cut/" + label, clean_email_body("%s\n\n%s" % (BODY, tail)), BODY)

# ...and the wider closing must not swallow a sentence that merely starts the same way
for label, body in [
    ("and + sentence", "Hi,\n\nPlease refund 4411.\nThanks and the team will confirm it today."),
    ("warmest + sentence", "Hi,\n\nThe room is cold.\nWarmest setting still reads 18 degrees."),
]:
    check("signoff kept/" + label, clean_email_body(body), body)

# ------------------------------------------- the name rule reaches lowercase beyond Latin-1
# #245 replaced the ASCII `[A-Z]` with "exclude the lowercase letters", but the ranges that
# exclude them stop at Latin-1 (`a-zß-öø-ÿ`). A word starting with ż, д or α *is* lowercase --
# not "capitalised in any script" -- yet the class read it as a name, so the line counted as a
# sign-off and the cut dropped whatever followed it. Case is a property of the character, not of
# the script: the port checks it per character (`\p{Lu}\p{Lt}\p{Lo}`), and Python now checks the
# same first-letter categories.
for label, body in [
    ("latin-extended lowercase name", "Hi,\n\nPlease refund invoice 4411.\nThanks, żaneta"),
    ("cyrillic lowercase name", "Hi,\n\nPlease refund invoice 4411.\nThanks, дмитрий"),
    ("greek lowercase name", "Hi,\n\nPlease refund invoice 4411.\nThanks, αλέξανδρος"),
    ("a roman numeral is not a name", "Hi,\n\nPlease refund invoice 4411.\nThanks, Ⅷ"),
]:
    check("signoff kept/" + label, clean_email_body(body), body)
# ...while a combining mark stays part of the letter before it: `Jose\u0301` spells `José` and
# still cuts as a sign-off (the port's `\p{M}`).
check(
    "signoff cut/decomposed accented name",
    clean_email_body("%s\n\n%s" % (BODY, "Regards, Jose\u0301")),
    BODY,
)
# ...in every script, not only the Latin one. `\u0300-\u036f` is the Latin combining block
# alone, so a name carrying a mark from any other script -- a Devanagari virama, an Arabic
# shadda, a Hebrew point, a Thai tone mark -- failed the tail and the signature stayed in the
# body. The port's `\p{M}` covers every mark, which is what these names need.
for label, tail in [
    ("devanagari", "Thanks, \u0928\u092e\u0938\u094d\u0924\u0947"),
    ("devanagari name", "Thanks, \u0930\u0935\u093f"),
    ("arabic", "Thanks, \u0645\u062d\u0645\u0651\u062f"),
    ("hebrew", "Regards, \u05e9\u05c1\u05dc\u05d5\u05dd"),
    ("thai", "Thanks, \u0e2a\u0e38\u0e0a\u0e32\u0e15\u0e34\u0e4c"),
    ("bengali", "Thanks, \u0985\u09ae\u09bf\u09a4"),
    ("tamil", "Thanks, \u0bb5\u0bc6\u0bb3\u0bcd\u0bb3\u0bbf"),
]:
    check("signoff cut/mark beyond latin-1, " + label,
          clean_email_body("%s\n\n%s" % (BODY, tail)), BODY)

# ...and a mark with no base is left where it is. Stripping one that opens the tail, or that
# follows a space, would join the tokens around it and cut a line the port keeps: `laya-ts`
# requires each token's first character to be `\p{Lu}\p{Lt}\p{Lo}`, which a leading mark fails.
# ZWJ and ZWNJ are `Cf` rather than `M`, so they are untouched here and in the port alike.
for label, body in [
    ("mark after a space", "Hi,\n\nPlease refund invoice 4411.\nThanks, Jose \u0301Smith"),
    ("mark opening the name", "Hi,\n\nPlease refund invoice 4411.\nThanks, \u0301Jose"),
    ("mark standing alone", "Hi,\n\nPlease refund invoice 4411.\nThanks, \u0301 Jose"),
    ("zwnj is not a mark", "Hi,\n\nPlease refund invoice 4411.\nThanks, \u0915\u094d\u200c\u0937"),
]:
    check("signoff kept/" + label, clean_email_body(body), body)

# ...and dropping the marks must not turn a lowercase name into one: the letter a mark rides on
# is what the case rule asks about, so a marked lowercase name is still not a sign-off.
for label, body in [
    ("greek lowercase, accented", "Hi,\n\nPlease refund invoice 4411.\nThanks, \u03b1\u0301\u03bb\u03c6\u03b1"),
    ("latin lowercase, decomposed", "Hi,\n\nPlease refund invoice 4411.\nThanks, jose\u0301"),
]:
    check("signoff kept/" + label, clean_email_body(body), body)


# ------------------------------------------------- the word, without the disclaimer
# `confidential` was a bare substring of `_DISCLAIMER`, so any sentence that merely
# mentioned it was dropped. A one-sentence body that mentions it was deleted whole and
# the model was then scored on an empty state, silently. The Portuguese branches beside
# it were already tied to disclaimer phrasing for this reason; English now is too.
for label, body in [
    ("a question about the word", "Is this confidential?"),
    ("a policy question", "What is your confidentiality policy?"),
    ("a request containing the word", "Please keep this confidential but process my refund."),
    ("a request about handling", "Please treat this as confidential."),
    ("a question with a dash", "This is confidential - can you help?"),
    ("a question about an attachment", "Is the attached document confidential?"),
    ("a label prefix", "Confidential: I need a refund."),
    ("a question about information", "What is the information policy for contractors?"),
]:
    check("word only/kept: " + label, clean_email_body(body), body)
check_true(
    "word only/body is never emptied",
    clean_email_body("Is this confidential?").strip() != "",
)
check(
    "word only/email_state keeps the request",
    email_state("Question", "Is this confidential?")["body"],
    "Is this confidential?",
)

# ...while the real footers those branches exist for are still dropped
for label, body in [
    ("named addressee", "This email is confidential and intended solely for the named addressee."),
    ("the individual addressed",
     "This message is confidential and intended solely for the use of the individual to whom it is addressed."),
    ("may be privileged", "The information in this email is confidential and may be privileged."),
    ("wrapped across lines",
     "This email and any files transmitted with it are\n"
     "confidential and intended solely for the named addressee."),
]:
    check_true("word only/still dropped: " + label, not clean_email_body(body).strip())
check(
    "word only/request before a footer survives",
    clean_email_body("My account is locked.\n"
                     "This email is confidential and intended solely for the named addressee.\n"
                     "Please unlock it."),
    "My account is locked. Please unlock it.",
)
check(
    "word only/request inside one sentence survives",
    clean_email_body("Please unlock it. This email is confidential and intended solely "
                     "for the named addressee."),
    "Please unlock it.",
)


# ------------------------------------------- email_questions has exactly one definition
check_true(
    "email_questions/one definition behind both module paths",
    email_module.email_questions is presets.email_questions,
    "laya.email.email_questions is not laya.presets.email_questions",
)
check_true(
    "email_questions/the package export is that same object",
    laya.email_questions is presets.email_questions,
)
check(
    "email_questions/both paths answer the same",
    email_module.email_questions(),
    laya.email_questions(),
)
check(
    "email_questions/a caller override reaches both paths",
    email_module.email_questions({"legal": "contracts"})["category"]["criteria"],
    laya.email_questions({"legal": "contracts"})["category"]["criteria"],
)


# --------------------------------- email_state owns the body budget (a `clean_email_body` control)
# `clean_email_body(body, max_chars=...)` takes a budget and `email_state` called it with none, so
# the state was always cut at 3000 characters with no lever. The one call a reader reaches for,
# `email_state(subject, body, max_chars=8000)`, went to `**extra` instead, which merges the keyword
# into the state that `serialize_state` renders into the model's input: the budget stayed at 3000
# and a `"max_chars": 8000` field was added to what the checkpoint reads.
LONG = ("The account review covers the last twelve statements and the support thread that goes with "
        "them, including the charges that were disputed and the credits that were applied. ") * 40
REQUEST = "Please refund the duplicate monthly charge of 49 dollars from the account ending 4417."
LONG_BODY = LONG + "\n\n" + REQUEST

check(
    "budget/default still cuts at 3000",
    len(email_state("Billing", LONG_BODY)["body"]),
    3000,
)
check_true(
    "budget/the default cuts away a late request",
    REQUEST not in email_state("Billing", LONG_BODY)["body"],
    "a 3000-character cap should not reach the closing line of a 7000-character body",
)
check_true(
    "budget/raising max_chars keeps the late request",
    REQUEST in email_state("Billing", LONG_BODY, max_chars=8000)["body"],
    "the whole point of the lever",
)
check(
    "budget/max_chars reaches clean_email_body unchanged",
    email_state("Billing", LONG_BODY, max_chars=8000)["body"],
    clean_email_body(LONG_BODY, max_chars=8000),
)
check_true(
    "budget/max_chars is not a field of the state",
    "max_chars" not in email_state("Billing", LONG_BODY, max_chars=8000),
    "it belongs to the cleaning, not to what the model reads",
)
check(
    "budget/clean=False passes the body whole",
    email_state("Billing", LONG_BODY, clean=False, max_chars=100)["body"],
    LONG_BODY,
)
# `max_chars` is added last, so a call that already passed four positional arguments still means
# subject, body, sender, clean. Reading the order off the signature is what fails if it moves.
check(
    "budget/positional order is unchanged",
    [p.name for p in inspect.signature(email_state).parameters.values()][:5],
    ["subject", "body", "sender", "clean", "max_chars"],
)
check(
    "budget/sender still lands under from",
    email_state("Billing", "Short body", sender="me@example.com").get("from"),
    "me@example.com",
)
# The two defaults are literals on purpose -- a signature is documentation, and `max_chars: int =
# 3000` reads better than a constant's name -- so this is what keeps the pair from drifting.
def budget_default(fn):
    param = inspect.signature(fn).parameters.get("max_chars")
    return None if param is None else param.default


check(
    "budget/the two defaults are the same value",
    budget_default(email_state),
    budget_default(clean_email_body),
)
check(
    "budget/email_state cuts where clean_email_body promises",
    email_state("Billing", LONG_BODY)["body"],
    clean_email_body(LONG_BODY),
)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
