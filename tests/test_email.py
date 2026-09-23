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
"""
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
    ("dash delimiter", "--\nAnna Meier\nSupport"),
]:
    check("signoff cut/" + label, clean_email_body("%s\n\n%s" % (BODY, tail)), BODY)


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


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
