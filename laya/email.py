"""Email utilities for cleaning and structuring email inputs in laya.

The markers below cover English, Portuguese and Spanish mail clients. The Router already sends
Portuguese and Spanish states to the multilingual checkpoint, but with English-only markers their
cleaning was a no-op: Gmail's `Em ... escreveu:`, Outlook's `-----Mensagem original-----`, the
`Atenciosamente` sign-off and the confidentiality footer all reached the model, and the quoted
history (often a *different* request) weighed on the answer as much as the new message did.
"""
import re
from typing import Dict, Optional

_QUOTE_HEADERS = [
    re.compile(r"^\s*On .{0,300}wrote:\s*$", re.I),
    # "Em resposta ao que você escreveu:" is body text; a client's attribution always carries a date
    re.compile(r"^\s*Em (?=.*\d).{0,300}escreveu:\s*$", re.I),
    re.compile(r"^\s*El (?=.*\d).{0,300}escribi[óo]:\s*$", re.I),
    re.compile(r"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", re.I),
    re.compile(r"^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado))\s*-{2,}", re.I),
    re.compile(r"^\s*_{8,}\s*$"),
    re.compile(r"^\s*From:\s.+$", re.I),
    # `De:` also opens ordinary Portuguese/Spanish lines ("De: 10/09 a 15/09"), so the Outlook
    # header is only recognised when it carries an address
    re.compile(r"^\s*De:\s.*[@<]", re.I),
]
# Gmail wraps a long attribution line, leaving `fulano@x.com> escreveu:` alone on the next line.
# That tail cuts too, and takes the `On/Em/El ...` head it belongs to with it.
_ATTRIBUTION_TAIL = re.compile(r"^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]):\s*$", re.I)
_ATTRIBUTION_HEAD = re.compile(r"^\s*(On|Em|El) (?=.*\d)", re.I)
_SIGNATURE_MARKERS = [
    re.compile(r"^\s*--\s*$"),
    re.compile(r"^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$", re.I),
    re.compile(r"^\s*sent from my (iphone|android|mobile|ipad)", re.I),
    # Portuguese/Spanish sign-offs match only on their own: "Obrigado pelo retorno, mas ..." is a
    # request, not a signature, so unlike the English marker no trailing words are allowed
    re.compile(
        r"^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?"
        r"( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|"
        r"(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?)[\s,!.]*$",
        re.I,
    ),
    re.compile(r"^\s*enviado (do meu|desde mi) (iphone|android|celular|telemóvel|ipad|móvil)", re.I),
]
_DISCLAIMER = re.compile(
    r"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|"
    r"if you (have )?received this (e-?mail|message) in error|"
    # Portuguese/Spanish: tied to "this message/e-mail" rather than the bare word `confidencial`,
    # which a sender's own request ("preciso do contrato confidencial") uses just as often
    r"\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)|"
    r"\b(uso exclusivo|exclusivamente|únicamente|unicamente)\b[^.]{0,30}"
    r"(destinatári|destinatari|pessoa|persona|entidade|entidad)|"
    r"\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\b[^.]{0,20} por (engano|erro)|"
    r"\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\b[^.]{0,20} por error)",
    re.I,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _strip_disclaimer(paragraph: str) -> str:
    """Drop boilerplate disclaimer text from one paragraph.

    A paragraph is dropped whole only when *every* sentence in it is boilerplate; otherwise only
    the boilerplate sentences go. A footer that runs on without a blank line used to take the
    sender's actual request with it, which is worse than leaving one boilerplate line behind.
    """
    if not _DISCLAIMER.search(paragraph):
        return paragraph                     # nothing to do: keep the original line structure
    parts = [p.strip() for p in _SENTENCE.split(paragraph) if p.strip()]
    return " ".join(p for p in parts if not _DISCLAIMER.search(p))


def clean_email_body(body: str, max_chars: int = 3000) -> str:
    """Remove quoted email history, signatures and disclaimers to keep input focused."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    lines = []
    for line in text.split("\n"):
        if any(p.match(line) for p in _QUOTE_HEADERS) and lines:
            break
        if _ATTRIBUTION_TAIL.match(line) and lines:
            if _ATTRIBUTION_HEAD.match(lines[-1]):
                lines.pop()
            break
        if line.lstrip().startswith(">"):
            continue
        lines.append(line.rstrip())
    cut = len(lines)
    for i in range(max(1, min(int(len(lines) * 0.6), len(lines) - 8)), len(lines)):
        if len(lines[i].strip()) <= 40 and any(p.match(lines[i]) for p in _SIGNATURE_MARKERS):
            cut = i
            break
    lines = lines[:cut]
    paragraphs = [_strip_disclaimer(p) for p in re.split(r"\n\s*\n", "\n".join(lines))]
    text = re.sub(r"[ \t]+", " ", "\n\n".join(p.strip() for p in paragraphs if p.strip()))
    return text[:max_chars]


def email_state(subject: str, body: str, sender: Optional[str] = None, clean: bool = True, **extra) -> Dict:
    """Construct a clean state dictionary for email classification."""
    state = {
        "subject": (subject or "").strip(),
        "body": clean_email_body(body) if clean else (body or ""),
    }
    if sender:
        state["from"] = sender
    state.update({k: v for k, v in extra.items() if v is not None})
    return state


def email_questions(categories: Optional[Dict[str, str]] = None) -> Dict:
    """Standard pre-built questions for email triage."""
    categories = categories or {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, integrations",
        "sales": "pricing, demos, new purchases",
        "security": "phishing, scams, account compromise",
        "hr": "hiring, leave, payroll",
        "other": "none of the above",
    }
    return {
        "category": {
            "type": "choice",
            "instructions": "Which team should handle the email in `body`?",
            "criteria": categories,
        },
        "is_spam": {
            "type": "noul",
            "instructions": "Is this email unsolicited spam or bulk marketing?",
        },
        "is_phishing": {
            "type": "noul",
            "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
            "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the request in `body`?",
            "criteria": ["no time pressure", "needs attention soon", "blocking issue or hard deadline"],
        },
        "needs_reply": {
            "type": "noul",
            "instructions": "Does the sender expect a reply?",
        },
    }
