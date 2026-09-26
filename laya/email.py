"""Email utilities for cleaning and structuring email inputs in laya.

The markers below cover English, Portuguese and Spanish mail clients. The Router already sends
Portuguese and Spanish states to the multilingual checkpoint, but with English-only markers their
cleaning was a no-op: Gmail's `Em ... escreveu:`, Outlook's `-----Mensagem original-----`, the
`Atenciosamente` sign-off and the confidentiality footer all reached the model, and the quoted
history (often a *different* request) weighed on the answer as much as the new message did.
"""
import re
from typing import Dict, List, Optional

# One definition, in the module that holds the other presets. Re-exported here because
# `from laya.email import email_questions` is a path callers already have.
from .presets import email_questions  # noqa: F401

_QUOTE_HEADERS = [
    re.compile(r"^\s*On .{0,300}wrote:\s*$", re.I),
    # "Em resposta ao que você escreveu:" is body text; a client's attribution always carries a date
    re.compile(r"^\s*Em (?=.*\d).{0,300}escreveu:\s*$", re.I),
    re.compile(r"^\s*El (?=.*\d).{0,300}escribi[óo]:\s*$", re.I),
    re.compile(r"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", re.I),
    re.compile(r"^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado))\s*-{2,}", re.I),
    re.compile(r"^\s*_{8,}\s*$"),
    # `From:` opens ordinary prose too ("From: my side the integration works, but please
    # refund..."), and a reply header always carries the sender, so the header is only
    # recognised when an address follows -- the same rule as `De:` below. A bare
    # `From: Name` header is caught by _HEADER_FROM_NAME/_HEADER_NEXT instead, which need the
    # header's own `Sent:`/`Date:` line to tell it apart from a sentence.
    re.compile(r"^\s*From:\s.*[@<]", re.I),
    # `De:` also opens ordinary Portuguese/Spanish lines ("De: 10/09 a 15/09"), so the Outlook
    # header is only recognised when it carries an address
    re.compile(r"^\s*De:\s.*[@<]", re.I),
]
# Gmail wraps a long attribution line, leaving `fulano@x.com> escreveu:` alone on the next line.
# That tail cuts too, and takes the `On/Em/El ...` head it belongs to with it.
_ATTRIBUTION_TAIL = re.compile(r"^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]):\s*$", re.I)
_ATTRIBUTION_HEAD = re.compile(r"^\s*(On|Em|El) (?=.*\d)", re.I)
# Exchange often leaves the address out of Outlook's reply header ("De: Maria Souza"), so a bare `De:`
# only cuts when the header's own `Enviado:` line, or a dated `Data:`/`Fecha:` line, follows it.
# `Para:` is not enough: "De: 10/09 / Para: 15/09" is how a leave request reads.
#
# The same is true of a bare English `From: Maria Souza`, which is why the marker above needs
# this rule: the English client lines are the translations of the two `De:` neighbours. A line
# that only looks like prose still has to be told apart from a header by its neighbours, so the
# English pair is "From: <name>" followed by "Sent:"/"Date:".
_HEADER_FROM_NAME = re.compile(r"^\s*(De|From):\s+\S", re.I)
_HEADER_NEXT = re.compile(r"^\s*(Enviad[oa]( em| el)?:\s|Sent:\s|(Data|Fecha|Date):\s.*\d{4})", re.I)
_SIGNATURE_MARKERS = [
    re.compile(r"^\s*--\s*$"),
    # A closing line is the closing word plus punctuation and at most a name. Anything else on
    # the line is a sentence, and the case of the next word is what separates the two: a name is
    # capitalised, "for" in "Thanks for the quick reply." is not. The closing words are matched
    # case-insensitively, the name is not, so the flag is scoped instead of global.
    # `warmest` and `and/& regards` are closings the alternation did not reach, and a name is
    # capitalised in any script, so the name class excludes the lowercase letters instead of
    # listing the uppercase ones: `Regards, Łukasz` is a sign-off, `Thanks for the reply` is not.
    re.compile(
        r"^\s*(?i:best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)"
        r"(?i:\s+(?:and|&)\s+regards|\s+(?:regards|wishes|again|in advance|a lot|so much|very much))?"
        r"[\s,;:!.]*(?:[^\W\d_a-zß-öø-ÿ][\w'-]*[\s,.]*){0,3}$"
    ),
    re.compile(r"^\s*sent from my (iphone|android|mobile|ipad)", re.I),
    # Portuguese/Spanish sign-offs match only on their own: "Obrigado pelo retorno, mas ..." is a
    # request, not a signature, so unlike the English marker no trailing words are allowed
    re.compile(
        r"^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?"
        r"( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|"
        r"(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?)[\s,!.]*$",
        re.I,
    ),
]
# Mobile and mail-app footers. Only a line that is nothing *but* the footer matches -- "Enviado do meu
# celular o comprovante ontem." is a request -- and such a line may run to 60 characters, since
# Samsung's default ("Enviado do meu smartphone Samsung Galaxy.") is longer than a sign-off's 40.
_DEVICE = (r"iphone|ipad|android|ios|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|"
           r"outlook|yahoo|mail|e-?mail|gmail|windows")
_DEVICE_FOOTER = re.compile(
    r"^\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?)"
    r" (%s)( (%s|para|for|no|na|\d+))*|(obter o|get) outlook (para|for) (ios|android))[\s.!]*$"
    % (_DEVICE, _DEVICE),
    re.I,
)
_DISCLAIMER = re.compile(
    # English: tied to a disclaimer noun and a disclaimer tail, the way the Portuguese
    # branches below are. The bare word matched any sentence that merely mentioned it,
    # so "Is this confidential?" and "Confidential: I need a refund." were deleted whole.
    # `[^.]` rather than `[^.\n]`: a footer wraps, so "are\nconfidential" must still match.
    r"(\b(e-?mail|message|information|communication|transmission|contents?)\b[^.]{0,60}"
    r"\bconfidential\b[^.]{0,60}\b(intended|solely|addressee|recipient|privileged|"
    r"disclos|unauthori[sz]ed)|"
    r"\bconfidential\b[^.]{0,60}\b(and (may|is) (also )?privileged)|"
    r"if you (have )?received this (e-?mail|message) in error|"
    # Portuguese/Spanish: tied to "this message/e-mail" rather than the bare word `confidencial`,
    # which a sender's own request ("preciso do contrato confidencial") uses just as often
    r"\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)|"
    r"\b(uso exclusivo|exclusivamente|únicamente|unicamente)\b[^.]{0,30}"
    r"(destinatári|destinatari|pessoa|persona|entidade|entidad)|"
    r"\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\b[^.]{0,20} por (engano|erro)|"
    r"\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\b[^.]{0,20} por error|"
    # the "think before printing" footer, tied to its environmental ending rather than to
    # `antes de imprimir`, which a request uses too ("antes de imprimir o boleto, confira o valor")
    r"\bantes de imprimir\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|"
    r"\b(meio|medio) ambiente\b[^.]{0,30}antes de imprimir)",
    re.I,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _starts_new_sentence(line: str) -> bool:
    """First letter is uppercase: a fresh sentence, not a wrapped line.

    Lines in uncased scripts (CJK, Devanagari, ...) never start a new piece,
    so wrapped boilerplate in those scripts still drops whole.
    """
    for ch in line:
        if ch.isalpha():
            return ch.isupper()
    return False


def _split_fused_lines(sentence: str) -> List[str]:
    """Split a fused boilerplate-positive sentence at sentence-starting newlines.

    An unpunctuated request line glued to a disclaimer line ("locked\\nThis ...")
    splits at the newline because the next line starts uppercase; a lowercase
    continuation ("are\\nconfidential") belongs to the same sentence, so a wrapped
    boilerplate footer still drops whole.
    """
    if "\n" not in sentence:
        return [sentence]
    pieces, buf = [], ""
    for line in (ln.strip() for ln in sentence.split("\n")):
        if not line:
            continue
        if buf and _starts_new_sentence(line):
            pieces.append(buf)
            buf = line
        else:
            buf = (buf + " " + line) if buf else line
    if buf:
        pieces.append(buf)
    return pieces


def _strip_disclaimer(paragraph: str) -> str:
    """Drop boilerplate disclaimer text from one paragraph.

    A paragraph is dropped whole only when *every* sentence in it is boilerplate; otherwise only
    the boilerplate sentences go. A footer that runs on without a blank line used to take the
    sender's actual request with it, which is worse than leaving one boilerplate line behind.
    """
    if not _DISCLAIMER.search(paragraph):
        return paragraph                     # nothing to do: keep the original line structure
    parts = [p.strip() for p in _SENTENCE.split(paragraph) if p.strip()]
    pieces = []
    for p in parts:
        pieces.extend(_split_fused_lines(p) if _DISCLAIMER.search(p) else [p])
    return " ".join(p for p in pieces if not _DISCLAIMER.search(p))


def clean_email_body(body: str, max_chars: int = 3000) -> str:
    """Remove quoted email history, signatures and disclaimers to keep input focused."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    # Bound regex work before the expensive patterns below: _DISCLAIMER uses
    # [^.]{0,60/80/100} alternations whose cost grows with input length, and only
    # max_chars are ever returned. Truncate lines too so one MB-long line cannot
    # dominate matching.
    if len(text) > max_chars * 4:
        text = text[:max_chars * 4]
    lines = []
    src = text.split("\n")
    for i, line in enumerate(src):
        if any(p.match(line) for p in _QUOTE_HEADERS) and lines:
            break
        if (lines and _HEADER_FROM_NAME.match(line) and i + 1 < len(src)
                and _HEADER_NEXT.match(src[i + 1])):
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
        n = len(lines[i].strip())
        if (n <= 40 and any(p.match(lines[i]) for p in _SIGNATURE_MARKERS)) or (
                n <= 60 and _DEVICE_FOOTER.match(lines[i])):
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
