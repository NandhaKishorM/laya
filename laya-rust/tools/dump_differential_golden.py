"""Record a seeded, generated differential corpus for language detection, routing and email.

Where `dump_routing_golden.py` records hand-written probes, this script *generates* thousands of
inputs from fragments of the vocabulary the Python modules themselves use, imported live at generation
time and sorted for determinism (every stop list, the shared / English-only / English-collision /
Nordic-overlap / short-Swedish word sets and the non-English diacritics of `laya.lang`, a sample of
every script range, the device-footer words of `laya.email`, plus the quote/sign-off/footer/disclaimer
wording of `laya.email`), runs them through the real `laya.lang`, `laya.router` and `laya.email`, and writes
what Python answered to `tests/golden/routing/differential_probe.json`. A port replays every case
and must agree exactly. The file is an output of this script only; never edit it by hand.

SDK-neutral: only the standard library and `laya` are used, and the only SDK-specific line is
OUT_DEFAULT. Deterministic: one seeded `random.Random`, no clock, no environment.

Unicode version: characters that are unassigned (`Cn`) in the Unicode database of the Python that
runs this script are never generated. A newer database (the Rust crates', for instance) may have
assigned them since, which is a version difference between runtimes and not a behaviour difference.

    python laya-rust/tools/dump_differential_golden.py [--out FILE]
"""
import argparse
import json
import os
import random
import sys
import unicodedata

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, REPO)

from laya.email import clean_email_body, email_state  # noqa: E402
from laya.email import _DEVICE  # noqa: E402
from laya.lang import (  # noqa: E402
    _EN_COLLISION_WORDS, _EN_ONLY_WORDS, _NON_EN_DIACRITICS, _NORDIC_OVERLAP_WORDS, _SCRIPT_RANGES,
    _SHARED_WORDS, _SHORT_SWEDISH_WORDS, _STOP, analyse, state_text,
)
from laya.router import Router  # noqa: E402

OUT_DEFAULT = os.path.join(REPO, "laya-rust", "tests", "golden", "routing", "differential_probe.json")

SEED = 20261019
N_STATES = 800
N_EMAILS = 600
N_EMAIL_STATES = 150
EMAIL_BUDGETS = [3000, 0, -1, -10, 5, 37, 200]
LANG_ARGS = ["en", "en_US", "en_US.UTF-8", " EN ", "C", "POSIX", "und", "zxx", "mul", "", ".", "fr",
             "pt-BR", "xx", "x\u200by"]
GUESS_CODES = ["en", "en-US", "pt", "ro", "C", "und", "", " fr ", "zxx", "de_DE.UTF-8"]

rng = random.Random(SEED)


def keep(text):
    """Drop characters that are unassigned (or surrogates) in this Python's Unicode database."""
    return "".join(c for c in text if unicodedata.category(c) not in ("Cn", "Cs"))


def pick(seq):
    return seq[rng.randrange(len(seq))]


# ------------------------------------------------------------------------------ state fragments
STOP_LANGS = sorted(_STOP)
STOP_WORDS = {k: sorted(v) for k, v in _STOP.items()}
# The other word sets and character tables `laya.lang` decides with, read live: a change to any of
# them changes the corpus (and therefore the golden), so a port that missed it is caught.
SPECIAL_WORD_SETS = [sorted(_EN_ONLY_WORDS), sorted(_SHARED_WORDS), sorted(_EN_COLLISION_WORDS),
                     sorted(_NORDIC_OVERLAP_WORDS), sorted(_SHORT_SWEDISH_WORDS)]
DIACRITICS = sorted(_NON_EN_DIACRITICS)
# `laya.email._DEVICE` is a regex alternation of plain words (and `e-?mail`); take the literal ones.
DEVICE_WORDS = sorted(w for w in _DEVICE.split("|") if w.isalpha())
ENGLISH = ["please", "refund", "invoice", "account", "locked", "order", "delivery", "password",
           "reset", "charged", "twice", "support", "ticket", "update", "my", "your", "help",
           "urgent", "payment", "confirmation", "shipping", "address", "cancel", "subscription"]
LOANWORDS = ["café", "résumé", "naïve", "façade", "jalapeño", "piñata", "über", "doppelgänger",
             "fiancée", "señor", "smörgåsbord", "déjà", "vu", "crème", "brûlée"]
ACRONYMS = ["NASA", "IBM", "API", "HTTP", "USB", "PDF", "FAQ", "ASAP", "CEO", "VPN"]
IDENTIFIERS = ["os.path", "user@example.com", "https://example.com/a/b?c=d", "foo_bar", "my-var.name",
               "C:\\temp\\file.txt", "node_modules/.bin/x", "v1.2.3", "a.b.c.d", "SELECT * FROM t;",
               "if (x == 1) { y(); }", "arr[0] = f(1);", "#include <stdio.h>", "def f(x): return x"]
EMOJI = ["\U0001F600", "\U0001F680", "\U0001F44D", "\u2764\uFE0F", "\U0001F1E7\U0001F1F7"]
WHITESPACE = [" ", " ", " ", "\u00a0", "\x1c", "\x1d", "\x1e", "\x1f", "\x85", "\u3000", "\t", "\u2003",
              "\u2028"]
# Characters with special case-folding or word-class behaviour: dotted/dotless i, long s, the Kelvin
# and Angstrom signs, titlecase digraphs, German sharp s, final sigma, a superscript digit and a
# connector, a circled letter, a Roman numeral, a vulgar fraction, a modifier letter, a mark.
SPECIAL = ["\u0130", "\u0131", "\u017f", "\u212a", "\u212b", "\u01c5", "\u00df", "\u1e9e", "\u03c2",
           "\u03a3", "\u00b2", "\u00b3", "\u00bd", "\u203f", "\u2040", "\u24b6", "\u2160", "\u02b0",
           "\u0149", "\u00b5", "\u03bc", "\u0345", "\u200d", "\u2460", "\u0660", "\uff10"]
SWAPS = {"i": ["\u0131", "\u0130", "I"], "I": ["\u0130", "\u0131", "i"], "s": ["\u017f", "S"],
         "S": ["\u017f", "s"], "k": ["\u212a", "K"], "K": ["\u212a", "k"], "a": ["\u212b", "A"],
         "u": ["\u00b5"], "2": ["\u00b2", "\u0662", "\uff12"]}
COMBINING = ["\u0301", "\u0308", "\u0323", "\u0327", "\u0345", "\u200d", "\u200c"]


def script_word(ranges):
    cps = []
    for lo, hi in ranges:
        cps.extend(range(lo, hi + 1))
    word = []
    for _ in range(rng.randint(2, 9)):
        c = chr(pick(cps))
        if unicodedata.category(c)[0] in "LMN":
            word.append(c)
    return "".join(word) or chr(ranges[0][0])


def devanagari_word():
    consonants = [chr(c) for c in range(0x0915, 0x093A)]
    matras = [chr(c) for c in list(range(0x093E, 0x094D)) + [0x0902, 0x0903]]
    out = []
    for _ in range(rng.randint(2, 6)):
        out.append(pick(consonants))
        if rng.random() < 0.7:
            out.append(pick(matras))
        if rng.random() < 0.15:
            out.append("\u094d")
    return "".join(out)


def fullwidth_word():
    return "".join(chr(0xFF21 + rng.randrange(26)) if rng.random() < 0.5 else chr(0xFF41 + rng.randrange(26))
                   for _ in range(rng.randint(2, 7)))


def ipa_word():
    return "".join(chr(0x250 + rng.randrange(0x60)) for _ in range(rng.randint(2, 6)))


def astral_word():
    base = pick([0x1D400, 0x1D434, 0x1D5A0, 0x10400, 0x10428])
    return "".join(chr(base + rng.randrange(26)) for _ in range(rng.randint(2, 6)))


def special_word():
    """A word with some letters swapped for look-alikes that fold or classify differently."""
    w = pick(ENGLISH + LOANWORDS + sorted(sum((STOP_WORDS[k][:6] for k in STOP_LANGS), [])))
    out = []
    for ch in w:
        out.append(pick(SWAPS[ch]) if ch in SWAPS and rng.random() < 0.4 else ch)
    w = "".join(out)
    if rng.random() < 0.4:
        w = w + pick(SPECIAL) if rng.random() < 0.5 else pick(SPECIAL) + w
    return w


def accent_word():
    w = pick(ENGLISH + LOANWORDS)
    return w + pick(COMBINING) if rng.random() < 0.5 else w[:1] + pick(COMBINING) + w[1:]


def diacritic_word():
    """A live stop word with a non-English diacritic spliced in, so the diacritic rate moves."""
    w = pick(STOP_WORDS[pick(STOP_LANGS)])
    i = rng.randrange(len(w) + 1)
    return w[:i] + pick(DIACRITICS) + w[i:]


def fragment():
    r = rng.random()
    if r < 0.30:
        sub = rng.random()
        if sub < 0.20:
            return pick(pick(SPECIAL_WORD_SETS))
        if sub < 0.26:
            return diacritic_word()
        return pick(STOP_WORDS[pick(STOP_LANGS)])
    if r < 0.45:
        return pick(ENGLISH)
    if r < 0.50:
        return pick(LOANWORDS)
    if r < 0.60:
        return script_word(pick(_SCRIPT_RANGES)[1])
    if r < 0.63:
        return devanagari_word()
    if r < 0.66:
        return fullwidth_word()
    if r < 0.68:
        return ipa_word()
    if r < 0.71:
        return astral_word()
    if r < 0.75:
        return pick(ACRONYMS)
    if r < 0.80:
        return pick(IDENTIFIERS)
    if r < 0.84:
        return accent_word()
    if r < 0.87:
        return pick(EMOJI)
    if r < 0.90:
        return pick(ENGLISH).upper()
    if r < 0.93:
        return pick(ENGLISH) + pick([".", "/", "_", "\\"]) + pick(ENGLISH)
    if r < 0.95:
        return special_word()
    if r < 0.96:
        return str(rng.randrange(100000))
    return pick(["!", "?", ",", ".", ":", ";", "-", "(", ")", "\"", "'"])


def line():
    parts = [fragment() for _ in range(rng.randint(1, 12))]
    if rng.random() < 0.08:
        parts = [p.upper() for p in parts]
    out = parts[0]
    for p in parts[1:]:
        out += (pick(WHITESPACE) if rng.random() < 0.25 else " ") + p
    return out


def text():
    return pick(["\n", "\n", "\n\n", "\r\n"]).join(line() for _ in range(rng.randint(1, 6)))


def leaf():
    r = rng.random()
    if r < 0.8:
        return text()
    if r < 0.85:
        return rng.randrange(1000)
    if r < 0.9:
        return rng.random() * 100
    if r < 0.95:
        return pick([None, True, False])
    return ""


def mapping(depth=0):
    d = {}
    for i in range(rng.randint(1, 4)):
        key = pick(["subject", "body", "note", "message", "title", "field%d" % i, "\u4e3b\u9898"])
        r = rng.random()
        if depth < 2 and r < 0.2:
            d[key] = mapping(depth + 1)
        elif depth < 2 and r < 0.3:
            d[key] = [leaf() for _ in range(rng.randint(1, 3))]
        else:
            d[key] = leaf()
    return d


def long_text():
    base = text()
    return (base + " ") * (4100 // max(1, len(base)) + 2)


def make_state():
    r = rng.random()
    if r < 0.50:
        return text()
    if r < 0.53:
        return long_text()
    if r < 0.70:
        return mapping()
    if r < 0.82:
        return [{"role": pick(["user", "assistant"]), "content": text()} if rng.random() < 0.6 else text()
                for _ in range(rng.randint(1, 6))]
    if r < 0.85:
        d = mapping()
        d["note"] = long_text()
        return d
    if r < 0.88:
        return " ".join(pick(STOP_WORDS[pick(STOP_LANGS)]) for _ in range(rng.randint(2, 8)))
    if r < 0.94:
        # short requests built from the English-only / shared / short-Swedish sets: the
        # "Why hasn't my card been delivered?" shape that the English stop list decides
        pool = pick(SPECIAL_WORD_SETS)
        return " ".join(pick(pool) if rng.random() < 0.6 else pick(ENGLISH) for _ in range(rng.randint(1, 7)))
    return line()


def sanitize(state):
    if isinstance(state, str):
        return keep(state)
    if isinstance(state, list):
        return [sanitize(v) for v in state]
    if isinstance(state, dict):
        return {keep(k): sanitize(v) for k, v in state.items()}
    return state


# ------------------------------------------------------------------------------ email fragments
SENTENCES = [
    "Please refund the duplicate charge on invoice 4411.",
    "I cannot log in to my account since yesterday.",
    "Preciso de ajuda com o meu pedido.",
    "Necesito ayuda con mi cuenta.",
    "Je demande un remboursement pour ma commande.",
    "Is this confidential?",
    "Confidential: I need a refund.",
    "From: my side the integration works, but please refund it.",
    "Thanks for the quick reply, but I need a refund.",
    "Antes de imprimir o boleto, confira o valor.",
    "Preciso do contrato confidencial.",
    "Je veux le contrat confidentiel.",
    "De: 10/09 a 15/09 estarei fora.",
    "Em resposta ao que voc\u00ea escreveu: n\u00e3o.",
    "Hello team, the order arrived broken.",
]
NAMES = ["Jane", "Jane Smith", "Maria", "\u0141ukasz", "\u0414\u043c\u0438\u0442\u0440\u0438\u0439", "\u5c71\u7530",
         "Jose\u0301", "\u017caneta", "john", "\U00010400\U00010428", "\u0301Jose", "Marie Dupont", "Jo\u00e3o"]
ATTR = [
    "On Mon, Jan 1, 2024 at 10:00 AM Jane Doe <jane@example.com> wrote:",
    "On Mon, Jan 1, 2024, Jane wrote:",
    "On someone wrote:",
    "Em 10/09/2024 \u00e0s 10:00, Jo\u00e3o <joao@example.com> escreveu:",
    "Em resposta ao que voc\u00ea escreveu:",
    "El 10/09/2024 Juan escribi\u00f3:",
    "El mi\u00e9rcoles Juan escribi\u00f3:",
    "Le 10/09/2024 \u00e0 10:00, Jean a \u00e9crit :",
    "Le 10/09 Jean a ecrit:",
    "Le rapport que vous avez \u00e9crit :",
    "-----Original Message-----",
    "----- Forwarded Message -----",
    "-----Mensagem original-----",
    "-----Mensaje reenviado-----",
    "-----Message d'origine-----",
    "________________",
    "From: jane@example.com",
    "From: Jane <jane@example.com>",
    "De: maria@example.com",
    "De : Marie <marie@example.com>",
    "> quoted line",
    "  > indented quote",
]
WRAPPED = [("On 10/09 Jane Smith", "jane@example.com> wrote:"),
           ("Em 10/09 Maria", "maria@example.com> escreveu:"),
           ("Le 10/09 Jean", "jean@example.com> a \u00e9crit :"),
           ("El 10/09 Juan", "juan@example.com> escribi\u00f3:"),
           ("On Monday Jane", "jane@example.com> wrote:")]
HEADER_PAIRS = [("From: Jane Smith", "Sent: Monday"), ("From: Jane Smith", "Date: Monday 12 January 2024"),
                ("De: Maria Souza", "Enviado: segunda-feira"), ("De: Maria Souza", "Data: 10/09/2024"),
                ("De : Marie Dupont", "Envoy\u00e9 : lundi"), ("From: Jane Smith", "Subject: hello"),
                ("De: 10/09", "Para: 15/09"), ("De: Pedro", "Fecha: 10 de enero de 2024")]
SIGNOFFS = ["Best regards,", "Kind regards", "Warmest regards,", "Many thanks", "Thanks,", "Thank you",
            "Cheers", "Sincerely,", "Regards and wishes", "Best and regards", "Thanks again", "Thanks a lot",
            "Thanks for the reply", "Thanks in advance", "Atenciosamente,", "Obrigado", "Muito obrigada!",
            "Com os melhores cumprimentos", "Sauda\u00e7\u00f5es", "Saludos cordiales,",
            "Muchas gracias de antemano", "Cordialement,", "Bien cordialement", "Merci d'avance",
            "Merci pour votre aide, mais", "Salutations distingu\u00e9es", "Bonne journ\u00e9e",
            "Bien \u00e0 vous", "--", "-- ", "Gracias", "Um abra\u00e7o", "Att"]
DEVICE = ["Sent from my iPhone", "Sent from my iPad", "Enviado do meu celular",
          "Enviado do meu smartphone Samsung Galaxy.", "Envoy\u00e9 depuis mon iPhone",
          "Envoy\u00e9 de mon iPad", "Get Outlook for iOS", "Obter o Outlook para Android",
          "Sent from Yahoo Mail for Android", "Sent from my Samsung Galaxy smartphone",
          "Enviado desde mi M\u00f3vil", "Sent from my device using Mail 2", "Sent from my phone"]
DISCLAIMERS = [
    "This email is confidential and intended solely for the named addressee.",
    "This message and any attachments are confidential and may be privileged.",
    "The information in this e-mail is confidential and intended solely for the recipient.",
    "If you have received this email in error, please delete it.",
    "If you received this message in error, notify the sender.",
    "Esta mensagem \u00e9 confidencial e destinada exclusivamente ao destinat\u00e1rio.",
    "Se voc\u00ea recebeu esta mensagem por engano, apague-a.",
    "Antes de imprimir, pense no meio ambiente.",
    "Pense no meio ambiente antes de imprimir.",
    "Este mensaje es confidencial y de uso exclusivo del destinatario.",
    "Si ha recibido este mensaje por error, b\u00f3rrelo.",
    "Ce message est confidentiel et r\u00e9serv\u00e9 au destinataire.",
    "Si vous avez re\u00e7u ce message par erreur, supprimez-le.",
    "Usage exclusif du destinataire.",
    "The email is confidential and intended solely for you.",
]
NEIGHBOURS = ["\u0130", "\u0131", "\u017f", "\u212a", "\u00bd", "\u2040", "\u00b2", "\u0301", "\u203f", "\u200d", "\u24b6", "\U0001D400", "\u2160", "_", "1"]


def vary(s):
    """Case, dotted/dotless i, odd whitespace and neighbouring-symbol variations of a marker line."""
    upper_lower = rng.random() < 0.15
    out = []
    for ch in s:
        x = ch
        if upper_lower:
            x = ch.upper() if rng.random() < 0.5 else ch.lower()
        if ch in SWAPS and rng.random() < 0.12:
            x = pick(SWAPS[ch])
        if ch == " " and rng.random() < 0.06:
            x = pick(["\u00a0", "\x1f", "\u3000", "\x85", "\t"])
        out.append(x)
    s = "".join(out)
    r2 = rng.random()
    if r2 < 0.08:
        s = pick(NEIGHBOURS) + s
    elif r2 < 0.16:
        i = s.find(" ", 1)
        if i > 0:
            s = s[:i] + pick(NEIGHBOURS) + s[i:]
    return s


def device_footer():
    w = pick(DEVICE_WORDS)
    return pick(["Sent from my %s", "Enviado do meu %s", "Envoyé depuis mon %s", "Get %s for iOS",
                 "Sent from %s", "%s"]) % w


def email_block():
    r = rng.random()
    if r < 0.30:
        return [vary(pick(ATTR))] + [pick(SENTENCES) for _ in range(rng.randint(0, 2))]
    if r < 0.38:
        a, b = pick(WRAPPED)
        return [vary(a), vary(b), pick(SENTENCES)]
    if r < 0.50:
        a, b = pick(HEADER_PAIRS)
        return [vary(a), vary(b), pick(SENTENCES)]
    if r < 0.68:
        return [vary(pick(SIGNOFFS))] + ([pick(NAMES)] if rng.random() < 0.7 else [])
    if r < 0.78:
        return [vary(pick(DEVICE) if rng.random() < 0.6 else device_footer())]
    if r < 0.92:
        return [vary(pick(DISCLAIMERS))]
    return ["> " + pick(SENTENCES), ">> " + pick(SENTENCES)]


def email_body():
    lines = []
    for _ in range(rng.randint(1, 4)):
        lines.append(pick(SENTENCES) if rng.random() < 0.8 else text()[:80])
        if rng.random() < 0.4:
            lines.append("")
    if rng.random() < 0.15:  # a footer fused to a request line, or in its own paragraph
        lines.append(pick(DISCLAIMERS))
    for _ in range(rng.randint(0, 3)):
        lines.extend(email_block())
        if rng.random() < 0.3:
            lines.append("")
    body = pick(["\n", "\n", "\n", "\r\n", "\r", "\\n"]).join(lines)
    if rng.random() < 0.05:
        body = body + "\n" + "Please help me. " * 400
    return keep(body)


# ------------------------------------------------------------------------------ lower() cases
# Characters whose lower-casing is context-dependent or one-to-many: Greek capital sigma (final
# sigma), dotted capital I, combining marks and apostrophes (case-ignorable), the Kelvin and
# Angstrom signs, sharp s, a titlecase digraph, plus plain letters and separators.
LOWER_ALPHABET = ["\u03a3", "\u03c3", "\u03c2", "\u0391", "\u03b1", "A", "a", "\u0130", "\u212a", "\u212b",
                  "\u1e9e", "\u00df", "\u01c5", "\u0301", "\u0308", "'", ".", " ", "-", "1", "\u00b7",
                  "\u2019", "\u0345", "\u200d", "\n", "\u03a9", "\u03bf"]


def build_lower(count=600):
    out = []
    for text in ["\u03a3", "\u0391\u03a3", "\u0391\u03a3 \u0391", "\u039f\u0394\u039f\u03a3", "\u03a3\u0391",
                 "\u0391\u0301\u03a3", "\u0391'\u03a3", "\u0391.\u03a3", "\u0391\u03a3'", "\u0391\u03a3'\u0391",
                 "\u0391 \u03a3", "1\u03a3", "\u0130", "\u212a", "I\u0307", "\u1e9e"]:
        out.append({"text": text, "lower": text.lower()})
    while len(out) < count:
        text = "".join(pick(LOWER_ALPHABET) for _ in range(rng.randint(1, 12)))
        out.append({"text": text, "lower": text.lower()})
    return out


# ------------------------------------------------------------------------------ JSON error cases
JSON_BASES = [
    '{"a": 1, "b": [true, false, null], "c": {"d": "e"}}',
    '[1, 2.5, -3e2, "x\\ny", {"k": []}]',
    '{\n  "name": "caf\u00e9",\n  "values": [1, 2, 3],\n  "nested": {"x": NaN, "y": -Infinity}\n}',
    '{"s": "tab\\there", "t": "\u00e9\u4e2d", "n": 12345678901234567890}',
    '"just a string"',
    '[[[1], [2]], [[3], [4]]]',
    '{"\u00e9": "\u4e2d", "k": 0.5}',
]
JSON_INSERTS = ["{", "}", "[", "]", ",", ":", '"', chr(92), " ", "\n", "\t", "x", "0", "-", ".", "e", "+", "N", "I",
                "\u00e9", "\u4e2d", "'", "nul", "tru", "u"]


def build_json_cases(count=1500):
    texts = [
        "", "  ", "{", '{"a"', '{"a":', "[1,]", "[,1]", "1 2", "{} x", "\ufeff{}", "-", "--1", "01", "[01]",
        # \u escapes at the end of the text: CPython wants one character after the four digits
        *['"a' + chr(92) + "u" + d + tail for d in ("", "0", "00", "004", "0041", "00411")
          for tail in ("", '"')],
        *['"' + chr(92) + "ud835" + chr(92) + "udc00" + tail for tail in ("", "x", '"', 'x"')],
        '"' + chr(92) + "ud835" + chr(92) + "udc0", '"' + chr(92) + "ud835" + chr(92) + "u00",
        '"' + chr(92) + "ud835", '"' + chr(92) + "u0041", '"' + chr(92) + "u0041x",
        '{"temperature": "A' + chr(92) + 'u0041',
        "NaN1", "Infinit", '"abc', '"ab' + chr(92) + 'x"', '"ab' + chr(92) + 'u12"', '"a\nb"', "1" * 4300,
        "-" + "1" * 4300, "[" + "1" * 4301 + "]", "-" + "1" * 4301, "\n\n  x", '{"a":\n  [1,\n  2 3]}',
    ]
    texts += JSON_BASES
    while len(texts) < count:
        base = pick(JSON_BASES)
        kind = rng.randrange(4)
        i = rng.randrange(len(base) + 1)
        if kind == 0:
            text = base[:i] + base[i + 1:]
        elif kind == 1:
            text = base[:i] + pick(JSON_INSERTS) + base[i:]
        elif kind == 2:
            text = base[:i]
        else:
            text = base[:i] + pick(JSON_INSERTS) + base[i + 1:]
        texts.append(text)
    out = []
    for text in texts:
        try:
            json.loads(text)
            error = None
        except ValueError as e:  # JSONDecodeError and the integer digit limit are both ValueError
            error = str(e)
        out.append({"text": text, "error": error})
    return out


# ------------------------------------------------------------------------------ build
def decision(router, **kwargs):
    d = router.route(**kwargs)
    return {"model": d["model"], "reason": d["reason"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    plain = Router()
    langs = []
    routes = []
    for i in range(N_STATES):
        state = sanitize(make_state())
        langs.append({"label": "gen/state_%04d" % i, "state": state, "state_text": state_text(state),
                      "analyse": analyse(state)})
        routes.append({"state": i, "kwargs": {}, **decision(plain, state=state)})
        for lang in rng.sample(LANG_ARGS, 2):
            routes.append({"state": i, "kwargs": {"lang": lang}, **decision(plain, state=state, lang=lang)})
        code = pick(GUESS_CODES)
        routes.append({"state": i, "kwargs": {"lang_guess": code},
                       **decision(plain, state=state, lang_guess=code)})
        code = pick(GUESS_CODES)
        routes.append({"state": i, "kwargs": {"router_lang_guess": code},
                       **decision(Router(lang_guess=code), state=state)})
        if i % 7 == 0:
            lang = pick(LANG_ARGS)
            code = pick(GUESS_CODES)
            routes.append({"state": i, "kwargs": {"lang": lang, "lang_guess": code},
                           **decision(plain, state=state, lang=lang, lang_guess=code)})

    clean = []
    for i in range(N_EMAILS):
        body = email_body()
        clean.append({"label": "gen/email_%04d" % i, "body": body,
                      "results": [clean_email_body(body, m) for m in EMAIL_BUDGETS]})
    email_states = []
    for i in range(N_EMAIL_STATES):
        subject = keep(pick(["Locked out", "  Refund request  ", "", "\u00a0 Re: order \u2003", "Fwd: invoice"]))
        body = email_body()
        sender = pick([None, "", "user@example.com"])
        clean_flag = rng.random() < 0.8
        extra = pick([{}, {"priority": "high"}, {"max_chars": 37}, {"max_chars": 0}, {"max_chars": -1},
                      {"max_chars": 200, "ticket": 4411}, {"nullable": None, "kept": "value"}])
        email_states.append({"label": "gen/email_state_%04d" % i, "subject": subject, "body": body,
                             "sender": sender, "clean": clean_flag, "extra": extra,
                             "result": email_state(subject, body, sender, clean_flag, **extra)})

    out = {"seed": SEED, "email_budgets": EMAIL_BUDGETS, "lang": langs, "route": routes,
           "email_clean": clean, "email_state": email_states, "lower": build_lower(),
           "json_errors": build_json_cases()}
    text_out = json.dumps(out, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text_out)
    print("  wrote %-28s %8.1f KB (lang %d, route %d, email_clean %d x %d budgets, email_state %d)" % (
        os.path.basename(args.out), os.path.getsize(args.out) / 1024.0, len(langs), len(routes), len(clean),
        len(EMAIL_BUDGETS), len(email_states)))


if __name__ == "__main__":
    main()
