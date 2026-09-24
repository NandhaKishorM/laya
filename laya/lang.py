"""Dependency-free language/script detection used to route between Laya checkpoints.

Routing only needs one decision: *is this English Latin text, or is it something the English
checkpoint cannot read?* Benchmarks on MASSIVE (14 languages) showed the English checkpoint
collapsing to near-random on non-Latin scripts (Hindi 0.100, Korean 0.103, Swahili 0.103,
Tamil 0.113 at 20 options, where random is 0.050), while holding up far better on Latin-script
languages (French 0.487, Spanish 0.480). So the signal that matters most is *script*, and the
secondary signal is whether Latin text is English.

Script detection is exact. The Latin-script language guess is a stopword/diacritic heuristic and
is explicitly best-effort: pass an explicit model or `lang=` when you already know the language.
"""
import re
import unicodedata
from typing import Dict, List, Optional, Union

# Unicode blocks that the English (ModernBERT-large, 50k English BPE) checkpoint cannot read.
_SCRIPT_RANGES = [
    ("greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("cyrillic", ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))),
    ("armenian", ((0x0530, 0x058F),)),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("bengali", ((0x0980, 0x09FF),)),
    ("gurmukhi", ((0x0A00, 0x0A7F),)),
    ("gujarati", ((0x0A80, 0x0AFF),)),
    ("oriya", ((0x0B00, 0x0B7F),)),
    ("tamil", ((0x0B80, 0x0BFF),)),
    ("telugu", ((0x0C00, 0x0C7F),)),
    ("kannada", ((0x0C80, 0x0CFF),)),
    ("malayalam", ((0x0D00, 0x0D7F),)),
    ("sinhala", ((0x0D80, 0x0DFF),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("lao", ((0x0E80, 0x0EFF),)),
    ("tibetan", ((0x0F00, 0x0FFF),)),
    ("myanmar", ((0x1000, 0x109F),)),
    ("georgian", ((0x10A0, 0x10FF),)),
    ("ethiopic", ((0x1200, 0x137F),)),
    ("khmer", ((0x1780, 0x17FF),)),
    ("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF))),
    ("han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
]

# Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
# weighted and a margin is required before calling something non-English.
#
# The Romance lists (fr/es/pt/it) deliberately carry the *unaccented* function words as well as the
# accented ones. A state that lost its accents -- mail clients, ticket systems and any pipeline that
# normalises to ASCII strip them -- keeps no diacritic rate for the non-English signal to read, so
# `la`, `un`, `y`, `e`, `et`, `deux` and friends are the only evidence left. With a list of mostly
# accented words such a state produced one hit or none, fell under the two-hit margin below, and was
# handed to the English checkpoint as undecided-but-not-non-English text (see #172 and #54).
_STOP = {
    "en": {"the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
           "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
           "will", "can", "would", "there", "their", "what", "which", "please", "we", "i"},
    "fr": {"le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
           "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
           "et", "du", "au", "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton",
           "ma", "ta", "sa", "mes", "tes", "ses", "ces", "deux", "trois", "très", "bien",
           "tout", "tous", "toute", "fait", "veux", "veut", "peux", "peut", "dois", "doit",
           "merci", "bonjour", "jour", "jours", "mois", "fois", "quand", "comment", "pourquoi",
           "alors", "donc"},
    "de": {"der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
           "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
           "ich", "wir", "mir", "mich", "dir", "dich", "uns", "mein", "meine", "meinen",
           "meinem", "meiner", "diese", "dieser", "diesen", "dieses", "einen", "einem", "einer",
           "wie", "wo", "wann", "welche", "im", "zum", "zur", "aus", "bei", "nach", "noch", "bitte",
           "heute", "jetzt", "kann", "kannst", "habe", "gibt", "wird",
           # shared with English on purpose: counted for English alone, they outvoted short German
           "in", "was"},
    "es": {"el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
           "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
           # `de`/`en` are Spanish too, but they are common English tokens as well (`de facto`,
           # `en-US`, `en route`, `Rio de Janeiro`), and a state of those alone already carries
           # no English function word for the margin below to weigh them against, so they stay out.
           "la", "un", "y", "al", "lo", "le", "les", "su", "mi", "tu", "nos",
           "ni", "dos", "tres", "fue", "fueron", "ser", "tiene", "tienen", "tengo", "puede",
           "pueden", "quiero", "necesito", "hemos", "han", "sobre", "entre", "cuando", "donde",
           "porque", "aunque", "también", "ya", "eso", "esto", "esa", "ese", "nada", "algo",
           "aquí", "hoy", "gracias"},
    "pt": {"os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
           "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
           # `no` is Portuguese too, and among the most frequent words it has; it is also one of the
           # most frequent English words, so it stays out and short Portuguese states that lean on it
           # alone are left to the diacritic rate, as before.
           "o", "e", "na", "nas", "nos", "ao", "aos", "por", "foi", "era", "ser", "sou",
           "tem", "tenho", "pode", "podem", "quero", "preciso", "eu", "meu", "minha", "seu",
           "sua", "isso", "isto", "aqui", "ali", "como", "quando", "onde", "porque", "mais",
           "já", "ainda", "agora", "hoje", "ontem", "dois", "três", "tudo", "nada", "obrigado",
           "olá",
           # Brazilian support text: `você` and the unaccented `nao`/`voce`/`sao`/`ja` that a stripped
           # state keeps (#172), and the chat abbreviations `vc`/`pra`. Without them "Voce pode me
           # mandar a nota fiscal?" matched one word and went to the English checkpoint, which on
           # `pt` reports 0.97 mean confidence at 0.47 accuracy. `ate`, `bom`, `sim` and `cade` stay
           # out: each is an English token too (ate, BOM, SIM, Cade).
           "você", "vocês", "voce", "voces", "vc", "vcs", "nao", "sao", "ja", "até", "tá", "pra",
           "gostaria", "obrigada", "também", "tambem", "estou", "estamos", "meus", "minhas",
           "nosso", "nossa", "consigo", "cadê", "boa", "tarde", "noite",
           # the words a ticket keeps once the jargon is English ("Deu erro 500 no endpoint de login
           # depois do update"): time and person words plus the past tenses a bug report is told in
           "depois", "antes", "então", "entao", "ninguém", "ninguem", "alguém", "alguem", "nenhum",
           "nenhuma", "estava", "ficou", "fiz", "deu"},
    "it": {"il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
           "questo", "questa", "anche", "come", "più", "sono", "nella", "alla",
           "la", "le", "un", "uno", "una", "e", "ed", "o", "da", "su", "tra", "fra", "mi",
           "ci", "ne", "ho", "hai", "ha", "abbiamo", "avete", "hanno", "era", "stato", "stata",
           "devo", "deve", "devono", "voglio", "vorrei", "mio", "mia", "tuo", "sua", "quando",
           "dove", "perche", "molto", "poco", "sempre", "mai", "già", "ancora", "adesso", "oggi",
           "ieri", "grazie", "ciao", "scusa",
           # the articulated prepositions: Italian-only words, which is what lets a state made of
           # shared articles (`la fattura`) still name the language rather than stay undecided
           "nel", "nell", "negli", "sul", "sulla", "sulle", "dal", "dalla", "dallo", "dagli", "dei",
           "delle", "dello", "degli", "agli", "alle", "col"},
    "nl": {"het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
           "door", "maar", "ook", "worden", "deze", "naar", "wordt"},
    # Romanian words that its Romance neighbours do not share, so adding `ro` cannot steal a
    # French/Spanish/Italian/Portuguese state: `la`, `o`, `un`, `de`, `pe`, `ca` are deliberately
    # left out for that reason, and the diacritic signal below carries the rest.
    "ro": {"și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
           "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
           "acesta", "aceasta", "mi", "ți", "vă", "nu"},
    # Romanized Bangla ("Banglish"): how Bangla is typed in chats, tickets and email when no Bengali
    # keyboard is at hand. It has no diacritics, so without a list it read as undecided-but-English
    # and went to the English checkpoint, which scores 0.08 on Bangla at 0.94 confidence. Spelling
    # is not standardised, so the common variants are listed (`bhalo`/`valo`, `korchi`/`korsi`).
    # Left out on purpose: frequent Bangla words that are also English words -- `ache` (is),
    # `are` (is there), `to` (so), `take` (to him), `age` (before), `pore` (later), `mane`
    # (meaning), `din` (give), `sob` (all), `tar` (his), `dao` (give), `eta` (this; ETA) --
    # ordinary words of a neighbouring language (`ora`, `nei`, `vai`), and words another list
    # already claims (`na`, `o`, `e`, `je`, `ta`, `por`, `hoy`), so adding `bn` cannot move a
    # state of any other language.
    "bn": {"ami", "amar", "amake", "amra", "amader", "apni", "apnar", "apnake", "apnara",
           "tumi", "tomar", "tomake", "tomra", "tader", "ota", "eita", "oita",
           "ekta", "ei", "oi", "ki", "keno", "kivabe", "kibhabe", "kothay", "kokhon", "kobe",
           "koto", "kintu", "jodi", "tahole", "ar", "theke", "jonno", "sathe", "shathe", "diye",
           "niye", "moddhe", "kore", "korte", "korchi", "korsi", "korbo", "korechi", "koreche",
           "korun", "koren", "korlam", "hobe", "hoyeche", "hoise", "hocche", "hoyni",
           "chai", "chaina", "lagbe", "parchi", "parbo", "parchina", "peyechi", "paini",
           "dite", "dilam", "diyechi", "nai", "khub", "onek", "ekhon", "akhon", "ekhono",
           "abar", "ekbar", "duibar", "ajke", "kalke", "taka", "bhalo", "valo", "kharap",
           "shomossa", "somossa", "dhonnobad", "bhai", "shob", "keu", "kichu", "bolte", "bolun",
           "parben", "asbe", "jabe", "pabo", "ferot", "dorkar", "hoye", "geche", "gese"},

    # "her", "ne", "men", "de" collide with English, French and Romanian, and "ki" with the
    # romanized Bangla list, so they are left out.
    "az": {"və", "ve", "bir", "bu", "üçün", "ucun", "ilə", "ile", "olan", "olub", "olmasa",
           "var", "yox", "yoxdur", "mən", "sən", "biz", "siz", "onlar", "daha", "çox", "cox",
           "hər", "nə", "kimi", "görə", "sonra", "əgər", "eger", "deyil", "lakin", "amma",
           "ancaq", "artıq", "artiq", "də", "isə", "həm", "yalnız", "yalniz"},
}
# Letters that ordinary English does not use. This is the signal that catches a Latin-script
# language we hold no stopwords for at all (Romanian, Polish, Czech, Turkish, Baltic, ...),
# which is the difference between routing it to the multilingual checkpoint and silently
# handing it to the English one.
_NON_EN_DIACRITICS = set(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ"          # Western European
    "ăâîșțşţ"                                   # Romanian
    "ąćęłńśźż"                                  # Polish
    "čďěňřšťůž"                                 # Czech / Slovak
    "őű"                                        # Hungarian
    "ğı"                                        # Turkish (text is lowercased before matching)
    "āēģīķļņūž"                                 # Baltic
    "đ"                                         # Serbo-Croatian / Vietnamese
    "ə"                                         # Azerbaijani
)
# Words that more than one list claims. `la`, `un`, `e`, `que`, `una` and friends are function words
# of several of these languages at once, so matching one says "not English" without saying *which*
# language: a shared word may not name a winner by itself (Romanian text was reported as French that
# way), though it still counts toward the total of a language that also matched a word of its own.
_SHARED_WORDS = {w for w in {word for words in _STOP.values() for word in words}
                 if sum(w in words for words in _STOP.values()) > 1}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# A token whose dot or @ joins word characters is an identifier, not prose: `github.com`,
# `user@acme.com`, `v1.2.3`, `U.S.A.`. `_WORD` splits them into pieces that collide with real
# function words -- `com` is Portuguese for "with", `o` is its article, `e` is Italian "e" -- so a
# state that was mostly links scored a language it does not contain, and two domains were enough to
# cross the margin below. A sentence-final period (`arrivato.`) keeps its word: the pattern needs
# word characters on both sides of the dot.
_IDENTIFIER = re.compile(r"[\w-]*(?:[.@][\w-]+)+", re.UNICODE)


def _iter_text(state: Union[str, dict, list, None], _depth: int = 0) -> List[str]:
    """Collect the string leaves of a state (str / dict / list), so detection sees real content."""
    if _depth > 6 or state is None:
        return []
    if isinstance(state, str):
        return [state]
    if isinstance(state, dict):
        out = []
        for v in state.values():
            out.extend(_iter_text(v, _depth + 1))
        return out
    if isinstance(state, (list, tuple)):
        out = []
        for v in state:
            out.extend(_iter_text(v, _depth + 1))
        return out
    return []


def state_text(state: Union[str, dict, list, None], max_chars: int = 4000) -> str:
    """Flatten a state into the text used for detection (keys are ignored: they are usually English)."""
    parts: List[str] = []
    budget = max_chars
    for leaf in _iter_text(state):
        if budget <= 0:
            break
        if len(leaf) > budget:
            parts.append(leaf[:budget])
            break
        parts.append(leaf)
        # Account for the joining space without materializing the full text first.
        budget -= len(leaf) + 1
    return " ".join(parts)[:max_chars]


def detect_script(text: str) -> str:
    """Dominant script of `text`: 'latin', 'han', 'devanagari', ... or 'unknown' if there are no letters."""
    counts: Dict[str, int] = {}
    latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x02B0 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
            latin += 1                                   # Latin, IPA Extensions, Ext-Additional, fullwidth
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            # An alphabetic character no range claims used to be counted nowhere, so text
            # written only in an unlisted script produced a total of 0 and was reported as
            # "unknown" -- and `analyse` treats "unknown" as English, sending it to the
            # checkpoint that has no tokens for it. 68% of Unicode's alphabetic codepoints
            # are outside _SCRIPT_RANGES (the CJK extensions, kana supplements, bopomofo,
            # halfwidth katakana, and dozens of smaller scripts), so enumerating them all is
            # not maintainable. Counting the remainder under "other" keeps them visible and
            # non-Latin, which is the safe direction: an unreadable script must not be
            # handed to the English checkpoint.
            counts["other"] = counts.get("other", 0) + 1
    counts["latin"] = latin
    total = sum(counts.values())
    if total == 0:
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def script_profile(text: str) -> Dict[str, float]:
    """Fraction of alphabetic characters belonging to each detected script."""
    counts: Dict[str, int] = {"latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x02B0 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
            counts["latin"] += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    total = sum(counts.values())
    if not total:
        return {}
    return {k: v / total for k, v in counts.items() if v}


# A diacritic rate above this is taken as evidence the text is not English, even when no
# stopword list matches it.
NON_EN_DIACRITIC_RATE = 0.02

# Non-Latin text is not for the English checkpoint even when Latin letters are the plurality: a
# brand name or order code outvotes the CJK request around it letter for letter, though one CJK
# character carries far more than a letter. A short message needs a large share to count; a long
# payload (ticket fields, English agent turns) dilutes the share, so there a sentence's worth of
# letters counts too.
NON_LATIN_FRACTION = 0.2
NON_LATIN_MIN_FRACTION = 0.1
NON_LATIN_MIN_LETTERS = 10


def _script_of(ch: str) -> Optional[str]:
    """The named non-Latin script of one letter, or None for Latin and for unclaimed letters."""
    cp = ord(ch)
    if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return None
    for name, ranges in _SCRIPT_RANGES:
        if any(lo <= cp <= hi for lo, hi in ranges):
            return name
    return None


def _non_latin_words(text: str) -> List[str]:
    """Non-Latin runs that read as words rather than as annotation inside English prose.

    English prose carries three kinds of non-Latin letters that are not a request written in
    another script, and each is excluded here: a symbol (`Set α to 0.05`, one letter), a proper
    name (`Дмитрий Петрович Савицкий`, capitalised), and a pronunciation (`[vlɐˈdʲimʲɪr]`, which
    no script range claims). A combining mark belongs to the letter before it and never splits a
    word, so `Влади́мир` stays one capitalised name rather than becoming `Влади` + `мир`.
    """
    runs, cur, script = [], "", None
    for ch in text:
        if unicodedata.combining(ch):
            continue
        s = _script_of(ch)
        if s is not None and s == script:
            cur += ch
            continue
        if cur:
            runs.append(cur)
        cur, script = (ch, s) if s is not None else ("", None)
    if cur:
        runs.append(cur)
    return [w for w in runs if len(w) >= 2 and not w[0].isupper()]


def latin_profile(text: str) -> Dict[str, object]:
    """Evidence behind the Latin-script language guess.

    Returns `language` (may be None when undecided), `english_hits`, `diacritic_rate` and
    `looks_non_english`. `analyse` needs the evidence and not just the verdict, because
    "undecided" and "English" are different answers and only one of them is safe to send to the
    English checkpoint.

    A non-English language is only named when it matched at least one word that no other list
    claims: shared function words alone (`la`, `e`, `o`) identify no particular language.
    """
    # 'İ'.lower() is 'i' + a combining dot, which matches no word list
    words = _WORD.findall(_IDENTIFIER.sub(" ", text).replace("İ", "i").lower())
    lowered = text.lower()
    diac = sum(1 for ch in lowered if ch in _NON_EN_DIACRITICS)
    diac_rate = diac / max(1, len(lowered))
    non_english = diac_rate >= NON_EN_DIACRITIC_RATE
    if len(words) < 4:
        return {"language": None, "english_hits": 0, "diacritic_rate": diac_rate,
                "looks_non_english": non_english}

    scores = {lg: sum(1 for w in words if w in sw) for lg, sw in _STOP.items()}
    en = scores.get("en", 0)
    # Only a language that matched at least one word no other list claims may be named. Without
    # that condition the top score can be pure overlap -- `la` and `e` in Romanian text made
    # Italian the winner -- which is a guess dressed as a detection. Such a language is dropped
    # from the running rather than merely losing the tie, so a lesser score with real evidence
    # still gets named, and the text stays undecided when no list has any.
    evidenced = {lg: s for lg, s in scores.items()
                 if lg != "en" and any(w not in _SHARED_WORDS for w in set(words) & _STOP[lg])}
    best_lg, best = max(evidenced.items(), key=lambda kv: kv[1], default=(None, 0))

    lang = None
    if best_lg and best >= max(2, en + 2):
        # a non-English language needs a clear margin over English function words
        lang = best_lg
    elif best_lg and non_english and best >= max(2, en):
        # Needs two hits here too. One shared function word ("para" in Turkish text) named Spanish
        # on the strength of the diacritics alone, which is a guess dressed as a detection.
        lang = best_lg
    elif en and not non_english:
        lang = "en"
    return {"language": lang, "english_hits": en, "diacritic_rate": diac_rate,
            "looks_non_english": non_english}


def guess_latin_language(text: str) -> Optional[str]:
    """Best-effort language code for Latin-script text, or None when undecided.

    Scores function-word hits per language and requires the winner to beat English by a margin and
    to have matched at least one word of its own, so ordinary English is never misrouted and a
    word of several languages at once names none of them. Short inputs usually return None on
    purpose.
    """
    return latin_profile(text)["language"]


def analyse(state: Union[str, dict, list, None]) -> Dict[str, object]:
    """Full detection result for a state.

    Returns `script`, `script_profile`, `language` (best effort, may be None),
    `is_english` and `non_latin_fraction`.
    """
    text = state_text(state)
    prof = script_profile(text)
    script = detect_script(text)
    non_latin = round(1.0 - prof.get("latin", 0.0), 4) if prof else 0.0
    n_non_latin = round(non_latin * sum(ch.isalpha() for ch in text))
    if script == "latin" and _non_latin_words(text) and (
            non_latin >= NON_LATIN_FRACTION or (
                non_latin >= NON_LATIN_MIN_FRACTION and n_non_latin >= NON_LATIN_MIN_LETTERS)):
        script = max((s for s in prof if s != "latin"), key=prof.get)
    if script == "unknown":
        return {"script": "unknown", "script_profile": prof, "language": None,
                "is_english": True, "language_undecided": True, "diacritic_rate": 0.0,
                "non_latin_fraction": 0.0}
    if script != "latin":
        return {"script": script, "script_profile": prof, "language": None,
                "is_english": False, "language_undecided": True, "diacritic_rate": 0.0,
                "non_latin_fraction": non_latin}
    prof_lat = latin_profile(text)
    lang = prof_lat["language"]
    # Undecided is not English. Treating it as English sent every Latin-script language we hold no
    # stopwords for to the checkpoint that cannot read it, silently. When nothing identifies the
    # language, non-English letters are enough to prefer the multilingual checkpoint; text with no
    # such letters (including short English) still goes to the English one.
    undecided = lang is None
    english = lang == "en" or (undecided and not prof_lat["looks_non_english"])
    return {"script": "latin", "script_profile": prof, "language": lang,
            "is_english": english, "language_undecided": undecided,
            "diacritic_rate": round(float(prof_lat["diacritic_rate"]), 4),
            "non_latin_fraction": non_latin}


def is_english(state: Union[str, dict, list, None]) -> bool:
    """True when the English checkpoint can be expected to read this state."""
    return bool(analyse(state)["is_english"])
