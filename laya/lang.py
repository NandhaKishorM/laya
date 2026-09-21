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
_STOP = {
    "en": {"the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
           "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
           "will", "can", "would", "there", "their", "what", "which", "please", "we", "i"},
    "fr": {"le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
           "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce"},
    "de": {"der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
           "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber"},
    "es": {"el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
           "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus"},
    "pt": {"os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
           "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela"},
    "it": {"il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
           "questo", "questa", "anche", "come", "più", "sono", "nella", "alla"},
    "nl": {"het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
           "door", "maar", "ook", "worden", "deze", "naar", "wordt"},
    # Romanian words that its Romance neighbours do not share, so adding `ro` cannot steal a
    # French/Spanish/Italian/Portuguese state: `la`, `o`, `un`, `de`, `pe`, `ca` are deliberately
    # left out for that reason, and the diacritic signal below carries the rest.
    "ro": {"și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
           "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
           "acesta", "aceasta", "mi", "ți", "vă", "nu"},
}
# Letters that ordinary English does not use. This is the signal that catches a Latin-script
# language we hold no stopwords for at all (Romanian, Polish, Czech, Turkish, Baltic, ...),
# which is the difference between routing it to the multilingual checkpoint and silently
# handing it to the English one.
# Latin script languages beyond the seven above. Every one of them is a language
# research/results/cpu_51_language_sweep.json already scores higher on the multilingual
# checkpoint than on the English one: Polish +0.27, Turkish +0.23, Latvian +0.22, Hungarian,
# Norwegian, Vietnamese and Azerbaijani +0.20. #42 made undecided Latin text prefer the
# multilingual checkpoint when it carries non-English letters, which covers the accented
# languages. It cannot cover a language written in plain ASCII: Indonesian, Malay, Javanese,
# Swahili and Tagalog have no diacritics to rate, so they still read as English. Naming them
# is what closes that half.
#
# The word lists come from the MASSIVE *train* split: frequent, present in at least eight
# distinct intents so domain specific words drop out, absent from the English top 120, then
# merged with function words written by hand. The test split was never read while building
# them.
_STOP.update({
    "af": {"aan", "af", "asseblief", "dat", "die", "dit", "ek", "en", "enige", "gaan", "het",
           "hierdie", "hoe", "in", "is", "jy", "kan", "lys", "maak", "met", "na", "nie", "nuwe", "om",
           "oor", "op", "pos", "se", "speel", "te", "tyd", "van", "vandag", "vir", "volgende", "waar",
           "wat", "watter", "wil"},
    "az": {"amma", "aç", "bilərsən", "bir", "biz", "bu", "daha", "de", "deyil", "et", "göstər", "gün",
           "günü", "hansı", "haqqında", "harada", "hava", "hər", "ilə", "istəyirəm", "mən", "mənim",
           "mənə", "necə", "növbəti", "nə", "nədir", "olan", "olmasa", "onlar", "oxut", "saat", "siz",
           "son", "səhər", "sən", "var", "və", "yeni", "yox", "zəhmət", "çox", "üçün", "əlavə", "ən"},
    "ca": {"al", "als", "amb", "aquest", "aquesta", "avui", "com", "correu", "de", "del", "el", "els",
           "en", "ha", "hi", "hora", "la", "les", "llista", "meu", "meva", "molt", "més", "no", "per",
           "plau", "posa", "pots", "que", "quin", "quina", "què", "si", "tinc", "un", "una", "us", "és"},
    "cy": {"ac", "am", "angen", "ar", "beth", "ble", "bod", "bore", "chwarae", "dda", "di", "diolch",
           "dydd", "ebost", "ei", "faint", "fi", "fy", "gloch", "gyda", "gyfer", "heddiw", "hefyd",
           "mae", "mi", "nesaf", "newydd", "newyddion", "oes", "os", "pa", "rhestr", "sut", "sy",
           "sydd", "unrhyw", "wedi", "yma", "yn", "yr", "yw"},
    "da": {"af", "at", "blev", "dag", "den", "denne", "der", "det", "du", "eller", "en", "er", "et",
           "for", "fortæl", "fra", "har", "have", "her", "hvad", "hvor", "hvordan", "hvornår", "ikke",
           "jeg", "kan", "klokken", "kunne", "liste", "med", "meget", "men", "mig", "min", "morgen",
           "møde", "noget", "nu", "næste", "og", "også", "om", "på", "sang", "skal", "som", "spil",
           "tak", "til", "ved", "vil", "været"},
    "fi": {"aseta", "ei", "etsi", "että", "haluan", "hyvä", "ja", "jos", "kaikki", "kanssa", "kello",
           "kerro", "kiitos", "kuin", "kuinka", "kun", "laita", "lista", "lisää", "lähetä", "mikä",
           "minua", "minulla", "minulle", "minun", "minä", "mitä", "muistuta", "mutta", "niin", "nyt",
           "näytä", "ole", "olen", "on", "onko", "ovat", "poista", "se", "soita", "tai", "tämä",
           "tämän", "tänään", "voi", "voit", "voitko", "yhtään"},
    "hu": {"az", "csak", "de", "egy", "el", "emailt", "ezt", "fel", "hogy", "hozzá", "hány", "idő",
           "is", "játszd", "kapcsold", "kell", "ki", "kérem", "kérlek", "következő", "küldj", "le",
           "lehet", "lesz", "listát", "ma", "meg", "mennyi", "mi", "mikor", "milyen", "mondd", "most",
           "mutasd", "már", "még", "nekem", "nem", "reggel", "van", "és", "új"},
    "id": {"acara", "ada", "adalah", "akan", "aku", "apa", "apakah", "bagaimana", "baru", "berapa",
           "beri", "berita", "bisa", "bisakah", "buat", "daftar", "dan", "dari", "dengan", "di", "dua",
           "hari", "ingin", "ini", "itu", "jam", "kamu", "ke", "lagu", "lampu", "pada", "pagi",
           "putar", "saya", "tahu", "tentang", "tidak", "tolong", "untuk", "yang"},
    "is": {"af", "að", "dag", "eftir", "ekki", "er", "eru", "frá", "fyrir", "get", "geturðu", "hvar",
           "hvað", "hvaða", "hvenær", "hver", "hvernig", "hversu", "klukkan", "með", "mig", "minn",
           "morgun", "mér", "mínum", "núna", "og", "segðu", "sem", "spila", "takk", "til", "tölvupóst",
           "um", "vinsamlegast", "við", "á", "ég", "í", "þarf", "það"},
    "jv": {"acara", "aku", "ana", "anyar", "apa", "bisa", "dhaptar", "dina", "gawe", "iki", "iku",
           "ing", "jam", "kabeh", "kang", "kanggo", "karo", "kowe", "ku", "lagu", "lampu", "lan",
           "menyang", "minggu", "nang", "ning", "opo", "ora", "paling", "pira", "rapat", "saiki",
           "saka", "sepur", "setel", "sing", "tembang", "tulung", "wis"},
    "lv": {"ar", "atskaņo", "atskaņot", "bet", "cik", "dziesmu", "epastu", "es", "ir", "ka", "kad",
           "kas", "ko", "kur", "kā", "kāda", "kādi", "kāds", "lai", "lūdzu", "man", "manu", "manā",
           "no", "par", "parādi", "pasaki", "rīta", "sarakstu", "tas", "tu", "un", "uz", "vai", "var",
           "vari", "ziņas", "šo", "šodien"},
    "ms": {"acara", "ada", "adakah", "adalah", "akan", "apa", "apakah", "baru", "berapa", "beritahu",
           "boleh", "dalam", "dan", "dari", "dengan", "di", "emel", "hari", "ini", "itu", "ke",
           "kepada", "lagu", "lampu", "mainkan", "mana", "mesyuarat", "nak", "pada", "penggera",
           "pukul", "saya", "senarai", "sila", "tentang", "tidak", "tolong", "untuk", "yang"},
    "nb": {"av", "ble", "dag", "den", "denne", "der", "det", "du", "eller", "en", "epost", "er", "for",
           "fortell", "fra", "gi", "har", "her", "hva", "hvor", "hvordan", "ikke", "jeg", "kan",
           "klokken", "kunne", "legg", "med", "meg", "men", "min", "mye", "neste", "noe", "noen", "nå",
           "når", "og", "også", "om", "på", "siste", "skal", "som", "spill", "takk", "til", "ved",
           "vennligst", "vil", "vært"},
    "pl": {"ale", "bardzo", "chcę", "co", "czy", "dla", "do", "dodaj", "dzisiaj", "ile", "jak", "jaka",
           "jaki", "jakie", "jakieś", "jest", "jestem", "jutro", "już", "która", "który", "mam", "mi",
           "mnie", "moje", "mojej", "może", "możesz", "na", "nie", "od", "po", "podaj", "pokaż",
           "powiedz", "proszę", "przez", "rano", "się", "spotkanie", "są", "tak", "teraz", "to",
           "tylko", "wiadomości", "wydarzenie", "włącz", "że"},
    "ro": {"această", "acest", "acum", "am", "azi", "care", "ce", "cu", "cum", "dar", "de", "despre",
           "din", "este", "la", "lista", "lui", "mai", "mea", "meu", "mi", "nu", "ora", "pe", "pentru",
           "pune", "redă", "rog", "sa", "si", "spune", "sunt", "să", "te", "trimite", "un", "vreau",
           "în", "și"},
    "sl": {"ali", "bilo", "biti", "bo", "da", "dan", "danes", "dodaj", "dogodek", "imam", "in", "iz",
           "je", "kaj", "kako", "kateri", "kdaj", "ki", "kje", "koliko", "lahko", "mi", "moj", "na",
           "nastavi", "ne", "ob", "od", "pa", "povej", "pošlji", "predvajaj", "prosim", "se", "sem",
           "sestanek", "seznam", "so", "sporočilo", "ta", "tudi", "za", "zdaj", "zelo"},
    "sq": {"dhe", "dua", "duhet", "faleminderit", "fundit", "janë", "ka", "kam", "ku", "kur", "këngë",
           "këtë", "listën", "luaj", "lutem", "me", "mund", "më", "ndonjë", "nga", "një", "nuk", "në",
           "pasdite", "po", "për", "që", "sa", "si", "sot", "tani", "tim", "trego", "të", "unë",
           "vendos", "çfarë", "është"},
    "sv": {"att", "av", "berätta", "den", "det", "du", "en", "ett", "från", "för", "har", "hur", "här",
           "idag", "inte", "jag", "kan", "klockan", "med", "mig", "min", "mitt", "när", "några", "och",
           "om", "på", "senaste", "som", "spela", "ta", "tack", "till", "vad", "vill", "är"},
    "sw": {"barua", "cha", "cheza", "gani", "habari", "hali", "hii", "huo", "je", "kama", "katika",
           "kengele", "kuhusu", "kutoka", "kuwa", "kwa", "kwenye", "la", "leo", "lini", "mimi",
           "mkutano", "na", "ngapi", "ni", "niambie", "nini", "orodha", "pepe", "saa", "sasa", "siku",
           "tafadhali", "unaweza", "wa", "wapi", "weka", "ya", "yangu", "za"},
    "tl": {"akin", "aking", "ako", "alas", "ang", "ano", "anong", "araw", "ay", "ba", "bang", "gusto",
           "hindi", "isang", "ito", "kaganapan", "kailan", "kay", "ko", "kong", "kung", "listahan",
           "may", "mga", "mo", "mula", "na", "ng", "ngayon", "ngayong", "ni", "oras", "paki", "para",
           "po", "sa", "saan", "sabihin", "susunod", "tungkol"},
    "tr": {"ama", "aç", "bana", "ben", "beni", "benim", "bir", "biz", "bu", "bugün", "daha", "değil",
           "ekle", "en", "et", "gibi", "gönder", "günü", "hangi", "her", "ile", "istiyorum", "için",
           "kadar", "kaç", "lütfen", "misin", "mı", "nasıl", "ne", "nedir", "nerede", "olarak",
           "onlar", "saat", "sabah", "sen", "siz", "son", "sonra", "söyle", "tüm", "var", "ve", "ver",
           "veya", "yeni", "yok", "zaman", "çal", "çok", "önce", "şu"},
    "vi": {"ba", "bao", "biết", "báo", "bạn", "cho", "các", "có", "của", "danh", "giờ", "gì", "hôm",
           "khi", "không", "là", "lòng", "lịch", "một", "mới", "nay", "ng", "ngày", "những", "nào",
           "này", "phát", "sách", "sự", "thể", "trong", "tôi", "vui", "và", "vào", "về", "với", "được",
           "đến"},
})

# Letters that belong to essentially one Latin script language. Function words need four words
# before they say anything and a typed question is often two, so one exclusive letter settles
# the short cases that the word tables and the diacritic rate both miss.
_UNIQUE = {
    "tr": "ığ",          # az shares these; the schwa below separates the two
    "az": "ə",
    "pl": "łążźńś",
    "ro": "ășț",         # comma below forms; the cedilla forms are ambiguous with Turkish
    "hu": "őű",
    "is": "þð",
    "cy": "ŵŷ",
    "lv": "āēīūķļņģ",
    "vi": "ơư",
    "sq": "ë",
    "de": "ß",
}

# Turkish, Polish, Romanian and Vietnamese are routinely typed without their diacritics, and
# clinical or ticket free text especially so. "icin" and "degil" are the same words as "için"
# and "değil" but score zero against an accented table, and stripping the accents also removes
# the diacritic rate that #42 leans on, so those inputs have no signal left at all. Matching a
# folded copy of every table restores it for all of them at once instead of duplicating each
# list by hand.
_FOLD_EXTRA = {"ı": "i", "ł": "l", "đ": "d", "ø": "o", "þ": "th", "ð": "d",
               "ß": "ss", "æ": "ae", "œ": "oe", "ə": "e", "ŋ": "n"}


def _fold(word: str) -> str:
    """Lowercase word with its diacritics removed, the way people type it on an ASCII keyboard."""
    out = []
    for ch in word:
        if ch in _FOLD_EXTRA:
            out.append(_FOLD_EXTRA[ch])
            continue
        stripped = "".join(c for c in unicodedata.normalize("NFD", ch)
                           if not unicodedata.combining(c))
        out.append(stripped or ch)
    return "".join(out)


_STOP_FOLDED = {lg: {_fold(w) for w in words} for lg, words in _STOP.items()}

# A function word English shares with the rival language ("to", "in", "have", "me", "am", "om")
# is evidence for neither side of that comparison, so it is taken off both scores. Counting it
# for English made a Danish sentence look English; counting it for the rival instead made
# "tell me about my alarms" look Albanian, because Albanian "me" means "with". The discount is
# per rival, so English is never penalised for colliding with a language that nothing else in
# the text points to.
_EN_SHARED = {lg: (_STOP["en"] | _STOP_FOLDED["en"]) & (words | _STOP_FOLDED[lg])
              for lg, words in _STOP.items() if lg != "en"}

_NON_EN_DIACRITICS = set(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ"          # Western European
    "ăâîșțşţ"                                   # Romanian
    "ąćęłńśźż"                                  # Polish
    "čďěňřšťůž"                                 # Czech / Slovak
    "őű"                                        # Hungarian
    "ğı"                                        # Turkish (text is lowercased before matching)
    "āēģīķļņūž"                                 # Baltic
    "đ"                                         # Serbo-Croatian / Vietnamese
)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


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
    return " ".join(_iter_text(state))[:max_chars]


def detect_script(text: str) -> str:
    """Dominant script of `text`: 'latin', 'han', 'devanagari', ... or 'unknown' if there are no letters."""
    counts: Dict[str, int] = {}
    latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF:      # Latin + Latin Extended Additional
            latin += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
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
        if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF:
            counts["latin"] += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    total = sum(counts.values())
    if not total:
        return {}
    return {k: v / total for k, v in counts.items() if v}


# A diacritic rate above this is taken as evidence the text is not English, even when no
# stopword list matches it.
NON_EN_DIACRITIC_RATE = 0.02


def latin_profile(text: str) -> Dict[str, object]:
    """Evidence behind the Latin-script language guess.

    Returns `language` (may be None when undecided), `english_hits`, `diacritic_rate` and
    `looks_non_english`. `analyse` needs the evidence and not just the verdict, because
    "undecided" and "English" are different answers and only one of them is safe to send to the
    English checkpoint.
    """
    words = [w.lower() for w in _WORD.findall(text)]
    folded = [_fold(w) for w in words]
    lowered = text.lower()
    diac = sum(1 for ch in lowered if ch in _NON_EN_DIACRITICS)
    diac_rate = diac / max(1, len(lowered))
    non_english = diac_rate >= NON_EN_DIACRITIC_RATE

    # An exclusive letter is enough on its own and does not need four words.
    uniq = {lg: sum(lowered.count(ch) for ch in chars) for lg, chars in _UNIQUE.items()}
    uniq_lg, uniq_hits = max(uniq.items(), key=lambda kv: kv[1], default=(None, 0))
    if uniq_hits:
        return {"language": uniq_lg, "english_hits": 0, "diacritic_rate": diac_rate,
                "looks_non_english": True}

    if len(words) < 4:
        return {"language": None, "english_hits": 0, "diacritic_rate": diac_rate,
                "looks_non_english": non_english}

    def hits(lg):
        """Word positions matching the table, accented or folded to plain ASCII."""
        table, table_folded = _STOP[lg], _STOP_FOLDED[lg]
        return sum(1 for w, f in zip(words, folded) if w in table or f in table_folded)

    scores = {lg: hits(lg) for lg in _STOP}
    en_raw = scores.get("en", 0)

    def shared(lg):
        table = _EN_SHARED.get(lg, frozenset())
        return sum(1 for w, f in zip(words, folded) if w in table or f in table)

    # Words English and the rival both claim are taken off both sides.
    net = {lg: scores[lg] - shared(lg) for lg in scores if lg != "en"}
    best_lg, best = max(net.items(), key=lambda kv: kv[1], default=(None, 0))
    en = en_raw - (shared(best_lg) if best_lg else 0)

    # Function words for some language other than English, and nothing for English, is
    # evidence the text is not English even when it is too thin to say which language it is.
    # "Musteriden iki kez ucret alindi ve para iadesi istiyor" gives seven languages one hit
    # each: naming a winner there would be a coin toss, but the text is plainly not English.
    # Feeding that into looks_non_english lets #42's routing use it without inventing a name.
    if best >= 1 and en <= 0:
        non_english = True
    # No stopword hit for any non-English language is no evidence for a *particular* one. Naming
    # the winner of a 0-0 tie invented a language (Romanian text was reported as French), so stay
    # undecided and let the diacritic rate speak.
    if best == 0:
        best_lg = None

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

    Scores function-word hits per language and requires the winner to beat English by a margin,
    so ordinary English is never misrouted. Short inputs usually return None on purpose.
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
