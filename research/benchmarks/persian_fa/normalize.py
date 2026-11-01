"""Optional Persian text normalisation, used only when ``run.py --normalize`` is passed.

It is an ablation, not part of the default protocol: running with and without it shows how
much of the error on the ``orthography`` and ``digits`` families comes from surface form.
Finglish is intentionally left untouched (transliteration is out of scope).
"""
import re

_CHAR_MAP = str.maketrans({
    "ي": "ی",  # Arabic yeh  -> Persian yeh
    "ى": "ی",  # alef maksura -> Persian yeh
    "ك": "ک",  # Arabic kaf  -> Persian keheh
    "ة": "ه",  # teh marbuta -> heh
    "ـ": None,      # tatweel
})
# Arabic-Indic and Persian digits -> ASCII digits
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile("[ً-ٰٟ]")
# "می خوام" / "میخوام" -> "می‌خوام" ; "نمی خوام" -> "نمی‌خوام"
_MI_PREFIX = re.compile(r"(?<![؀-ۿ])(ن?می)[ ]?(?=[؀-ۿ]{2,})")
# "هزینه ی" -> "هزینه‌ی" ; "پیامک های" -> "پیامک‌های"
_SUFFIX = re.compile(r"(?<=[؀-ۿ]) (ی|های|ها|ای|ام|ات|اش|تر|ترین)(?![؀-ۿ])")
ZWNJ = "‌"


def normalize(text: str) -> str:
    text = text.translate(_CHAR_MAP).translate(_DIGITS)
    text = _DIACRITICS.sub("", text)
    text = _MI_PREFIX.sub(lambda m: m.group(1) + ZWNJ, text)
    text = _SUFFIX.sub(lambda m: ZWNJ + m.group(1), text)
    return re.sub(r"[ \t]+", " ", text).strip()
