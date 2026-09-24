"""Regression: blank/whitespace explicit `lang` must fall through to detection.

`Router.route(lang=...)` is an explicit override, but only a code that actually
names a language should override anything. `_english_from_code` returns None for
None/blank/whitespace (documented as "no usable hint"), and the old code did
`if lang is not None: key = "english" if _english_from_code(lang) else "multilingual"`.
Because None is falsy, `lang=""` or `lang="   "` silently forced the multilingual
checkpoint instead of abstaining like a `lang_guess` that resolves to nothing.

The fix checks `_english_from_code`'s return value directly and falls through to
`lang_guess`/detection when it is None. Real English/non-English codes still route
immediately.

Run: python tests/test_blank_lang_routing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.router import Router  # noqa: E402

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


GENERIC = {"intent": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}

# A state the built-in detector reads as English. On this state "blank lang must
# abstain" and "blank lang forced multilingual" are distinguishable, because
# abstaining keeps the English checkpoint while the bug would jump to multilingual.
ENGLISH = "I was charged twice for invoice 4411"
# A state the detector reads as non-English, to show a real code is unchanged.
GERMAN = "Mein Konto wurde zweimal belastet, bitte erstatten Sie"

r = Router()

# ------------------------------------------------------------------ baseline
check("baseline/plain English state routes english", r.route(ENGLISH, GENERIC)["model"], "english")
check("baseline/no lang emits no explicit reason",
      "explicit lang=" in r.route(ENGLISH, GENERIC)["reason"], False)

# ------------------------------------------------------------------ blank abstains
blank = r.route(ENGLISH, GENERIC, lang="")
ws = r.route(ENGLISH, GENERIC, lang="   ")
none_val = r.route(ENGLISH, GENERIC, lang=None)

check("blank/empty lang falls through to detection (english)", blank["model"], "english")
check("blank/whitespace lang falls through to detection (english)", ws["model"], "english")
check("blank/None lang keeps detected model", none_val["model"], "english")
check("blank/empty lang does not claim an explicit override", "explicit lang=" in blank["reason"], False)
check("blank/whitespace lang does not claim an explicit override", "explicit lang=" in ws["reason"], False)
# fall-through means the detection block is present, not just an equal model
check_true("blank/empty lang keeps the detection block", blank["detection"] is not None)
check_true("blank/whitespace lang keeps the detection block", ws["detection"] is not None)

# ------------------------------------------------------------------ real codes still win
check("explicit/en forces english", r.route(GERMAN, GENERIC, lang="en")["model"], "english")
check("explicit/de forces multilingual", r.route(ENGLISH, GENERIC, lang="de")["model"], "multilingual")
check("explicit/en keeps the explicit reason", "explicit lang=" in r.route(GERMAN, GENERIC, lang="en")["reason"], True)
check("explicit/de keeps the explicit reason", "explicit lang=" in r.route(ENGLISH, GENERIC, lang="de")["reason"], True)

for code in ("en", "EN", "en-US", "en_US", "en_US.UTF-8"):
    check("code/%s routes english" % code, r.route(ENGLISH, GENERIC, lang=code)["model"], "english")

for code in ("de", "fr", "zh_CN", "pt-BR"):
    check("code/%s routes multilingual" % code, r.route(ENGLISH, GENERIC, lang=code)["model"], "multilingual")

# ------------------------------------------------------------------ blank does not mask a hint
# With a real hint installed, a blank lang must still abstain rather than pin multilingual.
r_hint = Router(lang_guess="de")
check("blank/does not mask an installed lang_guess",
      r_hint.route(GERMAN, GENERIC, lang="")["model"], "multilingual")
check("blank/does not break plain detection with a hint installed",
      r_hint.route(GERMAN, GENERIC)["model"], "multilingual")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
