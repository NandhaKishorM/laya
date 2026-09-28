"""Regression checks for localized built-in question presets.

Run: python tests/test_presets.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya  # noqa: E402
from laya.presets import state_field  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, condition):
    if condition:
        PASS.append(name)
    else:
        FAIL.append(name)


PRESETS = {
    "email": laya.email_questions,
    "guard": laya.guard_questions,
    "moderation": laya.moderation_questions,
    "router": laya.router_questions,
    "triage": laya.triage_questions,
}

for name, build in PRESETS.items():
    english = build()
    check("%s/default remains English" % name, english, build(language="en"))
    localized = build(language="sv-SE")
    check("%s/question keys remain stable" % name, sorted(localized), sorted(english))
    check("%s/state field remains stable" % name,
          state_field(localized), state_field(english))
    check_true("%s/every Swedish instruction names its state field" % name,
               all(state_field(localized) in question["instructions"]
                   for question in localized.values()))
    for qid in english:
        check("%s/%s/question type remains stable" % (name, qid),
              localized[qid]["type"], english[qid]["type"])
        en_criteria, sv_criteria = english[qid].get("criteria"), localized[qid].get("criteria")
        if isinstance(en_criteria, dict):
            check("%s/%s/choice keys remain stable" % (name, qid),
                  sorted(sv_criteria), sorted(en_criteria))
        elif isinstance(en_criteria, list):
            check("%s/%s/score levels remain stable" % (name, qid),
                  len(sv_criteria), len(en_criteria))
    check("%s/Swedish locale alias is stable" % name, localized, build(language="sv"))
    try:
        build(language="de")
    except ValueError:
        check_true("%s/unsupported language rejected" % name, True)
    else:
        check_true("%s/unsupported language rejected" % name, False)

check("guard/Swedish topic descriptions are localized",
      laya.guard_questions("sv")["topic"]["criteria"]["product_support"],
      "produkthjälp och kundsupport")
check("guard/Swedish harm severity preserves the four ordered meanings",
      laya.guard_questions("sv")["harm_severity"]["criteria"],
      [
          "ingen: vanlig och ofarlig begäran",
          "liten: olämpligt innehåll utan tydlig skaderisk",
          "allvarlig: osäkra råd eller kränkningar",
          "svår: farligt eller olagligt innehåll",
      ])
check("moderation/Swedish severity preserves the four ordered meanings",
      laya.moderation_questions("sv")["severity"]["criteria"],
      [
          "inget regelbrott: vanligt och relevant inlägg",
          "lindrigt: otrevlig ton eller utanför ämnet, utan angrepp på någon",
          "tydligt regelbrott: förolämpningar, trakasserier eller riktad spam",
          "grovt: hot, hatpropaganda eller uppmaningar till våld",
      ])
check("router/Swedish difficulty preserves the four ordered meanings",
      laya.router_questions("sv")["difficulty"]["criteria"],
      [
          "trivialt: en enkel uppgift eller ett kort svar",
          "enkelt: kort svar utan resonemang i flera steg",
          "måttligt: kräver flera steg",
          "svårt: långt resonemang i flera steg eller specialistkunskap",
      ])
check("router/Swedish domain keys remain stable",
      sorted(laya.router_questions("sv")["domain"]["criteria"]),
      sorted(laya.router_questions("en")["domain"]["criteria"]))

if FAIL:
    print("\n%d failed, %d passed" % (len(FAIL), len(PASS)))
    print("\n".join(FAIL))
    raise SystemExit(1)
print("\n%d passed, 0 failed" % len(PASS))
