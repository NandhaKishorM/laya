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
          "ingen: begäran är ofarlig och följsamhet orsakar ingen skada",
          "lindrig: olämpligt innehåll men liten risk för faktisk skada",
          "allvarlig: följsamhet kan orsaka konkret skada, till exempel genom osäkra råd eller kränkningar",
          "mycket allvarlig: hög risk för betydande skada, exempelvis farligt eller olagligt agerande",
      ])
check("email/default urgency remains English",
      laya.email_questions()["urgency"]["criteria"],
      ["no time pressure", "needs attention soon", "blocking issue or hard deadline"])
check("guard/Swedish harm scale matches English level count",
      len(laya.guard_questions("sv")["harm_severity"]["criteria"]),
      len(laya.guard_questions("en")["harm_severity"]["criteria"]))
check("email/Swedish urgency has distinct ascending anchors",
      laya.email_questions(language="sv")["urgency"]["criteria"],
      ["ingen tidspress: ärendet kan vänta utan märkbar påverkan",
       "viss brådska: bör hanteras snart men blockerar inte arbetet",
       "hög brådska: problemet blockerar arbetet eller tidsfristen är nära"])
check("moderation/Swedish severity preserves the four ordered meanings",
      laya.moderation_questions("sv")["severity"]["criteria"],
      [
          "inget regelbrott: vanligt och relevant inlägg",
          "lindrigt regelbrott: otrevlig ton eller utanför ämnet, utan angrepp på någon",
          "allvarligt regelbrott: riktade förolämpningar, trakasserier eller spam",
          "mycket allvarligt regelbrott: hot, hatpropaganda eller uppmaning till våld",
      ])
check("moderation/Swedish threat covers harm and intimidation naturally",
      laya.moderation_questions("sv")["threat"]["instructions"],
      "Innehåller `post` hot om våld eller annan skada, eller försök att skrämma någon?")
check("router/Swedish difficulty preserves the four ordered meanings",
      laya.router_questions("sv")["difficulty"]["criteria"],
      [
          "direkt faktasvar eller en enkel åtgärd",
          "kort svar som kräver viss tolkning men inga resonemang i flera steg",
          "flera steg, alternativ eller jämförelser krävs",
          "långt resonemang i flera steg, noggrann analys eller specialistkunskap krävs",
      ])
check("router/Swedish domain keys remain stable",
      sorted(laya.router_questions("sv")["domain"]["criteria"]),
      sorted(laya.router_questions("en")["domain"]["criteria"]))
check_true("triage/Swedish churn question separates intent from terms questions",
           "avsikt att lämna tjänsten" in laya.triage_questions("sv")["churn_risk"]["instructions"]
           and "uppsägningsvillkor" in laya.triage_questions("sv")["churn_risk"]["instructions"])
check("triage/Swedish churn polarity is explicit",
      sorted(laya.triage_questions("sv")["churn_risk"]["criteria"]), ["false", "true"])

if FAIL:
    print("\n%d failed, %d passed" % (len(FAIL), len(PASS)))
    print("\n".join(FAIL))
    raise SystemExit(1)
print("\n%d passed, 0 failed" % len(PASS))
