"""Evaluate high-cardinality choice: single-call vs. two-stage.

The question P0 #3 leaves open: two_stage_choice lifts the option ceiling, but does the
coarse-then-fine split cost more accuracy than the ceiling it removes?

Builds a synthetic labelled set where the correct option is recoverable from the state,
then compares single-call accuracy (where it fits) against two-stage at each cardinality.
"""
import argparse
import json
import random
import sys
import time

import laya
from laya import patterns
from laya.common import ece_score
from laya.errors import OptionsTooLongError

import numpy as np

# Distinct, unambiguous domains so the label is genuinely inferable from the text.
TOPICS = [
    ("billing_invoice", "an invoice was issued for the wrong amount"),
    ("billing_refund", "the customer wants money returned for a duplicate charge"),
    ("login_password", "the user cannot reset their password"),
    ("login_mfa", "two-factor authentication codes are not arriving"),
    ("outage_api", "the public API returns 500 errors for every request"),
    ("outage_dashboard", "the web dashboard fails to load any charts"),
    ("data_export", "a CSV export job never finishes"),
    ("data_import", "uploading a spreadsheet fails validation"),
    ("account_delete", "the customer asks to delete their account and data"),
    ("account_transfer", "ownership must move to a different administrator"),
    ("shipping_delay", "a physical package has not arrived on time"),
    ("shipping_address", "the delivery address needs correcting before dispatch"),
    ("security_phishing", "a suspicious email asks for account credentials"),
    ("security_breach", "an unauthorized login appeared from a new country"),
    ("feature_request", "the customer suggests a new product capability"),
    ("bug_report", "a button in the settings page does nothing when clicked"),
]


def make_options(n, rng):
    """n options: the real topics plus filler, each with a description."""
    opts = {}
    for name, desc in TOPICS:
        opts[name] = desc
    i = 0
    while len(opts) < n:
        opts["misc_category_%03d" % i] = "unrelated internal category number %d" % i
        i += 1
    items = list(opts.items())
    rng.shuffle(items)
    return dict(items)


def make_cases(n_cases, rng):
    cases = []
    for _ in range(n_cases):
        label, desc = rng.choice(TOPICS)
        state = "Support ticket: %s. Please route this to the correct team." % desc
        cases.append((state, label))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases", type=int, default=64)
    ap.add_argument("--cardinalities", default="16,50,100,255")
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--out", default="high_cardinality_results.json")
    args = ap.parse_args()

    rng = random.Random(0)
    agent = laya.Agent(args.model, device=args.device)
    cases = make_cases(args.cases, rng)
    results = []

    for n in [int(x) for x in args.cardinalities.split(",")]:
        options = make_options(n, rng)
        instructions = "Which category does this support ticket belong to?"

        row = {"n_options": n, "n_cases": len(cases)}

        # ---- single call ----
        single_ok, single_conf, single_correct, fits = 0, [], [], True
        t0 = time.time()
        for state, label in cases:
            try:
                ans = agent.system_one(
                    state,
                    {"q": {"type": "choice", "instructions": instructions, "criteria": options}},
                )["answers"]["q"]
            except OptionsTooLongError:
                fits = False
                break
            hit = ans["choice"] == label
            single_ok += hit
            single_conf.append(ans["confidence"])
            single_correct.append(float(hit))
        if fits:
            row["single"] = {
                "accuracy": round(single_ok / len(cases), 4),
                "ece": round(ece_score(np.array(single_conf), np.array(single_correct)), 4),
                "mean_confidence": round(float(np.mean(single_conf)), 4),
                "seconds": round(time.time() - t0, 2),
            }
        else:
            row["single"] = {"fits": False, "reason": "OptionsTooLongError"}

        # ---- two stage ----
        two_ok, two_conf, two_correct = 0, [], []
        t0 = time.time()
        for state, label in cases:
            out = patterns.two_stage_choice(
                agent, state, instructions, options, group_size=args.group_size
            )
            hit = out["choice"] == label
            two_ok += hit
            two_conf.append(out["confidence"])
            two_correct.append(float(hit))
        row["two_stage"] = {
            "accuracy": round(two_ok / len(cases), 4),
            "ece": round(ece_score(np.array(two_conf), np.array(two_correct)), 4),
            "mean_confidence": round(float(np.mean(two_conf)), 4),
            "seconds": round(time.time() - t0, 2),
        }

        results.append(row)
        print(json.dumps(row), flush=True)

    with open(args.out, "w") as f:
        json.dump({"cases": args.cases, "results": results}, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
