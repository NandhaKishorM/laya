#!/usr/bin/env python3
"""Fit and calibrate on your own labels, end to end: the cold engine, then the fitted one,
on the same held-out rows, with every number measured here rather than typed.

    python examples/42_fit_and_calibrate_vegaml.py

Reads examples/data/routing_labels.csv (60 messages routed to billing, HR, PR, sales or
technical, each with an urgency level), holds out four rows per team, fits heads and the
calibration on the other forty, and prints: accuracy and ECE before and after, the fit report
(which heads earned serving, where the calibration was measured), the conformal set sizes and
abstentions, the auto/review split, one full answer before and after so the shape is visible,
and an ablation over how many fitting rows the same held-out set gets, from the minimum laya
accepts upward.

Needs laya 0.4.24 or newer (vegaml ships with it) and a few minutes on a laptop GPU, most of
it reading the fitting rows through the frozen reader.
"""
import csv
import json
import os
import sys
import time
from collections import Counter

import laya

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "routing_labels.csv")

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "invoices, payments, refunds",
                          "hr": "jobs, payroll, leave, policy",
                          "pr": "press, media, public statements",
                          "sales": "pricing, demos, quotes, trials",
                          "technical": "bugs, outages, API and product questions"}},
    "urgency": {"type": "score", "instructions": "How soon does this need a reply?",
                "criteria": ["can wait", "this week", "today"]},
}
HOLD_OUT_PER_TEAM = 4


def ece(pairs, bins=10):
    """Expected calibration error of (confidence, was_correct) pairs, equal-width bins."""
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if inside:
            conf = sum(c for c, _ in inside) / len(inside)
            acc = sum(ok for _, ok in inside) / len(inside)
            total += len(inside) / len(pairs) * abs(acc - conf)
    return total


def measure(agent, rows):
    team_pairs, urg_pairs, set_sizes, abstains, routes, mae = [], [], [], 0, Counter(), 0.0
    for r in rows:
        out = agent.predict(r["state"], QUESTIONS)
        t, u = out["answers"]["team"], out["answers"]["urgency"]
        team_pairs.append((t["answer_confidence"], t["choice"] == r["team"]))
        urg_pairs.append((u["answer_confidence"], u["score"] == int(r["urgency"])))
        mae += abs(u["score"] - int(r["urgency"]))
        set_sizes.append(len(t["prediction_set"]))
        abstains += int(bool(t["abstain"]))
        routes[t["route"]] += 1
    n = len(rows)
    return {"n": n,
            "team_acc": sum(ok for _, ok in team_pairs) / n, "team_ece": ece(team_pairs),
            "team_conf": sum(c for c, _ in team_pairs) / n,
            "urg_acc": sum(ok for _, ok in urg_pairs) / n, "urg_ece": ece(urg_pairs), "urg_mae": mae / n,
            "set_sizes": dict(sorted(Counter(set_sizes).items())), "abstain": abstains,
            "routes": dict(routes)}


def show(label, m):
    print("\n== %s, %d held-out rows" % (label, m["n"]))
    print("   team accuracy      %.3f   ECE %.3f   mean confidence %.3f" % (m["team_acc"], m["team_ece"], m["team_conf"]))
    print("   urgency accuracy   %.3f   ECE %.3f   MAE %.2f levels" % (m["urg_acc"], m["urg_ece"], m["urg_mae"]))
    print("   team set sizes     %s   abstain %d   route %s" % (m["set_sizes"], m["abstain"], m["routes"]))


def fit_on(agent, fit_rows):
    """Fresh heads and calibration from these rows; a quarter is held out inside fit()."""
    agent.vega.heads.clear()
    agent.calibration = None
    return agent.fit([{"state": r["state"], "gold": {"team": r["team"], "urgency": r["urgency"]}}
                      for r in fit_rows], questions=QUESTIONS)


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    by_team = {}
    for r in rows:
        by_team.setdefault(r["team"], []).append(r)
    held = [r for rs in by_team.values() for r in rs[-HOLD_OUT_PER_TEAM:]]
    fit_rows = [r for rs in by_team.values() for r in rs[:-HOLD_OUT_PER_TEAM]]
    print("%d rows: %d to fit, %d held out (%d per team)" % (len(rows), len(fit_rows), len(held), HOLD_OUT_PER_TEAM))
    print("minimum rows fit() accepts for these questions: %d (5 per option of the widest question "
          "for the heads, plus a quarter held out to calibrate on, never fewer than 10)"
          % laya.VegaAgent.min_rows(QUESTIONS))

    t0 = time.time()
    agent = laya.Agent(vegaml=True)
    print("loaded %s in %.0fs" % (agent, time.time() - t0))

    probe = held[0]
    before_answer = agent.predict(probe["state"], QUESTIONS)["answers"]["team"]
    before = measure(agent, held)
    show("before: the cold engine", before)

    t0 = time.time()
    report = fit_on(agent, fit_rows)
    print("\n== fit + calibrate on %d rows in %.0fs: %d fit the heads, %d held out to calibrate on"
          % (report["n_rows"], time.time() - t0, report["n_fit"], report["n_calibrate"]))
    for q, h in report["heads"].items():
        print("   %-8s head cv %.3f vs engine %.3f on the same rows  ->  %s"
              % (q, h["cv_accuracy"] or 0, h["engine_accuracy"] or 0,
                 "SERVED" if h["used_by_auto"] else "left to the engine"))
    cal = report["calibration"]
    for q, c in cal["questions"].items():
        print("   %-8s temperature %.2f  conformal q-hat %.3f  ECE on the calibration rows %.3f -> %.3f"
              % (q, c["temperature"], c["qhat"], c["ece_before"], c["ece_after"]))
    print("   review line per question: %s" % cal["band"])
    if cal["band_unreachable"]:
        print("   %s: no line holds precision %.2f with a 95%% bound on %d calibration rows, so the "
              "conformal set routes those answers" % (cal["band_unreachable"], cal["target_precision"], cal["n_rows"]))

    after = measure(agent, held)
    show("after: heads + calibration", after)
    print("\n== change on held-out rows: team accuracy %+.3f, team ECE %+.3f, urgency MAE %+.2f"
          % (after["team_acc"] - before["team_acc"], after["team_ece"] - before["team_ece"],
             after["urg_mae"] - before["urg_mae"]))

    after_answer = agent.predict(probe["state"], QUESTIONS)["answers"]["team"]
    keep = ("choice", "probabilities", "answer_confidence", "prediction_set", "abstain", "unbound",
            "readout", "route", "route_reason")
    print("\n== one held-out message: %r  (gold: %s)" % (probe["state"][:70], probe["team"]))
    for label, ans in (("before", before_answer), ("after", after_answer)):
        print("   %s: %s" % (label, json.dumps({k: ans[k] for k in keep if k in ans}, ensure_ascii=False)))

    agent.save_calibration(os.path.join(HERE, "..", "fit-routing"))
    print("\nsaved heads + calibration to fit-routing/; reload with agent.load_calibration(\"fit-routing/\")")

    # Ablation: the same 20 held-out rows, fewer fitting rows. 7 per team (35) is the floor
    # fit() accepts for these questions; the run above used 8 (40).
    print("\n== ablation: fitting rows per team, same %d held-out rows" % len(held))
    print("   %-9s %-5s %-9s %-9s %-9s %-9s %s" % ("per team", "rows", "team acc", "team ECE", "urg. acc", "urg. MAE", "route"))
    print("   %-9s %-5s %-9.3f %-9.3f %-9.3f %-9.2f %s" % ("cold", "0", before["team_acc"], before["team_ece"],
                                                       before["urg_acc"], before["urg_mae"], before["routes"]))
    for per_team in (7, 8):
        subset = [r for rs in by_team.values() for r in rs[:per_team]]
        fit_on(agent, subset)
        m = measure(agent, held)
        print("   %-9d %-5d %-9.3f %-9.3f %-9.3f %-9.2f %s" % (per_team, len(subset), m["team_acc"], m["team_ece"],
                                                           m["urg_acc"], m["urg_mae"], m["routes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
