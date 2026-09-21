"""What the router does with Latin-script languages, and what it costs.

#42 stopped treating an undecided Latin language as English and lets the diacritic rate
send accented text to the multilingual checkpoint. That covers the accented languages. A
language written in plain ASCII has no diacritic rate to read, so Indonesian, Malay,
Javanese, Swahili and Tagalog still went to the English checkpoint at 99 to 100 percent,
and `results/cpu_51_language_sweep.json` shows the multilingual checkpoint winning on all
of them. This script measures how much of each locale still lands on the wrong one.

Part A  routing audit: share of each locale sent to the English checkpoint. Pure Python,
        no weights, runs in a second.
Part B  end-to-end accuracy through the router for one locale, with weights.

  python3 research/scripts/bench_latin_routing.py --data /path/to/massive/1.1/data
  python3 research/scripts/bench_latin_routing.py --data ... --with-models tr-TR

MASSIVE 1.1: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from laya import lang  # noqa: E402

SWEEP = os.path.join(REPO, "research", "results", "cpu_51_language_sweep.json")
OUT = os.path.join(REPO, "research", "results", "latin_routing.json")
N_OPTIONS = 20
SEED = 0


def sweep_delta():
    """multilingual - english accuracy per language, from the 51-language CPU sweep."""
    if not os.path.exists(SWEEP):
        return {}
    by_model = json.load(open(SWEEP, encoding="utf-8"))["part_a"]["by_model"]
    en, ml = by_model["english"]["per_language"], by_model["multilingual"]["per_language"]
    return {l: round(ml[l]["accuracy"] - en[l]["accuracy"], 4) for l in en if l in ml}


def test_rows(path, limit):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r for r in rows if r["partition"] == "test"][:limit]


def audit(data_dir, limit):
    delta = sweep_delta()
    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        locale = os.path.basename(path)[:-6]
        rows = test_rows(path, limit)
        if not rows:
            continue
        if lang.detect_script(" ".join(r["utt"] for r in rows[:5])) != "latin":
            continue          # non-Latin scripts are already routed by script alone
        tags = collections.Counter()
        to_english = 0
        for r in rows:
            a = lang.analyse({"body": r["utt"]})
            to_english += bool(a["is_english"])
            tags[str(a["language"])] += 1
        d = delta.get(locale[:2])
        out.append({
            "locale": locale,
            "n": len(rows),
            "to_english_checkpoint": round(to_english / len(rows), 4),
            "multilingual_minus_english": d,
            "top_detected": tags.most_common(3),
        })
    return out


def accuracy(data_dir, locale, n, device):
    import laya
    rows = [json.loads(l) for l in open(os.path.join(data_dir, locale + ".jsonl"), encoding="utf-8")]
    counts = collections.Counter(r["intent"] for r in rows)
    intents = sorted(i for i, _ in counts.most_common(N_OPTIONS))
    question = {"intent": {
        "type": "choice",
        "instructions": "Classify the user's utterance into exactly one intent.",
        "criteria": {i: i.replace("_", " ") for i in intents}}}
    test = [r for r in rows if r["partition"] == "test" and r["intent"] in intents]
    random.Random(SEED).shuffle(test)
    test = test[:n]

    english = laya.load("convaiinnovations/laya", device=device)
    multi = laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)
    english.predict({"body": "warmup"}, question)
    multi.predict({"body": "warmup"}, question)

    def run(pick):
        hits = to_en = 0
        for r in test:
            agent = english if pick(r["utt"]) else multi
            to_en += agent is english
            if agent.predict({"body": r["utt"]}, question)["answers"]["intent"]["choice"] == r["intent"]:
                hits += 1
        return round(hits / len(test), 4), to_en

    routed, routed_en = run(lambda u: lang.analyse({"body": u})["is_english"])
    only_ml, _ = run(lambda u: False)
    only_en, _ = run(lambda u: True)
    return {"locale": locale, "n": len(test), "n_options": N_OPTIONS,
            "router": routed, "router_sent_to_english": routed_en,
            "multilingual_only": only_ml, "english_only": only_en}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="MASSIVE 1.1 data directory")
    ap.add_argument("--limit", type=int, default=400, help="test rows per locale")
    ap.add_argument("--with-models", default=None, metavar="LOCALE",
                    help="also measure end-to-end accuracy for this locale (downloads weights)")
    ap.add_argument("--n", type=int, default=300, help="rows for the accuracy run")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    rows = audit(a.data, a.limit)
    print("Part A  routing audit (Latin-script locales)\n")
    print("%-9s %6s  %-24s %s" % ("locale", "-> en", "ml - en accuracy", "top detections"))
    print("-" * 78)
    gap = 0.0
    for r in rows:
        d = r["multilingual_minus_english"]
        cost = r["to_english_checkpoint"] * d if d else 0.0
        if d and d > 0:
            gap += cost
        print("%-9s %5.0f%%  %-24s %s" % (
            r["locale"], r["to_english_checkpoint"] * 100,
            ("%+.3f" % d) if d is not None else "n/a",
            ", ".join("%s=%d" % t for t in r["top_detected"])))
    print("\naccuracy left on the table by routing to the English checkpoint: %.3f "
          "(summed over locales where multilingual wins)" % gap)

    results = {"n_options": N_OPTIONS, "seed": SEED, "part_a": rows}
    if a.with_models:
        print("\nPart B  end-to-end accuracy through the router\n")
        acc = accuracy(a.data, a.with_models, a.n, a.device)
        results["part_b"] = acc
        print("  %-28s %.3f" % ("router", acc["router"]))
        print("  %-28s %.3f" % ("multilingual only", acc["multilingual_only"]))
        print("  %-28s %.3f" % ("english only", acc["english_only"]))
        print("  router sent %d/%d to the English checkpoint" %
              (acc["router_sent_to_english"], acc["n"]))

    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
