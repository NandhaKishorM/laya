"""Many-option choice on three public intent sets: one question, a wider budget, the embedding
shortlist, and `predict_tournament`.

Every strategy answers one row per call, as a caller would. Needs `pyarrow` for the parquet files.

    python research/benchmarks/tournament/benchmark.py --dataset banking77 --split test \
        --strategies single,wide,shortlist,tournament --out banking77.json
"""
import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import laya  # noqa: E402
from laya.evals import ece  # noqa: E402

REVISIONS = {
    "mteb/banking77": "18072d2685ea682290f7b8924d94c62acc19c0b2",
    "clinc/clinc_oos": "155b9c710419136e17307b80d0a13e68cd46b4ec",
    "SetFit/amazon_massive_intent_en-US": "f7672a018e8ceb37fc0184dcfbb7e665155ffea6",
}


def fetch(repo, filename):
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo]))


def banking77(split):
    import pyarrow.parquet as pq

    def read(name):
        return pq.read_table(fetch("mteb/banking77", "data/%s-00000-of-00001.parquet" % name)).to_pylist()

    names = dict((r["label"], r["label_text"]) for r in read("test"))
    return [names[i] for i in sorted(names)], [(r["text"], r["label_text"]) for r in read(split)]


def clinc150(split):
    import pyarrow.parquet as pq

    path = fetch("clinc/clinc_oos", "plus/%s-00000-of-00001.parquet" % split)
    names = json.loads(pq.read_schema(path).metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
    rows = [(r["text"], names[r["intent"]]) for r in pq.read_table(path).to_pylist()]
    # In-scope intents only: "oos" is not an intent a label can describe.
    return [n for n in names if n != "oos"], [(text, gold) for text, gold in rows if gold != "oos"]


def massive(split):
    def read(name):
        path = fetch("SetFit/amazon_massive_intent_en-US", "%s.jsonl" % name)
        return [json.loads(line) for line in path.read_text().splitlines()]

    names = dict((r["label"], r["label_text"]) for r in read("train"))
    return [names[i] for i in sorted(names)], [(r["text"], r["label_text"]) for r in read(split)]


DATASETS = {
    "banking77": (banking77, "Which banking intent does `message` express?"),
    "clinc150": (clinc150, "Which intent does `message` express?"),
    "massive": (massive, "Which intent does `message` express?"),
}


def strategy(name, agent):
    """`(state, question) -> (answer, model calls)` for one strategy name."""
    if name in ("single", "wide"):
        budget = {"head_max_len": 512, "max_len": 1024} if name == "wide" else {}
        return lambda state, q: (agent.predict(state, {"q": q}, **budget)["answers"]["q"], 1)
    if name == "shortlist":
        embed_fn = laya.cached_embed_fn(laya.embed_fn_from_agent(agent))
        return lambda state, q: (laya.predict_shortlist(agent, state, {"q": q}, embed_fn)["answers"]["q"], 1)
    if name.startswith("tournament"):
        size = {"group_size": int(name.split("-")[1])} if "-" in name else {}

        def run(state, q):
            out = laya.predict_tournament(agent, state, {"q": q}, **size)
            return out["answers"]["q"], out["tournament"]["q"]["rounds"] + 1
        return run
    raise SystemExit("unknown strategy %r" % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, help="score a seeded sample of this many rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategies", default="single,wide,shortlist,tournament",
                    help="comma list of single, wide, shortlist, tournament, tournament-<group size>")
    ap.add_argument("--model", default="english")
    ap.add_argument("--device")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load, instructions = DATASETS[args.dataset]
    names, rows = load(args.split)
    if args.limit and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)
    question = {"type": "choice", "instructions": instructions,
                "criteria": {name.replace("_", " "): None for name in names}}
    agent = laya.load(args.model, device=args.device)
    few = dict(question, criteria=dict(list(question["criteria"].items())[:8]))
    for text, _gold in rows[:3]:  # warm up before anything is timed
        agent.predict({"message": text}, {"q": few})

    report = {"dataset": args.dataset, "split": args.split, "rows": len(rows), "options": len(names),
              "revisions": REVISIONS, "model": args.model, "device": str(agent.device),
              "laya": laya.__version__, "machine": platform.machine(), "strategies": {}}
    first = None
    for name in args.strategies.split(","):
        answer_fn = strategy(name, agent)
        correct, pairs, ms, calls, errors = [], [], [], [], []
        for text, gold in rows:
            started = time.perf_counter()
            try:
                answer, n_calls = answer_fn({"message": text}, question)
            except ValueError as exc:  # the question does not fit the window at all
                errors.append(str(exc))
                correct.append(0)
                if len(errors) == len(correct) == 3:
                    break
                continue
            ms.append((time.perf_counter() - started) * 1000)
            correct.append(int(answer["choice"] == gold.replace("_", " ")))
            pairs.append((answer["answer_confidence"], bool(correct[-1])))
            calls.append(n_calls)
        entry = {"errors": len(errors), "first_error": errors[0] if errors else None}
        if len(correct) == len(rows):
            entry["accuracy"] = round(sum(correct) / len(rows), 4)
        if ms:
            ms.sort()
            entry.update(ece=round(ece([c for c, _ in pairs], [ok for _, ok in pairs]), 4),
                         mean_ms=round(sum(ms) / len(ms), 1), p50_ms=round(ms[len(ms) // 2], 1),
                         p95_ms=round(ms[int(len(ms) * 0.95)], 1),
                         calls_per_row=round(sum(calls) / len(calls), 3))
        if first is None:
            first = (name, correct)
        elif len(correct) == len(first[1]) == len(rows):
            entry["vs_" + first[0]] = {"gained": sum(a > b for a, b in zip(correct, first[1])),
                                       "lost": sum(a < b for a, b in zip(correct, first[1]))}
        report["strategies"][name] = entry
        print(args.dataset, name, json.dumps(entry), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
