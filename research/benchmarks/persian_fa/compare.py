"""Side-by-side table of several audited runs, per family.

    python compare.py runs/multilingual runs/english runs/multilingual_norm
    python compare.py runs/* --mode choice_en --out comparison.md

Every run must pass the audit first (this script re-runs it).
"""
import argparse
import sys
from pathlib import Path

from audit import Audit, check_dataset, check_manifest, check_run, check_source, summarise
from common import FAMILIES, ROOT, load_cases
from prompts import MODES


def label_for(meta):
    b = meta.get("backend", {})
    name = b.get("checkpoint") or b.get("backend")
    return name + (" +norm" if meta.get("normalize") else "")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--mode", choices=tuple(MODES), default="choice_fa")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if args.out and (args.out.exists() or (ROOT / "results").resolve() in args.out.resolve().parents):
        p.error("use a new output file outside the historical archive")

    cases = load_cases()
    a = Audit()
    check_dataset(a, cases)
    if not a.problems:
        check_manifest(a, cases)
        check_source(a)
    if a.problems:
        sys.exit("invalid frozen benchmark: " + a.problems[0])
    cols = []
    protocol = None
    for d in args.run_dirs:
        a = Audit()
        meta, rows = check_run(a, cases, d)
        if a.problems:
            sys.exit(f"{d}: audit failed ({a.problems[0]})")
        if args.mode not in meta["modes"]:
            sys.exit(f"{d}: no {args.mode} responses")
        if meta["not_a_model_result"]:
            sys.exit("keyword pipeline results cannot be used as a model baseline")
        current = (meta["case_ids"], meta["repeats"])
        if protocol is not None and protocol != current:
            sys.exit("comparison requires the same cases and repeat count")
        protocol = current
        cols.append((label_for(meta), summarise(cases, meta, rows)["modes"][args.mode]))
    if not cols:
        sys.exit("no comparable runs")

    head = "| | " + " | ".join(name for name, _ in cols) + " |"
    lines = [f"# Comparison ({args.mode}, first repeat)", "", head, "|---" * (len(cols) + 1) + "|"]
    lines.append("| **overall** | " + " | ".join(f"**{s['correct']}/{s['n']}**" for _, s in cols) + " |")
    for fam in FAMILIES:
        cells = []
        for _, s in cols:
            v = s["by_family"].get(fam)
            cells.append(f"{v['correct']}/{v['n']}" if v else "-")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    lines.append("| false cancel | " + " | ".join(str(s["false_cancel"]) for _, s in cols) + " |")
    lines.append("| macro-F1 | " + " | ".join(f"{s['macro_f1']:.3f}" for _, s in cols) + " |")
    lines.append("| ECE | " + " | ".join("-" if s["ece"] is None else f"{s['ece']:.3f}" for _, s in cols) + " |")
    lines.append("| p50 ms | " + " | ".join("-" if s["latency_ms_p50"] is None else f"{s['latency_ms_p50']:.1f}" for _, s in cols) + " |")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
