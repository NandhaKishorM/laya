"""Verify a fresh laya_eval run against committed results — or against another run.

The published tables in README.md and BENCHMARKS.md are only as trustworthy as a
reader's ability to re-derive them. `laya_eval.py` produces the numbers; this
script answers the follow-up question without anyone writing their own differ:
does this run match the file it is compared against, cell by cell?

It understands the three JSON shapes this repository uses for per-language
results:

- a `laya_eval.py` report (`--out` file): top-level `report` mapping
- `research/results/cpu_51_language_sweep.json`: `part_a.by_model.<model>.per_language`
- `research/results/cpu_51_language_sweep_clamped.json`: `per_language.<lang>.<branch>`

Usage:

    # fresh run vs the clamped re-run branch (the default)
    python research/eval/verify_results.py my_run.json \
        research/results/cpu_51_language_sweep_clamped.json

    # fresh multilingual run vs the original sweep
    python research/eval/verify_results.py my_multi.json \
        research/results/cpu_51_language_sweep.json --model-key multilingual

    # device parity: one fresh run against another
    python research/eval/verify_results.py sweep_mps.json sweep_cpu.json

Exit code is 0 when every compared cell is within tolerance and 1 otherwise,
so it can gate a release checklist (the #285 ask).

Offline: stdlib only, no checkpoint, no network.
"""
import argparse
import json
import sys

METRICS = ["accuracy", "macro_f1", "ece", "mean_confidence", "acc_at_50_coverage"]
DEFAULT_TOL = 5e-5  # committed figures are stored rounded to four decimals

BRANCHES = ("committed", "unclamped_rerun", "clamped_rerun")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def extract(doc, model_key="english", branch="clamped_rerun"):
    """Normalise any supported document to {lang: {metric: float}} plus a label."""
    if "report" in doc:
        label = "laya_eval report (device=%s)" % doc.get("config", {}).get("device", "?")
        return {lang: row for lang, row in doc["report"].items()}, label
    if "part_a" in doc:
        models = doc["part_a"]["by_model"]
        if model_key not in models:
            raise SystemExit("no by_model.%s; available: %s" % (model_key, ", ".join(sorted(models))))
        return dict(models[model_key]["per_language"]), "sweep by_model.%s" % model_key
    if "per_language" in doc:
        pl = doc["per_language"]
        sample = next(iter(pl.values()))
        if branch in sample:
            return {lang: row[branch] for lang, row in pl.items() if branch in row}, \
                "clamped file, branch %s" % branch
        return dict(pl), "flat per_language"
    raise SystemExit("unrecognised results document (no report / part_a / per_language)")


def compare(run, ref, metrics, tol):
    """Cell-by-cell diff over languages present in both. Returns (rows, misses)."""
    langs = sorted(set(run) & set(ref))
    only_run = sorted(set(run) - set(ref))
    only_ref = sorted(set(ref) - set(run))
    rows = []
    misses = 0
    for m in metrics:
        exact = 0
        worst_lang, worst = None, 0.0
        for lang in langs:
            if m not in run[lang] or m not in ref[lang]:
                continue
            d = abs(run[lang][m] - ref[lang][m])
            if d < tol:
                exact += 1
            else:
                misses += 1
            if d > worst:
                worst_lang, worst = lang, d
        rows.append((m, exact, len(langs), worst_lang, worst))
    return rows, misses, only_run, only_ref, langs


def macro(report, langs, metric):
    vals = [report[l][metric] for l in langs if metric in report[l]]
    return sum(vals) / len(vals) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="fresh laya_eval --out JSON (left side)")
    ap.add_argument("against", help="results document to verify against (right side)")
    ap.add_argument("--model-key", default="english",
                    help="by_model key for sweep files (default: english)")
    ap.add_argument("--branch", default="clamped_rerun", choices=BRANCHES,
                    help="branch for clamped-format files (default: clamped_rerun)")
    ap.add_argument("--metrics", default=",".join(METRICS),
                    help="comma-separated metrics (default: the five shared ones)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="per-cell tolerance (default: %(default)s, half the last stored digit)")
    args = ap.parse_args(argv)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    run, run_label = extract(load(args.run))
    ref, ref_label = extract(load(args.against), model_key=args.model_key, branch=args.branch)
    rows, misses, only_run, only_ref, langs = compare(run, ref, metrics, args.tol)

    print("%s\n  vs %s" % (run_label, ref_label))
    print("  %d languages compared" % len(langs))
    for m, exact, n, worst_lang, worst in rows:
        print("  %-20s exact %d/%d   worst diff %.4f (%s)" % (m, exact, n, worst, worst_lang))
    for name, missing in (("only in run", only_run), ("only in reference", only_ref)):
        if missing:
            print("  %s: %s" % (name, ",".join(missing)))
    acc = macro(run, langs, "accuracy")
    ece = macro(run, langs, "ece")
    if acc is not None:
        print("  run macro: acc=%.4f ece=%.4f" % (acc, ece))
    if misses:
        print("  FAIL: %d cell(s) beyond tolerance %.1e" % (misses, args.tol))
        return 1
    print("  OK: every compared cell within %.1e" % args.tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
