"""Offline audit. Standard library only: no model, no network, no API keys.

    python audit.py                         # check frozen inputs and all historical runs
    python audit.py --run-dir runs/x        # also check a run and recompute its metrics
    python audit.py --run-dir runs/x --write   # ...and write summary.json / summary.md into it

Exit code is non-zero when anything fails, so it can gate CI.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from common import (FAMILIES, PINNED_FILES, ROOT, load_cases, load_jsonl, load_manifest, pinned_hashes,
                    request_for, sha256_file)
from metrics import repeat_agreement, score
from prompts import LABELS, MODES, QUESTION_ID

REQUIRED_FIELDS = ("id", "family", "label", "text", "rationale", "trap")


class Audit:
    def __init__(self):
        self.problems, self.warnings = [], []

    def fail(self, msg):
        self.problems.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


# --------------------------------------------------------------------------- dataset
def check_dataset(a: Audit, cases: list):
    if not isinstance(cases, list) or len(cases) != 64 or any(not isinstance(c, dict) for c in cases):
        a.fail("dataset must contain 64 case objects")
        return
    for c in cases:
        if any(not isinstance(c.get(f), str) or not c[f].strip()
               for f in REQUIRED_FIELDS if f != "trap"):
            a.fail("case fields must be non-empty strings")
            return
        if c.get("trap") is not None and not isinstance(c["trap"], str):
            a.fail(f"{c['id']}: trap must be a string or null")
            return
    ids = [c.get("id") for c in cases]
    for cid, n in Counter(ids).items():
        if n > 1:
            a.fail(f"duplicate case id {cid}")
    for c in cases:
        missing = [f for f in REQUIRED_FIELDS if f not in c]
        if missing:
            a.fail(f"{c.get('id')}: missing fields {missing}")
        if c.get("label") not in LABELS:
            a.fail(f"{c.get('id')}: label {c.get('label')!r} not in {LABELS}")
        if c.get("family") not in FAMILIES:
            a.fail(f"{c.get('id')}: unknown family {c.get('family')!r}")
        if not str(c.get("text", "")).strip():
            a.fail(f"{c.get('id')}: empty text")
    # balance: every family has every label equally often
    grid = Counter((c.get("family"), c.get("label")) for c in cases)
    per_cell = {grid[(f, lab)] for f in FAMILIES for lab in LABELS}
    if per_cell != {2}:
        a.fail(f"families x labels is not balanced: {dict(grid)}")
    texts = Counter(c.get("text") for c in cases)
    for t, n in texts.items():
        if n > 1:
            a.fail(f"duplicate text: {t[:40]}...")


def check_manifest(a: Audit, cases: list):
    try:
        manifest = load_manifest()
    except (OSError, ValueError):
        a.fail("data/manifest.json is missing or invalid")
        return
    if not isinstance(manifest, dict):
        a.fail("manifest must be an object")
        return
    current = pinned_hashes()
    if not isinstance(manifest.get("sha256"), dict):
        a.fail("manifest sha256 must be an object")
        return
    for rel in PINNED_FILES:
        if manifest.get("sha256", {}).get(rel) != current[rel]:
            a.fail(f"{rel} changed since the manifest was written (hash mismatch)")
    if manifest.get("n_cases") != len(cases):
        a.fail(f"manifest n_cases={manifest.get('n_cases')} but dataset has {len(cases)}")
    if manifest.get("labels") != list(LABELS) or manifest.get("modes") != list(MODES):
        a.fail("manifest labels/modes differ from the frozen prompts")
    if manifest.get("families") != dict(Counter(c["family"] for c in cases)):
        a.fail("manifest family counts differ from the dataset")
    if manifest.get("label_counts") != dict(Counter(c["label"] for c in cases)):
        a.fail("manifest label counts differ from the dataset")
    if not manifest.get("reviewed_by") or manifest["reviewed_by"] == "TODO":
        a.fail("manifest must retain the source's reviewer attribution")


def check_source(a, root=ROOT):
    source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    for item in source["files"]:
        if sha256_file(root / item["path"]) != item["sha256"]:
            a.fail(f"{item['path']}: differs from the frozen source artifact")


def finite(value, low=0, high=float("inf")):
    try:
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    except OverflowError:
        return False


def _historical_run(run_dir):
    # Only the three supplied archives use projected answer rows without full requests/responses.
    source = json.loads((ROOT / "SOURCE.json").read_text(encoding="utf-8"))
    hashes = {f["path"]: f["sha256"] for f in source["files"]}
    for name in ("multilingual", "multilingual_norm", "english"):
        prefix = f"results/v1/{name}/"
        if all(sha256_file(run_dir / f) == hashes[prefix + f]
               for f in ("metadata.json", "responses.jsonl")):
            return True
    return False


# --------------------------------------------------------------------------- run
def check_run(a: Audit, cases: list, run_dir: Path):
    meta_path, resp_path = run_dir / "metadata.json", run_dir / "responses.jsonl"
    if not meta_path.exists() or not resp_path.exists():
        a.fail(f"{run_dir}: needs metadata.json and responses.jsonl")
        return None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = load_jsonl(resp_path)
    except (OSError, ValueError) as exc:
        a.fail(f"{run_dir.name}: cannot read run ({type(exc).__name__})")
        return None, None
    if not isinstance(meta, dict):
        a.fail("metadata must be an object")
        return None, None
    modes, repeats, case_ids = meta.get("modes"), meta.get("repeats"), meta.get("case_ids")
    if (not isinstance(modes, list) or not modes or any(not isinstance(m, str) or m not in MODES for m in modes)
            or len(set(modes)) != len(modes) or type(repeats) is not int or repeats < 1):
        a.fail("metadata needs unique known modes and a positive integer repeat count")
        return None, None
    known = {c["id"]: c for c in cases}
    if (not isinstance(case_ids, list) or not case_ids
            or any(not isinstance(cid, str) or cid not in known for cid in case_ids)
            or len(set(case_ids)) != len(case_ids)
            or type(meta.get("n_cases")) is not int or meta["n_cases"] != len(case_ids)):
        a.fail("metadata case_ids must be unique known cases matching n_cases")
        return None, None
    for field in ("normalize", "smoke_test", "not_a_model_result"):
        if type(meta.get(field)) is not bool:
            a.fail(f"metadata {field} must be a boolean")
    backend = meta.get("backend")
    if not isinstance(backend, dict) or backend.get("backend") not in ("laya", "keyword"):
        a.fail("metadata must identify the laya or keyword backend")
    elif meta.get("not_a_model_result") != (backend["backend"] == "keyword"):
        a.fail("keyword results must be marked not_a_model_result")
    if type(meta.get("errors")) is not int or meta["errors"] < 0:
        a.fail("metadata errors must be a non-negative integer")
    version = meta.get("record_format")
    if version is None:
        if not _historical_run(run_dir):
            a.fail("legacy answer-only format is reserved for the unchanged historical archives")
    elif type(version) is not int or version != 2:
        a.fail("unsupported record_format")
    else:
        partial = len(case_ids) != 64 or repeats != 3 or set(modes) != set(MODES)
        if meta.get("smoke_test") != partial:
            a.fail("smoke_test must identify a partial protocol")
        if type(meta.get("completed_requests")) is not int or meta["completed_requests"] != len(rows):
            a.fail("completed_requests differs from the recorded responses")
    if a.problems:
        return None, None

    if meta.get("sha256") != pinned_hashes():
        a.fail("run was recorded against different data/prompts (sha256 mismatch)")
    if meta.get("smoke_test"):
        a.warn("smoke-test run (--limit): not a full-protocol result")
    if meta.get("not_a_model_result"):
        a.warn("keyword matcher: pipeline check only, not a model result or baseline")

    # Check count before expanding the product, so malformed repeat counts cannot allocate
    # an unbounded expected set for a tiny archive.
    if len(rows) != len(modes) * repeats * len(case_ids):
        a.fail("response count does not match the declared cases, modes and repeats")
        return None, None
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("mode"), str)
                or not isinstance(row.get("case_id"), str) or type(row.get("repeat")) is not int):
            a.fail("response identity must contain a mode, case_id and integer repeat")
            return None, None
    expected = {(m, r, cid) for m in modes for r in range(repeats) for cid in case_ids}
    seen = Counter((row["mode"], row["repeat"], row["case_id"]) for row in rows)
    for key, n in seen.items():
        if n > 1:
            a.fail(f"duplicate response {key}")
        if key not in expected:
            a.fail(f"unexpected response {key}")
    for key in expected - set(seen):
        a.fail(f"missing response {key}")
    if a.problems:
        return None, None

    for row in rows:
        tag = f"{row['case_id']}/{row['mode']}/r{row['repeat']}"
        if not finite(row.get("latency_ms")):
            a.fail(f"{tag}: latency must be finite and non-negative")
        if "error" not in row or row["error"] is not None and (not isinstance(row["error"], str) or not row["error"]):
            a.fail(f"{tag}: error must be null or a non-empty error type")
        if version == 2:
            if row.get("request") != request_for(known[row["case_id"]], row["mode"], meta["normalize"]):
                a.fail(f"{tag}: request differs from the frozen input")
            if "response" not in row:
                a.fail(f"{tag}: raw response is missing")
        if row.get("error") is not None:
            if any(row.get(f) is not None for f in ("prediction", "probabilities", "answer_confidence")):
                a.fail(f"{tag}: failed response must not contain a scored answer")
            continue  # counted as wrong by metrics, not an audit failure
        pred, probs = row.get("prediction"), row.get("probabilities")
        if pred not in LABELS:
            a.fail(f"{tag}: prediction {pred!r} not a label")
            continue
        if not isinstance(probs, dict) or set(probs) != set(LABELS):
            a.fail(f"{tag}: probabilities must cover exactly {LABELS}")
            continue
        vals = list(probs.values())
        if not all(finite(v, high=1) for v in vals):
            a.fail(f"{tag}: non-finite or out-of-range probability")
        elif abs(sum(vals) - 1) > 0.001:  # archive probabilities are rounded to four decimals
            a.fail(f"{tag}: probabilities sum to {sum(vals):.4f}")
        elif probs[pred] < max(vals) - 1e-6:
            a.fail(f"{tag}: prediction is not the argmax")
        conf = row.get("answer_confidence")
        if not finite(conf, high=1):
            a.fail(f"{tag}: answer confidence must be finite and in [0, 1]")
        # The historical SDK used the winning probability. Current Laya may apply a
        # binning calibration map to answer_confidence; format 2 verifies the SDK value below.
        elif version is None and finite(probs[pred], high=1) and abs(conf - probs[pred]) > 1e-6:
            a.fail(f"{tag}: confidence differs from the answer probability")
        if version == 2:
            response = row.get("response")
            answers = response.get("answers") if isinstance(response, dict) else None
            answer = answers.get(QUESTION_ID) if isinstance(answers, dict) else None
            if not isinstance(answer, dict) or answer.get("type") != "choice":
                a.fail(f"{tag}: raw response has no choice answer")
            elif any(answer.get(k) != row.get(v) for k, v in
                     (("choice", "prediction"), ("probabilities", "probabilities"),
                      ("answer_confidence", "answer_confidence"))):
                a.fail(f"{tag}: scored fields differ from the raw response")
    if meta["errors"] != sum(row.get("error") is not None for row in rows):
        a.fail("metadata errors differs from the recorded failures")
    return meta, rows


# --------------------------------------------------------------------------- report
def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def summarise(cases, meta, rows):
    run_cases = [c for c in cases if c["id"] in set(meta.get("case_ids") or [c["id"] for c in cases])]
    out = {"backend": meta.get("backend"), "normalize": meta.get("normalize"), "modes": {}}
    for mode in meta["modes"]:
        s = score(run_cases, rows, mode, repeat=0)
        s["repeat_agreement"] = repeat_agreement(rows, mode)
        out["modes"][mode] = s
    return out


def render_markdown(summary):
    b = summary["backend"] or {}
    title = b.get("checkpoint") or b.get("backend")
    lines = [f"# Results: {title}{' + normalize' if summary.get('normalize') else ''}", ""]
    for mode, s in summary["modes"].items():
        ece = "-" if s["ece"] is None else "%.3f" % s["ece"]
        p50 = "-" if s["latency_ms_p50"] is None else "%.1f ms" % s["latency_ms_p50"]
        lines += [f"## {mode}", "",
                  f"- accuracy **{s['correct']}/{s['n']} ({pct(s['accuracy'])})**, macro-F1 {s['macro_f1']:.3f}, failures {s['failures']}",
                  f"- false cancellation routing: **{s['false_cancel']}**; missed `cancel`: {s['missed_cancel']}",
                  f"- mean answer confidence {pct(s['mean_answer_confidence'])}, ECE {ece}",
                  f"- latency p50 {p50}, repeat agreement {pct(s['repeat_agreement'])}",
                  "", "| family | correct |", "|---|---|"]
        for fam, v in s["by_family"].items():
            lines.append(f"| {fam} | {v['correct']}/{v['n']} |")
        lines += ["", "| gold \\ predicted | " + " | ".join(LABELS) + " | failed |",
                  "|---" * (len(LABELS) + 2) + "|"]
        for g in LABELS:
            row = s["confusion"][g]
            lines.append(f"| {g} | " + " | ".join(str(row.get(p, 0)) for p in LABELS) + f" | {row.get('FAILED', 0)} |")
        lines.append("")
    lines.append("Quality is scored on the first repeat (preselected, not the best). Failures count as wrong.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--write", action="store_true", help="write summary.json and summary.md into the run dir")
    args = p.parse_args(argv)
    if args.write and not args.run_dir:
        p.error("--write requires --run-dir")
    if args.write and (ROOT / "results").resolve() in args.run_dir.resolve().parents:
        p.error("the historical archive is read-only; use a new run directory")

    a = Audit()
    try:
        cases = load_cases()
        check_dataset(a, cases)
        if not a.problems:
            check_manifest(a, cases)
            check_source(a)
        if not a.problems:
            print(f"dataset: {len(cases)} cases, {len(FAMILIES)} families, {len(LABELS)} labels, {len(MODES)} prompt modes")
            dirs = [args.run_dir] if args.run_dir else [ROOT / "results/v1" / name for name in
                                                      ("multilingual", "multilingual_norm", "english")]
            for run_dir in dirs:
                meta, rows = check_run(a, cases, run_dir)
                if meta is None or a.problems:
                    continue
                summary = summarise(cases, meta, rows)
                saved = run_dir / "summary.json"
                if saved.exists() and not args.write:
                    if json.loads(saved.read_text(encoding="utf-8")) != summary:
                        a.fail(f"{run_dir.name}: saved summary differs from recomputed results")
                if args.run_dir:
                    print(render_markdown(summary))
                else:
                    scores = ", ".join(f"{m}: {s['correct']}/{s['n']}" for m, s in summary["modes"].items())
                    print(f"{run_dir.name}: {len(rows)} records; {scores}")
                if args.write:
                    saved.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8", newline="\n")
                    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        a.fail(f"cannot audit artifacts: {type(exc).__name__}: {exc}")

    for w in a.warnings:
        print(f"WARNING: {w}")
    if a.problems:
        for prob in a.problems[:50]:
            print(f"FAIL: {prob}")
        print(f"\naudit FAILED ({len(a.problems)} problems)")
        return 1
    print("audit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
