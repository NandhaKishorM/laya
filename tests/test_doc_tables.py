"""The benchmark tables that cite a committed JSON must still equal it (#298).

`BENCHMARKS.md:274` promises the fast-path comparison "can be re-checked without a GPU"
because every per-option probability is in `benchmarks/results/parity_*.json`. Two cells
did not match it, and the sentence under the table claimed the fast path is never further
from fp32 than the stock path is, which the `laya-multilingual` `noul` row contradicts.

Re-measuring would not have caught either: the run is not stale, the transcription is. So
this reads the published table and the committed JSON and compares them, which is
deterministic and needs no weights, no GPU and no network.

Only tables with a committed artifact are checked. Everything that cannot be checked from
this repository is listed at the end rather than silently skipped, so the gap stays visible.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

PASS: List[str] = []
FAIL: List[str] = []


def check(what: str, got: Any, want: Any) -> None:
    if got == want:
        PASS.append(what)
    else:
        FAIL.append("%s: got %r, want %r" % (what, got, want))


def check_true(what: str, cond: bool, detail: Any = "") -> None:
    if cond:
        PASS.append(what)
    else:
        FAIL.append("%s: %s" % (what, detail))


def read(path: str) -> str:
    # newline="" so a CRLF checkout and an LF checkout parse identically
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def rounded(value: float, places: int = 3) -> str:
    """Format the way a person transcribing a run would: nearest, halves away from zero.

    `round()` is not usable here. It rounds halves to even, so `round(0.0005, 3)` is `0.0`
    and `round(0.0095, 3)` is `0.009`, either of which would report a correct cell as
    drifted. The table's own `0.045` from `0.045488` and `0.010` from `0.009501` show the
    convention actually used is the ordinary one.
    """
    from decimal import ROUND_HALF_UP, Decimal

    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP))


# --------------------------------------------------------------- the published parity table
BENCHMARKS = "BENCHMARKS.md"
PARITY_DIR = os.path.join("benchmarks", "results")

# checkpoint -> the file its row cites, and the row's own name for the model
SOURCES = {
    "laya": "parity_english_rtx4070.json",
    "laya-multilingual": "parity_multilingual_rtx4070.json",
}

HEADER = ("| checkpoint | type | n | max \\|p_fast - p_stock\\| | max \\|p_fast - p_fp32\\| "
          "| max \\|p_stock - p_fp32\\| | argmax fast = stock | fast = fp32 |")

# (checkpoint, backend, dtype) -> the file that row of the per-backend table cites
BACKEND_SOURCES = {
    ("laya", "tilelang", "bf16"): "parity_english_rtx4070.json",
    ("laya-multilingual", "tilelang", "bf16"): "parity_multilingual_rtx4070.json",
    ("laya", "compile", "bf16"): "parity_english_compile_rtx4070.json",
    ("laya-multilingual", "compile", "bf16"): "parity_multilingual_compile_rtx4070.json",
    ("laya-multilingual", "compile", "fp16"): "parity_multilingual_compile_fp16_rtx4070.json",
    ("laya-multilingual", "onnx", "fp32"): "parity_multilingual_onnx_cpu.json",
}
BACKEND_HEADER = ("| checkpoint | backend | dtype | type | n | max \\|p - p_stock\\| | max \\|p - p_fp32\\| "
                  "| max \\|p_stock - p_fp32\\| | argmax = stock | = fp32 |")


def split_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def unstyled(cell: str) -> str:
    """The table bolds the columns it wants read first; the value is what matters."""
    return cell.replace("**", "").replace("\\|", "|").strip()


def parse_table(text: str, header: str) -> List[List[str]]:
    lines = text.split("\n")
    try:
        start = lines.index(header)
    except ValueError:
        return []
    rows = []
    for line in lines[start + 2:]:                       # skip the |---| separator
        if not line.startswith("|"):
            break
        rows.append(split_row(line))
    return rows


def main() -> int:
    for path in (BENCHMARKS, PARITY_DIR):
        check_true("%s exists" % path, os.path.exists(path), "missing")

    rows = parse_table(read(BENCHMARKS), HEADER)
    check_true("parity table/parsed six rows", len(rows) == 6, "got %d" % len(rows))
    check_true("parity table/every row has eight cells",
               all(len(r) == 8 for r in rows), [len(r) for r in rows])

    summaries: Dict[str, Dict[str, Any]] = {}
    for ckpt, name in SOURCES.items():
        with open(os.path.join(PARITY_DIR, name), encoding="utf-8") as fh:
            summaries[ckpt] = json.load(fh)["summary"]

    cols = ("d_fast_stock", "d_fast_fp32", "d_stock_fp32")
    for row in rows:
        ckpt, qtype = row[0], row[1]
        if ckpt not in summaries:
            FAIL.append("parity table/unknown checkpoint %r" % ckpt)
            continue
        summary = summaries[ckpt].get(qtype)
        if summary is None:
            FAIL.append("parity table/%s has no %r summary in its JSON" % (ckpt, qtype))
            continue

        label = "%s/%s" % (ckpt, qtype)
        check("parity table/%s n" % label, int(row[2]), summary["n"])
        for cell, key in zip(row[3:6], cols):
            check("parity table/%s %s" % (label, key), unstyled(cell), rounded(summary[key]))
        for cell, key in zip(row[6:8], ("agree_fast_stock", "agree_fast_fp32")):
            check("parity table/%s %s" % (label, key), unstyled(cell),
                  "%d/%d" % (summary[key], summary["n"]))

    # ------------------------------------------------- the sentence under the table
    # BENCHMARKS.md and README.md both asserted the fast path is never further from fp32 than
    # the stock bf16 path is. That is checkable from the same JSON, and it was false on one
    # row, so the sentence now states the bound the data actually supports instead. What is
    # asserted here is that bound, and that the row which falsified the old claim is still
    # the one the new wording accounts for.
    for ckpt, summary in summaries.items():
        for qtype, s in summary.items():
            if not isinstance(s, dict):
                continue
            worst = max(s["d_fast_fp32"], s["d_stock_fp32"], s["d_fast_stock"])
            check_true("parity bound/%s/%s within the stated 0.076" % (ckpt, qtype),
                       worst <= 0.076 + 5e-4, worst)
            # the per-row agreement counts are checked above against the table itself; this
            # pins the floor the prose quotes, which is the worst row in the set (47/48)
            check_true("parity bound/%s/%s agreement floor" % (ckpt, qtype),
                       min(s["agree_fast_fp32"], s["agree_fast_stock"]) >= 47,
                       (s["agree_fast_fp32"], s["agree_fast_stock"]))

    # the old wording was "at least as close to fp32 as the stock bf16 path is"; this records
    # that it is not universally true, so a future edit cannot quietly restore it
    contradicted = [(c, q) for c, s in summaries.items() for q, v in s.items()
                    if isinstance(v, dict) and v["d_fast_fp32"] > v["d_stock_fp32"]]
    check("the row that falsified the old sentence is still the multilingual noul one",
          contradicted, [("laya-multilingual", "noul")])

    # ------------------------------------------------- the per-backend parity table
    # `benchmarks/parity.py --backend ...` writes the same JSON shape for every backend ("fast"
    # is the backend under test), so the table BENCHMARKS.md prints per backend is checked the
    # same way, cell by cell, against the file each row cites.
    rows = parse_table(read(BENCHMARKS), BACKEND_HEADER)
    check_true("backend table/present", len(rows) >= 1, "no rows under the backend header")
    check_true("backend table/every row has ten cells", all(len(r) == 10 for r in rows), [len(r) for r in rows])
    seen = set()
    for row in rows:
        ckpt, backend, dtype, qtype = row[0], unstyled(row[1]).strip("`"), row[2], row[3]
        name = BACKEND_SOURCES.get((ckpt, backend, dtype))
        if name is None:
            FAIL.append("backend table/no committed JSON for %s %s %s" % (ckpt, backend, dtype))
            continue
        path = os.path.join(PARITY_DIR, name)
        if not os.path.exists(path):
            FAIL.append("backend table/%s is missing" % path)
            continue
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        label = "%s/%s/%s/%s" % (ckpt, backend, dtype, qtype)
        check("backend table/%s file names the backend" % label, doc.get("backend", "tilelang"), backend)
        summary = doc["summary"].get(qtype)
        if summary is None:
            FAIL.append("backend table/%s has no %r summary" % (label, qtype))
            continue
        check("backend table/%s n" % label, int(row[4]), summary["n"])
        for cell, key in zip(row[5:8], cols):
            check("backend table/%s %s" % (label, key), unstyled(cell), rounded(summary[key]))
        for cell, key in zip(row[8:10], ("agree_fast_stock", "agree_fast_fp32")):
            check("backend table/%s %s" % (label, key), unstyled(cell), "%d/%d" % (summary[key], summary["n"]))
        seen.add((ckpt, backend, dtype))
    check("backend table/every committed backend run is in the table", sorted(seen), sorted(BACKEND_SOURCES))

    # ------------------------------------------------- what this cannot check
    unbacked = [
        ("README.md typed-decisions rows", "no committed artifact for the fine-tuned checkpoint"),
        ("README.md / BENCHMARKS.md 103-332 q/s throughput", "hand-picked from a range spanning 55-553"),
        ("BENCHMARKS.md fast-path latency table", "the committed fast_*.json have no 'eval' key"),
        ("README.md CPU/T4 reload latencies", "research/latency_benchmark_results.json is absent"),
    ]
    check_true("unbacked tables are reported, not skipped", len(unbacked) == 4, unbacked)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    if not FAIL:
        print("every cell of the parity table matches its committed JSON")
        print("not checkable from this repository (not asserted either way):")
        for name, why in unbacked:
            print("  - %s: %s" % (name, why))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
