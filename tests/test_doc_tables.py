"""The benchmark tables that cite a committed JSON must still equal it (#298).

`BENCHMARKS.md:274` promises the fast-path comparison "can be re-checked without a GPU"
because every per-option probability is in `benchmarks/results/parity_*.json`. Two cells
did not match it, and the sentence under the table claimed the fast path is never further
from fp32 than the stock path is, which the `laya-multilingual` `noul` row contradicts.

Re-measuring would not have caught either: the run is not stale, the transcription is. So
this reads the published table and the committed JSON and compares them, which is
deterministic and needs no weights, no GPU and no network.

The headline "English vs the rest" figures were the exception until #776: three tables publish
them, none named its artifact or its protocol, and no check tied them to the run that produced
them. They now name it, and the cells are checked here.

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


FP16_HEADER = ("| checkpoint | type | n | max \\|p_fast - p_fp32\\| bf16 | max \\|p_fast - p_fp32\\| fp16 "
               "| argmax fast = fp32, bf16 | fp16 |")


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

    # ------------------------------------------------- the fp16 table (same files, plus fp16 runs)
    fp16_sources = {
        "laya": ("parity_english_rtx4070.json", "parity_english_fp16_rtx4070.json"),
        "laya-multilingual": ("parity_multilingual_rtx4070.json", "parity_multilingual_fp16_rtx4070.json"),
        "laya-typed-decisions": ("parity_typed_decisions_rtx4070.json", "parity_typed_decisions_fp16_rtx4070.json"),
    }
    pairs: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for ckpt, (bf, fp) in fp16_sources.items():
        loaded = []
        for name in (bf, fp):
            with open(os.path.join(PARITY_DIR, name), encoding="utf-8") as fh:
                loaded.append(json.load(fh))
        check("fp16 table/%s bf16 file is a bf16 run" % ckpt, loaded[0]["dtype"], "torch.bfloat16")
        check("fp16 table/%s fp16 file is an fp16 run" % ckpt, loaded[1]["dtype"], "torch.float16")
        pairs[ckpt] = (loaded[0]["summary"], loaded[1]["summary"])
    rows16 = parse_table(read(BENCHMARKS), FP16_HEADER)
    check_true("fp16 table/parsed nine rows", len(rows16) == 9, "got %d" % len(rows16))
    for row in rows16:
        ckpt, qtype = row[0], row[1]
        if ckpt not in pairs or qtype not in pairs[ckpt][0] or qtype not in pairs[ckpt][1]:
            FAIL.append("fp16 table/unknown row %s/%s" % (ckpt, qtype))
            continue
        b, f = pairs[ckpt][0][qtype], pairs[ckpt][1][qtype]
        label = "fp16 table/%s/%s" % (ckpt, qtype)
        check(label + " n", int(row[2]), f["n"])
        check(label + " bf16 d_fast_fp32", unstyled(row[3]), rounded(b["d_fast_fp32"]))
        check(label + " fp16 d_fast_fp32", unstyled(row[4]), rounded(f["d_fast_fp32"]))
        check(label + " bf16 agree_fast_fp32", unstyled(row[5]), "%d/%d" % (b["agree_fast_fp32"], b["n"]))
        check(label + " fp16 agree_fast_fp32", unstyled(row[6]), "%d/%d" % (f["agree_fast_fp32"], f["n"]))
    # the prose under it: fp16 fast agrees with fp16 stock everywhere, and moves nothing by more than 0.009
    fp16_rows = [v for _, f in pairs.values() for v in f.values() if isinstance(v, dict)]
    check("fp16 prose/argmax fast = stock on every question",
          sum(v["agree_fast_stock"] for v in fp16_rows), sum(v["n"] for v in fp16_rows))
    check("fp16 prose/max |fast - stock| rounds to 0.009",
          rounded(max(v["d_fast_stock"] for v in fp16_rows)), "0.009")
    check("fp16 prose/README's 'within 0.009 of fp32'",
          rounded(max(v["d_fast_fp32"] for v in fp16_rows)), "0.009")
    # ---------------------------- the published headline: English vs the rest (#776)
    # Three tables publish these figures -- README.md twice ("Why Route: The Evidence" and the
    # 51-language section) and BENCHMARKS.md once ("English vs the rest") -- and every one of
    # them comes from `research/results/t4_colab_benchmark.json`. None of the three said which
    # artifact or protocol produced it, and none was checked here. #776 re-ran the four README
    # figures with an independently written harness and landed within 0.0003, which is the
    # table's own rounding to three places; that is the second opinion. What has to hold
    # without a GPU is the transcription: each printed cell is recomputed from the run's
    # per-language suites and compared, and the run's own `*_english_vs_rest` summary is
    # compared against the same recomputation, so an aggregation change cannot pass silently.
    # README's `Router` (Routed) column is deliberately not pinned: no routed run is in the
    # artifact, so a check on it would be arithmetic on the other two columns, not evidence.
    T4 = os.path.join("research", "results", "t4_colab_benchmark.json")
    if not os.path.exists(T4):
        FAIL.append("headline/%s is missing, so three published tables are unbacked" % T4)
    else:
        with open(T4, encoding="utf-8") as fh:
            t4 = json.load(fh)
        t4_suites, t4_summary, t4_meta = t4["suites"], t4["summary"], t4["meta"]
        T4_MODELS = ("laya", "laya-multilingual")
        FAMILIES = ("massive_intent", "massive_scenario", "xnli")

        # the protocol, read out of the run rather than transcribed from prose
        check("headline/protocol seed", t4_meta["seed"], 13)
        check("headline/protocol options per choice question", t4_meta["n_opts"], 20)
        check("headline/protocol cases per language", t4_meta["per_lang"], 300)

        LANG_SUITES = {n: s for n, s in t4_suites.items() if n.split(".")[0] in FAMILIES}
        # Averaging the languages and pooling the questions are the same number only because
        # each language contributed exactly `per_lang` cases -- which is what lets the prose
        # say "average" unqualified, so it is pinned rather than assumed.
        wrong_size = ["%s/%s=%d" % (n, m, s[m]["calibrated"]["n"])
                      for n, s in LANG_SUITES.items() for m in T4_MODELS
                      if s[m]["calibrated"]["n"] != t4_meta["per_lang"]]
        check("headline/every language suite holds per_lang questions", wrong_size, [])

        def language_group(family: str, model: str, english: bool) -> Tuple[float, int]:
            names = sorted(n for n in LANG_SUITES
                           if n.startswith(family + ".") and n.endswith(".en") == english)
            accs = [t4_suites[n][model]["calibrated"]["accuracy"] for n in names]
            return sum(accs) / len(accs), len(names)

        for family in FAMILIES:
            fam_suites = [n for n in LANG_SUITES if n.startswith(family + ".")]
            for model in T4_MODELS:
                for english, group in ((True, "english"), (False, "non_english")):
                    mine, count = language_group(family, model, english)
                    stored = t4_summary["%s_english_vs_rest" % family][group][model]
                    check_true("headline/artifact agrees with the per-suite recompute/%s/%s/%s"
                               % (family, model, group), abs(mine - stored) < 5e-5,
                               "recomputed %.6f, stored %s" % (mine, stored))
                check("headline/%s/%s suite count" % (family, model),
                      t4_summary[family][model]["n_suites"], len(fam_suites))
                check("headline/%s/%s question count" % (family, model),
                      t4_summary[family][model]["n_questions"],
                      sum(t4_suites[n][model]["calibrated"]["n"] for n in fam_suites))

        # The printed tables. `default_family` covers the README 51-language table, whose two
        # MASSIVE rows are labelled only "English" and "13 other languages"; the section
        # heading supplies the family there, and no other table shares these headers.
        HEADLINE_TABLES = [
            ("README.md", "| Benchmark / Task | English (`laya`) | Multilingual (`laya-multilingual`) | `Router` (Routed) |",
             (1, 2), None, 4),
            ("README.md", "| | `laya` | `laya-multilingual` |", (1, 2), "massive_intent", 4),
            ("BENCHMARKS.md", "| task | laya | laya-multilingual |", (1, 2), None, 6),
        ]
        cells_checked = 0
        for path, header, cols, default_family, want_rows in HEADLINE_TABLES:
            found = []
            for row in parse_table(read(path), header):
                label = unstyled(row[0]).lower()
                family = default_family
                for key, name in (("massive scenario", "massive_scenario"),
                                  ("massive intent", "massive_intent"), ("xnli", "xnli")):
                    if key in label:
                        family = name
                if family is not None:
                    found.append((label, family, row))
            where = "%s/%s-family" % (path, default_family or "named")
            check_true("headline/table %s parsed" % where, len(found) == want_rows,
                       "got %d figure rows" % len(found))
            for label, family, row in found:
                stated = re.search(r"(\d+) other", label)
                if stated:
                    _, count = language_group(family, T4_MODELS[0], False)
                    check("headline/%s %r language count" % (where, label),
                          int(stated.group(1)), count)
                for col, model in zip(cols, T4_MODELS):
                    mine, _ = language_group(family, model, "other" not in label)
                    check("headline/%s %s/%s" % (where, label, model),
                          unstyled(row[col]), rounded(mine))
                    cells_checked += 1
        check("headline/published cells compared", cells_checked, 28)

        total = {m: sum(s[m]["calibrated"]["n"] for s in t4_suites.values()) for m in T4_MODELS}
        check_true("headline/both checkpoints answered identical question counts",
                   total["laya"] == total["laya-multilingual"], total)
        stated_total = re.search(r"\(([\d,]+) questions, one T4 GPU", read("README.md"))
        check_true("headline/README states a question denominator", stated_total is not None,
                   "the '(N questions, one T4 GPU)' caption moved")
        if stated_total:
            check("headline/README's denominator equals the run",
                  int(stated_total.group(1).replace(",", "")), total["laya"])

    # ------------------------------------- the 51-language table (#208)
    # The multilingual half of the committed sweep does not reproduce on current code, so the
    # table prints the refreshed re-run instead. That makes `cpu_51_language_sweep_refreshed.json`
    # the artifact behind every cell, and it is checkable here: the `laya` column must still
    # equal the committed file (it reproduces exactly, which is what makes the multilingual
    # divergence interesting rather than a harness change), and the `laya-multilingual` column
    # must equal the refresh.
    REFRESH = os.path.join("research", "results", "cpu_51_language_sweep_refreshed.json")
    if os.path.exists(REFRESH):
        with open(REFRESH, encoding="utf-8") as fh:
            refreshed = json.load(fh)
        committed_sweep = json.loads(read(os.path.join("research", "results",
                                                       "cpu_51_language_sweep.json")))
        en = refreshed["by_model"]["english"]
        ml = refreshed["by_model"]["multilingual"]

        # the refreshed `laya` column is the committed one, per language
        same = [lg for lg in en["per_language"]
                if abs(en["per_language"][lg]["refreshed_clamped"]["accuracy"]
                       - committed_sweep["part_a"]["by_model"]["english"]["per_language"][lg]["accuracy"]) > 5e-5]
        check("51-language/laya reproduces the committed file exactly", same, [])

        # ...and the multilingual one does not, which is why the refresh exists
        differ = [lg for lg in ml["per_language"]
                  if abs(ml["per_language"][lg]["refreshed_clamped"]["accuracy"]
                         - ml["per_language"][lg]["committed"]["accuracy"]) > 5e-5]
        check_true("51-language/multilingual differs from committed in most languages",
                   len(differ) > 40, "%d of 51 differ" % len(differ))

        bench = read(BENCHMARKS)
        header = "| lang | laya | laya-multilingual | \u0394 | laya ECE | multilingual ECE |"
        start = bench.find(header)
        check_true("51-language/table found in BENCHMARKS.md", start != -1, "header not found")
        if start != -1:
            body = bench[start:bench.find("</details>", start)]
            rows = [l for l in body.splitlines() if l.startswith("| `")]
            check("51-language/table has one row per language", len(rows), 51)
            mismatched = []
            for row in rows:
                cells = [c.strip() for c in row.strip("|").split("|")]
                lg = cells[0].strip("`")
                e = en["per_language"].get(lg, {}).get("refreshed_clamped")
                m = ml["per_language"].get(lg, {}).get("refreshed_clamped")
                if e is None or m is None:
                    mismatched.append("%s: not in the artifact" % lg)
                    continue
                want = ["%.3f" % e["accuracy"], "%.3f" % m["accuracy"],
                        "%+.3f" % (m["accuracy"] - e["accuracy"]),
                        "%.3f" % e["ece"], "%.3f" % m["ece"]]
                if cells[1:] != want:
                    mismatched.append("%s: %s != %s" % (lg, cells[1:], want))
            check("51-language/every cell matches the refreshed artifact", mismatched, [])
    else:
        FAIL.append("51-language/%s is missing, so the table is unbacked" % REFRESH)

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
