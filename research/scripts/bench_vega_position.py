"""Vega, not the classic encoder: accuracy against evidence position at a FIXED length.

`bench_position_sensitivity.py` (PR #697) measured `laya-multilingual`. Since 0.5.0,
`laya.Agent()` loads the Vega engine, whose reader takes 73,728 tokens whole. This script asks
whether the evidence-position effect that #697 found survives that reader.

It reuses the same 20 requests, filler, question and "\n\nActual request: " delimiter, so a
cell at `needle_position=1.0` is the cell #697 measures. Documents are sized with the classic
tokenizer (the same one #697 used); the Vega reader's own `usage["input_tokens"]` and
`usage["truncated"]` are reported for every case, because the two tokenizers do not agree.

    python research/scripts/bench_vega_position.py --device cuda \
        --out research/results/vega_position_sensitivity.json

Every prediction is written to the JSON, so the table re-derives without the model:

    python research/scripts/bench_vega_position.py --analyse \
        research/results/vega_position_sensitivity.json

The default top cell is ~48k filler tokens: on an 8 GB RTX 4070 Laptop the 72k nominal cell
OOMs even with `flash-linear-attention` installed (6.78 GiB in use, 276 MiB short), so the
committed run uses 48k and records that attempt in `run_note`. The question, options and
wrapper share the reader's 73,728-token window; a cell whose `truncated` count is above zero
is a truncation result, not a long-context result, and the JSON records it. The 7k cell is the
paired control: at 7k both engines read the whole document, so a difference there is an engine
difference, not truncation.

On an 8 GB card install `flash-linear-attention`; without it, Qwen3.5's reference
gated-delta-rule fallback upcasts and OOMs even at 48k. `pip install flash-linear-attention`.
"""
import argparse, json, math, os, platform, statistics, sys, time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import transformers
from transformers import AutoTokenizer
import laya

try:
    import vegaml
except ImportError:  # the classic arm still runs; the vega arm says what to install
    vegaml = None

QUESTIONS = {"department": {"type": "choice", "instructions": "Which department should handle this request?",
                            "criteria": {"billing": "invoices, payments, refunds",
                                         "technical": "bugs, outages, system errors",
                                         "sales": "pricing, new contracts, plan upgrades",
                                         "other": "everything else"}}}

# Identical to bench_long_context.py: 20 requests, English + 7 other languages.
REQUESTS = [
    ("I was charged twice for invoice 4411, please refund the duplicate payment.", "billing"),
    ("The dashboard crashes with an error every time I open the reports page.", "technical"),
    ("What would an enterprise contract for 200 seats cost per year?", "sales"),
    ("Please refund my subscription payment from last month, it was billed by mistake.", "billing"),
    ("Our API returns 500 errors since this morning and the service is down.", "technical"),
    ("Can you send me pricing for upgrading our plan to the business tier?", "sales"),
    ("Me cobraron dos veces la factura de marzo, devuélvanme el cargo duplicado.", "billing"),
    ("La aplicación se cierra cada vez que abro la configuración.", "technical"),
    ("Quisiera una cotización para un contrato anual de 50 licencias.", "sales"),
    ("Fui cobrado duas vezes na fatura de março, quero o reembolso da cobrança duplicada.", "billing"),
    ("O sistema cai toda vez que tento gerar o relatório mensal.", "technical"),
    ("J'ai été facturé deux fois ce mois-ci, merci de rembourser le doublon.", "billing"),
    ("L'application plante dès que j'ouvre la page des paramètres.", "technical"),
    ("Ich wurde zweimal belastet, bitte erstatten Sie die doppelte Zahlung.", "billing"),
    ("Die Anwendung stürzt jedes Mal ab, wenn ich die Einstellungen öffne.", "technical"),
    ("Was kostet ein Jahresvertrag für 100 Nutzer?", "sales"),
    ("मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।", "billing"),
    ("ऐप हर बार सेटिंग्स खोलते ही बंद हो जाता है।", "technical"),
    ("請求書が二重に請求されました。重複分を返金してください。", "billing"),
    ("تم خصم المبلغ مرتين من بطاقتي، أرجو استرداد المبلغ المكرر.", "billing"),
]
FILLER = ("Thanks for the update on the quarterly planning meeting. We reviewed the roadmap slides, discussed "
          "hiring for the design team, agreed on the offsite venue, and noted that the parking garage will be "
          "closed next week. ")


def place(reps, frac, request):
    """bench_long_context.py's document, with the request moved to fraction `frac`.

    Built from the same pieces as `bench_long_context.py` -- whole repeats of FILLER and the
    same "\n\nActual request: " delimiter -- so that `frac=1.0` produces a byte-identical
    string. That matters: the filler is not token-truncated there either, and the delimiter is
    a semantic cue, so a construction that omits either lands on different numbers for the same
    nominal cell and cannot be compared with the existing bench.
    """
    marker = "\n\nActual request: " if reps else ""
    cut = int(round(reps * frac))
    head, tail = FILLER * cut, FILLER * (reps - cut)
    text = head + marker + request
    if tail:                                  # keeps frac=1.0 identical to bench_long_context.py
        text += "\n\n" + tail
    return text


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def exact_mcnemar(b, c):
    """Exact two-sided McNemar p for b discordant pairs one way and c the other."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, p), 6)


def _prediction(arm, item, gold, state, result, latency_s):
    """One answer in the shape the JSON keeps, from either arm's result."""
    ans = result["answers"]["department"]
    usage = result.get("usage") or {}
    choice = ans.get("choice")
    probs = ans.get("probabilities") or {}
    return {
        "arm": arm,
        "item": item,
        "gold": gold,
        "choice": choice,
        "p_gold": round(float(probs.get(gold, 0.0)), 6),
        "correct": choice == gold,
        "abstain": ans.get("abstain"),
        "route": ans.get("route"),
        "prediction_set": ans.get("prediction_set"),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "truncated": bool(usage.get("truncated")),
        "latency_s": round(latency_s, 4),
        "doc_chars": len(state),
    }


def summarise(arm, pad, frac, preds):
    """Per-cell summary; every number is recomputable from `preds`."""
    n = len(preds)
    all_labels = [g for _, g in REQUESTS]
    majority_all = max(all_labels.count(lab) for lab in set(all_labels)) / len(all_labels)
    correct = sum(1 for p in preds if p["correct"])
    choices = [p["choice"] for p in preds]
    modal = max((choices.count(c) for c in set(choices)), default=0) / n if n else 0.0
    labels = sorted({p["gold"] for p in preds})
    recall = {}
    for lab in labels:
        rows = [p for p in preds if p["gold"] == lab]
        recall[lab] = round(sum(1 for p in rows if p["correct"]) / len(rows), 4) if rows else None
    lat = [p["latency_s"] for p in preds]
    toks = [p["input_tokens"] for p in preds]
    abstain = [p for p in preds if p.get("abstain") is True]
    review = [p for p in preds if p.get("route") == "review"]
    return {
        "arm": arm,
        "pad_tokens": pad,
        "needle_position": frac,
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "wilson_ci95": wilson(correct, n),
        "majority_class": round(majority_all, 4),
        "per_class_recall": recall,
        "modal_prediction_share": round(modal, 4),
        "abstain_rate": round(len(abstain) / n, 4) if n else 0.0,
        "review_rate": round(len(review) / n, 4) if n else 0.0,
        "truncated_count": sum(1 for p in preds if p["truncated"]),
        "median_latency_s": round(statistics.median(lat), 4) if lat else 0.0,
        "median_input_tokens": int(statistics.median(toks)) if toks else 0,
        "predictions": preds,
    }


def _paired(a, b):
    """Discordant counts and exact p for two cells on the same items."""
    am = {p["item"]: p["correct"] for p in a["predictions"]}
    bm = {p["item"]: p["correct"] for p in b["predictions"]}
    shared = sorted(set(am) & set(bm))
    only_a = sum(1 for i in shared if am[i] and not bm[i])
    only_b = sum(1 for i in shared if bm[i] and not am[i])
    return len(shared), only_a, only_b, exact_mcnemar(only_a, only_b)


def comparisons(rows):
    """The four pre-specified paired tests, and only those.

    Position is tested at the top length (start vs middle, middle vs end); length is tested at
    the end position (7k vs 32k, 32k vs top). Every other cell is descriptive. Testing every
    pair would be 14 comparisons at n=20 and would not survive any multiplicity correction.
    """
    out = []
    by = {(r["arm"], r["pad_tokens"], r["needle_position"]): r for r in rows}
    vega = [r for r in rows if r["arm"] == "vega"]
    if not vega:
        return out
    pads = sorted({r["pad_tokens"] for r in vega})
    positions = sorted({r["needle_position"] for r in vega})
    if len(pads) < 3 or len(positions) < 3:
        return out
    top, mid = pads[-1], pads[len(pads) // 2]
    for lo, hi in [(0.0, 0.5), (0.5, 1.0)]:
        a, b = by.get(("vega", top, lo)), by.get(("vega", top, hi))
        if a and b:
            n, only_a, only_b, p = _paired(a, b)
            out.append({"arm": "vega", "axis": "position", "pad_tokens": top, "low": lo, "high": hi,
                        "n_paired": n, "low_only_correct": only_a, "high_only_correct": only_b,
                        "p_exact_mcnemar": p, "pre_specified": True})
    for lo, hi in [(pads[0], mid), (mid, top)]:
        a, b = by.get(("vega", lo, 1.0)), by.get(("vega", hi, 1.0))
        if a and b:
            n, only_a, only_b, p = _paired(a, b)
            out.append({"arm": "vega", "axis": "length", "needle_position": 1.0, "low": lo, "high": hi,
                        "n_paired": n, "low_only_correct": only_a, "high_only_correct": only_b,
                        "p_exact_mcnemar": p, "pre_specified": True})
    return out


def print_rows(rows):
    print("arm     pad    pos    n  acc    CI95            maj   modal abstain review trunc  med_tok  med_s")
    for r in rows:
        print("%-7s %-6d %-5.2f %3d  %.3f  [%.3f,%.3f]  %.2f  %.2f  %.2f    %.2f   %d      %6d   %.3f"
              % (r["arm"], r["pad_tokens"], r["needle_position"], r["n"], r["accuracy"],
                 r["wilson_ci95"][0], r["wilson_ci95"][1], r["majority_class"],
                 r["modal_prediction_share"], r["abstain_rate"], r["review_rate"],
                 r["truncated_count"], r["median_input_tokens"], r["median_latency_s"]))
    print()
    print("Pre-specified paired comparisons (exact McNemar):")
    for c in comparisons(rows):
        where = ("pad=%d" % c["pad_tokens"]) if c["axis"] == "position" else ("pos=%.2f" % c["needle_position"])
        print("  %-7s %-8s %s %s vs %s  n=%d  low-only=%d high-only=%d  p=%.4g"
              % (c["arm"], c["axis"], where, c["low"], c["high"], c["n_paired"],
                 c["low_only_correct"], c["high_only_correct"], c["p_exact_mcnemar"]))


def analyse(path):
    data = json.loads(Path(path).read_text())
    rows = data["rows"]
    print_rows(rows)
    print("\nmetadata: %s" % json.dumps({k: data.get(k) for k in
          ("laya", "vegaml", "torch", "transformers", "device", "platform", "classic_limit", "vega_context_tokens")},
          indent=1, ensure_ascii=False))


def build_agent(args, arm):
    if arm == "vega":
        kw = {"vegaml": args.vega_model or True, "device": args.device}
        if args.revision:
            kw["revision"] = args.revision
        if args.vega_max_len:
            kw["max_len"] = args.vega_max_len
        return laya.Agent(**kw)
    return laya.load(args.classic_model, subfolder=args.classic_subfolder, device=args.device,
                     revision=args.revision) if args.revision else \
        laya.load(args.classic_model, subfolder=args.classic_subfolder, device=args.device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lengths", default="7000,32000,48000",
                    help="Vega filler tokens, sized with the classic tokenizer (default: 7k,32k,48k)")
    ap.add_argument("--classic-lengths", default="7000",
                    help="classic-arm filler tokens (default: 7k; 8,192 would cut the head)")
    ap.add_argument("--positions", default="0.0,0.5,1.0")
    ap.add_argument("--n", type=int, default=20, help="requests to run (default: all 20)")
    ap.add_argument("--arms", default="vega,classic", help="comma-separated: vega,classic")
    ap.add_argument("--out", default="research/results/vega_position_sensitivity.json")
    ap.add_argument("--analyse", metavar="JSON", help="print a committed result without loading a model")
    ap.add_argument("--tokenizer", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default="multilingual/tokenizer",
                    help="tokenizer subfolder (default: multilingual/tokenizer)")
    ap.add_argument("--classic-model", default="convaiinnovations/laya")
    ap.add_argument("--classic-subfolder", default="multilingual",
                    help="checkpoint subfolder (default: multilingual)")
    ap.add_argument("--classic-limit", type=int, default=8192)
    ap.add_argument("--vega-model", default=None, help="default: the shipped vega-U reader")
    ap.add_argument("--vega-max-len", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--run-note", default="",
                    help="free-text note recorded in the JSON (e.g. an OOM fallback)")
    ap.add_argument("--smoke", action="store_true",
                    help="one request at the top length, end position, Vega only")
    a = ap.parse_args()

    if a.analyse:
        analyse(a.analyse)
        return
    if a.smoke:
        a.n, a.lengths, a.positions, a.arms = 1, "48000", "1.0", "vega"

    lengths = [int(x) for x in a.lengths.split(",")]
    classic_lengths = [int(x) for x in a.classic_lengths.split(",")]
    positions = [float(x) for x in a.positions.split(",")]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    requests = REQUESTS[:a.n]

    tok = AutoTokenizer.from_pretrained(a.tokenizer, subfolder=a.subfolder)
    per_rep = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    labels = [g for _, g in REQUESTS]
    majority = max(labels.count(lab) for lab in set(labels)) / len(labels)

    meta = {
        "checkpoint": {"vega": None,
                       "classic": a.classic_model + "/" + a.classic_subfolder if "classic" in arms else None},
        "device": a.device,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "laya": getattr(laya, "__version__", None),
        "vegaml": getattr(vegaml, "__version__", None) if vegaml else None,
        "classic_limit": a.classic_limit,
        "lengths": lengths,
        "classic_lengths": classic_lengths,
        "vega_context_tokens": None,
        "tokenizer": a.tokenizer + "/" + a.subfolder,
        "filler_tokens_per_rep": per_rep,
        "questions": QUESTIONS,
        "filler": FILLER,
        "label_counts": {lab: labels.count(lab) for lab in sorted(set(labels))},
        "majority_class": round(majority, 4),
        "run_note": a.run_note,
        "requests": [{"request": req, "gold": gold} for req, gold in REQUESTS],
        "note": ("Documents reuse the 20 requests, filler, question and delimiter of "
                 "bench_position_sensitivity.py; the request sits at `needle_position` of the "
                 "filler, so 1.0 is the cell that script measures. Lengths are filler tokens "
                 "under the classic tokenizer; the Vega reader's own input_tokens and truncation "
                 "flag are reported per case. n=20 per cell: intervals are wide and a null is "
                 "underpowered, not equivalence."),
    }

    all_rows = []
    for arm in arms:
        agent = build_agent(a, arm)
        if arm == "vega":
            meta["checkpoint"]["vega"] = getattr(agent, "model_id", None)
            meta["vega_context_tokens"] = getattr(agent, "context_tokens", None)
        first_cell = True
        for pad in (classic_lengths if arm == "classic" else lengths):
            reps = round(pad / per_rep)
            for frac in positions:
                preds = []
                for i, (req, gold) in enumerate(requests):
                    state = place(reps, frac, req)
                    if first_cell and i == 0:
                        # Warm-up, as bench_long_context.py does; not counted.
                        if arm == "vega":
                            agent.predict_long(state, QUESTIONS)
                        else:
                            agent.predict(state, QUESTIONS, max_len=a.classic_limit)
                        first_cell = False
                    t0 = time.perf_counter()
                    if arm == "vega":
                        result = agent.predict_long(state, QUESTIONS)
                    else:
                        result = agent.predict(state, QUESTIONS, max_len=a.classic_limit)
                    preds.append(_prediction(arm, i, gold, state, result, time.perf_counter() - t0))
                row = summarise(arm, pad, frac, preds)
                all_rows.append(row)
                print("arm=%-7s pad=%-6d pos=%-5.2f acc=%.3f (%2d/%d)  trunc=%d  med_tok=%d  med=%.3fs"
                      % (arm, pad, frac, row["accuracy"], row["correct"], row["n"],
                         row["truncated_count"], row["median_input_tokens"], row["median_latency_s"]),
                      flush=True)
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = dict(meta)
    out["rows"] = all_rows
    out["comparisons"] = comparisons(all_rows)
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\nwrote %s" % dest)
    print_rows(all_rows)


if __name__ == "__main__":
    main()
