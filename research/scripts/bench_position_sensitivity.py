"""Where in the document the answer sits: accuracy vs evidence position at a FIXED length.

`bench_long_context.py` varies only `pad`, and always appends the request at the END of the
filler. That confounds two different things: how long the document is, and how far the
evidence is from the option markers (which `build_sequence` places at the *start* of the
sequence, before the state). A run that varies only `pad` cannot tell them apart.

This script holds the document length fixed and sweeps the request through it, so the two
axes are separable:

    python research/scripts/bench_position_sensitivity.py --device cuda \
        --out research/results/position_sensitivity_multilingual.json

It uses the same 20 requests, the same filler and the same question as `bench_long_context.py`,
so a cell at position 1.0 is the same case that script measures. Every prediction is written to
the JSON, so the table re-derives without re-running the model.

Reported beside every cell: the majority-class rate of the item set. The 20 requests are
`billing` 9 / `technical` 7 / `sales` 4 / `other` 0, so a model that answers `billing` for
everything scores 0.45 — above the 0.35 that `bench_long_context.py` reports for a document the
1,024-token limit truncated. Position and length effects are only meaningful against that floor.
"""
import argparse, json, platform, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import transformers
import laya

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=8192, help="max_len for every cell")
    ap.add_argument("--lengths", default="4000,7000", help="document lengths (filler tokens)")
    ap.add_argument("--positions", default="0.0,0.25,0.5,0.75,1.0", help="needle position fractions")
    ap.add_argument("--out", default="research/results/position_sensitivity_multilingual.json")
    ap.add_argument("--subfolder", default="multilingual")
    ap.add_argument("--model", default="convaiinnovations/laya",
                    help="hub repo id, or a local checkpoint directory")
    a = ap.parse_args()

    model = a.model + (("/" + a.subfolder) if a.subfolder else "")
    if Path(a.model).is_dir():            # a local directory already contains the checkpoint
        agent = laya.load(a.model, device=a.device)
    else:
        agent = laya.load(a.model, subfolder=a.subfolder or None, device=a.device)
    tok = agent.tok

    labels = [g for _, g in REQUESTS]
    counts = {lab: labels.count(lab) for lab in sorted(set(labels))}
    majority = max(counts.values()) / len(labels)

    lengths = [int(x) for x in a.lengths.split(",")]
    positions = [float(x) for x in a.positions.split(",")]

    per_rep = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    rows = []
    for pad in lengths:
        reps = round(pad / per_rep)
        for frac in positions:
            correct, lat, preds = 0, [], []
            for i, (req, gold) in enumerate(REQUESTS):
                state = place(reps, frac, req)
                if i == 0 and frac == positions[0]:          # warm-up, as bench_long_context.py does
                    agent.predict(state, QUESTIONS, max_len=a.limit)
                t0 = time.perf_counter()
                r = agent.predict(state, QUESTIONS, max_len=a.limit)
                lat.append(time.perf_counter() - t0)
                ans = r["answers"]["department"]
                ok = ans["choice"] == gold
                correct += int(ok)
                preds.append({"request": req, "gold": gold, "prediction": ans["choice"],
                              "p_gold": round(float(ans["probabilities"].get(gold, 0.0)), 5),
                              "correct": ok, "input_tokens": r["usage"]["input_tokens"]})
            acc = correct / len(REQUESTS)
            row = {"pad_tokens": pad, "reps": reps, "needle_position": frac, "limit": a.limit,
                   "n": len(REQUESTS), "correct": correct, "accuracy": round(acc, 4),
                   "majority_class": round(majority, 4),
                   "modal_prediction_share": round(
                       max(sum(1 for p in preds if p["prediction"] == lab) for lab in set(p["prediction"] for p in preds))
                       / len(preds), 4),
                   "median_latency_s": round(statistics.median(lat), 4),
                   "median_input_tokens": int(statistics.median([p["input_tokens"] for p in preds])),
                   "predictions": preds}
            rows.append(row)
            print("pad=%-5d pos=%-5.2f acc=%.3f (%2d/%d)  majority=%.3f  modal=%.2f  median=%.3fs"
                  % (pad, frac, acc, correct, len(REQUESTS), majority,
                     row["modal_prediction_share"], row["median_latency_s"]), flush=True)

    out = {"checkpoint": model, "device": a.device, "platform": platform.platform(),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "laya": laya.__version__, "limit": a.limit, "filler_tokens_per_rep": per_rep,
           "questions": QUESTIONS, "filler": FILLER,
           "label_counts": counts, "majority_class": round(majority, 4),
           "note": ("Position is the fraction of the filler the request is inserted at, so 1.0 is the "
                    "case bench_long_context.py measures and 0.0 places the request immediately after "
                    "the option markers. Length is held fixed within a `pad_tokens` group, so the "
                    "position effect is separable from the length effect."),
           "rows": rows}
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % dest)

    # Length held fixed: the position effect alone, which a pad-only sweep cannot show.
    for pad in lengths:
        group = [r for r in rows if r["pad_tokens"] == pad]
        if len(group) < 2:
            continue
        best = max(group, key=lambda r: r["accuracy"])
        worst = min(group, key=lambda r: r["accuracy"])
        print("pad=%-5d  position spread: %.3f (best pos=%.2f) to %.3f (worst pos=%.2f)  -> %.3f"
              % (pad, best["accuracy"], best["needle_position"],
                 worst["accuracy"], worst["needle_position"], best["accuracy"] - worst["accuracy"]))


if __name__ == "__main__":
    main()
