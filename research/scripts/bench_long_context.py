"""Long-document accuracy and latency on laya-multilingual at the default limit and at 8,192 tokens.

Each case is a support request placed at the END of a document padded with unrelated meeting
notes, so the request sits after `pad` tokens of other text. That is the case a 1,024-token limit
gets wrong: `build_sequence` keeps the start of the state by default and cuts the rest.

    python research/scripts/bench_long_context.py --device mps --out research/results/long_context_multilingual.json

The JSON keeps every prediction, so the table can be re-derived without re-running the model.
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

# 20 requests: English, Spanish, Portuguese, French, German, Hindi, Japanese, Arabic.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pads", default="0,1000,2000,4000,7000", help="approximate filler tokens before the request")
    ap.add_argument("--out", default="research/results/long_context_multilingual.json")
    a = ap.parse_args()

    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device=a.device)
    per_rep = len(agent.tok(FILLER, add_special_tokens=False)["input_ids"])
    default_limit = int(agent.cfg.get("max_len", 1024))
    rows, cases = [], []
    for pad in [int(p) for p in a.pads.split(",")]:
        reps = round(pad / per_rep)
        for limit in (default_limit, 8192):
            correct, lats, toks = 0, [], []
            for i, (text, gold) in enumerate(REQUESTS):
                state = FILLER * reps + ("\n\nActual request: " if reps else "") + text
                if i == 0:
                    agent.predict(state, QUESTIONS, max_len=limit)          # warm-up
                t0 = time.perf_counter()
                r = agent.predict(state, QUESTIONS, max_len=limit)
                lat = time.perf_counter() - t0
                ans = r["answers"]["department"]
                ok = ans["choice"] == gold
                correct += ok; lats.append(lat); toks.append(r["usage"]["input_tokens"])
                cases.append({"pad_tokens": pad, "limit": limit, "request": text, "gold": gold,
                              "choice": ans["choice"], "probabilities": ans["probabilities"],
                              "correct": ok, "input_tokens": r["usage"]["input_tokens"], "latency_s": round(lat, 4)})
            row = {"pad_tokens": pad, "limit": limit, "n": len(REQUESTS), "correct": correct,
                   "accuracy": round(correct / len(REQUESTS), 3),
                   "median_latency_s": round(statistics.median(lats), 3),
                   "median_input_tokens": int(statistics.median(toks))}
            rows.append(row)
            print("pad %5d  limit %5d  accuracy %2d/%d  median %.3fs  tokens read %d"
                  % (pad, limit, correct, len(REQUESTS), row["median_latency_s"], row["median_input_tokens"]), flush=True)
    out = {"checkpoint": "convaiinnovations/laya (multilingual/)", "default_limit": default_limit,
           "device": a.device, "platform": platform.platform(), "torch": torch.__version__,
           "transformers": transformers.__version__, "laya": laya.__version__,
           "questions": QUESTIONS, "rows": rows, "cases": cases}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
