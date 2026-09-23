"""Stock forward vs TileLang fast path: numerics, latency, and accuracy on real datasets.

    python benchmarks/bench_fast.py                       # english checkpoint
    python benchmarks/bench_fast.py --subfolder multilingual
    python benchmarks/bench_fast.py --eval 1000           # + AG News / dair-ai emotion accuracy & ECE

Set HF_ENDPOINT to a mirror if huggingface.co is slow for you.
"""
import argparse, json, os, sys, time
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import laya
from laya.common import QTYPES, build_sequence, collate_items, ece_score

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="convaiinnovations/laya"); ap.add_argument("--subfolder", default=None)
ap.add_argument("--eval", type=int, default=0, help="samples per dataset for the accuracy comparison (0 = skip)")
ap.add_argument("--iters", type=int, default=30); ap.add_argument("--json", default=None)
args = ap.parse_args()

agent = laya.load(args.model, subfolder=args.subfolder)
if agent.device.type != "cuda":
    sys.exit("needs a CUDA device")
report = {"model": args.model, "subfolder": args.subfolder, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

Q = {"department": {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages", "sales": "pricing", "shipping": "delivery"}},
     "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]},
     "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"}}
def qs(n): return {f"{k}{i}": v for i in range(n) for k, v in Q.items()}
short = {"subject": "Duplicate charge on invoice 4411", "body": "We were billed twice for March. Please refund the duplicate or we're moving to a competitor."}
long_ = {"subject": "Outage report", "body": "Since yesterday our whole team cannot log in, the dashboard returns 502 errors and our release is blocked. " * 40}

def batch(state, q):
    items = []
    for qid in q:
        qq = agent._to_internal(q[qid]); seq, m = build_sequence(agent.tok, state, qq, agent.cfg["max_len"], agent.cfg["head_max_len"])
        items.append({"ids": seq, "markers": m, "qtype": QTYPES[qq["t"]]})
    return {k: v.cuda() for k, v in collate_items([items], agent.tok.pad_token_id).items() if torch.is_tensor(v)}

def fwd(b, amp=True):
    with torch.no_grad(), torch.autocast("cuda", dtype=agent.dtype, enabled=amp):
        return agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])

def timeit(fn, iters=args.iters):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1000

cases = [("short x1", short, {"department": Q["department"]}), ("short x3", short, Q), ("short x30", short, qs(10)),
         ("long x3", long_, Q), ("long x30", long_, qs(10))]
P = lambda l: torch.softmax(l.float(), -1)
print(f"\n== {report['gpu']}  {args.model}/{args.subfolder or ''}  dtype={agent.dtype}")
print("== numerics: max |p - p_fp32| over all options")
rows = []
for name, st, q in cases:
    b = batch(st, q)
    agent.deaccelerate()
    l32, _ = fwd(b, amp=False); lo, _ = fwd(b); t_stock = timeit(lambda: fwd(b))
    assert agent.accelerate(strict=True)
    lf, _ = fwd(b); t_fast = timeit(lambda: fwd(b))
    d_o, d_f, d_of = [(P(a) - P(c)).abs().max().item() for a, c in ((lo, l32), (lf, l32), (lf, lo))]
    agree = (lf.argmax(-1) == l32.argmax(-1)).float().mean().item()
    L = b["input_ids"].shape[1]
    print(f"{name:10s} L={L:4d}  stock-bf16={d_o:.4f}  fast={d_f:.4f}  fast-vs-stock={d_of:.4f}  argmax agree={agree:.2f}")
    rows.append(dict(case=name, L=L, stock_ms=t_stock, fast_ms=t_fast, dp_stock=d_o, dp_fast=d_f, agree=agree))
print("\n== model forward latency (ms)")
print(f"{'case':10s} {'L':>5s} {'stock':>9s} {'fast':>9s} {'speedup':>8s}")
for r in rows:
    print(f"{r['case']:10s} {r['L']:5d} {r['stock_ms']:9.2f} {r['fast_ms']:9.2f} {r['stock_ms']/r['fast_ms']:7.1f}x")
print("\n== end-to-end agent.predict() incl. tokenization (ms)")
for name, st, q in cases:
    agent.deaccelerate(); ts = timeit(lambda: agent.predict(st, q)); agent.accelerate(strict=True); tf = timeit(lambda: agent.predict(st, q))
    print(f"{name:10s} stock={ts:8.2f}  fast={tf:8.2f}  {ts/tf:5.1f}x")
    [r for r in rows if r["case"] == name][0].update(e2e_stock_ms=ts, e2e_fast_ms=tf)
report["latency"] = rows

if args.eval:
    from datasets import load_dataset
    evals = {
        "ag_news": ("fancyzhx/ag_news", "test", "text", "label",
                    {"world": "international news, politics, conflicts", "sports": "sports, games, athletes",
                     "business": "companies, markets, economy", "sci/tech": "science, technology, software, space"}),
        "emotion": ("dair-ai/emotion", "test", "text", "label",
                    {"sadness": None, "joy": None, "love": None, "anger": None, "fear": None, "surprise": None}),
    }
    report["eval"] = {}
    print(f"\n== accuracy on real datasets ({args.eval} samples each), stock vs fast")
    print(f"{'dataset':9s} {'acc stock':>10s} {'acc fast':>9s} {'ECE stock':>10s} {'ECE fast':>9s} {'agree':>6s} {'stock ms/it':>12s} {'fast ms/it':>11s}")
    for name, (repo, split, tcol, lcol, crit) in evals.items():
        ds = load_dataset(repo, split=split).shuffle(seed=0).select(range(args.eval))
        labels = list(crit.keys())
        q = {"label": {"type": "choice", "instructions": f"Which category does this {name.replace('_', ' ')} text belong to?", "criteria": crit}}
        out = {}
        for mode in ("stock", "fast"):
            agent.deaccelerate() if mode == "stock" else agent.accelerate(strict=True)
            preds, confs, correct = [], [], []
            torch.cuda.synchronize(); t = time.perf_counter()
            for ex in ds:
                a = agent.predict({"text": ex[tcol]}, q)["answers"]["label"]
                preds.append(a["choice"]); confs.append(max(a["probabilities"].values())); correct.append(labels.index(a["choice"]) == ex[lcol])
            torch.cuda.synchronize(); dt = (time.perf_counter() - t) / len(ds) * 1000
            out[mode] = dict(acc=float(np.mean(correct)), ece=ece_score(np.array(confs), np.array(correct, dtype=float)), ms=dt, preds=preds)
        agree = float(np.mean([a == b for a, b in zip(out["stock"]["preds"], out["fast"]["preds"])]))
        print(f"{name:9s} {out['stock']['acc']:10.3f} {out['fast']['acc']:9.3f} {out['stock']['ece']:10.3f} {out['fast']['ece']:9.3f} {agree:6.3f} {out['stock']['ms']:12.1f} {out['fast']['ms']:11.1f}")
        report["eval"][name] = dict(n=args.eval, stock={k: v for k, v in out["stock"].items() if k != "preds"}, fast={k: v for k, v in out["fast"].items() if k != "preds"}, agreement=agree)
if args.json:
    json.dump(report, open(args.json, "w"), indent=1); print("saved", args.json)
