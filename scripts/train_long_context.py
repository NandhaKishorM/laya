"""Extend the laya checkpoint to a longer context window.

Fine-tunes the shipped checkpoint (trained at max_len=512) at a longer max_len so that
accuracy and calibration hold past the trained length. Synthetic long-document training
data places the evidence at varying depths, so the model must actually attend over the
whole window rather than the first 512 tokens.

Usage:
    python train_long_context.py --max-len 8192 --steps 400
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from laya.common import QTYPES, build_model, build_sequence, collate_items

FILLER = [
    "The quarterly report notes steady performance across all regions.",
    "Meeting minutes were circulated to the distribution list on Friday.",
    "The maintenance window is scheduled outside of business hours.",
    "Procurement confirmed the vendor contract renews automatically.",
    "Headcount remained flat compared with the previous reporting period.",
    "The office will observe the standard public holiday calendar.",
    "Travel bookings must be submitted through the internal portal.",
    "Documentation for the legacy system is archived in the wiki.",
]

# (label, evidence sentence) -- the signal the model must find in a sea of filler.
SIGNALS = [
    # Deliberately subtle: no keyword shortcut like "URGENT" separates the classes, so the
    # model must read the evidence rather than pattern-match a marker token.
    ("urgent", "The checkout flow has been failing for all customers since this morning."),
    ("urgent", "Payments are being declined and revenue has stopped as of an hour ago."),
    ("urgent", "Every API request is timing out and the on-call engineer is unreachable."),
    ("urgent", "Customer data appears to be exposed publicly and must be locked down."),
    ("normal", "The onboarding guide could use a refresh when someone has a spare moment."),
    ("normal", "A minor typo appears on the pricing page footer; no hurry on the fix."),
    ("normal", "Consider adding a dark mode option to the settings screen eventually."),
    ("normal", "The quarterly archive job finished successfully with nothing to action."),
]


def make_example(rng, target_chars, question_type="noul"):
    """A long document with one decisive sentence buried at a random depth."""
    label_name, evidence = rng.choice(SIGNALS)
    body = []
    while sum(len(s) for s in body) < target_chars:
        body.append(rng.choice(FILLER))
    pos = rng.randrange(len(body) + 1)
    body.insert(pos, evidence)
    state = " ".join(body)
    label = 1 if label_name == "urgent" else 0
    return state, label, pos / max(1, len(body))


def build_batch(tok, rng, batch_size, max_len, head_max_len, target_chars, device):
    items = []
    for _ in range(batch_size):
        state, label, _ = make_example(rng, target_chars)
        q = {
            "t": "noul",
            "ins": "Does this document describe an urgent, time-critical problem?",
            "crit": {"true": "urgent and time-critical", "false": "routine, can wait"},
        }
        ids, markers = build_sequence(tok, state, q, max_len, head_max_len)
        target = [0.0, 0.0]
        target[label] = 1.0
        items.append(
            {"ids": ids, "markers": markers, "qtype": QTYPES["noul"], "label": label, "target": target}
        )
    b = collate_items([items], tok.pad_token_id)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, tok, rng, max_len, head_max_len, target_chars, device, n=64, bs=8):
    model.eval()
    correct, confs = [], []
    for _ in range(max(1, n // bs)):
        b = build_batch(tok, rng, bs, max_len, head_max_len, target_chars, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                              b["marker_mask"], b["qtype"])
        p = torch.softmax(logits.float(), -1)
        pred = p.argmax(-1)
        correct.extend((pred == b["label"]).float().cpu().tolist())
        confs.extend(p.max(-1).values.cpu().tolist())
    model.train()
    acc = float(np.mean(correct))
    # ECE over 10 bins
    conf, corr = np.array(confs), np.array(correct)
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - corr[sel].mean())
    return acc, float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="local snapshot of the laya checkpoint")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--head-max-len", type=int, default=192)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", default="/root/laya_8k")
    ap.add_argument("--log", default="train_log.jsonl")
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    rng = random.Random(0)

    cfg = json.load(open(os.path.join(args.model_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))

    enc_dir = os.path.join(args.model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)
    model.load_state_dict(load_file(os.path.join(args.model_dir, "model.safetensors")), strict=True)

    # widen the encoder's position limit
    model.encoder.config.max_position_embeddings = max(
        args.max_len, model.encoder.config.max_position_embeddings
    )
    model.to(device).train()
    model.encoder.gradient_checkpointing_enable()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1
    )
    # ~4 chars/token, fill most of the window
    target_chars = int(args.max_len * 3.6)
    logf = open(args.log, "w")

    acc0, ece0 = evaluate(model, tok, random.Random(1), args.max_len, args.head_max_len,
                          target_chars, device)
    rec = {"step": 0, "eval_acc": round(acc0, 4), "eval_ece": round(ece0, 4), "phase": "before"}
    print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            b = build_batch(tok, rng, args.batch_size, args.max_len, args.head_max_len,
                            target_chars, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                  b["marker_mask"], b["qtype"])
            loss = F.cross_entropy(logits.float(), b["label"]) / args.accum
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 10 == 0 or step == 1:
            rec = {"step": step, "loss": round(total, 4),
                   "lr": round(sched.get_last_lr()[0], 8),
                   "elapsed_s": round(time.time() - t0, 1),
                   "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()

        if step % args.eval_every == 0 or step == args.steps:
            acc, ece = evaluate(model, tok, random.Random(1), args.max_len, args.head_max_len,
                                target_chars, device)
            rec = {"step": step, "eval_acc": round(acc, 4), "eval_ece": round(ece, 4)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()

    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(args.out, "model.safetensors"))
    cfg_out = dict(cfg)
    cfg_out["max_len"] = args.max_len
    cfg_out["head_max_len"] = args.head_max_len
    json.dump(cfg_out, open(os.path.join(args.out, "rl_agent_config.json"), "w"), indent=2)
    print(json.dumps({"saved": args.out, "total_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
