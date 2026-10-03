"""Fine-tune a Laya checkpoint on a single device (CPU or GPU).

Reproduces the 2xT4 notebook's training loop without DDP, so any checkpoint
(including the multilingual one) can be fine-tuned on one GPU or on a CPU.
Preprocesses a JSONL dataset into tokenized items, trains with RLCD, fits one
calibration temperature per question type, and saves a loadable checkpoint.

Each JSONL line is one case: {"state": "...", "questions": {...}, "gold": {...}}
where questions use the choice / score / noul primitives and gold carries the
teacher probabilities plus a label, matching the schema in docs/finetune.md.

Usage:
  python research/scripts/finetune_single_device.py \
      --data dataset.jsonl --model-dir /path/to/multilingual --output-dir out/
"""

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_model,
    build_sequence,
    proper_reward,
    render_options,
)


def build_training_item(tok, cfg, state, q, gold_q):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [
            gold_q["probabilities"].get("false", 0.5),
            gold_q["probabilities"].get("true", 0.5),
        ]
    else:
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    total = sum(target)
    target = (
        [v / total for v in target] if total > 0 else [1.0 / len(target)] * len(target)
    )
    seq, markers = build_sequence(
        tok,
        state,
        {"t": t, "ins": q["instructions"], "crit": crit},
        cfg["max_len"],
        cfg["head_max_len"],
    )
    if len(markers) != len(render_options({"t": t, "crit": crit})):
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[t],
        "target": target,
        "label": target.index(max(target)),
    }


def preprocess(tok, cfg, data_path):
    items = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for qid, q in row["questions"].items():
                if qid not in row["gold"]:
                    continue
                item = build_training_item(tok, cfg, row["state"], q, row["gold"][qid])
                if item is not None:
                    items.append(item)
    return items


def collate(items, pad_id):
    n, length = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def forward(model, batch, device, use_amp):
    if use_amp:
        with torch.autocast("cuda", dtype=torch.float16):
            return model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
    return model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["marker_pos"].to(device),
        batch["marker_mask"].to(device),
        batch["qtype"].to(device),
    )


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), TEMP_MIN, TEMP_MAX).item())


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Laya checkpoint on one device."
    )
    parser.add_argument(
        "--data", required=True, help="JSONL dataset of {state, questions, gold} cases."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Checkpoint directory; defaults to the multilingual subfolder.",
    )
    parser.add_argument("--output-dir", default="laya_finetuned")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "auto":
        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    print("Device:", device)

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join(
            snapshot_download(
                "convaiinnovations/laya",
                allow_patterns=["multilingual/*", "multilingual/*/*"],
            ),
            "multilingual",
        )
    _fix_tokenizer_config(model_dir)

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = use_amp
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(
        load_file(os.path.join(model_dir, "model.safetensors")), strict=True
    )
    if use_amp:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True
    model.to(device)
    model.train()

    all_items = preprocess(tok, cfg, args.data)
    if not all_items:
        raise SystemExit("No training items produced from --data.")

    order = list(range(len(all_items)))
    random.shuffle(order)
    n_calib = min(400, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]

    micro_batch = 8
    group_size = 4
    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": 2.5e-5}, {"params": head_params, "lr": 1.0e-4}],
        weight_decay=0.01,
    )
    steps_per_epoch = max(1, (len(train_items) + micro_batch - 1) // micro_batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    print(
        "Train items: {} ({} held out) | {} epochs | {} steps/epoch".format(
            len(train_items), len(calib_items), args.epochs, steps_per_epoch
        )
    )

    for epoch in range(args.epochs):
        random.seed(args.seed + epoch)
        random.shuffle(train_items)
        sigma = 0.4 + (0.1 - 0.4) * (epoch / max(1, args.epochs - 1))
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for b_idx in range(0, len(train_items), micro_batch):
            chunk = train_items[b_idx : b_idx + micro_batch]
            if not chunk:
                continue
            batch = collate(chunk, tok.pad_token_id)
            logits, _act = forward(model, batch, device, use_amp)
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            eps = (
                torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
            )
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            with torch.no_grad():
                r = proper_reward(
                    q,
                    target.unsqueeze(0),
                    batch["qtype"].to(device),
                    mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss = (
                -(adv * logp).mean()
                - (target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1))
                .sum(-1)
                .mean()
            )

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            n_batches += 1
        print(
            "Epoch {}/{} | avg loss {:.4f}".format(
                epoch + 1, args.epochs, epoch_loss / max(1, n_batches)
            ),
            flush=True,
        )

    model.eval()
    calib_preds = []
    with torch.no_grad():
        for c_idx in range(0, len(calib_items), 16):
            c_chunk = calib_items[c_idx : c_idx + 16]
            cb = collate(c_chunk, tok.pad_token_id)
            l_sub, _ = forward(model, cb, device, use_amp)
            l_np = l_sub.float().cpu().numpy()
            for r, it in enumerate(c_chunk):
                kk = len(it["markers"])
                calib_preds.append((it["qtype"], l_np[r, :kk], it["target"]))

    fitted = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
        if sel:
            fitted[qt] = fit_one_temp(sel)
    print("Fitted temperatures (choice, score, noul):", [round(t, 3) for t in fitted])

    os.makedirs(args.output_dir, exist_ok=True)
    save_file(
        {key: val.half().contiguous().cpu() for key, val in model.state_dict().items()},
        os.path.join(args.output_dir, "model.safetensors"),
    )
    model.encoder.config.save_pretrained(os.path.join(args.output_dir, "encoder"))
    tok.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["temperature"] = fitted
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved checkpoint to", args.output_dir)


if __name__ == "__main__":
    main()
