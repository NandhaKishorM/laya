"""Laya typed-decisions fine-tuning on Apple Silicon.

This is a standalone replacement for the Kaggle 2xT4 notebook. It performs:
1. dependency/model/data preparation
2. local preprocessing and caching
3. single-process MPS/CPU fine-tuning
4. temperature calibration
5. checkpoint/model export

Useful examples:
    python notebooks/laya_finetune_typed_decisions_mps.py --epochs 2 --micro-batch 1 --grad-accum 32
    python notebooks/laya_finetune_typed_decisions_mps.py --model-dir ./laya_base --items ./train_items.pt
    python notebooks/laya_finetune_typed_decisions_mps.py --device cpu
"""

import argparse
import gc
import json
import math
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options

MODEL_ID = "convaiinnovations/laya"
DATASET_ID = "LocalLLaMA/typed-decisions"
DEFAULT_MODEL_DIR = "./laya_base"
DEFAULT_ITEMS = "./train_items.pt"
DEFAULT_OUTPUT_DIR = "./laya_finetuned_typed_decisions"


def choose_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS is unavailable; using CPU instead.")
        return torch.device("cpu")
    return torch.device(requested)


def prepare_model(model_dir):
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").exists():
        print(f"Downloading {MODEL_ID} to {model_dir} ...")
        snapshot_download(MODEL_ID, local_dir=str(model_dir))
    _fix_tokenizer_config(str(model_dir))
    return str(model_dir)


def build_training_item(tokenizer, cfg, state, question, gold_question):
    qtype = question["type"]
    criteria = question.get("criteria", {})

    if qtype == "choice":
        keys = list(criteria.keys())
        target = [gold_question["probabilities"].get(k, 0.0) for k in keys]
    elif qtype == "noul":
        target = [
            gold_question["probabilities"].get("false", 0.5),
            gold_question["probabilities"].get("true", 0.5),
        ]
    elif qtype == "score":
        n_levels = len(criteria) if isinstance(criteria, list) else 4
        target = [gold_question["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    else:
        return None

    total = sum(target)
    if total > 0:
        target = [float(x) / total for x in target]
    else:
        target = [1.0 / len(target)] * len(target)

    label = target.index(max(target))
    n_options = len(render_options({"t": qtype, "crit": criteria}))
    sequence, markers = build_sequence(
        tokenizer,
        state,
        {"t": qtype, "ins": question["instructions"], "crit": criteria},
        cfg["max_len"],
        cfg["head_max_len"],
    )
    if len(markers) != n_options:
        return None

    return {
        "ids": sequence,
        "markers": markers,
        "qtype": QTYPES[qtype],
        "target": target,
        "label": label,
    }


def prepare_items(model_dir, items_path, force=False):
    items_path = Path(items_path)
    model_path = Path(model_dir).resolve()
    with open(model_path / "rl_agent_config.json") as f:
        cfg = json.load(f)
    max_len = cfg.get("max_len", 1024)
    head_max_len = cfg.get("head_max_len", 256)
    cache_meta_path = items_path.with_name(items_path.name + ".meta.json")
    cache_key = {
        "model_dir": str(model_path),
        "max_len": max_len,
        "head_max_len": head_max_len,
        "dataset": DATASET_ID,
        "split": "train",
    }
    if items_path.exists() and not force:
        if not cache_meta_path.exists():
            # Legacy item files predate the sidecar metadata. They remain
            # usable offline; newly written caches always receive a key.
            print(f"Using legacy cached training items without metadata: {items_path}")
            return
        try:
            with open(cache_meta_path) as f:
                cached_key = json.load(f)
        except (OSError, json.JSONDecodeError):
            cached_key = None
        if cached_key == cache_key:
            print(f"Using cached training items: {items_path}")
            return
        print("Training-item cache key changed; rebuilding the cache.")

    # Keep datasets optional when a compatible local cache is already available.
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
    print(f"Downloading dataset {DATASET_ID} ...")
    dataset = load_dataset(DATASET_ID, "all", split="train")

    items = []
    skipped = 0
    for row in dataset:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, question in questions.items():
            if qid not in gold:
                continue
            item = build_training_item(
                tokenizer,
                {**cfg, "max_len": max_len, "head_max_len": head_max_len},
                state,
                question,
                gold[qid],
            )
            if item is None:
                skipped += 1
            else:
                items.append(item)

    items_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, items_path)
    with open(cache_meta_path, "w") as f:
        json.dump(cache_key, f, indent=2)
    print(f"Saved {len(items)} training items to {items_path}; skipped={skipped}")


def collate(items, pad_id):
    batch_size = len(items)
    seq_len = max(len(item["ids"]) for item in items)
    kmax = max(len(item["markers"]) for item in items)
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    attention = torch.zeros((batch_size, seq_len), dtype=torch.long)
    marker_pos = torch.zeros((batch_size, kmax), dtype=torch.long)
    marker_mask = torch.zeros((batch_size, kmax), dtype=torch.bool)
    target = torch.zeros((batch_size, kmax), dtype=torch.float32)

    for i, item in enumerate(items):
        length = len(item["ids"])
        input_ids[i, :length] = torch.tensor(item["ids"], dtype=torch.long)
        attention[i, :length] = 1
        k = len(item["markers"])
        marker_pos[i, :k] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[i, :k] = True
        target[i, :len(item["target"])] = torch.tensor(item["target"], dtype=torch.float32)

    return (
        input_ids,
        attention,
        marker_pos,
        marker_mask,
        target,
        torch.tensor([item["qtype"] for item in items], dtype=torch.long),
    )


def fit_temperature(samples):
    if len(samples) < 10:
        return 1.0
    kmax = max(len(logits) for logits, _ in samples)
    logits = torch.full((len(samples), kmax), -1e4)
    targets = torch.zeros((len(samples), kmax))
    for i, (values, target) in enumerate(samples):
        logits[i, :len(values)] = torch.as_tensor(values)
        targets[i, :len(target)] = torch.as_tensor(target, dtype=torch.float32)

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_temperature.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temperature.exp(), 0.1, 10.0).item())


def save_checkpoint(model, tokenizer, cfg, output_dir, epoch, final=False):
    path = Path(output_dir) if final else Path(output_dir) / "checkpoint_latest"
    path.mkdir(parents=True, exist_ok=True)
    weights = {
        name: value.detach().half().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(weights, str(path / "model.safetensors"))
    model.encoder.config.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "tokenizer")
    with open(path / "checkpoint_meta.json", "w") as f:
        json.dump({"epoch": epoch, "final": final}, f, indent=2)
    with open(path / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def train(args, model_dir, items_path, device):
    with open(Path(model_dir) / "rl_agent_config.json") as f:
        cfg = json.load(f)
    cfg.update({"max_tokens_per_batch": 2048, "max_len": 1024, "head_max_len": 256})
    if not args.no_checkpointing:
        cfg["gradient_checkpointing"] = True

    tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
    model = build_model(cfg, encoder_dir=Path(model_dir) / "encoder")
    model.load_state_dict(load_file(str(Path(model_dir) / "model.safetensors")), strict=True)
    model.float()

    if not args.no_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True
    model.to(device).train()

    all_items = torch.load(items_path, map_location="cpu", weights_only=False)
    order = list(range(len(all_items)))
    random.Random(20260922).shuffle(order)
    n_calib = min(args.calib_max, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]

    encoder_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": 2.5e-5}, {"params": head_params, "lr": 1e-4}],
        weight_decay=0.01,
    )
    updates = max(1, math.ceil(len(train_items) / args.micro_batch / args.grad_accum) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=1e-6
    )

    print(f"Device: {device}")
    print(f"Training items: {len(train_items)}; calibration items: {len(calib_items)}")
    print(f"micro_batch={args.micro_batch}; grad_accum={args.grad_accum}; epochs={args.epochs}")

    for epoch in range(args.epochs):
        random.Random(42 + epoch).shuffle(train_items)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_batches = 0
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, args.epochs - 1)

        for start in range(0, len(train_items), args.micro_batch):
            chunk = train_items[start:start + args.micro_batch]
            ids, attention, positions, mask, target, qtype = collate(chunk, tokenizer.pad_token_id)
            ids, attention = ids.to(device), attention.to(device)
            positions = positions.to(device)
            mask = mask.to(device)
            target = target.to(device)
            qtype = qtype.to(device)

            logits, activation = model(ids, attention, positions, mask, qtype)
            logits = logits.float()
            k = mask.sum(-1, keepdim=True).float()

            eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            noisy_logits = logits.detach().unsqueeze(0) + eps
            probabilities = torch.softmax(noisy_logits.masked_fill(~mask, -1e4), -1)

            with torch.no_grad():
                reward = proper_reward(
                    probabilities,
                    target.unsqueeze(0),
                    qtype,
                    mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)

            logp = -(((noisy_logits - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(advantage * logp).mean()
            loss_ce = -(
                target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
            ).sum(-1).mean()
            loss = (loss_rl + loss_ce + 0.0 * activation.sum()) / args.grad_accum
            loss.backward()

            n_batches += 1
            if n_batches % args.grad_accum == 0 or start + args.micro_batch >= len(train_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.grad_accum
            if n_batches % 100 == 0:
                print(f"epoch {epoch + 1}/{args.epochs}, step {n_batches}, loss={loss.item() * args.grad_accum:.4f}")

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch + 1}/{args.epochs} complete; avg_loss={avg_loss:.4f}")
        save_checkpoint(model, tokenizer, cfg, args.output_dir, epoch + 1)

    print("Running temperature calibration ...")
    model.eval()
    samples = [[] for _ in range(3)]
    with torch.no_grad():
        for start in range(0, len(calib_items), args.micro_batch):
            chunk = calib_items[start:start + args.micro_batch]
            ids, attention, positions, mask, target, qtype = collate(chunk, tokenizer.pad_token_id)
            logits, _ = model(
                ids.to(device), attention.to(device), positions.to(device), mask.to(device), qtype.to(device)
            )
            for i, item in enumerate(chunk):
                qtype_id = item["qtype"]
                samples[qtype_id].append(
                    (logits[i, :len(item["markers"])].cpu(), item["target"])
                )

    temperatures = [fit_temperature(group) if group else 1.2 for group in samples]
    save_checkpoint(model, tokenizer, cfg, args.output_dir, args.epochs, final=True)
    cfg.update({
        "fine_tuned": True,
        "model_name": "laya-typed-decisions",
        "temperature": temperatures,
    })
    cfg.pop("temperature_by_options", None)
    with open(Path(args.output_dir) / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(Path(args.output_dir) / "checkpoint_latest" / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Model saved to {args.output_dir}")
    print(f"Temperatures: {temperatures}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya on Apple Silicon MPS")
    parser.add_argument("model_dir", nargs="?", default=None)
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--model-dir", dest="model_dir_option")
    parser.add_argument("--output-dir", dest="output_dir_option")
    parser.add_argument("--items", default=DEFAULT_ITEMS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--calib-max", type=int, default=400)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.model_id != MODEL_ID:
        # Keep the requested model ID local to preparation by updating the
        # module constant before prepare_model() is called.
        globals()["MODEL_ID"] = args.model_id

    args.model_dir = args.model_dir_option or args.model_dir or DEFAULT_MODEL_DIR
    args.output_dir = args.output_dir_option or args.output_dir or DEFAULT_OUTPUT_DIR

    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    model_dir = prepare_model(args.model_dir)
    prepare_items(model_dir, args.items, force=args.force_preprocess)
    train(args, model_dir, args.items, device)
    gc.collect()


if __name__ == "__main__":
    main()
