"""Device-agnostic RLCD fine-tuning for Laya decision models.

The Kaggle notebook (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`) is the reference
run: two T4s, DDP, fp16 autocast. This module is the same training loop with the device handling
factored out, so it also runs on

* an **AMD GPU through Metal (MPS)** on macOS -- `device="mps"`,
* a **ROCm build of PyTorch** on Linux -- ROCm presents itself as `cuda`, so `device="cuda"` and
  the fp16 autocast path are used unchanged,
* **CPU** -- `device="cpu"`, fp32, slower but exact.

Mixed precision is CUDA-only here. MPS and CPU run fp32 with no gradient scaler, which is slower
per step but numerically safe: `torch.autocast("mps", ...)` is not supported by torch 2.2 (the
version macOS Intel tops out at) and loss scaling has nothing to scale in fp32.

    python -m laya.finetune --model-dir models/laya --items train_items.pt \
        --output-dir finetuned --device mps --epochs 4

Multi-GPU is still DDP: launch with `torchrun --nproc_per_node=N` and the world size is detected.
Items are pre-tokenized by the caller (see the notebook), so this module needs no tokenizer to
train -- only to save one alongside the weights.
"""
import argparse
import contextlib
import json
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .common import build_model, proper_reward

__all__ = [
    "available_devices",
    "pick_device",
    "device_report",
    "collate_train_batch",
    "fit_temperature",
    "train_rlcd",
    "main",
]


# ------------------------------------------------------------------ device selection
def available_devices() -> Dict[str, object]:
    """What this machine can train on, in preference order."""
    info: Dict[str, object] = {"cuda": 0, "cuda_names": [], "mps": False, "cpu": True}
    if torch.cuda.is_available():
        info["cuda"] = torch.cuda.device_count()
        info["cuda_names"] = [torch.cuda.get_device_name(i) for i in range(info["cuda"])]
        # ROCm reports itself as cuda; say so rather than calling it NVIDIA.
        info["backend"] = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps"] = True
    return info


def pick_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve the training device: explicit request first, then CUDA/ROCm, MPS, CPU.

    An explicit request is honoured even if it looks unavailable, so the failure names the
    device the caller asked for rather than silently training somewhere else.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report() -> str:
    info = available_devices()
    lines = []
    if info["cuda"]:
        backend = str(info.get("backend", "cuda")).upper()
        lines.append("%s: %d device(s) -- %s" % (backend, info["cuda"], ", ".join(info["cuda_names"])))
    else:
        lines.append("CUDA/ROCm: not available")
    lines.append("MPS (Metal, AMD/Apple GPU): %s" % ("available" if info["mps"] else "not available"))
    lines.append("CPU: always available")
    return "\n".join("  " + line for line in lines)


# ------------------------------------------------------------------ mixed precision
@contextlib.contextmanager
def _amp(device: torch.device):
    """fp16 autocast on CUDA/ROCm; a no-op everywhere else.

    `torch.autocast` rejects a device type it has no backend for -- including 'mps' even with
    enabled=False -- so the context is only entered where it actually applies.
    """
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            yield
    else:
        yield


def _grad_scaler(device: torch.device) -> Optional["torch.amp.GradScaler"]:
    """A GradScaler for CUDA/ROCm, or None where fp32 needs no scaling."""
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (TypeError, RuntimeError):            # older torch: torch.cuda.amp.GradScaler
        return torch.cuda.amp.GradScaler(enabled=True)


# ------------------------------------------------------------------ batching and calibration
def collate_train_batch(items: Sequence[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a list of pre-tokenized items into one batch. CPU tensors; the caller moves them."""
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
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
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_temperature(sel: List) -> float:
    """Fit one temperature to (logits, target) pairs by minimising log loss. CPU, tiny.

    One parameter, so LBFGS converges in a handful of iterations and no device is involved.
    """
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
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


# ------------------------------------------------------------------ the loop
def _distributed():
    """(rank, world_size) when launched under torchrun with more than one process."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1
    world_size = int(os.environ["WORLD_SIZE"])
    return int(os.environ["RANK"]), world_size


def _optimizer_step(loss, model, optimizer, scaler, max_norm: float = 1.0) -> None:
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()


def train_rlcd(
    items: List[Dict],
    model_dir: str,
    output_dir: str,
    *,
    device: Optional[object] = None,
    epochs: int = 4,
    micro_batch: int = 8,
    grad_accum: int = 4,
    group_size: int = 4,
    lr_encoder: float = 2.5e-5,
    lr_head: float = 1e-4,
    sigma_start: float = 0.4,
    sigma_end: float = 0.1,
    grad_checkpointing: bool = True,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    pad_id: Optional[int] = None,
    tokenizer_dir: Optional[str] = None,
    calib_every: int = 15,
    calib_max: int = 400,
    limit: Optional[int] = None,
    init_method: Optional[str] = None,
    seed: int = 42,
    log_every: int = 50,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Train one checkpoint with RLCD. Returns a small metrics dict.

    The reward is `proper_reward` with a spherical term (strictly proper), the policy gradient is
    taken over `group_size` noisy samples of the logits, and a soft cross-entropy term pulls the
    logits towards the teacher distribution. That math is identical to the notebook's; what
    differs here is only where the tensors live and whether fp16 is used.

    Under torchrun only rank 0 calibrates and writes the output, so the metrics returned by the
    rank-0 process are the ones that describe the saved checkpoint; the other ranks return their
    own training numbers with the unfitted temperatures.
    """
    rank, world_size = _distributed()
    target_device = device if isinstance(device, torch.device) else pick_device(device)

    if target_device.type == "cuda":
        # one process per GPU under torchrun; single process otherwise
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        target_device = torch.device("cuda", local_rank)
    if world_size > 1:
        if target_device.type == "mps":
            raise RuntimeError(
                "MPS has no distributed backend, so torchrun is not supported on it; "
                "run single-process (omit torchrun) or use LAYA_DEVICE=cpu for a gloo run")
        import torch.distributed as dist
        if not dist.is_initialized():
            # NCCL for CUDA/ROCm (ROCm ships RCCL under the same name); gloo for CPU.
            # init_method defaults to env:// (what torchrun sets up); a file:// store is a
            # useful fallback where the TCP store cannot resolve the host, e.g. gloo on macOS.
            kwargs = {"rank": rank, "world_size": world_size}
            if init_method:
                kwargs["init_method"] = init_method
            dist.init_process_group("nccl" if target_device.type == "cuda" else "gloo", **kwargs)

    if limit is not None:
        items = items[:limit]

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = bool(grad_checkpointing)
    cfg["max_tokens_per_batch"] = 4096
    if max_len is not None:
        cfg["max_len"] = max_len
    if head_max_len is not None:
        cfg["head_max_len"] = head_max_len

    tok_dir = tokenizer_dir or os.path.join(model_dir, "tokenizer")
    tokenizer = None
    if pad_id is None and os.path.isdir(tok_dir):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    from safetensors.torch import load_file

    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)

    if grad_checkpointing:
        try:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except (AttributeError, TypeError) as e:      # older transformers signature
            try:
                model.encoder.gradient_checkpointing_enable()
            except Exception:
                log("  gradient checkpointing unavailable (%s); continuing without it" % e)
                cfg["gradient_checkpointing"] = False
    model.head_checkpointing = bool(cfg["gradient_checkpointing"])
    model.to(target_device)
    model.train()

    train_model = model
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # device_ids is a CUDA concept; gloo/CPU passes None
        train_model = DDP(model, device_ids=[target_device.index] if target_device.type == "cuda" else None,
                          find_unused_parameters=True)

    my_items = items[rank::world_size] if world_size > 1 else list(items)
    micro_batch = max(1, min(micro_batch, len(my_items)))

    enc_params = [p for n, p in train_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in train_model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": lr_encoder},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=0.01)

    updates_per_epoch = max(1, len(my_items) // (micro_batch * grad_accum))
    total_updates = updates_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = _grad_scaler(target_device)

    if rank == 0:
        log("Training on %s | %d items (%d this rank) | %d epochs | micro_batch=%d x accum=%d"
            % (target_device, len(items), len(my_items), epochs, micro_batch, grad_accum))
        if target_device.type != "cuda":
            log("  fp32, no autocast and no loss scaler on %s" % target_device.type)

    t0 = time.time()
    losses, rewards, updates = [], [], 0
    for epoch in range(epochs):
        random.seed(seed + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        progress = epoch / max(1, epochs - 1)
        sigma = sigma_start + (sigma_end - sigma_start) * progress

        for b_idx in range(0, len(my_items), micro_batch):
            chunk = my_items[b_idx:b_idx + micro_batch]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, pad_id=pad_id)
            dev = target_device

            with _amp(dev):
                logits, act = train_model(
                    batch["input_ids"].to(dev),
                    batch["attention_mask"].to(dev),
                    batch["marker_pos"].to(dev),
                    batch["marker_mask"].to(dev),
                    batch["qtype"].to(dev),
                )

            logits = logits.float()
            mask = batch["marker_mask"].to(dev)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(dev)

            # 1. sample group_size noisy logit distributions with a zero-mean projection
            eps = torch.randn((group_size,) + logits.shape, device=dev) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. strictly proper reward, with a spherical term for soft-target matching
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(dev), mask,
                                  w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. policy gradient over the noise, plus soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / grad_accum + 0.0 * act.sum()

            _optimizer_step(loss, train_model, optimizer, scaler)
            accum_step += 1

            if accum_step % grad_accum == 0 or (b_idx + micro_batch) >= len(my_items):
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1

            epoch_loss += loss.item() * grad_accum
            n_batches += 1
            losses.append(loss.item() * grad_accum)
            rewards.append(float(r.mean().item()))

            if rank == 0 and log_every and (n_batches % log_every) == 0:
                log("  epoch %d/%d | step %d | loss %.4f | reward %.3f | lr %.2e"
                    % (epoch + 1, epochs, n_batches, loss.item() * grad_accum,
                       float(r.mean().item()), scheduler.get_last_lr()[0]))

        if rank == 0:
            log("  epoch %d/%d done in %.1fs | mean loss %.4f"
                % (epoch + 1, epochs, time.time() - t0, epoch_loss / max(1, n_batches)))

    # ---------------------------------------------------------------- save + calibrate
    temperatures = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    if rank == 0:
        model.eval()
        log("Fitting calibration temperatures on %d held-out items..." % min(calib_max, len(items)))
        calib_items = items[::calib_every][:calib_max]
        preds = []
        try:
            with torch.no_grad():
                for c_idx in range(0, len(calib_items), 16):
                    chunk = calib_items[c_idx:c_idx + 16]
                    cb = collate_train_batch(chunk, pad_id=pad_id)
                    with _amp(target_device):
                        l_sub, _ = model(
                            cb["input_ids"].to(target_device),
                            cb["attention_mask"].to(target_device),
                            cb["marker_pos"].to(target_device),
                            cb["marker_mask"].to(target_device),
                            cb["qtype"].to(target_device),
                        )
                    arr = l_sub.float().cpu().numpy()
                    for row, it in enumerate(chunk):
                        preds.append((it["qtype"], arr[row, : len(it["markers"])], it["target"]))
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in preds if q_type == qt]
                if sel:
                    temperatures[qt] = fit_temperature(sel)
            log("Fitted temperatures (choice, score, noul): %s"
                % [round(t, 3) for t in temperatures])
        except Exception as e:                                  # noqa: BLE001
            log("Temperature fitting fallback (%s: %s)" % (type(e).__name__, e))

        os.makedirs(output_dir, exist_ok=True)
        from safetensors.torch import save_file
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-typed-decisions"
        cfg["temperature"] = temperatures
        with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        log("Saved to %s" % output_dir)

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    return {
        "device": str(target_device),
        "world_size": world_size,
        "items": len(items),
        "epochs": epochs,
        "updates": updates,
        "seconds": round(time.time() - t0, 1),
        "final_loss": round(losses[-1], 4) if losses else None,
        "first_loss": round(losses[0], 4) if losses else None,
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "temperatures": [round(t, 4) for t in temperatures],
        "output_dir": output_dir,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RLCD fine-tuning, on CUDA/ROCm, MPS or CPU")
    ap.add_argument("--model-dir", required=True, help="checkpoint holding rl_agent_config.json, encoder/, model.safetensors")
    ap.add_argument("--items", required=True, help="torch.save'd list of pre-tokenized items")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=None, help="train on the first N items (smoke runs)")
    ap.add_argument("--max-len", type=int, default=None,
                    help="context written into the exported config (default: keep the checkpoint's)")
    ap.add_argument("--head-max-len", type=int, default=None,
                    help="option-head budget written into the exported config")
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--init-method", default=None,
                    help="distributed init method, e.g. file:///tmp/laya_ddp_init "
                         "(default: env:// as set up by torchrun)")
    a = ap.parse_args(argv)

    print(device_report(), flush=True)
    device = pick_device(a.device)
    items = torch.load(a.items, weights_only=False)
    print("Loaded %d pre-tokenized items from %s" % (len(items), a.items), flush=True)
    metrics = train_rlcd(
        items, a.model_dir, a.output_dir,
        device=device, epochs=a.epochs, micro_batch=a.micro_batch, grad_accum=a.grad_accum,
        group_size=a.group_size, lr_encoder=a.lr_encoder, lr_head=a.lr_head,
        grad_checkpointing=not a.no_grad_checkpointing, limit=a.limit, log_every=a.log_every,
        max_len=a.max_len, head_max_len=a.head_max_len, init_method=a.init_method,
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
