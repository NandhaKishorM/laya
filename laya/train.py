"""Fine-tune a Laya checkpoint on typed decisions: one training loop the scripts can share.

The notebook, the Apple Silicon script and `research/scripts/finetune_single_device.py` each
carry their own copy of the same loop. This module is that loop once, with two additions the
copies do not have:

- **A choice of objective.** `loss="rlcd"` (the default) is the notebook's objective: a
  GRPO-style term over noisy logit samples rewarded by `proper_reward`, plus soft
  cross-entropy against the target distribution. `loss="soft-ce"` trains on the soft
  cross-entropy alone; #741 measured no gain from the extra term on the typed-decisions split.
- **Option-order augmentation (opt-in).** The scripts always present options in the order the
  question lists them, so the model can learn a position prior: the 20-option order-flip rate in
  `BENCHMARKS.md` and #131 are both that symptom. Each epoch, items of the types named in
  `shuffle_options` (for example `("choice",)`) are re-encoded with a random `option_order` -- the same path inference
  uses -- and the target is permuted with them, so slot `s` is trained on option
  `option_order[s]`.

Calibration goes through `laya.calibrate.fit_temperature_map`, so a fine-tuned checkpoint is
fitted with the same clamp and buckets the runtime applies instead of a local copy of the
fitter.

Items keep the question and the tokenized state rather than a finished sequence, because a
shuffled epoch needs to rebuild the head. States are tokenized once per row and shared by
every question on it.

No new dependency: torch, transformers and safetensors are already required by the package.
"""
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .calibrate import fit_temperature_map
from .common import (
    QTYPES,
    build_head,
    build_model,
    build_sequence,
    collate_items,
    encode_text,
    proper_reward,
    render_options,
    serialize_state,
)

LOSSES = ("soft-ce", "rlcd")
_MASKED = -1e4


@dataclass
class TrainConfig:
    """Training knobs. The defaults train the way the published notebook does.

    The notebook's effective batch of 64 is 8 per micro-batch x 2 GPUs x 4 accumulation steps;
    on one device that is `micro_batch=8, grad_accum=8`.

    `max_len` / `head_max_len` default to the checkpoint's own config. Whatever is used is
    written into the saved config, so inference sees the same budgets training did.
    `amp` defaults to on for CUDA only; `gradient_checkpointing` defaults to following `amp`.
    """

    epochs: int = 4
    micro_batch: int = 8
    grad_accum: int = 8
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    loss: str = "rlcd"
    rl_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0
    shuffle_options: Tuple[str, ...] = ()
    freeze_encoder: bool = False
    calib_max: int = 400
    calib_frac: float = 0.1
    calib_seed: int = 20260922
    seed: int = 0
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    amp: Optional[bool] = None
    gradient_checkpointing: Optional[bool] = None
    log_every: int = 100

    def validate(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError("loss must be one of %s, got %r" % (", ".join(LOSSES), self.loss))
        unknown = sorted(set(self.shuffle_options) - set(QTYPES))
        if unknown:
            raise ValueError("shuffle_options names unknown question type(s): %s" % ", ".join(unknown))
        for name in ("epochs", "micro_batch", "grad_accum", "rl_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("%s must be a positive integer, got %r" % (name, value))
        if not 0.0 <= self.calib_frac < 1.0:
            raise ValueError("calib_frac must be in [0, 1), got %r" % (self.calib_frac,))


# ------------------------------------------------------------------------------------- data

def to_internal(qid: Any, question: Dict[str, Any]) -> Dict[str, Any]:
    """The question as inference sees it: `Agent._check_question` then `Agent._to_internal`.

    Training on any other rendering would teach the model option texts it never sees at
    inference -- a list-form choice, custom noul `labels`, a non-string instruction.
    """
    from .agent import Agent

    Agent._check_question(qid, question)
    q = Agent._to_internal(question)
    q.pop("option_order", None)    # training draws its own order per epoch
    return q


def target_from_gold(q: Dict[str, Any], gold_q: Dict[str, Any]) -> List[float]:
    """The normalised target distribution for an internal question (`to_internal`), in option order.

    `gold_q["probabilities"]` is keyed by choice label, by `"false"`/`"true"` for noul, and by
    the level index as a string for score -- the `{state, questions, gold}` schema in
    `docs/finetune.md`. A choice label that is not a string is also looked up as its JSON key.
    A row whose probabilities sum to zero becomes uniform, as in the scripts. A negative or
    non-finite probability raises `ValueError`.
    """
    probs = gold_q["probabilities"]

    def get(key, default):
        return probs.get(key, probs.get(str(key), default))

    t, crit = q["t"], q["crit"]
    if t == "choice":
        target = [float(get(k, 0.0)) for k in crit.keys()]
    elif t == "noul":
        target = [float(get("false", 0.5)), float(get("true", 0.5))]
    else:
        target = [float(get(str(i), 0.0)) for i in range(len(crit))]
    if any(not math.isfinite(v) or v < 0 for v in target):
        raise ValueError("probabilities must be finite and non-negative, got %r" % (target,))
    total = sum(target)
    if total > 0:
        return [v / total for v in target]
    return [1.0 / len(target)] * len(target)


def encode_state(tok, state: Any, max_len: int) -> List[int]:
    """State token ids, capped at `max_len`: no sequence can hold more of the state than that."""
    text = serialize_state(state).replace(tok.mask_token, " ")
    return encode_text(tok, text, add_special_tokens=False)["input_ids"][:max_len]


def make_item(tok, q: Dict[str, Any], target: Sequence[float], state_ids: List[int],
              head_max_len: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """`(item, None)`, or `(None, reason)` when the question cannot be trained on as written.

    `q` is an internal question (`to_internal`). The reasons are `target_mismatch` (the target
    does not have one entry per option) and `options_collapsed` (the head budget left two options
    with the same token span, #538). Training on either would teach the model to tell apart
    options it cannot see as different.
    """
    k = len(render_options(q))
    if len(target) != k:
        return None, "target_mismatch"
    _ids, markers, stats = build_head(tok, q, head_max_len)
    if len(markers) != k or stats["options_distinct"] < stats["options"]:
        return None, "options_collapsed"
    return {"q": q, "state_ids": state_ids, "target": [float(v) for v in target],
            "qtype": QTYPES[q["t"]], "k": k}, None


def items_from_rows(tok, rows: Iterable[Dict[str, Any]], max_len: int,
                    head_max_len: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Items from `{state, questions, gold}` rows; returns `(items, skipped)`.

    `skipped` counts, by reason, questions that were labelled but could not become an item:
    `invalid_question` and `invalid_target` (a `ValueError` from `to_internal` or
    `target_from_gold`), plus the two reasons `make_item` gives. Questions with no gold entry
    are not counted -- the row simply does not label them.
    """
    items, skipped = [], {}
    for row in rows:
        state_ids = encode_state(tok, row["state"], max_len)
        for qid, question in row["questions"].items():
            gold_q = row["gold"].get(qid)
            if gold_q is None:
                continue
            reason = None
            try:
                q = to_internal(qid, question)
            except ValueError:
                reason = "invalid_question"
            if reason is None:
                try:
                    target = target_from_gold(q, gold_q)
                except (ValueError, TypeError, KeyError):
                    reason = "invalid_target"
            if reason is None:
                item, reason = make_item(tok, q, target, state_ids, head_max_len)
            if reason is None:
                items.append(item)
            else:
                skipped[reason] = skipped.get(reason, 0) + 1
    return items, skipped


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def encode_item(tok, item: Dict[str, Any], max_len: int, head_max_len: int,
                option_order: Optional[List[int]] = None) -> Dict[str, Any]:
    """The model input for an item, with options in `option_order` and the target to match.

    `build_sequence` puts option `option_order[s]` in slot `s`, so the slot-ordered target is
    `target[option_order[s]]` -- the inverse of what `unpermute_probs` does at inference.
    """
    ids, markers = build_sequence(tok, None, item["q"], max_len, head_max_len,
                                  option_order=option_order, state_ids=item["state_ids"])
    target = item["target"]
    if option_order is not None:
        target = [target[i] for i in option_order]
    return {"ids": ids, "markers": markers, "qtype": item["qtype"], "target": target}


def draw_option_order(item: Dict[str, Any], rng: random.Random,
                      shuffle_types: Sequence[str]) -> Optional[List[int]]:
    """A random permutation for items whose type is in `shuffle_types`, else None (canonical)."""
    if item["q"]["t"] not in shuffle_types or item["k"] < 2:
        return None
    order = list(range(item["k"]))
    rng.shuffle(order)
    return order


def split_calibration(items: Sequence[Any], calib_max: int, calib_frac: float,
                      seed: int) -> Tuple[List[Any], List[Any]]:
    """`(train, calibration)`, with the calibration slice taken before any training.

    Fitting temperatures on items the run has trained on measures the fit, not the calibration
    (`docs/finetune.md`). The split depends only on `seed` and the item count, so every DDP rank
    and every rerun hold out the same items.
    """
    n_calib = min(calib_max, int(len(items) * calib_frac))
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    held = set(order[:n_calib])
    train = [it for i, it in enumerate(items) if i not in held]
    calib = [it for i, it in enumerate(items) if i in held]
    return train, calib


# ----------------------------------------------------------------------------------- losses

def soft_ce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a target distribution, over each row's own options only."""
    logp = torch.log_softmax(logits.masked_fill(~mask, _MASKED), -1)
    return -(target * logp * mask).sum(-1).mean()


def rlcd_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, qtype: torch.Tensor,
              sigma: float, samples: int = 4, w_sph: float = 0.75, w_rps: float = 1.0) -> torch.Tensor:
    """The notebook objective: a policy-gradient term over noisy logits plus soft cross-entropy.

    Zero-mean Gaussian noise (scaled by `sigma`, projected onto each row's options) is added to
    detached logits; each sample is rewarded by `proper_reward` against the target, and the
    normalised advantage weights the Gaussian log-density of the sample.
    """
    k = mask.sum(-1, keepdim=True).float()
    fmask = mask.float()
    eps = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * fmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, _MASKED), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, fmask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * sigma ** 2)
    return -(adv * logp).mean() + soft_ce_loss(logits, target, mask)


def sigma_at(epoch: int, epochs: int, start: float, end: float) -> float:
    """Exploration noise for `epoch`, annealed linearly from `start` to `end`."""
    return start + (end - start) * (epoch / max(1, epochs - 1))


# ------------------------------------------------------------------------------- checkpoint

def resolve_device(device: Optional[str] = "auto") -> torch.device:
    if device not in (None, "auto"):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def load_checkpoint(model_dir: str):
    """`(model, tokenizer, cfg)` from a local checkpoint directory, model on CPU in fp32."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from .agent import _fix_tokenizer_config

    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return model.float(), tok, cfg


def save_checkpoint(model, tok, cfg: Dict[str, Any], path: str) -> None:
    """Write the layout `laya.load` reads: weights in fp16, encoder config, tokenizer, config."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    save_file({k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()},
              os.path.join(path, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(path, "encoder"))
    tok.save_pretrained(os.path.join(path, "tokenizer"))
    tmp = os.path.join(path, "rl_agent_config.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, os.path.join(path, "rl_agent_config.json"))


# ------------------------------------------------------------------------------------ train

def _forward(model, batch, device, amp: bool, detach_encoder: bool):
    args = (batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
    if amp:
        with torch.autocast(device.type, dtype=torch.float16):
            logits, _act = model(*args, detach_encoder=detach_encoder)
    else:
        logits, _act = model(*args, detach_encoder=detach_encoder)
    return logits.float()


def train_model(model, tok, items: Sequence[Dict[str, Any]], config: TrainConfig, device: torch.device,
                max_len: int, head_max_len: int,
                on_epoch_end: Optional[Callable[[int, float], None]] = None) -> List[float]:
    """Train `model` in place on `items`; returns the mean loss of each epoch."""
    config.validate()
    if not items:
        raise ValueError("no training items")
    amp = (device.type == "cuda") if config.amp is None else bool(config.amp)
    checkpointing = amp if config.gradient_checkpointing is None else bool(config.gradient_checkpointing)

    if config.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = checkpointing
    model.to(device).train()
    if config.freeze_encoder:
        # A frozen encoder is a fixed feature extractor: dropout in it would only add noise.
        model.encoder.eval()

    groups = [{"params": [p for n, p in model.named_parameters()
                          if not n.startswith("encoder.") and p.requires_grad], "lr": config.head_lr}]
    if not config.freeze_encoder:
        groups.insert(0, {"params": [p for n, p in model.named_parameters()
                                     if n.startswith("encoder.") and p.requires_grad], "lr": config.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(items) / config.micro_batch)
    updates = max(1, math.ceil(steps_per_epoch / config.grad_accum) * config.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=config.min_lr)
    scaler = torch.amp.GradScaler("cuda") if amp and device.type == "cuda" else None

    torch.manual_seed(config.seed)
    order_rng = random.Random(config.seed)
    params = [p for g in groups for p in g["params"]]
    history = []
    for epoch in range(config.epochs):
        epoch_items = list(items)
        random.Random(config.seed + epoch).shuffle(epoch_items)
        sigma = sigma_at(epoch, config.epochs, config.sigma_start, config.sigma_end)
        total, n_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(epoch_items), config.micro_batch):
            chunk = [encode_item(tok, it, max_len, head_max_len,
                                 draw_option_order(it, order_rng, config.shuffle_options))
                     for it in epoch_items[start:start + config.micro_batch]]
            batch = collate_items([chunk], tok.pad_token_id)
            logits = _forward(model, batch, device, amp, config.freeze_encoder)
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            if config.loss == "rlcd":
                loss = rlcd_loss(logits, target, mask, batch["qtype"].to(device), sigma,
                                 config.rl_samples, config.w_sph, config.w_rps)
            else:
                loss = soft_ce_loss(logits, target, mask)
            scaled = loss / config.grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            n_steps += 1
            if n_steps % config.grad_accum == 0 or start + config.micro_batch >= len(epoch_items):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item()
            if config.log_every and n_steps % config.log_every == 0:
                print("epoch %d/%d step %d loss %.4f" % (epoch + 1, config.epochs, n_steps, loss.item()), flush=True)
        mean = total / max(1, n_steps)
        history.append(mean)
        print("epoch %d/%d mean loss %.4f" % (epoch + 1, config.epochs, mean), flush=True)
        if on_epoch_end is not None:
            on_epoch_end(epoch, mean)
    model.eval()
    return history


@torch.no_grad()
def calibration_records(model, tok, items: Sequence[Dict[str, Any]], device: torch.device,
                        max_len: int, head_max_len: int, batch_size: int = 16):
    """`(qtype, logits, target, k)` records for `fit_temperature_map`, options in canonical order."""
    model.eval()
    records = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = collate_items([[encode_item(tok, it, max_len, head_max_len) for it in chunk]], tok.pad_token_id)
        logits = _forward(model, batch, device, amp=False, detach_encoder=False).cpu().numpy()
        for row, it in zip(logits, chunk):
            records.append((it["qtype"], row[:it["k"]], it["target"], it["k"]))
    return records


def finetune(data: str, model_dir: str, output_dir: str, config: Optional[TrainConfig] = None,
             device: Optional[str] = "auto") -> Dict[str, Any]:
    """Preprocess, train, calibrate and save; returns a summary of the run.

    `data` is a JSONL file of `{state, questions, gold}` rows (the schema in `docs/finetune.md`).
    `model_dir` is a local checkpoint directory -- the layout `laya.load` reads. The result in
    `output_dir` loads with `laya.load(output_dir)`.
    """
    config = config or TrainConfig()
    config.validate()
    dev = resolve_device(device)
    model, tok, cfg = load_checkpoint(model_dir)
    max_len = config.max_len or cfg.get("max_len", 512)
    head_max_len = config.head_max_len or cfg.get("head_max_len", 192)

    items, skipped = items_from_rows(tok, read_jsonl(data), max_len, head_max_len)
    if not items:
        raise ValueError("%s produced no training items (skipped: %r)" % (data, skipped))
    train_items, calib_items = split_calibration(items, config.calib_max, config.calib_frac, config.calib_seed)
    print("train items %d, calibration items %d, skipped %r, device %s"
          % (len(train_items), len(calib_items), skipped, dev), flush=True)

    def checkpoint_latest(epoch, _loss):
        save_checkpoint(model, tok, dict(cfg, max_len=max_len, head_max_len=head_max_len),
                        os.path.join(output_dir, "checkpoint_latest"))

    history = train_model(model, tok, train_items, config, dev, max_len, head_max_len,
                          on_epoch_end=checkpoint_latest)

    fitted = fit_temperature_map(calibration_records(model, tok, calib_items, dev, max_len, head_max_len))
    out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True,
                   temperature=fitted["temperature"])
    # An inherited bucket map takes precedence at inference and would mask the new fit.
    out_cfg.pop("temperature_by_options", None)
    if fitted["temperature_by_options"]:
        out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
    out_cfg["training"] = dict(out_cfg.get("training") or {}, laya_train=asdict(config))
    save_checkpoint(model, tok, out_cfg, output_dir)
    return {
        "train_items": len(train_items),
        "calibration_items": len(calib_items),
        "skipped": skipped,
        "epoch_loss": history,
        "temperature": fitted["temperature"],
        "temperature_by_options": fitted["temperature_by_options"],
        "n_by_bucket": fitted["n_by_bucket"],
        "output_dir": output_dir,
    }
