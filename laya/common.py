"""Core model architecture, token sequence construction, and confidence estimation for laya."""
import json
import math
import os
import threading
from contextlib import nullcontext
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}

# A fast tokenizer is not read-only: `truncation=True` / `padding=True` make it call
# `enable_truncation` / `enable_padding`, which mutates the shared Rust object. One tokenizer is
# parsed per checkpoint directory and shared by every Agent that wants it, so concurrent
# `predict()` calls -- on one Agent or on two sharing the cache -- raced and raised
# `RuntimeError: Already borrowed`. Serialise encoding instead: it is a small fraction of a call
# next to the forward pass, and this keeps the cache's single parse.
_TOKENIZE_LOCK = threading.RLock()


def encode_text(tok, text, **kwargs):
    """Tokenize `text` while holding the lock a shared fast tokenizer needs."""
    with _TOKENIZE_LOCK:
        return tok(text, **kwargs)


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes compact JSON, so a
    rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _resolve_noul_labels(labels=None):
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return false_label, true_label


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul semantic order is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values.
        # `str(k)` unconditionally: a label with no description is rendered as itself, so an int
        # label used to come back as an int from a function annotated `-> List[str]` and then
        # reached `build_sequence`, which calls `.replace` on it and raised an AttributeError
        # naming neither the question nor the label. With a description the same label already
        # went through `"%s: %s" %` and was a str, which is why only the undescribed form broke.
        # `structured._enum_field` stringifies labels the same way; the returned answer still
        # carries the caller's original label, which is unchanged.
        return [str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
                for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label + ": "
        + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label + ": "
        + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
    state_ids: Optional[List[int]] = None,
    return_stats: bool = False,
):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    `state_ids` lets a caller tokenize the shared state once and reuse it across every question,
    instead of re-serializing and re-tokenizing the same document per question.

    `return_stats` adds a third return value describing what the head budget did to the options:
    `options` (how many the question defines), `options_distinct` (how many still have a token
    span of their own) and `tokens_per_option` (the cap applied to each, or None when none was).
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = encode_text(tok, "%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        # Cap at the tokenizer, not after the fact: `[:48]` still makes the tokenizer process the
        # whole (possibly long) description. truncation=True, max_length=48 keeps the first 48
        # tokens, which is exactly what the previous slice produced.
        opt_tokens = encode_text(
            tok,
            " " + opts[i].replace(mask_tok, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )["input_ids"]
        opt_ids.append([tok.mask_token_id] + opt_tokens)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    per_option = None
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        per_option = per
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = encode_text(tok, serialize_state(state).replace(mask_tok, " "),
                                add_special_tokens=False)["input_ids"]
    # not state_ids[-room:]: with no room left, state_ids[-0:] is the whole state rather than none of it
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    ids, markers = ids[:max_len], [m for m in markers if m < max_len]
    if not return_stats:
        return ids, markers
    # Two options that share a prefix can come out of the cut as the same token span: the marker
    # count still matches the option count, so the guard in `Agent._encode_state` passes and
    # nothing downstream can tell that the question lost the ability to name them apart. Counted
    # on the capped option ids, before assembly: re-slicing the finished sequence cannot close
    # the last option's span -- it runs on into the serialized state, which differs per request,
    # so the last option always looks distinguishable however it collided (#538).
    return ids, markers, {
        "options": len(opt_ids),
        "options_distinct": len({tuple(o) for o in opt_ids}),
        "tokens_per_option": per_option,
    }


def collapsed_options(qids, items) -> Dict[str, Dict[str, Optional[int]]]:
    """The questions whose options no longer have a token span each, from per-item stats.

    `total` is the number of options the question defines, not the number of markers that
    reached the sequence: a report counted from the markers would say "43/58" about a request
    where 28 options never made it into the input at all.
    """
    out = {}
    for qid, item in zip(qids, items):
        stats = item.get("options")
        if stats and stats["options_distinct"] < stats["options"]:
            out[qid] = {"total": stats["options"], "distinct": stats["options_distinct"],
                        "tokens_per_option": stats["tokens_per_option"]}
    return out


class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1,
                 no_init: bool = False):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        # `no_init` means the caller is about to load every parameter from a checkpoint, so the
        # head's initial values are pure overhead -- and not just time. transformers'
        # `no_init_weights()` patches the torch.nn.init functions, but something inside
        # nn.TransformerEncoderLayer draws from the RNG outside them, so building it still
        # advanced the global generator and made `load()` a visible side effect. On the meta
        # device no initialisation kernel runs at all; the layers are materialised empty and the
        # load fills them.
        with torch.device("meta") if no_init else nullcontext():
            nhead = max(1, d // 64)
            layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.register_buffer("temperature", torch.ones(3))
        if no_init:
            # Only the modules created above are on the meta device; the encoder is already real
            # and may hold non-persistent buffers (RoPE frequencies) that to_empty would wipe.
            for module in (self.head, self.type_emb, self.scorer, self.act_head):
                if module is not None:
                    module.to_empty(device="cpu")
            self.temperature = torch.empty_like(self.temperature, device="cpu")
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    # Non-reentrant checkpointing also trains the head when its input
                    # is frozen. Default RNG preservation keeps dropout consistent.
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            # A single-option question has exactly one marker, so p.topk(2, ...)
            # has nothing to select for the second slot and raises. The answer
            # is still well-defined: softmax over one logit is 1.0 regardless of
            # its value, so pad the missing second entry with 0.0 - that gives
            # the act head top1 - top2 == 1.0, the same "fully decided" signal
            # it would see for any other unambiguous top-1-vs-rest gap.
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def _apply_rope_config(ecfg) -> None:
    """Carry transformers>=5 per-layer RoPE settings over to the attributes 4.x reads.

    A checkpoint re-saved by transformers 5 stores RoPE as
    `rope_parameters = {"full_attention": {"rope_theta": ...}, "sliding_attention": {...}}`.
    transformers 4.x does not know that key, so it keeps its own defaults (global 160000,
    local 10000) and any checkpoint whose sliding-attention theta differs silently runs the
    wrong RoPE base -- mmBERT is exactly that case, both of its thetas are 160000. Map the
    values onto `global_rope_theta` / `local_rope_theta`, which 4.x does read. On
    transformers 5 this is a no-op beyond re-setting the same numbers.
    """
    rope = getattr(ecfg, "rope_parameters", None)
    if not isinstance(rope, dict):
        return
    flat = rope.get("rope_theta")
    for layer_type, attr in (("full_attention", "global_rope_theta"),
                             ("sliding_attention", "local_rope_theta")):
        params = rope.get(layer_type)
        theta = params.get("rope_theta") if isinstance(params, dict) else flat
        if theta is not None and hasattr(ecfg, attr):
            setattr(ecfg, attr, float(theta))


def _no_init_weights():
    """`no_init_weights` lives in different modules across transformers versions."""
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers 4.x
        from transformers.modeling_utils import no_init_weights
    return no_init_weights()


def build_model(cfg: Dict, encoder_dir: Optional[str] = None, pretrained: bool = True,
                revision: Optional[str] = None) -> DecisionModel:
    """Build the decision model described by `cfg`.

    With `pretrained=False`, or when `encoder_dir` holds a saved encoder config, nothing is
    downloaded and **no parameter is initialised**: the caller is expected to load a checkpoint
    into the result with `load_state_dict(..., strict=True)` immediately. Skipping initialisation
    keeps `load()` from spending time on, or consuming RNG for, weights it is about to overwrite.
    """
    from transformers import AutoConfig, AutoModel

    head_layers, n_act = cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1
    if not pretrained or (encoder_dir and os.path.exists(encoder_dir)):
        ecfg = AutoConfig.from_pretrained(encoder_dir or cfg["encoder"])
        _apply_rope_config(ecfg)
        with _no_init_weights():
            enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        return DecisionModel(enc, head_layers, n_act, no_init=True)
    # Training-time Hub load of the base encoder; allow pinning it like the checkpoints.
    kw = {"attn_implementation": "sdpa"}
    if revision:
        kw["revision"] = revision
    enc = AutoModel.from_pretrained(cfg["encoder"], **kw)
    return DecisionModel(enc, head_layers, n_act)


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def answer_confidence(p: np.ndarray, k: int) -> float:
    """Probability mass on the answer being reported: max(p).

    This is the quantity temperature scaling fits, and the quantity every calibration figure in
    this repository is computed on -- both benchmark harnesses take `conf = max(probs)` before
    calling `ece_score`. It is therefore the one confidence with the property the README's
    gating section relies on: of the answers returned at confidence c, about c of them are right.

    `confidence_from_probs` below reports a different quantity on a different scale and carries
    no such guarantee, so the two must not be compared against the same threshold.
    """
    if k < 1:
        return 1.0
    return float(np.clip(np.max(p[:k]), 0.0, 1.0))


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k).

    How concentrated the whole distribution is. Useful, but not calibrated: it is not what
    temperature scaling fits and not what the reported ECE measures. See `answer_confidence`.
    """
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


# A fitted temperature below 1 sharpens the logits instead of softening them. The shipped
# `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is published as
# 0.99, so a caller gating on confidence is told a coin flip is a certainty. No honest calibration
# needs to sharpen this hard, so refuse to apply one that does.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):    # NaN / inf
        return 1.0
    return min(hi, max(lo, t))


def amp_dtype(name: Optional[str]) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    has_target = any("target" in it for it in items)
    target = torch.zeros((n, kmax), dtype=torch.float32) if has_target else None

    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if has_target and "target" in it:
            if len(it["target"]) > k:
                # Otherwise this lands as "The expanded size of the tensor (k) must match the
                # existing size (kmax)" from inside the assignment, which says nothing about the
                # actual mistake: a target with more entries than the item has options. The limit
                # is this item's own marker count, not the batch-wide kmax: in a mixed-width
                # batch a longer sibling row must not legitimise extra entries (#311).
                raise ValueError(
                    "collate_items: item %d has %d target entries but only %d marker positions; "
                    "a target needs one entry per option" % (i, len(it["target"]), k))
            target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)

    res = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it.get("label", -1) for it in items]),
        "meta": [{k: it[k] for k in it if k not in ("ids", "markers", "target")} for it in items],
    }
    if target is not None:
        res["target"] = target
    return res
