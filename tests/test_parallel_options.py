"""`option_layout="parallel"` must make the decision logits permute with the options, exactly.

In the sequential layout every option sits at later positions than the one before it and attends
to all the others, so the slot an option is listed in changes its score: five identical options
spread by 3.59 logits on the English checkpoint (`tests/test_option_order.py`). The parallel layout
gives every option the same starting position and stops options attending to each other; the head
on top has no positional encoding, so reordering the options can only reorder the logits.

The checks run a tiny randomly initialised ModernBERT inside the real `DecisionModel`, so no
weights are downloaded. They pin:

- the layout itself (`parallel_layout`): shared option positions, the state after the longest option;
- the masks: with no options marked, they reproduce the encoder's own padding/sliding-window masks;
- equivariance: under any option order the parallel logits are the canonical ones permuted,
  while the sequential layout on the same weights is not (so the check can fail);
- the plumbing: `collate_items` carries the layout, refuses a mixed batch, and the config gate.

Run: python tests/test_parallel_options.py
"""
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers  # noqa: E402

if int(transformers.__version__.split(".")[0]) < 5:
    print("skipped: the parallel layout needs transformers>=5, found %s" % transformers.__version__)
    sys.exit(0)

from transformers import ModernBertConfig, ModernBertModel  # noqa: E402

from laya.common import (  # noqa: E402
    DecisionModel,
    build_sequence,
    collate_items,
    parallel_layout,
    parallel_option_masks,
    uses_parallel_layout,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
        return
    except Exception as e:                       # noqa: BLE001 - the point is the wrong type
        FAIL.append("%s: raised %s" % (name, type(e).__name__))
        return
    FAIL.append("%s: did not raise" % name)


class StubTok:
    mask_token = "[MASK]"
    mask_token_id, cls_token_id, sep_token_id, pad_token_id = 3, 1, 2, 0

    def __init__(self):
        self._vocab = {}

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [self._vocab.setdefault(w, len(self._vocab) + 10) for w in str(text).split()]
        return {"input_ids": ids[:max_length] if (truncation and max_length) else ids}


# ------------------------------------------------------------------ the layout
# [CLS] a b [SEP] | [MASK] x | [MASK] y z | [MASK] | [SEP] s s [SEP]
#   0   1 2   3      4    5     6    7 8     9       10   11 12 13
lay = parallel_layout([4, 6, 9], head_len=11, length=14)
check("layout/instruction keeps its positions", lay["position_ids"][:4], [0, 1, 2, 3])
check("layout/every option starts at the same position",
      [lay["position_ids"][m] for m in (4, 6, 9)], [4, 4, 4])
check("layout/option tokens count up inside their span", lay["position_ids"][4:10], [4, 5, 4, 5, 6, 4])
check("layout/closing [SEP] and state follow the longest option", lay["position_ids"][10:], [7, 8, 9, 10])
check("layout/option ids mark each slot, 0 elsewhere", lay["option_ids"], [0] * 4 + [1, 1, 2, 2, 2, 3] + [0] * 4)
check("layout/no options is the identity", parallel_layout([], 4, 6)["position_ids"], list(range(6)))

tok = StubTok()
QI = {"t": "choice", "ins": "which team handles this",
      "crit": {"refund": "money back please", "tech": "a bug", "sales": "pricing", "other": "none of these fit"}}
STATE = {"msg": " ".join("w%d" % i for i in range(40))}
ids, markers, layout = build_sequence(tok, STATE, QI, 512, 192, return_layout=True)
check("build/one layout entry per token", (len(layout["position_ids"]), len(layout["option_ids"])), (len(ids),) * 2)
check("build/without the flag the return shape is unchanged", len(build_sequence(tok, STATE, QI, 512, 192)), 2)
short_ids, short_markers, short_layout = build_sequence(tok, STATE, QI, 24, 192, return_layout=True)
check("build/clamped to max_len with the ids", len(short_layout["option_ids"]), len(short_ids))
check_true("build/a clamp does not widen the last surviving option",
           short_layout["option_ids"][:len(short_ids)] == layout["option_ids"][:len(short_ids)])

# ------------------------------------------------------------------ the model
torch.manual_seed(0)
enc_cfg = ModernBertConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=4, local_attention=8, global_attn_every_n_layers=2,
                           pad_token_id=0, cls_token_id=1, sep_token_id=2, bos_token_id=1, eos_token_id=2,
                           attn_implementation="sdpa")
model = DecisionModel(ModernBertModel(enc_cfg), head_layers=2).eval()


def logits_for(order, parallel):
    seq, mk, *lay_ = build_sequence(tok, STATE, QI, 512, 192, option_order=order, return_layout=parallel)
    item = {"ids": seq, "markers": mk, "qtype": 0}
    if lay_:
        item["layout"] = lay_[0]
    b = collate_items([[item]], tok.pad_token_id)
    extra = {"position_ids": b["position_ids"], "option_ids": b["option_ids"]} if parallel else {}
    with torch.no_grad():
        logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                            **extra)
    return logits[0], act[0]


# ------------------------------------------------------------------ masks match the stock ones
# No options marked and plain positions: the hand-built masks must be the encoder's own, padding included.
batch = collate_items([[{"ids": ids, "markers": markers, "qtype": 0},
                        {"ids": ids[:30], "markers": markers, "qtype": 0}]], tok.pad_token_id)
plain_pos = torch.arange(batch["input_ids"].shape[1]).expand_as(batch["input_ids"])
masks = parallel_option_masks(batch["attention_mask"], plain_pos, torch.zeros_like(plain_pos),
                              enc_cfg.sliding_window)
with torch.no_grad():
    stock = model.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    ours = model.encoder(input_ids=batch["input_ids"], attention_mask=masks, position_ids=plain_pos).last_hidden_state
real = batch["attention_mask"].bool()
check_true("masks/reproduce the stock padding + sliding-window masks",
           torch.allclose(stock[real], ours[real], atol=1e-5),
           "max diff %.2e" % (stock[real] - ours[real]).abs().max().item())

# ------------------------------------------------------------------ equivariance
canon_par, canon_act = logits_for(None, parallel=True)
worst_par, worst_seq, worst_act = 0.0, 0.0, 0.0
canon_seq, _ = logits_for(None, parallel=False)
for order in itertools.permutations(range(4)):
    order = list(order)
    par, act = logits_for(order, parallel=True)
    seq_, _ = logits_for(order, parallel=False)
    # slot s shows option order[s], so slot s must score what option order[s] scored canonically
    worst_par = max(worst_par, (par[:4] - canon_par[order]).abs().max().item())
    worst_seq = max(worst_seq, (seq_[:4] - canon_seq[order]).abs().max().item())
    worst_act = max(worst_act, (act - canon_act).abs().max().item())
check_true("equivariant/parallel logits permute with the options (all 24 orders)", worst_par < 1e-5,
           "max diff %.2e" % worst_par)
check_true("equivariant/the act head is order-blind too", worst_act < 1e-5, "max diff %.2e" % worst_act)
check_true("equivariant/the sequential layout is not (the check can fail)", worst_seq > 1e-4,
           "max diff %.2e" % worst_seq)

# ------------------------------------------------------------------ plumbing
with_layout = {"ids": ids, "markers": markers, "qtype": 0, "layout": layout}
b = collate_items([[with_layout]], tok.pad_token_id)
check("collate/carries position ids", b["position_ids"][0].tolist(), layout["position_ids"])
check_true("collate/keeps the layout out of meta", "layout" not in b["meta"][0])
check_true("collate/no layout, no layout tensors",
           "option_ids" not in collate_items([[{"ids": ids, "markers": markers, "qtype": 0}]], 0))
check_raises("collate/refuses a batch mixing layouts", ValueError,
             lambda: collate_items([[with_layout, {"ids": ids, "markers": markers, "qtype": 0}]], 0))

check("config/published checkpoints stay sequential", uses_parallel_layout({}), False)
check("config/parallel is opt-in", uses_parallel_layout({"option_layout": "parallel"}), True)
check_raises("config/rejects an unknown layout", ValueError, lambda: uses_parallel_layout({"option_layout": "set"}))

if FAIL:
    print("FAILED:\n  " + "\n  ".join(FAIL))
    sys.exit(1)
print("all parallel-option tests passed (%d checks)" % len(PASS))
