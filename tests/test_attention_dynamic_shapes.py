"""The decision head's attention must keep its shape when traced.

`nn.MultiheadAttention` reshapes the packed projection with sizes captured while tracing, so an
ONNX graph exported from a short dummy input only runs at that length: `tests/test_onnx.py` failed
on a 53-token input from an export traced at 16. `_DynamicMultiheadAttention` writes the same
reshape with constant-only shape arguments, which keeps it symbolic.

This traces the module the way the exporter does and runs it at a different length, then checks the
maths still matches both the stock module and an explicit reference. It needs neither onnx nor a
checkpoint, so it runs in the plain CI lanes.

    python tests/test_attention_dynamic_shapes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import BertConfig, BertModel  # noqa: E402

from laya.common import DecisionModel, _DynamicMultiheadAttention  # noqa: E402

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
        FAIL.append("%s%s" % (name, (": " + detail) if detail else ""))


class Wrapped(nn.Module):
    """The call the encoder layer makes on the traced path: self-attention, no weights."""

    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]


D_MODEL, HEADS = 64, 4
STOCK = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True).eval()
DYNAMIC = _DynamicMultiheadAttention(D_MODEL, HEADS, batch_first=True).eval()
DYNAMIC.load_state_dict(STOCK.state_dict())
check("head/exports the same state_dict keys as the stock module",
      list(DYNAMIC.state_dict()), list(STOCK.state_dict()))

torch.manual_seed(0)
X = torch.randn(2, 53, D_MODEL)
with torch.no_grad():
    stock_out = STOCK(X, X, X, need_weights=False)[0]
    dynamic_out = DYNAMIC(X, X, X, need_weights=False)[0]
    parts = F.linear(X, STOCK.in_proj_weight, STOCK.in_proj_bias).chunk(3, dim=-1)
    q, k, v = (part.unflatten(-1, (HEADS, D_MODEL // HEADS)).transpose(1, 2) for part in parts)
    manual = STOCK.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(-2))

check_true("head/eager output matches the explicit reference",
           torch.allclose(manual, dynamic_out, atol=1e-6), "max diff %.2e" % (manual - dynamic_out).abs().max())
check_true("head/eager output matches the stock module",
           torch.allclose(stock_out, dynamic_out, atol=1e-6), "max diff %.2e" % (stock_out - dynamic_out).abs().max())

# The encoder layer canonicalises a bool padding mask to 0 / -inf before attention sees it.
padded = torch.zeros(2, 53)
padded[:, 40:] = float("-inf")
with torch.no_grad():
    stock_padded = STOCK(X, X, X, key_padding_mask=padded, need_weights=False)[0]
    dynamic_padded = DYNAMIC(X, X, X, key_padding_mask=padded, need_weights=False)[0]
check_true("head/padded output matches the stock module",
           torch.allclose(stock_padded, dynamic_padded, atol=1e-6),
           "max diff %.2e" % (stock_padded - dynamic_padded).abs().max())

# Trace at a short length, then run at a longer one -- the regression this guards.
traced = torch.jit.trace(Wrapped(DYNAMIC).eval(), (torch.randn(2, 5, D_MODEL),), strict=False)
out = traced(torch.randn(2, 53, D_MODEL))
check("head/traced at 5 tokens still runs at 53", tuple(out.shape), (2, 53, D_MODEL))

encoder = BertModel(BertConfig(vocab_size=8, hidden_size=D_MODEL, num_hidden_layers=1,
                               num_attention_heads=HEADS, intermediate_size=128))
model = DecisionModel(encoder, head_layers=1)
check("head/DecisionModel uses the dynamic attention",
      isinstance(model.head.layers[0].self_attn, _DynamicMultiheadAttention), True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
