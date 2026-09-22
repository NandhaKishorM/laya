"""Regression tests for the truncation signal (NandhaKishorM/laya#174).

build_sequence used to cut state/head tokens silently and return only
(ids, markers), so no caller could know input was lost. With
return_info=True it reports what survived; Agent.system_one exposes it
per answer as "truncated". Offline: stub tokenizer, no weights.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import build_sequence  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


class StubTok:
    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [100 + (abs(hash(w)) % 900) for w in text.split()]}


tok = StubTok()
Q = {"t": "choice", "ins": "pick one", "crit": {"a": "first", "b": "second"}}

# ---------------------------------------------------------------- default shape unchanged
ids, markers = build_sequence(tok, "hello world", Q, 512, 192)
check("default/returns pair", isinstance(ids, list) and isinstance(markers, list), True)

# ---------------------------------------------------------------- short state: nothing cut
ids2, m2, info = build_sequence(tok, "hello world", Q, 512, 192, return_info=True)
check("short/ids identical", ids2, ids)
check("short/markers identical", m2, markers)
check("short/not truncated", info["state_truncated"], False)
check("short/head intact", info["head_truncated"], False)
check("short/kept equals total", info["state_kept"], info["state_tokens"])

# ---------------------------------------------------------------- long state: cut, flagged, ids stable
# Varied words so head-keep and tail-keep produce different sequences.
long_state = " ".join("w%d" % i for i in range(2000))
plain = build_sequence(tok, long_state, Q, 512, 192)
ids3, m3, info3 = build_sequence(tok, long_state, Q, 512, 192, return_info=True)
check("long/ids identical with and without info", (ids3, m3), plain)
check("long/capped at max_len", len(ids3) <= 512, True)
check("long/state flagged truncated", info3["state_truncated"], True)
check("long/kept below total", info3["state_kept"] < info3["state_tokens"], True)
check("long/total matches tokenizer", info3["state_tokens"], len(tok(long_state)["input_ids"]))

# ---------------------------------------------------------------- truncate_left keeps the tail
_, _, infoL = build_sequence(tok, long_state, Q, 512, 192, truncate_left=True, return_info=True)
check("left/still flagged", infoL["state_truncated"], True)
check("left/flag records direction", infoL["truncate_left"], True)
idsL, _, _ = build_sequence(tok, long_state, Q, 512, 192, truncate_left=True, return_info=True)
idsR, _, _ = build_sequence(tok, long_state, Q, 512, 192, return_info=True)
check("left/keeps tail not head", idsL != idsR, True)

# ---------------------------------------------------------------- long instructions cut the head
big_q = {"t": "choice", "ins": "word " * 2000, "crit": {"a": "first", "b": "second"}}
_, _, infoH = build_sequence(tok, "hi", big_q, 512, 192, return_info=True)
check("head/long instructions flagged", infoH["head_truncated"], True)
check("head/short state untouched", infoH["state_truncated"], False)

# ---------------------------------------------------------------- dict state serializes then measures
_, _, infoD = build_sequence(tok, {"k": "v " * 2000}, Q, 512, 192, return_info=True)
check("dict/long dict state flagged", infoD["state_truncated"], True)

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all truncation-signal tests passed")
sys.exit(1 if FAIL else 0)
