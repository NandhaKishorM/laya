"""Concurrent sequence building must not race on the shared fast tokenizer.

`build_sequence` asks the tokenizer for truncated option text (`truncation=True`), and that mutates
the underlying Rust object through `enable_truncation`. One tokenizer is parsed per checkpoint
directory and shared across Agents, so two threads encoding at once used to raise
`RuntimeError: Already borrowed`. `common.encode_text` serialises those calls; this guards it.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from laya.common import build_sequence  # noqa: E402

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


VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
for i in range(200):
    VOCAB["tok%d" % i] = 5 + i

TOKENIZER = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(WordLevel(VOCAB, unk_token="[UNK]")),
    pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
    sep_token="[SEP]", mask_token="[MASK]",
)

# Long enough that truncation does real work on every option, not just enable itself.
QUESTION = {
    "t": "choice",
    "ins": "Which team should handle `message`?",
    "crit": {"billing": " ".join("tok%d" % (i * 3 % 200) for i in range(80)),
             "technical": " ".join("tok%d" % (i * 7 % 200) for i in range(80))},
}
STATE = " ".join("tok%d" % (i * 11 % 200) for i in range(600))

errors, results = [], []


def worker():
    try:
        for _ in range(40):
            seq, markers = build_sequence(TOKENIZER, STATE, QUESTION, 512, 192)
            results.append((tuple(seq), tuple(markers)))
    except Exception as e:  # noqa: BLE001
        errors.append("%s: %s" % (type(e).__name__, e))


threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check_true("concurrency/four threads finish without an error", not errors,
           errors[0] if errors else "")
check("concurrency/every thread built the same sequence", len(set(results)), 1)
check("concurrency/all calls produced output", len(results), 4 * 40)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
