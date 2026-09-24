"""The load-time and budget errors a user actually hits.

Every branch here was reachable but never executed by any suite in CI, measured with
`sys.settrace` over all 15 of them: `laya/agent.py` lines 126, 146, 154, 165 and 350 had
zero hits. They are the messages a user sees when a checkpoint is wrong or a question is
too large, so a regression in one is a regression in the only diagnostic they get.

No network: the checkpoint is a tiny local one built here, the same shape
`tests/test_download.py` uses.

Run: python tests/test_load_errors.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya import load  # noqa: E402
from laya.common import DecisionModel  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


TMP = tempfile.TemporaryDirectory()
REPO = Path(TMP.name) / "repo"


def build_checkpoint(root, max_len=64, head_max_len=32, vocab=("hello",)):
    """A loadable Laya checkpoint, small enough to build in-process."""
    words = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for i, word in enumerate(vocab):
        words[word] = 5 + i
    root.mkdir(parents=True, exist_ok=True)
    config = BertConfig(vocab_size=len(words), hidden_size=64, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=128)
    config.save_pretrained(root / "encoder")
    # `Whitespace` is required: without a pre-tokenizer, `WordLevel` sees the whole string
    # as one word and every label collapses to a single `[UNK]`, so the option block never
    # grows and the budget guard below cannot be reached.
    word_level = Tokenizer(WordLevel(words, unk_token="[UNK]"))
    word_level.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=word_level,
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )
    tokenizer.save_pretrained(root / "tokenizer")
    model = DecisionModel(BertModel(config), head_layers=0)
    save_file(model.state_dict(), root / "model.safetensors")
    (root / "rl_agent_config.json").write_text(json.dumps({
        "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
        "max_len": max_len, "head_max_len": head_max_len,
    }), encoding="utf-8")


build_checkpoint(REPO)


def load_error(path, **kw):
    """Load and return the exception, or None if it unexpectedly succeeded."""
    try:
        load(str(path), device="cpu", **kw)
    except Exception as exc:  # noqa: BLE001 -- the message is the thing under test
        return exc
    return None


# ------------------------------------------------------- a good checkpoint still loads
good = load(str(REPO), device="cpu")
check_true("baseline/local checkpoint loads", good is not None)
check("baseline/config round-trips", good.cfg["max_len"], 64)
del good


# ----------------------------------------------------------------- 1. missing directory
# `model_id_or_path` starting with ./ or a drive letter is treated as a local path and
# never handed to the Hub, so the user gets this message instead of a network error.
missing_dir = Path(TMP.name) / "not-a-checkpoint"
err = load_error(missing_dir)
check_true("missing path/raises FileNotFoundError", isinstance(err, FileNotFoundError), repr(err))
# The message carries the path through `!r`, so on Windows it is the backslash-escaped
# form; compare against that rather than the plain string.
check_true("missing path/names the path", repr(str(missing_dir)) in str(err), str(err))
check_true("missing path/says what to check",
           "does not exist" in str(err) or "training saved" in str(err), str(err))


# -------------------------------------------------------------- 2. missing subfolder
# Reachable when a repo exists but the requested sibling checkpoint is not in it.
err = load_error(REPO, subfolder="multilingual")
check_true("missing subfolder/raises FileNotFoundError", isinstance(err, FileNotFoundError), repr(err))
check_true("missing subfolder/names the subfolder", "multilingual" in str(err), str(err))

sub = Path(TMP.name) / "sub"
build_checkpoint(sub / "multilingual")
check_true("present subfolder/loads", load_error(sub, subfolder="multilingual") is None)


# --------------------------------------------------- 3. missing rl_agent_config.json
# This is the file that makes a directory a Laya checkpoint rather than a bare encoder,
# so the message has to point at the training run that would have written it.
no_cfg = Path(TMP.name) / "no-config"
shutil.copytree(REPO, no_cfg)
os.remove(no_cfg / "rl_agent_config.json")
err = load_error(no_cfg)
check_true("no config/raises FileNotFoundError", isinstance(err, FileNotFoundError), repr(err))
check_true("no config/names the missing file", "rl_agent_config.json" in str(err), str(err))
check_true("no config/says where it comes from",
           "ships with the weights" in str(err) or "training run" in str(err), str(err))


# --------------------------------------------------------- 4. missing model.safetensors
# The config is present and valid here, so this is reached only after that check passes.
no_weights = Path(TMP.name) / "no-weights"
shutil.copytree(REPO, no_weights)
os.remove(no_weights / "model.safetensors")
err = load_error(no_weights)
check_true("no weights/raises FileNotFoundError", isinstance(err, FileNotFoundError), repr(err))
check_true("no weights/names the missing file", "model.safetensors" in str(err), str(err))


# ------------------------------------------- 5. a question whose options do not fit
# `build_sequence` drops markers past `max_len`. Without this guard the model gets a
# selected index outside its own option count, which surfaces as an index error deep in
# the head rather than as a statement about the question.
#
# Measured against the shipped budget (`max_len=512`, `head_max_len=192`) with the real
# tokenizer: 100 options give 413 tokens and 100 markers, 140 give 512 tokens and 126
# markers. So the boundary sits between 100 and 140. Nothing documents that threshold,
# which is why this pins the behaviour rather than the number.
wide = Path(TMP.name) / "wide"
WORDS = ("department", "handling", "billing", "enquiries")
build_checkpoint(wide, max_len=512, head_max_len=192,
                 vocab=WORDS + tuple(str(i) for i in range(1, 201)))
wide_agent = load(str(wide), device="cpu")
many = {("department %d handling billing enquiries" % i): None for i in range(1, 141)}
try:
    wide_agent.system_one("hello",
                          {"q": {"type": "choice", "instructions": "Which department?",
                                 "criteria": many}})
    _outcome = None
except Exception as exc:  # noqa: BLE001
    _outcome = exc
check_true("options over budget/raises ValueError", isinstance(_outcome, ValueError), repr(_outcome))
check_true("options over budget/names the question", "'q'" in str(_outcome), str(_outcome))
check_true("options over budget/reports the budget",
           "head_max_len" in str(_outcome), str(_outcome))
# ...and a question that does fit still answers, so the guard is not refusing everything.
fits = {"q": {"type": "choice", "instructions": "Pick one",
              "criteria": {"department": None, "billing": None}}}
try:
    wide_agent.system_one("hello", fits)
    _ok = True
except Exception:  # noqa: BLE001
    _ok = False
check_true("options within budget/still answers", _ok)
del wide_agent


TMP.cleanup()

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
