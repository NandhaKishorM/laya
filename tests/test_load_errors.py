"""The load-time and budget errors a user actually hits.

Every branch here was reachable but never executed by any suite in CI, measured with
`sys.settrace` over all 15 of them: `laya/agent.py` lines 126, 146, 154, 165 and 350 had
zero hits. They are the messages a user sees when a checkpoint is wrong or a question is
too large, so a regression in one is a regression in the only diagnostic they get.

No network: the checkpoint is a tiny local one built here, the same shape
`tests/test_download.py` uses.

Section 7 spends that same tiny agent on `examples/39_error_handling.py`: it runs the page and holds
what it promises to what it prints.

Run: python tests/test_load_errors.py
"""
import ast
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

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
from laya.router import Router  # noqa: E402

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

# The message used to name only `head_max_len`, which pointed at the wrong knob in both
# directions. The option markers are placed at absolute positions and `build_sequence` drops the
# ones past `max_len`, so `head_max_len` is how much of the sequence the options were given --
# lowering it shortens the option block and can bring the question back inside `max_len`, while
# raising it overflows further. The message has to name `max_len` and say what was measured, or a
# caller follows it the wrong way. The direction itself is measured against a real checkpoint in
# the description rather than pinned here, since it needs one this suite does not build.
check_true("options over budget/names max_len too",
           "max_len=" in str(_outcome), str(_outcome))
# The count is the markers that SURVIVED, not `len(seq)`: `build_sequence` truncates to `max_len`
# first, so `len(seq)` is always exactly `max_len` at this point and reporting it stated the
# ceiling as though it were the requirement. `only N of its M` is the shape that distinguishes
# them, and `M` is checkable here because the question has 140 options.
check_true("options over budget/reports the markers that survived, not the ceiling",
           "only " in str(_outcome) and "fit in" in str(_outcome), str(_outcome))
check_true("options over budget/names the option count",
           "140 option markers" in str(_outcome), str(_outcome))
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


# ------------------------------------------- 6. a temperature list of the wrong length
# `_decode_answers` indexes `temperature` by question type (`QTYPES`), so a checkpoint that
# ships the wrong number of entries -- say one -- loads cleanly, answers `choice` questions,
# and then raises a bare `IndexError` on the first `score`/`noul` question: a decode-time
# crash whose cause is a single config field. The language-override path already refuses this
# shape ("must be a list of 3 floats"); this pins the same refusal for the checkpoint's own
# list, where there was none.
short_temp = Path(TMP.name) / "short-temperature"
build_checkpoint(short_temp)
_cfg = json.loads((short_temp / "rl_agent_config.json").read_text())
_cfg["temperature"] = [0.9]
(short_temp / "rl_agent_config.json").write_text(json.dumps(_cfg), encoding="utf-8")
err = load_error(short_temp)
check_true("short temperature/raises ValueError", isinstance(err, ValueError), repr(err))
check_true("short temperature/names the field", "temperature" in str(err), str(err))
check_true("short temperature/says the shape", "list of 3" in str(err), str(err))


# ------------------------------------ 7. examples/39 must run, and must promise what it prints
# Two defects on one page, and no job could see either: examples are not executed by the test job,
# and nothing read a page's prose.
#
#   * The file stopped at case 2 in a checkout without a vendored `models/` directory -- which is
#     every checkout, since `models/` is not in the repository. `os.path.dirname` was handed
#     `MODELS["english"]`, a `Router` spec, and `examples/_common.py`'s `checkpoint` returns that as
#     the tuple `(repo, subfolder)` when the directory is missing: `TypeError: expected str, bytes
#     or os.PathLike object, not tuple`, with cases 3, 4 and 5 never reached -- under a docstring
#     that says "Every case below is executed".
#   * Banner item 3 promised "a low-level RuntimeError" for a `choice` question with empty criteria,
#     three lines above the output that prints `ValueError`, and its conclusion said both shapes
#     "reach the scorer with zero options and blow up in top-k". They reach neither; the validator
#     measured one section up rejects them by name before the state is encoded.
#
# So the page is run here on the tiny checkpoint built above, and its own output is the witness: for
# each numbered case, the exception types the banner lists must be exactly the types that case's own
# section prints.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import _common  # noqa: E402 -- the helpers the example imports

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "39_error_handling.py"
SOURCE = EXAMPLE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
AGENT_SRC = (Path(__file__).resolve().parents[1] / "laya" / "agent.py").read_text(encoding="utf-8")

ERR_TOKEN = re.compile(r"\b[A-Z][A-Za-z]+(?:Error|Exception)\b")
# what a section shows it was given: the line the page prints from `type(e).__name__`, or from its
# own `except <Type>` clause. The banner is excluded by slicing the output per heading.
EVIDENCE = re.compile(r"(?:raised|->)\s+([A-Z][A-Za-z]+(?:Error|Exception))\b")
# A rejected question never reaches the decision head, so a page may not send the reader there.
LIE_WORDS = ("top-k", "reach the scorer", "reaches the scorer", "blow up", "blows up")


def _types(text):
    return sorted(set(ERR_TOKEN.findall(text)))


def _banner_blurb():
    for node in ast.walk(TREE):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "banner":
            for arg in node.args[2:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    return arg.value
    return ""


def _printed_prose():
    """Every string the page prints -- its claims, as opposed to the code that makes them."""
    parts = []
    for node in ast.walk(TREE):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    parts.append(arg.value)
    # A page wrapped across `print` calls makes one sentence: without this, "…window. 120" and
    # "options slip through…" would never read as the count it states.
    return re.sub(r"\s+", " ", " ".join(parts))


def _dotted(node):
    """`os.path.join` -> 'os.path.join'; anything not built from plain attribute reads -> ''."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


BLURB = _banner_blurb()
ITEM = {}
for _line in BLURB.splitlines():
    _m = re.match(r"^\s*(\d)\..*?->\s*(.*)$", _line)
    if _m:
        ITEM[int(_m.group(1))] = _m.group(2)
# Everything the page asserts -- its five banner lines and its printed conclusions. `EVIDENCE` is
# only searched over the per-case sections, so the banner cannot vouch for itself.
CLAIMS = PROSE = _printed_prose() + " " + BLURB

# `Router` specs and checkpoint directories are different types, and only one of them is a path.
_has_local = _common.has_local
_common.has_local = lambda name="english": False
SPEC_WITHOUT_MODELS = _common.checkpoint("english")
_common.has_local = _has_local
check("39/a Router spec with no models/ directory is not a usable path",
      isinstance(SPEC_WITHOUT_MODELS, str) and os.path.isdir(SPEC_WITHOUT_MODELS), False)

# Section 5's checkpoint is the one to answer the page's `load("english")` with: it ships the
# shipped budgets (`max_len=512`, `head_max_len=192`), so the page's four inference cases take the
# same code paths they take against the English checkpoint.
example_agent = load(str(wide), device="cpu")
_real_load = _common.load
_common.load = lambda name=None, *args, **kwargs: example_agent
_ran, _crash, _stdout = False, "", io.StringIO()
try:
    with contextlib.redirect_stdout(_stdout):
        exec(compile(SOURCE, str(EXAMPLE), "exec"),
             {"__name__": "example_39", "__file__": str(EXAMPLE)})
    _ran = True
except BaseException as exc:  # noqa: BLE001 -- running the page is the test
    _crash = "%s: %s" % (type(exc).__name__, exc)
finally:
    _common.load = _real_load

check_true("39/the whole page runs to its last case", _ran, _crash)
OUT = _stdout.getvalue()
_marks = list(re.finditer(r"-- (\d)\. [^\n]*? --", OUT))
SECTION = {}
for _i, _m in enumerate(_marks):
    _stop = _marks[_i + 1].start() if _i + 1 < len(_marks) else len(OUT)
    SECTION[int(_m.group(1))] = OUT[_m.end():_stop]
check("39/its own output shows all five cases ran", sorted(SECTION), [1, 2, 3, 4, 5])
check("39/the banner lists five cases", sorted(ITEM), [1, 2, 3, 4, 5])

for _case in (1, 2, 3, 4, 5):
    check("39/case %d promises the exception it prints" % _case,
          _types(ITEM.get(_case, "")), sorted(set(EVIDENCE.findall(SECTION.get(_case, "")))))
SHOWN = sorted({t for section in SECTION.values() for t in EVIDENCE.findall(section)})
check("39/the page claims no exception its output never shows",
      sorted(set(_types(CLAIMS)) - set(SHOWN)), [])
check("39/no sentence sends a rejected question to the decision head",
      [w for w in LIE_WORDS if w in CLAIMS], [])

# case 3's message comes from one validator, so the page has to name it: "ValueError" alone would
# fit a hundred other raises, and the reader is left pattern-matching the traceback.
check_true("39/case 3 names the validator that rejects it", "_check_question" in CLAIMS, CLAIMS[:120])
check_true("39/naming it is possible: the validator is in the runtime",
           "def _check_question" in AGENT_SRC, "")
check_true("39/what it reports is the question, as the page says",
           "'q'" in SECTION.get(3, ""), SECTION.get(3, "")[:120])

# case 4 quotes the guard as an expression, so the expression has to be the code's -- under any
# name `render_options` is bound to, and any name its count is stored in.
_assign = re.search(r"(\w+) = len\(render_options\((\w+)\)\)", AGENT_SRC)
check_true("39/the guard it quotes is the guard the code compares",
           _assign is not None and "len(markers) != %s:" % _assign.group(1) in AGENT_SRC,
           AGENT_SRC[AGENT_SRC.find("len(markers)"):][:60] if "len(markers)" in AGENT_SRC else "")
check_true("39/quoted rather than paraphrased",
           "len(markers)" in PROSE and "render_options" in PROSE, "")

# the crash class, statically: `MODELS` values can be `(repo, subfolder)` (witnessed above), so no
# filesystem call may take one, and the missing-path case must come from `LOCAL_MODELS`.
_IN_OSPATH = []
_MISSING = ""
for node in ast.walk(TREE):
    if isinstance(node, ast.Call) and _dotted(node.func).startswith("os.path."):
        if any(isinstance(n, ast.Name) and n.id == "MODELS" for n in ast.walk(node)):
            _IN_OSPATH.append("%s line %d" % (_dotted(node.func), node.lineno))
    if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "missing" for t in node.targets):
        _MISSING = ast.unparse(node.value)
check_true("39/the filesystem rule can see a call at all",
           any(_dotted(n.func).startswith("os.path.") for n in ast.walk(TREE) if isinstance(n, ast.Call)),
           "")
check("39/no filesystem call is handed a Router spec", _IN_OSPATH, [])
check_true("39/the missing path is built from the checkpoint directory",
           "LOCAL_MODELS" in _MISSING, _MISSING)

# the counts in its prose are the counts in its code, and each label names the dict it says.
SIZES_BY_NAME = {}
for node in ast.walk(TREE):
    if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
        for call in ast.walk(node.value):
            if (isinstance(call, ast.Call) and getattr(call.func, "id", "") == "range"
                    and call.args and isinstance(call.args[0], ast.Constant)):
                SIZES_BY_NAME[node.targets[0].id] = call.args[0].value
                break
check_true("39/the two option budgets it sets up are named",
           len(SIZES_BY_NAME) >= 2, str(SIZES_BY_NAME))
STATED = {int(x) for x in re.findall(r"(\d+) options", CLAIMS)}
check_true("39/and the prose really states them, so this rule is not reading an empty list",
           bool(STATED), CLAIMS[:80])
check("39/it states no option count it does not build",
      sorted(n for n in STATED if n not in set(SIZES_BY_NAME.values())), [])
_MISPAIRED = []
for node in ast.walk(TREE):
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        _label, _var = node.elts
        if (isinstance(_label, ast.Constant) and isinstance(_label.value, str)
                and re.fullmatch(r"\d+ options", _label.value.strip())
                and isinstance(_var, ast.Name) and _var.id in SIZES_BY_NAME
                and int(_label.value.split()[0]) != SIZES_BY_NAME[_var.id]):
            _MISPAIRED.append("%s with %s=%d" % (_label.value.strip(), _var.id, SIZES_BY_NAME[_var.id]))
check("39/each option-count label names the dict of that size", _MISPAIRED, [])

# Controls: the two sentences this section replaced are kept verbatim, so a rule that matches
# nothing cannot pass for being empty.
HISTORIC_BANNER_3 = "a low-level RuntimeError, not an answer;"
HISTORIC_PRINT_3 = ("   -> both the empty dict and the empty list reach the scorer with zero options "
                    "and blow      up in top-k. No answer dict is returned; treat an empty schema as "
                    "a caller bug.")
check("39/the case rule still rejects the banner line it replaced",
      _types(HISTORIC_BANNER_3) == sorted(set(EVIDENCE.findall(SECTION.get(3, "")))), False)
check("39/the sentence rule still rejects the conclusion it replaced",
      bool([w for w in LIE_WORDS if w in HISTORIC_PRINT_3]), True)
check("39/the exception-name rule still rejects a type the run never showed",
      sorted(set(_types(HISTORIC_BANNER_3)) - set(SHOWN)), ["RuntimeError"])

# ------------------------------------------- 8. a device fallback warns instead of printing
# Asking for a device this machine does not have falls back to CPU with a note. That note used
# to be a bare `print`, so it landed on stdout -- where `laya --json` and `laya --batch --json`
# write their records and where the MCP server speaks JSON-RPC. A warning goes to stderr and a
# caller can filter it; a print breaks the stream and cannot be silenced.
_stdout = io.StringIO()
with warnings.catch_warnings(record=True) as caught, \
        patch("torch.cuda.is_available", return_value=False), \
        redirect_stdout(_stdout):
    warnings.simplefilter("always")
    _fallback = load(str(REPO), device="cuda")
check("device fallback/lands on CPU", _fallback.device.type, "cpu")
check("device fallback/writes nothing to stdout", _stdout.getvalue(), "")
check_true("device fallback/warns instead",
           any(issubclass(w.category, RuntimeWarning) and "CUDA requested" in str(w.message)
               for w in caught),
           str([str(w.message) for w in caught]))
del _fallback


# ------------------------------------------- 9. unusable ordinals and "auto"
# `is_available` only says a GPU exists: on a one-GPU box `cuda:99` sails through it
# and dies later in `.to()` with a bare CUDA error. The ordinal is validated up front
# instead, with the same warn-and-CPU shape as section 8. "auto" (any case or
# surrounding whitespace) means "decide yourself", the way the serve layer already
# treats LAYA_DEVICE. A driver that reports CUDA but cannot answer the capability
# query gets safe fp16 defaults rather than a crash in the amp policy.
with warnings.catch_warnings(record=True) as caught, \
        patch("torch.cuda.is_available", return_value=True), \
        patch("torch.cuda.device_count", return_value=1):
    warnings.simplefilter("always")
    _bad_ordinal = load(str(REPO), device="cuda:99")
check("bad ordinal/lands on CPU", _bad_ordinal.device.type, "cpu")
check_true("bad ordinal/names the ordinal",
           any("cuda:99" in str(w.message) for w in caught),
           str([str(w.message) for w in caught]))
del _bad_ordinal

_auto = load(str(REPO), device="AUTO")
check_true("auto/resolves without crashing",
           _auto.device.type in ("cpu", "cuda", "mps", "xpu"), str(_auto.device))
del _auto

with warnings.catch_warnings(record=True) as caught, \
        patch("torch.cuda.is_available", return_value=True), \
        patch("torch.cuda.device_count", return_value=1), \
        patch("torch.cuda.get_device_capability", side_effect=RuntimeError("no driver")), \
        patch.object(torch.nn.Module, "to", lambda self, *a, **k: self):
    warnings.simplefilter("always")
    _no_probe = load(str(REPO), device="cuda:0")
check("unprobed GPU/stays on the requested device", _no_probe.device.type, "cuda")
check("unprobed GPU/falls back to safe fp16", _no_probe.dtype, torch.float16)
check_true("unprobed GPU/warns about the probe",
           any("capability" in str(w.message) for w in caught),
           str([str(w.message) for w in caught]))
del _no_probe


TMP.cleanup()

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
