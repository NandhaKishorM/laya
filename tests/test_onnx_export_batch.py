"""The ONNX export must stay batch-dynamic: `ONNXAgent` sends one row per question.

`scripts/export_onnx.py` declared `dynamic_axes` for a batch axis but traced the model with a
single row, and `torch.export` specializes any example dimension of extent 1. The graph it wrote
ran at batch 1 and failed inside ONNX Runtime at batch 2 ("Attempting to broadcast an axis by a
dimension other than 1"), so every `predict`/`system_one` call with two questions, and
`predict_batch`/`predict_long` at any size, was unservable -- while the export printed
"Successfully exported" and the graph's declared input shape still read `batch_size`. That last
part is why this suite runs the graph instead of reading its shapes.

No checkpoint and no network: the 322M-parameter encoder is replaced by a tiny embedding, and
the real `DecisionModel` head (`_DynamicMultiheadAttention`, the marker gather, the typed
scorer, the act head) is what the exported graph is made of -- the same head that bakes the
traced batch size in.

Run: python tests/test_onnx_export_batch.py
"""
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from laya.common import DecisionModel  # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("export_onnx",
                                               os.path.join(_here, os.pardir, "scripts", "export_onnx.py"))
export_onnx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_onnx)

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


def attempt(name, thunk):
    """Run `thunk`, recording any exception as a failure and returning None.

    Every failure mode has to arrive as a line in the `N passed, M failed` summary, which is how
    this repo's script suites are read. Some exporter breakages are hard rather than numerical --
    drop `dynamic_axes` and ONNX Runtime refuses the graph outright -- and an uncaught error there
    would end the script on a traceback with no summary at all.
    """
    try:
        return thunk()
    except (Exception, SystemExit) as error:
        FAIL.append("%s raised %s: %s" % (name, type(error).__name__,
                                          str(error).replace("\n", " ")[:200]))
        return None


# ---------------------------------------------------------------- a tiny stand-in encoder
HIDDEN, VOCAB = 32, 100


class _TinyEncoderConfig:
    hidden_size = HIDDEN


class _TinyEncoder(nn.Module):
    """Shape-compatible stand-in for the transformers encoder `DecisionModel` wraps.

    It consumes `attention_mask` so padding reaches the hidden states, as the real encoder does;
    what matters for this suite is that `DecisionModel.forward` and its head run unchanged.
    """

    def __init__(self):
        super().__init__()
        self.config = _TinyEncoderConfig()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.proj(torch.tanh(self.embed(input_ids)))
        if attention_mask is not None:
            hidden = hidden * attention_mask[:, :, None].to(hidden.dtype)
        return types.SimpleNamespace(last_hidden_state=hidden)


def _build_model():
    torch.manual_seed(0)
    # head_layers=2 is the shipped default, and the head is where the traced batch got baked in.
    model = DecisionModel(_TinyEncoder(), head_layers=2, n_act=2, dropout=0.0)
    return model.eval()


tmp = os.path.join(_here, "_tmp_onnx_export_batch")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
onnx_path = os.path.join(tmp, "tiny.onnx")

model = _build_model()
check("export/returns the output path",
      attempt("export/export_module", lambda: export_onnx.export_module(model, onnx_path)),
      onnx_path)
check_true("export/writes the file", os.path.exists(onnx_path))

# ---------------------------------------------------------------- the example inputs are >1 row
check("tracing batch is greater than one (a single row bakes batch=1 into the head)",
      export_onnx.TRACE_BATCH > 1, True)
traced = export_onnx.example_inputs()
check("example_inputs/one row per batch entry", [tuple(t.shape)[0] for t in traced],
      [export_onnx.TRACE_BATCH] * 5)
check_true("example_inputs/carries padding, so the trace sees src_key_padding_mask",
           int(traced[1].min()) == 0, traced[1])
check_true("example_inputs/carries a masked-off marker, so the trace sees the masked_fill",
           bool((~traced[3]).any()), traced[3])
# `TRACE_BATCH` is read at call time, not bound into the signature default -- otherwise the one
# experiment a reader of this file will run (set it to 1 and watch the suite fail) silently does
# nothing and looks like evidence the bug is not real.
export_onnx.TRACE_BATCH = 1
check("example_inputs/TRACE_BATCH is read at call time, so it can be overridden",
      [tuple(t.shape)[0] for t in export_onnx.example_inputs()], [1] * 5)
export_onnx.TRACE_BATCH = 2
# Both of these are latent traps for whoever adds the next shape case rather than bugs today:
# an all-padding row makes torch and ONNX Runtime legitimately disagree (measured 2.76e-01 on
# `act_logits` and 1.12e-01 on `logits`, against 2.38e-07 / 1.19e-07 clamped),
# and a marker position outside the sequence makes ONNX Runtime raise from `GatherElements`.
wide = export_onnx.example_inputs(batch=32, seq_len=16, num_markers=2)
check_true("example_inputs/no row is entirely padding, even at batch > seq_len",
           int(wide[1].sum(dim=1).min()) >= 1, wide[1].sum(dim=1))
try:
    export_onnx.example_inputs(batch=2, seq_len=8, num_markers=8)
    FAIL.append("example_inputs/rejects marker positions outside the sequence: it was accepted")
except ValueError as error:
    check_true("example_inputs/rejects marker positions outside the sequence",
               "seq_len" in str(error), str(error)[:200])

# ---------------------------------------------------------------- the graph declares a batch axis
loaded = attempt("graph/loads as ONNX", lambda: onnx.load(onnx_path))


def _dims(value_info):
    return [d.dim_param if d.HasField("dim_param") else d.dim_value
            for d in value_info.type.tensor_type.shape.dim]


if loaded is not None:
    graph = loaded.graph
    declared = {v.name: _dims(v) for v in list(graph.input) + list(graph.output)}
    check("graph/I/O names are the ones ONNXAgent feeds and reads",
          ([i.name for i in graph.input], [o.name for o in graph.output]),
          (export_onnx.INPUT_NAMES, export_onnx.OUTPUT_NAMES))
    check_true("graph/every input and output declares a symbolic leading axis",
               all(isinstance(d[0], str) for d in declared.values()), declared)

# ---------------------------------------------------------------- it actually runs at batch > 1
# A declared axis is not a working one: the batch-1-only graph this suite guards against
# declared "batch_size" too. Running it is the only check that tells them apart.
session = attempt("graph/opens in ONNX Runtime",
                  lambda: ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]))
worst = 0.0
_biggest_output = 0.0   # the graph must actually produce something
for batch, seq_len, num_markers in [(1, 16, 2), (2, 16, 2), (2, 24, 3), (3, 40, 5), (5, 64, 7)]:
    label = "batch=%d seq_len=%d num_markers=%d" % (batch, seq_len, num_markers)
    inputs = export_onnx.example_inputs(batch=batch, seq_len=seq_len, num_markers=num_markers)
    feed = {name: tensor.numpy() for name, tensor in zip(export_onnx.INPUT_NAMES, inputs)}
    try:
        got = session.run(export_onnx.OUTPUT_NAMES, feed)
    except Exception as error:
        FAIL.append("run/%s: %s: %s" % (label, type(error).__name__, str(error).replace("\n", " ")[:200]))
        continue
    PASS.append("run/%s" % label)
    with torch.no_grad():
        want = model(*inputs)
    check(("run/%s: logits shape" % label), tuple(got[0].shape), (batch, num_markers))
    check(("run/%s: act_logits shape" % label), tuple(got[1].shape), (batch, 2))
    diff = max(float(np.max(np.abs(a - w.numpy()))) for a, w in zip(got, want))
    worst = max(worst, diff)
    _biggest_output = max(_biggest_output, max(float(np.max(np.abs(a))) for a in got))
    check_true("parity/%s matches PyTorch within 1e-4" % label, diff <= 1e-4, "max abs diff %.2e" % diff)

# Non-vacuity belongs on the outputs, not on the disagreement: if a future ORT or opset happens to
# reproduce this tiny fp32 graph bit-exactly, `worst == 0.0` is the best possible result, not a
# failure. What must not be true is that the graph returns nothing.
check_true("parity/the comparison is real, not an all-zero graph", _biggest_output > 0.0,
           "max abs diff across every shape was exactly 0.0")

# ---------------------------------------------------------------- verify_batch_dynamic accepts it
try:
    export_onnx.verify_batch_dynamic(model, onnx_path)
    PASS.append("verify/accepts a batch-dynamic export")
except SystemExit as error:
    FAIL.append("verify/accepts a batch-dynamic export: raised %r" % (str(error),))

# ---------------------------------------------------------------- ...and rejects a batch-1 graph
# Pin the leading axis of the good graph to 1 and nothing else, which is what an export with a
# baked batch produces. Without this case the guard above could be vacuous.
pinned_path = os.path.join(tmp, "tiny.batch1.onnx")
pinned = attempt("graph/reloads for the pinned-axis check", lambda: onnx.load(onnx_path))
for value_info in list(pinned.graph.input) + list(pinned.graph.output):
    leading = value_info.type.tensor_type.shape.dim[0]
    leading.ClearField("dim_param")
    leading.dim_value = 1
onnx.save(pinned, pinned_path)
try:
    export_onnx.verify_batch_dynamic(model, pinned_path)
    FAIL.append("verify/rejects a batch-1-only graph: it was accepted, so the guard is vacuous")
except SystemExit as error:
    check_true("verify/rejects a batch-1-only graph", "batch 2" in str(error), str(error)[:200])

# ------------------------------------------------- the verifier compares NUMBERS, not just shapes
# Rejecting a batch-1-only graph only proves it notices a graph that fails to *run*. The reason this
# verifier compares probabilities at all is the case where the graph runs fine at every batch and
# quietly returns different values -- a decomposition or quantization change, say. Exporting one
# model and verifying a *perturbed* copy against that graph is that case, exactly.
import copy  # noqa: E402

_perturbed = copy.deepcopy(model)
with torch.no_grad():
    # Non-uniform, and on a weight rather than a final bias. A uniform shift of the last bias moves
    # no probability at all -- softmax is shift-invariant, which is the blind spot
    # `verify_batch_dynamic`'s own docstring records -- so perturbing that way would make this check
    # pass for the wrong reason.
    _w = [q for q in _perturbed.parameters() if q.ndim >= 2][-1]
    _w.add_(torch.linspace(0.0, 5.0, _w.numel()).reshape(_w.shape))
try:
    export_onnx.verify_batch_dynamic(_perturbed, onnx_path)
    FAIL.append("verify/rejects a graph whose numbers drifted: it was accepted, so the guard is vacuous")
except SystemExit as error:
    check_true("verify/rejects a graph whose numbers drifted",
               "differ" in str(error) or "probabilit" in str(error), str(error)[:160])

# ------------------------------------------------------- export_to_onnx verifies by default
# `verify_batch_dynamic` is the safety net for the day `dynamic_axes` is removed and this export
# goes silently static. A net that defaults to off is not a net, and flipping that default is a
# one-character change no other check here would notice: the suite calls `verify_batch_dynamic`
# directly, and asserting `--no-verify` is *documented* says nothing about what happens without it.
_sig = inspect.signature(export_onnx.export_to_onnx)
check("export_to_onnx/verify defaults to True", _sig.parameters["verify"].default, True)

_calls = []
_real_agent = getattr(export_onnx, "Agent", None)
_real_export = export_onnx.export_module
_real_verify = export_onnx.verify_batch_dynamic


class _StubAgent:
    def __init__(self, *a, **k):
        self.model = model


export_onnx.Agent = _StubAgent
export_onnx.export_module = lambda *a, **k: _calls.append("export")
export_onnx.verify_batch_dynamic = lambda *a, **k: _calls.append("verify")
try:
    export_onnx.export_to_onnx("stub", os.path.join(tmp, "unused.onnx"))
    check("export_to_onnx/verifies when not asked otherwise", _calls, ["export", "verify"])
    _calls.clear()
    export_onnx.export_to_onnx("stub", os.path.join(tmp, "unused.onnx"), verify=False)
    check("export_to_onnx/--no-verify skips it", _calls, ["export"])
finally:
    export_onnx.export_module = _real_export
    export_onnx.verify_batch_dynamic = _real_verify
    if _real_agent is not None:
        export_onnx.Agent = _real_agent

# ------------------------------------------- marker width is baked in, and the message says so
# `DecisionModel.forward` branches on `p.size(-1) >= 2` in Python, so the traced marker width bakes
# that branch into the graph. A one-criterion `score` question produces exactly one marker and is
# accepted by `Agent._to_internal`, so this is reachable -- and the graph raises an ONNX Runtime
# `TopK` error there. That predates this PR; what this check pins is that the export no longer
# *claims* marker width is dynamic, so nobody reads "Verification passed" as covering it.
_w1 = export_onnx.example_inputs(batch=2, seq_len=16, num_markers=1)
try:
    session.run(export_onnx.OUTPUT_NAMES,
                {n: t.numpy() for n, t in zip(export_onnx.INPUT_NAMES, _w1)})
    check_true("limitation/num_markers=1 is known to fail", False,
               "it succeeded -- the limitation is gone, so update the message and this check")
except Exception as error:
    check_true("limitation/num_markers=1 fails on a graph traced at width 2",
               "TopK" in str(error) or "topk" in str(error), str(error).replace("\n", " ")[:120])
_verify_doc = export_onnx.verify_batch_dynamic.__doc__ or ""
check_true("limitation/the verifier documents that marker width is not dynamic",
           "not dynamic" in _verify_doc.lower(), _verify_doc[:80])

# --------------------------------------------- onnxruntime missing is a message, not a traceback
# Verifying by default made onnxruntime a requirement of a command that never needed it: the export
# itself does not touch ORT. A bare ModuleNotFoundError after "Successfully exported" reads as a
# failed export, which it was not.
_hidden = {"onnxruntime": None}
_saved = {k: sys.modules.get(k) for k in _hidden}
sys.modules.update(_hidden)
try:
    export_onnx.verify_batch_dynamic(model, onnx_path)
    check_true("cli/missing onnxruntime is explained", False, "no SystemExit was raised")
except SystemExit as error:
    msg = str(error)
    check_true("cli/missing onnxruntime is explained, not a traceback",
               "onnxruntime" in msg and "--no-verify" in msg and "laya[onnx]" in msg, msg[:160])
except ModuleNotFoundError as error:
    check_true("cli/missing onnxruntime is explained, not a traceback", False,
               "raised a bare %s" % type(error).__name__)
finally:
    for k, v in _saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v

# ------------------------------------------------------ the quantized graph is verified too
check_true("quantize/export_to_onnx hands the model back so the INT8 graph can be checked",
           "return agent.model" in inspect.getsource(export_onnx.export_to_onnx),
           inspect.getsource(export_onnx.export_to_onnx)[-120:])
_main = inspect.getsource(export_onnx).split('if __name__ == "__main__":')[-1]
check_true("quantize/the INT8 graph is verified when --quantize is used",
           "verify_batch_dynamic(model, int8_path" in _main, _main[-300:])
check_true("quantize/INT8 uses its own looser tolerance",
           "INT8_ATOL" in _main and export_onnx.INT8_ATOL > 1e-3,
           "INT8_ATOL=%r" % (export_onnx.INT8_ATOL,))

# ------------------------------------------------------------------ the docs say it too
# The export gained a verification pass, a hard onnxruntime requirement and a --no-verify flag.
# docs/evals.md is the page that documents the invocation, and nothing else enforces that it keeps up.
_evals = os.path.join(_here, os.pardir, "docs", "evals.md")
with open(_evals, encoding="utf-8") as fh:
    _evals_text = fh.read()
for _needle in ("--no-verify", "laya[onnx]", "verifies itself"):
    check_true("docs/evals.md mentions %r" % _needle, _needle in _evals_text,
               "add it next to the export invocation")

# ---------------------------------------------------------------- CLI surface
help_text = subprocess.run(
    [sys.executable, os.path.join(_here, os.pardir, "scripts", "export_onnx.py"), "--help"],
    capture_output=True, text=True, timeout=300,
).stdout
check_true("cli/--no-verify is documented in --help", "--no-verify" in help_text,
           [ln for ln in help_text.splitlines() if "verify" in ln])

# ---------------------------------------------------------------- report
shutil.rmtree(tmp, ignore_errors=True)
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
