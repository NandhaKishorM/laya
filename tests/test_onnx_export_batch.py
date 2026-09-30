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
import tempfile
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from laya.common import QTYPES, DecisionModel  # noqa: E402

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


class _Skip(Exception):
    """Raised to skip a block this platform cannot run."""


#: Tolerance for the TINY model's PASS cases only, not for a real checkpoint. `verify_batch_dynamic`
#: defaults to 1e-3, which the 322M checkpoint passes at 6.14e-05; that default is what bounds
#: quality, it is asserted below (both its value and that it rejects a calibrated error), and an
#: earlier revision of this comment claimed it was "asserted separately" when nothing asserted it.
#: 5e-2 is deliberately loose and is NOT a quality bound: measured, it accepts a scorer weight scaled
#: by 3.0, which the 1e-3 default catches from 1.25 upward. It exists so a cross-machine fp32
#: difference cannot fail the suite, and nothing else. This 32-dim random model has near-tied logits, and softmax
#: amplifies the difference between one machine's fp32 kernels and another's: measured 4.69e-04 here
#: and 1.03e-02 on a GitHub runner, which straddles 1e-3 and made this suite fail by hardware. The
#: signal it has to separate is enormous by comparison -- a perturbed weight moves probabilities by
#: 6.4e-01 -- so 5e-2 sits 5x above the worst drift observed and 13x below the thing being detected.
TINY_ATOL = 5e-2


def _build_model():
    torch.manual_seed(0)
    # head_layers=2 is the shipped default, and the head is where the traced batch got baked in.
    model = DecisionModel(_TinyEncoder(), head_layers=2, n_act=2, dropout=0.0)
    return model.eval()


tmp = os.path.join(_here, "_tmp_onnx_export_batch")

# --- findings from an adversarial review -----------------------------------------------------

# The I/O names are a cross-file contract with laya/onnx_agent.py, which builds its feed and calls
# session.run(["logits", "act_logits"]) from hardcoded strings. Every check here previously built
# its feed from INPUT_NAMES too, so a permutation was self-consistent and survived: swapping the two
# OUTPUT_NAMES exported a graph whose `logits` is the act head, and ONNXAgent then reads the act
# head as the option scores -- silently wrong answers on every request, no error anywhere. Swapping
# input_ids/attention_mask (same dtype, same shape, so ORT accepts it) gave a max deviation of 5.27
# from PyTorch. Asserted against literals, and against the strings onnx_agent.py actually uses.
check("names/inputs are the literal contract", export_onnx.INPUT_NAMES,
      ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"])
check("names/outputs are the literal contract", export_onnx.OUTPUT_NAMES,
      ["logits", "act_logits"])

_agent_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "laya", "onnx_agent.py")).read()
check_true("names/onnx_agent runs the same output names in the same order",
           'session.run(["logits", "act_logits"]' in _agent_src, "")
for _n in export_onnx.INPUT_NAMES:
    check_true("names/onnx_agent feeds %s" % _n, '"%s":' % _n in _agent_src, _n)

# And the shapes those names carry, read BY NAME rather than by position, so a permutation is
# visible: logits is (batch, num_markers) and act_logits is (batch, 2).
_tmp2 = tempfile.mkdtemp()
try:
    _p2 = os.path.join(_tmp2, "names.onnx")
    _m2 = _build_model()
    export_onnx.export_module(_m2, _p2)
    _sess2 = ort.InferenceSession(_p2, providers=["CPUExecutionProvider"])
    _feed2 = {k: v for k, v in zip(export_onnx.INPUT_NAMES,
                                   [x.numpy() for x in export_onnx.example_inputs(batch=3, seq_len=16,
                                                                                  num_markers=4)])}
    _got = dict(zip(["logits", "act_logits"], _sess2.run(["logits", "act_logits"], _feed2)))
    check("names/logits is (batch, num_markers)", _got["logits"].shape, (3, 4))
    check("names/act_logits is (batch, 2)", _got["act_logits"].shape, (3, 2))
    # Released explicitly: module-level ORT sessions that outlive the block were measured causing
    # `libc++abi: recursive_mutex lock failed` at interpreter shutdown -- exit 134 AFTER the suite
    # printed "0 failed", i.e. a red CI job with a green summary.
    del _sess2, _got
finally:
    shutil.rmtree(_tmp2, ignore_errors=True)


# The --quantize path is what docs/evals.md tells people to deploy, and nothing exercised it on a
# real torch export: tests/test_onnx_quantize.py builds a one-node MatMul graph with no intermediate
# value_info at all, so it cannot reach the `del model.graph.value_info[:]` line, and this suite
# exported a real graph but never quantized it. Deleting that one line leaves both suites green and
# makes `python scripts/export_onnx.py --quantize` crash outright with
# `InferenceError: Inferred shape and existing shape differ in dimension 0`.
# Skipped on Windows: quantizing a graph this size dies there with an illegal instruction inside
# onnxruntime (exit 132), which no `except` can catch. `tests/test_onnx_quantize.py` passes on
# Windows because its graph is a single MatMul node. The `onnx export (quantization, weight-free)`
# job on Linux is where this path is gated, and that is the job the documented `--quantize`
# invocation has to keep green.
_tmp3 = None if sys.platform.startswith("win") else tempfile.mkdtemp()
if _tmp3 is None:
    check_true("quantize/skipped on Windows (covered by the Linux quantization job)", True, "")
try:
    if _tmp3 is None:
        raise _Skip()
    _p3 = os.path.join(_tmp3, "q.onnx")
    _m3 = _build_model()
    export_onnx.export_module(_m3, _p3)
    _int8 = export_onnx.quantize_model(_p3, export_onnx.int8_output_path(_p3))
    check_true("quantize/a real torch export quantizes without a shape-inference error",
               os.path.exists(_int8), _int8)
    # and the quantized graph is still batch-dynamic and still within tolerance of PyTorch
    export_onnx.verify_batch_dynamic(_m3, _int8, atol=export_onnx.INT8_ATOL)
    check_true("quantize/the INT8 graph verifies at batch 1, 2 and 3", True, "")
    _s3 = ort.InferenceSession(_int8, providers=["CPUExecutionProvider"])
    check("quantize/the INT8 graph keeps a symbolic batch axis",
          _s3.get_inputs()[0].shape[0], "batch_size")
except _Skip:
    pass
finally:
    shutil.rmtree(_tmp3, ignore_errors=True) if _tmp3 else None

# Every way of turning the new verification OFF survived: the CLI was pinned only by grepping
# __main__ for substrings, and the stubs recorded THAT a function was called, never with what. So
# `batches=(1,)` on either call, `--no-verify` flipped to store_false, `verify=False`, and
# `atol=1e9` on the INT8 check all shipped a success message while checking nothing. The real
# __main__ now runs against a stubbed Agent and the arguments it passes are asserted.
_seen = {}
_tmp4 = None if sys.platform.startswith("win") else tempfile.mkdtemp()
if _tmp4 is None:
    check_true("cli/skipped on Windows (it exports and quantizes; see above)", True, "")
try:
    if _tmp4 is None:
        raise _Skip()
    _p4 = os.path.join(_tmp4, "cli.onnx")

    def _fake_verify(model, path, atol=None, batches=(1, 2, 3)):
        _seen.setdefault("calls", []).append({"path": path, "atol": atol, "batches": tuple(batches)})

    class _FakeAgent:
        def __init__(self, *a, **kw):
            self.model = _build_model()

    _real_agent = export_onnx.Agent
    _real_verify = export_onnx.verify_batch_dynamic
    export_onnx.Agent = _FakeAgent
    export_onnx.verify_batch_dynamic = _fake_verify
    try:
        export_onnx.main(["--model", "x", "--output", _p4, "--quantize"])
    finally:
        export_onnx.Agent = _real_agent
        export_onnx.verify_batch_dynamic = _real_verify

    _calls = _seen.get("calls", [])
    check("cli/both the fp32 and the INT8 graph are verified", len(_calls), 2)
    check_true("cli/the fp32 check sweeps more than batch 1",
               _calls and set(_calls[0]["batches"]) >= {1, 2, 3}, _calls[:1])
    check_true("cli/the INT8 check sweeps more than batch 1",
               len(_calls) > 1 and set(_calls[1]["batches"]) >= {1, 2, 3}, _calls[1:])
    check("cli/the INT8 check uses the quantization tolerance",
          _calls[1]["atol"] if len(_calls) > 1 else None, export_onnx.INT8_ATOL)
    check_true("cli/the INT8 graph is written beside the fp32 one, not over it",
               len(_calls) > 1 and _calls[1]["path"] != _calls[0]["path"], _calls)
except _Skip:
    pass
finally:
    shutil.rmtree(_tmp4, ignore_errors=True) if _tmp4 else None


shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
onnx_path = os.path.join(tmp, "tiny.onnx")

model = _build_model()
check("export/returns the output path",
      attempt("export/export_module", lambda: export_onnx.export_module(model, onnx_path)),
      onnx_path)
check_true("export/writes the file", os.path.exists(onnx_path))

# ------------------------------------------------- the verifier varies the axes it says it varies
# `verify_batch_dynamic` claims to check "batch 1, 2 and 3 with 2-4 markers". Sizing every batch
# with the same `example_inputs(batch=batch)` left the suite green while the sequence and marker
# axes were never exercised at more than one width -- so the graph could be marker-static and the
# verification would still pass. Spy on the shapes it actually asks for.
_asked = []
_real_example_inputs = export_onnx.example_inputs


def _spy_example_inputs(batch=None, seq_len=16, num_markers=2):
    _asked.append((batch, seq_len, num_markers))
    return _real_example_inputs(batch=batch, seq_len=seq_len, num_markers=num_markers)


export_onnx.example_inputs = _spy_example_inputs
try:
    export_onnx.verify_batch_dynamic(_build_model(), onnx_path, batches=(1, 2, 3))
finally:
    export_onnx.example_inputs = _real_example_inputs

check("verify/asks for one input set per batch", [a[0] for a in _asked], [1, 2, 3])
# One width per batch, not the same shapes three times: sizing every batch identically left the
# suite green while the sequence and marker axes were never exercised at more than one width, so a
# marker-static graph would have verified clean.
check("verify/varies the sequence width across batches",
      len({a[1] for a in _asked}), len(_asked))
check("verify/varies the marker count across batches",
      len({a[2] for a in _asked}), len(_asked))
check_true("verify/and every marker count it asks for fits its sequence",
           all(a[2] < a[1] for a in _asked), _asked)

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
# Every question type the model branches on must appear in the traced batch. Flattening qtype to a
# single value left the suite green while tracing only one branch of `DecisionModel.forward`.
check_true("example_inputs/varies qtype across the batch, so more than one branch is traced",
           len(set(traced[4].tolist())) == min(export_onnx.TRACE_BATCH, len(QTYPES)),
           traced[4])
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
    # The axes the graph must declare, written out LITERALLY rather than read from
    # `export_onnx.DYNAMIC_AXES`. Deriving the expectation from the same constant under test is
    # what made the first version of this check vacuous: shrinking `DYNAMIC_AXES["logits"]` to the
    # batch axis shrank the expectation with it and 74 checks stayed green, so the graph could tell
    # a consumer the option count is fixed at the width it was traced with.
    _WANT_AXES = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }
    check("graph/the declared axes are the ones this suite requires, name by name",
          export_onnx.DYNAMIC_AXES, _WANT_AXES)
    # One `Dim` per name, SHARED across the inputs that use it. `torch.export.Dim("batch_size")`
    # called twice returns two unequal objects, so building one per occurrence would declare five
    # independent batch symbols -- "each input may have its own batch size" instead of "they must
    # agree". The exported graph names them identically either way and runs either way, so no
    # behavioural check here can tell them apart; this pins the specification directly, which is
    # the only place the difference exists.
    _shapes = export_onnx._dynamic_shapes()
    check("graph/one dynamic shape entry per input", len(_shapes), len(export_onnx.INPUT_NAMES))
    _by_label = {}
    for _name, _entry in zip(export_onnx.INPUT_NAMES, _shapes):
        for _axis, _dim in _entry.items():
            _by_label.setdefault(_WANT_AXES[_name][_axis], []).append(_dim)
    check_true("graph/inputs sharing an axis name share one Dim object",
               all(all(d is group[0] for d in group) for group in _by_label.values()),
               {k: len({id(d) for d in v}) for k, v in _by_label.items()})
    check("graph/and every declared axis name is represented",
          sorted(_by_label), sorted({lbl for n in export_onnx.INPUT_NAMES
                                     for lbl in _WANT_AXES[n].values()}))
    _wrong = []
    for _name, _axes in _WANT_AXES.items():
        _got = declared.get(_name, [])
        for _axis, _label in _axes.items():
            if _axis >= len(_got) or _got[_axis] != _label:
                _wrong.append("%s[%d]=%r want %r"
                              % (_name, _axis, _got[_axis] if _axis < len(_got) else "MISSING",
                                 _label))
    check("graph/every axis is symbolic in the graph under the name it was declared with",
          _wrong, [])

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
    export_onnx.verify_batch_dynamic(model, onnx_path, atol=TINY_ATOL)
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
    export_onnx.verify_batch_dynamic(_perturbed, onnx_path, atol=TINY_ATOL)
    FAIL.append("verify/rejects a graph whose numbers drifted: it was accepted, so the guard is vacuous")
except SystemExit as error:
    check_true("verify/rejects a graph whose numbers drifted",
               "differ" in str(error) or "probabilit" in str(error), str(error)[:160])

# The last 2-D parameter lives in the act head, so the check above only moves `act_logits`.
# `logits` IS the decision, and skipping it in the comparison loop left the suite green -- a
# regression that corrupts the option scores and leaves the act head intact would have passed both
# verification and CI. Perturb the scorer side too, as its own case.
# BOTH outputs must be compared, asserted on what the verifier reports rather than through a
# perturbation. `logits` IS the decision, and dropping it from the comparison loop left the suite
# green: every weight in this model reaches `act_logits` as well (the act head consumes the scorer's
# output), so no perturbation isolates `logits`, and whichever output the loop still compares
# catches the drift. What a mutation cannot fake is the verifier naming both.
import io as _io                                                             # noqa: E402
import contextlib as _contextlib                                             # noqa: E402

_cap = _io.StringIO()
with _contextlib.redirect_stdout(_cap):
    export_onnx.verify_batch_dynamic(model, onnx_path, atol=TINY_ATOL)
_reported = _cap.getvalue()
for _name in export_onnx.OUTPUT_NAMES:
    check_true("verify/compares %s against PyTorch" % _name,
               _reported.count(_name) >= 3, _reported[:200])
check("verify/compares every output at every batch it sweeps",
      len([ln for ln in _reported.splitlines() if "max abs prob diff" in ln]),
      3 * len(export_onnx.OUTPUT_NAMES))

# The tolerance that SHIPS is the default, and it was pinned by nothing: `atol: float = 1e-3` ->
# `1e9` left all three suites green, while `export_to_onnx` calls `verify_batch_dynamic` with no atol
# at all. Pin the value, and prove it rejects an error TINY_ATOL is blind to -- a calibrated 1.25x
# scaling of one scorer weight, not the linspace(0, 5) sledgehammer.

check("verify/the shipping tolerance is 1e-3",
      inspect.signature(export_onnx.verify_batch_dynamic).parameters["atol"].default, 1e-3)

_calibrated = copy.deepcopy(model)
with torch.no_grad():
    dict(_calibrated.named_parameters())["scorer.3.weight"].mul_(1.25)
try:
    export_onnx.verify_batch_dynamic(_calibrated, onnx_path)          # default atol, as it ships
    FAIL.append("verify/the shipping tolerance rejects a 1.25x weight error: it was accepted")
except SystemExit as error:
    check_true("verify/the shipping tolerance rejects a 1.25x weight error",
               "differ" in str(error) or "probabilit" in str(error), str(error)[:160])
# ... and that TINY_ATOL is not doing that work, so nobody mistakes it for a quality bound.
try:
    export_onnx.verify_batch_dynamic(_calibrated, onnx_path, atol=TINY_ATOL)
    check_true("verify/TINY_ATOL is a pass-case tolerance, not a quality bound", True, "")
except SystemExit:
    FAIL.append("verify/TINY_ATOL is a pass-case tolerance: it caught 1.25x, so the comment is wrong")

# INT8_ATOL was bounded only from below (`> 1e-3`), so 1e9 was green. Bound it from above too: the
# quantization drift this expects is 9.95e-04 over verify's own sweep, and per-tensor quantization --
# the variant the code comment says flips 3 of 20 real decisions -- moves it to 0.29.
# Both sides, and tightly: 1e-3 < x < 0.1 let 1.1e-3 through, which is below the 9.95e-04 drift
# this tolerance has to ACCEPT over verify's own sweep, so a real quantized graph would start
# failing. The upper end has to stay well under the 0.29 that per-tensor quantization produces --
# the variant the code comment says flips 3 of 20 real decisions.
check_true("quantize/INT8_ATOL accepts expected drift and rejects per-tensor drift",
           5e-3 < export_onnx.INT8_ATOL < 5e-2, export_onnx.INT8_ATOL)

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
    export_onnx.verify_batch_dynamic(model, onnx_path, atol=TINY_ATOL)
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
# Read `main`'s source, not the `__main__` guard: the CLI body moved into a function so the
# arguments it passes could be asserted for real, above. These substring checks stay as a cheap
# belt-and-braces, but they are no longer the only thing pinning the wiring.
_main = inspect.getsource(export_onnx.main)
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
# The sessions this file opened are dropped before the report, so a torn-down session cannot be
# blamed for a failure printed after it.
del session, loaded, pinned

shutil.rmtree(tmp, ignore_errors=True)
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)

# Exit without running interpreter finalization.
#
# This file is the one place in the suite that puts torch, onnxruntime and onnxruntime's quantizer
# in a single process, and on macOS that combination aborts during C++ static destruction:
# `libc++abi: terminating due to uncaught exception of type std::__1::system_error: recursive_mutex
# lock failed: Invalid argument`, exit 134, AFTER this summary has printed "0 failed". A green
# report and a red job.
#
# It is not this file's sessions being alive -- measured, 3 of 30 runs abort with every session
# explicitly released and 0 of 30 with the release removed, so that hypothesis is dead -- and it is
# not onnxruntime alone: 1 or 3 bare sessions opened and held in a fresh process abort 0 of 15
# times. It is the teardown ORDER of two native libraries, which Python cannot fix from inside.
#
# So finalization is skipped. `os._exit` ends the process with the status this file computed, after
# stdout is flushed, and the destructors that abort never run. The tests have all completed and
# their results are already printed; nothing below this line was going to run anyway. Measured on
# macOS with onnxruntime 1.30.0: 0 of 40 runs abort, against 4 of 42 for the same tests exiting
# normally.
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if FAIL else 0)
