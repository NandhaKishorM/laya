"""INT8 export: `scripts/export_onnx.py --quantize` writes a weight-only quantized copy.

`ONNXAgent` loads any graph that keeps the fp32 export's input/output names, so the whole
feature is one function: `quantize_model` (dynamic per-channel INT8 MatMul weights via
onnxruntime) plus
`int8_output_path` (the sidecar naming) and the CLI flag. No checkpoint and no torch model is
downloaded here: the quantizer is exercised on a tiny hand-built MatMul graph, which is exactly
the shape it targets in the real export, and the CLI is checked through `--help`.

Run: python tests/test_onnx_quantize.py
"""
import importlib.util
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

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


# ---------------------------------------------------------------- a tiny fp32 MatMul graph
def _build_model(path):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1, 8)).astype(np.float32)
    w = (np.eye(8, dtype=np.float32) + 0.02 * rng.normal(size=(8, 8)).astype(np.float32))
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["input_ids", "weight"], ["logits"])],
            "mini",
            [helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [1, 8])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 8])],
            [numpy_helper.from_array(w, name="weight")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 13  # keep the graph loadable by the installed onnxruntime
    onnx.save(model, path)
    return x


tmp = os.path.join(_here, "_tmp_onnx_quantize")
os.makedirs(tmp, exist_ok=True)
fp32_path = os.path.join(tmp, "mini.onnx")
int8_path = os.path.join(tmp, "mini.int8.onnx")
x = _build_model(fp32_path)

# ---------------------------------------------------------------- int8_output_path
check("naming/laya.onnx -> laya.int8.onnx", export_onnx.int8_output_path("laya.onnx"),
      "laya.int8.onnx")
check("naming/keeps the directory", export_onnx.int8_output_path("/models/laya.onnx"),
      "/models/laya.int8.onnx")
check("naming/extensionless input still gets .onnx", export_onnx.int8_output_path("laya"),
      "laya.int8.onnx")

# ---------------------------------------------------------------- quantize_model
out = export_onnx.quantize_model(fp32_path, int8_path)
check("quantize/returns the output path", out, int8_path)
check_true("quantize/writes the file", os.path.exists(int8_path))

q_model = onnx.load(int8_path)
int8_tensors = [t for t in q_model.graph.initializer if t.data_type == TensorProto.INT8]
check_true("quantize/MatMul weight is stored as INT8", len(int8_tensors) >= 1,
           [(t.name, t.data_type) for t in q_model.graph.initializer])
scales = [t for t in q_model.graph.initializer if t.name.endswith("_scale")]
check_true("quantize/scales are per output channel (per-tensor flips decisions on the real model)",
           scales and all(numpy_helper.to_array(t).size > 1 for t in scales),
           [(t.name, list(t.dims)) for t in scales])
check("quantize/graph keeps the I/O names ONNXAgent binds to",
      ([i.name for i in q_model.graph.input], [o.name for o in q_model.graph.output]),
      (["input_ids"], ["logits"]))

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
got = sess.run(["logits"], {"input_ids": x})[0]
fp32_sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
ref = fp32_sess.run(["logits"], {"input_ids": x})[0]
check_true("quantize/outputs stay close to fp32 on a well-conditioned weight",
           np.allclose(got, ref, rtol=0.05, atol=0.05),
           "max abs diff %.4f" % float(np.max(np.abs(got - ref))))
check_true("quantize/magnitude is actually exercised (not an empty pass)",
           float(np.max(np.abs(got - ref))) > 0.0)

def _weight_bytes(model):
    """Raw bytes held by initializers -- the honest size comparison on a tiny graph,
    where container overhead dwarfs an 8x8 weight."""
    return sum(numpy_helper.to_array(t).nbytes for t in model.graph.initializer)


check_true("quantize/weight storage shrinks vs fp32",
           _weight_bytes(q_model) < _weight_bytes(onnx.load(fp32_path)),
           (_weight_bytes(q_model), _weight_bytes(onnx.load(fp32_path))))

# the fp32 export itself is untouched -- the quantized model is a sidecar, so A/B is possible
check("quantize/fp32 model still loads and matches itself",
      np.allclose(fp32_sess.run(["logits"], {"input_ids": x})[0], ref, atol=0), True)

# ---------------------------------------------------------------- CLI surface
help_text = subprocess.run(
    [sys.executable, os.path.join(_here, os.pardir, "scripts", "export_onnx.py"), "--help"],
    capture_output=True, text=True, timeout=120,
).stdout
check_true("cli/--quantize is documented in --help", "--quantize" in help_text,
           [ln for ln in help_text.splitlines() if "quantize" in ln])
check_true("cli/help names the sidecar output", ".int8.onnx" in help_text)


# ---------------------------------------------------------------- report
import shutil  # noqa: E402

shutil.rmtree(tmp, ignore_errors=True)
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
