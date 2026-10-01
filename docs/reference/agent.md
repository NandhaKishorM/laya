# Agent

`laya.Agent` loads one checkpoint and answers typed questions about a state. `laya.load` is
a shortcut for `Agent(...)`, and `laya.RLAgent` is an alias of `Agent`. `ONNXAgent` runs an
exported ONNX model on CPU; import it from `laya.onnx_agent`.

::: laya.agent.Agent

::: laya.agent.load

::: laya.onnx_agent.ONNXAgent

## Quantized export

`scripts/export_onnx.py --quantize` writes an INT8 weight-only quantized copy beside the fp32
export (`laya.onnx` also produces `laya.int8.onnx`). Dynamic quantization converts the `MatMul`
weights to int8 with the activation scale computed per input at run time, so no calibration
dataset is needed, and `ONNXAgent` loads the result by pointing `onnx_path` at it. On CPU it is
roughly 2x faster than the eager model and ~1.8x faster than the fp32 ONNX graph, and 1.4-2.8x
smaller depending on the checkpoint.

INT8 trades real accuracy, so it is a size/latency option, not a free one — do not use it where
the calibrated probability or confidence matters. Scales are **per-tensor** by default; `--per-channel`
opts into per-channel weights but on the dynamic path that collapses the decision model (agreement
with the eager model dropped to ~32% on the English checkpoint and ~40% on the multilingual one,
vs ~67% / ~83% per-tensor; see issue #790). Even per-tensor drifts noticeably on the larger
checkpoint; accuracy-safe int8 would need QAT or SmoothQuant-style outlier handling. The int8
graph is CPU-only: ONNX Runtime has no INT8 MatMul kernel on the CUDAExecutionProvider, and a GPU
provider silently falls back per node.

```bash
python scripts/export_onnx.py --model convaiinnovations/laya --output laya.onnx --quantize
```
