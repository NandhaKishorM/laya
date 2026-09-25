# Agent

`laya.Agent` loads one checkpoint and answers typed questions about a state. `laya.load` is
a shortcut for `Agent(...)`, and `laya.RLAgent` is an alias of `Agent`. `ONNXAgent` runs an
exported ONNX model on CPU; import it from `laya.onnx_agent`.

::: laya.agent.Agent

::: laya.agent.load

::: laya.onnx_agent.ONNXAgent

## Quantized export

`scripts/export_onnx.py --quantize` writes an INT8 weight-only quantized copy beside the fp32
export (`laya.onnx` also produces `laya.int8.onnx`). Dynamic per-channel quantization converts
the `MatMul` weights to int8 with activations left in fp32, so no calibration dataset is needed,
and `ONNXAgent` loads the result by pointing `onnx_path` at it. On the English checkpoint, CPU
(M-series, 20 support-ticket states x choice/noul/score): model file 1.6 GB -> 581 MB, p50
per-state latency ~340 ms -> ~250 ms (~1.35x), and zero decision changes versus fp32 (largest single
probability drift 0.09). Per-tensor scales instead of per-channel flipped 3 of 20 states with
drift up to 0.29, which is why the exporter uses per-channel. The int8 graph is CPU-only: ONNX
Runtime has no INT8 MatMul kernel on the CUDAExecutionProvider, and a GPU provider silently
falls back per node.

```bash
python scripts/export_onnx.py --model convaiinnovations/laya --output laya.onnx --quantize
```
