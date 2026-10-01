import argparse
import os
import torch

from laya.agent import Agent


def quantize_model(model_path: str, output_path: str, per_channel: bool = False) -> str:
    """Write an INT8 weight-only dynamically quantized copy of `model_path`.

    Dynamic quantization converts the weights of every `MatMul` (the attention and MLP linear
    layers) to int8 while leaving activations in fp32; the activation scale is computed per input
    at run time, so no calibration dataset is needed. The graph structure and the input/output
    names are unchanged, which is what lets `ONNXAgent` load the result by pointing `onnx_path` at
    it. It is CPU-only: ONNX Runtime has no INT8 MatMul kernel on the CUDAExecutionProvider, so an
    int8 graph on GPU falls back to CPU.

    `per_channel` defaults to **False** (one scale per weight tensor). Per-channel scales are a
    static/QDQ feature; on the dynamic `MatMulInteger` path the per-channel weight scale does not
    combine correctly with the per-input activation scale, and on the real checkpoints it collapses
    the decision model (measured: the English ModernBERT-large agreed with the eager Agent on only
    31/96 held-out choice decisions at `per_channel=True` vs 64/96 at `per_channel=False`; the
    multilingual mmBERT was 40% vs 83%). See issue #790.

    INT8 is a size/latency option, not a free one. On CPU it is roughly 2x faster than eager and
    ~1.8x faster than the fp32 ONNX graph, and 1.4-2.8x smaller, but even at `per_channel=False` it
    trades real accuracy (the drift above is substantial, and worse on the larger checkpoint): do
    not use it where the calibrated probability or confidence matters. Accuracy-safe int8 for this
    model would need QAT or SmoothQuant-style outlier handling, not an export-time flag.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    import onnx

    model = onnx.load(model_path)
    # The torch exporter leaves intermediate `value_info` shapes that disagree with what the
    # quantizer's own shape-inference pass re-derives ("Inferred shape and existing shape
    # differ"). The declarations are informational only, so drop them and let quantization
    # recompute whatever it needs.
    del model.graph.value_info[:]
    quantize_dynamic(
        model_input=model,
        model_output=output_path,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    return output_path


def int8_output_path(output_path: str) -> str:
    """`laya.onnx` -> `laya.int8.onnx`, next to the fp32 export it was quantized from."""
    root, ext = os.path.splitext(output_path)
    return "%s.int8%s" % (root, ext or ".onnx")


def export_to_onnx(model_id_or_path: str, output_path: str):
    print(f"Loading PyTorch Agent from: {model_id_or_path}")
    agent = Agent(model_id_or_path, compile=False, device="cpu")
    
    print("Creating dummy input tensors...")
    # 1. Dummy tensors for tracing. torch.export specialises any dimension that is 1 (or equal
    # to another) at trace time, so a batch-1 dummy baked batch=1 into the decision head's
    # attention: the export ran at batch 1 and failed at batch >= 2 (#695). Batch, sequence and
    # marker counts are therefore all > 1 and pairwise different. laya-ts's exporter does the same.
    batch, seq_len, num_markers = 2, 17, 3
    dummy_input_ids = torch.randint(0, 100, (batch, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch, seq_len), dtype=torch.long)
    dummy_marker_pos = torch.tensor([[1, 5, 9]] * batch, dtype=torch.long)
    dummy_marker_mask = torch.ones((batch, num_markers), dtype=torch.bool)
    dummy_qtype = torch.zeros(batch, dtype=torch.long)

    inputs = (
        dummy_input_ids,
        dummy_attention_mask,
        dummy_marker_pos,
        dummy_marker_mask,
        dummy_qtype,
    )

    # 2. Dynamic dimensions. `dynamic_axes` rather than torch.export Dims: it is the declaration
    # every torch this package supports accepts (`dynamic_shapes` raises under the TorchScript
    # exporter, the default through torch 2.8, and does not exist before 2.5), and the dynamo
    # exporter converts it to Dims. With the batch-2 dummies above, both give the same graph;
    # the dummies are what fix #695, not the declaration.
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }

    input_names = [
        "input_ids",
        "attention_mask",
        "marker_pos",
        "marker_mask",
        "qtype",
    ]
    
    output_names = ["logits", "act_logits"]

    print(f"Exporting to {output_path} (this may take a minute)...")
    
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # We must detach the encoder because ONNX export runs the model in trace mode.
    # The `detach_encoder` flag in forward() just detaches the hidden state gradient, 
    # but we don't even need to pass it since kwargs are ignored by tracing.
    
    torch.onnx.export(
        agent.model,
        inputs,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    print(f"Successfully exported ONNX model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Laya model to ONNX format")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya", help="HuggingFace Hub ID or local path")
    parser.add_argument("--output", type=str, default="laya.onnx", help="Output path for the ONNX file")
    parser.add_argument("--quantize", action="store_true",
                        help="Also write an INT8 weight-only quantized copy (CPU-only speed and "
                             "size win, at a real accuracy cost) next to --output, named "
                             "<output>.int8.onnx")
    parser.add_argument("--per-channel", action="store_true",
                        help="Quantize weights per output channel instead of per tensor. Off by "
                             "default: on the dynamic path per-channel collapses the model (see "
                             "issue #790). Only meaningful with --quantize.")
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)
    if args.quantize:
        int8_path = quantize_model(args.output, int8_output_path(args.output),
                                   per_channel=args.per_channel)
        print(f"Successfully wrote INT8 quantized model to: {int8_path}")
