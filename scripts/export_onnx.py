import argparse
import inspect
import os
from typing import Any, Dict, Optional

import torch

from laya.agent import Agent


def quantize_model(model_path: str, output_path: str) -> str:
    """Write an INT8 weight-only dynamically quantized copy of `model_path`.

    Dynamic quantization converts the weights of every `MatMul` (the attention and MLP linear
    layers) to int8 while leaving activations in fp32; the quantization scales are computed per
    output channel at load time, so no calibration dataset is needed. The graph structure and
    the input/output names are unchanged, which is what lets `ONNXAgent` load the result by
    pointing `onnx_path` at it. It is CPU-only: ONNX Runtime has no INT8 MatMul kernel on the
    CUDAExecutionProvider, so an int8 graph on GPU falls back to CPU.
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
        # One scale per output channel rather than one per tensor. On the English checkpoint
        # measured on 20 support-ticket states x choice/noul/score, per-tensor int8 flipped 3
        # of 20 decisions (max probability drift 0.29); per-channel flipped none (max 0.09)
        # at the same size and speed.
        per_channel=True,
    )
    return output_path


def int8_output_path(output_path: str) -> str:
    """`laya.onnx` -> `laya.int8.onnx`, next to the fp32 export it was quantized from."""
    root, ext = os.path.splitext(output_path)
    return "%s.int8%s" % (root, ext or ".onnx")


def _shape_kwargs(supports_dynamic_shapes: Optional[bool] = None) -> Dict[str, Any]:
    """Shape declarations in the spelling the installed `torch.onnx.export` accepts.

    Two spellings exist. The dynamo exporter (torch >= 2.6) reads `dynamic_shapes` with
    `torch.export.Dim` symbols and supports opset 18; the TorchScript exporter that torch 2.2
    ships reads `dynamic_axes` and stops at opset 17. This fork pins torch 2.2 on macOS/Intel
    (LOCAL_SETUP.md), where passing `dynamic_shapes` raised `TypeError: export() got an
    unexpected keyword argument 'dynamic_shapes'` and the exporter could not run at all. Pick by
    the installed signature rather than `torch.__version__`, so a torch that backports either
    spelling is handled too. The outputs follow from the inputs either way: `act_logits` is
    (batch_size, 2) rather than the traced (2, 2).
    """
    if supports_dynamic_shapes is None:
        supports_dynamic_shapes = "dynamic_shapes" in inspect.signature(torch.onnx.export).parameters
    if not supports_dynamic_shapes:
        return {
            "opset_version": 17,
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "marker_pos": {0: "batch_size", 1: "num_markers"},
                "marker_mask": {0: "batch_size", 1: "num_markers"},
                "qtype": {0: "batch_size"},
                "logits": {0: "batch_size", 1: "num_markers"},
                "act_logits": {0: "batch_size"},
            },
        }
    batch_dim = torch.export.Dim("batch_size")
    seq_dim = torch.export.Dim("seq_len")
    marker_dim = torch.export.Dim("num_markers")
    return {
        "opset_version": 18,
        "dynamic_shapes": (
            {0: batch_dim, 1: seq_dim},
            {0: batch_dim, 1: seq_dim},
            {0: batch_dim, 1: marker_dim},
            {0: batch_dim, 1: marker_dim},
            {0: batch_dim},
        ),
    }


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
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        **_shape_kwargs(),
    )
    
    print(f"Successfully exported ONNX model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Laya model to ONNX format")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya", help="HuggingFace Hub ID or local path")
    parser.add_argument("--output", type=str, default="laya.onnx", help="Output path for the ONNX file")
    parser.add_argument("--quantize", action="store_true",
                        help="Also write an INT8 weight-only quantized copy (CPU-only speed and "
                             "size win) next to --output, named <output>.int8.onnx")
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)
    if args.quantize:
        int8_path = quantize_model(args.output, int8_output_path(args.output))
        print(f"Successfully wrote INT8 quantized model to: {int8_path}")
