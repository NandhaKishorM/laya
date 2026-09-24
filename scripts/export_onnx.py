import argparse
import os
import torch

from laya.agent import Agent

def export_to_onnx(model_id_or_path: str, output_path: str):
    print(f"Loading PyTorch Agent from: {model_id_or_path}")
    agent = Agent(model_id_or_path, compile=False, device="cpu")
    
    # 1. Ensure model is in evaluation mode
    # If agent.model is wrapped (e.g. OptimizedModule or custom wrapper), unwrap if necessary
    model_to_export = getattr(agent.model, "_orig_mod", agent.model)
    model_to_export.eval()

    print("Creating dummy input tensors...")
    # 2. Dummy tensors for tracing
    # (batch_size=1, seq_len=16)
    dummy_input_ids = torch.randint(0, 100, (1, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 16), dtype=torch.long)
    
    # (batch_size=1, num_markers=2)
    dummy_marker_pos = torch.tensor([[1, 5]], dtype=torch.long)
    
    # Using int64 (or int32) instead of bool for mask avoids cast/type issues in older ONNX runtimes
    dummy_marker_mask = torch.tensor([[1, 1]], dtype=torch.long)
    
    # (batch_size=1)
    dummy_qtype = torch.tensor([0], dtype=torch.long)
    
    inputs = (
        dummy_input_ids,
        dummy_attention_mask,
        dummy_marker_pos,
        dummy_marker_mask,
        dummy_qtype,
    )

    # 3. Define dynamic axes
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
    
    # Disable gradient computation during tracing export
    with torch.no_grad():
        torch.onnx.export(
            model_to_export,
            inputs,
            output_path,
            export_params=True,
            opset_version=17,  # Opset 17 offers high stability across runtimes (e.g., ONNX Runtime, TensorRT)
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
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output)
