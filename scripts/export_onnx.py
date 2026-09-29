import argparse
import os
from typing import Optional

import numpy as np
import torch

from laya.agent import Agent
from laya.common import QTYPES


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


# Tracing shapes. `torch.export` specializes every example dimension whose extent is 1, so a
# batch-of-one example bakes batch=1 into the decision head's attention reshapes no matter how
# the batch axis is declared. Nothing about that failure is visible at export time: the graph
# still *reports* a symbolic batch axis, ONNX Runtime still accepts a wider feed, and the run
# then dies inside the graph ("Attempting to broadcast an axis by a dimension other than 1") --
# which is every `ONNXAgent` request carrying more than one question or state.
# Two rows keep the axis symbolic. laya-ts/scripts/export_onnx.py carries the same note for the
# encoder/head pair it writes.
TRACE_BATCH = 2

# Tolerance for the INT8 graph, which is looser than the fp32 one on purpose: weight-only
# quantization is *meant* to change the numbers. Measured on the tiny synthetic model in
# tests/test_onnx_export_batch.py, the worst probability drift across the swept shapes is 9.95e-04 --
# already at the fp32 atol of 1e-3, so reusing that would flake. What this check is really for is
# that the quantized graph still RUNS at batch > 1: `quantize_model` deletes every `value_info` and
# rewrites every `MatMul`, which is exactly the kind of rewrite that can re-specialize an axis.
INT8_ATOL = 2e-2

INPUT_NAMES = [
    "input_ids",
    "attention_mask",
    "marker_pos",
    "marker_mask",
    "qtype",
]

OUTPUT_NAMES = ["logits", "act_logits"]

# `dynamic_axes` rather than `torch.export.Dim`s, per #726: it is the declaration every torch this
# package supports accepts (`dynamic_shapes` raises under the TorchScript exporter -- the default
# through torch 2.8 -- and does not exist before 2.5), and the dynamo exporter converts it to Dims.
# With the batch-2 example inputs above, both produce the same graph; `TRACE_BATCH` is what fixes
# #695, not the declaration. An earlier revision of this branch switched to `dynamic_shapes` before
# that reasoning existed upstream; it is reverted here rather than carried.
#
# One `batch_size` symbol shared by every input and output, so a feed is only valid when its rows
# agree -- which is what `ONNXAgent` always sends.
DYNAMIC_AXES = {
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
    "marker_pos": {0: "batch_size", 1: "num_markers"},
    "marker_mask": {0: "batch_size", 1: "num_markers"},
    "qtype": {0: "batch_size"},
    "logits": {0: "batch_size", 1: "num_markers"},
    "act_logits": {0: "batch_size"},
}


def example_inputs(batch: Optional[int] = None, seq_len: int = 16, num_markers: int = 2):
    """Positional inputs for `DecisionModel.forward`, shaped `(batch, ...)`.

    Ragged past the first row on purpose -- padding in `attention_mask` and a masked-off
    trailing marker -- so tracing and verification both run through `src_key_padding_mask` and
    the `masked_fill` on the option logits instead of the degenerate all-ones case. Values are
    deterministic so a verification failure is reproducible.

    `batch` defaults to `TRACE_BATCH` at call time rather than in the signature, so that setting
    `export_onnx.TRACE_BATCH` reaches this function -- which is the one experiment anybody
    revisiting the batch bug will want to run.
    """
    if batch is None:
        batch = TRACE_BATCH
    # `marker_pos` indexes the sequence axis, so it has to stay inside it: the largest position
    # written below is `num_markers`, and a position at or past `seq_len` makes the head's marker
    # gather read out of bounds -- `IndexError` in torch, and in ONNX Runtime the much less
    # obvious "GatherElements op: Out of range value in index tensor".
    if num_markers >= seq_len:
        raise ValueError(
            "num_markers=%d needs seq_len > %d: marker positions index the sequence axis, and "
            "position %d does not exist in a length-%d sequence"
            % (num_markers, num_markers, num_markers, seq_len))
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, 100, (batch, seq_len), generator=generator, dtype=torch.long)
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long)
    marker_pos = torch.arange(1, num_markers + 1, dtype=torch.long).repeat(batch, 1)
    marker_mask = torch.ones((batch, num_markers), dtype=torch.bool)
    for row in range(1, batch):
        # One more padded token per row, but never the whole row: `max(1, ...)` keeps at least
        # one real token. Unclamped, `seq_len - row` reaches 0 at `row == seq_len` and turns
        # negative after it, so `batch > seq_len` would hand the model an all-padding row -- on
        # which torch and ONNX Runtime legitimately disagree: measured at batch=32, seq_len=16,
        # where 16 of the 32 rows come out entirely padding, 2.76e-01 on `act_logits` and
        # 1.12e-01 on `logits`, against 2.38e-07 and 1.19e-07 once clamped. The caller would read
        # that as an export failure.
        attention_mask[row, max(1, seq_len - row):] = 0
        marker_mask[row, -1] = False
    qtype = torch.arange(batch, dtype=torch.long) % len(QTYPES)
    return input_ids, attention_mask, marker_pos, marker_mask, qtype


def export_module(model, output_path: str, opset_version: int = 18) -> str:
    """Write `model` to `output_path` as an ONNX graph with a symbolic batch, sequence and
    marker axis.

    Split out from `export_to_onnx` so the batch-dynamic contract can be checked on a small
    hand-built model, with no checkpoint to download; see `tests/test_onnx_export_batch.py`.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # The encoder is not detached here: ONNX export runs the model under `torch.no_grad`
    # semantics anyway, and `detach_encoder` only affects the hidden-state gradient, so the
    # keyword would change nothing in the exported graph.
    # `dynamic_shapes`, not `dynamic_axes`. Both work from a two-row example, and neither fixes
    # the bug on its own -- `TRACE_BATCH` does, because `torch.export` specializes any example
    # dimension of extent 1 whatever the axis declaration says. What decides between them is which
    # one keeps working: torch calls `dynamic_axes` deprecated under `dynamo=True` ("Prefer
    # specifying ``dynamic_shapes``"), and on the release that drops the conversion an export
    # declaring only `dynamic_axes` would go *silently* static -- a graph that serves one question,
    # no error, exactly the failure this file exists to close. `verify_batch_dynamic` is the only
    # thing that would catch that day, which is why it runs by default and compares numbers rather
    # than shapes.
    torch.onnx.export(
        model,
        example_inputs(),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=DYNAMIC_AXES,
    )
    return output_path


def probabilities(logits):
    """Softmax over the last axis -- what `ONNXAgent` turns both outputs into."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def verify_batch_dynamic(model, output_path: str, batches=(1, 2, 3), atol: float = 1e-3) -> None:
    """Run the graph at `output_path` in ONNX Runtime at each batch size and compare to `model`.

    This is the check the exporter did not have. Nothing short of running the graph at batch > 1
    distinguishes a usable export from a batch-1-only one: the declared input shapes read
    `batch_size` in both cases, and torch reports the problem as a `UserWarning` the script
    printed straight past. Raises `SystemExit`, because a graph that cannot serve two questions
    is not a successful export, and writing one without a word is how a user ends up debugging
    ONNX Runtime instead of reading an error here.

    Compared as probabilities rather than raw logits, because probabilities are what `ONNXAgent`
    returns and all a decision depends on. The `torch.export` decompositions move a raw logit far
    more than they move the answer: measured on the 322M multilingual checkpoint, raw `act_logits`
    differ by up to 1.2e-02 and raw `logits` by 4.3e-04, while the probabilities both produce
    agree to 6.1e-05. laya-ts/scripts/export_onnx.py compares its act head the same way, for the
    same reason.

    **Marker width is dynamic for two or more markers, and baked at exactly one.**
    `DecisionModel.forward` branches on `p.size(-1) >= 2` in Python, so the traced width bakes THAT
    BRANCH into the graph -- not the width itself. Measured on a graph traced at 2 markers: widths
    2, 3, 4, 5 and 8 all run and match PyTorch; width 1 raises an ONNX Runtime `TopK` error
    (`k argument [2] should not be greater than specified axis dim value [1]`), which a one-criterion
    `score` question produces (`criteria=["only"]` is accepted and yields exactly one marker). An
    earlier revision of this docstring said the width was "not dynamic", which would have told a user
    to re-export per criteria count; that is not necessary. The limitation is narrower still, because
    `collate_items` sizes markers to the batch maximum, so a one-criterion question only hits it when
    every item in the collated batch has exactly one marker. Removing the Python branch from the model
    is a separate change.

    Probability space has one blind spot, and it is deliberate: softmax is shift-invariant, so a
    constant added to every logit in a row is invisible here -- measured on the exported graph,
    `logits + 100.0` passes while `logits * 3.0` is caught. That is safe only because every
    consumer softmaxes. `ONNXAgent._decode_answers` is the only reader of `logits`, and it divides
    by `t_scale` first, so for the `t_scale > 1` part of the clamped [0.5, 5.0] range this check is
    strictly more sensitive than production; at the bottom of that range production is at most
    twice as sensitive, still far inside `atol` at the 6.1e-05 the checkpoint produces. A consumer
    that ever read a raw logit would need a raw-logit check added here.
    """
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        # The export itself never needed onnxruntime -- `torch.onnx.export` does not touch it -- so
        # verifying by default makes it a new requirement of a command that used to work without it.
        # A bare traceback after "Successfully exported" reads as a failed export, which it was not.
        raise SystemExit(
            "the export succeeded, but verifying it needs onnxruntime, which is not installed.\n"
            "Install it with `pip install laya[onnx]`, or skip the check with `--no-verify`.")

    try:
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    except Exception as error:
        # A graph ONNX Runtime will not even load is a failed export too, and the caller should
        # read that here rather than in a raw ORT traceback.
        raise SystemExit(
            "verification failed: ONNX Runtime cannot load the graph just written to %s, so "
            "ONNXAgent cannot use it. ONNX Runtime said: %s" % (output_path, error))
    for batch in batches:
        # Sequence and marker counts move with the batch so no run can pass by accidentally
        # matching the traced shape.
        inputs = example_inputs(batch=batch, seq_len=16 + 8 * batch, num_markers=1 + batch)
        with torch.no_grad():
            expected = model(*inputs)
        feed = {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, inputs)}
        try:
            got = session.run(OUTPUT_NAMES, feed)
        except Exception as error:
            raise SystemExit(
                "verification failed: the exported graph does not run at batch %d, so ONNXAgent "
                "cannot serve a request with %d questions or states. ONNX Runtime said: %s"
                % (batch, batch, error)
            )
        for name, actual, want in zip(OUTPUT_NAMES, got, expected):
            want = want.numpy()
            diff = float(np.max(np.abs(probabilities(actual) - probabilities(want))))
            if not diff <= atol:
                raise SystemExit("verification failed: %s probabilities differ from PyTorch by "
                                 "%.2e (> %.0e) at batch %d" % (name, diff, atol, batch))
            print("  batch %-2d %-10s max abs prob diff vs PyTorch %.2e (raw logits %.2e)"
                  % (batch, name, diff, float(np.max(np.abs(actual - want)))))


def export_to_onnx(model_id_or_path: str, output_path: str, verify: bool = True):
    print(f"Loading PyTorch Agent from: {model_id_or_path}")
    agent = Agent(model_id_or_path, compile=False, device="cpu")

    print(f"Exporting to {output_path} (this may take a minute)...")
    export_module(agent.model, output_path)
    print(f"Successfully exported ONNX model to: {output_path}")

    if verify:
        print("Verifying the export against PyTorch at batch 1, 2 and 3...")
        verify_batch_dynamic(agent.model, output_path)
        print("Verification passed: the graph runs at batch 1, 2 and 3 with 2-4 markers and\n"
              "              matches PyTorch. Width 1 is baked out by a Python branch -- see\n"
              "              verify_batch_dynamic.")
    # Handed back so the caller can verify a quantized copy against the same weights without
    # loading the checkpoint a second time.
    return agent.model


def main(argv=None):
    """The CLI, as a function so the arguments it passes can be asserted.

    It was previously inline under `if __name__ == "__main__":`, which meant the only way to pin it
    was to grep this file for substrings -- and every way of turning the new verification off
    survived that: `batches=(1,)` on either call, `--no-verify` flipped to `store_false`,
    `verify=False`, and `atol=1e9` on the INT8 check all shipped a success message while checking
    nothing, with both suites green.
    """
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
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the post-export check that runs the graph in ONNX Runtime at "
                             "batch 1, 2 and 3 and compares it with PyTorch")
    args = parser.parse_args(argv)

    model = export_to_onnx(args.model, args.output, verify=not args.no_verify)
    if args.quantize:
        int8_path = quantize_model(args.output, int8_output_path(args.output),
                                   per_channel=args.per_channel)
        print(f"Successfully wrote INT8 quantized model to: {int8_path}")
        # The quantized graph is the one docs/evals.md tells people to deploy, and quantization is
        # precisely the "runs but returns different numbers" case this verifier exists for -- so it
        # gets checked too, at a tolerance that expects quantization to move the numbers.
        if not args.no_verify:
            print("Verifying the INT8 graph at batch 1, 2 and 3...")
            verify_batch_dynamic(model, int8_path, atol=INT8_ATOL)
            print("INT8 verification passed: it runs at batch > 1 and stays within "
                  f"{INT8_ATOL:.0e} of PyTorch in probability space.")


if __name__ == "__main__":
    main()
