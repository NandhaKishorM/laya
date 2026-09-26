"""Export a Laya checkpoint to split ONNX (encoder.onnx + head.onnx).

Source of truth stays ``model.safetensors``; this script produces the two ONNX
files the ``laya-ts`` Node/Web providers load. Run once per checkpoint:
    python laya-ts/scripts/export_onnx.py --repo convaiinnovations/laya --out-dir ./model
    python laya-ts/scripts/export_onnx.py --repo convaiinnovations/laya --subfolder multilingual --out-dir ./model-ml
    python laya-ts/scripts/export_onnx.py --model-dir <local ckpt> --out-dir ./model

With ``--repo`` the checkpoint is downloaded from Hugging Face (safetensors +
tokenizer + encoder) into the HF cache first, then converted. With
``--model-dir`` an already-downloaded directory is used directly.

The encoder graph takes ``(input_ids, attention_mask)`` and returns
``last_hidden_state``. The head graph takes ``(hidden_states, marker_pos,
marker_mask, qtype)`` and returns ``(logits, act_logits)``. Verification runs
a torch forward at ``--seq-len`` and at ``--verify-len`` (default 512, past the
local_attention=128 sliding window) and requires the ONNX outputs to match
within 1e-4 at both lengths before anything is written.
"""
import argparse
import json
import os
import shutil

try:
    from laya.common import build_model
except ImportError:  # --help must work without torch installed
    build_model = None

DEFAULT_REPO = "convaiinnovations/laya"


def _download(repo, subfolder=None, token=None):
    """Download a checkpoint from HF into the HF cache; return the checkpoint dir."""
    # Windows without Developer Mode can't symlink the HF cache; the hub prints
    # a long warning and falls back to copies. Silence it, copies still work.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("download needs huggingface_hub: pip install huggingface_hub")
    prefix = ("%s/" % subfolder) if subfolder else ""
    allow = [prefix + name for name in (
        "rl_agent_config.json", "model.safetensors", "config.json",
        "tokenizer.json", "tokenizer/*", "encoder/*",
    )]
    root = snapshot_download(repo, allow_patterns=allow,
                             token=token or os.environ.get("HF_TOKEN"))
    path = os.path.join(root, subfolder) if subfolder else root
    if not os.path.isdir(path):
        raise SystemExit("subfolder %r not found in %r" % (subfolder, repo))
    return path


def _ensure_tokenizer_json(model_dir):
    """Copies tokenizer/tokenizer.json up to the checkpoint root when needed."""
    dest = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(dest):
        src = os.path.join(model_dir, "tokenizer", "tokenizer.json")
        if os.path.exists(src):
            shutil.copy(src, dest)


def _checkpoint_weights_path(model_dir):
    """Require the checkpoint's safetensors weights before building a model."""
    path = os.path.join(model_dir, "model.safetensors")
    if not os.path.isfile(path):
        raise SystemExit("checkpoint is missing model.safetensors: %s" % model_dir)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HF repo to download first (default %s with --model-dir absent)." % DEFAULT_REPO)
    src.add_argument("--model-dir", help="Local checkpoint dir with rl_agent_config.json (+ tokenizer.json).")
    p.add_argument("--subfolder", default=None, help="Subfolder inside the repo (e.g. multilingual, typed-decisions).")
    p.add_argument("--token", default=None, help="HF token (or HF_TOKEN env).")
    p.add_argument("--out-dir", required=True, help="Output dir for encoder.onnx, head.onnx, tokenizer.json, rl_agent_config.json.")
    p.add_argument("--opset", type=int, default=18, help="ONNX opset (default 18; the exporter implements 18 natively and a downgrade to 17 emits invalid Split nodes).")
    p.add_argument("--seq-len", type=int, default=16, help="Dummy sequence length for tracing/verification (default 16).")
    p.add_argument("--verify-len", type=int, default=512, help="Second length for the torch-vs-ONNX 1e-4 check (default 512; exercises sliding-window layers, local_attention=128).")
    p.add_argument("--no-verify", action="store_true", help="Skip the torch-vs-ONNX 1e-4 check.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if build_model is None:
        raise SystemExit("export needs torch + laya installed: pip install torch transformers onnx onnxruntime")
    model_dir = args.model_dir or _download(args.repo or DEFAULT_REPO, args.subfolder, args.token)
    weights_path = _checkpoint_weights_path(model_dir)
    _ensure_tokenizer_json(model_dir)
    import torch

    class _EncoderOnly(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.encoder

        def forward(self, input_ids, attention_mask):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    class _HeadOnly(torch.nn.Module):
        """Decision head tail: type_emb + transformer head + scorer + act_head."""

        def __init__(self, model):
            super().__init__()
            self.head = model.head
            self.type_emb = model.type_emb
            self.scorer = model.scorer
            self.act_head = model.act_head

        def forward(self, hidden_states, marker_pos, marker_mask, qtype, attention_mask):
            h = hidden_states + self.type_emb(qtype.squeeze(-1))[:, None, :]
            if self.head is not None:
                # Same as DecisionModel.forward: padding must not be attended.
                # Without this, batch mates of unequal length corrupt each
                # other's markers through the head transformer.
                pad = attention_mask == 0
                for layer in self.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m = torch.gather(h, 1, idx)
            logits = self.scorer(m).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)
            p = torch.softmax(logits.detach(), -1)
            k = marker_mask.sum(-1).clamp(min=2).float()
            ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
            top2 = p.topk(2, -1).values
            feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
            act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1))
            return logits, act_logits

    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    # The encoder arch lives in <ckpt>/encoder/config.json (same as Agent:
    # enc_dir = model_dir/"encoder"); the checkpoint root itself has no
    # config.json, so passing model_dir here makes AutoConfig fail with
    # "Unrecognized model ... no model_type key".
    enc_dir = os.path.join(model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None,
                        pretrained=False)
    from safetensors.torch import load_file as _load_safetensors

    model.load_state_dict(_load_safetensors(weights_path), strict=False)
    model.eval().float()

    # Batch must stay symbolic: tracing with batch=1 lets dynamo bake batch=1
    # into a head reshape (batch=2 then fails with "204 by 408" broadcast).
    # Explicit Dims + batch=2 dummies keep it dynamic; verify covers 1 and 2.
    batch_dim = torch.export.Dim("batch", min=1, max=64)
    seq_dim = torch.export.Dim("seq", min=1, max=8192)
    markers_dim = torch.export.Dim("markers", min=1, max=256)
    enc_shapes = ({0: batch_dim, 1: seq_dim}, {0: batch_dim, 1: seq_dim})
    head_shapes = ({0: batch_dim, 1: seq_dim}, {0: batch_dim, 1: markers_dim},
                   {0: batch_dim, 1: markers_dim}, {0: batch_dim},
                   {0: batch_dim, 1: seq_dim})

    seq = int(args.seq_len)
    input_ids = torch.ones(2, seq, dtype=torch.long)
    attention_mask = torch.ones(2, seq, dtype=torch.long)
    ref_hidden, ref_logits, ref_act, marker_pos, marker_mask, qtype, ref_att = _run_ref(model, _HeadOnly(model).eval(), seq, batch=2)

    os.makedirs(args.out_dir, exist_ok=True)
    enc_path = os.path.join(args.out_dir, "encoder.onnx")
    head_path = os.path.join(args.out_dir, "head.onnx")

    torch.onnx.export(
        _EncoderOnly(model).eval(),
        (input_ids, attention_mask),
        enc_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "last_hidden_state": {0: "batch", 1: "seq"}},
        dynamic_shapes=enc_shapes,
        opset_version=args.opset,
    )
    torch.onnx.export(
        _HeadOnly(model).eval(),
        (ref_hidden, marker_pos, marker_mask, qtype, ref_att),
        head_path,
        input_names=["hidden_states", "marker_pos", "marker_mask", "qtype", "attention_mask"],
        output_names=["logits", "act_logits"],
        dynamic_axes={"hidden_states": {0: "batch", 1: "seq"}, "marker_pos": {0: "batch", 1: "markers"}, "marker_mask": {0: "batch", 1: "markers"}, "qtype": {0: "batch"}},
        dynamic_shapes=head_shapes,
        opset_version=args.opset,
    )

    if not args.no_verify:
        _verify(enc_path, head_path, input_ids, attention_mask, marker_pos, marker_mask, qtype, ref_att, ref_logits, ref_act)
        b1 = _run_ref(model, _HeadOnly(model).eval(), seq, batch=1)
        _verify(enc_path, head_path, torch.ones(1, seq, dtype=torch.long),
                torch.ones(1, seq, dtype=torch.long), *b1[3:6], b1[6], b1[1], b1[2])
        if int(args.verify_len) != seq:
            v_hidden, v_logits, v_act, v_pos, v_mask, v_qtype, v_att2 = _run_ref(model, _HeadOnly(model).eval(), int(args.verify_len), batch=2)
            v_ids = torch.ones(2, int(args.verify_len), dtype=torch.long)
            v_att = torch.ones(2, int(args.verify_len), dtype=torch.long)
            _verify(enc_path, head_path, v_ids, v_att, v_pos, v_mask, v_qtype, v_att2, v_logits, v_act)

    for name in ("tokenizer.json", "rl_agent_config.json"):
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out_dir, name))
    print("wrote %s and %s" % (enc_path, head_path))


def _run_ref(model, head, seq, batch=1):
    import torch

    input_ids = torch.ones(batch, seq, dtype=torch.long)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    with torch.inference_mode():
        ref_hidden = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        marker_pos = torch.tensor([[1, 2]] * batch, dtype=torch.long)
        marker_mask = torch.tensor([[True, True]] * batch)
        qtype = torch.tensor([[0]] * batch, dtype=torch.long)
        ref_logits, ref_act = head(ref_hidden, marker_pos, marker_mask, qtype, attention_mask)
    return ref_hidden, ref_logits, ref_act, marker_pos, marker_mask, qtype, attention_mask


def _verify(enc_path, head_path, input_ids, attention_mask, marker_pos, marker_mask, qtype, head_att, ref_logits, ref_act):
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("export --no-verify not given but onnxruntime is missing: pip install onnxruntime")
    enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    head = ort.InferenceSession(head_path, providers=["CPUExecutionProvider"])
    (hidden,) = enc.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()})
    logits, act = head.run(None, {"hidden_states": hidden.astype(np.float32), "marker_pos": marker_pos.numpy(), "marker_mask": marker_mask.numpy(), "qtype": qtype.numpy(), "attention_mask": head_att.numpy()})
    diff = float(np.abs(logits - ref_logits.detach().numpy()).max())
    if diff > 1e-4:
        raise SystemExit("verification failed: logits max abs diff %.2e > 1e-4" % diff)
    # act_logits are consumed only through softmax (rounded to 4dp downstream),
    # and dynamo decompositions add ~5e-4 raw-logit noise that softmax washes
    # out. Compare the probabilities TS actually reads instead of raw logits.
    got_p = np.exp(act - act.max(axis=-1, keepdims=True))
    got_p /= got_p.sum(axis=-1, keepdims=True)
    ref = ref_act.detach().numpy()
    ref_p = np.exp(ref - ref.max(axis=-1, keepdims=True))
    ref_p /= ref_p.sum(axis=-1, keepdims=True)
    pdiff = float(np.abs(got_p - ref_p).max())
    if pdiff > 1e-4:
        raise SystemExit("verification failed: act_probs max abs diff %.2e > 1e-4" % pdiff)
    print("verify ok: torch vs onnx match within 1e-4 (logits %.1e, act_probs %.1e)" % (diff, pdiff))


if __name__ == "__main__":
    main()
