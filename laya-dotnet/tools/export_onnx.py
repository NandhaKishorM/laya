"""Export a Laya checkpoint to ONNX with external weights.

Produces the same artifact layout as the pre-built exports in onnx/:

    <out>/<name>/
        model.onnx          # graph structure (~2-3 MB)
        model.onnx.data     # weight sidecar (all initializers, ~1.3 GB)
        rl_agent_config.json
        tokenizer/
            tokenizer.json
            tokenizer_config.json
        fixtures.npz        # PyTorch reference outputs for ONNX parity checks

The tokenizer post_processor (TemplateProcessing) is deliberately kept in
tokenizer.json.  The .NET SDK strips it at load time and caches the result as
tokenizer.nopost.json -- that file is created by the SDK, not by this script.

Usage:
    # Export the multilingual checkpoint to onnx/multilingual/
    python laya-dotnet/tools/export_onnx.py --model multilingual

    # Re-export, overwriting an existing artifact
    python laya-dotnet/tools/export_onnx.py --model multilingual --force

    # Export all three checkpoints and verify each against PyTorch
    python laya-dotnet/tools/export_onnx.py --model all --verify

    # Export to a custom directory and verify
    python laya-dotnet/tools/export_onnx.py --model multilingual --out /tmp/laya-test --verify

    # Export a local checkpoint directory (must contain rl_agent_config.json)
    python laya-dotnet/tools/export_onnx.py /path/to/checkpoint --out onnx/my-model --verify

    # Use a private HuggingFace token
    python laya-dotnet/tools/export_onnx.py --model typed-decisions --token hf_...

    # Redirect HuggingFace cache to a drive with more free space (typed-decisions is 842 MB)
    HF_HOME=/d/hf_cache python laya-dotnet/tools/export_onnx.py --model typed-decisions
"""

import argparse
import io
import json
import os
import shutil
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, REPO)

from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import (  # noqa: E402
    QTYPES,
    build_model,
    build_sequence,
    collate_items,
)
from laya.router import DEFAULT_MODELS  # noqa: E402


# ─── model catalogue ────────────────────────────────────────────────────────

# Canonical short names for the three shipped checkpoints.
_NAMES = ["english", "multilingual", "typed-decisions"]


def _resolve_spec(name_or_path: str, token: str, revision: str = None):
    """Return (model_dir, subfolder, label) for a name or a local path."""
    if os.path.exists(name_or_path):
        # Local directory: treat as a self-contained checkpoint.
        return name_or_path, None, os.path.basename(name_or_path.rstrip("/\\"))

    if name_or_path not in _NAMES:
        raise SystemExit(
            "unknown model %r; use one of %s or a local path" % (name_or_path, _NAMES)
        )

    repo, subfolder = DEFAULT_MODELS[name_or_path]

    import huggingface_hub

    prefix = (subfolder + "/") if subfolder else ""
    kw = {
        "token": token or os.environ.get("HF_TOKEN"),
        "revision": revision,
        "allow_patterns": [
            prefix + n for n in (
                "rl_agent_config.json",
                "model.safetensors",
                "tokenizer/*",
                "encoder/*",
            )
        ],
    }
    print("  downloading %s/%s from HuggingFace Hub..." % (repo, subfolder or "(root)"))
    model_dir = huggingface_hub.snapshot_download(repo, **kw)
    return model_dir, subfolder, name_or_path


# ─── fixture cases ──────────────────────────────────────────────────────────

# Two synthetic test cases, chosen to cover noul (2 markers) and a mixed
# multi-question batch (max 5 markers).  These are identical across all
# checkpoints: the state and question texts are the same; only the tokenised
# lengths differ.

_NOUL_STATE = (
    "Subject: Double charged on invoice 88213\n\n"
    "Hi, I was billed twice this month for the Pro plan and the second "
    "charge has not been refunded. This is the third time I have written in."
)
_NOUL_QUESTIONS = {
    "churn_risk": {
        "t": "noul",
        "ins": "The customer threatens to cancel the account.",
        "crit": None,
    }
}

_TRIAGE_STATE = _NOUL_STATE
_TRIAGE_QUESTIONS = {
    "department": {
        "t": "choice",
        "ins": "Which department should handle this?",
        "crit": {
            "billing": "invoices, payments, double charges, refunds",
            "technical": "bugs, outages, integration errors",
            "account": "logins, seats, plan changes",
            "escalation": "manager required, repeat issue, SLA breach",
            "retention": "cancellation threats, churn risk",
        },
    },
    "urgency": {
        "t": "score",
        "ins": "How urgent is this request?",
        "crit": ["not urgent", "this week", "today", "drop everything"],
    },
    "churn_risk": {
        "t": "noul",
        "ins": "The customer threatens to cancel the account.",
        "crit": None,
    },
    "needs_human": {
        "t": "noul",
        "ins": "This requires a human agent rather than an automated reply.",
        "crit": {"false": "a template reply resolves it", "true": "a person must intervene"},
    },
}


def _build_fixture_group(tok, questions, state, max_len, head_max_len, pad_id):
    """Collate one fixture group and return the numpy arrays."""
    import numpy as np

    items = []
    for qid, q in questions.items():
        seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
        items.append({
            "ids": seq,
            "markers": markers,
            "qtype": QTYPES[q["t"]],
        })
    batch = collate_items([items], pad_id)
    return {
        "input_ids": batch["input_ids"].numpy().astype(np.int64),
        "attention_mask": batch["attention_mask"].numpy().astype(np.int64),
        "marker_pos": batch["marker_pos"].numpy().astype(np.int64),
        "marker_mask": batch["marker_mask"].numpy(),           # bool
        "qtype": batch["qtype"].numpy().astype(np.int64),
    }


def _time_forward(model, inputs_pt, n=5):
    """Return n wall-clock timings (seconds) for a forward pass, discarding the first."""
    import torch
    times = []
    with torch.no_grad():
        for _ in range(n + 1):
            t0 = time.perf_counter()
            model(*inputs_pt)
            times.append(time.perf_counter() - t0)
    return times[1:]    # skip first (warm-up)


def generate_fixtures(model, tok, cfg, out_dir):
    """Save PyTorch reference outputs as fixtures.npz."""
    import numpy as np
    import torch

    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)
    pad_id = tok.pad_token_id

    groups = {
        "single_noul": _build_fixture_group(
            tok, _NOUL_QUESTIONS, _NOUL_STATE, max_len, head_max_len, pad_id
        ),
        "triage_4q": _build_fixture_group(
            tok, _TRIAGE_QUESTIONS, _TRIAGE_STATE, max_len, head_max_len, pad_id
        ),
    }

    arrays = {}
    model.eval()
    with torch.no_grad():
        for g, tensors in groups.items():
            inp_pt = [
                torch.from_numpy(tensors["input_ids"]),
                torch.from_numpy(tensors["attention_mask"]),
                torch.from_numpy(tensors["marker_pos"]),
                torch.from_numpy(tensors["marker_mask"]),
                torch.from_numpy(tensors["qtype"]),
            ]
            logits, act_logits = model(*inp_pt)
            arrays["%s/input_ids" % g] = tensors["input_ids"]
            arrays["%s/attention_mask" % g] = tensors["attention_mask"]
            arrays["%s/marker_pos" % g] = tensors["marker_pos"]
            arrays["%s/marker_mask" % g] = tensors["marker_mask"]
            arrays["%s/qtype" % g] = tensors["qtype"]
            arrays["%s/ref_logits" % g] = logits.float().cpu().numpy()
            arrays["%s/ref_act" % g] = act_logits.float().cpu().numpy()
            timings = _time_forward(model, inp_pt)
            arrays["%s/torch_ms" % g] = np.array(
                [t * 1000.0 for t in timings], dtype=np.float64
            )

    path = os.path.join(out_dir, "fixtures.npz")
    np.savez(path, **arrays)

    for g in groups:
        shape_ids = arrays["%s/input_ids" % g].shape
        shape_mp = arrays["%s/marker_pos" % g].shape
        print(
            "  fixture %-14s  input_ids=%s  marker_pos=%s" % (g, shape_ids, shape_mp)
        )
    print("  wrote %s" % path)


# ─── ONNX export ────────────────────────────────────────────────────────────

def _dummy_inputs():
    """Dummy inputs for the ONNX export trace.

    batch=2 (not 1): using batch=1 causes the dynamo exporter to specialise
    (fix) the batch axis as a static 1 even when batch is listed in
    dynamic_shapes.  batch=2 forces a symbolic batch node into the graph.
    seq=32, markers=2 are near-minimum values; the Dim constraints extend them
    to the full legal range.
    """
    import torch

    B, S, M = 2, 32, 2
    return (
        torch.zeros(B, S, dtype=torch.long),
        torch.ones(B, S, dtype=torch.long),
        torch.zeros(B, M, dtype=torch.long),
        torch.ones(B, M, dtype=torch.bool),
        torch.zeros(B, dtype=torch.long),
    )


def export_onnx(model, out_dir: str, opset: int = 18):
    """Export model to model.onnx + model.onnx.data in out_dir.

    Uses torch.onnx.export with dynamo=True (the new torch.export-based path,
    default from PyTorch 2.9+).  The dynamo exporter inserts Shape/Concat nodes
    to compute tensor shapes symbolically at runtime, giving fully dynamic axes
    for batch, seq, and markers -- matching the pre-built artifacts exactly.

    The legacy TorchScript tracer (dynamo=False) bakes concrete shapes from the
    dummy inputs, so e.g. the ModernBERT sliding-window attention would always
    expect the traced sequence length.
    """
    import torch
    from torch.export.dynamic_shapes import Dim

    model.eval()
    dummy = _dummy_inputs()

    # Declare all three free dimensions: batch, seq, markers.
    batch = Dim("batch", min=1, max=1024)
    seq = Dim("seq", min=1, max=8192)
    markers = Dim("markers", min=2, max=512)   # min=2: TopK(k=2) constraint

    dynamic_shapes = (
        {0: batch, 1: seq},      # input_ids
        {0: batch, 1: seq},      # attention_mask
        {0: batch, 1: markers},  # marker_pos
        {0: batch, 1: markers},  # marker_mask
        {0: batch},              # qtype
    )

    # The dynamo exporter prints unicode emoji (✅, etc.) which crashes on
    # Windows consoles with a non-UTF-8 code page.  Reconfigure stdout/stderr to
    # UTF-8 with replacement for the duration of the export.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, io.UnsupportedOperation):
            pass

    with torch.no_grad():
        prog = torch.onnx.export(
            model,
            dummy,
            f=None,
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "act_logits"],
            dynamic_shapes=dynamic_shapes,
            opset_version=opset,
            dynamo=True,
            external_data=False,   # we handle external data ourselves below
            optimize=True,
            report=False,
        )

    if prog is None:
        raise RuntimeError("torch.onnx.export returned None -- export failed")

    import onnx

    # Save to a temp file first, then re-save with our specific external-data layout.
    # For large models (≥ ~1.6 GB weights), prog.save may create an external-data
    # sidecar alongside the temp file even when external_data=False.  Clean up both.
    tmp_path = os.path.join(out_dir, "_model_tmp.onnx")
    tmp_data_path = tmp_path + ".data"
    try:
        prog.save(tmp_path, external_data=False)
        proto = onnx.load(tmp_path)
    finally:
        for _p in (tmp_path, tmp_data_path):
            if os.path.exists(_p):
                os.remove(_p)

    # onnx.save_model opens the external-data sidecar in append mode, so if a
    # previous export left a stale file it would be silently inflated.  Always
    # remove both output files before writing.
    final_path = os.path.join(out_dir, "model.onnx")
    data_path = os.path.join(out_dir, "model.onnx.data")
    for p in (final_path, data_path):
        if os.path.exists(p):
            os.remove(p)

    # Persist with large initializers (weight matrices etc.) in one sidecar named
    # exactly "model.onnx.data".  Tensors ≤ 1 KB stay inline so ONNX Runtime can
    # perform shape inference without having to parse external-data files (which it
    # cannot do for shape-inference purposes).
    onnx.save_model(
        proto,
        final_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",   # must be exactly this name
        size_threshold=1024,          # keep tensors ≤ 1 KB inline
        convert_attribute=False,
    )

    size_graph = os.path.getsize(final_path)
    size_data = os.path.getsize(os.path.join(out_dir, "model.onnx.data"))
    print(
        "  model.onnx %.1f MB   model.onnx.data %.1f MB"
        % (size_graph / 1e6, size_data / 1e6)
    )


# ─── tokenizer copy ─────────────────────────────────────────────────────────

def copy_tokenizer(model_dir: str, out_dir: str):
    """Copy tokenizer/ from model_dir to out_dir/tokenizer/."""
    src_tok = os.path.join(model_dir, "tokenizer")
    dst_tok = os.path.join(out_dir, "tokenizer")

    if not os.path.isdir(src_tok):
        raise FileNotFoundError("tokenizer directory not found: %s" % src_tok)

    os.makedirs(dst_tok, exist_ok=True)
    n = 0
    for fname in os.listdir(src_tok):
        # Copy the canonical tokenizer files; skip any pre-existing nopost cache
        # (the .NET SDK regenerates them at load time from tokenizer.json).
        if fname.endswith(".nopost.json") or fname.endswith(".nopost.json.meta"):
            continue
        shutil.copy2(os.path.join(src_tok, fname), os.path.join(dst_tok, fname))
        n += 1
    print("  tokenizer/  (%d files)" % n)

    # Apply the same tokenizer_config fix that Agent.__init__ does, so the directory
    # works with AutoTokenizer.from_pretrained() without any environment correction.
    _fix_tokenizer_config(out_dir)


# ─── ONNX parity verification ────────────────────────────────────────────────

def _run_onnx(sess, feeds_np):
    """Run one ONNX inference and return (logits, act_logits) as numpy arrays."""
    input_meta = {i.name: i.type for i in sess.get_inputs()}
    _NP = {
        "tensor(int64)": __import__("numpy").int64,
        "tensor(int32)": __import__("numpy").int32,
        "tensor(bool)": __import__("numpy").bool_,
        "tensor(float)": __import__("numpy").float32,
    }
    feeds = {n: feeds_np[n].astype(_NP[input_meta[n]]) for n in input_meta}
    return sess.run(["logits", "act_logits"], feeds)


def verify(out_dir: str, cfg: dict, seq_lens=None, batch_sizes=None, marker_counts=None):
    """Verify the exported ONNX matches PyTorch across a range of tensor shapes.

    Feeds synthetic zero-valued tensors so no text or tokenizer is needed.
    Reports max absolute difference on logits and act_logits for every shape.
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    if seq_lens is None:
        seq_lens = [16, 32, 64, 128, 256, cfg.get("max_len", 512)]
    if batch_sizes is None:
        batch_sizes = [1, 3]
    if marker_counts is None:
        marker_counts = [2, 5, 9]

    # Load the exported model weights for PyTorch re-evaluation.
    # Re-use the fixtures.npz rather than re-downloading to avoid an extra
    # snapshot_download; the fixtures were written from the same weights.
    fixtures_path = os.path.join(out_dir, "fixtures.npz")
    if not os.path.exists(fixtures_path):
        raise SystemExit("fixtures.npz not found in %s; run without --verify first" % out_dir)

    sess = ort.InferenceSession(
        os.path.join(out_dir, "model.onnx"),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )

    # --- self-check against fixtures ----------------------------------------
    z = np.load(fixtures_path)
    groups = sorted({k.split("/")[0] for k in z.files})
    print("  self-check against fixtures.npz:")
    for g in groups:
        need = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
        if not all("%s/%s" % (g, n) in z.files for n in need):
            continue
        feeds = {n: z["%s/%s" % (g, n)] for n in need}
        logits, act = _run_onnx(sess, feeds)
        d_logits = float(np.abs(logits - z["%s/ref_logits" % g]).max())
        d_act = float(np.abs(act - z["%s/ref_act" % g]).max())
        ok = "OK" if d_logits <= 1e-3 and d_act <= 1e-2 else "FAIL"
        print(
            "    %-14s  logits %.3e  act_logits %.3e  %s" % (g, d_logits, d_act, ok)
        )
        if ok == "FAIL":
            raise SystemExit(
                "ONNX parity check failed for %r (logits %.3e > 1e-3 or "
                "act_logits %.3e > 1e-2)" % (g, d_logits, d_act)
            )

    # --- shape sweep --------------------------------------------------------
    print(
        "  shape sweep: %d seq_lens × %d batch_sizes × %d marker_counts = %d combos"
        % (len(seq_lens), len(batch_sizes), len(marker_counts),
           len(seq_lens) * len(batch_sizes) * len(marker_counts))
    )

    # We need PyTorch weights for the shape sweep comparison.  Load from the
    # original checkpoint directory if it is still cached locally.
    # If not available, skip PyTorch comparison and only check that the ONNX
    # session runs without error and produces the expected output shapes.
    pt_model = _load_pt_model_if_cached(cfg, out_dir)

    max_d_logits = max_d_act = 0.0
    n_ok = n_skip = 0
    max_len = cfg.get("max_len", 512)

    for seq in seq_lens:
        if seq > max_len:
            continue
        for B in batch_sizes:
            for M in marker_counts:
                feeds = _make_synthetic_feeds(B, seq, M)
                logits_onnx, act_onnx = _run_onnx(sess, feeds)

                # Shape assertions.
                assert logits_onnx.shape == (B, M), \
                    "logits shape %s != (%d, %d)" % (logits_onnx.shape, B, M)
                assert act_onnx.shape[0] == B, \
                    "act_logits batch %d != %d" % (act_onnx.shape[0], B)

                if pt_model is not None:
                    inp = [
                        torch.from_numpy(feeds["input_ids"]),
                        torch.from_numpy(feeds["attention_mask"]),
                        torch.from_numpy(feeds["marker_pos"]),
                        torch.from_numpy(feeds["marker_mask"]),
                        torch.from_numpy(feeds["qtype"]),
                    ]
                    with torch.no_grad():
                        logits_pt, act_pt = pt_model(*inp)
                    logits_pt = logits_pt.float().numpy()
                    act_pt = act_pt.float().numpy()

                    d_l = float(abs(logits_onnx - logits_pt).max())
                    d_a = float(abs(act_onnx - act_pt).max())
                    max_d_logits = max(max_d_logits, d_l)
                    max_d_act = max(max_d_act, d_a)
                    n_ok += 1
                else:
                    n_skip += 1

    if pt_model is not None:
        print(
            "  shape sweep done: %d combos, max diff  logits %.3e  act_logits %.3e"
            % (n_ok, max_d_logits, max_d_act)
        )
    else:
        print(
            "  shape sweep done: %d combos (PyTorch not available, "
            "shape/session checks only)" % (n_ok + n_skip)
        )


def _make_synthetic_feeds(B, seq, M):
    """Synthetic zero inputs of the given shape for the shape sweep."""
    import numpy as np

    return {
        "input_ids":      np.zeros((B, seq), dtype=np.int64),
        "attention_mask": np.ones((B, seq), dtype=np.int64),
        "marker_pos":     np.zeros((B, M), dtype=np.int64),
        "marker_mask":    np.ones((B, M), dtype=np.bool_),
        "qtype":          np.zeros(B, dtype=np.int64),
    }


def _load_pt_model_if_cached(cfg, out_dir):
    """Return a loaded PyTorch model if weights are available locally, else None."""
    from safetensors.torch import load_file

    # The export was done from a snapshot_download() cache; try to locate the
    # safetensors file there.  If it is gone (evicted or never downloaded because
    # the user supplied a local path that has since been removed), skip PT compare.
    cache_candidates = []
    # Check if there is a safetensors file alongside the onnx output (for local-path exports).
    out_parent = os.path.dirname(out_dir)
    for candidate in [
        os.path.join(out_parent, "model.safetensors"),
        os.path.join(out_dir, "model.safetensors"),   # shouldn't normally exist here
    ]:
        if os.path.exists(candidate):
            cache_candidates.append(candidate)

    # Also check the HF cache for a known pattern.
    try:
        import huggingface_hub
        cache_info = huggingface_hub.scan_cache_dir()
        for repo in cache_info.repos:
            for rev in repo.revisions:
                for f in rev.files:
                    if f.file_name == "model.safetensors" and os.path.exists(f.file_path):
                        cache_candidates.append(str(f.file_path))
    except Exception:
        pass

    # Find a safetensors that is compatible with this cfg's architecture.
    for sf_path in cache_candidates:
        try:
            model = build_model(cfg)
            weights = load_file(sf_path)
            model.load_state_dict(weights, strict=True)
            try:
                model.encoder.config.reference_compile = False
            except Exception:
                pass
            model.eval()
            return model
        except Exception:
            continue

    return None


# ─── ONNX graph signature check ─────────────────────────────────────────────

def check_graph_signature(out_dir: str, reference_onnx: str):
    """Compare input/output names, dtypes, and dynamic-axis names to a reference graph."""
    import onnx

    def sig(path):
        m = onnx.load(path, load_external_data=False)
        return {
            "opset": m.opset_import[0].version,
            "inputs": {
                inp.name: {
                    "dtype": inp.type.tensor_type.elem_type,
                    "dims": [
                        d.dim_param if d.dim_param else d.dim_value
                        for d in inp.type.tensor_type.shape.dim
                    ],
                }
                for inp in m.graph.input
            },
            "outputs": {
                out.name: {
                    "dtype": out.type.tensor_type.elem_type,
                    "dims": [
                        d.dim_param if d.dim_param else d.dim_value
                        for d in out.type.tensor_type.shape.dim
                    ],
                }
                for out in m.graph.output
            },
        }

    new_sig = sig(os.path.join(out_dir, "model.onnx"))
    ref_sig = sig(reference_onnx)

    ok = True
    # Opset
    if new_sig["opset"] != ref_sig["opset"]:
        print("  WARN  opset %d != reference %d" % (new_sig["opset"], ref_sig["opset"]))
        ok = False
    # Inputs
    for name, ref_inp in ref_sig["inputs"].items():
        if name not in new_sig["inputs"]:
            print("  WARN  input %r missing" % name)
            ok = False
            continue
        new_inp = new_sig["inputs"][name]
        if new_inp["dtype"] != ref_inp["dtype"]:
            print("  WARN  input %r dtype %d != %d" % (name, new_inp["dtype"], ref_inp["dtype"]))
            ok = False
        if new_inp["dims"] != ref_inp["dims"]:
            print("  WARN  input %r dims %s != %s" % (name, new_inp["dims"], ref_inp["dims"]))
            ok = False
    # Outputs
    for name, ref_out in ref_sig["outputs"].items():
        if name not in new_sig["outputs"]:
            print("  WARN  output %r missing" % name)
            ok = False
            continue
        new_out = new_sig["outputs"][name]
        if new_out["dtype"] != ref_out["dtype"]:
            print("  WARN  output %r dtype %d != %d" % (name, new_out["dtype"], ref_out["dtype"]))
            ok = False
        if new_out["dims"] != ref_out["dims"]:
            print("  WARN  output %r dims %s != %s" % (name, new_out["dims"], ref_out["dims"]))
            ok = False
    if ok:
        print("  graph signature matches reference (%s)" % os.path.basename(os.path.dirname(reference_onnx)))
    return ok


# ─── main per-model export ───────────────────────────────────────────────────

def export_one(name_or_path: str, out_root: str, token: str, opset: int, do_verify: bool,
               force: bool = False, revision: str = None):
    """Download (if needed), export, and verify one checkpoint.

    Raises RuntimeError if <out_root>/<name>/model.onnx already exists and
    force is False.  Pass force=True (--force on the CLI) to overwrite.
    The check happens before any download so nothing is wasted on a refusal.
    """
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    import torch

    print("\n=== %s ===" % name_or_path)
    t0 = time.perf_counter()

    # 1. Compute out_dir early so we can detect an existing export BEFORE
    #    downloading or building the model (a slow, expensive step).
    if os.path.exists(name_or_path):
        label_hint = os.path.basename(name_or_path.rstrip("/\\"))
    else:
        label_hint = name_or_path  # e.g., "multilingual"
    out_dir = os.path.join(out_root, label_hint)
    final_onnx = os.path.join(out_dir, "model.onnx")
    if os.path.exists(final_onnx) and not force:
        raise RuntimeError(
            "%s already exists.  Pass --force to overwrite." % final_onnx
        )

    # 2. Resolve model directory (may trigger a HuggingFace download).
    model_dir, subfolder, label = _resolve_spec(name_or_path, token, revision)
    if subfolder:
        model_dir = os.path.join(model_dir, subfolder)

    _fix_tokenizer_config(model_dir)

    cfg_path = os.path.join(model_dir, "rl_agent_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("rl_agent_config.json not found in %s" % model_dir)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    # 3. Prepare output directory.
    os.makedirs(out_dir, exist_ok=True)

    # 4. Build model and load weights.
    print("  building model from config...")
    enc_dir = os.path.join(model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)
    weights_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        raise FileNotFoundError("model.safetensors not found in %s" % model_dir)
    weights = load_file(weights_path)
    model.load_state_dict(weights, strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    # Switch the encoder to eager attention for ONNX export.  The SDPA path
    # (default for ModernBERT) calls torch.nn.functional.scaled_dot_product_attention
    # and the sliding-window attention utility in masking_utils.py, both of which
    # bake sequence-length shapes into the traced graph.  The eager path uses plain
    # matmul and produces fully dynamic ONNX nodes.  The weights are identical in
    # both paths; we are only changing the forward dispatch.
    try:
        model.encoder.config._attn_implementation = "eager"
    except Exception:
        pass
    model.eval()
    print("  model loaded (%.1f M parameters)" % (sum(p.numel() for p in model.parameters()) / 1e6))

    # 5. Load tokenizer.
    tok_dir = os.path.join(model_dir, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir)

    # 6. Copy config and tokenizer.
    shutil.copy2(cfg_path, os.path.join(out_dir, "rl_agent_config.json"))
    print("  rl_agent_config.json copied")
    copy_tokenizer(model_dir, out_dir)

    # 7. Generate PyTorch reference fixtures before export (they are independent).
    print("  generating fixtures...")
    with torch.no_grad():
        generate_fixtures(model, tok, cfg, out_dir)

    # 8. Export to ONNX.
    print("  exporting to ONNX (opset=%d)..." % opset)
    export_onnx(model, out_dir, opset=opset)

    # 9. Verify if requested.
    if do_verify:
        print("  verifying...")
        verify(out_dir, cfg)

    elapsed = time.perf_counter() - t0
    print("  done in %.1f s -> %s" % (elapsed, out_dir))
    return out_dir


# ─── graph signature check against reference ────────────────────────────────

def _check_against_reference(out_dir: str, label: str):
    """If a pre-built reference exists in onnx/, compare signatures."""
    ref_candidates = {
        "english": os.path.join(REPO, "onnx", "english", "model.onnx"),
        "multilingual": os.path.join(REPO, "onnx", "multilingual", "model.onnx"),
    }
    ref_path = ref_candidates.get(label)
    if ref_path and os.path.exists(ref_path):
        try:
            check_graph_signature(out_dir, ref_path)
        except Exception as e:
            print("  signature check error: %s" % e)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "local_path",
        nargs="?",
        help="Local checkpoint directory (alternative to --model).",
    )
    ap.add_argument(
        "--model",
        choices=_NAMES + ["all"],
        default=None,
        help="Which checkpoint to export: english, multilingual, typed-decisions, or all.",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(REPO, "onnx"),
        help="Output root directory (default: %(default)s).  "
             "Each model is written to <out>/<model-name>/.",
    )
    ap.add_argument(
        "--token",
        default=None,
        metavar="HF_TOKEN",
        help="HuggingFace API token (default: HF_TOKEN env var).",
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: %(default)s).",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="After export, reload in ONNX Runtime and compare to PyTorch "
             "across a range of tensor shapes.",
    )
    ap.add_argument(
        "--revision",
        default=None,
        metavar="SHA",
        help="Hugging Face commit (or branch/tag) to download the checkpoint from "
             "(default: the repo's main branch).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing export.  By default the script refuses to "
             "overwrite <out>/<model>/model.onnx if it already exists.",
    )
    args = ap.parse_args()

    if args.local_path and args.model:
        ap.error("Specify either a local path or --model, not both.")
    if not args.local_path and not args.model:
        ap.error("Specify --model {english,multilingual,typed-decisions,all} or a local path.")

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    targets = (
        [args.local_path]
        if args.local_path
        else (_NAMES if args.model == "all" else [args.model])
    )

    multi_model = len(targets) > 1
    out_dirs = []
    failures = []
    for target in targets:
        try:
            od = export_one(
                target,
                out_root=out_root,
                token=args.token,
                opset=args.opset,
                do_verify=args.verify,
                force=args.force,
                revision=args.revision,
            )
            out_dirs.append((target, od))
            # Signature check against the pre-built reference (if any).
            label = os.path.basename(od)
            _check_against_reference(od, label)
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit) as e:
            if multi_model:
                msg = str(e).strip()
                print("\n[%s] skipped: %s" % (target, msg or type(e).__name__))
                failures.append((target, msg))
            else:
                raise

    print("\nexport complete:")
    for target, od in out_dirs:
        files = [f for f in os.listdir(od) if not os.path.isdir(os.path.join(od, f))]
        print("  %-20s -> %s  (%d files)" % (target, od, len(files)))
    if failures:
        print("\nskipped (%d):" % len(failures))
        for target, msg in failures:
            print("  %-20s  %s" % (target, msg))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
