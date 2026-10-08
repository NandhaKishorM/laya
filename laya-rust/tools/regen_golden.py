"""Regenerate Rust parity fixtures from the Python laya package at this checkout.

Uses the same pinned checkpoint and exporters as the .NET parity lane. Exports may
be reused only when complete and stamped with the current exporter/model inputs;
goldens are ALWAYS rewritten, never read as an input or cached.

    python laya-rust/tools/regen_golden.py
    python laya-rust/tools/regen_golden.py --checkpoint routing
    python laya-rust/tools/regen_golden.py --checkpoint repr
    python laya-rust/tools/regen_golden.py --checkpoint english --artifacts-root D:/laya-artifacts

Install requirements-regen.txt and CPU torch first. The tests discover fused
exports via LAYA_ONNX_ROOT and split exports via LAYA_ONNX_SPLIT_ROOT.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

from google.protobuf.message import DecodeError

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, REPO)

HF_REPO = "convaiinnovations/laya"
HF_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
CHECKPOINTS = ["english", "multilingual", "typed-decisions"]
EXPORT_FUSED = os.path.join(REPO, "laya-dotnet", "tools", "export_onnx.py")
EXPORT_SPLIT = os.path.join(REPO, "laya-ts", "scripts", "export_onnx.py")
EXPORT_INPUTS = [
    EXPORT_FUSED, EXPORT_SPLIT, os.path.join(TOOLS, "requirements-regen.txt"),
    os.path.join(REPO, "laya", "common.py"),
]
STAMP = ".regen-stamp.json"
FUSED_FILES = [
    "model.onnx", "model.onnx.data", "rl_agent_config.json",
    os.path.join("tokenizer", "tokenizer.json"), "fixtures.npz",
]
SPLIT_FILES = ["encoder.onnx", "head.onnx", "rl_agent_config.json", "tokenizer.json"]


def inputs_hash(revision):
    h = hashlib.sha256(revision.encode("utf-8"))
    for path in EXPORT_INPUTS:
        with open(path, "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def is_valid(path, files, stamp):
    """Reject stale exports, missing graphs and truncated external-data sidecars."""
    try:
        if not all(os.path.getsize(os.path.join(path, f)) > 0 for f in files):
            return False
        with open(os.path.join(path, STAMP), encoding="utf-8") as f:
            if json.load(f).get("inputs_sha256") != stamp:
                return False
        import onnx

        for file in files:
            if not file.endswith(".onnx"):
                continue
            model = onnx.load(os.path.join(path, file), load_external_data=False)
            for tensor in model.graph.initializer:
                if tensor.data_location != onnx.TensorProto.EXTERNAL:
                    continue
                data = {entry.key: entry.value for entry in tensor.external_data}
                size = os.path.getsize(os.path.join(path, data["location"]))
                if size <= 0 or size < int(data.get("offset", 0)) + int(data.get("length", 0)):
                    return False
        return True
    except (OSError, ValueError, KeyError, DecodeError):
        return False


def run(cmd, env=None):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


def snapshot(name, revision):
    from huggingface_hub import snapshot_download
    from laya.router import DEFAULT_MODELS

    repo, subfolder = DEFAULT_MODELS[name]
    if repo != HF_REPO:
        raise SystemExit("checkpoint repository changed; update HF_REPO/HF_REVISION")
    prefix = subfolder + "/" if subfolder else ""
    root = snapshot_download(repo, revision=revision, allow_patterns=[
        prefix + n for n in ("rl_agent_config.json", "model.safetensors", "config.json",
                            "tokenizer.json", "tokenizer/*", "encoder/*")
    ])
    return os.path.join(root, subfolder) if subfolder else root


def publish(partial, final, stamp):
    with open(os.path.join(partial, STAMP), "w", encoding="utf-8") as f:
        json.dump({"inputs_sha256": stamp}, f)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.replace(partial, final)


def regen_checkpoint(name, root, revision, force):
    stamp = inputs_hash(revision)
    fused = os.path.join(root, "onnx", name)
    split = os.path.join(root, "onnx-split", name)
    need_fused = force or not is_valid(fused, FUSED_FILES, stamp)
    need_split = force or not is_valid(split, SPLIT_FILES, stamp)
    if need_fused or need_split:
        model_dir = snapshot(name, revision)
        env = dict(os.environ, HF_HUB_OFFLINE="1",
                   PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
        if need_fused:
            staging = os.path.join(root, ".partial-rust", "onnx")
            shutil.rmtree(staging, ignore_errors=True)
            run([sys.executable, EXPORT_FUSED, "--model", name, "--out", staging,
                 "--revision", revision], env)
            publish(os.path.join(staging, name), fused, stamp)
        if need_split:
            staging = os.path.join(root, ".partial-rust", "onnx-split", name)
            shutil.rmtree(staging, ignore_errors=True)
            run([sys.executable, EXPORT_SPLIT, "--model-dir", model_dir,
                 "--out-dir", staging], env)
            publish(staging, split, stamp)
    run([sys.executable, os.path.join(TOOLS, "dump_golden.py"),
         "--checkpoint", name, "--onnx", fused])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", choices=CHECKPOINTS + ["routing", "repr", "all"])
    ap.add_argument("--artifacts-root", default=REPO)
    ap.add_argument("--revision", default=HF_REVISION)
    ap.add_argument("--force-export", action="store_true")
    args = ap.parse_args()
    selected = args.checkpoint or ["all"]
    if "all" in selected:
        selected = CHECKPOINTS + ["routing"]
    root = os.path.abspath(args.artifacts_root)
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        for name in CHECKPOINTS:
            if name in selected:
                regen_checkpoint(name, root, args.revision, args.force_export)
        if "routing" in selected:
            run([sys.executable, os.path.join(TOOLS, "dump_routing_golden.py"), "--force"])
        if "repr" in selected:
            run([sys.executable, os.path.join(TOOLS, "repr_cases.py")])
    except subprocess.CalledProcessError as e:
        print("FAILED: %s" % " ".join(e.cmd), file=sys.stderr)
        return e.returncode or 1
    finally:
        shutil.rmtree(os.path.join(root, ".partial-rust"), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
