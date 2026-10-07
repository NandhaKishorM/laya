"""Regenerate the .NET golden fixtures from the shipping Python code, end to end.

This is the single entry point the `.NET` CI lane runs (`.github/workflows/dotnet.yml`) and the
one command a contributor needs after changing anything in `laya/` that the goldens capture
(sequence building, option rendering, calibration, language detection, ...). It only chains the
existing tools; none of their logic is repeated here:

    1. downloads the checkpoint at a *pinned* Hugging Face revision (`HF_REVISION` below),
    2. exports the fused layout   -> <artifacts-root>/onnx/<checkpoint>/        (tools/export_onnx.py)
    3. exports the split layout   -> <artifacts-root>/onnx-split/<checkpoint>/  (laya-ts/scripts/export_onnx.py)
    4. records tests/Laya.Tests/golden/<checkpoint>/ from the real `laya.Agent`  (tools/dump_golden.py)
    5. records tests/Laya.Tests/golden/routing/                                  (tools/dump_routing_golden.py)

Exports are the slow part, so an artifact directory that is already present, complete and was
produced from the same inputs (revision + exporter scripts + `laya/common.py`, which holds the
model definition) is reused. That is what lets CI cache `<artifacts-root>/onnx*` between runs.
The goldens themselves are always rewritten.

The default artifacts root is the repository root, which is where the test project looks for
`onnx/<checkpoint>` and `onnx-split/<checkpoint>` on its own (it walks up from the test binary).
Point `--artifacts-root` elsewhere and set `LAYA_ONNX_ROOT=<root>/onnx` for the fused layout; the
split layout has no environment variable, so it is only found at `<some ancestor>/onnx-split`.

Usage:
    # Everything: three checkpoints, split layouts and the routing goldens
    python laya-dotnet/tools/regen_golden.py

    # One checkpoint, artifacts somewhere with room (each checkpoint is ~1-3 GB, fused + split)
    python laya-dotnet/tools/regen_golden.py --checkpoint english --artifacts-root D:/laya-artifacts

    # Only the model-free routing goldens
    python laya-dotnet/tools/regen_golden.py --checkpoint routing

    # Re-export even when a valid artifact directory exists
    python laya-dotnet/tools/regen_golden.py --checkpoint multilingual --force-export

Needs the packages in `tools/requirements-regen.txt` plus a CPU build of torch. Exits non-zero
as soon as any step fails.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from google.protobuf.message import DecodeError

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, REPO)

# The Hugging Face commit every checkpoint is exported from. The weights, encoder configs and
# tokenizers are byte-identical from 458d7563 (the commit that bundled the typed-decisions
# checkpoint) through this one; later commits only touch the model card. Bump it deliberately,
# then verify the regenerated goldens against the SDK.
HF_REPO = "convaiinnovations/laya"
HF_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

CHECKPOINTS = ["english", "multilingual", "typed-decisions"]

GOLDEN_ROOT = os.path.join(REPO, "laya-dotnet", "tests", "Laya.Tests", "golden")
EXPORT_FUSED = os.path.join(TOOLS, "export_onnx.py")
EXPORT_SPLIT = os.path.join(REPO, "laya-ts", "scripts", "export_onnx.py")
DUMP_GOLDEN = os.path.join(TOOLS, "dump_golden.py")
DUMP_ROUTING = os.path.join(TOOLS, "dump_routing_golden.py")

# Everything that decides what an export looks like. Its hash is stored beside each exported
# directory; a mismatch (or a missing stamp) means the export is stale and is redone.
EXPORT_INPUTS = [
    EXPORT_FUSED,
    EXPORT_SPLIT,
    os.path.join(TOOLS, "requirements-regen.txt"),
    os.path.join(REPO, "laya", "common.py"),
]

STAMP = ".regen-stamp.json"
FUSED_FILES = [
    "model.onnx",
    "model.onnx.data",
    "rl_agent_config.json",
    os.path.join("tokenizer", "tokenizer.json"),
    "fixtures.npz",
]
SPLIT_FILES = ["encoder.onnx", "head.onnx", "rl_agent_config.json", "tokenizer.json"]


def inputs_hash(revision: str) -> str:
    h = hashlib.sha256(revision.encode("utf-8"))
    for path in EXPORT_INPUTS:
        with open(path, "rb") as f:
            # Normalise line endings so a CRLF checkout hashes like the LF one.
            h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def is_valid(path: str, files: list, stamp: str) -> bool:
    """The directory holds a complete export made from exactly these inputs."""
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
                sidecar = os.path.join(path, data["location"])
                size = os.path.getsize(sidecar)
                if size <= 0 or size < int(data.get("offset", 0)) + int(data.get("length", 0)):
                    return False
        return True
    except (OSError, ValueError, KeyError, DecodeError):
        return False


def run(cmd: list, env: dict = None):
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=REPO, env=env)
    print("  (%.1f s)" % (time.perf_counter() - t0), flush=True)


def snapshot(name: str, revision: str) -> str:
    """The pinned checkpoint directory in the Hugging Face cache (downloaded on first use)."""
    from huggingface_hub import snapshot_download

    from laya.router import DEFAULT_MODELS

    repo, subfolder = DEFAULT_MODELS[name]
    if repo != HF_REPO:
        raise SystemExit(
            "%s now lives in %r, but the pinned revision belongs to %r; update HF_REPO/HF_REVISION"
            % (name, repo, HF_REPO)
        )
    prefix = (subfolder + "/") if subfolder else ""
    # A superset of what the two exporters ask for, so each of them hits the cache.
    allow = [prefix + n for n in (
        "rl_agent_config.json", "model.safetensors", "config.json",
        "tokenizer.json", "tokenizer/*", "encoder/*",
    )]
    root = snapshot_download(repo, revision=revision, allow_patterns=allow)
    return os.path.join(root, subfolder) if subfolder else root


def publish(partial: str, final: str, stamp: str):
    """Stamp a finished export and move it into place, so an interrupted run leaves nothing half-valid."""
    with open(os.path.join(partial, STAMP), "w", encoding="utf-8") as f:
        json.dump({"inputs_sha256": stamp}, f)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.replace(partial, final)


def export_fused(name: str, root: str, revision: str, stamp: str, env: dict):
    final = os.path.join(root, "onnx", name)
    staging = os.path.join(root, ".partial", "onnx")
    shutil.rmtree(staging, ignore_errors=True)
    run([sys.executable, EXPORT_FUSED, "--model", name, "--out", staging, "--revision", revision], env=env)
    publish(os.path.join(staging, name), final, stamp)


def export_split(name: str, root: str, model_dir: str, stamp: str, env: dict):
    final = os.path.join(root, "onnx-split", name)
    staging = os.path.join(root, ".partial", "onnx-split", name)
    shutil.rmtree(staging, ignore_errors=True)
    run([sys.executable, EXPORT_SPLIT, "--model-dir", model_dir, "--out-dir", staging], env=env)
    publish(staging, final, stamp)


def regen_checkpoint(name: str, root: str, revision: str, force: bool):
    print("\n=== %s ===" % name)
    stamp = inputs_hash(revision)
    fused = os.path.join(root, "onnx", name)
    split = os.path.join(root, "onnx-split", name)
    need_fused = force or not is_valid(fused, FUSED_FILES, stamp)
    need_split = force or not is_valid(split, SPLIT_FILES, stamp)

    if need_fused or need_split:
        model_dir = snapshot(name, revision)
        # Everything the exporters need is in the cache now; going back to the network would
        # only be a way to pick up something that is not the pinned revision.
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
        if need_fused:
            export_fused(name, root, revision, stamp, env)
        if need_split:
            export_split(name, root, model_dir, stamp, env)
    else:
        print("fused and split exports are up to date; reusing %s and %s" % (fused, split))

    run([sys.executable, DUMP_GOLDEN, "--checkpoint", name, "--onnx", fused])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--checkpoint",
        action="append",
        choices=CHECKPOINTS + ["routing", "all"],
        metavar="{%s,routing,all}" % ",".join(CHECKPOINTS),
        help="What to regenerate; repeat to combine. 'routing' is the model-free routing/language/"
             "email/shortlist goldens, 'all' is every checkpoint plus routing (default: all).",
    )
    ap.add_argument(
        "--artifacts-root",
        default=REPO,
        help="Where onnx/<checkpoint> and onnx-split/<checkpoint> are written and reused "
             "(default: the repository root, which the tests find on their own).",
    )
    ap.add_argument(
        "--revision",
        default=HF_REVISION,
        metavar="SHA",
        help="Hugging Face commit to export from (default: %(default)s).",
    )
    ap.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export even when a valid artifact directory already exists.",
    )
    args = ap.parse_args()

    # The exporters print check marks; a Windows console code page cannot encode them.
    os.environ.setdefault("PYTHONUTF8", "1")
    selected = args.checkpoint or ["all"]
    if "all" in selected:
        selected = CHECKPOINTS + ["routing"]
    checkpoints = [c for c in CHECKPOINTS if c in selected]
    root = os.path.abspath(args.artifacts_root)

    t0 = time.perf_counter()
    try:
        for name in checkpoints:
            regen_checkpoint(name, root, args.revision, args.force_export)
        if "routing" in selected:
            print("\n=== routing ===")
            run([sys.executable, DUMP_ROUTING, "--force"])
    except subprocess.CalledProcessError as e:
        print("\nFAILED (exit %d): %s" % (e.returncode, " ".join(e.cmd)), file=sys.stderr)
        return e.returncode or 1
    finally:
        shutil.rmtree(os.path.join(root, ".partial"), ignore_errors=True)

    print("\nregenerated %s in %.0f s" % (", ".join(checkpoints + (["routing"] if "routing" in selected else [])), time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
