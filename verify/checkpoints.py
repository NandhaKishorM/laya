"""Verify the local checkpoints against the recorded manifest.

    .venv/bin/python verify/checkpoints.py [--models ./models] [--quiet]
    .venv/bin/python verify/checkpoints.py --fetch        # download, then verify

Checks that every checkpoint in `verify/checkpoints.json` is present and that its weights hash
to the recorded sha256, so a truncated download or an upstream change is caught rather than
turning into quietly wrong answers. Exits non-zero on a missing file or a hash mismatch.

`--fetch` downloads each checkpoint at the manifest's pinned `revision`, so the hashes keep
matching when the Hub repo moves on; without it, nothing is downloaded. A file whose size does
not match the manifest is re-downloaded by `--fetch` (a truncated transfer is repaired, not
trusted), while a file that is present at the right size is left alone.

A mismatch is reported, not hidden: if the weights really did change, the manifest needs
re-recording (`--record`).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "verify", "checkpoints.json")

# The files a checkpoint needs to load: config, weights, tokenizer and encoder config.
RUNTIME_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_entry(entry, revision, models_dir):
    """Download one checkpoint at `revision` into the directory its manifest path names.

    A file that is already present is only trusted when its size matches the manifest. The
    point of `--fetch` is to end with a checkpoint the verify pass accepts, and a truncated
    download -- the case the module docstring promises this script catches -- used to be
    skipped by the bare existence check here, so the advice printed on failure ("fetch it
    with --fetch") could never repair it. A wrong-sized file is removed first, because
    `snapshot_download(local_dir=...)` resumes from its own bookkeeping and would otherwise
    keep the file it believes it already wrote.
    """
    from huggingface_hub import snapshot_download

    dest = os.path.join(models_dir, os.path.dirname(entry["path"]))
    present = os.path.join(dest, entry["file"])
    if os.path.exists(present):
        if os.path.getsize(present) == entry["bytes"]:
            return dest
        os.remove(present)
    prefix = entry["subfolder"] + "/" if entry["subfolder"] else ""
    snapshot_download(entry["repo"], revision=revision, local_dir=dest,
                      allow_patterns=[prefix + name for name in RUNTIME_FILES])
    if prefix:
        # `local_dir` keeps the repo-relative path, so a subfolder checkpoint arrives nested one
        # level down; flatten it so every checkpoint is <models>/<manifest dir>/model.safetensors.
        nested = os.path.join(dest, entry["subfolder"])
        for name in os.listdir(nested):
            shutil.move(os.path.join(nested, name), os.path.join(dest, name))
        os.rmdir(nested)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    ap.add_argument("--fetch", action="store_true",
                    help="download each checkpoint at the manifest's pinned revision first")
    ap.add_argument("--record", action="store_true",
                    help="re-record sha256/size for the files on disk (use after a deliberate update)")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    if args.fetch:
        os.makedirs(args.models, exist_ok=True)
        for entry in manifest["checkpoints"]:
            dest = fetch_entry(entry, manifest["revision"], args.models)
            if not args.quiet:
                print("   fetched %-16s -> %s" % (entry["name"], dest))

    problems, recorded = [], []
    truncated, rewritten = [], []
    for entry in manifest["checkpoints"]:
        path = os.path.join(args.models, entry["path"])
        if not os.path.exists(path):
            problems.append("%s: missing (%s) -- fetch it with "
                            ".venv/bin/python verify/checkpoints.py --fetch" % (entry["name"], path))
            continue
        size = os.path.getsize(path)
        digest = sha256(path)
        if args.record:
            entry["bytes"], entry["sha256"] = size, digest
            recorded.append(entry["name"])
            continue
        ok = size == entry["bytes"] and digest == entry["sha256"]
        if not ok:
            problems.append("%s: %s\n       expected %s (%d bytes)\n       found    %s (%d bytes)"
                            % (entry["name"], path, entry["sha256"], entry["bytes"], digest, size))
            # The two causes need opposite remedies: a wrong size is an incomplete download,
            # repaired by re-fetching, while identical bytes and a different hash is content
            # that changed under a pinned revision. The old advice told both cases to
            # `--record`, which for a truncated file means writing the truncation into the
            # manifest as the new truth.
            (truncated if size != entry["bytes"] else rewritten).append(entry["name"])
        elif not args.quiet:
            print("   OK   %-16s %-34s %8.1f MB  %s"
                  % (entry["name"], entry["encoder"] + " ctx=" + str(entry["context"]),
                     size / 1e6, digest[:16]))

    if args.record:
        json.dump(manifest, open(MANIFEST, "w"), indent=2)
        open(MANIFEST, "a").write("\n")
        print("re-recorded: %s" % ", ".join(recorded))
        return 0

    if problems:
        print("\n%d checkpoint problem(s):" % len(problems))
        for p in problems:
            print("   FAIL " + p)
        if truncated:
            print("\nA size mismatch means the file is incomplete or was replaced (%s):"
                  "\n   .venv/bin/python verify/checkpoints.py --fetch   "
                  "# re-downloads at the pinned revision" % ", ".join(truncated))
        if rewritten:
            print("\nIf the upstream hub revision really moved (%s), review the files and "
                  "re-record with:\n   .venv/bin/python verify/checkpoints.py --record"
                  % ", ".join(rewritten))
        return 1
    if not args.quiet:
        print("\n   all %d checkpoints match the manifest (hub revision %s)"
              % (len(manifest["checkpoints"]), manifest["revision"][:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
