"""Verify the local checkpoints against the recorded manifest.

    .venv/bin/python verify/checkpoints.py [--models ./models] [--quiet]

Checks that every checkpoint in `verify/checkpoints.json` is present and that its weights hash
to the recorded sha256, so a truncated download or an upstream change is caught rather than
turning into quietly wrong answers. Exits non-zero on a missing file or a hash mismatch.

A mismatch is reported, not hidden: if the upstream hub revision has moved on, the weights are
legitimately different and the manifest needs re-recording (`--record`).
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "verify", "checkpoints.json")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    ap.add_argument("--record", action="store_true",
                    help="re-record sha256/size for the files on disk (use after a deliberate update)")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    problems, recorded = [], []
    for entry in manifest["checkpoints"]:
        path = os.path.join(args.models, entry["path"])
        if not os.path.exists(path):
            problems.append("%s: missing (%s) -- run ./setup_laya.sh" % (entry["name"], path))
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
        print("\nIf the upstream hub revision moved, re-record with: "
              ".venv/bin/python verify/checkpoints.py --record")
        return 1
    if not args.quiet:
        print("\n   all %d checkpoints match the manifest (hub revision %s)"
              % (len(manifest["checkpoints"]), manifest["revision"][:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
