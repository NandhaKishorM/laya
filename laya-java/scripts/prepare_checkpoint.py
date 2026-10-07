#!/usr/bin/env python3
"""Download a pinned laya checkpoint and export the fused ONNX graph the Java tests need.

Separated from `gen_fixtures.py` on purpose: the download and the export are the slow, cacheable
part of a CI run (minutes), while regenerating the fixtures from the Python code is the part that
must happen on every commit (seconds). Splitting them lets the workflow cache the first and never
cache the second.

Writes, under `laya-java/.work/` so the repository root stays clean and one `.gitignore` covers it
    laya-java/.work/checkpoints/<name>/        the checkpoint, as `Agent.open` expects it
    laya-java/.work/onnx/<name>/laya.onnx      the fused graph, as `LayaSession.open` expects it

Usage:
    python laya-java/scripts/prepare_checkpoint.py --checkpoint multilingual
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# The same pin the .NET parity lane uses (laya-dotnet/tools/regen_golden.py). Both lanes measure
# the same checkpoint, so they must not drift to different revisions: a port that matched one and
# not the other would look like a port bug.
HF_REPO = "convaiinnovations/laya"
HF_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

# What `Agent.open` and the fixture generator read: the budgets, the fitted temperatures and the
# tokenizer. No weights.
WANTED_LIGHT = ("rl_agent_config.json", "config.json", "tokenizer.json", "tokenizer/*")
# Plus what the exporter needs to trace the model.
WANTED = WANTED_LIGHT + ("model.safetensors", "encoder/*")


STAMP = ".laya-revision"


def stamped(path, revision, wanted_weights=None):
    """Whether `path` was produced by THIS revision (and, if asked, with weights).

    Everything cacheable here is keyed on the pinned revision, and nothing on disk otherwise
    records which revision produced it. Without that, "the graph is already exported" and "the
    checkpoint is already downloaded" are claims about a PATH rather than about a revision, so a
    restored cache from before a `HF_REVISION` bump is reused and the whole lane measures the old
    model while reporting on the new one. Both sides of the comparison come from the same stale
    artifact, so they agree and the cell goes green having tested nothing it claims to.
    """
    marker = os.path.join(path, STAMP) if os.path.isdir(path) else path + STAMP
    try:
        with open(marker, "r", encoding="utf-8") as handle:
            recorded = handle.read().strip().split()
    except OSError:
        return False
    if not recorded or recorded[0] != revision:
        return False
    # A weightless checkpoint cannot be traced, so it must not satisfy a request that needs one.
    if wanted_weights is not None:
        return (len(recorded) > 1 and recorded[1] == "weights") == bool(wanted_weights)
    return True


def stamp(path, revision, with_weights=None):
    """Record the revision `path` was produced from, once it is complete."""
    marker = os.path.join(path, STAMP) if os.path.isdir(path) else path + STAMP
    text = revision + ("" if with_weights is None else (" weights" if with_weights else " light"))
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def fetch(name, revision, destination, with_weights=True):
    """Materialise exactly the wanted files of one checkpoint at `destination`.

    `local_dir`, not the shared cache. `snapshot_download` returns the cache ROOT, and
    `allow_patterns` governs only what it DOWNLOADS -- anything already cached from another run is
    still sitting there. The english checkpoint has no subfolder, so its directory IS the
    repository root and contains `multilingual/` and `typed-decisions/` as children: copying that
    root produced a 2.2 GB "tokenizer only" checkpoint, which is how this was found.

    `with_weights=False` fetches only what the tokenizer and sequence fixtures need. Three of the
    fixture families cover BOTH checkpoints in one document, so a parity run has to hold both -- but
    only the one the end-to-end golden was recorded from needs a graph, and therefore weights. The
    other is a few megabytes this way instead of a few gigabytes.
    """
    from huggingface_hub import snapshot_download

    sys.path.insert(0, REPO)
    from laya.router import DEFAULT_MODELS

    if name not in DEFAULT_MODELS:
        raise SystemExit("unknown checkpoint %r; laya.router knows %s"
                         % (name, sorted(DEFAULT_MODELS)))
    repo, subfolder = DEFAULT_MODELS[name]
    if repo != HF_REPO:
        raise SystemExit(
            "%s now lives in %r, but the pinned revision belongs to %r; update HF_REPO and "
            "HF_REVISION here and in laya-dotnet/tools/regen_golden.py together"
            % (name, repo, HF_REPO))
    prefix = (subfolder + "/") if subfolder else ""
    wanted = WANTED if with_weights else WANTED_LIGHT

    staging = destination + ".partial"
    for path in (staging, destination):
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    snapshot_download(repo, revision=revision, local_dir=staging,
                      allow_patterns=[prefix + pattern for pattern in wanted])
    source = os.path.join(staging, subfolder) if subfolder else staging
    if not os.path.isfile(os.path.join(source, "rl_agent_config.json")):
        raise SystemExit(
            "the download produced no rl_agent_config.json at %s; the pinned revision's layout "
            "may have changed" % source)
    # Moved rather than copied, so a checkpoint is never on disk twice.
    os.replace(source, destination)
    if os.path.exists(staging):
        shutil.rmtree(staging)
    stamp(destination, revision, with_weights)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="english, multilingual, ...")
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument("--force", action="store_true",
                        help="re-download and re-export even when a stamp says this revision's "
                             "checkpoint and graph are already here")
    parser.add_argument("--no-graph", action="store_true",
                        help="fetch only the tokenizer and config, and do not export a graph: "
                             "enough for the tokenizer and sequence fixtures, which cover both "
                             "checkpoints, without paying for weights this run will not trace")
    args = parser.parse_args(argv)

    work = os.path.join(REPO, "laya-java", ".work")
    model_dir = os.path.join(work, "checkpoints", args.checkpoint)
    graph_dir = os.path.join(work, "onnx", args.checkpoint)
    graph = os.path.join(graph_dir, "laya.onnx")

    wants_weights = not args.no_graph
    if not args.force and stamped(model_dir, args.revision, wanted_weights=wants_weights):
        print("  checkpoint already at %s for %s, not re-downloading"
              % (os.path.relpath(model_dir, REPO), args.revision[:12]))
    else:
        print("downloading %s at %s%s" % (args.checkpoint, args.revision[:12],
                                          " (tokenizer and config only)" if args.no_graph else ""),
              flush=True)
        fetch(args.checkpoint, args.revision, model_dir, with_weights=wants_weights)
    size = sum(os.path.getsize(os.path.join(root, f))
               for root, _dirs, files in os.walk(model_dir) for f in files)
    print("  checkpoint at %s (%.1f MB)" % (os.path.relpath(model_dir, REPO), size / 1e6))

    if args.no_graph:
        print("  --no-graph: no export, as asked")
        return 0

    if (os.path.exists(graph) and os.path.getsize(graph) > 0 and not args.force
            and stamped(graph, args.revision)):
        print("  graph already exported at %s for %s (use --force to redo it)"
              % (os.path.relpath(graph, REPO), args.revision[:12]))
        return 0

    os.makedirs(graph_dir, exist_ok=True)
    started = time.perf_counter()
    print("exporting the fused graph (this is the slow part)", flush=True)
    # The repository's own exporter, so the graph the Java tests run is the graph laya ships.
    #
    # PYTHONPATH, not just cwd: a script's `sys.path[0]` is the SCRIPT's directory, so
    # `scripts/export_onnx.py` gets `<repo>/scripts` and `import laya` fails with
    # ModuleNotFoundError even when the subprocess runs from the repository root. Setting it here
    # rather than installing the package keeps the export measuring the working tree.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_onnx.py"),
                    "--model", model_dir, "--output", graph],
                   check=True, cwd=REPO, env=env)
    if not os.path.exists(graph) or os.path.getsize(graph) == 0:
        raise SystemExit("the exporter reported success but wrote no graph at %s" % graph)
    stamp(graph, args.revision)
    print("  %s (%.1f MB) in %.1f s"
          % (os.path.relpath(graph, REPO), os.path.getsize(graph) / 1e6,
             time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
