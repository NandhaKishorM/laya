"""Bare-message retrieval diagnostic, separate from formatted-query pipelines.

This loads only a pinned sentence encoder. It does not run a Laya decision model,
choose new model families or change the frozen fixed-k and amendment protocols.
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from laya.common import serialize_state  # noqa: E402
from laya.evals import Dataset, Example, file_fingerprint, questions_fingerprint  # noqa: E402
from laya.shortlist import shortlist_choice  # noqa: E402
from research.benchmarks.tournament import compare_embeddings as fixed  # noqa: E402
from research.benchmarks.tournament.diagnostics import fingerprint  # noqa: E402
from research.benchmarks.tournament.embedders import RoleEmbedder, SPECS, TextEncoder  # noqa: E402


SCHEMA = "laya-dedicated-shortlist/query-text-diagnostic/1"
VARIANTS = {"minilm": SPECS["minilm"], "nomic-qd": SPECS["nomic"]}
QUERY_FORMAT = "bare-message; no instructions or JSON wrapper; model query prefix retained"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selected_cases(name, split, limit, offset=0, seed=0):
    if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("offset must be nonnegative and limit positive")
    cases, dataset, identity = fixed.load_cases(name, split, limit + offset, seed)
    cases = cases[offset:]
    dataset = Dataset(dataset.examples[offset:])
    identity = dict(identity, case_ids_sha256=fingerprint([row["id"] for row in cases]),
                    questions_sha256=questions_fingerprint(dataset), offset=offset)
    return cases, dataset, identity


def source_identity():
    directory = Path(__file__).parent
    return {name: file_fingerprint(str(directory / name)) for name in
            ("query_text_diagnostic.py", "compare_embeddings.py", "embedders.py", "diagnostics.py", "benchmark.py")}


def encoder_runtime(encoder, requested_device):
    import torch
    requested = torch.device(requested_device)
    if requested.type == "cuda" and requested.index is None:
        requested = torch.device("cuda", torch.cuda.current_device())
    actual = torch.device(encoder.device)
    if actual.type == "cuda" and actual.index is None:
        actual = torch.device("cuda", torch.cuda.current_device())
    tensors = list(encoder.model.parameters()) + list(encoder.model.buffers())
    devices = sorted({str(tensor.device) for tensor in tensors})
    if actual != requested or not tensors or devices != [str(requested)]:
        raise ValueError("actual encoder tensors/device differ from requested device")
    floating = {str(tensor.dtype) for tensor in tensors if tensor.is_floating_point()}
    if floating != {"torch.float32"}:
        raise ValueError("bare-query diagnostic requires fp32 encoder weights")
    return {"requested_device": str(requested), "actual_device": str(actual),
            "tensor_devices": devices, "requested_dtype": "torch.float32", "floating_dtypes": sorted(floating),
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}


def retrieval(cases, questions, identity, embedder, requested_device="cpu"):
    """Keep reference row identity while embedding exactly each message string."""
    if set(questions) != {"q"} or questions["q"].get("type") != "choice":
        raise ValueError("diagnostic requires one choice question named q")
    criteria = questions["q"]["criteria"]
    if not isinstance(criteria, dict) or len(criteria) <= 20:
        raise ValueError("diagnostic requires more than 20 canonical choice labels")
    if not cases or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("missing or duplicate cases")
    if identity.get("case_ids_sha256") != fingerprint([row["id"] for row in cases]):
        raise ValueError("case identity differs")
    schema = Dataset([Example(cases[0]["state"], questions, {})])
    if identity.get("questions_sha256") != questions_fingerprint(schema):
        raise ValueError("question identity differs")
    for row in cases:
        state = row["state"]
        if (not isinstance(state, dict) or set(state) != {"message"} or
                not isinstance(state["message"], str)):
            raise ValueError("each diagnostic state must contain only a string message")
        if row["state_sha256"] != fingerprint(state) or row["expected"] not in criteria:
            raise ValueError("invalid state fingerprint or gold label")
        if serialize_state(state["message"]) != state["message"]:
            raise ValueError("string serialization would alter the bare query")
    runtime = encoder_runtime(embedder.encoder, requested_device)
    fixed.reset_cache(embedder, questions["q"])
    rows = []
    for case in cases:
        message = case["state"]["message"]
        # Passing the original dict or question instructions would change the treatment.
        ranked = shortlist_choice(message, criteria, embedder, k=20, instructions=None)
        if encoder_runtime(embedder.encoder, requested_device) != runtime:
            raise ValueError("encoder device, dtype or threads changed during retrieval")
        if len(ranked) != 20 or len(set(ranked)) != 20 or not set(ranked) <= set(criteria):
            raise ValueError("invalid retrieved candidate set")
        gold = case["expected"]
        rank = ranked.index(gold) + 1 if gold in ranked else None
        rows.append({"id": case["id"], "state_sha256": case["state_sha256"], "expected": gold,
                     "choice": ranked[0], "correct": ranked[0] == gold, "rank": rank, "gold_rank": rank,
                     "shortlist": ranked, "query_text_sha256": text_hash(message),
                     "prefixed_query_text_sha256": text_hash(embedder.spec.query_prefix + message)})
    return {"status": "complete", "identity": dict(identity), "strategy": "bare-message-top1",
            "query_format": QUERY_FORMAT, "query_hash_encoding": "SHA256 of exact UTF-8 text",
            "gold_rank_scope": "top 20 only; null means absent from top 20", "runtime": runtime,
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "recall": {"20": sum(row["rank"] is not None for row in rows) / len(rows)}, "cases": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(fixed.DATASETS), required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.limit < 1 or args.offset < 0 or args.threads < 1:
        ap.error("limit/threads must be positive and offset nonnegative")
    if Path(args.out).exists():
        ap.error("refusing to overwrite an existing report")
    import torch
    torch.set_num_threads(args.threads)
    report = {"schema": SCHEMA, "status": "running", "decision_model_loaded": False,
              "query_format": QUERY_FORMAT, "parameters": {key: value for key, value in vars(args).items() if key != "out"},
              "source_sha256": source_identity(), "environment": fixed.environment(),
              "interpretation": "Comparison with formatted-query shortlist/combined changes both query representation "
                                "and decision pipeline; it does not isolate the causal effect of adding Laya."}
    fixed.write_report(args.out, report)
    try:
        cases, dataset, identity = selected_cases(args.dataset, args.split, args.limit, args.offset, args.seed)
        spec = VARIANTS[args.variant]
        encoder = TextEncoder(spec, args.device, local_files_only=args.local_files_only)
        embedder = RoleEmbedder(encoder, spec)
        identity = dict(identity, checkpoint_revision=fixed.CHECKPOINT_REVISION,
                        embedder_id=fingerprint(embedder.identity()))
        report.update(identity=identity, embedder=dict(asdict(spec), dtype="float32", device=str(encoder.device)),
                      reference_checkpoint={"repo": "convaiinnovations/laya", "revision": fixed.CHECKPOINT_REVISION,
                                            "role": "pairing reference only; no decision weights loaded"})
        report["retrieval"] = retrieval(cases, dataset.examples[0].questions, identity, embedder, args.device)
        if source_identity() != report["source_sha256"]:
            raise RuntimeError("source changed during execution")
        report["memory"] = fixed.memory_snapshot(args.device)
        report["status"] = "complete"
        print(args.dataset, args.variant, report["retrieval"]["accuracy"], flush=True)
    except Exception as exc:
        report.update(status="failed", error="%s: %s" % (type(exc).__name__, exc))
        raise
    finally:
        fixed.write_report(args.out, report)


if __name__ == "__main__":
    main()
