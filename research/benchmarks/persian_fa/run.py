"""Run the benchmark against one backend and record every raw response.

Examples (from this folder)::

    # pipeline check, no model download (keyword matcher: NOT a model result or baseline)
    python run.py --backend keyword --output runs/keyword

    # smoke test on the real model: 2 cases, 1 repeat, 1 mode
    python run.py --backend laya --checkpoint multilingual --limit 2 --repeats 1 \
        --modes choice_fa --output runs/smoke

    # full protocol (64 cases x 2 modes x 3 repeats)
    python run.py --backend laya --checkpoint multilingual --output runs/multilingual

Every run goes to a NEW directory containing ``metadata.json`` and ``responses.jsonl``.
Check it with ``python audit.py --run-dir <dir>``.
"""
import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, load_cases, pinned_hashes, request_for, sha256_file
from prompts import LABELS, MODES, QUESTION_ID

NAMED_CHECKPOINTS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
HUB_REPO = "convaiinnovations/laya"


# --------------------------------------------------------------------------- backends
class KeywordBackend:
    """Deterministic keyword matcher, used only to test the pipeline without downloading
    weights. It is NOT a model result and NOT a fair baseline: its keywords were written with
    these 64 messages in view, so its score is inflated. Never publish it as a comparison."""

    name = "keyword"
    KEYWORDS = {
        "cancel": ("لغو", "کنسل", "ببندید", "حذف", "غیرفعال", "غيرفعال", "cancel", "deactivate",
                   "delete", "laghv"),
        "billing": ("پول", "مبلغ", "فاکتور", "فاكتور", "قیمت", "پرداخت", "کسر", "كسر", "هزینه",
                    "invoice", "refund", "charge", "pardakht", "pool", "factor", "شارژ"),
        "technical": ("خطا", "ارور", "باگ", "کرش", "باز نمی", "نمیشه", "نمی‌شه", "لاگ", "login",
                      "رمز", "کد تأیید", "api", "کند", "قطع", "baz nemishe", "error"),
    }

    def describe(self):
        return {"backend": "keyword"}

    def predict(self, state, questions):
        text = state["message"].lower()
        scores = {lab: sum(kw.lower() in text for kw in kws) for lab, kws in self.KEYWORDS.items()}
        best = max(scores, key=scores.get)
        pred = best if scores[best] > 0 else "other"
        probs = {lab: (0.7 if lab == pred else 0.1) for lab in LABELS}
        return {"answers": {QUESTION_ID: {"type": "choice", "choice": pred,
                                          "probabilities": probs, "answer_confidence": 0.7}}}


class LayaBackend:
    name = "laya"

    def __init__(self, checkpoint, device, revision):
        # Use this checkout when invoked as `python research/benchmarks/persian_fa/run.py`.
        sys.path.insert(0, str(ROOT.parents[2]))
        import laya  # imported lazily so the audit and tests never need torch

        self._laya = laya
        self.checkpoint = checkpoint
        self.revision = revision
        if checkpoint in NAMED_CHECKPOINTS:
            self.source = HUB_REPO
            self.subfolder = NAMED_CHECKPOINTS[checkpoint]
        else:  # a local directory or another hub repo id
            self.source, self.subfolder = checkpoint, None
        kwargs = {"device": device}
        if self.subfolder is not None:
            kwargs["subfolder"] = self.subfolder
        if revision is not None:
            if Path(self.source).exists():
                raise ValueError("--revision applies to Hub checkpoints; local files are identified by hashes")
            kwargs["revision"] = revision
        self.agent = laya.load(self.source, **kwargs)

    def describe(self):
        info = {"backend": "laya", "checkpoint": self.checkpoint, "source": self.source,
                "subfolder": self.subfolder, "revision_requested": self.revision,
                "revision_resolved": getattr(self.agent, "revision", None),
                "binning_map": getattr(self.agent, "binning_map", None),
                "laya_version": getattr(self._laya, "__version__", None)}
        for attr in ("device", "dtype"):
            if hasattr(self.agent, attr):
                info[attr] = str(getattr(self.agent, attr))
        if Path(self.source).is_dir():
            info["checkpoint_files"] = {f: sha256_file(Path(self.source) / f)
                                        for f in ("model.safetensors", "rl_agent_config.json")}
        module_file = getattr(self._laya, "__file__", None)
        if module_file:
            package = Path(module_file).resolve().parent
            info["source_files"] = {f: sha256_file(package / f) for f in
                                    ("agent.py", "common.py", "decision_model.py") if (package / f).exists()}
            try:
                info["source_commit"] = subprocess.check_output(
                    ["git", "-C", str(package), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                info["source_commit"] = None
        return info

    def synchronize(self):
        device = getattr(self.agent, "device", None)
        kind = str(device).split(":")[0]
        if kind in ("cuda", "mps"):
            import torch

            if kind == "cuda":
                torch.cuda.synchronize(device)
            else:
                torch.mps.synchronize()

    def predict(self, state, questions):
        return self.agent.predict(state, questions)


# --------------------------------------------------------------------------- helpers
def environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "numpy"):
        try:
            env[mod] = importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:
            env[mod] = None
    return env


def to_row(case, mode, repeat, result, latency_ms, error):
    row = {"case_id": case["id"], "mode": mode, "repeat": repeat, "prediction": None,
           "probabilities": None, "answer_confidence": None,
           "latency_ms": round(latency_ms, 3) if latency_ms is not None else None, "error": error}
    if result is not None and error is None:
        ans = result["answers"][QUESTION_ID]
        row["prediction"] = ans["choice"]
        row["probabilities"] = ans["probabilities"]
        row["answer_confidence"] = ans["answer_confidence"]
    return row


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("laya", "keyword"), default="laya")
    p.add_argument("--checkpoint", default="multilingual",
                   help="english | multilingual | typed-decisions | local dir | hub repo id")
    p.add_argument("--revision", default=None, help="hub commit to pin (recommended for published runs)")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    p.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="first N cases only (smoke tests)")
    p.add_argument("--normalize", action="store_true", help="ablation: Persian normalisation first")
    p.add_argument("--output", required=True, help="new directory for this run")
    args = p.parse_args(argv)
    if args.repeats < 1 or args.limit is not None and not 1 <= args.limit <= 64:
        p.error("repeats must be positive and limit must be in [1, 64]")
    if len(set(args.modes)) != len(args.modes):
        p.error("modes must be unique")
    if args.revision and (len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision)):
        p.error("--revision must be a full 40-character Hub commit SHA")
    if args.backend == "keyword" and args.revision:
        p.error("the keyword pipeline check has no model revision")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}; use a new --output directory")
    if (ROOT / "results").resolve() in out.resolve().parents:
        sys.exit("write fresh runs outside the historical archive")

    from audit import Audit, check_dataset, check_manifest, check_run, check_source
    cases = load_cases()
    audit = Audit()
    check_dataset(audit, cases)
    if not audit.problems:
        check_manifest(audit, cases)
        check_source(audit)
    if audit.problems:
        sys.exit("invalid benchmark inputs: " + "; ".join(audit.problems))
    if args.limit:
        cases = cases[: args.limit]

    t0 = time.perf_counter()
    backend = KeywordBackend() if args.backend == "keyword" else LayaBackend(args.checkpoint, args.device, args.revision)
    load_s = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=False)  # only after the model loaded

    metadata = {
        "record_format": 2,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "backend": backend.describe(),
        "environment": environment(),
        "sha256": pinned_hashes(),
        "modes": args.modes,
        "repeats": args.repeats,
        "n_cases": len(cases),
        "case_ids": [c["id"] for c in cases],
        "normalize": args.normalize,
        "smoke_test": len(cases) != 64 or args.repeats != 3 or len(args.modes) != len(MODES),
        "load_seconds": round(load_s, 2),
        "warmup": "one excluded warm-up request per mode",
        "not_a_model_result": args.backend == "keyword",
        "runner_sha256": sha256_file(Path(__file__)),
        "errors": 0,
    }

    total = len(args.modes) * args.repeats * len(cases)
    done = errors = 0
    def save_metadata():
        metadata["errors"], metadata["completed_requests"] = errors, done
        (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8", newline="\n")

    save_metadata()
    try:
        with open(out / "responses.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for mode in args.modes:
                req = request_for(cases[0], mode, args.normalize)
                try:  # same input transformation as the timed requests
                    backend.predict(**req)
                except Exception:
                    pass
                for repeat in range(args.repeats):
                    for case in cases:
                        req = request_for(case, mode, args.normalize)
                        result = error = None
                        sync = getattr(backend, "synchronize", lambda: None)
                        start = time.perf_counter()
                        try:
                            sync()
                            start = time.perf_counter()
                            result = backend.predict(**req)
                            sync()
                            row = to_row(case, mode, repeat, result, (time.perf_counter() - start) * 1000, None)
                        except Exception as exc:  # type only; no private paths/exception messages
                            error = type(exc).__name__
                            errors += 1
                            row = to_row(case, mode, repeat, None, (time.perf_counter() - start) * 1000, error)
                        row.update(request=req, response=result)
                        fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        fh.flush()
                        done += 1
                    print(f"{done}/{total}  {mode} r{repeat}  errors={errors}", flush=True)
    finally:
        save_metadata()
    print()
    print(f"wrote {out}/responses.jsonl and metadata.json")
    print(f"next: python audit.py --run-dir {out}")
    check_run(audit, cases, out)
    for problem in audit.problems:
        print(f"FAIL: {problem}")
    return 1 if errors or audit.problems else 0


if __name__ == "__main__":
    sys.exit(main())
