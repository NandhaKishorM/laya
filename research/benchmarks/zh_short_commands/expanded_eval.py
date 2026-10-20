"""Evaluate frozen additional Chinese fixtures with the upstream seven-rung harness.

This is a diagnostic, not an independently annotated test set. Run on CPU in FP32
for comparison with the committed archive. No fitting or prompt selection occurs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import prompts
import run as upstream


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def datasets():
    base = upstream.load_cases()
    extra = upstream.load_cases(str(HERE / "expanded_cases.jsonl"))
    assert len(extra) == 96 and Counter(c["gold"] for c in extra) == dict.fromkeys(prompts.LABELS, 16)
    assert len({c["id"] for c in extra}) == 96
    assert len({c["text"] for c in extra}) == 96
    assert not ({c["text"] for c in base} & {c["text"] for c in extra})
    return {"baseline18": base, "expanded96": extra}


def make_records(config, cases, logits, agent):
    function = upstream.noul_records if config in prompts.NOUL_CONFIGS else upstream.choice_records
    result = function(config, cases, logits, agent, False)
    for row, raw in zip(result["cases"], logits):
        row["logits"] = [float(v) for v in raw]
    return result


def audit(path):
    """Rebuild requests and summaries from frozen inputs and archived raw logits."""
    import numpy as np
    doc = read(path)
    frozen = read(HERE / "expanded_manifest.json")
    for name, expected in frozen["sha256"].items():
        assert digest(HERE / name) == expected, ("changed frozen input", name)
    assert doc["manifest"] == frozen
    assert doc["config"]["runner_sha256"] == digest(__file__)
    cfg = doc["config"]["calibration"]
    agent = SimpleNamespace(temperature=cfg["temperature"], temperature_by_options=cfg["temperature_by_options"])
    suites = datasets()
    assert set(doc["suites"]) == set(suites)
    count = 0
    for suite, cases in suites.items():
        assert set(doc["suites"][suite]) == set(prompts.CONFIGS)
        for config, block in doc["suites"][suite].items():
            rows = block["cases"]
            expected_count = len(cases) * (4 if config in prompts.NOUL_CONFIGS else 1)
            assert len(rows) == len(block["repeat_logits"]) == expected_count
            z = [r["logits"] for r in rows]
            assert all(np.isfinite(v).all() for v in z)
            assert all(np.isfinite(v).all() for v in block["repeat_logits"])
            expected = make_records(config, cases, z, agent)
            assert rows == expected["cases"], (suite, config, "request/decision mismatch")
            assert block["report"] == expected["report"], (suite, config, "metric mismatch")
            repeat = make_records(config, cases, block["repeat_logits"], agent)
            changes = sum(a.get("pred_label", a.get("pred_bool")) != b.get("pred_label", b.get("pred_bool"))
                          for a, b in zip(rows, repeat["cases"]))
            assert changes == block["repeat_decision_changes"] == 0
            assert max(abs(a-b) for first, second in zip(z, block["repeat_logits"])
                       for a, b in zip(first, second)) == block["repeat_max_logit_delta"]
            count += len(rows)
    print("PASS: frozen requests, logits, decisions, metrics and repeats; %d primary records" % count)
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--checkpoint", default=str(ROOT / "models/laya-multilingual"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "expanded_report.json")
    args = parser.parse_args()
    if args.audit:
        audit(args.audit)
        return
    assert args.batch_size > 0
    assert not args.out.exists(), "Refuse to overwrite an existing run"
    suites = datasets()
    manifest = read(HERE / "expanded_manifest.json")
    for name, value in manifest["sha256"].items():
        assert digest(HERE / name) == value
    import torch
    import laya
    import transformers
    torch.manual_seed(13)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    agent = laya.load(args.checkpoint, device=args.device)
    agent.model.float().eval()
    agent.amp_enabled = False
    info = upstream.checkpoint_provenance(SimpleNamespace(checkpoint=args.checkpoint, subfolder=None, stub=False), agent)
    assert info.get("weight_sha256") == "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"
    checkpoint = Path(args.checkpoint)
    doc = {
        "manifest": manifest,
        "config": {"timestamp": datetime.now(timezone.utc).isoformat(),
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                   "transformers": transformers.__version__, "laya": laya.__version__,
                   "device": args.device, "dtype": "float32", "batch_size_states": args.batch_size,
                   "threads": 4, "seed": 13, "deterministic": True, "runner_sha256": digest(__file__),
                   "checkpoint": info,
                   "checkpoint_files": {str(p.relative_to(checkpoint)).replace('\\', '/'): digest(p)
                                        for p in checkpoint.rglob('*') if p.is_file() and '.cache' not in p.parts},
                   "calibration": {"temperature": [float(v) for v in agent.temperature],
                                   "temperature_by_options": agent.temperature_by_options}},
        "suites": {}, "sdk_parity": []}
    for suite, cases in suites.items():
        doc["suites"][suite] = {}
        for config in prompts.CONFIGS:
            pairs = prompts.cases_for(config, [c["text"] for c in cases])
            started = time.perf_counter()
            runs = []
            for repeat in range(2):
                logits = []
                for start in range(0, len(pairs), args.batch_size):
                    logits.extend(upstream.score_cases(agent, pairs[start:start + args.batch_size]))
                runs.append(logits)
            result = make_records(config, cases, runs[0], agent)
            second = make_records(config, cases, runs[1], agent)
            result["repeat_logits"] = [[float(v) for v in row] for row in runs[1]]
            result["repeat_decision_changes"] = sum(a.get("pred_label", a.get("pred_bool")) != b.get("pred_label", b.get("pred_bool"))
                                                    for a, b in zip(result["cases"], second["cases"]))
            result["repeat_max_logit_delta"] = max(abs(float(a)-float(b)) for x,y in zip(runs[0],runs[1]) for a,b in zip(x,y))
            result["elapsed_seconds_two_runs"] = time.perf_counter() - started
            doc["suites"][suite][config] = result
            print("%s %s accuracy=%s repeats_changed=%s" % (suite, config, result["report"]["accuracy"], result["repeat_decision_changes"]), flush=True)
            # Check a real SDK call against the research scorer, at identical FP32 settings.
            answers = agent.predict(pairs[0][0], pairs[0][1])["answers"]
            for row in result["cases"][:len(pairs[0][1])]:
                if config in prompts.CHOICE_CONFIGS:
                    answer = answers["intent"]
                    delta = abs(answer["probabilities"][row["pred_label"]] - row["probability"])
                    assert answer["choice"] == row["pred_label"]
                else:
                    answer = answers[row["dimension"]]
                    delta = abs(answer["noul"] - row["p_true"])
                    assert (answer["noul"] >= 0.5) == row["pred_bool"]
                assert delta < 0.001, (suite, config, "SDK/research probability mismatch", delta)
                doc["sdk_parity"].append({"suite": suite, "config": config, "case_id": row["case_id"], "delta": delta})
    write(args.out, doc)
    audit(args.out)


if __name__ == "__main__":
    main()
