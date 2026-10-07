"""CPU smoke test for the Step-4 benchmark harness.

Proves the wiring and the metric plumbing without a GPU, the network or real weights:

1. `prepare_data.case_to_record` handles the dataset's JSON-string columns;
2. the metric definitions match hand-computed values from the notebook's cell 14;
3. `run_benchmark.build_command` emits the notebook's hyperparameters;
4. the whole loop runs: a tiny local checkpoint is fine-tuned through the `laya-train` CLI on a
   tiny typed-decisions-shaped JSONL, then scored by `evaluate.evaluate_checkpoint`.

Run: python research/scripts/step4_benchmark/smoke_test.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya.common import DecisionModel  # noqa: E402
from metrics import accuracy, brier_score, expected_calibration_error, soft_accuracy  # noqa: E402
from prepare_data import case_to_record  # noqa: E402
from run_benchmark import NOTEBOOK_HYPERPARAMS, build_command  # noqa: E402

WORDS = ["please", "refund", "invoice", "charged", "twice", "app", "crash", "error", "screen",
         "billing", "technical", "other", "urgent", "low", "high", "which", "team", "is", "it",
         "level", "statement", "does", "not", "hold"]


def make_tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for w in WORDS:
        vocab[w] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def make_checkpoint(path: Path, tok) -> None:
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
    config.save_pretrained(path / "encoder")
    tok.save_pretrained(path / "tokenizer")
    model = DecisionModel(BertModel(config), head_layers=1, n_act=2)
    save_file(model.state_dict(), path / "model.safetensors")
    (path / "rl_agent_config.json").write_text(json.dumps({
        "encoder": "unused/offline", "head_layers": 1, "act_costs": {"act": 0},
        "max_len": 64, "head_max_len": 40, "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": {},
    }))


DEPARTMENT = {"type": "choice", "instructions": "which team ?",
              "criteria": {"billing": "", "technical": "", "other": ""}}
URGENT = {"type": "noul", "instructions": "is it urgent ?"}
LEVEL = {"type": "score", "instructions": "which level ?", "criteria": ["low", "high"]}

CASES = [
    ("please refund invoice", {"department": {"label": "billing", "probabilities": {"billing": 0.9, "technical": 0.05, "other": 0.05}},
                               "urgent": {"label": "false", "probabilities": {"false": 0.8, "true": 0.2}},
                               "level": {"label": 0, "score": 0.2, "probabilities": {"0": 0.7, "1": 0.3}}}),
    ("charged twice invoice", {"department": {"label": "billing", "probabilities": {"billing": 0.85, "technical": 0.1, "other": 0.05}},
                               "urgent": {"label": "true", "probabilities": {"false": 0.3, "true": 0.7}},
                               "level": {"label": 0, "score": 0.1, "probabilities": {"0": 0.8, "1": 0.2}}}),
    ("app crash error", {"department": {"label": "technical", "probabilities": {"billing": 0.05, "technical": 0.9, "other": 0.05}},
                         "urgent": {"label": "true", "probabilities": {"false": 0.2, "true": 0.8}},
                         "level": {"label": 1, "score": 0.9, "probabilities": {"0": 0.3, "1": 0.7}}}),
    ("screen error crash", {"department": {"label": "technical", "probabilities": {"billing": 0.1, "technical": 0.8, "other": 0.1}},
                            "urgent": {"label": "false", "probabilities": {"false": 0.6, "true": 0.4}},
                            "level": {"label": 1, "score": 0.8, "probabilities": {"0": 0.2, "1": 0.8}}}),
]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -> ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(1)


def metric_plumbing() -> None:
    print("metric plumbing")
    check("accuracy", abs(accuracy([1.0, 0.0, 1.0, 1.0]) - 0.75) < 1e-12)
    check("brier exact", abs(brier_score([1.0, 0.0], [1.0, 0.0])) < 1e-12)
    check("brier off", abs(brier_score([0.0, 1.0], [1.0, 0.0]) - 2.0) < 1e-12)
    check("soft accuracy", abs(soft_accuracy([0.5, 0.5], [1.0, 0.0]) - 0.5) < 1e-12)
    # ECE over a single bin: all confidence 0.5, half correct -> |0.5 - 0.5| = 0.
    ece = expected_calibration_error([0.5, 0.5], [1.0, 0.0])
    check("ece single bin", abs(ece) < 1e-12, f"{ece}")
    # ECE over the top bin: all confidence 1.0, half correct -> |1.0 - 0.5| = 0.5.
    ece = expected_calibration_error([1.0, 1.0], [1.0, 0.0])
    check("ece top bin", abs(ece - 0.5) < 1e-12, f"{ece}")


def case_conversion() -> None:
    print("prepare_data.case_to_record")
    row = {"state": json.dumps({"x": 1}), "questions": json.dumps({"q": DEPARTMENT}),
           "gold": json.dumps({"q": {"label": "billing", "probabilities": {"billing": 1.0}}})}
    rec = case_to_record(row)
    check("state decoded", rec["state"] == {"x": 1})
    check("questions decoded", rec["questions"]["q"]["type"] == "choice")
    check("gold decoded", rec["gold"]["q"]["label"] == "billing")


def command_contract() -> None:
    print("run_benchmark.run_laya_train command")
    cmd = build_command("train.jsonl", "out/rlcd-seed0", "rlcd", 0,
                        "convaiinnovations/laya", "cuda", NOTEBOOK_HYPERPARAMS)
    joined = " ".join(cmd)
    check("module entrypoint", "-m laya.train_cli" in joined)
    for token in ("--loss rlcd", "--seed 0", "--epochs 4", "--micro-batch 8", "--grad-accum 8",
                  "--encoder-lr 2.5e-05", "--head-lr 0.0001", "--max-len 1024",
                  "--head-max-len 256"):
        check(f"flag {token}", token in joined)


def end_to_end(tmp: Path) -> None:
    print("end-to-end: CLI fine-tune -> evaluate")
    tok = make_tokenizer()
    make_checkpoint(tmp / "base", tok)

    train_rows = []
    for _ in range(6):
        for state, gold in CASES:
            train_rows.append({"state": state,
                               "questions": {"department": DEPARTMENT, "urgent": URGENT, "level": LEVEL},
                               "gold": gold})
    data_path = tmp / "train.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in train_rows), encoding="utf-8")

    out_dir = tmp / "run_rlcd"
    cmd = build_command(str(data_path), str(out_dir), "soft-ce", 0, str(tmp / "base"), "cpu",
                        dict(NOTEBOOK_HYPERPARAMS, epochs=1, micro_batch=8, grad_accum=8,
                             max_len=64, head_max_len=40))
    result = subprocess.run(cmd, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
    check("laya-train exit 0", result.returncode == 0, f"rc={result.returncode}")
    check("checkpoint written", (out_dir / "rl_agent_config.json").exists())
    saved = json.loads((out_dir / "rl_agent_config.json").read_text())
    check("training recipe recorded", saved["training"]["laya_train"]["loss"] == "soft-ce")
    check("calibration recorded", "laya_train_calibration" in saved["training"])

    test_meta = [{"id": f"c{i}", "workflow": "smoke", "state": state,
                  "questions": {"department": DEPARTMENT, "urgent": URGENT, "level": LEVEL},
                  "gold": gold}
                 for i, (state, gold) in enumerate(CASES)]

    from evaluate import evaluate_checkpoint
    for temperatures in ("shipped", "raw"):
        res = evaluate_checkpoint(str(out_dir), test_meta, device="cpu", temperatures=temperatures)
        check(f"{temperatures} accuracy in [0,1]", 0.0 <= res["accuracy"] <= 1.0, f"{res['accuracy']}")
        check(f"{temperatures} ece finite", res["ece"] == res["ece"], f"{res['ece']}")
        check(f"{temperatures} brier finite", res["brier"] == res["brier"], f"{res['brier']}")
        check(f"{temperatures} n_decisions", res["n_decisions"] == len(CASES) * 3, f"{res['n_decisions']}")
    print("   note: accuracy here is a randomly-initialised tiny encoder, not a benchmark number")


def main() -> int:
    metric_plumbing()
    case_conversion()
    command_contract()
    with tempfile.TemporaryDirectory() as d:
        end_to_end(Path(d))
    print("\nSMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
