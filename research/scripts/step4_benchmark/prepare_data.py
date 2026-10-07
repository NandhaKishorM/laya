"""Prepare LocalLLaMA/typed-decisions for `laya-train` and for evaluation.

The benchmark's own rows are `{state, questions, gold}` with the teacher distribution per
question. That is exactly the schema `laya.train` reads, so the conversion is a parse, not a
reshape: the two JSON-string columns are decoded and the row is written back out as JSONL.

Splits:

* ``train.jsonl`` — the dataset's ``train`` split (1,200 cases / 6,000 decisions), the input to
  ``laya-train``. The calibration slice is held out *inside* ``laya-train`` at a fixed seed
  (``calib_frac`` 0.1, capped at 400, seed 20260922), matching the published notebook.
* ``test.jsonl`` / ``test_meta.json`` — the dataset's ``test`` split (400 cases / 2,000
  decisions), used by ``evaluate.py``. ``test_meta.json`` keeps ``id`` and ``workflow`` so
  per-workflow accuracy can be reported alongside the pooled number.

Usage:
    python prepare_data.py --out-dir data/typed_decisions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def case_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """One dataset row as a `laya-train` record (`{state, questions, gold}`)."""
    return {
        "state": _parse(row["state"]),
        "questions": _parse(row["questions"]),
        "gold": _parse(row["gold"]),
    }


def load_split(split: str, dataset: str = "LocalLLaMA/typed-decisions") -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: the datasets package is required (pip install datasets)", file=sys.stderr)
        sys.exit(1)
    ds = load_dataset(dataset, "all", split=split)
    return list(ds)


def write_split(rows: List[Dict[str, Any]], out_dir: str, split: str) -> int:
    records = [case_to_record(r) for r in rows]
    path = os.path.join(out_dir, f"{split}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def write_test_meta(rows: List[Dict[str, Any]], out_dir: str) -> int:
    meta = []
    for r in rows:
        meta.append({
            "id": r.get("id", ""),
            "workflow": r.get("workflow", ""),
            "state": _parse(r["state"]),
            "questions": _parse(r["questions"]),
            "gold": _parse(r["gold"]),
        })
    path = os.path.join(out_dir, "test_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return len(meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/typed_decisions")
    parser.add_argument("--dataset", default="LocalLLaMA/typed-decisions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ("train", "test"):
        rows = load_split(split, args.dataset)
        n = write_split(rows, args.out_dir, split)
        print(f"{split}: {n} cases -> {os.path.join(args.out_dir, split + '.jsonl')}")
        if split == "test":
            m = write_test_meta(rows, args.out_dir)
            print(f"test meta: {m} cases -> {os.path.join(args.out_dir, 'test_meta.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
