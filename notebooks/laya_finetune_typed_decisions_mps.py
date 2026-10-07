"""Laya typed-decisions fine-tuning on Apple Silicon.

This is a standalone replacement for the Kaggle 2xT4 notebook. It performs:
1. dependency/model/data preparation
2. local preprocessing and caching
3. single-process MPS/CPU fine-tuning
4. temperature calibration
5. checkpoint/model export

Useful examples:
    python notebooks/laya_finetune_typed_decisions_mps.py --epochs 2 --micro-batch 1 --grad-accum 32
    python notebooks/laya_finetune_typed_decisions_mps.py --model-dir ./laya_base --items ./train_items.pt
    python notebooks/laya_finetune_typed_decisions_mps.py --device cpu
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from laya.agent import _fix_tokenizer_config

MODEL_ID = "convaiinnovations/laya"
DATASET_ID = "LocalLLaMA/typed-decisions"
DEFAULT_MODEL_DIR = "./laya_base"
DEFAULT_ITEMS = "./train_items.pt"
DEFAULT_OUTPUT_DIR = "./laya_finetuned_typed_decisions"


def choose_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS is unavailable; using CPU instead.")
        return torch.device("cpu")
    return torch.device(requested)


def prepare_model(model_dir):
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").exists():
        print(f"Downloading {MODEL_ID} to {model_dir} ...")
        snapshot_download(MODEL_ID, local_dir=str(model_dir))
    _fix_tokenizer_config(str(model_dir))
    return str(model_dir)


def prepare_items(model_dir, items_path, force=False):
    # The convergence target is `laya.train`; its input is row-level JSONL, not
    # pre-tokenized item tensors. Keep the cache honest: when the requested cache
    # changes, also invalidate the row cache.
    items_path = Path(items_path)
    rows_path = items_path.with_suffix(items_path.suffix + '.rows.jsonl')
    model_path = Path(model_dir).resolve()
    with open(model_path / 'rl_agent_config.json') as f:
        cfg = json.load(f)
    max_len = cfg.get('max_len', 1024)
    head_max_len = cfg.get('head_max_len', 256)
    cache_meta_path = rows_path.with_name(rows_path.name + '.meta.json')
    cache_key = {
        'model_dir': str(model_path),
        'max_len': max_len,
        'head_max_len': head_max_len,
        'dataset': DATASET_ID,
        'split': 'train',
        'format': 'rows_jsonl',
    }
    if rows_path.exists() and not force:
        if not cache_meta_path.exists():
            print(f'Using legacy cached training rows without metadata: {rows_path}')
            return str(rows_path)
        try:
            with open(cache_meta_path) as f:
                cached_key = json.load(f)
        except (OSError, json.JSONDecodeError):
            cached_key = None
        if cached_key == cache_key:
            print(f'Using cached training rows: {rows_path}')
            return str(rows_path)
        print('Training-row cache key changed; rebuilding the cache.')

    from datasets import load_dataset

    print(f'Downloading dataset {DATASET_ID} ...')
    dataset = load_dataset(DATASET_ID, 'all', split='train')
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(rows_path, 'w', encoding='utf-8') as f:
        for row in dataset:
            state = json.loads(row['state'])
            questions = json.loads(row['questions'])
            gold = json.loads(row['gold'])
            f.write(json.dumps({'state': state, 'questions': questions, 'gold': gold}, ensure_ascii=False) + '\n')
            n_rows += 1
    with open(cache_meta_path, 'w') as f:
        json.dump(cache_key, f, indent=2)
    print(f'Saved {n_rows} training rows to {rows_path}')
    return str(rows_path)


def train(args, model_dir, rows_path, device):
    from laya.train import TrainConfig, finetune

    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        calib_max=args.calib_max,
        calib_seed=20260922,
        seed=42,
        max_len=1024,
        head_max_len=256,
        gradient_checkpointing=not args.no_checkpointing,
        amp=False,
    )
    summary = finetune(
        data=str(rows_path),
        model_dir=str(model_dir),
        output_dir=str(args.output_dir),
        config=config,
        device=args.device if args.device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    )
    # Wrapper-visible metadata and persistence preserved from the old entry point.
    for rel in [Path(args.output_dir) / 'rl_agent_config.json', Path(args.output_dir) / 'checkpoint_latest' / 'rl_agent_config.json']:
        if rel.exists():
            cfg = json.loads(rel.read_text(encoding='utf-8'))
            for name, entry in (summary.get('calibration') or {}).items():
                if isinstance(entry, dict) and entry.get('items') == 0:
                    idx = {'choice': 0, 'score': 1, 'noul': 2}.get(name)
                    if idx is not None:
                        cfg['temperature'][idx] = 1.2
            cfg.pop('temperature_by_options', None)
            cfg['model_name'] = 'laya-typed-decisions'
            cfg['max_tokens_per_batch'] = 2048
            rel.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(f'Train items: {summary["train_items"]}; calibration items: {summary["calibration_items"]}')
    print('Fitted temperatures (choice, score, noul):', [round(t, 3) for t in summary['temperature']])
    print(f'Model saved to {args.output_dir}')


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya on Apple Silicon MPS")
    parser.add_argument("model_dir", nargs="?", default=None)
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--model-dir", dest="model_dir_option")
    parser.add_argument("--output-dir", dest="output_dir_option")
    parser.add_argument("--items", default=DEFAULT_ITEMS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--calib-max", type=int, default=400)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.model_id != MODEL_ID:
        # Keep the requested model ID local to preparation by updating the
        # module constant before prepare_model() is called.
        globals()["MODEL_ID"] = args.model_id

    args.model_dir = args.model_dir_option or args.model_dir or DEFAULT_MODEL_DIR
    args.output_dir = args.output_dir_option or args.output_dir or DEFAULT_OUTPUT_DIR

    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    model_dir = prepare_model(args.model_dir)
    rows_path = prepare_items(model_dir, args.items, force=args.force_preprocess)
    train(args, model_dir, rows_path, device)
    gc.collect()


if __name__ == "__main__":
    main()
