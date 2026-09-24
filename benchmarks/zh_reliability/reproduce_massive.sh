#!/usr/bin/env bash
set -Eeuo pipefail
# Relative paths are interpreted from the repository root. Outputs must be new.
BENCH_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$BENCH_REPO"
EXP_DIR=${1:?Usage: bash benchmarks/zh_reliability/reproduce_massive.sh OUTPUT_DIR [MODEL_DIR [DATA_JSONL [SPLIT_JSON]]]}
BENCH_PY=${LAYA_ZH_PYTHON:-.venv/bin/python}
BENCH_BASE=${2:-${LAYA_ZH_MODEL:-runs/zh-download/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67}}
BENCH_DATA=${3:-${LAYA_ZH_DATA:-runs/massive-alarm-data/data.jsonl}}
BENCH_SPLIT=${4:-${LAYA_ZH_SPLIT:-runs/massive-alarm-data/split_manifest.json}}
BENCH_REVISION=e4e9ddf21a7b1903b7acffd8814ad4307bf63a67
mkdir -- "$EXP_DIR"
"$BENCH_PY" - "$EXP_DIR" "$BENCH_BASE" "$BENCH_DATA" "$BENCH_SPLIT" <<'PY_INIT'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from benchmarks.zh_reliability.data import atomic_json, file_hash, partition, read_data

root, model, data, split_path = (Path(value).resolve() for value in sys.argv[1:])
template = Path('benchmarks/zh_reliability/massive_protocol.json')
protocol = json.loads(template.read_text(encoding='utf-8'))
rows = read_data(data, formal=True)
split = partition(rows, split_path, protocol['seed'], formal=True)
if split['counts'] != protocol['expected_split_counts'] or split['group_counts'] != protocol['expected_group_counts']:
    raise ValueError('data partitions do not match the fixed MASSIVE alarm protocol')
lineage = json.loads((model / 'zh_lineage.json').read_text(encoding='utf-8'))
if (lineage.get('model_id') != 'convaiinnovations/laya-multilingual'
        or lineage.get('base_revision') != protocol['model_revision'] or lineage.get('trained') is not False):
    raise ValueError('use the pinned original multilingual checkpoint with its zh_lineage.json')
protocol.update(created_utc=datetime.now(timezone.utc).isoformat(), status='FIXED_BEFORE_MODEL_PREDICTIONS',
                data=str(data), data_file_sha256=file_hash(data), split_manifest=str(split_path),
                split_file_sha256=file_hash(split_path), base_model=str(model),
                base_model_weights_sha256=file_hash(model / 'model.safetensors'),
                base_model_lineage_sha256=file_hash(model / 'zh_lineage.json'),
                template_sha256=file_hash(template))
atomic_json(root / 'protocol.json', protocol)
atomic_json(root / 'manifest.json', {
    'status': 'RUNNING', 'gpu_experiment': 'RUNNING', 'command': 'formal-massive-experiment-replay',
    'completed': [], 'protocol_sha256': file_hash(root / 'protocol.json')})
PY_INIT
BENCH_STAGE=initialization
mark_failed() {
  BENCH_EXIT_CODE=$?
  "$BENCH_PY" - "$EXP_DIR" "$BENCH_STAGE" "$BENCH_EXIT_CODE" <<'PY_FAIL'
import json
from pathlib import Path
import sys
from benchmarks.zh_reliability.data import atomic_json
root, stage, code = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
path = root / 'manifest.json'
manifest = json.loads(path.read_text())
manifest.update(status='FAILED', gpu_experiment='FAILED', active_stage=stage,
                error={'stage': stage, 'exit_code': code})
atomic_json(path, manifest)
PY_FAIL
  return "$BENCH_EXIT_CODE"
}
trap mark_failed ERR
run_benchmark() {
  "$BENCH_PY" -m benchmarks.zh_reliability "$@" --data "$BENCH_DATA" --formal \
    --split-manifest "$BENCH_SPLIT" --max-len 512 --head-max-len 192 --dtype fp16 \
    --seed 20260924 --revision "$BENCH_REVISION"
}
BENCH_STAGE=base-calibration
run_benchmark calibrate --model "$BENCH_BASE" --device 'RTX 2080 Ti' --minimum 30 \
  --run-dir "$EXP_DIR/base-calibration"
BENCH_STAGE=head
run_benchmark train-head --model "$BENCH_BASE" --device 'TITAN RTX' --full --max-steps 198 \
  --checkpoint-every 11 --micro-batch 1 --accumulation 8 --lr 0.0001 \
  --weight-decay 0.01 --max-grad-norm 1 --run-dir "$EXP_DIR/head"
BENCH_STAGE=head-calibration
run_benchmark calibrate --model "$EXP_DIR/head/exported" --device 'RTX 2080 Ti' --minimum 30 \
  --run-dir "$EXP_DIR/head-calibration"
BENCH_STAGE=E0
run_benchmark eval --model "$BENCH_BASE" --device 'RTX 2080 Ti' --split test \
  --measure --warmup 10 --batches 186 --run-dir "$EXP_DIR/E0"
BENCH_STAGE=E1
run_benchmark eval --model "$BENCH_BASE" --device 'RTX 2080 Ti' --split test \
  --calibration "$EXP_DIR/base-calibration/calibration.json" --run-dir "$EXP_DIR/E1"
BENCH_STAGE=E2
run_benchmark eval --model "$EXP_DIR/head/exported" --device 'RTX 2080 Ti' --split test \
  --measure --warmup 10 --batches 186 --run-dir "$EXP_DIR/E2"
BENCH_STAGE=E3
run_benchmark eval --model "$EXP_DIR/head/exported" --device 'RTX 2080 Ti' --split test \
  --calibration "$EXP_DIR/head-calibration/calibration.json" --run-dir "$EXP_DIR/E3"
BENCH_STAGE=aggregation
"$BENCH_PY" -m benchmarks.zh_reliability.aggregate_massive "$EXP_DIR"
