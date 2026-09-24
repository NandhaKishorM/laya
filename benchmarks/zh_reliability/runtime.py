"""Environment diagnostics and a thin adapter over the repository's actual Agent."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from .data import atomic_json, digest, file_hash, options

MODEL_ID = 'convaiinnovations/laya-multilingual'
INPUT_KEYS = ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)


def environment():
    result = {'python': sys.version, 'executable': sys.executable, 'os': platform.platform(),
              'commit': command(['git', 'rev-parse', 'HEAD']),
              'dirty': bool(command(['git', 'status', '--porcelain'])),
              'source_hashes': {str(p): file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'environment': {k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                  'HF_HUB_OFFLINE', 'USE_TF', 'USE_TORCH', 'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS')
                              if k in os.environ}, 'dependencies': {},
              'nvidia_smi': command(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version',
                                     '--format=csv,noheader']), 'gpus': []}
    for package in ('torch', 'transformers', 'numpy', 'safetensors', 'huggingface_hub', 'tokenizers'):
        try:
            result['dependencies'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['dependencies'][package] = None
    try:
        import torch
        result.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      pytorch_arches=torch.cuda.get_arch_list())
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpu = {'process_index': i, 'name': p.name, 'uuid': str(getattr(p, 'uuid', 'unavailable')),
                   'memory_bytes': p.total_memory, 'capability': [p.major, p.minor]}
            if gpu['uuid'] != 'unavailable' and not gpu['uuid'].startswith('GPU-'):
                gpu['uuid'] = 'GPU-' + gpu['uuid']
            try:
                with torch.cuda.device(i):
                    x = torch.ones((16, 16), device=f'cuda:{i}')
                    value = (x @ x).sum().item()
                    torch.cuda.synchronize(i)
                    if value != 4096:
                        raise RuntimeError('CUDA tensor result mismatch')
                    del x
                gpu['status'] = 'SUCCESS'
            except Exception as e:
                gpu.update(status='FAILED', reason=str(e))
            # UUID matching avoids confusing process-local index with physical index.
            matches = [line.split(',')[0].strip() for line in result['nvidia_smi'].splitlines()
                       if gpu['uuid'] != 'unavailable' and gpu['uuid'] in line]
            gpu['physical_index'] = matches[0] if len(matches) == 1 else None
            result['gpus'].append(gpu)
        result['gpu_status'] = ('SUCCESS' if result['gpus'] and all(g['status'] == 'SUCCESS' for g in result['gpus'])
                                else 'FAILED' if result['gpus'] else 'NOT_RUN')
    except Exception as e:
        result.update(gpu_status='NOT_RUN', reason=str(e))
    return result


def select_device(selector, env):
    matches = [g for g in env['gpus'] if selector.lower() in g['name'].lower() or selector == g['uuid']
               or selector == f"cuda:{g['process_index']}"]
    if len(matches) != 1 or matches[0]['status'] != 'SUCCESS':
        raise RuntimeError('device must uniquely match one visible GPU with a successful CUDA operation: ' + selector)
    return 'cuda:' + str(matches[0]['process_index'])


def resolve_model(source, revision, offline, work):
    """Make a private snapshot so Agent's tokenizer compatibility patch cannot touch shared caches."""
    source_path = Path(source)
    sha = None
    if not source_path.is_dir():
        if source != MODEL_ID:
            raise ValueError('only the multilingual model is supported')
        from huggingface_hub import HfApi, snapshot_download
        if not offline:
            sha = HfApi().model_info(MODEL_ID, revision=revision).sha
        path = snapshot_download(MODEL_ID, revision=sha or revision, local_files_only=offline,
                                 allow_patterns=['rl_agent_config.json', 'model.safetensors',
                                                 'encoder/*.json', 'tokenizer/*'])
        source_path = Path(path)
        sha = source_path.name
    lineage_path = source_path / 'zh_lineage.json'
    lineage = json.loads(lineage_path.read_text()) if lineage_path.exists() else None
    if lineage and lineage['model_id'] != MODEL_ID:
        raise ValueError('checkpoint lineage is not multilingual')
    cfg = json.loads((source_path / 'rl_agent_config.json').read_text())
    # Local imports require explicit, inspectable multilingual identity, not just a folder name.
    if not lineage and not any(x in str(cfg.get('encoder', '')).lower() for x in ('mmbert', 'multilingual')):
        raise ValueError('local checkpoint config does not identify a multilingual encoder')
    files = [source_path / 'model.safetensors', source_path / 'rl_agent_config.json']
    files += sorted((source_path / 'encoder').glob('*.json')) + sorted((source_path / 'tokenizer').glob('*'))
    hashes = {str(p.relative_to(source_path)): file_hash(p) for p in files if p.is_file()}
    if lineage and lineage.get('weights_hash') and lineage['weights_hash'] != hashes['model.safetensors']:
        raise ValueError('exported checkpoint weight hash mismatch')
    identity = {'model_id': MODEL_ID, 'revision': sha or (lineage or {}).get('base_revision'),
                'content_hash': digest(hashes), 'files': hashes,
                'trained': bool(lineage and lineage.get('trained'))}
    target = Path(work) / 'model_input'
    target.mkdir()
    for name in hashes:
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path / name, dst)
    atomic_json(Path(work) / 'model_identity.json', identity)
    return target, identity


def ensure_device(agent, device):
    import torch
    expected = torch.device(device)
    if agent.device != expected or any(p.device != expected for p in agent.model.parameters()):
        raise RuntimeError('device mismatch / SDK CPU fallback: requested ' + device)


class Adapter:
    def __init__(self, agent, identity, max_len=512, head_max_len=192, dtype='fp16'):
        import torch
        self.agent, self.identity = agent, identity
        self.max_len, self.head_max_len, self.dtype = max_len, head_max_len, dtype
        self.device = str(agent.device)
        if head_max_len < 16 or max_len <= head_max_len + 4:
            raise ValueError('max_len must leave state space beyond head_max_len + 4')
        agent.amp_enabled = dtype == 'fp16' and agent.device.type == 'cuda'
        agent.dtype = torch.float16 if agent.amp_enabled else torch.float32
        self.original = {'temperature_raw': agent.temperature_raw,
                         'temperature_by_options_raw': agent.temperature_by_options_raw,
                         'temperature': list(agent.temperature),
                         'temperature_by_options': dict(agent.temperature_by_options),
                         'lang_temperatures': dict(agent.lang_temperatures)}

    def temperature(self, row, neutral=False):
        from laya.common import QTYPES, temp_bucket
        if neutral:
            return 1.
        qt = QTYPES[row['question']['type']]
        cfg = self.agent.lang_temperatures.get('zh', {'temperature': self.agent.temperature,
                                                    'temperature_by_options': self.agent.temperature_by_options})
        return cfg['temperature_by_options'].get(temp_bucket(qt, len(options(row))), cfg['temperature'][qt])

    def prepare(self, row):
        from laya.common import build_sequence, render_options
        a, q = self.agent, row['question']
        a._check_question(row['id'], q)
        internal = a._to_internal(q)
        item = a._encode_state(row['state'], [row['id']], {row['id']: internal},
                               self.max_len, self.head_max_len)[0]
        state_ids = a.tok(row['state'].replace(a.tok.mask_token, ' '), add_special_tokens=False)['input_ids']
        empty, markers = build_sequence(a.tok, '', internal, self.max_len, self.head_max_len, state_ids=[])
        if len(markers) != len(options(row)) or any(item['ids'][m] != a.tok.mask_token_id for m in markers):
            raise ValueError('option marker truncated')
        retained = max(0, len(item['ids']) - len(empty))
        head = a.tok(f"{internal['t']} question: {internal['ins'].replace(a.tok.mask_token, ' ')}",
                     add_special_tokens=False)['input_ids']
        opt_full = [a.tok(' ' + o.replace(a.tok.mask_token, ' '), add_special_tokens=False)['input_ids']
                    for o in render_options(internal)]
        head_cut = len(head) > markers[0] - 2
        opt_cut = any(len(full) > (markers[i + 1] if i + 1 < len(markers) else len(empty) - 2) - m - 1
                      for i, (m, full) in enumerate(zip(markers, opt_full)))
        item['label'] = options(row).index(row['label'])
        tokens = {'input_tokens': len(item['ids']), 'state_tokens': len(state_ids), 'state_retained': retained,
                  'option_count': len(markers), 'state_truncated': retained < len(state_ids),
                  'instructions_truncated': head_cut, 'options_truncated': opt_cut,
                  'truncated': retained < len(state_ids) or head_cut or opt_cut}
        return item, tokens

    def batch(self, items):
        from laya.common import collate_items
        return collate_items([items], self.agent.tok.pad_token_id)

    def forward(self, batch, training=False):
        import torch
        ensure_device(self.agent, self.device)
        with torch.autocast(device_type=self.agent.device.type, dtype=torch.float16,
                            enabled=self.dtype == 'fp16' and self.agent.device.type == 'cuda'):
            logits, _action_logits = self.agent.model(
                **{k: batch[k].to(self.agent.device) for k in INPUT_KEYS}, detach_encoder=training)
        if not torch.isfinite(logits).all():
            raise RuntimeError('nonfinite decision logits')
        return logits

    def predict(self, rows, condition, split, neutral=False):
        import torch
        from .metrics import probability_record
        self.agent.model.eval()
        results = []
        # One state has exactly one decision sequence in this schema.
        with torch.no_grad():
            for r in rows:
                item, tokens = self.prepare(r)
                logits = self.forward(self.batch([item]))[0, :len(options(r))].float().cpu().tolist()
                rec = probability_record(r, logits, self.temperature(r, neutral))
                rec.update(tokens=tokens, condition=condition, split=split['assignments'][r['id']], model=self.identity)
                results.append(rec)
        return results

    def parity(self, rows):
        import numpy as np
        from .metrics import probability_record
        import torch
        errors = []
        self.agent.model.eval()
        for row in rows:
            item, _ = self.prepare(row)
            with torch.no_grad():
                z = self.forward(self.batch([item]))[0, :len(options(row))].cpu().tolist()
            p = probability_record(row, z, self.temperature(row))['probabilities']
            answer = self.agent.system_one(row['state'], {row['id']: row['question']}, lang='zh',
                max_len=self.max_len, head_max_len=self.head_max_len)['answers'][row['id']]
            ensure_device(self.agent, self.device)
            public = ([1 - answer['noul'], answer['noul']] if row['question']['type'] == 'noul'
                      else [answer['probabilities'][k] for k in options(row)])
            error = float(np.max(np.abs(np.array(p) - public)))
            if error > .000051:
                raise AssertionError('Agent parity exceeds four-decimal rounding: ' + str(error))
            errors.append(error)
        return {'n': len(rows), 'max_absolute_error': max(errors), 'rounding_tolerance': .000051}


def measure(adapter, rows, warmup=10, batches=30):
    import numpy as np
    import torch
    if adapter.agent.device.type != 'cuda':
        raise RuntimeError('GPU measurements prohibit CPU fallback')
    if warmup < 1 or batches < 1 or not rows:
        raise ValueError('measurement needs positive warmup/batches and nonempty samples')
    device = adapter.agent.device
    adapter.agent.model.eval()
    prepared = [adapter.batch([adapter.prepare(r)[0]]) for r in rows]
    prepared_gpu = [{k: b[k].to(device) for k in INPUT_KEYS} for b in prepared]
    result = {'warmup_batches': warmup, 'measured_batches': batches, 'batch_states': 1,
              'batch_decisions': 1, 'independent_samples': len(rows),
              'tokens': [adapter.prepare(r)[1] for r in rows], 'repetitions': batches / len(rows),
              'model_stage_transfers': 'excludes H2D/D2H; inputs already on device'}
    from .metrics import probability_record
    with torch.no_grad():
        for stage in ('end_to_end', 'model_stage'):
            times = []
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            for i in range(warmup + batches):
                row = rows[i % len(rows)]
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                if stage == 'end_to_end':
                    item, _ = adapter.prepare(row)
                    z = adapter.forward(adapter.batch([item]))[0, :len(options(row))].cpu().tolist()
                    probability_record(row, z, adapter.temperature(row))
                else:
                    adapter.forward(prepared_gpu[i % len(rows)])
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                if i >= warmup:
                    times.append(elapsed)
            result[stage] = {'p50_batch_ms': float(np.quantile(times, .5) * 1000),
                'p95_batch_ms': float(np.quantile(times, .95) * 1000),
                'state_per_s': batches / sum(times), 'decision_per_s': batches / sum(times),
                'max_memory_allocated': torch.cuda.max_memory_allocated(device),
                'max_memory_reserved': torch.cuda.max_memory_reserved(device),
                'memory_scope': 'inference including warmup, resident model and prepared inputs; excludes loading/training'}
    return result
