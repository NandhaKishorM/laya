"""Single-device supervised head tuning with immutable, resumable step checkpoints."""
import hashlib
import json
from pathlib import Path
import random
import shutil

from .data import atomic_json, file_hash
from .metrics import quality


def tensor_hash(named):
    h = hashlib.sha256()
    for name, tensor in named:
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze_decision(model):
    names = []
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(('head.', 'type_emb.', 'scorer.')))
        p.grad = None
        if p.requires_grad:
            names.append(name)
    if not names or not any(n.startswith('scorer.') for n in names):
        raise ValueError('unsupported decision architecture')
    model.encoder.eval()
    return names


def make_scaler(enabled):
    import torch
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler('cuda', enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train(adapter, rows, split, output, config, resume=None):
    import torch
    from safetensors.torch import load_file, save_file
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not config.get('full') and config['max_steps'] > 20:
        raise ValueError('smoke is limited to 20 actual optimizer steps; use --full explicitly')
    if config['micro_batch'] != 1 or config['accumulation'] < 1 or config['max_steps'] < 1:
        raise ValueError('initial protocol supports micro_batch=1 state/decision and positive accumulation/steps')
    checkpoint_every = config.get('checkpoint_every', 1)
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError('checkpoint_every must be a positive integer')
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    model = adapter.agent.model.float()
    names = freeze_decision(model)
    params = [p for p in model.parameters() if p.requires_grad]
    frozen = lambda: ((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    frozen_before = tensor_hash(frozen())
    initial_decision = tensor_hash((n, p) for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = make_scaler(adapter.dtype == 'fp16' and adapter.agent.device.type == 'cuda')
    training = [r for r in rows if split['assignments'][r['id']] == 'train']
    dev = [r for r in rows if split['assignments'][r['id']] == 'dev']
    if not training or not dev:
        raise ValueError('training and dev partitions must both be nonempty')
    contract_config = {k: v for k, v in config.items() if k not in ('max_steps', 'full', 'checkpoint_every')}
    # Default scheduling retains compatibility with checkpoints written before this option existed.
    if checkpoint_every != 1:
        contract_config['checkpoint_every'] = checkpoint_every
    contract = {'model': adapter.identity, 'data_hash': split['data_hash'], 'split_hash': split['split_hash'],
                'config': contract_config,
                'max_len': adapter.max_len, 'head_max_len': adapter.head_max_len, 'dtype': adapter.dtype,
                'device': adapter.device, 'trainable_names': names}
    step, epoch, cursor, skipped, best_loss, best_path = 0, 0, 0, 0, float('inf'), None
    if resume:
        resume = Path(resume)
        meta = json.loads((resume / 'progress.json').read_text())
        if meta['contract'] != contract:
            raise ValueError('resume model/data/config/device contract mismatch')
        if meta['frozen_hash'] != frozen_before:
            raise ValueError('resume frozen base mismatch')
        if (meta.get('head_hash') != file_hash(resume / 'head.safetensors')
                or meta.get('state_hash') != file_hash(resume / 'local_state.pt')):
            raise ValueError('resume checkpoint file hash mismatch')
        # This file is produced locally. weights_only also refuses arbitrary pickle globals.
        state = torch.load(resume / 'local_state.pt', map_location='cpu', weights_only=True)
        head = load_file(resume / 'head.safetensors')
        if set(head) != set(names):
            raise ValueError('resume decision parameter names mismatch')
        model.load_state_dict(head, strict=False)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        if adapter.agent.device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], adapter.agent.device)
        step, epoch, cursor, skipped = (meta[k] for k in ('step', 'epoch', 'cursor', 'skipped_updates'))
        best_loss, best_path = meta['best_dev_nll'], meta['best_path']
        if config['max_steps'] <= step:
            raise ValueError('resume max_steps must exceed saved actual optimizer steps')
    atomic_json(output / 'training_config.json', {'contract': contract, 'requested': config,
        'checkpoint_every': checkpoint_every, 'checkpoint_schedule': 'successful update multiples and final update',
        'trainable_parameters': sum(p.numel() for p in params), 'frozen_parameters': sum(p.numel() for n, p in frozen()),
        'batch_unit': 'one state = one expanded decision sequence; accumulation averages actual remaining states'})
    if adapter.agent.device.type == 'cuda':
        torch.cuda.synchronize(adapter.agent.device)
        torch.cuda.reset_peak_memory_stats(adapter.agent.device)
    log_path = output / 'training.jsonl'
    latest = None
    while step < config['max_steps']:
        order = list(range(len(training)))
        random.Random(seed + epoch).shuffle(order)
        if cursor >= len(order):
            epoch, cursor = epoch + 1, 0
            continue
        window = order[cursor:cursor + config['accumulation']]
        model.train()
        model.encoder.eval()
        model.act_head.eval()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for index in window:
            item, _ = adapter.prepare(training[index])
            batch = adapter.batch([item])
            logits = adapter.forward(batch, training=True)
            mask = batch['marker_mask'].to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits.masked_fill(~mask, -float('inf')),
                                                       batch['label'].to(logits.device))
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss_sum += float(loss.detach())
            scaler.scale(loss / len(window)).backward()
        scaler.unscale_(optimizer)
        grads_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
        if any(p.grad is not None for _, p in frozen()):
            raise AssertionError('frozen parameter received gradient')
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in params):
            raise AssertionError('decision head received no nonzero gradient')
        if grads_finite:
            torch.nn.utils.clip_grad_norm_(params, config['max_grad_norm'], error_if_nonfinite=True)
        elif not scaler.is_enabled():
            raise RuntimeError('nonfinite FP32 gradients')
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        did_update = grads_finite and scaler.get_scale() >= scale_before
        cursor += len(window)
        if not did_update:
            skipped += 1
            with log_path.open('a') as f:
                f.write(json.dumps({'step': step, 'status': 'SKIPPED', 'reason': 'GradScaler overflow',
                                    'epoch': epoch, 'cursor': cursor, 'scale': scaler.get_scale()}) + '\n')
            if skipped > 20:
                raise RuntimeError('more than 20 skipped scaler updates')
            continue
        step += 1
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise RuntimeError('nonfinite updated parameters')
        dev_nll = None
        save_checkpoint = step % checkpoint_every == 0 or step == config['max_steps']
        if save_checkpoint:
            dev_nll = quality(adapter.predict(dev, 'E2-dev', split, neutral=True))['nll']
            checkpoint = output / f'step-{step:05d}'
            checkpoint.mkdir()
            if dev_nll < best_loss:
                best_loss, best_path = dev_nll, str(checkpoint.resolve())
            save_file({n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad},
                      checkpoint / 'head.safetensors')
            state = {'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                     'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                     'cuda_rng': torch.cuda.get_rng_state(adapter.agent.device) if adapter.agent.device.type == 'cuda' else None}
            torch.save(state, checkpoint / 'local_state.pt')
            atomic_json(checkpoint / 'progress.json', {'contract': contract, 'step': step, 'epoch': epoch,
                'cursor': cursor, 'skipped_updates': skipped, 'best_dev_nll': best_loss,
                'best_path': best_path, 'frozen_hash': frozen_before,
                'head_hash': file_hash(checkpoint / 'head.safetensors'),
                'state_hash': file_hash(checkpoint / 'local_state.pt')})
            # Completion marker is written only after both checkpoint files are saved.
            latest = checkpoint
        with log_path.open('a') as f:
            f.write(json.dumps({'step': step, 'epoch': epoch, 'cursor': cursor, 'status': 'SUCCESS',
                               'states': len(window), 'train_loss': loss_sum / len(window), 'dev_nll': dev_nll,
                               'checkpoint_saved': save_checkpoint, 'scaler_scale': scaler.get_scale()}) + '\n')
    training_peak = ({'max_memory_allocated': torch.cuda.max_memory_allocated(adapter.agent.device),
                      'max_memory_reserved': torch.cuda.max_memory_reserved(adapter.agent.device),
                      'scope': 'training plus scheduled dev selection, excludes initial loading'}
                     if adapter.agent.device.type == 'cuda' else None)
    if tensor_hash(frozen()) != frozen_before:
        raise AssertionError('frozen encoder/action parameters changed')
    if tensor_hash((n, p) for n, p in model.named_parameters() if p.requires_grad) == initial_decision:
        raise AssertionError('no decision parameter updated')
    model.load_state_dict(load_file(Path(best_path) / 'head.safetensors'), strict=False)
    model.eval()
    result = {'status': 'SUCCESS', 'steps': step, 'skipped_updates': skipped, 'best_dev_nll': best_loss,
              'checkpoint_every': checkpoint_every,
              'best_path': best_path, 'latest': str(latest.resolve()), 'frozen_unchanged': True,
              'decision_updated': True, 'training_memory': training_peak,
              'action_outputs': 'not supervised or validated'}
    atomic_json(output / 'training_result.json', result)
    return result


def export_agent(adapter, model_input, output):
    from safetensors.torch import save_file
    output = Path(output)
    output.mkdir()
    for name in ('encoder', 'tokenizer'):
        shutil.copytree(Path(model_input) / name, output / name)
    cfg = dict(adapter.agent.cfg)
    cfg['temperature'] = [1., adapter.agent.temperature[1], 1.]
    cfg['temperature_by_options'] = {k: v for k, v in adapter.agent.temperature_by_options.items()
                                     if not k.startswith(('choice:', 'noul:'))}
    cfg.update(max_len=adapter.max_len, head_max_len=adapter.head_max_len)
    atomic_json(output / 'rl_agent_config.json', cfg)
    save_file({n: p.detach().cpu().contiguous() for n, p in adapter.agent.model.state_dict().items()},
              output / 'model.safetensors')
    atomic_json(output / 'zh_lineage.json', {'model_id': adapter.identity['model_id'],
        'base_revision': adapter.identity['revision'], 'base_hash': adapter.identity['content_hash'],
        'trained': True, 'selection': 'dev NLL; locked before calibration/test',
        'weights_hash': file_hash(output / 'model.safetensors')})
