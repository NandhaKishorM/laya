"""Offline logic tests; tiny random BERT is NOT a real multilingual Laya experiment."""
import copy
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_TORCH', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from benchmarks.zh_reliability.calibration import calibrate, fit_temperature, validate_calibration
from benchmarks.zh_reliability.data import FIXTURES, atomic_json, options, partition, read_data
from benchmarks.zh_reliability.metrics import log_probs, paired_bootstrap, probability_record, quality, summarize
from benchmarks.zh_reliability.runtime import Adapter, ensure_device, select_device
from benchmarks.zh_reliability.training import export_agent, freeze_decision, train
from laya.agent import Agent
from laya.common import DecisionModel, collate_items


class DataMetricsTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_data(FIXTURES)
        self.split = partition(self.rows)
        self.model = {'content_hash': 'mock-only', 'trained': False}

    def prediction(self, row=None, logits=(0., 0.), t=1.):
        row = row or self.rows[0]
        p = probability_record(row, logits, t)
        p.update(split=self.split['assignments'][row['id']], model=self.model)
        return p

    def test_schema_groups_manifest_and_relations(self):
        self.assertEqual(len(self.rows), 80)
        self.assertEqual(len(self.split['groups']), 40)
        self.assertEqual(self.split['counts'], {'test': 12, 'train': 48, 'dev': 8, 'calibration': 12})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'split.json'
            saved = partition(self.rows, path)
            self.assertEqual(partition(self.rows, path, seed=1), saved)
            changed = copy.deepcopy(self.rows)
            changed[0]['state'] += '!'
            with self.assertRaisesRegex(ValueError, 'hash'):
                partition(changed, path)
            saved['assignments'][self.rows[0]['id']] = 'train'
            atomic_json(path, saved)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                partition(self.rows, path)

    def test_schema_rejections(self):
        for change in ({'label': 'unknown'}, {'provenance': {}}, {'tags': 'negation'},
                       {'question': {'type': 'score', 'instructions': 'bad'}},
                       {'provenance': {'kind': 'agent_authored', 'human_verified': True}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                rows = copy.deepcopy(self.rows)
                rows[0].update(change)
                path = Path(tmp) / 'data.jsonl'
                path.write_text('\n'.join(json.dumps(r) for r in rows))
                with self.assertRaises(ValueError):
                    read_data(path)
        with self.assertRaises(ValueError):
            read_data(FIXTURES, formal=True)

    def test_template_family_may_not_cross_groups(self):
        rows = copy.deepcopy(self.rows)
        rows[2]['provenance']['template_family'] = rows[0]['provenance']['template_family']
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'data.jsonl'
            p.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, 'family crosses'):
                read_data(p)

    def test_formal_requires_fixed_licensed_verified_data(self):
        rows = copy.deepcopy(self.rows)
        for r in rows:
            r['provenance'].update(kind='user_supplied', human_verified=True, source='test source',
                                   license='CC0', annotation_status='double reviewed')
            r['split'] = self.split['assignments'][r['id']]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'real.jsonl'
            p.write_text('\n'.join(json.dumps(r) for r in rows))
            accepted = read_data(p, formal=True)
            self.assertEqual(partition(accepted, formal=True)['assignments'], self.split['assignments'])
            accepted[0]['split'] = 'train'
            with self.assertRaisesRegex(ValueError, 'split a group'):
                partition(accepted, formal=True)

    def test_choice_permutation_preserves_semantic_mapping(self):
        row = self.rows[40]
        before = probability_record(row, [4., 3., 2., 1.], 1.)
        perm = copy.deepcopy(row)
        perm['question']['criteria'] = dict(reversed(list(perm['question']['criteria'].items())))
        after = probability_record(perm, [1., 2., 3., 4.], 1.)
        self.assertEqual(before['predicted'], after['predicted'])
        self.assertEqual(before['gold'], after['gold'])
        for key, value in zip(before['option_order'], before['probabilities']):
            self.assertAlmostEqual(value, dict(zip(after['option_order'], after['probabilities']))[key])

    def test_noul_probability_is_true_despite_display_labels(self):
        row = copy.deepcopy(self.rows[0])
        row['question']['labels'] = {'false': '拒绝', 'true': '接受'}
        row['question']['criteria'] = dict(reversed(list(row['question']['criteria'].items())))
        p = self.prediction(row, [0., math.log(3)])
        self.assertEqual(p['option_order'], ['false', 'true'])
        self.assertAlmostEqual(p['p_true'], .75)
        self.assertAlmostEqual(p['confidence'], .75)

    def test_valid_mask_and_temperature_invariance(self):
        expected = log_probs([1., 2.], 1.)
        np.testing.assert_allclose(expected, log_probs([1., 99., 2.], 1., [True, False, True]))
        np.testing.assert_allclose(np.exp(expected), torch.softmax(torch.tensor([1., 2.], dtype=torch.float64), 0))
        for t in (.5, 1., 2., 5.):
            self.assertEqual(log_probs([1., 4., -2.], t).argmax(), 1)
        for t in (0., -.1, .1, 6., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                log_probs([0., 1.], t)

    def test_hand_computable_nll_brier_ece_and_confidence(self):
        p = self.prediction(logits=(0., math.log(3)))  # gold true, correct with p=.75
        m = quality([p])
        self.assertAlmostEqual(m['nll'], -math.log(.75))
        self.assertAlmostEqual(m['brier'], .125)
        self.assertAlmostEqual(m['ece'], .25)
        self.assertAlmostEqual(p['confidence'], .75)  # not 1 - entropy/log(2)
        self.assertEqual(m['accuracy'], 1.)
        uniform = self.prediction()
        self.assertEqual(uniform['confidence'], .5)
        self.assertIsNone(quality([])['nll'])
        json.dumps(summarize([]), allow_nan=False)

    def test_nll_does_not_underflow_via_probabilities(self):
        p = self.prediction(logits=(10000., -10000.))
        self.assertEqual(quality([p])['nll'], 20000.)

    def test_risk_ties_do_not_depend_on_labels(self):
        a, b = self.prediction(), self.prediction()
        a.update(id='a', gold='false')
        b.update(id='b', gold='true')
        self.assertEqual(quality([b, a])['risk_coverage'][0]['risk'], 0)
        a['gold'], b['gold'] = 'true', 'false'
        self.assertEqual(quality([b, a])['risk_coverage'][0]['risk'], 1)
        self.assertEqual(quality([a, b])['risk_coverage'], quality([b, a])['risk_coverage'])

    def test_task_local_macro_f1(self):
        a, b = self.prediction(), self.prediction()
        a.update(task_id='a', gold='false', predicted='false')
        b.update(task_id='b', gold='true', predicted='true')
        # Each task has one perfectly predicted class and one absent class: F1=.5 each.
        self.assertEqual(quality([a, b])['macro_f1'], .5)

    def calibration_rows(self):
        return [self.prediction(r, [0.] * len(options(r))) for r in self.rows
                if self.split['assignments'][r['id']] == 'calibration']

    def test_fit_export_reload_no_double_scaling_or_bucket_shadowing(self):
        rows = self.calibration_rows()
        original = {'temperature': [2., 3., 4.], 'temperature_by_options': {'choice:3-5': 5., 'score:2': 2.},
                    'lang_temperatures': {}}
        a = calibrate(rows, self.model, self.split, original, minimum=1, smoke=True)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'cal.json'
            atomic_json(p, a)
            a = json.loads(p.read_text())
        cfg = validate_calibration(a, self.model, self.split, True)
        self.assertEqual(cfg['zh']['temperature'][1], 3.)
        self.assertNotIn('choice:3-5', cfg['zh']['temperature_by_options'])
        self.assertEqual(cfg['zh']['temperature_by_options']['score:2'], 2.)
        altered = copy.deepcopy(rows)
        for r in altered:
            r['temperature'] = 5.
            r['probabilities'] = [0.] * len(r['raw_logits'])
        self.assertEqual(fit_temperature(rows, 1), fit_temperature(altered, 1))
        with self.assertRaisesRegex(ValueError, 'smoke'):
            validate_calibration(a, self.model, self.split)
        with self.assertRaisesRegex(ValueError, 'model or data'):
            validate_calibration(a, {'content_hash': 'other'}, self.split, True)
        bad = copy.deepcopy(self.split)
        bad['data_hash'] = 'wrong'
        with self.assertRaises(ValueError):
            validate_calibration(a, self.model, bad, True)
        bad = copy.deepcopy(a)
        bad['calibration_hash'] = 'wrong'
        with self.assertRaises(ValueError):
            validate_calibration(bad, self.model, self.split, True)
        bad = copy.deepcopy(a)
        bad['lang_temperatures']['zh']['temperature'][0] = .1
        with self.assertRaises(ValueError):
            validate_calibration(bad, self.model, self.split, True)

    def test_fit_bounds_skipped_and_partition_guard(self):
        rows = self.calibration_rows()
        self.assertEqual(fit_temperature(rows)['status'], 'SKIPPED')
        a = self.prediction(logits=(10., -10.))
        result = fit_temperature([a] * 30)
        self.assertEqual(result['temperature'], 5.)
        self.assertTrue(result['boundary_hit'])
        self.assertLessEqual(result['nll_fitted'], result['nll_t1'])
        rows[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'calibration partition'):
            calibrate(rows, self.model, self.split, {})

    def test_bootstrap_pairs_and_insufficient_groups(self):
        rows = self.calibration_rows()
        self.assertEqual(paired_bootstrap(rows, rows)['status'], 'SKIPPED')
        result = paired_bootstrap(rows, rows, repeats=10, min_groups=2)
        self.assertEqual(result['after_minus_before']['nll']['ci95'], [0., 0.])
        with self.assertRaises(ValueError):
            paired_bootstrap(rows, rows[:-1])

    def test_visible_device_selector_and_cpu_fallback(self):
        env = {'gpus': [{'name': 'TITAN RTX', 'process_index': 0, 'uuid': 'GPU-b', 'status': 'SUCCESS'},
                        {'name': 'RTX 2080 Ti', 'process_index': 1, 'uuid': 'GPU-a', 'status': 'FAILED'}]}
        self.assertEqual(select_device('GPU-b', env), 'cuda:0')
        with self.assertRaises(RuntimeError):
            select_device('2080', env)
        with self.assertRaises(RuntimeError):
            select_device('physical:0', env)
        fake = type('A', (), {'device': torch.device('cpu')})()
        with self.assertRaisesRegex(RuntimeError, 'fallback'):
            ensure_device(fake, 'cuda:0')

    def test_cli_failure_is_nonzero_and_cannot_reuse_success(self):
        from benchmarks.zh_reliability.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'failed'
            with patch('benchmarks.zh_reliability.__main__.environment', return_value={'gpus': [], 'gpu_status': 'NOT_RUN'}):
                self.assertEqual(main(['import-data', '--data', str(run / 'missing'), '--run-dir', str(run)]), 1)
            m = json.loads((run / 'manifest.json').read_text())
            self.assertEqual(m['status'], 'FAILED')
            self.assertEqual(m['error']['stage'], 'data')
            self.assertEqual(main(['import-data', '--run-dir', str(run)]), 2)

    def test_no_cuda_smoke_and_report_saved_results(self):
        from benchmarks.zh_reliability.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            run, out = Path(tmp) / 'no-gpu', Path(tmp) / 'report'
            with patch('benchmarks.zh_reliability.__main__.environment', return_value={'gpus': [], 'gpu_status': 'NOT_RUN'}):
                self.assertEqual(main(['smoke', '--gpu', '--run-dir', str(run)]), 0)
                self.assertEqual(json.loads((run / 'manifest.json').read_text())['status'], 'SKIPPED')
                self.assertEqual(main(['report', '--source', str(run), '--run-dir', str(out)]), 0)
            self.assertEqual((run / 'split_manifest.json').read_bytes(), (out / 'split_manifest.json').read_bytes())
            self.assertEqual((run / 'metrics.json').read_bytes(), (out / 'metrics.json').read_bytes())
            self.assertIn('SKIPPED', (out / 'report.md').read_text())

    def test_checkpoint_interval_cli_configuration(self):
        from benchmarks.zh_reliability.__main__ import parser, training_config
        default = parser().parse_args(['train-head'])
        self.assertEqual(training_config(default)['checkpoint_every'], 1)
        requested = parser().parse_args(['train-head', '--checkpoint-every', '11', '--max-steps', '198', '--full'])
        self.assertEqual(training_config(requested)['checkpoint_every'], 11)

    def test_all_cli_help(self):
        from benchmarks.zh_reliability.__main__ import parser
        for name in ('check-env', 'import-data', 'smoke', 'eval', 'calibrate', 'train-head', 'resume', 'report'):
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as e:
                parser().parse_args([name, '--help'])
            self.assertEqual(e.exception.code, 0)


class TinyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.base = Path(cls.tmp.name) / 'tiny'
        cls.base.mkdir()
        torch.manual_seed(7)
        config = BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=1, intermediate_size=32)
        config.save_pretrained(cls.base / 'encoder')
        backend = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
                                      'false': 5, 'true': 6, 'billing': 7, 'technical': 8,
                                      'shipping': 9, 'other': 10}, unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]',
                                      cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
        tok.save_pretrained(cls.base / 'tokenizer')
        model = DecisionModel(BertModel(config), head_layers=1)
        save_file(model.state_dict(), cls.base / 'model.safetensors')
        atomic_json(cls.base / 'rl_agent_config.json', {'encoder': 'tiny-test-only', 'head_layers': 1,
                    'act_costs': {'act': 0}, 'temperature': [2., 1., 3.],
                    'temperature_by_options': {'choice:3-5': 1.5}, 'max_len': 128, 'head_max_len': 64})
        cls.rows = read_data(FIXTURES)
        cls.split = partition(cls.rows)
        cls.config = {'seed': 17, 'max_steps': 2, 'micro_batch': 1, 'accumulation': 8, 'lr': 1e-4,
                      'weight_decay': .01, 'max_grad_norm': 1., 'full': False}

    def adapter(self):
        return Adapter(Agent(str(self.base), device='cpu', fast=False, compile=False),
                       {'model_id': 'tiny-mock-only', 'revision': None, 'content_hash': 'tiny', 'trained': False},
                       max_len=128, head_max_len=64, dtype='fp32')

    def test_raw_forward_official_parity_and_mask(self):
        a = self.adapter()
        result = a.parity([self.rows[0], self.rows[40]])
        self.assertLessEqual(result['max_absolute_error'], .000051)
        items = [a.prepare(r)[0] for r in (self.rows[0], self.rows[40])]
        b = collate_items([items], a.agent.tok.pad_token_id)
        self.assertEqual(b['marker_mask'].tolist(), [[True, True, False, False], [True] * 4])
        a.agent.model.eval()
        with torch.no_grad():
            z = a.forward(b)
        self.assertTrue(torch.equal(z[0, 2:], torch.full((2,), -1e4)))

    def test_mock_decision_not_action_and_noul_display_labels(self):
        a = self.adapter()
        row = copy.deepcopy(self.rows[0])
        row['question']['labels'] = {'false': '不', 'true': '是'}
        with patch.object(a.agent.model, 'forward', return_value=(torch.tensor([[0., 3.]]), torch.tensor([[999., -999.]]))):
            p = a.predict([row], 'E0', self.split)[0]
            self.assertAlmostEqual(p['p_true'], 1 / (1 + math.exp(-1)))
            a.parity([row])

    def test_token_lengths_state_truncation_and_missing_marker(self):
        a = self.adapter()
        r = copy.deepcopy(self.rows[0])
        r['state'] = 'hello ' * 500
        item, tokens = a.prepare(r)
        self.assertTrue(tokens['state_truncated'])
        self.assertEqual(tokens['input_tokens'], 128)
        self.assertLess(tokens['state_retained'], tokens['state_tokens'])
        a.max_len = 4
        with self.assertRaises(ValueError):
            a.prepare(r)

    def test_freeze_update_resume_and_agent_reload(self):
        a = self.adapter()
        frozen = {n: p.clone() for n, p in a.agent.model.named_parameters()
                  if n.startswith(('encoder.', 'act_head.'))}
        names = freeze_decision(a.agent.model)
        self.assertTrue(all(n.startswith(('head.', 'type_emb.', 'scorer.')) for n in names))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = train(a, self.rows, self.split, root / 'full', self.config)
            for n, p in a.agent.model.named_parameters():
                if n in frozen:
                    self.assertIsNone(p.grad)
                    self.assertTrue(torch.equal(p, frozen[n]))
            first = train(self.adapter(), self.rows, self.split, root / 'first', {**self.config, 'max_steps': 1})
            resumed = train(self.adapter(), self.rows, self.split, root / 'resumed', self.config, first['latest'])
            self.assertEqual(resumed['steps'], 2)
            for name, tensor in load_file(Path(full['latest']) / 'head.safetensors').items():
                self.assertTrue(torch.equal(tensor, load_file(Path(resumed['latest']) / 'head.safetensors')[name]), name)
            meta = json.loads((Path(resumed['latest']) / 'progress.json').read_text())
            self.assertEqual(meta['cursor'], 16)
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                train(self.adapter(), self.rows, self.split, root / 'bad', {**self.config, 'lr': .2}, first['latest'])
            with (Path(first['latest']) / 'local_state.pt').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'file hash mismatch'):
                train(self.adapter(), self.rows, self.split, root / 'corrupt', self.config, first['latest'])
            before = a.predict(self.rows[:2], 'E2', self.split, True)
            export_agent(a, self.base, root / 'export')
            reloaded = Adapter(Agent(str(root / 'export'), device='cpu'), a.identity, 128, 64, 'fp32')
            after = reloaded.predict(self.rows[:2], 'E2', self.split, True)
            self.assertEqual([r['raw_logits'] for r in before], [r['raw_logits'] for r in after])
            self.assertEqual(reloaded.agent.temperature, [1., 1., 1.])

    def test_periodic_checkpoints_include_final_update_and_reload_best(self):
        adapter = self.adapter()
        config = {**self.config, 'max_steps': 5, 'accumulation': 1, 'checkpoint_every': 2}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(adapter, 'predict', wraps=adapter.predict) as predict, \
                    patch('benchmarks.zh_reliability.training.quality',
                          side_effect=[{'nll': .6}, {'nll': .4}, {'nll': .8}]):
                result = train(adapter, self.rows, self.split, root, config)
            self.assertEqual(predict.call_count, 3)
            self.assertEqual(sorted(p.name for p in root.glob('step-*')), ['step-00002', 'step-00004', 'step-00005'])
            logs = [json.loads(line) for line in (root / 'training.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in logs], [1, 2, 3, 4, 5])
            self.assertEqual([r['dev_nll'] for r in logs], [None, .6, None, .4, .8])
            self.assertEqual([r['checkpoint_saved'] for r in logs], [False, True, False, True, True])
            self.assertEqual(Path(result['best_path']).name, 'step-00004')
            self.assertEqual(Path(result['latest']).name, 'step-00005')
            selected = load_file(root / 'step-00004' / 'head.safetensors')
            latest = load_file(root / 'step-00005' / 'head.safetensors')
            parameters = dict(adapter.agent.model.named_parameters())
            self.assertTrue(all(torch.equal(parameters[name], tensor) for name, tensor in selected.items()))
            self.assertTrue(any(not torch.equal(selected[name], tensor) for name, tensor in latest.items()))
            self.assertEqual(result['checkpoint_every'], 2)

    def test_checkpoint_schedule_resume_contract_and_default_compatibility(self):
        config = {**self.config, 'max_steps': 5, 'accumulation': 1, 'checkpoint_every': 2}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = train(self.adapter(), self.rows, self.split, root / 'full', config)
            first = train(self.adapter(), self.rows, self.split, root / 'first', {**config, 'max_steps': 2})
            resumed = train(self.adapter(), self.rows, self.split, root / 'resumed', config, first['latest'])
            self.assertEqual(sorted(p.name for p in (root / 'resumed').glob('step-*')), ['step-00004', 'step-00005'])
            full_head = load_file(Path(full['latest']) / 'head.safetensors')
            resumed_head = load_file(Path(resumed['latest']) / 'head.safetensors')
            self.assertTrue(all(torch.equal(tensor, resumed_head[name]) for name, tensor in full_head.items()))
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                train(self.adapter(), self.rows, self.split, root / 'mismatch',
                      {**config, 'checkpoint_every': 3}, first['latest'])
            old_default = train(self.adapter(), self.rows, self.split, root / 'old-default',
                                {**self.config, 'max_steps': 1})
            old_meta = json.loads((Path(old_default['latest']) / 'progress.json').read_text())
            self.assertNotIn('checkpoint_every', old_meta['contract']['config'])
            compatible = train(self.adapter(), self.rows, self.split, root / 'explicit-default',
                               {**self.config, 'checkpoint_every': 1}, old_default['latest'])
            self.assertEqual(compatible['steps'], 2)

    def test_checkpoint_schedule_counts_successful_updates_not_overflows(self):
        class SkipOnceScaler:
            def __init__(self):
                self.value = 1.
                self.skip = True

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def is_enabled(self):
                return True

            def get_scale(self):
                return self.value

            def step(self, optimizer):
                if not self.skip:
                    optimizer.step()

            def update(self):
                if self.skip:
                    self.value /= 2
                    self.skip = False

            def state_dict(self):
                return {'test_only_scale': self.value}

        config = {**self.config, 'max_steps': 3, 'accumulation': 1, 'checkpoint_every': 2}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('benchmarks.zh_reliability.training.make_scaler', return_value=SkipOnceScaler()):
            root = Path(tmp)
            result = train(self.adapter(), self.rows, self.split, root, config)
            self.assertEqual(result['steps'], 3)
            self.assertEqual(result['skipped_updates'], 1)
            self.assertEqual(sorted(p.name for p in root.glob('step-*')), ['step-00002', 'step-00003'])
            logs = [json.loads(line) for line in (root / 'training.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in logs], ['SKIPPED', 'SUCCESS', 'SUCCESS', 'SUCCESS'])
            self.assertEqual([r['cursor'] for r in logs], [1, 2, 3, 4])
            self.assertEqual([r['step'] for r in logs if r.get('checkpoint_saved')], [2, 3])

    def test_checkpoint_interval_rejects_nonpositive_or_noninteger_values(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp, \
                    self.assertRaisesRegex(ValueError, 'checkpoint_every must be a positive integer'):
                train(self.adapter(), self.rows, self.split, tmp, {**self.config, 'checkpoint_every': value})

    def test_tail_accumulation_uses_actual_window(self):
        # 10 train states -> one window of 8 and one of 2, both are real optimizer steps.
        rows = [r for r in self.rows if self.split['assignments'][r['id']] == 'train'][:10]
        rows += [r for r in self.rows if self.split['assignments'][r['id']] == 'dev'][:2]
        with tempfile.TemporaryDirectory() as tmp:
            result = train(self.adapter(), rows, self.split, tmp, self.config)
            logs = [json.loads(line) for line in (Path(tmp) / 'training.jsonl').read_text().splitlines()]
            self.assertEqual([r['states'] for r in logs], [8, 2])
            self.assertEqual(result['steps'], 2)


@unittest.skipUnless(os.environ.get('LAYA_ZH_REAL_MODEL'), 'opt-in: set LAYA_ZH_REAL_MODEL to multilingual local weights')
class RealCheckpointTests(unittest.TestCase):
    def test_official_parity(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        # Keep opt-in load within a private copy; never let Agent patch a shared snapshot.
        from benchmarks.zh_reliability.runtime import resolve_model
        with tempfile.TemporaryDirectory() as tmp:
            source, identity = resolve_model(os.environ['LAYA_ZH_REAL_MODEL'], 'main', True, tmp)
            agent = Agent(str(source), device='cuda:0', fast=False, compile=False)
            ensure_device(agent, 'cuda:0')
            a = Adapter(agent, identity, 256, 192, 'fp16')
            rows = read_data(FIXTURES)
            a.parity(rows[:2] + rows[40:42])


if __name__ == '__main__':
    unittest.main()
