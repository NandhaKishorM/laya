import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import ROOT, audit_run, load_cases, verify_archive
from metrics import quality


class AuditTests(unittest.TestCase):
    def test_archived_source_bytes(self):
        verify_archive()

    def test_archived_results(self):
        expected = {'jev': {'choice': 64, 'four_noul': 63}, 'laya': {'choice': 20, 'four_noul': 18}}
        for backend, modes in expected.items():
            result = audit_run(ROOT / 'results/v1' / backend)
            for mode, correct in modes.items():
                self.assertEqual((result[mode]['correct'], result[mode]['n']), (correct, 64))
                self.assertEqual(result[mode]['failed_requests'], 0)

    def fixture(self, folder):
        metadata = json.loads((ROOT / 'results/v1/laya/metadata.json').read_text(encoding='utf-8'))
        rows = [json.loads(x) for x in (ROOT / 'results/v1/laya/raw.jsonl').read_text(encoding='utf-8').splitlines()]
        rows = [r for r in rows if r['mode'] == 'choice' and r['repeat'] == 0]
        metadata.update(repeats=1, modes=['choice'], evaluated_ids=[r['id'] for r in rows])
        (folder / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        return rows

    def write(self, folder, rows):
        (folder / 'raw.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows), encoding='utf-8')

    def test_reject_corrupted_records(self):
        for corruption in ['missing', 'duplicate', 'gold', 'request', 'prediction', 'negative_latency']:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                rows = self.fixture(folder)
                if corruption == 'missing': rows.pop()
                elif corruption == 'duplicate': rows.append(copy.deepcopy(rows[0]))
                elif corruption == 'gold': rows[0]['expected'] = 'NOT_A_LABEL'
                elif corruption == 'request': rows[0]['request_sha256'] = 'wrong'
                elif corruption == 'prediction': rows[0]['predicted'] = 'noise' if rows[0]['predicted'] != 'noise' else 'todo'
                else: rows[0]['elapsed_ms'] = -1
                self.write(folder, rows)
                with self.assertRaises(ValueError): audit_run(folder)

    def test_failed_request_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = self.fixture(folder)
            rows[0] = {k: v for k, v in rows[0].items() if k in ['id', 'family', 'expected', 'mode', 'repeat', 'request_sha256']}
            rows[0].update(status='error', error_type='TimeoutError')
            self.write(folder, rows)
            result = audit_run(folder)['choice']
            self.assertEqual((result['n'], result['errors'], result['failed_requests']), (64, 1, 1))
            self.assertEqual(result['timing']['n'], 63)

    def test_balanced_frozen_cases(self):
        cases, _ = load_cases()
        for label in ['urgent', 'todo', 'valuable', 'noise']:
            self.assertEqual(sum(c['expected'] == label for c in cases), 16)

    def test_perfect_and_failed_classification(self):
        rows = [{'status': 'ok', 'expected': c, 'predicted': c} for c in ['urgent', 'todo', 'valuable', 'noise']]
        self.assertEqual(quality(rows)['macro_f1'], 1)
        rows[0] = {'status': 'error', 'expected': 'urgent'}
        result = quality(rows)
        self.assertEqual((result['n'], result['correct'], result['missed_action_count']), (4, 3, 1))


if __name__ == '__main__':
    unittest.main()
