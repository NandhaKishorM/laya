"""Recompute the published E0/E1 evidence and reject inconsistent archived results."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.benchmarks.zh_reliability.aggregate_massive import render_report, verify_artifact
from research.benchmarks.zh_reliability.data import digest
from research.benchmarks.zh_reliability.calibration import FIT_METHOD, fit_temperature

RESULTS = Path(__file__).resolve().parents[1] / 'research/benchmarks/zh_reliability/results'


class PublishedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads((RESULTS / 'massive_alarm_2026-09-24.json').read_text(encoding='utf-8'))
        cls.result = verify_artifact(cls.artifact)

    def altered(self):
        return copy.deepcopy(self.artifact)

    def test_published_table_recomputes_from_decisions(self):
        self.assertEqual(self.result['counts']['test_source_utterances'], 93)
        self.assertEqual(self.result['counts']['test_groups'], 89)
        self.assertEqual(self.result['counts']['test_decisions'], 186)
        before, after = (self.result[c]['overall'] for c in ('E0', 'E1'))
        self.assertAlmostEqual(before['accuracy'], 99 / 186)
        self.assertEqual(before['accuracy'], after['accuracy'])
        self.assertEqual(before['macro_f1'], after['macro_f1'])
        self.assertAlmostEqual(before['nll'], 1.383747, places=6)
        self.assertAlmostEqual(after['nll'], .785033, places=6)
        self.assertEqual(self.result['fitted']['noul']['temperature'], 5.)
        self.assertTrue(self.result['fitted']['noul']['boundary_hit'])
        paired = self.result['paired']['E1-E0']
        self.assertEqual(paired['repeats'], 1000)
        self.assertEqual(paired['after_minus_before']['accuracy']['ci95'], [0., 0.])
        self.assertLess(paired['after_minus_before']['nll']['ci95'][1], 0.)
        self.assertEqual(render_report(self.artifact, self.result),
                         (RESULTS / 'massive_alarm_2026-09-24.md').read_text(encoding='utf-8'))

    def test_recalibration_preserves_historical_inference(self):
        provenance = self.artifact['provenance']
        note = provenance['offline_recalibration']
        self.assertEqual(provenance['measurement_date'], '2026-09-24')
        self.assertEqual(note['source_artifact_sha256'],
                         '48ee7fb1000e664d242ed1a96a71cdddc65b807a496bf313c051fcca031b46c2')
        raw = [{k: v for k, v in row.items() if k != 'calibrated_temperature'}
               for row in self.artifact['records']]
        self.assertEqual(digest(raw), '8f3defe00a6b9bda9e827b781e054e2753531c43e92e0fbb7a67d65b2a0410ca')
        self.assertEqual(self.artifact['calibration']['method'], FIT_METHOD)
        self.assertEqual(note['historical_fitted']['choice']['temperature'], 3.1355035588563744)
        self.assertEqual(self.result['fitted']['choice']['temperature'], 3.1355036471)
        self.assertEqual(self.result['E1']['overall']['accuracy'], note['historical_E1_overall']['accuracy'])
        self.assertLess(abs(self.result['E1']['overall']['nll'] - note['historical_E1_overall']['nll']), 1e-8)

    def test_archived_fit_is_invariant_to_sample_order(self):
        from research.benchmarks.zh_reliability.aggregate_massive import evidence_prediction
        rows = [evidence_prediction(r, 'E0') for r in self.artifact['records']
                if r['split'] == 'calibration' and r['type'] == 'choice']
        expected = self.artifact['calibration']['fitted']['choice']
        self.assertEqual(fit_temperature(rows), expected)
        self.assertEqual(fit_temperature(rows[::-1]), expected)
        self.assertEqual(fit_temperature(sorted(rows, key=lambda r: r['gold'])), expected)

    def test_fit_temperature_is_compared_exactly(self):
        artifact = self.altered()
        artifact['calibration']['fitted']['choice']['temperature'] += 1e-10
        with self.assertRaisesRegex(ValueError, 'fitted temperature/objective mismatch'):
            verify_artifact(artifact, bootstrap=False)

    def test_fit_method_cannot_be_substituted(self):
        artifact = self.altered()
        artifact['calibration']['method'] = 'unrecorded'
        with self.assertRaisesRegex(ValueError, 'fit method mismatch'):
            verify_artifact(artifact, bootstrap=False)

    def test_records_hash_rejects_changed_logits(self):
        artifact = self.altered()
        artifact['records'][0]['raw_logits'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'record hash mismatch'):
            verify_artifact(artifact, bootstrap=False)

    def test_rehashed_logits_must_match_between_conditions(self):
        artifact = self.altered()
        record = next(r for r in artifact['records'] if r['split'] == 'test')
        record['calibrated_raw_logits'][0] += 1.
        artifact['records_sha256'] = digest(artifact['records'])
        with self.assertRaisesRegex(ValueError, 'changed raw decision logits'):
            verify_artifact(artifact, bootstrap=False)

    def test_rehashed_duplicate_ids_are_rejected(self):
        artifact = self.altered()
        artifact['records'][1]['id'] = artifact['records'][0]['id']
        artifact['records_sha256'] = digest(artifact['records'])
        with self.assertRaisesRegex(ValueError, 'duplicate decision IDs'):
            verify_artifact(artifact, bootstrap=False)

    def test_rehashed_cross_partition_group_is_rejected(self):
        artifact = self.altered()
        calibration = next(r for r in artifact['records'] if r['split'] == 'calibration')
        record = next(r for r in artifact['records'] if r['split'] == 'test')
        record['group_id'] = calibration['group_id']
        artifact['records_sha256'] = digest(artifact['records'])
        with self.assertRaisesRegex(ValueError, 'group leakage'):
            verify_artifact(artifact, bootstrap=False)

    def test_source_pair_hash_must_match(self):
        artifact = self.altered()
        artifact['records'][0]['source_row_hash'] = '0' * 64
        artifact['records_sha256'] = digest(artifact['records'])
        with self.assertRaisesRegex(ValueError, 'source utterance split/group/hash mismatch'):
            verify_artifact(artifact, bootstrap=False)

    def test_temperature_is_refitted_from_calibration_only(self):
        artifact = self.altered()
        artifact['calibration']['fitted']['choice']['temperature'] += .1
        with self.assertRaisesRegex(ValueError, 'fitted temperature/objective mismatch'):
            verify_artifact(artifact, bootstrap=False)

    def test_stored_metric_cannot_override_recomputed_value(self):
        artifact = self.altered()
        artifact['summary']['E1']['overall']['accuracy'] = 1.
        with self.assertRaisesRegex(ValueError, 'stored summary mismatch: E1'):
            verify_artifact(artifact, bootstrap=False)


if __name__ == '__main__':
    unittest.main()
