import copy
import unittest
from unittest.mock import patch

import audit


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = audit.verify()

    def test_complete_archive(self):
        self.assertEqual(audit.audit_samples(self.plan), 11570)

    def changed(self, change):
        plan = copy.deepcopy(self.plan)
        change(plan['archive']['benchmark'])
        with self.assertRaises(ValueError): audit.audit_samples(plan)

    def test_missing_duplicate_and_unknown_samples(self):
        self.changed(lambda b: b['timings'].pop())
        self.changed(lambda b: b['timings'].append(dict(b['timings'][0])))
        self.changed(lambda b: b['timings'][0].update(id='unknown-case'))

    def test_nonfinite_negative_and_boolean_durations(self):
        for value in (float('nan'), float('inf'), -1, True):
            self.changed(lambda b: b['timings'][0].update(ms=value))

    def test_statistics_are_recomputed(self):
        self.changed(lambda b: b['summary']['distinct-claims-exact-32']['v6'].update(mean=1))
        self.changed(lambda b: b['cohorts']['invalid']['prepared'].update(p95=1))
        self.changed(lambda b: b['summary'].pop('distinct-claims-exact-32'))

    def test_dense_checker_regression_is_rejected(self):
        original = audit.prepared.check
        def broken(**kwargs):
            result = original(**kwargs)
            records = [r['value'] for r in result['source_records']]
            if sum('var' in r for r in records) == 24 and sum('left' in r for r in records) == 64:
                return dict(result, grounding='fail')
            return result
        with patch.object(audit.prepared, 'check', side_effect=broken):
            with self.assertRaisesRegex(ValueError, 'full-result mismatch: distinct-claims-exact-1'):
                audit.audit()

    def test_invalid_dense_input_is_rejected_without_timing(self):
        plan = copy.deepcopy(self.plan)
        plan['archive']['dense'][0]['input']['candidate'] = 'not-json'
        with patch.object(audit, 'verify', return_value=plan):
            with self.assertRaisesRegex(ValueError, 'full-result mismatch: distinct-claims-exact-1'):
                audit.audit()


if __name__ == '__main__': unittest.main()
