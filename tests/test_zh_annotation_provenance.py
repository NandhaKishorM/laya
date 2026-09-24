"""Offline regressions for upstream review evidence; all rows here are synthetic."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.zh_reliability.data import partition, read_data


class UpstreamAnnotationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'data.jsonl'
        self.row = {
            'id': 'synthetic_alarm_1', 'group_id': 'synthetic_source_1', 'task_id': 'alarm_operation',
            'state': '设置明早的闹钟', 'split': 'train',
            'question': {'type': 'choice', 'instructions': '选择闹钟操作',
                         'criteria': {'alarm_set': '设置闹钟', 'alarm_remove': '删除闹钟'}},
            'label': 'alarm_set', 'tags': [],
            'provenance': {
                'kind': 'public_human_annotated', 'human_verified': False,
                'source': 'synthetic source for offline validation only', 'license': 'CC-BY-4.0',
                'annotation_status': 'upstream intent judgments; no local human review',
                'source_intent': 'alarm_set',
                'upstream_annotation': {
                    'dataset': 'MASSIVE', 'version': '1.1', 'source_id': '1', 'intent': 'alarm_set',
                    'source_row_hash': 'b' * 64,
                    'evidence_url': 'https://github.com/alexa/massive/blob/' + 'a' * 40 + '/README.md#data-format',
                    'judgments': [{'worker_id': str(i), 'intent_score': 1} for i in range(3)],
                },
            },
        }

    def read(self, rows, formal=True):
        self.path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        return read_data(self.path, formal=formal)

    def test_source_review_keeps_local_review_false_and_preserves_group_splits(self):
        rows = []
        for i, split in enumerate(('train', 'dev', 'calibration', 'test')):
            source = copy.deepcopy(self.row)
            source.update(id=str(i), group_id='source_' + str(i), split=split)
            source['provenance']['upstream_annotation']['source_id'] = str(i)
            binary = copy.deepcopy(source)
            binary.update(id=str(i) + '_binary', task_id='alarm_set_binary', label='true')
            binary['question'] = {'type': 'noul', 'instructions': '是否设置闹钟？',
                                  'criteria': {'false': '不是设置闹钟', 'true': '设置闹钟'}}
            rows.extend((source, binary))
        loaded = self.read(rows)
        self.assertTrue(all(r['provenance']['human_verified'] is False for r in loaded))
        manifest = partition(loaded, formal=True)
        self.assertEqual(manifest['group_counts'], {s: 1 for s in ('train', 'dev', 'calibration', 'test')})
        self.assertEqual(manifest['assignments'], {r['id']: r['split'] for r in rows})
        rows[-1]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'split a group'):
            partition(self.read(rows), formal=True)

    def test_missing_or_invalid_identity_evidence_is_rejected(self):
        changes = {
            'dataset': (None, 'Other'), 'version': (None, '1.0'),
            'source_id': (None, 1, '', ' '), 'intent': (None, '', 'alarm_remove'),
            'source_row_hash': (None, '', 'b' * 63, 'g' * 64),
            'evidence_url': (None, 'https://github.com/alexa/massive/blob/main/README.md',
                             'https://example.com/' + 'a' * 40 + '/README.md',
                             'https://github.com/alexa/massive/blob/' + 'a' * 40 + '/README.md?branch=main'),
        }
        for field, values in changes.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    row = copy.deepcopy(self.row)
                    row['provenance']['upstream_annotation'][field] = value
                    with self.assertRaisesRegex(ValueError, 'source-review evidence'):
                        self.read([row])
        for value in (None, [], 'claimed review'):
            row = copy.deepcopy(self.row)
            row['provenance']['upstream_annotation'] = value
            with self.assertRaisesRegex(ValueError, 'source-review evidence'):
                self.read([row])

    def test_contested_missing_or_duplicate_judgments_are_rejected(self):
        valid = self.row['provenance']['upstream_annotation']['judgments']
        invalid = [None, {}, [], valid[:2], valid[:2] + [valid[0]], valid + [None]]
        for score in (0, 2, True, 1.0, '1', None):
            judgments = copy.deepcopy(valid)
            judgments[2]['intent_score'] = score
            invalid.append(judgments)
        for worker in ('', ' ', None, 2):
            judgments = copy.deepcopy(valid)
            judgments[2]['worker_id'] = worker
            invalid.append(judgments)
        invalid.append(valid + [{'worker_id': 'fourth', 'intent_score': 0}])
        for judgments in invalid:
            with self.subTest(judgments=judgments):
                row = copy.deepcopy(self.row)
                row['provenance']['upstream_annotation']['judgments'] = judgments
                with self.assertRaisesRegex(ValueError, 'source-review evidence'):
                    self.read([row])

    def test_public_kind_cannot_claim_local_human_review(self):
        row = copy.deepcopy(self.row)
        row['provenance']['human_verified'] = True
        for formal in (False, True):
            with self.subTest(formal=formal), self.assertRaisesRegex(ValueError, 'source-review evidence'):
                self.read([row], formal=formal)

    def test_formal_source_license_status_and_split_are_still_required(self):
        for field in ('source', 'license', 'annotation_status'):
            row = copy.deepcopy(self.row)
            del row['provenance'][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'formal data requires'):
                self.read([row])
        row = copy.deepcopy(self.row)
        del row['split']
        with self.assertRaisesRegex(ValueError, 'formal data requires'):
            self.read([row])
        row = copy.deepcopy(self.row)
        del row['provenance']['source_intent']
        with self.assertRaisesRegex(ValueError, 'source-review evidence'):
            self.read([row])

    def test_pinned_raw_document_url_is_accepted(self):
        row = copy.deepcopy(self.row)
        row['provenance']['upstream_annotation']['evidence_url'] = (
            'https://raw.githubusercontent.com/alexa/massive/' + 'a' * 40 + '/README.md')
        self.assertEqual(len(self.read([row])), 1)

    def test_existing_local_verification_and_fixture_rules_are_preserved(self):
        row = copy.deepcopy(self.row)
        row['provenance']['kind'] = 'user_annotated'
        row['provenance']['human_verified'] = True
        del row['provenance']['upstream_annotation']
        self.assertEqual(len(self.read([row])), 1)
        row['provenance']['kind'] = 'agent_authored_fixture'
        with self.assertRaisesRegex(ValueError, 'cannot claim human verification'):
            self.read([row])
        row['provenance']['human_verified'] = False
        self.assertEqual(len(self.read([row], formal=False)), 1)
        with self.assertRaisesRegex(ValueError, 'formal data requires'):
            self.read([row])
        row['provenance']['kind'] = 'unverified_external'
        with self.assertRaisesRegex(ValueError, 'formal data requires'):
            self.read([row])


if __name__ == '__main__':
    unittest.main(verbosity=2)
