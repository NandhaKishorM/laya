"""Offline MASSIVE converter checks using fabricated, test-only source records."""
import copy
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.zh_reliability import prepare_massive as massive
from benchmarks.zh_reliability.data import atomic_json, digest, file_hash, partition, read_data


class MassiveConversionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.source = {'archive_sha256': 'b' * 64}
        self.chinese, self.english = [], []
        for i, split in enumerate(('train', 'train', 'train', 'train', 'dev', 'test')):
            self.add(str(i), split, tuple(massive.INTENTS)[i % 3])

    def add(self, source_id, split, intent='alarm_set', utterance=None, template=None):
        chinese = {
            'id': source_id, 'locale': 'zh-CN', 'partition': split, 'scenario': 'alarm', 'intent': intent,
            'utt': utterance or '仅供测试的合成句子' + source_id,
            'annot_utt': '仅供测试的合成句子' + source_id,
            'judgments': [{'worker_id': str(i), 'intent_score': 1} for i in range(3)],
        }
        english = {**copy.deepcopy(chinese), 'locale': 'en-US',
                   'utt': 'Fabricated source for test only ' + source_id,
                   'annot_utt': template or 'Fabricated source for test only ' + source_id}
        self.chinese.append(chinese)
        self.english.append(english)
        return chinese, english

    def convert(self, **kwargs):
        return massive.convert(self.chinese, self.english, self.source, **kwargs)

    @staticmethod
    def by_source(rows):
        result = {}
        for row in rows:
            source_id = row['provenance']['upstream_annotation']['source_id']
            result.setdefault(source_id, {})[row['question']['type']] = row
        return result

    def test_exact_labels_shared_groups_and_upstream_provenance(self):
        rows, diagnostics, summary = self.convert()
        self.assertEqual(diagnostics, [])
        self.assertEqual(summary['derived_decisions'], 2 * len(self.chinese))
        for source in self.chinese:
            pair = self.by_source(rows)[source['id']]
            choice, noul = pair['choice'], pair['noul']
            self.assertEqual(choice['label'], source['intent'])
            self.assertEqual(choice['question']['criteria'], massive.INTENTS)
            self.assertEqual(noul['label'], 'true' if source['intent'] == 'alarm_set' else 'false')
            self.assertEqual(list(noul['question']['criteria']), ['false', 'true'])
            self.assertEqual(choice['group_id'], noul['group_id'])
            self.assertEqual(choice['split'], noul['split'])
            self.assertEqual(choice['state'], source['utt'])
            self.assertEqual(noul['state'], source['utt'])
            provenance = choice['provenance']
            self.assertIs(provenance['human_verified'], False)
            self.assertIs(provenance['local_human_review'], False)
            self.assertEqual(provenance['kind'], 'public_human_annotated')
            evidence = provenance['upstream_annotation']
            self.assertEqual(evidence['intent'], source['intent'])
            self.assertEqual(evidence['source_row_hash'], digest(source))
            self.assertEqual(evidence['judgments'], source['judgments'])
            self.assertEqual(evidence['evidence_url'], massive.EVIDENCE_URL)
        massive.write_jsonl(self.path / 'derived.jsonl', rows)
        validated = read_data(self.path / 'derived.jsonl', formal=True)
        self.assertEqual(partition(validated, formal=True)['counts'],
                         {s: summary['by_split'][s]['decisions'] for s in summary['by_split']})

    def test_official_dev_test_are_preserved_and_calibration_is_only_from_train(self):
        rows, _, summary = self.convert(calibration_fraction=.5)
        pairs = self.by_source(rows)
        self.assertEqual(pairs['4']['choice']['split'], 'dev')
        self.assertEqual(pairs['5']['choice']['split'], 'test')
        source_splits = {r['id']: r['partition'] for r in self.chinese}
        for source_id, pair in pairs.items():
            if pair['choice']['split'] == 'calibration':
                self.assertEqual(source_splits[source_id], 'train')
            else:
                self.assertEqual(pair['choice']['split'], source_splits[source_id])
        self.assertEqual(summary['by_split']['train']['source_rows'], 2)
        self.assertEqual(summary['by_split']['calibration']['source_rows'], 2)
        self.assertEqual(sum(x['source_rows'] for x in summary['by_split'].values()), summary['retained_source_rows'])
        self.assertEqual(sum(x['decisions'] for x in summary['by_split'].values()), len(rows))
        self.assertEqual(sum(x['groups'] for x in summary['by_split'].values()), len({r['group_id'] for r in rows}))

    def test_english_delexicalized_template_overlap_quarantines_train(self):
        self.add('overlap_train', 'train', template='Test only: set an alarm at [time : seven]')
        self.add('overlap_test', 'test', template='Test only: set an alarm at [time : eleven]')
        rows, diagnostics, summary = self.convert()
        pairs = self.by_source(rows)
        self.assertNotIn('overlap_train', pairs)
        self.assertEqual(pairs['overlap_test']['choice']['split'], 'test')
        excluded = next(r for r in diagnostics if r['source_id'] == 'overlap_train')
        self.assertIn('shares_source_template_or_text_with_test', excluded['reasons'])
        self.assertEqual(excluded['group_id'], pairs['overlap_test']['choice']['group_id'])
        self.assertEqual(summary['retained_source_rows'] + summary['diagnostic_source_rows'], len(self.chinese))

    def test_chinese_normalization_and_transitive_template_grouping(self):
        self.add('chain_train', 'train', utterance='测试：ＡＢＣ！', template='Synthetic bridge one [time : seven]')
        self.add('chain_dev', 'dev', utterance='测 试 abc', template='Synthetic bridge two [time : eight]')
        self.add('chain_test', 'test', utterance='仅供测试的第三句', template='Synthetic bridge two [time : ten]')
        rows, diagnostics, _ = self.convert()
        pairs = self.by_source(rows)
        self.assertNotIn('chain_train', pairs)
        self.assertNotIn('chain_dev', pairs)
        owner_group = pairs['chain_test']['choice']['group_id']
        excluded = [r for r in diagnostics if r['source_id'].startswith('chain_')]
        self.assertEqual({r['source_id'] for r in excluded}, {'chain_train', 'chain_dev'})
        self.assertTrue(all(r['group_id'] == owner_group for r in excluded))
        self.assertTrue(all('shares_source_template_or_text_with_test' in r['reasons'] for r in excluded))

    def test_duplicate_chinese_sources_in_one_partition_share_a_group(self):
        self.add('duplicate_a', 'test', utterance='测试：相同句子。')
        self.add('duplicate_b', 'test', utterance='测试 相同句子')
        rows, diagnostics, summary = self.convert()
        pairs = self.by_source(rows)
        self.assertEqual(pairs['duplicate_a']['choice']['group_id'], pairs['duplicate_b']['choice']['group_id'])
        self.assertEqual(summary['by_split']['test']['source_rows'], 3)
        self.assertEqual(summary['by_split']['test']['groups'], 2)
        self.assertEqual(diagnostics, [])

    def test_conflicting_chinese_labels_are_retained_only_in_diagnostics(self):
        self.add('conflict_a', 'test', intent='alarm_set', utterance='测试：相同但冲突的句子')
        self.add('conflict_b', 'test', intent='alarm_remove', utterance='测试 相同但冲突的句子。')
        rows, diagnostics, summary = self.convert()
        pairs = self.by_source(rows)
        self.assertNotIn('conflict_a', pairs)
        self.assertNotIn('conflict_b', pairs)
        excluded = [r for r in diagnostics if r['source_id'].startswith('conflict_')]
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all('conflicting_intents_for_identical_normalized_chinese' in r['reasons'] for r in excluded))
        self.assertEqual(summary['excluded_reasons']['conflicting_intents_for_identical_normalized_chinese'], 2)

    def test_contested_or_missing_reviews_are_preserved_in_diagnostics(self):
        for source_id, score in (('contested', 0), ('reasonable_only', 2), ('boolean_not_score', True)):
            row, _ = self.add(source_id, 'test')
            row['judgments'][2]['intent_score'] = score
        missing, _ = self.add('missing_review', 'test')
        del missing['judgments']
        duplicate, _ = self.add('duplicate_review', 'test')
        duplicate['judgments'][2] = copy.deepcopy(duplicate['judgments'][0])
        original = copy.deepcopy(self.chinese)
        rows, diagnostics, summary = self.convert()
        expected = {'contested', 'reasonable_only', 'boolean_not_score', 'missing_review', 'duplicate_review'}
        self.assertEqual({r['source_id'] for r in diagnostics}, expected)
        self.assertTrue(expected.isdisjoint(self.by_source(rows)))
        self.assertTrue(all('upstream_intent_review_not_unanimous_or_missing' in r['reasons'] for r in diagnostics))
        source_lookup = {r['id']: r for r in original}
        self.assertTrue(all(d['source_row'] == source_lookup[d['source_id']] for d in diagnostics))
        self.assertEqual(self.chinese, original)
        self.assertEqual(summary['diagnostic_source_rows'], len(expected))

    def test_source_mismatch_is_rejected(self):
        variants = [('locale', 'zh-TW'), ('partition', 'test'), ('intent', 'alarm_remove')]
        for field, value in variants:
            english = copy.deepcopy(self.english)
            english[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'source mismatch'):
                massive.convert(self.chinese, english, self.source)
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            massive.convert(self.chinese, self.english[1:], self.source)
        chinese = copy.deepcopy(self.chinese)
        chinese[0]['locale'] = 'zh-TW'
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            massive.convert(chinese, self.english, self.source)

    def test_deterministic_rerun_and_source_input_not_mutated(self):
        source_before = copy.deepcopy((self.chinese, self.english, self.source))
        first = self.convert(seed=7, calibration_fraction=.5)
        second = self.convert(seed=7, calibration_fraction=.5)
        self.assertEqual(first, second)
        self.assertEqual((self.chinese, self.english, self.source), source_before)
        reversed_result = massive.convert(list(reversed(self.chinese)), list(reversed(self.english)), self.source,
                                          seed=7, calibration_fraction=.5)
        self.assertEqual(first, reversed_result)

    def test_insufficient_groups_and_invalid_calibration_fraction_fail(self):
        for fraction in (0, 1, -.1, 1.1):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, 'calibration fraction'):
                self.convert(calibration_fraction=fraction)
        with self.assertRaisesRegex(ValueError, 'insufficient independent train groups'):
            massive.convert(self.chinese[3:], self.english[3:], self.source)


class MassiveSourceChecksTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def make_source(self):
        source = self.path / 'source'
        source.mkdir()
        for name in massive.SOURCE_FILES:
            text = 'FABRICATED TEST-ONLY DOCUMENT ' + name
            if name == 'NOTICE.md':
                text += '\nCC BY 4.0\n'
            if name == 'DATA_LICENSE':
                text += '\nAttribution 4.0 International\n'
            (source / name).write_text(text, encoding='utf-8')
        archive = source / 'massive-1.1.tar.gz'
        archive.write_bytes(b'FABRICATED TEST-ONLY ARCHIVE')
        metadata = {'archive_url': massive.ARCHIVE_URL, 'publisher_revision': massive.PUBLISHER_REVISION,
                    'archive_sha256': file_hash(archive),
                    'files': {name: file_hash(source / name) for name in massive.SOURCE_FILES}}
        atomic_json(source / 'source_manifest.json', metadata)
        return source, metadata

    def test_source_file_and_archive_checksum_mismatch_are_rejected(self):
        source, metadata = self.make_source()
        self.assertEqual(massive.check_source(source), metadata)
        text = (source / 'zh-CN.jsonl').read_bytes()
        (source / 'zh-CN.jsonl').write_bytes(text + b'changed')
        with self.assertRaisesRegex(ValueError, 'source file hash mismatch: zh-CN.jsonl'):
            massive.check_source(source)
        (source / 'zh-CN.jsonl').write_bytes(text)
        (source / 'massive-1.1.tar.gz').write_bytes(b'changed archive')
        with self.assertRaisesRegex(ValueError, 'source archive hash mismatch'):
            massive.check_source(source)

    def test_version_and_notice_mismatch_are_rejected(self):
        source, metadata = self.make_source()
        for field, value in (('archive_url', 'https://example.com/not-massive'), ('publisher_revision', 'main')):
            changed = {**metadata, field: value}
            atomic_json(source / 'source_manifest.json', changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'unexpected dataset version'):
                massive.check_source(source)
        (source / 'NOTICE.md').write_text('test-only document without required notice')
        metadata['files']['NOTICE.md'] = file_hash(source / 'NOTICE.md')
        atomic_json(source / 'source_manifest.json', metadata)
        with self.assertRaisesRegex(ValueError, 'missing publisher dataset-license notice'):
            massive.check_source(source)

    @staticmethod
    def archive_bytes(extra=None):
        out = io.BytesIO()
        members = [('release/data/zh-CN.jsonl', b'{"test_only":true}\n'),
                   ('release/data/en-US.jsonl', b'{"test_only":true}\n'),
                   ('release/LICENSE', b'FABRICATED TEST ONLY Attribution 4.0 International')]
        if extra:
            members.append(extra)
        with tarfile.open(fileobj=out, mode='w:gz') as archive:
            for name, value in members:
                member = tarfile.TarInfo(name)
                if value is None:
                    member.type = tarfile.SYMTYPE
                    member.linkname = '../../outside'
                    archive.addfile(member)
                else:
                    member.size = len(value)
                    archive.addfile(member, io.BytesIO(value))
        return out.getvalue()

    def responses(self, archive):
        def open_url(url, timeout):
            self.assertGreater(timeout, 0)
            if url == massive.ARCHIVE_URL:
                return io.BytesIO(archive)
            self.assertTrue(url.startswith(massive.RAW_DOCS))
            return io.BytesIO(b'FABRICATED TEST-ONLY DOCUMENT CC BY 4.0')
        return open_url

    def test_download_reads_selected_regular_members_without_extracting_paths(self):
        archive = self.archive_bytes(('../outside', b'not allowed outside source'))
        destination = self.path / 'download'
        with mock.patch.object(massive.urllib.request, 'urlopen', side_effect=self.responses(archive)):
            massive.fetch_source(destination)
        self.assertFalse((self.path / 'outside').exists())
        self.assertFalse((destination / 'release').exists())
        self.assertEqual((destination / 'zh-CN.jsonl').read_text(), '{"test_only":true}\n')
        self.assertEqual(massive.check_source(destination)['archive_sha256'], file_hash(destination / 'massive-1.1.tar.gz'))

    def test_download_rejects_selected_symlinks_and_duplicate_members(self):
        for i, extra in enumerate((('other/data/zh-CN.jsonl', None),
                                   ('other/data/en-US.jsonl', b'duplicate test-only member'))):
            archive = self.archive_bytes(extra)
            with self.subTest(extra=extra), mock.patch.object(massive.urllib.request, 'urlopen',
                                                            side_effect=self.responses(archive)):
                with self.assertRaisesRegex(ValueError, 'unexpected/duplicate archive member'):
                    massive.fetch_source(self.path / ('download_' + str(i)))


class MassiveParquetTests(unittest.TestCase):
    """Fabricated HF-shaped records and transport mocks; no Parquet library or network."""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.features = {
            'scenario': {'_type': 'ClassLabel', 'names': ['music', 'alarm']},
            'intent': {'_type': 'ClassLabel', 'names': ['alarm_query', 'alarm_remove', 'alarm_set']},
        }
        self.row = {
            'id': 'test-only-1', 'locale': 'zh-CN', 'partition': 'dev', 'scenario': 1, 'intent': 2,
            'utt': '仅供测试：请设置ＡＭ七点的闹钟！',
            'annot_utt': '仅供测试：请设置[time : ＡＭ七点]的闹钟！',
            'judgments': {'worker_id': ['0', '1', '2'], 'intent_score': [1, 1, 1],
                          'grammar_score': [3, 4, 3]},
            'slot_method': {'slot': ['time'], 'method': ['localization']},
        }

    def decode(self, row=None, features=None):
        return massive.decode_parquet_row(self.row if row is None else row,
                                          self.features if features is None else features,
                                          'zh-CN', 'validation')

    def test_decoder_uses_embedded_labels_and_transposes_sequences_without_text_changes(self):
        original = copy.deepcopy(self.row)
        result = self.decode()
        self.assertEqual(result['scenario'], 'alarm')
        self.assertEqual(result['intent'], 'alarm_set')
        self.assertEqual(result['utt'], original['utt'])
        self.assertEqual(result['annot_utt'], original['annot_utt'])
        self.assertEqual(result['partition'], 'dev')
        self.assertEqual(result['judgments'], [
            {'worker_id': '0', 'intent_score': 1, 'grammar_score': 3},
            {'worker_id': '1', 'intent_score': 1, 'grammar_score': 4},
            {'worker_id': '2', 'intent_score': 1, 'grammar_score': 3},
        ])
        self.assertEqual(result['slot_method'], [{'slot': 'time', 'method': 'localization'}])
        self.assertEqual(self.row, original)
        self.row['slot_method'] = {'slot': [], 'method': []}
        self.assertEqual(self.decode()['slot_method'], [])

    def test_decoder_rejects_invalid_label_indices_and_feature_type(self):
        for field in ('scenario', 'intent'):
            for value in (-1, len(self.features[field]['names']), True, 1.0, '1', None):
                row = copy.deepcopy(self.row)
                row[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'invalid parquet class label'):
                    self.decode(row)
            features = copy.deepcopy(self.features)
            features[field]['_type'] = 'Value'
            with self.assertRaisesRegex(ValueError, 'invalid parquet class label'):
                self.decode(features=features)

    def test_decoder_rejects_wrong_locale_or_partition(self):
        for field, value in (('locale', 'en-US'), ('partition', 'validation'), ('partition', 'test')):
            row = copy.deepcopy(self.row)
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'locale/partition mismatch'):
                self.decode(row)
        row = copy.deepcopy(self.row)
        row['partition'] = 'train'
        self.assertEqual(massive.decode_parquet_row(row, self.features, 'zh-CN', 'train')['partition'], 'train')

    def test_decoder_rejects_malformed_or_unequal_sequence_columns(self):
        for field in ('judgments', 'slot_method'):
            for value in (None, [], {}, {'one': 'not a list'}, {'one': [1], 'two': [1, 2]}):
                row = copy.deepcopy(self.row)
                row[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'parquet sequence'):
                    self.decode(row)

    def make_source(self):
        directory = self.path / 'source'
        directory.mkdir()
        for name in massive.SOURCE_FILES + massive.PARQUET_FILES:
            text = 'FABRICATED TEST-ONLY SOURCE ' + name
            if name == 'NOTICE.md':
                text += ' CC BY 4.0'
            if name == 'DATA_LICENSE':
                text += ' Attribution 4.0 International'
            (directory / name).write_text(text)
        metadata = {
            'archive_url': massive.ARCHIVE_URL, 'publisher_revision': massive.PUBLISHER_REVISION,
            'transport': 'hf_parquet', 'mirror_revision': massive.PARQUET_REVISION,
            'mirror_base_url': massive.PARQUET_BASE, 'license_text_url': massive.LICENSE_TEXT_URL,
            'files': {name: file_hash(directory / name) for name in massive.SOURCE_FILES + massive.PARQUET_FILES},
        }
        atomic_json(directory / 'source_manifest.json', metadata)
        return directory, metadata

    def test_source_check_requires_fixed_mirror_metadata_and_parquet_hashes(self):
        directory, metadata = self.make_source()
        self.assertEqual(massive.check_source(directory), metadata)
        for field, value in (('mirror_revision', 'main'), ('mirror_base_url', 'https://example.com/'),
                             ('license_text_url', 'https://example.com/LICENSE')):
            atomic_json(directory / 'source_manifest.json', {**metadata, field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'unexpected parquet source revision'):
                massive.check_source(directory)
        atomic_json(directory / 'source_manifest.json', metadata)
        for name in massive.PARQUET_FILES:
            target = directory / name
            original = target.read_bytes()
            target.write_bytes(original + b'CHANGED')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'source file hash mismatch'):
                massive.check_source(directory)
            target.write_bytes(original)

    def test_mirror_fetch_uses_pinned_urls_and_publishes_completed_files(self):
        directory = self.path / 'download'
        expected = {f'{locale}-{split}.parquet': f'{massive.PARQUET_BASE}{locale}/{split}/0000.parquet'
                    for locale in ('zh-CN', 'en-US') for split in ('train', 'validation', 'test')}
        expected.update({name: massive.RAW_DOCS + name for name in massive.SOURCE_FILES[3:]})
        expected['DATA_LICENSE'] = massive.LICENSE_TEXT_URL
        def open_url(url, timeout):
            self.assertIn(url, expected.values())
            self.assertGreater(timeout, 0)
            return io.BytesIO(('FABRICATED TEST-ONLY ' + url).encode())
        with mock.patch.object(massive.urllib.request, 'urlopen', side_effect=open_url) as download, \
                mock.patch.object(massive, 'prepare_parquet_source', return_value=directory) as prepare:
            self.assertEqual(massive.fetch_parquet_source(directory), directory)
        self.assertEqual(download.call_count, len(expected))
        prepare.assert_called_once_with(directory)
        self.assertEqual({p.name for p in directory.iterdir()}, set(expected))
        self.assertFalse(list(directory.glob('*.part')))
        for name, url in expected.items():
            self.assertEqual((directory / name).read_bytes(), ('FABRICATED TEST-ONLY ' + url).encode())

    def test_interrupted_mirror_download_is_not_published_or_prepared(self):
        class InterruptedStream(io.BytesIO):
            def read(self, size=-1):
                data = super().read(size)
                if not data:
                    raise OSError('fabricated download interruption')
                return data
        directory = self.path / 'download'
        with mock.patch.object(massive.urllib.request, 'urlopen', return_value=InterruptedStream(b'test only')), \
                mock.patch.object(massive, 'prepare_parquet_source') as prepare:
            with self.assertRaisesRegex(OSError, 'fabricated download interruption'):
                massive.fetch_parquet_source(directory)
        prepare.assert_not_called()
        self.assertFalse((directory / massive.PARQUET_FILES[0]).exists())
        self.assertEqual((directory / (massive.PARQUET_FILES[0] + '.part')).read_bytes(), b'test only')


if __name__ == '__main__':
    unittest.main(verbosity=2)
