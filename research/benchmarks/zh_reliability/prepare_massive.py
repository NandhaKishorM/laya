"""Prepare a licensed, source-reviewed MASSIVE 1.1 Chinese alarm subset (no model calls)."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import shutil
import sys
import tarfile
import unicodedata
import urllib.request

from .data import atomic_json, digest, file_hash, partition, read_data

PUBLISHER_REVISION = 'f966f21846043aabef9b0f974fa7970027f43738'
ARCHIVE_URL = 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz'
EVIDENCE_URL = f'https://github.com/alexa/massive/blob/{PUBLISHER_REVISION}/README.md'
RAW_DOCS = f'https://raw.githubusercontent.com/alexa/massive/{PUBLISHER_REVISION}/'
PARQUET_REVISION = 'ed58ac423a2f4121720918bf5301577edce4ffd3'
PARQUET_BASE = f'https://huggingface.co/datasets/AmazonScience/massive/resolve/{PARQUET_REVISION}/'
PARQUET_FILES = tuple(f'{locale}-{split}.parquet' for locale in ('zh-CN', 'en-US')
                      for split in ('train', 'validation', 'test'))
# The publisher NOTICE establishes that this license applies to MASSIVE.
LICENSE_TEXT_URL = 'https://raw.githubusercontent.com/creativecommons/cc-legal-tools-data/b40ad648826d12d38207b111d66a708bfa6c5ce4/docs/licenses/by/4.0/legalcode.txt'
INTENTS = {'alarm_set': '设置闹钟', 'alarm_query': '查询闹钟', 'alarm_remove': '删除或取消闹钟'}
SOURCE_FILES = ('zh-CN.jsonl', 'en-US.jsonl', 'DATA_LICENSE', 'README.md', 'NOTICE.md', 'THIRD-PARTY.md', 'LICENSE.txt')


def fetch_source(directory):
    directory = Path(directory)
    directory.mkdir()
    archive = directory / 'massive-1.1.tar.gz'
    with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as source, archive.with_suffix('.part').open('wb') as out:
        shutil.copyfileobj(source, out)
    archive.with_suffix('.part').replace(archive)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = member.name
            target = None
            if name.endswith('/data/zh-CN.jsonl'):
                target = 'zh-CN.jsonl'
            elif name.endswith('/data/en-US.jsonl'):
                target = 'en-US.jsonl'
            elif name.endswith('/LICENSE'):
                target = 'DATA_LICENSE'
            if target:
                if not member.isfile() or (directory / target).exists():
                    raise ValueError('unexpected/duplicate archive member: ' + name)
                # Read only exact members; never extract arbitrary archive paths/symlinks.
                with tar.extractfile(member) as src, (directory / target).open('wb') as out:
                    shutil.copyfileobj(src, out)
    for name in SOURCE_FILES[3:]:
        with urllib.request.urlopen(RAW_DOCS + name, timeout=30) as src:
            (directory / name).write_bytes(src.read())
    metadata = {'archive_url': ARCHIVE_URL, 'archive_sha256': file_hash(archive),
                'publisher_revision': PUBLISHER_REVISION,
                'files': {name: file_hash(directory / name) for name in SOURCE_FILES}}
    atomic_json(directory / 'source_manifest.json', metadata)
    return directory


def check_source(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'source_manifest.json').read_text())
    if metadata['archive_url'] != ARCHIVE_URL or metadata['publisher_revision'] != PUBLISHER_REVISION:
        raise ValueError('unexpected dataset version or documentation revision')
    transport = metadata.get('transport', 'publisher_archive')
    if transport not in ('publisher_archive', 'hf_parquet'):
        raise ValueError('unexpected source transport')
    if transport == 'hf_parquet' and (metadata.get('mirror_revision') != PARQUET_REVISION
            or metadata.get('mirror_base_url') != PARQUET_BASE
            or metadata.get('license_text_url') != LICENSE_TEXT_URL):
        raise ValueError('unexpected parquet source revision or license URL')
    source_files = SOURCE_FILES + (PARQUET_FILES if transport == 'hf_parquet' else ())
    for name in source_files:
        if metadata['files'].get(name) != file_hash(directory / name):
            raise ValueError('source file hash mismatch: ' + name)
    archive = directory / 'massive-1.1.tar.gz'
    if archive.exists() and file_hash(archive) != metadata['archive_sha256']:
        raise ValueError('source archive hash mismatch')
    if 'CC BY 4.0' not in (directory / 'NOTICE.md').read_text():
        raise ValueError('missing publisher dataset-license notice')
    if 'Attribution 4.0 International' not in (directory / 'DATA_LICENSE').read_text():
        raise ValueError('unexpected dataset license')
    return metadata


def decode_parquet_row(row, features, locale, split):
    """Decode HF label indices and column-oriented sequences without changing source text."""
    row = dict(row)
    if row.get('locale') != locale or row.get('partition') != ('dev' if split == 'validation' else split):
        raise ValueError('parquet locale/partition mismatch')
    for field in ('scenario', 'intent'):
        feature, value = features[field], row[field]
        names = feature.get('names', [])
        if feature.get('_type') != 'ClassLabel' or type(value) is not int or not 0 <= value < len(names):
            raise ValueError('invalid parquet class label: ' + field)
        row[field] = names[value]
    for field in ('judgments', 'slot_method'):
        value = row.get(field)
        if not isinstance(value, dict) or not value or any(not isinstance(v, list) for v in value.values()):
            raise ValueError('invalid parquet sequence: ' + field)
        if len({len(v) for v in value.values()}) != 1:
            raise ValueError('unequal parquet sequence lengths: ' + field)
        row[field] = [dict(zip(value, values)) for values in zip(*value.values())]
    return row


def prepare_parquet_source(directory):
    """Decode six pinned files and record artifact and decoded hashes; no network calls."""
    import pyarrow
    import pyarrow.parquet as pq
    directory = Path(directory)
    for locale in ('zh-CN', 'en-US'):
        rows = []
        for split in ('train', 'validation', 'test'):
            table = pq.read_table(directory / f'{locale}-{split}.parquet')
            features = json.loads(table.schema.metadata[b'huggingface'])['info']['features']
            rows.extend(decode_parquet_row(row, features, locale, split) for row in table.to_pylist())
        write_jsonl(directory / f'{locale}.jsonl', rows)
    metadata = {'archive_url': ARCHIVE_URL, 'publisher_revision': PUBLISHER_REVISION,
        'transport': 'hf_parquet', 'mirror_revision': PARQUET_REVISION, 'mirror_base_url': PARQUET_BASE,
        'license_text_url': LICENSE_TEXT_URL, 'pyarrow_version': pyarrow.__version__,
        'decoding': 'ClassLabel indices use embedded feature names; column-oriented judgments/slot_method are transposed to lists. Row hashes cover decoded JSON, not original archive bytes.',
        'files': {name: file_hash(directory / name) for name in SOURCE_FILES + PARQUET_FILES}}
    atomic_json(directory / 'source_manifest.json', metadata)
    return directory


def fetch_parquet_source(directory):
    directory = Path(directory)
    directory.mkdir()
    downloads = {f'{locale}-{split}.parquet': f'{PARQUET_BASE}{locale}/{split}/0000.parquet'
                 for locale in ('zh-CN', 'en-US') for split in ('train', 'validation', 'test')}
    downloads.update({name: RAW_DOCS + name for name in SOURCE_FILES[3:]})
    downloads['DATA_LICENSE'] = LICENSE_TEXT_URL
    for name, url in downloads.items():
        target = directory / name
        temporary = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(url, timeout=60) as src, temporary.open('wb') as out:
            shutil.copyfileobj(src, out)
        temporary.replace(target)
    return prepare_parquet_source(directory)


def normalized(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def source_template(row):
    # Slot values in English source annotations must not create independent groups.
    return normalized(re.sub(r'\[\s*([^:\]]+?)\s*:\s*[^\]]*\]', r'[\1]', row['annot_utt']))


def records(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate source ID')
    return rows


def confirmed(row):
    judgments = row.get('judgments', [])
    return (isinstance(judgments, list) and len(judgments) >= 3
            and all(isinstance(j, dict) and isinstance(j.get('worker_id'), str) and j['worker_id'].strip()
                    and type(j.get('intent_score')) is int and j['intent_score'] == 1 for j in judgments)
            and len({j['worker_id'] for j in judgments}) == len(judgments))


def convert(chinese, english, source, seed=20260924, calibration_fraction=.20):
    if not 0 < calibration_fraction < 1:
        raise ValueError('calibration fraction must be between zero and one')
    en = {r['id']: r for r in english}
    selected = [r for r in chinese if r.get('scenario') == 'alarm' and r.get('intent') in INTENTS]
    if not selected or len({r['id'] for r in selected}) != len(selected):
        raise ValueError('empty alarm selection or duplicate source IDs')
    parent = {r['id']: r['id'] for r in selected}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    anchors, labels = {}, defaultdict(set)
    for r in selected:
        e = en.get(r['id'])
        if (r.get('locale') != 'zh-CN' or r.get('partition') not in ('train', 'dev', 'test')
                or not isinstance(r.get('utt'), str) or not normalized(r['utt']) or not e
                or e.get('locale') != 'en-US' or e['partition'] != r['partition'] or e['intent'] != r['intent']):
            raise ValueError('Chinese/English source mismatch: ' + str(r['id']))
        chinese_key = normalized(r['utt'])
        labels[chinese_key].add(r['intent'])
        for key in (('zh', chinese_key), ('en_template', source_template(e))):
            if key in anchors:
                parent[root(r['id'])] = root(anchors[key])
            else:
                anchors[key] = r['id']
    groups = defaultdict(list)
    for r in selected:
        groups[root(r['id'])].append(r)
    decisions, diagnostic = [], []
    priority = {'train': 0, 'dev': 1, 'test': 2}
    for members in groups.values():
        owner = max((r['partition'] for r in members), key=priority.get)
        group_id = 'massive-alarm-' + digest(sorted(r['id'] for r in members))[:20]
        conflict = any(len(labels[normalized(r['utt'])]) > 1 for r in members)
        for r in members:
            reasons = []
            if r['partition'] != owner:
                reasons.append('shares_source_template_or_text_with_' + owner)
            if conflict:
                reasons.append('conflicting_intents_for_identical_normalized_chinese')
            if not confirmed(r):
                reasons.append('upstream_intent_review_not_unanimous_or_missing')
            if reasons:
                diagnostic.append({'source_id': r['id'], 'group_id': group_id, 'reasons': reasons,
                                   'original_partition': r['partition'], 'source_row': r})
            else:
                decisions.append((r, group_id, owner))
    train_groups = sorted({g for _, g, s in decisions if s == 'train'})
    if len(train_groups) < 2:
        raise ValueError('insufficient independent train groups after source-leakage quarantine')
    random.Random(seed).shuffle(train_groups)
    cal_groups = set(train_groups[:max(1, min(len(train_groups) - 1, round(len(train_groups) * calibration_fraction)))])
    output = []
    for r, group_id, original_split in sorted(decisions, key=lambda x: x[0]['id']):
        split = 'calibration' if group_id in cal_groups else original_split
        provenance = {'kind': 'public_human_annotated', 'human_verified': False,
            'source': ARCHIVE_URL, 'license': 'CC-BY-4.0', 'source_intent': r['intent'],
            'annotation_status': 'Unanimous upstream human intent-match judgments; derived mapping has no independent local human review',
            'local_human_review': False, 'template_family': group_id,
            'source_partition': original_split, 'source_manifest_hash': digest(source),
            'source_transport': source.get('transport', 'publisher_archive'),
            'upstream_annotation': {'dataset': 'MASSIVE', 'version': '1.1', 'source_id': r['id'],
                'intent': r['intent'], 'source_row_hash': digest(r), 'evidence_url': EVIDENCE_URL,
                'judgments': r['judgments']}}
        if 'archive_sha256' in source:
            provenance['source_archive_sha256'] = source['archive_sha256']
        if source.get('transport') == 'hf_parquet':
            provenance['source_mirror_revision'] = source['mirror_revision']
        common = {'group_id': group_id, 'state': r['utt'], 'split': split,
                  'tags': ['public_localized', 'alarm'], 'provenance': provenance}
        output.append({**common, 'id': 'massive-' + r['id'] + '-choice', 'task_id': 'massive_alarm_choice_v1',
            'question': {'type': 'choice', 'instructions': '这句话的主要意图是哪一种闹钟操作？', 'criteria': dict(INTENTS)},
            'label': r['intent']})
        output.append({**common, 'id': 'massive-' + r['id'] + '-noul', 'task_id': 'massive_alarm_set_noul_v1',
            'question': {'type': 'noul', 'instructions': '这句话的主要意图是否是设置闹钟？',
                         'criteria': {'false': '意图是查询、删除或取消闹钟', 'true': '意图是设置闹钟'}},
            'label': 'true' if r['intent'] == 'alarm_set' else 'false'})
    summary = {'source_zh_rows': len(chinese), 'selected_alarm_source_rows': len(selected),
        'retained_source_rows': len(decisions), 'diagnostic_source_rows': len(diagnostic),
        'source_groups_before_quarantine': len(groups), 'derived_decisions': len(output),
        'source_partition_counts': dict(Counter(r['partition'] for r in selected)),
        'excluded_reasons': dict(Counter(reason for r in diagnostic for reason in r['reasons'])),
        'seed': seed, 'calibration_fraction_of_remaining_train_groups': calibration_fraction,
        'local_human_review': False, 'scope': 'MASSIVE-derived source-reviewed alarm subset; not official full benchmark or natural customer-service logs',
        'grouping_limit': 'source IDs, exact normalized Chinese and delexicalized English templates; undisclosed semantic/template relations may remain',
        'base_model_training_overlap': 'unknown; no unseen-data claim',
        'by_split': {s: {'decisions': sum(r['split'] == s for r in output),
                        'source_rows': sum(('calibration' if g in cal_groups else t) == s for _, g, t in decisions),
                        'groups': len({r['group_id'] for r in output if r['split'] == s}),
                        'source_intents': dict(Counter(r['label'] for r in output if r['split'] == s and r['question']['type'] == 'choice'))}
                     for s in ('train', 'dev', 'calibration', 'test')}}
    return output, diagnostic, summary


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, help='existing source with source_manifest.json; default offline')
    p.add_argument('--download', action='store_true', help='download publisher release into this new run')
    p.add_argument('--download-parquet', action='store_true', help='download only zh-CN/en-US from pinned publisher HF mirror; requires pyarrow')
    p.add_argument('--run-dir', type=Path, required=True, help='new output directory')
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--calibration-fraction', type=float, default=.20)
    args = p.parse_args(argv)
    if sum((bool(args.source_dir), args.download, args.download_parquet)) != 1:
        p.error('choose exactly one of --source-dir, --download or --download-parquet')
    run = args.run_dir
    if run.exists():
        print('refusing existing output directory: ' + str(run), file=sys.stderr)
        return 2
    run.mkdir(parents=True)
    manifest = {'status': 'RUNNING', 'command': 'prepare-massive', 'gpu_experiment': 'NOT_RUN',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'completed': [],
                'converter_sha256': file_hash(__file__),
                'validation_sha256': file_hash(Path(__file__).with_name('data.py')),
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    atomic_json(run / 'manifest.json', manifest)
    stage = 'source'
    try:
        if args.download:
            source_dir = fetch_source(run / 'source')
        elif args.download_parquet:
            source_dir = fetch_parquet_source(run / 'source')
        else:
            source_dir = args.source_dir
        source = check_source(source_dir)
        atomic_json(run / 'source_manifest.json', source)
        license_dir = run / 'attribution'
        license_dir.mkdir()
        for name in SOURCE_FILES[2:]:
            shutil.copyfile(source_dir / name, license_dir / name)
        (license_dir / 'DERIVATION.md').write_text(
            '# Derived MASSIVE alarm subset\n\n'
            'MASSIVE Copyright Amazon.com, Inc. or its affiliates. Source: ' + ARCHIVE_URL + '\n\n'
            'Dataset and underlying SLURP text: CC BY 4.0; see NOTICE.md and DATA_LICENSE.\n\n'
            'Changes: select three alarm intents with unanimous upstream intent-match reviews; '
            'quarantine template overlaps and conflicting normalized utterances; derive choice and noul '
            'questions; split calibration groups from original train. Original Chinese text and '
            'intent labels remain unchanged. No local human review was performed.\n\n'
            'Transport and decoding: see ../source_manifest.json. For Parquet, DATA_LICENSE is '
            'the CC legal text fetched from its recorded official URL, not archive/LICENSE bytes.\n',
            encoding='utf-8')
        manifest['completed'].append('source hashes and publisher attribution')
        stage = 'conversion'
        rows, diagnostics, summary = convert(records(source_dir / 'zh-CN.jsonl'), records(source_dir / 'en-US.jsonl'),
                                               source, args.seed, args.calibration_fraction)
        write_jsonl(run / 'data.jsonl', rows)
        write_jsonl(run / 'diagnostic.jsonl', diagnostics)
        # Full normal importer validation is a gate, not just converter-specific assertions.
        validated = read_data(run / 'data.jsonl', formal=True)
        split = partition(validated, run / 'split_manifest.json', args.seed, formal=True)
        summary.update(data_hash=split['data_hash'], split_hash=split['split_hash'])
        atomic_json(run / 'metrics.json', summary)
        manifest.update(status='SUCCESS', data_sha256=file_hash(run / 'data.jsonl'),
                        diagnostic_sha256=file_hash(run / 'diagnostic.jsonl'))
        manifest['completed'].append('source-reviewed conversion and fixed group partition validation')
        atomic_json(run / 'manifest.json', manifest)
        (run / 'report.md').write_text('# MASSIVE Chinese alarm data preparation\n\n'
            'Public human-localized data; no local human review or model evaluation has been performed.\n\n'
            'The original Chinese utterances and publisher intent labels are preserved. '
            'Choice and noul are deterministic views of the same source and share a group. '
            'Official test/dev membership is retained for eligible rows; lower-partition template overlaps '
            'and contested/missing human reviews remain in diagnostic.jsonl. This is a filtered subset, '
            'not the full MASSIVE benchmark.\n\n```json\n' + json.dumps(summary, ensure_ascii=False, indent=2)
            + '\n```\n\nAttribution: Amazon MASSIVE and SLURP, CC BY 4.0; see attribution/.\n', encoding='utf-8')
        print(str(run.resolve()))
        return 0
    except Exception as e:
        manifest.update(status='FAILED', error={'stage': stage, 'type': type(e).__name__, 'reason': str(e)})
        atomic_json(run / 'manifest.json', manifest)
        print(f'{stage}: {e}; see {run / "manifest.json"}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
