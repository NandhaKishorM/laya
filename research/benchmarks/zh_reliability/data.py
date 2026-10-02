"""Strict fixture/real-data validation and immutable group partitions."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
from collections import Counter

SPLITS = ('train', 'dev', 'calibration', 'test')
FIXTURES = Path(__file__).with_name('fixtures.jsonl')


def digest(value):
    # Preserve criteria insertion order: order is part of the actual model input.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def options(row):
    return ['false', 'true'] if row['question']['type'] == 'noul' else list(row['question']['criteria'])


def upstream_annotation_verified(provenance):
    """Validate supplied source-review evidence, not a local review of the derived task."""
    if provenance.get('kind') != 'public_human_annotated' or provenance.get('human_verified') is not False:
        return False
    evidence = provenance.get('upstream_annotation')
    if not isinstance(evidence, dict):
        return False
    if (evidence.get('dataset') != 'MASSIVE' or evidence.get('version') != '1.1'
            or not isinstance(evidence.get('source_id'), str) or not evidence['source_id'].strip()
            or not isinstance(provenance.get('source_intent'), str) or not provenance['source_intent'].strip()
            or evidence.get('intent') != provenance['source_intent']):
        return False
    source_hash = evidence.get('source_row_hash')
    evidence_url = evidence.get('evidence_url')
    if (not isinstance(source_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', source_hash)
            or not isinstance(evidence_url, str) or not re.fullmatch(
                r'https://(?:github\.com/alexa/massive/blob/|raw\.githubusercontent\.com/alexa/massive/)'
                r'[0-9a-f]{40}/README\.md(?:#[A-Za-z0-9_-]+)?', evidence_url)):
        return False
    judgments = evidence.get('judgments')
    if not isinstance(judgments, list) or len(judgments) < 3:
        return False
    workers = set()
    for judgment in judgments:
        if (not isinstance(judgment, dict) or not isinstance(judgment.get('worker_id'), str)
                or not judgment['worker_id'].strip() or judgment['worker_id'] in workers
                or type(judgment.get('intent_score')) is not int or judgment['intent_score'] != 1):
            return False
        workers.add(judgment['worker_id'])
    return True


def read_data(path, formal=False):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('empty dataset')
    seen, tasks, families, relations = set(), {}, {}, {}
    for r in rows:
        for key in ('id', 'group_id', 'task_id', 'state'):
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError('nonempty string required: ' + key)
        if r['id'] in seen:
            raise ValueError('duplicate id: ' + r['id'])
        seen.add(r['id'])
        q = r.get('question', {})
        if q.get('type') not in ('choice', 'noul') or not isinstance(q.get('instructions'), str):
            raise ValueError('only choice/noul with string instructions are supported')
        c = q.get('criteria')
        if not isinstance(c, dict) or len(c) < 2 or any(not isinstance(v, str) for v in c.values()):
            raise ValueError('criteria must map at least two stable option IDs to string descriptions')
        if any(not isinstance(k, str) or not k.strip() for k in c):
            raise ValueError('invalid option_id')
        if q['type'] == 'noul' and set(c) != {'false', 'true'}:
            raise ValueError('noul criteria must be [false, true]')
        if 'labels' in q:
            labels = q['labels']
            if (q['type'] != 'noul' or not isinstance(labels, dict) or set(labels) != {'false', 'true'}
                    or any(not isinstance(x, str) or not x.strip() for x in labels.values())
                    or len(set(labels.values())) != 2):
                raise ValueError('invalid noul display labels')
        if r.get('label') not in options(r):
            raise ValueError('label not in semantic options')
        if not isinstance(r.get('tags'), list) or any(not isinstance(t, str) for t in r['tags']):
            raise ValueError('tags must be a list of strings')
        p = r.get('provenance', {})
        if not isinstance(p.get('kind'), str) or type(p.get('human_verified')) is not bool:
            raise ValueError('provenance kind and human_verified required')
        if p['kind'].startswith('agent_authored') and p['human_verified']:
            raise ValueError('agent fixtures cannot claim human verification')
        upstream_verified = upstream_annotation_verified(p)
        if p['kind'] == 'public_human_annotated' and not upstream_verified:
            raise ValueError('public human annotations require uncontested pinned source-review evidence')
        if formal and (p['kind'].startswith('agent_authored') or not (p['human_verified'] or upstream_verified)
                       or any(not isinstance(p.get(k), str) or not p[k].strip()
                              for k in ('source', 'license', 'annotation_status'))
                       or r.get('split') not in SPLITS):
            raise ValueError('formal data requires source, license, verified annotation and fixed split')
        task = (q['type'], tuple(sorted(c.items())), q['instructions'])
        if r['task_id'] in tasks and tasks[r['task_id']] != task:
            raise ValueError('task_id must have consistent semantic criteria/instructions')
        tasks[r['task_id']] = task
        family = p.get('template_family', r['group_id'])
        if family in families and families[family] != r['group_id']:
            raise ValueError('template family crosses groups')
        families[family] = r['group_id']
        rel = r.get('relation')
        if rel:
            if rel.get('kind') not in ('preserve', 'flip') or not isinstance(rel.get('pair_id'), str):
                raise ValueError('invalid relation')
            relations.setdefault(rel['pair_id'], []).append(r)
    for pair in relations.values():
        if len(pair) != 2 or len({r['group_id'] for r in pair}) != 1:
            raise ValueError('each relation must pair exactly two rows in the same group')
        if len({r['relation']['kind'] for r in pair}) != 1 or len({r['task_id'] for r in pair}) != 1:
            raise ValueError('inconsistent pair')
        if (pair[0]['label'] == pair[1]['label']) != (pair[0]['relation']['kind'] == 'preserve'):
            raise ValueError('relation contradicts labels')
    return rows


def partition(rows, path=None, seed=20260924, formal=False):
    hashes = {r['id']: digest(r) for r in rows}
    content_hash = digest(hashes)
    groups = sorted({r['group_id'] for r in rows})
    if path and Path(path).exists():
        out = json.loads(Path(path).read_text())
        if out['data_hash'] != content_hash or out['row_hashes'] != hashes:
            raise ValueError('split manifest data hash mismatch')
    else:
        if formal:
            mapping = {}
            for r in rows:
                if r['group_id'] in mapping and mapping[r['group_id']] != r['split']:
                    raise ValueError('fixed partitions split a group')
                mapping[r['group_id']] = r['split']
        else:
            random.Random(seed).shuffle(groups)
            n = len(groups)
            cuts = [round(n * .6), round(n * .7), round(n * .85), n]
            mapping, start = {}, 0
            for split, end in zip(SPLITS, cuts):
                mapping.update({g: split for g in groups[start:end]})
                start = end
        out = {'seed': seed, 'formal': formal, 'data_hash': content_hash, 'row_hashes': hashes,
               'groups': mapping, 'assignments': {r['id']: mapping[r['group_id']] for r in rows}}
    if set(out['groups']) != set(groups) or set(out['assignments']) != set(hashes):
        raise ValueError('split manifest membership mismatch')
    for r in rows:
        s = out['assignments'][r['id']]
        if s not in SPLITS or s != out['groups'][r['group_id']] or (formal and s != r['split']):
            raise ValueError('group leakage or fixed split mismatch')
    if out['formal'] != formal:
        raise ValueError('split mode mismatch')
    out['counts'] = dict(Counter(out['assignments'].values()))
    out['group_counts'] = dict(Counter(out['groups'].values()))
    split_hash = digest(out['assignments'])
    if 'split_hash' in out and out['split_hash'] != split_hash:
        raise ValueError('split manifest assignment hash mismatch')
    out['split_hash'] = split_hash
    if any(not out['counts'].get(s) for s in SPLITS):
        raise ValueError('all four partitions must be nonempty')
    if path and not Path(path).exists():
        atomic_json(path, out)
    return out
