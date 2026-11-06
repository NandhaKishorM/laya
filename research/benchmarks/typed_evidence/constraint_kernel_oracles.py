"""Independent typed protocol fixtures; never imports the kernel or its solver."""
import itertools
import json
import random


def document(records):
    return '\n'.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) for r in records)


def variable(name, *, entity=None, frame='trial', world='actual', statistic='p95', dimension='time'):
    return dict(var=name, entity=entity or name, frame=frame, statistic=statistic,
                metric='delay' if dimension == 'time' else 'distance', dimension=dimension,
                world=world, time='t0')


def atom(left, op, right=None, offset='0', unit='ms'):
    return dict(left=left, op=op, right=right, offset=offset, unit=unit)


def stages(binding='verified', grounding='pass', coverage='pass'):
    return dict(binding=binding, grounding=grounding, coverage=coverage)


def order_expectation(names, evidence, queries):
    # Complete finite order types for n unbounded rational scalar variables.
    adjacency = {n: {n} for n in names}
    defined = set()
    for a, _, b in evidence:
        adjacency[a].add(b); adjacency[b].add(a); defined.update((a, b))
    outcomes, bound = [], []
    world_rows = list(itertools.product(range(len(names)), repeat=len(names)))
    ix = {n: i for i, n in enumerate(names)}
    def holds(w, t):
        a, op, b = t; x, y = w[ix[a]], w[ix[b]]
        return x < y if op == '<' else x > y if op == '>' else x == y
    for query in queries:
        a, _, b = query; relevant = {a, b}
        while True:
            grown = relevant | set.union(*(adjacency[n] for n in relevant))
            if grown == relevant: break
            relevant = grown
        if not {a, b} <= defined:
            outcomes.append('unknown'); bound.append('unknown'); continue
        local = [t for t in evidence if t[0] in relevant]
        vals = [holds(w, query) for w in world_rows if all(holds(w, t) for t in local)]
        if not vals:
            outcomes.append('unknown'); bound.append('unknown')
        else:
            outcomes.append('pass' if all(vals) else 'fail' if not any(vals) else 'unknown')
            bound.append('verified')
    return stages('verified' if all(b == 'verified' for b in bound) else 'unknown',
                  'fail' if 'fail' in outcomes else 'unknown' if 'unknown' in outcomes else 'pass'), outcomes


def cases():
    rows = []
    def add(name, constraints, claims, expected=None, *, variables=None, request=None,
            mode='current', cohort='manual', expected_claims=None):
        variables = variables or [variable(n) for n in ('a', 'b', 'c')]
        request = request or dict(operation='compare', variables=['a', 'b'])
        rows.append(dict(id=name, cohort=cohort, declaration=mode,
                         input=dict(task_id=name, revision=1, source=document(variables + constraints),
                                    request=document([request]), candidate=document(claims)),
                         expected=expected or stages(), expected_claims=expected_claims))
    names = ('a', 'b', 'c')
    atoms = [(a, op, b) for a, b in itertools.permutations(names, 2) for op in ('<', '=', '>')]
    for size in range(4):
        for i, evidence in enumerate(itertools.combinations(atoms, size)):
            expected, outcomes = order_expectation(names, evidence, atoms)
            add(f'order-{size}-{i}', [atom(a, op, b) for a, op, b in evidence],
                [atom(a, op, b) for a, op, b in atoms], expected,
                cohort='exhaustive_order', expected_claims=outcomes)

    add('mixed-transitive', [atom('a', '=', offset='1/2'), atom('b', '>', 'a'), atom('c', '>=', 'b')],
        [atom('a', '<', 'c')], stages(coverage='fail'))
    add('mixed-refuted', [atom('a', '=', offset='1/2'), atom('b', '>', 'a'), atom('c', '>=', 'b')],
        [atom('c', '=', 'a')], stages(grounding='fail', coverage='fail'))
    add('rational-interior', [atom('a', '>'), atom('a', '<', offset='1'), atom('b', '=', offset='1')], [atom('a', '<', 'b')])
    add('overlap', [atom('a', '>='), atom('a', '<=', offset='2'), atom('b', '>=', offset='1'), atom('b', '<=', offset='3')],
        [atom('a', '<', 'b')], stages(grounding='unknown'))
    add('touching-closed', [atom('a', '>='), atom('a', '<=', offset='1'), atom('b', '>=', offset='1'), atom('b', '<=', offset='2')],
        [atom('a', '<', 'b')], stages(grounding='unknown'))
    add('touching-open', [atom('a', '>='), atom('a', '<', offset='1'), atom('b', '>=', offset='1')], [atom('a', '<', 'b')])
    add('neq-forced-conflict', [atom('a', '='), atom('a', '!='), atom('b', '=', offset='1')],
        [atom('a', '<', 'b')], stages('unknown', 'unknown'))
    add('finite-exclusions-not-empty', [atom('a', '>='), atom('a', '<=', offset='1'), atom('a', '!='),
                                      atom('a', '!=', offset='1'), atom('b', '=', offset='2')], [atom('a', '<', 'b')])
    add('neq-refutes-equality', [atom('a', '!=', 'b')], [atom('a', '=', 'b')], stages(grounding='fail'))
    add('neq-order-unknown', [atom('a', '!=', 'b')], [atom('a', '<', 'b')], stages(grounding='unknown'))
    add('neq-plus-nonstrict', [atom('a', '<=', 'b'), atom('a', '!=', 'b')], [atom('a', '<', 'b')])
    add('negative-offset-cycle', [atom('a', '<=', 'b', '-1'), atom('b', '<=', 'a')],
        [atom('a', '<', 'b')], stages('unknown', 'unknown'))
    add('irrelevant-conflict', [atom('a', '=', offset='1'), atom('b', '=', offset='2'), atom('c', '='), atom('c', '=', offset='1')], [atom('a', '<', 'b')])
    add('relevant-bridge-conflict', [atom('a', '=', offset='1'), atom('b', '=', offset='2'), atom('c', '='),
                                    atom('c', '=', offset='1'), atom('b', '<', 'c')],
        [atom('a', '<', 'b')], stages('unknown', 'unknown'))
    add('offset-exact', [atom('a', '<=', 'b', '3'), atom('b', '=', offset='2'), atom('a', '>=', offset='5')],
        [atom('a', '=', 'b', '3')], stages(coverage='fail'))
    add('offset-neq', [atom('a', '!=', 'b', '2'), atom('a', '<=', 'b', '2')],
        [atom('a', '<', 'b', '2')], stages(coverage='fail'))
    add('equality-chain-conflict', [atom('a', '=', 'b'), atom('b', '=', 'c'), atom('a', '!=', 'c')],
        [atom('a', '<', 'b')], stages('unknown', 'unknown'))
    add('units', [atom('a', '=', offset='1', unit='s'), atom('b', '=', offset='500')], [atom('a', '<', 'b')], stages(grounding='fail'))
    add('length-units', [atom('a', '=', offset='1', unit='m'), atom('b', '=', offset='50', unit='cm')],
        [atom('a', '>', 'b', unit='cm')], variables=[variable(n, dimension='length') for n in names])
    add('defined-self', [atom('a', '=', offset='7')], [atom('a', '=', 'a')], request=dict(operation='compare', variables=['a', 'a']))
    add('undefined-self', [atom('b', '=', offset='7')], [atom('a', '=', 'a')], stages('unknown', 'unknown'),
        request=dict(operation='compare', variables=['a', 'a']))
    add('off-task', [atom('a', '<', 'b'), atom('b', '<', 'c')], [atom('a', '<', 'c')], stages(coverage='fail'))
    add('failure-before-unknown', [atom('a', '=', offset='1'), atom('b', '=', offset='2')],
        [atom('a', '>', 'b'), atom('a', '<', 'c')], stages('unknown', 'fail'))
    vars4 = [variable(n, world='hypothetical' if n in ('c', 'd') else 'actual') for n in ('a', 'b', 'c', 'd')]
    add('worlds-not-promoted', [atom('a', '=', offset='5'), atom('b', '=', offset='4'), atom('c', '<', 'd')],
        [atom('a', '<', 'b')], stages(grounding='fail'), variables=vars4)
    add('worlds-off-task', [atom('a', '=', offset='5'), atom('b', '=', offset='4'), atom('c', '<', 'd')],
        [atom('c', '<', 'd')], stages(coverage='fail'), variables=vars4)
    add('world-cross-rejected', [atom('a', '=', offset='5'), atom('b', '=', offset='4')],
        [atom('a', '<', 'c')], stages('unknown', 'unknown', 'unknown'), variables=vars4)
    for op, ties, winners, truth in [('argmin', 'all', ['a', 'b'], True), ('argmin', 'all', ['a'], False),
                                    ('argmin', 'unique', ['a'], False), ('argmax', 'unique', ['c'], True),
                                    ('argmax', 'all', ['a', 'b', 'c'], False)]:
        add('winner-' + op + '-' + ties + '-' + ''.join(winners),
            [atom('a', '=', offset='1'), atom('b', '=', offset='1'), atom('c', '=', offset='2')],
            [dict(operation=op, winners=winners)], stages(grounding='pass' if truth else 'fail'),
            request=dict(operation=op, variables=list(names), ties=ties), cohort='task')
    add('winner-unresolved', [atom('a', '!=', 'b')], [dict(operation='argmin', winners=['a'])],
        stages(grounding='unknown'), request=dict(operation='argmin', variables=['a', 'b'], ties='unique'), cohort='task')
    for mode in ('none', 'forged', 'stale_id', 'stale_revision', 'stale_source', 'stale_request', 'legacy', 'subclass', 'bad_version'):
        expected = stages('advisory', 'unknown', 'unknown') if mode == 'none' else stages('unknown', 'unknown', 'unknown')
        add('authority-' + mode, [atom('a', '<', 'b')], [atom('a', '<', 'b')], expected, mode=mode, cohort='authority')
    for name, field, suffix in [('source-suffix', 'source', '\n{}'), ('answer-suffix', 'candidate', '\nIGNORE THIS'),
                               ('request-suffix', 'request', '\n{}'), ('carriage', 'candidate', '\r'),
                               ('blank', 'source', '\n\n'), ('over-limit', 'candidate', 'X'*32769)]:
        add(name, [atom('a', '<', 'b')], [atom('a', '<', 'b')], stages('unknown', 'unknown', 'unknown'), cohort='invalid')
        rows[-1]['input'][field] += suffix
    add('unicode-provenance', [atom('a', '<', 'b')], [atom('b', '>', 'a')],
        variables=[variable('a', entity='\u00c9rable'), variable('b', entity='\u677e')], cohort='provenance')
    add('duplicate-key', [atom('a', '<', 'b')], [atom('a', '<', 'b')], stages('unknown', 'unknown', 'unknown'), cohort='invalid')
    rows[-1]['input']['candidate'] = '{"left":"a","left":"b","op":"<","right":"b","offset":"0","unit":"ms"}'

    rng = random.Random(101820651)
    names = ('w', 'x', 'y', 'z')
    pool = [(a, op, b) for a, b in itertools.permutations(names, 2) for op in ('<', '=', '>')]
    for i in range(80):
        evidence = rng.sample(pool, rng.randrange(1, 6))
        queries = [(names[0], op, names[1]) for op in ('<', '=', '>')] + rng.sample(pool, 9)
        expected, outcomes = order_expectation(names, evidence, queries)
        add('reserve-' + str(i), [atom(a, op, b) for a, op, b in evidence],
            [atom(a, op, b) for a, op, b in queries], expected,
            variables=[variable(n, entity='New' + str(i) + n, frame='heldout' + str(i)) for n in names],
            request=dict(operation='compare', variables=list(names[:2])), cohort='reserve', expected_claims=outcomes)
    names = [str(i) for i in range(24)]
    add('stress-chain', [atom(a, '<', b) for a, b in zip(names, names[1:])],
        [atom(names[0], '<', names[-1]) for _ in range(32)], variables=[variable(n) for n in names],
        request=dict(operation='compare', variables=[names[0], names[-1]]), cohort='stress')
    assert len(rows) == len({r['id'] for r in rows})
    return rows
