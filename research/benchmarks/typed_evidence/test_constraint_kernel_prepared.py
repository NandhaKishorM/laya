import itertools
import random
import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from unittest.mock import patch

import constraint_kernel_prepared as candidate
import constraint_kernel_v6 as reference
import audit as study
from constraint_kernel_oracles import document, variable, atom


class PreparedTests(unittest.TestCase):
    def test_immutable_branch_isolation_and_all_exclusions(self):
        atoms = [('a', '<=', 'x', Q(0)), ('x', '<=', 'y', Q(0)),
                 ('y', '<=', 'b', Q(0)), ('x', '!=', 'y', Q(0))]
        state = candidate._prepare(atoms, ('a', 'x', 'y', 'b'))
        before = state.matrix
        self.assertFalse(state.feasible([('b', '<=', 'a', Q(0))]))
        self.assertTrue(state.feasible([('a', '<', 'b', Q(0))]))
        self.assertEqual(state.matrix, before)
        with self.assertRaises(TypeError): state.matrix[0][0] = None
        with self.assertRaises(TypeError): state.index['a'] = 0
        with self.assertRaises(FrozenInstanceError): state.names = ()
        state = candidate._prepare([('a', '!=', 'b', Q(0))], ('a', 'b'))
        for _ in range(2):
            self.assertTrue(state.feasible([('a', '<=', 'b', Q(0))]))
            self.assertTrue(state.feasible([('b', '<=', 'a', Q(0))]))
            self.assertFalse(state.feasible([('a', '<=', 'b', Q(0)), ('b', '<=', 'a', Q(0))]))
            self.assertFalse(state.feasible([('a', '=', 'b', Q(0))]))
            self.assertTrue(state.feasible([('b', '<', 'a', Q(0))]))

    def test_new_exclusions_strictness_and_exact_precision(self):
        state = candidate._prepare([('a', '=', 'b', Q(1, 3))], ('a', 'b'))
        self.assertFalse(state.feasible([('a', '!=', 'b', Q(1, 3))]))
        self.assertFalse(state.feasible([('a', '<', 'b', Q(1, 3))]))
        self.assertTrue(state.feasible([('a', '<=', 'b', Q(1, 3))]))
        n = 10**18
        state = candidate._prepare([('a', '=', None, Q(n, n-1))], ('a',))
        self.assertFalse(state.feasible([('a', '<=', None, Q(1))]))
        self.assertTrue(state.feasible([('a', '>', None, Q(1))]))

    def test_complete_weak_order_theories(self):
        names = ('a', 'b', 'c')
        operations = {'<': lambda a,b:a<b, '<=': lambda a,b:a<=b,
                      '=': lambda a,b:a==b, '!=': lambda a,b:a!=b,
                      '>': lambda a,b:a>b, '>=': lambda a,b:a>=b}
        atoms = [(a, op, b, Q(0)) for a,b in itertools.combinations(names, 2) for op in operations]
        assignments = [dict(zip(names, values)) for values in itertools.product(range(3), repeat=3)]
        def holds(aa, w): return all(operations[op](w[a], w[b]) for a,op,b,_ in aa)
        for size in range(4):
            for evidence in itertools.combinations(atoms, size):
                worlds = [w for w in assignments if holds(evidence, w)]
                state = candidate._prepare(evidence, names)
                self.assertEqual(state is not None, bool(worlds), evidence)
                if state is None: continue
                for query in atoms:
                    expected = any(holds([query], w) for w in worlds)
                    self.assertEqual(state.feasible([query]), expected, (evidence,query))

    def test_seeded_rational_compound_queries(self):
        rng = random.Random(101820681)
        names = ('a', 'b', 'c', 'd', 'e')
        def atom_():
            return rng.choice(names), rng.choice(tuple(reference.NEGATE)), rng.choice(names+(None,)), Q(rng.randint(-4,4), rng.randint(1,5))
        for _ in range(500):
            source = [atom_() for _ in range(rng.randrange(1,15))]
            prepared = candidate._prepare(source, names)
            self.assertEqual(prepared is not None, reference._sat(source))
            if prepared is None: continue
            matrix = prepared.matrix
            queries = [[atom_() for _ in range(rng.randrange(1,9))] for _ in range(12)]
            for query in queries+list(reversed(queries)):
                self.assertEqual(prepared.feasible(query), reference._sat(source+query), (source,query))
                self.assertEqual(prepared.matrix, matrix)

    def test_source_reuse_is_local_and_provenance_exact(self):
        data = dict(task_id='reuse', revision=1,
                    source=document([variable('a'),variable('b'),atom('a','<','b')]),
                    request=document([dict(operation='compare',variables=['a','b'])]),
                    candidate=document([atom('a','<','b'),atom('a','<=','b'),atom('b','>','a')]))
        prepare = candidate._prepare
        with patch.object(candidate, '_prepare', wraps=prepare) as calls:
            for _ in range(2):
                declaration = candidate.TypedDeclaration(**{k:data[k] for k in ('task_id','revision','source','request')})
                actual = candidate.check(**data,declaration=declaration)
                expected = reference.check(**data,declaration=declaration)
                self.assertEqual(study.stable(actual), study.stable(expected))
            self.assertEqual(calls.call_count, 2)
        self.assertIs(candidate.TypedDeclaration, reference.TypedDeclaration)

    def test_independent_strict_bound_oracle(self):
        # Independent Bellman-Ford over symbolic epsilon; enumerate != branches.
        def possible(atoms):
            excluded = [a for a in atoms if a[1] == '!=']
            ordinary = [a for a in atoms if a[1] != '!=']
            for directions in itertools.product(('<', '>'), repeat=len(excluded)):
                expanded = ordinary + [(a, op, b, c) for (a, _, b, c), op in zip(excluded, directions)]
                edges = []
                for a, op, b, c in expanded:
                    if op in ('<', '<=', '='): edges.append((b, a, (c, -int(op == '<'))))
                    if op in ('>', '>=', '='): edges.append((a, b, (-c, -int(op == '>'))))
                nodes = {n for e in edges for n in e[:2]}
                distance = {n: (Q(0), 0) for n in nodes}
                for _ in range(len(nodes)):
                    changed = False
                    for u, v, bound in edges:
                        value = (distance[u][0] + bound[0], distance[u][1] + bound[1])
                        if value < distance[v]: distance[v] = value; changed = True
                    if not changed: return True
                if not edges: return True
            return False
        self.assertFalse(possible([('a', '<', 'a', Q(0))]))
        self.assertFalse(possible([('a', '!=', 'a', Q(0))]))
        self.assertTrue(possible([('a', '>', None, Q(0)), ('a', '<', None, Q(1)),
                                  ('a', '!=', None, Q(1, 2))]))
        rng = random.Random(12821018); names = ('a', 'b', 'c', 'd')
        ops = {'<': lambda a,b:a<b, '<=': lambda a,b:a<=b, '=': lambda a,b:a==b,
               '!=': lambda a,b:a!=b, '>=': lambda a,b:a>=b, '>': lambda a,b:a>b}
        for _ in range(240):
            witness = {n: Q(rng.randrange(-5, 6), rng.randrange(1, 6)) for n in names}
            witness[None] = Q(0); source = []
            while len(source) < 10:
                a, b = rng.choice(names), rng.choice(names + (None,))
                op = rng.choice(tuple(ops))
                q = witness[a] - witness[b] + Q(rng.randrange(-2, 3), rng.randrange(1, 6))
                if op == '!=' and sum(t[1] == '!=' for t in source) >= 2: continue
                if ops[op](witness[a] - witness[b], q): source.append((a, op, b, q))
            state = candidate._prepare(source, rng.sample(names, len(names)))
            self.assertIsNotNone(state); self.assertTrue(possible(source)); snapshot = state.matrix
            for _ in range(12):
                query = [(rng.choice(names), rng.choice(tuple(ops)), rng.choice(names + (None,)),
                          Q(rng.randrange(-3, 4), rng.randrange(1, 6))) for _ in range(rng.randrange(1, 4))]
                self.assertEqual(state.feasible(query), possible(source + query), (source, query))
                self.assertEqual(state.matrix, snapshot)

    def test_full_frozen_outputs(self):
        plan = study.verify()
        for row in plan['rows']+plan['projections']+plan['controls']:
            declaration = study.declaration(row)
            kwargs = dict(row['input'], declaration=declaration, proposal=row.get('proposal'))
            self.assertEqual(study.stable(candidate.check(**kwargs)), study.stable(reference.check(**kwargs)), row['id'])


if __name__ == '__main__': unittest.main()
