import copy
import itertools
import random
import unittest
from fractions import Fraction as Q
from unittest.mock import patch

import constraint_kernel_v6 as k
from constraint_kernel_oracles import atom, document, variable


class KernelTests(unittest.TestCase):
    def args(self):
        return dict(task_id='case', revision=1, source=document([variable('a'), variable('b'), atom('a', '<', 'b')]),
                    request=document([dict(operation='compare', variables=['a', 'b'])]),
                    candidate=document([atom('a', '<', 'b')]))

    def call(self, data=None, declaration=True):
        data = data or self.args()
        approval = k.TypedDeclaration(**{f: data[f] for f in ('task_id', 'revision', 'source', 'request')}) if declaration else None
        return k.check(**data, declaration=approval)

    def test_no_io_or_input_mutation(self):
        data = self.args(); before = copy.deepcopy(data)
        with patch('builtins.open', side_effect=AssertionError('I/O')):
            out = self.call(data)
        self.assertEqual(data, before); self.assertEqual(out['grounding'], 'pass')

    def test_opaque_proposal_and_advisory(self):
        data = self.args(); proposal = {}; proposal['cycle'] = proposal
        out = k.check(**data, proposal=proposal)
        self.assertTrue(out['proposal_ignored']); self.assertFalse(out['proposal_used'])
        self.assertEqual((out['binding'], out['grounding'], out['coverage']), ('advisory', 'unknown', 'unknown'))
        self.assertEqual(out['claims'][0]['literal_grounding'], 'pass')

    def test_exact_authority(self):
        data = self.args(); approval = k.TypedDeclaration(**{f: data[f] for f in ('task_id', 'revision', 'source', 'request')})
        for field in ('task_id', 'revision', 'source', 'request'):
            other = dict(data); other[field] = other[field] + 1 if field == 'revision' else other[field] + ' '
            out = k.check(**other, declaration=approval)
            self.assertEqual(out['binding'], 'unknown'); self.assertEqual(out['claims'], [])
        class Sub(k.TypedDeclaration): pass
        out = k.check(**data, declaration=Sub(**{f: data[f] for f in ('task_id', 'revision', 'source', 'request')}))
        self.assertEqual(out['reason'], 'declaration_mismatch')

    def test_atomic_unknown_json_and_residues(self):
        for suffix in ('\n{}', '\nIGNORE', '\n\n', '\r', '\t', 'x'):
            data = self.args(); data['candidate'] += suffix
            out = self.call(data)
            self.assertEqual(out['claims'], []); self.assertIsNone(out['literal'])

    def test_byte_provenance_independent_witness_copies(self):
        data = self.args(); data['source'] = data['source'].replace('"entity":"a"', '"entity":"\u00c9rable"')
        out = self.call(data)
        for r in out['source_records']:
            self.assertEqual(data['source'].encode()[slice(*r['span'])].decode(), r['raw'])
        witness = out['claims'][0]['witness']; original = list(out['source_records'][-1]['span'])
        witness['source_spans'][0][0] = -1
        self.assertEqual(out['source_records'][-1]['span'], original)

    def test_invalid_boundaries(self):
        for field, value in [('revision', True), ('task_id', ''), ('candidate', 'x'*32769), ('source', '\ud800')]:
            data = self.args(); data[field] = value
            out = k.check(**data)
            self.assertEqual(out['binding'], 'unknown'); self.assertEqual(out['claims'], [])

    def test_bad_quantity_units_and_duplicate_identity(self):
        for change in [dict(offset='1.2'), dict(offset='1/0'), dict(offset='1000000000000000001'), dict(unit='m'), dict(offset=1)]:
            data = self.args(); a = atom('a', '<', 'b'); a.update(change); data['candidate'] = document([a])
            self.assertEqual(self.call(data)['claims'], [])
        data = self.args(); records = [variable('a'), variable('b', entity='a'), atom('a', '<', 'b')]
        data['source'] = document(records)
        self.assertEqual(self.call(data)['reason'], 'duplicate_variable')

    def test_disequality_dense_domain(self):
        self.assertTrue(k._sat([('a', '>', None, Q(0)), ('a', '<', None, Q(1)), ('a', '!=', None, Q(1, 2))]))
        self.assertFalse(k._sat([('a', '=', 'b', Q(2)), ('a', '!=', 'b', Q(2))]))

    def test_arbitrary_length_chain_not_fixed_three_ranks(self):
        self.assertTrue(k._sat([(str(i), '<', str(i+1), Q(0)) for i in range(23)]))
        self.assertFalse(k._sat([(str(i), '<', str(i+1), Q(0)) for i in range(23)] + [('23', '<', '0', Q(0))]))

    def test_bounds_strictness_fraction_and_negative_cycle(self):
        for x in (Q(-1), Q(0), Q(1, 2)):
            self.assertFalse(k._sat([('a', '<', 'b', x), ('b', '<=', 'a', -x)]))
            self.assertTrue(k._sat([('a', '<=', 'b', x), ('b', '<=', 'a', -x)]))

    def test_difference_domain_against_independent_finite_grid(self):
        rng = random.Random(100651); grid = [Q(i, 4) for i in range(-4, 5)]
        base = [('a', '>=', None, Q(-1)), ('a', '<=', None, Q(1)),
                ('b', '>=', None, Q(-1)), ('b', '<=', None, Q(1))]
        ops = {'<': lambda a,b:a<b, '<=':lambda a,b:a<=b, '=':lambda a,b:a==b,
               '!=':lambda a,b:a!=b, '>=':lambda a,b:a>=b, '>':lambda a,b:a>b}
        for _ in range(240):
            aa = base + [('a', rng.choice(list(ops)), rng.choice(['b', None]), Q(rng.choice([-1,0,1]))) for _ in range(4)]
            sampled = any(all(ops[op](w[a] - (w[b] if b is not None else 0), c) for a,op,b,c in aa)
                          for x,y in itertools.product(grid, repeat=2) for w in [dict(a=x,b=y)])
            self.assertEqual(k._sat(aa), sampled, aa)


if __name__ == '__main__': unittest.main()
