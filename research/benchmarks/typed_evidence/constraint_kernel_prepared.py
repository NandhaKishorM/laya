"""Candidate: immutable exact source closure with branch-local query updates.

The external v6 contract and declaration class are unchanged. No persistent cache.
"""
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
import hashlib
import time

from constraint_kernel_v6 import (
    VERSION, UNITS, NEGATE, VAR_KEYS, TypedDeclaration, Unsupported,
    _metadata, _text, _records, _label, _frame, _atom, _compatible,
    _hash, _normalized, _less,
)


def _edges(atoms, index):
    edges, excluded = [], []
    for a, op, b, c in atoms:
        if op in ('<', '<=', '='):
            edges.append((index[b], index[a], (c, op == '<')))
        if op in ('>', '>=', '='):
            edges.append((index[a], index[b], (-c, op == '>')))
        if op == '!=':
            excluded.append((index[b], index[a], c))
    return edges, excluded


def _exclusions_hold(matrix, excluded):
    for u, v, c in excluded:
        hi, lo = matrix[u][v], matrix[v][u]
        if hi is not None and lo is not None and hi[0] <= c and lo[0] <= -c:
            return False
    return True


@dataclass(frozen=True)
class PreparedSource:
    names: tuple
    index: object
    matrix: tuple
    excluded: tuple

    def feasible(self, additions=()):
        if not additions:
            return True
        edges, excluded = _edges(additions, self.index)
        # Every branch owns these rows; even a rejected update cannot touch base.
        matrix = [list(row) for row in self.matrix]
        zero = (Fraction(0), False)
        size = len(matrix)
        for u, v, w in edges:
            back = matrix[v][u]
            if back is not None and _less((w[0] + back[0], w[1] or back[1]), zero):
                return False
            if not _less(w, matrix[u][v]):
                continue
            column = [matrix[i][u] for i in range(size)]
            row = list(matrix[v])
            for i, left in enumerate(column):
                if left is None:
                    continue
                prefix = left[0] + w[0]
                strict = left[1] or w[1]
                for j, right in enumerate(row):
                    if right is None:
                        continue
                    bound = (prefix + right[0], strict or right[1])
                    if _less(bound, matrix[i][j]):
                        matrix[i][j] = bound
        return _exclusions_hold(matrix, self.excluded + tuple(excluded))


def _prepare(atoms, names):
    names = tuple(dict.fromkeys((None,) + tuple(names)))
    index = {n: i for i, n in enumerate(names)}
    edges, excluded = _edges(atoms, index)
    size = len(names)
    zero = (Fraction(0), False)
    matrix = [[None] * size for _ in names]
    for i in range(size):
        matrix[i][i] = zero
    for u, v, bound in edges:
        if _less(bound, matrix[u][v]):
            matrix[u][v] = bound
    for k in range(size):
        for i in range(size):
            left = matrix[i][k]
            if left is None:
                continue
            for j in range(size):
                right = matrix[k][j]
                if right is None:
                    continue
                bound = (left[0] + right[0], left[1] or right[1])
                if _less(bound, matrix[i][j]):
                    matrix[i][j] = bound
        if any(_less(matrix[i][i], zero) for i in range(size)):
            return None
    if not _exclusions_hold(matrix, excluded):
        return None
    return PreparedSource(names, MappingProxyType(index),
                          tuple(tuple(row) for row in matrix), tuple(excluded))


def _sat(atoms):
    names = tuple(dict.fromkeys(n for a, _, b, _ in atoms for n in (a, b)))
    return _prepare(atoms, names) is not None


def check(*, task_id, revision, source, request, candidate, declaration=None, proposal=None):
    started = time.perf_counter()
    result = dict(api_version=VERSION, status='unknown', binding='unknown', grounding='unknown',
                  coverage='unknown', source_records=[], claims=[], task=None, literal=None,
                  proposal_used=False, proposal_ignored=proposal is not None, whole_answer_certified=False)
    try:
        _metadata(task_id, revision)
        for text in (source, request, candidate): _text(text)
        if declaration is not None and (type(declaration) is not TypedDeclaration or
                declaration.api_version != VERSION or
                (task_id, revision, source, request) !=
                (declaration.task_id, declaration.revision, declaration.source, declaration.request)):
            raise Unsupported('declaration_mismatch')
        result.update(source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),
                      input_sha256=_hash(dict(task_id=task_id, revision=revision, source=source,
                                             request=request, candidate=candidate, version=VERSION)))
        records = _records(source, 96); variables, semantic_keys, constraints = {}, set(), []
        constraint_phase = False
        for i, r in enumerate(records):
            obj = r['value']
            if 'var' in obj:
                if constraint_phase or set(obj) != VAR_KEYS: raise Unsupported('variable_schema')
                for value in obj.values(): _label(value)
                key = (obj['entity'],) + _frame(obj)
                if obj['var'] in variables or key in semantic_keys: raise Unsupported('duplicate_variable')
                if obj['dimension'] not in UNITS: raise Unsupported('dimension_domain')
                variables[obj['var']] = obj; semantic_keys.add(key)
            else:
                constraint_phase = True; constraints.append((i, _atom(obj, variables)))
        if not 1 <= len(variables) <= 24 or len(constraints) > 64: raise Unsupported('source_bounds')
        task = _records(request, 1)[0]; t = task['value']; operation = t.get('operation')
        if type(operation) is not str or operation not in ('compare', 'argmin', 'argmax'):
            raise Unsupported('task_operation')
        keys = {'operation', 'variables'} | (set() if operation == 'compare' else {'ties'})
        if set(t) != keys or type(t.get('variables')) is not list or not 2 <= len(t['variables']) <= 8:
            raise Unsupported('task_schema')
        operands = t['variables']; _compatible(operands, variables)
        if operation == 'compare':
            if len(operands) != 2: raise Unsupported('compare_arity')
        elif (len(set(operands)) != len(operands) or type(t['ties']) is not str or
              t['ties'] not in ('all', 'unique')): raise Unsupported('tie_policy')
        claims = _records(candidate, 32); claim_atoms = []
        if operation == 'compare':
            claim_atoms = [[_atom(r['value'], variables)] for r in claims]
            coverage = any(a[1] in ('<', '=', '>') and a[3] == 0 and
                           sorted((a[0], a[2] or '')) == sorted(operands) for aa in claim_atoms for a in aa)
        else:
            obj = claims[0]['value']; winners = obj.get('winners')
            if (len(claims) != 1 or set(obj) != {'operation', 'winners'} or obj['operation'] != operation or
                    type(winners) is not list or not winners or
                    any(type(n) is not str or n not in operands for n in winners) or
                    len(winners) != len(set(winners)) or (t['ties'] == 'unique' and len(winners) != 1)):
                raise Unsupported('winner_schema')
            aa = [(winners[0], '=', w, Fraction(0)) for w in winners[1:]]
            aa += [(winners[0], '<' if operation == 'argmin' else '>', n, Fraction(0))
                   for n in operands if n not in winners]
            claim_atoms = [aa]; coverage = True
        adjacency = {n: {n} for n in variables}; defined = set()
        for _, (a, _, b, _) in constraints:
            defined.add(a)
            if b is not None: adjacency[a].add(b); adjacency[b].add(a); defined.add(b)
        prepared_cache, sat_cache = {}, {}
        def sat(deps, relevant, additions=()):
            # Record identity and mapping belong to this validated call only.
            source_key = (tuple(i for i, _ in deps), tuple(sorted(relevant)))
            if source_key not in prepared_cache:
                prepared_cache[source_key] = _prepare(
                    [a for _, a in deps], source_key[1])
            state = prepared_cache[source_key]
            key = (source_key, tuple(additions))
            if key not in sat_cache:
                sat_cache[key] = state is not None and state.feasible(additions)
            return sat_cache[key]
        derived = []
        for r, aa in zip(claims, claim_atoms):
            needed = set(operands) if operation != 'compare' else {n for a, _, b, _ in aa for n in (a, b) if n is not None}
            relevant = set(needed)
            while True:
                grown = relevant | set.union(*(adjacency[n] for n in relevant))
                if grown == relevant: break
                relevant = grown
            deps = [(i, a) for i, a in constraints if a[0] in relevant]
            evidence = [a for _, a in deps]
            reason = 'resolved'; verdict = 'unknown'; witness = None
            if not needed <= defined: reason = 'missing_evidence'
            elif not sat(deps, relevant): reason = 'inconsistent_evidence'
            else:
                supports = sat(deps, relevant, aa)
                counter = any(sat(deps, relevant, [(a, NEGATE[op], b, c)]) for a, op, b, c in aa)
                verdict = 'fail' if not supports else 'unknown' if counter else 'pass'
                witness = dict(method='rational_difference_logic', source_indices=[i for i, _ in deps],
                               source_spans=[list(records[i]['span']) for i, _ in deps],
                               relevant_variables=sorted(relevant), normalized_claim=[_normalized(a) for a in aa],
                               claim_satisfiable=supports, counterclaim_satisfiable=counter)
            derived.append(dict(**r, binding='verified' if reason == 'resolved' else 'unknown',
                                grounding=verdict, reason=reason, witness=witness))
        binding = 'verified' if all(c['binding'] == 'verified' for c in derived) else 'unknown'
        grounded = ('fail' if any(c['grounding'] == 'fail' for c in derived) else
                    'unknown' if any(c['grounding'] == 'unknown' for c in derived) else 'pass')
        literal = dict(binding=binding, grounding=grounded, coverage='pass' if coverage else 'fail')
        result.update(source_records=records, claims=derived, task=task, literal=literal,
                      reason='declared_typed_contract' if declaration is not None else 'undeclared_contract')
        if declaration is None:
            result.update(binding='advisory', status='literal_derivation_checked')
            for c in derived:
                c.update(binding='advisory', literal_grounding=c['grounding'], grounding='unknown')
        else:
            result.update(**literal, status='verified_derivation' if binding == 'verified' else 'unknown',
                          declaration_sha256=_hash(dict(task_id=task_id, revision=revision, source=source,
                                                       request=request, version=VERSION)))
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as error:
        result.update(binding='unknown', grounding='unknown', coverage='unknown', status='unknown',
                      source_records=[], claims=[], task=None, literal=None,
                      reason=str(error) if isinstance(error, Unsupported) else 'invalid_typed_input')
    result['wall_ms'] = (time.perf_counter() - started) * 1000
    return result
