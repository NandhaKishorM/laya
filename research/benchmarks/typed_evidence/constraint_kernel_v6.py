"""Pure bounded rational difference logic over an explicitly approved typed task."""
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import re
import time

VERSION = 'constraint-kernel-v6'
UNITS = {'time': {'s': Fraction(1000), 'ms': Fraction(1), 'us': Fraction(1, 1000)},
         'length': {'m': Fraction(1), 'cm': Fraction(1, 100)}}
NEGATE = {'<': '>=', '<=': '>', '=': '!=', '!=': '=', '>=': '<', '>': '<='}
VAR_KEYS = {'var', 'entity', 'frame', 'statistic', 'metric', 'dimension', 'world', 'time'}
ATOM_KEYS = {'left', 'op', 'right', 'offset', 'unit'}
NUMBER = re.compile(r'[+-]?[0-9]{1,19}(?:/[1-9][0-9]{0,18})?\Z', re.ASCII)


class Unsupported(ValueError):
    pass


def _label(value):
    if (type(value) is not str or not 0 < len(value.encode('utf-8')) <= 128 or
            any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise Unsupported('label_domain')
    return value


def _metadata(task_id, revision):
    _label(task_id)
    if type(revision) is not int or not 0 <= revision <= 10**9:
        raise Unsupported('revision_domain')


def _text(text):
    if (type(text) is not str or not 0 < len(text.encode('utf-8')) <= 32768 or
            any((ord(c) < 32 and c != '\n') or ord(c) == 127 for c in text)):
        raise Unsupported('text_domain')


@dataclass(frozen=True)
class TypedDeclaration:
    task_id: str
    revision: int
    source: str
    request: str
    api_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _metadata(self.task_id, self.revision)
        _text(self.source); _text(self.request)


def _pairs(items):
    result = {}
    for k, v in items:
        if k in result: raise Unsupported('duplicate_key')
        result[k] = v
    return result


def _records(text, limit):
    lines = (text[:-1] if text.endswith('\n') else text).split('\n')
    if not 0 < len(lines) <= limit: raise Unsupported('record_count')
    out, pos = [], 0
    for line in lines:
        size = len(line.encode('utf-8'))
        if not 0 < size <= 2048: raise Unsupported('record_bounds')
        obj = json.loads(line, object_pairs_hook=_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(Unsupported('nonfinite')))
        if type(obj) is not dict: raise Unsupported('object_required')
        out.append(dict(value=obj, raw=line, span=[pos, pos + size]))
        pos += size + 1
    return out


def _frame(v):
    return tuple(v[k] for k in ('frame', 'statistic', 'metric', 'dimension', 'world', 'time'))


def _compatible(names, variables):
    if any(type(n) is not str or n not in variables for n in names):
        raise Unsupported('unknown_variable')
    if len({_frame(variables[n]) for n in names}) != 1:
        raise Unsupported('incompatible_frame')


def _atom(obj, variables):
    if set(obj) != ATOM_KEYS: raise Unsupported('constraint_schema')
    a, b, op, value, unit = (obj[k] for k in ('left', 'right', 'op', 'offset', 'unit'))
    _compatible([a] if b is None else [a, b], variables)
    if type(op) is not str or op not in NEGATE: raise Unsupported('operator_domain')
    if type(value) is not str or not NUMBER.fullmatch(value): raise Unsupported('quantity_domain')
    n, _, d = value.partition('/'); n, d = int(n), int(d or 1)
    if abs(n) > 10**18 or d > 10**18: raise Unsupported('quantity_bounds')
    units = UNITS[variables[a]['dimension']]
    if type(unit) is not str or unit not in units: raise Unsupported('unit_dimension')
    q = Fraction(n, d) * units[unit]
    if abs(q) > 10**9: raise Unsupported('canonical_bounds')
    return a, op, b, q


def _less(a, b):
    return b is None or a[0] < b[0] or (a[0] == b[0] and a[1] and not b[1])


def _sat(atoms):
    """Strict Floyd closure, then excluded-hyperplane feasibility over Q."""
    names = list(dict.fromkeys([None] + [n for a, _, b, _ in atoms for n in (a, b)]))
    ix = {n: i for i, n in enumerate(names)}; size = len(names)
    matrix = [[None] * size for _ in names]
    for i in range(size): matrix[i][i] = (Fraction(0), False)
    excluded = []
    def edge(a, b, c, strict):
        i, j = ix[b], ix[a]; w = (c, strict)
        if _less(w, matrix[i][j]): matrix[i][j] = w
    for a, op, b, c in atoms:
        if op in ('<', '<=', '='): edge(a, b, c, op == '<')
        if op in ('>', '>=', '='): edge(b, a, -c, op == '>')
        if op == '!=': excluded.append((a, b, c))
    for k in range(size):
        for i in range(size):
            x = matrix[i][k]
            if x is None: continue
            for j in range(size):
                y = matrix[k][j]
                if y is None: continue
                w = (x[0] + y[0], x[1] or y[1])
                if _less(w, matrix[i][j]): matrix[i][j] = w
        if any(_less(matrix[i][i], (Fraction(0), False)) for i in range(size)):
            return False
    for a, b, c in excluded:
        hi, lo = matrix[ix[b]][ix[a]], matrix[ix[a]][ix[b]]
        if hi is not None and lo is not None and hi[0] <= c and lo[0] <= -c:
            return False
    return True


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _normalized(a):
    return dict(left=a[0], op=a[1], right=a[2], offset=str(a[3]))


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
        sat_cache = {}
        def sat(aa):
            key = tuple(aa)
            if key not in sat_cache: sat_cache[key] = _sat(aa)
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
            elif not sat(evidence): reason = 'inconsistent_evidence'
            else:
                supports = sat(evidence + aa)
                counter = any(sat(evidence + [(a, NEGATE[op], b, c)]) for a, op, b, c in aa)
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
