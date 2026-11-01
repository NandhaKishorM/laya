# P1/P2 review of PR 1282

Review baseline: `65d8b68d07a47e6a80e253ddfc8a237f435336ea`.
Scope: exact solver, source/query ownership, schema/authority boundaries, offline
audit, fixtures and the new CI steps. No model, training, controller, production
configuration or running service was changed. No related upstream fix existed
for this newly introduced benchmark at review time.

## Confirmed P2: mandatory audit skipped dense benchmark correctness

At the review baseline, `audit.audit()` replayed only `rows` and `projections`.
The 14 dense/small controls were used to audit timing statistics, but their actual
checker outputs were tested only by the optional `--benchmark` path. The ordinary
unit suite also omitted these full-size workloads. Thus the benchmark CI gate
could be green even if the optimization broke its main advertised workloads.

### Falsification and observed evidence

- The initial broad claim that **all** 64-constraint workloads were untested was
  falsified: a pre-existing `bounded-large` projection catches a generic
  64-constraint corruption. That broader finding was dropped.
- The actual uncovered shape was **24 variables and 64 constraints**. No ordinary
  replay row/projection had that combination. Eleven dense controls did.
- A process-local `unittest.mock.patch` wrapper returned the ordinary checker
  result except that it changed grounding to `fail` for that shape. No source or
  archived input was edited. Default audit still reported `full_result_parity=true`;
  all 21 tests passed; the faulty path was called zero times. A direct exact-dense
  invocation changed the correct `pass` to `fail`.
- A sibling falsifier replaced one dense candidate with `not-json` in an isolated
  in-memory plan after verification. Default audit still passed, while the optional
  benchmark raised `benchmark oracle`. This isolates the missing replay boundary,
  rather than a flaw in the immutable source solver or JSON rejection itself.
- Two new regression tests failed against the original gate with `ValueError not
  raised`, before the audit implementation was changed. Both pass with the repair.

This is a research/CI correctness gate defect, not evidence that the actual solver
currently returns a wrong numerical answer or that production is affected.

### Repair invariant

Every workload underlying a published comparison must receive mandatory full-output
correctness replay, even when fresh latency measurement is not requested.

The default audit and ordinary full-fixture unit replay now include all 14 controls.
Control inputs have a separate manifest digest. Full expected outputs were first
matched to every original frozen profile result digest, then canonically hashed
for the public archive. The provenance records that profile's SHA-256:

```text
76015fdd9c10869140425c7395c55da72382f8469a791b72c7b6a3bda0fa83a9
```

No gold came from the candidate. Existing protocol/projection expected results,
model-retention history and all 11,570 timing observations are unchanged. The new
default count is 1,421 full-output input pairs; 18,744 protocol per-claim checks and
43,461 original-byte/witness checks retain their previous meanings. Fresh timing
also verifies the same full-output gold before accepting a sample.

## Other passes and rejected hypotheses

- Reviewed incremental edge orientation, strict-zero/negative cycles, redundant
  bounds, two-edge equality, cumulative winner queries, and every source/query
  exclusion. No solver counterexample was found.
- Added a separate Bellman-Ford test oracle over exact rationals with symbolic
  strictness. It explicitly enumerates `<`/`>` branches for disequalities rather
  than using the candidate's closure/excluded-hyperplane algorithm. Tests cover
  240 feasible source graphs and 2,880 compound extensions, randomized mappings,
  failed branches and immutable snapshots, plus strict-self/dense-domain controls.
  This is finite independent testing, not formal verification over all inputs.
- A separate eight-thread probe compared 160 unique explicit task/revision/source
  calls against full reference outputs; no cross-call state contamination occurred.
- Existing malformed-input, stale/forged approval, frame/unit, irrelevant-conflict,
  byte-span, missing-evidence, failure-dominance and winner controls remain unchanged.
- Parser, frozen reference, prepared solver and oracle generator code are unchanged
  by this repair. No new fallback, regular-expression policy or runtime behavior
  was added to the model or package.

## Reproduction

```bash
python research/benchmarks/typed_evidence/audit.py
python -m unittest discover -s research/benchmarks/typed_evidence -p 'test_*.py' -v
ruff check laya/ research/benchmarks/typed_evidence/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/ research/benchmarks/typed_evidence/
git diff --check
```

The independent oracle and both fault regressions execute in the ordinary test
suite; no additional packages are required. No P1 or additional solver P2 was
confirmed by these bounded passes. This does not establish absence of all bugs.
