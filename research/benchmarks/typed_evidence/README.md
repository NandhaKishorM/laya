# Typed evidence verification: a model-free research boundary

This benchmark explores what can be checked **after** a decision model proposes
an answer. It is not a Laya inference optimization, controller, trained checkpoint
or new public package API. Nothing here is imported by `laya/`. It requires only
the Python standard library; no weights, downloads, keys or external services.

Run from the repository root:

```bash
python research/benchmarks/typed_evidence/audit.py
python -m unittest discover -s research/benchmarks/typed_evidence -p 'test_*.py' -v
# Optional fresh CPU measurements, not a replay of the archived timing values:
python research/benchmarks/typed_evidence/audit.py --benchmark --repeats 10
```

## Why this boundary

Our research started with effort and supplied/web/private evidence routing. Better
label agreement did not necessarily improve actual answer quality. Attempts to
repair answers introduced rewrites, scope mistakes and unsupported claims. A
specialized entailment model also made numerical comparison mistakes. The
[research summary](RESEARCH-SUMMARY.md) records the sequence and rejected avenues.

Here, interpretation remains advisory. A trusted application independently binds
an exact source and task; the checker only tests a candidate against that contract.
The same principle applies to Laya decisions: model confidence is a signal, not
evidence that a fact is true or permission to execute a tool. This prototype
does not implement unrestricted prose binding or automatically authorize any action.

## Contract and limitations

Source is JSONL: variable records first, then exact constraints. Each variable
names an entity, frame, statistic, metric, dimension, world and time. Constraints
specify `left`, `op`, `right` (variable or null), rational-string `offset`, and
`unit`. Supported operations are `<`, `<=`, `=`, `!=`, `>=`, `>` over **rationals**;
there is no integer/nonlinear solver. Units are ms/us/s and m/cm with exact
conversion. Frames cannot be silently mixed.

Task is one JSON object: `compare` with two variables, or `argmin`/`argmax` with
two to eight variables and an explicit `all`/`unique` tie policy. Candidate JSONL
contains compare constraints, or a winner declaration. Source/candidate limits
remain 24 variables, 64 source constraints, 32 candidate records and 32 KiB per
text. Duplicate keys, unsupported schemas and partial parses do not pass.

The `TypedDeclaration` object binds the exact task ID, revision, source and request.
Construct it only after an independent trusted binding step, **not from a model's
assertion that its own interpretation is correct**. It is a local explicit opt-in,
not a cryptographic authentication mechanism. Without it, results remain advisory.
The checker does not establish real-world source truth, scope correctness of the
caller's binding or the correctness of surrounding prose.

`binding`, `grounding` and `coverage` are separate outputs. A true off-task statement
does not satisfy task coverage. Missing or inconsistent relevant evidence remains
unknown. Irrelevant conflicts do not poison a separate component. Rejection is
atomic; source/answer spans use original UTF-8 byte offsets. Witnesses retain exact
source record indices. `whole_answer_certified` and `proposal_used` remain false.

## Prepared-source invariant

`constraint_kernel_v6.py` is the frozen full-recomputation reference.
`constraint_kernel_prepared.py` is a separate implementation candidate sharing the
same declaration class and parser helpers. Both retain the validated checkpoint's
code, not simplified benchmark solvers; the prepared export drops one empty EOF line.

Within one validated check, prepare each exact relevant source union once. Keep
the closed rational difference-bound matrix, variable mapping and all disequality
exclusions immutable. Every query gets private rows. Equality's two edges and
compound winner conditions apply sequentially within that query. A failed branch
cannot change its source or siblings. There is no cross-call or global cache.

For a feasible closed matrix D, adding edge u -> v with weight w is inconsistent
if `w + D[v,u]` is negative, or zero and strict. Otherwise:

```text
D'[i,j] = min(D[i,j], D[i,u] + w + D[v,j])
```

Weights are exact Fractions plus a strictness bit. Update from the old row/column,
then check **every** source and query disequality, including exclusions far from
the changed edge. Finite non-forced exclusions cannot cover a nonempty convex
rational region; that argument does not apply to integer domains. The zero anchor
does not merge unrelated source-incidence components. Source-state ownership is
keyed by exact source record IDs and the variable mapping within the current call.

Target work is O(U*n^3 + Q*k*n^2 + Q*E), rather than repeated O(Q*n^3) closure:
U relevant unions, Q queries, k added bound edges, E exclusions. Parsing, copying
and witness work remain included. A general solver or a prose verifier is out of scope.

## Historical CPU measurements

Same-input randomized/interleaved whole-check measurements on a Threadripper PRO
7965WX, Python 3.12.3. Declaration construction/model loading are excluded equally;
validation, parsing, proof, exclusions and output construction are included.
Normal unrelated workloads were not stopped. Ten repetitions per engine/control,
five per engine/protocol case: 11,570 archived observations, no discarded samples.

| Shape | Reference mean ms | Prepared mean ms | Ratio |
| --- | ---: | ---: | ---: |
| Exact dense, 1 claim | 28.546 | 14.092 | 2.03x |
| Exact dense, 8 distinct | 127.938 | 14.739 | 8.68x |
| Exact dense, 16 distinct | 242.903 | 15.034 | 16.16x |
| Exact dense, 32 distinct | 475.294 | 18.814 | 25.26x |
| Interval dense, 32 distinct | 363.604 | 70.833 | 5.13x |
| Inconsistent source, 32 claims | 2.132 | 1.896 | 1.12x |
| Small unit negative control | 0.199 | 0.147 | 1.36x |
| Compound tied-winner control | 0.365 | 0.192 | 1.90x |

Dense controls have 24 variables and 64 constraints. Exact 32-distinct prepares one
source instead of 65 full closure attempts; interval 32-distinct still prepares
eight distinct relevant unions instead of 72 closure attempts. Not every attempted
reference closure completes: inconsistent queries stop early.

Keep the tails: prepared exact-32 median 15.669 ms, mean 18.814, p95 46.911;
interval-32 median 65.230, mean 70.833, p95 87.688. Ten samples make p95 coarse.
No cohort mean regressed, but not every observation improved: exhaustive-order
median was 0.31030 -> 0.31065 ms and invalid-input p95 was 0.0854 -> 0.0878 ms.
These are **checker** speedups, not faster Laya inference or agent response time.

## What the audit establishes

- Regenerate 1,118 frozen synthetic cases and load the 11 explicitly archived
  quantity-transport cases: 1,129 independently labelled protocol cases.
- Replay both implementations and 278 archived legacy-to-typed projections;
  compare **all non-timing output fields** to each other and archived SHA-256
  digests. The one historical typed self-comparison extension relative to legacy
  v4 is recorded explicitly, not introduced or hidden by this optimization.
- Always replay all 14 dense/small benchmark controls too, without enabling
  timing. Their full-output gold is anchored to the original frozen profile,
  not regenerated from the candidate. Fresh measurements check that gold as well.
- Check 18,744 per-claim oracles and 43,461 original-byte/witness fields.
- Recompute every archived control/cohort timing statistic, sample identity,
  multiplicity and speedup ratio from the 11,570 raw samples in `archive.json`.
- Exercise strict boundaries, exact fractions, source/new/distant exclusions,
  equality conjunctions, branch isolation, authority, parsing and scope controls.
  Exhaustive weak-order tests and seeded compound-query differential tests are
  finite falsification attempts, not a formal proof.

The post-publication [P1/P2 review](REVIEW-1282.md) found and repaired a P2 in the
mandatory audit: it previously audited dense timings without executing those
workloads. The two fault-injection regressions failed before the repair. Replay
now covers 1,421 inputs (1,129 protocol + 278 projections + 14 controls), while
the existing protocol/per-claim counts and every historical timing sample remain
unchanged. An independent strict-bound oracle is included in the tests.

The original full study separately reran fresh CPU inference: 1,364 Laya general
fields, 32 effort controls, 265 entailment-model inputs and 1,232 legacy results
were unchanged, as were weights/configurations. **That used Laya 0.4.8**, not this
checkout's current SDK. Its aggregate provenance and original artifact hashes are
recorded here; this model-free export does not rerun or independently substantiate
that entire private model study. Known incorrect decisions remained incorrect.
Retention is not accuracy, and no classifier accuracy improvement is claimed.

## Files and provenance

- `manifest.json`: public-file hashes and the exact regenerated protocol digest.
- `archive.json`: selected original fixtures, full-result digests and **all** A/B
  timing samples; compact JSON keeps the export modest without dropping records.
- `constraint_kernel_oracles.py`: solver-independent fixtures and weak-order oracle.
- `audit.py`: model-free replay, archive audit and optional fresh measurements.
- `test_constraint_kernel_*.py`: software and mathematical boundary tests.
- `test_audit.py`: negative controls for archive corruption.

Exported reference/oracle bytes match their original hashes. The prepared solver
differs only by one trailing empty line (its AST is unchanged); both original and
export hashes are recorded. The prepared test's private study import is replaced by this audit's
small fixture/declaration adapter; the legacy negative authority control uses a
distinct local class rather than importing the private v4 experiment. Private
paths/configuration, live conversations, account identifiers, downloaded datasets,
weights and environments are not included. Public transformed artifacts have new
hashes, distinct from their cited original private artifacts. License: repository
Apache-2.0. No changes to the core Laya API, inference, hooks, training or defaults.
