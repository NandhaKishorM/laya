# Research progression and rejected avenues

This summarizes the complete local experiment progression, not only the final
optimization. The upstream contribution is intentionally a focused, runnable
model-free research benchmark. The much larger experiment tree, raw archives,
training exports and operational launch details are checkpointed privately;
they are not prerequisites for running the included audit.

## Routing and training

1. CPU benchmarks established that small decision models were viable for short
   inputs but did not reliably choose effort or supplied/web/private evidence.
   Label agreement, calibration, latency and actual downstream answer correctness
   were treated as different measurements.
2. Frozen-head training, conservative ablations and real encoder LoRA runs were
   tested. Evidence-source gains sometimes came with private-history-to-web errors.
   Early stopping and fresh retention were necessary; no run was promoted just
   because development accuracy reached 100%.
3. A three-seed evidence ladder compared head objectives, rank-4/rank-16 rsLoRA and
   frozen E5 with a linear classifier. E5 obtained 44/48 on a small new-phrasing
   audit versus original Laya's 27/48, with 28.40 versus 66.33 ms warm median in
   separate sequential CPU runs. This is a different, fixed-label model and
   rendering, not an architectural Laya speedup or a replacement for arbitrary
   decision schemas. Private-history errors and imbalanced external data limited
   the conclusion. It was not deployed.
4. Broader general/layout training exposed transfer and representation failures.
   Improved probability loss did not imply improved general decision accuracy.
   No generic recipe cleared the strict retention gate. Input-key order, option
   IDs and wording were isolated as real sources of variation, not diagnosed
   merely from a single anomalous prediction.

## Contracts and real answer quality

5. Frozen-weight effort and retrieval interfaces stabilized input shape, chronology,
   token budgets and label decoding. Inactive calls passed through; combined calls
   kept the original full SDK batch to avoid small numerical changes in siblings.
   Dedicated effort labels improved on small declared sets, but a real Qwen
   answer/cost crossover did not establish better answers than fixed medium.
6. Source availability and routing were tested independently. A 48-case set left
   five semantic errors despite stabilized input shape. Further training was
   paused; no global string rules or automatic policy promotion followed.
7. Advisory routing against actual generated answers exposed typing, output-format
   and tool-use failures. Semantic answer rewriting changed previously correct
   values, so the repair path was narrowed to preserving original answers and
   checking explicit output contracts.
8. A task/evidence ledger prototype separated missing evidence from reasoning
   difficulty. Typed binding, reservations/revisions and fixture-backed acquisition
   addressed ownership/lifecycle mistakes. Free-form interpretations stayed
   advisory. Scope disclosures were limited to verified tasks after broader
   wording caused regressions. None of this became a production controller.

## Outcome checks and numerical verification

9. Task-bound outcome checks and cheaper recovery proposals were frozen and
   falsified before coding. Binding/clock validation and negative-literal oracles
   were repaired without silently changing task definitions. No retry controller
   or new training was added.
10. Coverage/grounding checks missed omitted facts and unsupported assertions.
    A frozen entailment specialist was measured on manually bound pairs; numerical
    comparison and arithmetic remained problematic. Model confidence or a second
    model's agreement was not treated as proof.
11. Manually bound exact quantity proofs were extended with a bounded grammar,
    original-byte spans, quantity conversion, ASCII layouts and singleton relations.
    Growing coverage one fixture at a time was rejected as an architectural path.
12. The unknowns were grouped by mechanism. A shared typed evidence/task API and
    rational difference-constraint kernel addressed relational families without
    inventing unrestricted prose authority. Relevant source components, strictness,
    units/frames, dense-domain disequalities, requested winners/ties and coverage
    were kept explicit. Legacy results and independent finite oracles were retained.

## Dense-case optimization

13. Profiling found repeated closure, not JSON parsing or unexpectedly huge rational
    coefficients, dominated the dense workloads. Separate source/claim-count and
    identical-claim controls isolated the cache boundary.
14. Before implementing reuse, falsification rejected matrix-only snapshots,
    endpoint-only exclusion scans, independently checked conjunctions, mutable
    sibling state, coarse source keys and floating-point substitution.
15. The final candidate prepares immutable source state per exact relevant union,
    updates isolated queries with exact arithmetic, and checks every exclusion.
    Whole-check means improved 475.294 -> 18.814 ms for 32 distinct exact claims and
    363.604 -> 70.833 ms for intervals, with full non-timing result parity. The
    [benchmark README](README.md) gives complete methods, tails and limitations.

## What remains deliberately unsolved

- General prose-to-proof binding and arbitrary fact verification remain advisory.
- The explicit declaration object is not proof that caller-supplied source text is
  true, nor an authentication mechanism or an instruction-execution permission.
- Classifier errors, numeric support outside this rational language, calibration,
  optimal effort for a new backend, long-context behavior and concurrency are not
  solved by a faster checker.
- Model-retention equality preserves known mistakes as well as good decisions.
- Public model-free replay is distinct from the private historical model-inference
  study. The PR does not claim fresh current-main classifier accuracy evidence.
- No production rollout, retry controller, new training objective or runtime hook
  is proposed. All abandoned/failed avenues remain part of the documented record.
