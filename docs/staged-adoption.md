# Staged adoption

Laya returns a typed decision, not permission to execute it. Adopt a decision engine in stages so
that the application keeps control of the real action while evidence accumulates.

This guide describes the application-side rollout around Laya. Laya supplies the decision,
probabilities, confidence and hook events; the application owns the incumbent action, rollout
policy, review boundary and rollback.

## 1. Shadow: observe without side effects

Run Laya on representative real traffic, but keep the incumbent action authoritative. A shadow
record should contain enough context to reproduce a comparison later:

- the request class and the Laya question schema;
- Laya's answer, per-option probabilities, confidence, checkpoint and `run_id`;
- the incumbent action and the eventual reviewed or ground-truth outcome;
- latency, errors and any fallback or review decision.

Use the existing prediction hooks for Laya-side evidence. The logging functions below are
application-owned placeholders; they are not Laya APIs:

```python
from laya import Router

class ShadowLog:
    def on_predict_end(self, ctx):
        result = ctx.results[0] if ctx.results else None
        write_shadow_record({
            "run_id": ctx.run_id,
            "model": ctx.model,
            "decision": ctx.decision,
            "result": result,
            "elapsed_ms": ctx.elapsed_ms,
            "error": None if ctx.error is None else repr(ctx.error),
        })

router = Router(hooks=[ShadowLog()])

def handle(request):
    try:
        laya_result = router.predict(request.state, request.questions)
    except Exception as exc:
        record_laya_failure(request, exc)
        return run_incumbent_action(request)
    # The shadow result is recorded by the hook; do not execute it here.
    return run_incumbent_action(request)
```

Keep sensitive fields redacted according to the application's policy. Catch and log exceptions
around `router.predict(...)` at the application boundary. The prediction hook covers the prediction
lifecycle, but failures before that lifecycle require application-level capture; do not assume
`on_predict_end` saw them. A shadow logger must not turn logging into a new user-facing action.

See [Prediction hooks](hooks/index.md), the [hook lifecycle](hooks/lifecycle.md), and
[Tracing](hooks/tracing.md) for the event order and `run_id` correlation.

## 2. Compare: disagreement is a signal, not a verdict

Compare Laya with the incumbent on the same request and question meaning. A disagreement is not
automatically an error: the incumbent may be wrong, the cases may be ambiguous, or the action may
require human judgment. Use reviewed labels or ground-truth outcomes where available, and keep an
explicit `unknown` or review bucket instead of forcing every disagreement into a binary score.

Review comparisons by checkpoint, language or route, question schema, action type and risk class.
Record coverage and disagreement alongside accuracy. A high agreement rate on an easy subset does
not justify promotion for a different language, action or question shape.

## 3. Choose a policy from held-out evidence

A confidence threshold is an application policy, not a property supplied by Laya. Fit or calibrate
the decision scores on representative held-out data, then choose a threshold from the measured
accuracy and error cost at the coverage your application can tolerate. There is no universal number
that transfers across checkpoints, question types, languages or action risks.

Record the checkpoint and question-schema version, calibration method, threshold, evaluation set and
owner with the policy. Re-evaluate it when those inputs change. Confidence orders decisions; it does
not establish that a decision is correct, and high confidence is never execution permission by
itself.

The README's [Automated Confidence Gating](https://github.com/NandhaKishorM/laya#automated-confidence-gating),
[Calibration](https://github.com/NandhaKishorM/laya#calibration), and [Honest limits](https://github.com/NandhaKishorM/laya#honest-limits) sections
give the existing calibration and confidence context. Keep irreversible or high-cost actions behind
an explicit review boundary even when their confidence is high.

## 4. Promote a bounded slice

Promotion should be a measured, reversible change rather than a global on/off switch. Define an
eligibility boundary before enabling automation, for example:

- the checkpoint, language/route and question schema are in the evaluated set;
- the action is reversible or has an explicit human review path;
- the request is not missing required context and has no Laya error;
- the slice has a size or traffic cap and a named rollback condition.

Start with a small canary. Keep review or fallback for ineligible, ambiguous and failed cases. Continue
sampling promoted decisions, compare them with the incumbent and reviewed outcomes, and monitor
disagreement, coverage, fallback rate, errors and latency. Roll back when the agreed guardrail is
breached; promotion is a bounded step, not a permanent declaration that the model is correct.

## Application/Laya boundary

Laya provides the decision evidence and exposes it through the existing API and hooks. The
application owns the incumbent result, action execution, eligibility rules, threshold, review,
fallback and rollback. Hooks can log or annotate evidence, but they do not make a high-impact action
safe to execute.

A practical rollout is therefore:

```text
real request
    ├─ incumbent action (authoritative)
    └─ Laya shadow decision ──> log, compare, evaluate
                                  └─ bounded eligible slice
                                      └─ review / fallback / rollback
```

## Rollout checklist

- [ ] Shadow logging is side-effect free and correlated by `run_id`.
- [ ] Comparison data includes the incumbent outcome and reviewed or ground-truth labels where
      available.
- [ ] Thresholds are fitted and validated on representative held-out data.
- [ ] Irreversible or high-cost actions have an explicit review boundary.
- [ ] Promotion is bounded, sampled and reversible, with a named fallback and rollback path.
- [ ] The policy owner and re-evaluation trigger are recorded.

## See also

- [Prediction hooks](hooks/index.md) — the extension seam for audit, metrics and gating.
- [Hook API reference](hooks/api.md) — `PredictContext` fields and lifecycle events.
- [Tracing](hooks/tracing.md) — `run_id` and span correlation.
- [README: Automated Confidence Gating](https://github.com/NandhaKishorM/laya#automated-confidence-gating) — confidence is a
  policy input, not a correctness guarantee.
- [README: Calibration](https://github.com/NandhaKishorM/laya#calibration) and [Honest limits](https://github.com/NandhaKishorM/laya#honest-limits).
