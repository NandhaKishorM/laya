# Optional numeric-comparison specialist

`ComparisonAgent` adds an **explicitly selected** numeric-comparison path alongside an
unchanged LAYA `Agent`. Ordinary calls still use the base model. The feature does not
automatically route requests, replace base weights, or change existing SDK defaults.

The specialist interprets comparison language, predicts the referenced operands, and
uses exact unit conversions and comparisons to produce `supported`, `contradicted`, or
`insufficient_context`. The local-compositional backend separates predicate recognition,
operand reversal and negation; a direct-scoring `flat` backend is also supported.

## Loading and calling

You need a **compatible, separately trained comparison bundle** on disk, as well as the
base checkpoint. Comparison weights are not included in the Python package. A normal
root, multilingual or typed-decisions LAYA checkpoint is not a comparison bundle.
`ComparisonSpecialist.from_pretrained` accepts only an existing local directory: it
does not download missing weights or call a hosted service.

```python
from laya import Agent, ComparisonAgent, ComparisonSpecialist

base = Agent("./models/laya", device="cuda")
specialist = ComparisonSpecialist.from_pretrained("./models/comparison", device="cuda")
agent = ComparisonAgent(base, specialist)

state = {
    "task": "numeric_comparison",
    "assertion": "The quantity named measured is at most the quantity named permitted.",
    "fields": [
        {"name": "measured", "value": "900", "unit": "g"},
        {"name": "permitted", "value": "1", "unit": "kg"},
    ],
}
result = agent.predict(state, comparison=True)
answer = result["answers"]["verdict"]
print(answer["choice"], answer["probabilities"])
```

Without `comparison=True`, both `predict` and `predict_batch` delegate to the base,
including its question definitions, keyword arguments and hooks. This also applies to
numeric-looking text. `system_one` aliases `predict`; `predict_many` aliases
`predict_batch`.

For enabled calls, omit `questions` or supply exactly `comparison_questions()`:

```python
from laya import comparison_questions

results = agent.predict_batch(
    [state], comparison_questions(), comparison=True, batch_size=8
)
```

The flag applies to the whole batch, not individual rows. Enabled calls reject other
question schemas and base prediction options, including hooks, rather than silently
ignoring them. They do not execute the base action head or its hooks. An application
requiring audit callbacks on this path should wrap the comparison call explicitly.

## Input contract

- `task` must be `numeric_comparison` and `assertion` must be nonempty text.
- `fields` contains at most seven uniquely named fields. Field keys are `name`, `value`
  and `unit`. Use decimal strings when the original decimal value must be preserved.
- Comparison meaning is expressed relative to the first two named quantities in the
  assertion. Their listing order is not the order of records in `fields`.
- The optional `semantics` field may only restate the built-in contract; custom
  execution rules are rejected.
- Missing, nonfinite, unparseable or unsupported quantities and incompatible units
  produce unavailable entries in the exact comparison table. Final probabilities also
  depend on the learned operand selection and neural mixture.
- Duplicate names, overlapping token-level name links, reserved tokenizer tokens,
  unsupported keys and oversized inputs raise errors. No gold labels, precomputed
  operator probabilities or cached answers are accepted.

| Dimension | Supported units |
|---|---|
| Mass | `mg`, `g`, `kg` |
| Length | `mm`, `cm`, `m` |
| Duration | `ms`, `s`, `min` |
| Storage | `B`, `kB`, `KiB`, `MB`, `MiB` |

Unit names are case-sensitive. Decimal storage units and binary storage units are
distinct. Booleans are not numeric values. Offset units, currencies, percentages and
arbitrary arithmetic expressions are not implemented as executable operators.

The operator input is limited to 512 tokens and the operand input to 1,024, including
instructions and options. Inputs exceeding either budget are rejected, not truncated.
Basic character limits also bound tokenization and numeric parsing.

## Output and confidence

The result contains the usual `answers` and `usage` envelopes, plus `specialist`
diagnostics: backend, operator distribution, operand indices/names, mixture gate and
precision. Operator order is `lt`, `le`, `gt`, `ge`, `eq`, `ne`; operand index 7 is the
explicit unavailable candidate. Input usage counts both encoder sequences.

Both the answer and diagnostics explicitly mark `calibrated: false`. `answer_confidence`
is the maximum returned probability, **not** a validated probability of correctness.
No act/escalate action is produced. Exact arithmetic cannot repair a misinterpreted
sentence or wrong operand binding. Unfamiliar wording, nested scope and domain shift
require application-specific evaluation before relying on results.

## Bundles, resources and portability

The versioned `laya-comparison-v1` bundle contains `comparison_config.json`,
`model.safetensors`, `operator_encoder/config.json`, `operand_encoder/config.json`, a
local fast tokenizer and `manifest.json`. The loader verifies every manifest digest,
requires the complete weight key/shape set, and refuses unknown bundle versions.
The bundle stores its backend and precision choices; no backend is selected from an
evaluation score at load time.

For independently trusted file digests, pass `expected_sha256={relative_path: digest}`
to `from_pretrained`, as with `Agent`. The bundled manifest detects accidental changes
but does not authenticate an untrusted producer.

`specialist.save_pretrained("./models/comparison-copy")` exports a portable bundle to
a **new** directory. Existing directories are never overwritten. The manifest is
written last, so a failed partial export cannot be loaded. Keep weights, raw data and
evaluation reports outside source control.

The current implementation retains two specialist encoders in addition to the base
encoder. Loading a specialist does not make it free when disabled: its weights still
occupy memory. Enabled requests execute the two specialist encoders and exact arithmetic;
they are not a promised speedup over base inference. Size and latency depend on the
bundle, device, precision, batch size and input length.

CPU and CUDA inference are supported. CPU uses FP32; CUDA follows the bundle's FP32/BF16
settings and requires BF16-capable hardware when applicable. Explicit CUDA failures are
reported rather than silently falling back to CPU. Tokenization/inference calls are
serialized per specialist; concurrent external mutation or training is unsupported.

## Checks

These checks use mocks and tiny randomly initialized models, not downloaded weights:

```bash
python tests/test_comparison.py
python tests/test_comparison_models.py
python tests/test_hooks_api.py
```

Validate any trained bundle separately against its original inference implementation
and on independently reviewed application examples. Unit-test parity is not a quality
or calibration guarantee.
