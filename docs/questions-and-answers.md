# Questions and answers

A Laya question is a typed decision, and the type decides both what you ask and what you get
back. There are three, and one state can carry all of them in a single forward pass:

```python
import laya

agent = laya.load("convaiinnovations/laya")

questions = {
    "dept": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "money and invoices",
                          "technical": "bugs and outages",
                          "sales": "pricing and contracts"}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "severity": {"type": "score", "instructions": "How severe is this?",
                 "criteria": ["trivial", "minor", "moderate", "serious", "critical"]},
}

result = agent.system_one({"text": "I was charged twice and nobody has replied for a week. "
                                   "Please refund me."}, questions)
result["answers"]["dept"]["choice"]        # 'billing'
result["answers"]["urgent"]["noul"]        # 0.8727
result["answers"]["severity"]["score"]     # 2.9046
```

One forward pass answers all three. That is the point of the typed interface: a `noul` question
is not a `choice` with two options that happen to be "yes" and "no" — it is a different head
with a different output shape, and the type tells Laya which to use.

## The three types

### `choice` — pick one of a set

```python
{"type": "choice",
 "instructions": "Which team should handle this?",
 "criteria": {"billing": "money and invoices", "technical": "bugs and outages"}}
```

`criteria` is an ordered mapping of label to description. **Order is positional** in the rendered
question, so two questions with the same labels in a different order are different questions.
A description is optional; `{"billing": None}` renders the label alone. Labels are returned
exactly as you wrote them, so a non-string label comes back as itself in `choice` and as the key
in `probabilities`.

The description is worth writing. It is not decoration: the rendered question text is what the
model reads, so a bare `{"a": None, "b": None}` gives it nothing to distinguish the options.

```python
{"type": "choice", "choice": "billing",
 "probabilities": {"billing": 0.9881, "technical": 0.0057, "sales": 0.0062},
 "confidence": 0.9339, "answer_confidence": 0.9881,
 "action": {"act_probability": 1.0}}
```

### `noul` — yes or no

```python
{"type": "noul", "instructions": "Is this urgent?"}
```

`noul` is [the project's name](https://en.wikipedia.org/wiki/Yes%E2%80%93no_question) for a
two-way decision, and its answer is **the probability of true**, not a thresholded boolean:

```python
{"type": "noul", "noul": 0.8727, "confidence": 0.8727, "answer_confidence": 0.8727,
 "action": {"act_probability": 1.0}}
```

The threshold is yours to choose, because it depends on what a false positive costs you. There is
no `bool` field to mistake for a decision.

You can relabel the two options when the decision is not naturally a yes/no — `labels` takes
exactly the keys `false` and `true`, and the polarity is unchanged: `noul` is still P(true).

```python
{"type": "noul", "instructions": "Does this need a human?",
 "labels": {"false": "automatic", "true": "escalate"}}
```

Labels are not a way to fix a question the model gets wrong. `noul` follows its own option
labels, most strongly on the English checkpoint, so a label pair that reads as a decision
("approve" / "reject") can pull the answer toward the label rather than the state. Validate any
relabelling on your own data before relying on it.

### `score` — an ordered level

```python
{"type": "score", "instructions": "How severe is this?",
 "criteria": ["trivial", "minor", "moderate", "serious", "critical"]}
```

`criteria` is an ordered list, and it must be ordered ascending — position *is* the scale.

```python
{"type": "score", "score": 2.9046,
 "legend": {"0": "trivial", "1": "minor", "2": "moderate", "3": "serious", "4": "critical"},
 "probabilities": {"0": 0.0134, "1": 0.05, "2": 0.0518, "3": 0.7881, "4": 0.0967},
 "confidence": 0.5187, "answer_confidence": 0.7881,
 "action": {"act_probability": 1.0}}
```

**`score` is an expected value, not the most likely level.** Above, `score` is 2.90 while the
single most likely level is `serious` (3) at 0.788. Both are useful and they answer different
questions: the expected value minimises squared error over the scale, the argmax minimises
disagreement with the model. If you want the label, take the argmax of `probabilities` or read
`answer_confidence`'s counterpart from the legend — do not round `score` and assume it is the
label. `legend` exists so you never have to guess which index means what.

## Reading confidence

Every answer carries two confidence numbers, and they measure different things.

| field | what it is | gating on it? |
|---|---|---|
| `answer_confidence` | `max(p)` — the probability of the answer being reported | **yes, after fitting** |
| `confidence` | `1 - H(p) / log(k)` — how concentrated the whole distribution is | no |
| `probabilities` | the full distribution (`choice`, `score`) | — |

`answer_confidence` is the quantity temperature scaling fits and the quantity every calibration
figure in the repository is computed on, which is what makes it the one to gate on. It is **not
calibrated as shipped**: the property usually attributed to it — that of the answers returned at
confidence *c*, about *c* of them are right — holds only once temperatures have been fitted and
validated on **held-out data for your checkpoint at your option count**. The shipped checkpoints
are over-confident and how far depends on the option count, so an untuned threshold can select
below the model's own accuracy ([#394](https://github.com/NandhaKishorM/laya/issues/394)).

```python
# THRESHOLD is a number you measured on your own held-out data, not one the model ships.
# Fit and validate the temperatures first — the fine-tuning notebook has the loop:
#   notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
ans = result["answers"]["dept"]
if ans["answer_confidence"] >= THRESHOLD:
    ...
```

`confidence` is normalized entropy: high when the distribution is peaked, low when it is spread
out, regardless of whether the top answer is correct. It is a useful signal and it is **not** on
the same scale, so the two must not be gated against one number:

```python
# the same three answers, and the two numbers are not the same
dept      confidence 0.9339   answer_confidence 0.9881
urgent    confidence 0.8727   answer_confidence 0.8727
severity  confidence 0.5187   answer_confidence 0.7881
```

For `noul` they are equal by construction — over two options, `max(p, 1-p)` is `max(p)` — so a
`noul` answer cannot tell you which one you have been reading. Both keys are present on every
type so the choice is explicit rather than implied.

**A threshold is a policy, not a property of the model.** Both checkpoints ship over-confident,
and how over-confident depends on the option count, so a number measured on a 3-option question
does not transfer to a 20-option one. Measure it on your own data; the Calibration section of
[`BENCHMARKS.md`](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) has the fitting
loop and the fitted values.

### `action` and `act_probability`

`action.act_probability` is a separate head's score for "should an agent act on this at all",
distinct from the answer's own confidence. It is reported for every question type. Nothing in
the library thresholds it for you.

## Presets

Three ready-made question sets, so the common cases do not need hand-written criteria:

```python
from laya import triage_questions, guard_questions, moderation_questions

agent.system_one(ticket, triage_questions())
```

Use them as a starting point rather than a contract — read the questions they produce with
`render_options` and check the labels fit your domain before shipping them.

## Reading the options back

Because option order is positional and the option text is what the model reads, it is worth being
able to see exactly what was sent:

```python
from laya import render_options

render_options({"t": "choice", "crit": {"billing": None, "sales": "pricing"}})
# ['billing', 'sales: pricing']

render_options({"t": "score", "crit": ["low", "high"]})
# ['level 0: low', 'level 1: high']
```

The same labels in a different order render in that order, which is why order is part of the
question's identity:

```python
render_options({"t": "choice", "crit": {"x": "first", "y": "second"}})
# ['x: first', 'y: second']
render_options({"t": "choice", "crit": {"y": "second", "x": "first"}})
# ['y: second', 'x: first']
```

**Note the keys.** `render_options` takes the *internal* short-key form `{"t": ..., "crit": ...}`,
not the `{"type": ..., "criteria": ...}` form you write in a question — passing the public shape
raises `KeyError: 't'`.

The conversion is a short, stable mapping you can inline, which avoids reaching into a private
helper — `Agent._to_internal` is internal and may change:

```python
def as_internal(q):
    """The short-key shape `render_options` reads, from a question as you wrote it."""
    crit = q.get("criteria")
    if q["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    return {"t": q["type"], "ins": q["instructions"], "crit": crit}

render_options(as_internal(question))
```

That mirrors what the library does for a `choice` question written as a list of labels; a
`criteria` dict and a `score` list pass through unchanged.

## Limits worth knowing before you design around this

- **Confidence is not a correctness guarantee at high option counts.** On a 20-option question the
  distributions for right and wrong answers overlap heavily, and a threshold can end up selecting
  *below* the model's own accuracy. See [#394](https://github.com/NandhaKishorM/laya/issues/394).
- **Negation is not reliably handled in forced-choice questions.** A cancellation question can
  return the cancellation label for a state that says *not* to cancel, with high confidence, on
  both checkpoints. See [#377](https://github.com/NandhaKishorM/laya/issues/377).
- **`noul` can follow its labels rather than the state**, so validate any relabelling.
- **More options is not free.** Past roughly 20 the model degrades quickly; use the shortlist
  helper to reduce a large label space before asking, or split it into a coarse and a fine
  question.
