# Routing

`laya-multilingual` and `laya` fail in opposite directions: the English checkpoint scores near
random on non-Latin scripts, and the multilingual checkpoint degrades on English-heavy Latin
text. Neither is a general-purpose default, so the choice of checkpoint is a decision Laya makes
before the forward pass — and `Router` is where it makes it.

```python
import laya
from laya import Router

router = Router()                       # english + multilingual, max_loaded=2
result = router.predict(
    "Ich möchte mein Konto kündigen",
    {"intent": {"type": "noul", "instructions": "Does the user want to cancel?"}},
)
result["routing"]["model"]              # 'multilingual'
result["routing"]["reason"]             # why it chose that
```

Routing costs microseconds and no model load — `Router.route` is separate from inference
precisely so you can inspect or aggregate decisions before paying for a checkpoint.

## What it decides on

`Router` reads the **script** first and the **language** second. Script settles most of the
question on its own, because the English checkpoint cannot read a non-Latin script at all:

```python
from laya import detect_language

detect_language("I was charged twice")
# {'script': 'latin', 'script_profile': {'latin': 1.0}, 'language': 'en',
#  'is_english': True, 'language_undecided': False, 'diacritic_rate': 0.0, ...}

detect_language("Что делать?")
# {'script': 'cyrillic', ..., 'language': None, 'is_english': False,
#  'language_undecided': True, ...}
```

`detect_language` is public, so you can run the same test the router runs without building a
router. `detect_script` returns just the script name when that is all you need.

Four outcomes matter, and they route differently:

| input | `language` | `is_english` | routes to |
|---|---|---|---|
| `I was charged twice` | `'en'` | `True` | `english` |
| `Das ist ein Test` | `'de'` | `False` | `multilingual` |
| `Что делать?` | `None` | `False` | `multilingual` |
| `Ik wil mijn account opzeggen` | `None` | `True` | `english` |

The last row is the one to understand: short Latin text with no evidence either way is
**undecided**, and undecided Latin text goes to the English checkpoint rather than the
multilingual one. That is deliberate. The English checkpoint handles English well and degrades
on other Latin languages; the multilingual checkpoint handles other Latin languages and
degrades on English-heavy text. For a short ambiguous string on an English-heavy workload, the
English arm is the better bet, and the decision is reported rather than hidden:

```python
router.route("Ik wil mijn account opzeggen", questions).reason
# 'Latin script, language not identified and no non-English letters; using default (english)'
```

Every `reason` names the evidence, so a misroute is diagnosable from the result alone. The
reasons you will see most often:

| reason | means |
|---|---|
| `English Latin text` | Latin script with English evidence |
| `Latin script but language looks like 'de', not English` | a Latin language identified as non-English |
| `non-Latin script (cyrillic, 100% of letters); the English checkpoint cannot read it` | script alone decided it |
| `Latin script, language not identified and no non-English letters; using default (english)` | undecided, defaulted |

## Reading the decision

`RouteDecision` subclasses `dict`, so it serialises straight into an API response, and it also
has `.model` and `.reason` attributes for the common reads:

```python
d = router.route("Ich möchte mein Konto kündigen", questions)
d.model        # 'multilingual'
d.reason       # "Latin script but language looks like 'de', not English"
dict(d)        # {'model': ..., 'reason': ..., 'repo': ..., 'detection': ..., 'workflow': ...}
```

`detection` is the full `detect_language` payload for that state, so it carries `script`,
`language`, `is_english`, `language_undecided`, `diacritic_rate` and `non_latin_fraction`.
`repo` is the Hub repository the chosen checkpoint comes from, and `workflow` is populated when
`auto_task_detection=True`.

## Overriding the route

Four ways, in increasing order of how much you want to decide yourself.

**Pin the checkpoint** when you already know which one fits:

```python
router.predict(state, questions, model="multilingual")
```

**Name the language** when you know it and the detector is guessing:

```python
router.route("Ik wil mijn account opzeggen", questions, lang="nl").model   # 'multilingual'
router.route("Ik wil mijn account opzeggen", questions, lang="en").model   # 'english'
```

`lang` also reaches the checkpoint, where it selects any per-language temperatures that
checkpoint carries. Passing `lang="nl"` therefore changes both which model answers and how its
probabilities are scaled.

**Bring your own detector** with `lang_guess`, a callable taking the state. It is consulted when
the built-in detection is undecided, and it can be set per call or once on the router:

```python
router = Router(lang_guess=lambda s: my_lid(s))          # router-wide
router.predict(state, questions, lang_guess=lambda s: "nl")   # or per call
```

Both produce a reason naming the source, so you can tell a caller-supplied verdict from a
detected one:

```
Router(lang_guess=...): the caller identified this as non-English text
lang_guess:             the caller identified this as non-English text
```

The callable should return a language code, or something falsy when it has no opinion.

**Replace the decision outright** with an `on_route` hook, which sees the `RouteDecision` before
it is used and may return a different one. This is the general escape hatch — pin by request
attribute, A/B a checkpoint, or route on something the detector cannot see. See
[Prediction hooks](hooks/index.md).

## Memory and load cost

A checkpoint is built on first use, which costs seconds; routing itself costs microseconds.
`max_loaded` (default `2`) caps how many **loaded** checkpoints stay resident, evicting the
least recently used:

```python
router = Router(max_loaded=2)
router.load("english")            # resident
router.load("multilingual")       # resident
router.load("typed-decisions")    # evicts the least recently used
router.unload("multilingual")     # free one
router.unload()                   # free all
```

For a server, pay the loads up front so no request does:

```python
router = Router(preload=True)                  # every default checkpoint resident
router.preload(["multilingual"])               # or name them
```

`preload` raises `max_loaded` to fit the requested checkpoints plus whatever is already
resident, so preloading incrementally never evicts what you loaded before.

**Already have an agent?** Hand it over instead of loading a second copy:

```python
agent = laya.load("convaiinnovations/laya")
router = Router()
router.attach("english", agent)
```

Worth doing: each checkpoint is hundreds of megabytes, and `attach` avoids holding two copies
of the same weights.

An attached agent is never evicted — `attach` raises `max_loaded` to fit it, so the cap you
passed to the constructor is a floor once you start attaching:

```python
router = Router(max_loaded=1)
router.attach("english", a); router.attach("multilingual", b)
router.max_loaded                     # 2, raised by the second attach
len(router.loaded)                    # 2
```

That is deliberate rather than an oversight: `attach` is for weights the caller already owns
and is holding anyway, so evicting one would free nothing while silently detaching an agent the
caller still has a reference to. Keep using `attach` for agents your process owns, and let
`max_loaded` govern the checkpoints the router loads for itself.

## Batching

`predict_batch` routes every request, groups by the checkpoint it chose, and shares forward
passes within a group — so a mixed-language batch costs one pass per checkpoint, not one per
request. Requests may carry their own `model`, `task`, `lang` and `lang_guess`:

```python
results = router.predict_batch([
    {"state": "I was charged twice", "questions": questions},
    {"state": "Je veux annuler mon abonnement", "questions": questions},
    {"state": "Please cancel my plan", "questions": questions},
    {"state": "Ich möchte kündigen", "questions": questions},
])
[out["routing"]["model"] for out in results]
# ['english', 'multilingual', 'english', 'multilingual']
```

Results come back in input order. Requests are also split on question schema, token budget and
(for a checkpoint with per-language temperatures) language, because those change what a shared
forward pass would compute — grouping never merges requests that would answer differently apart.

If you only want the routing decisions, `route_batch` returns them without loading anything:

```python
decisions = router.route_batch(requests)     # no checkpoint is built
```

That is useful for capacity planning or for rejecting a batch before paying to load a model for
it.

## Custom checkpoint sets

`models` maps a routing name to a repository and an optional subfolder. The default is the three
public ones:

```python
Router().models
# {'english': ('convaiinnovations/laya', None),
#  'multilingual': ('convaiinnovations/laya', 'multilingual'),
#  'typed-decisions': ('convaiinnovations/laya', 'typed-decisions')}
```

Point a name at your own fine-tune and it is routed to exactly as the built-ins are:

```python
Router(models={
    "english": ("your-org/your-checkpoint", None),
    "multilingual": ("convaiinnovations/laya", "multilingual"),
})
```

`standalone_repos=True` treats each name as its own repository rather than subfolders of one,
which is what you want when your checkpoints are separate Hub repos. `default` names the arm
used when detection is undecided.

## What routing cannot do

- **It does not read the state's meaning.** The decision is script and language, not topic or
  difficulty. A short English sentence and a short German one look alike to it except for the
  language evidence, which is why short undecided text defaults rather than refusing.
- **It cannot rescue a checkpoint that does not fit the task.** Choosing correctly between two
  checkpoints does not help if neither was trained on your label space — see the high-cardinality
  limits in `BENCHMARKS.md` at the repository root.
- **A single accented loanword is not enough to leave English**, and one bare function word is
  not enough to enter it; short text in a Latin language the detector cannot identify will
  default to the English arm. If that matters for your workload, pass `lang` or `lang_guess`
  rather than relying on detection.
