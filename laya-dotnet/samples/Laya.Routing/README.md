# Laya.Routing

**The combined sample for `LanguageDetection`, `LayaRouter`, `LayaEmail` and `LayaShortlist`.**
Its stdout follows a fixed output contract, so two runs can be diffed line for line (timings
excepted).

For the full library API, see the [SDK README](../../README.md). For the single-checkpoint
quickstart, see [`Laya.Sample`](../Laya.Sample/README.md).

---

## Output contract

This is the normative spec both language samples must match byte-for-byte (apart from timings,
which only ever go to stderr — never build a diff that depends on them).

**Flags:** `--model-root <dir>` (defaults to the `LAYA_ONNX_ROOT` environment variable already
set for the process), `--download`, `--preload`, `-h`/`--help`.

**Exit codes:** `0` everything printed; `1` a runtime/artifact error (a checkpoint could not be
resolved); `2` bad usage (missing option value or an unrecognised argument).

**Section headers**, one per section, printed exactly as:

```text
== 1. Language detection ==
== 2. Routing decisions ==
== 3. Router predict ==
== 4. Email cleaning ==
== 5. Shortlist ==
```

A single blank line separates each section's output from the next section's header (no blank
line after the last section).

**Number formatting:** every probability, confidence and score is formatted with exactly 4
decimal places under the invariant culture, e.g. `0.1234`, `1.0000`, `-0.0000`. Percentages
(only ever inside a routing `reason` string, formatted by `LayaRouter` itself) carry no `%`
decoration beyond the literal `%` character Python's `%.0f%%` already produces.

### Section 1 — Language detection

One line per state, in this exact field order and spacing:

```text
{name,-22} script={script,-10} language={language ?? "-",-4} is_english={true|false}
```

`is_english` is lowercase `true`/`false` (not C#'s `True`/`False`). The 8 states are, in order:
`english`, `spanish_accent_stripped`, `french`, `romanian`, `hindi`, `arabic`, `japanese`,
`dict_state` (a two-key dictionary state) — the exact literals in
`laya-dotnet/tools/routing_cases.py`'s `SAMPLE_DETECTION_STATES`, copied verbatim (also recorded in
`laya-dotnet/tests/Laya.Tests/golden/routing/sample_inputs.json` under `detection_states`).

### Section 2 — Routing decisions

`LayaRouter.Route` only — no checkpoint is loaded. One line per decision:

```text
{name,-26} -> {model subdir}  ({reason})
```

Note the double space before the parenthesized reason. `{model subdir}` is one of `english`,
`multilingual`, `typed-decisions` (never the C# enum name). The reason string is whatever
`RouteDecision.Reason` produced — it must be byte-identical to Python's, including `%r`-style
quoting and half-even `%.0f%%` rounding; this sample does not reformat it. Seven decisions, in
order: `auto_english`, `auto_hindi`, `auto_unknown_latin_romanian`,
`explicit_model_multilingual`, `explicit_lang_de`, `workflow_opt_in_customer_service` (routed
through a router built with `AutoTaskDetection = true`), `workflow_opt_in_off_by_default`
(routed through the default router, so the same question ids do **not** trigger the workflow).

### Section 3 — Router predict

For each of the two support emails (English, then Hindi), sharing one `LayaRouter` with
`MaxLoaded = 2` so both checkpoints stay resident:

```text
[{lang}] routed to {model subdir}: {reason}
  {id,-18} {value}
  {id,-18} {value}
  ...
```

`{lang}` is the sample's own case label for the email — `English` or `Hindi` — **not** the
detected ISO language code (the Hindi state's detected `language` is `null`, since its script
is Devanagari, not Latin, so an ISO-code label would not exist for it). `{value}` is: the choice
label as-is for a `ChoiceQuestion`; the score at 4 decimals for a `ScoreQuestion`; the raw
probability at 4 decimals for a `NoulQuestion` (not `yes`/`no`, and not rounded to a boolean).
No confidence or action-probability line is printed for any answer — the per-answer output is
exactly the one line above, nothing more.

### Section 4 — Email cleaning

```text
--- cleaned ---
{LayaEmail.CleanBody output, verbatim, including internal blank lines between paragraphs}
--- end ---
[{lang}] routed to {model subdir}: {reason}
  {id,-18} {value}
  ...
```

The cleaned body is printed with `Console.WriteLine`, so it ends with exactly one trailing
newline before `--- end ---`. The routed-to header line and the per-answer lines that follow use
the **same** format as section 3 (see above), over `LayaPresets.Email()` answered by the same
shared router on `LayaEmail.State(subject, cleaned-body-input, sender)`. `{lang}` is `"en"` for
this sample's email.

### Section 5 — Shortlist

A single 40-option choice question run through `LayaShortlist.Predict` (over the same shared
router) with the demo hashing embedder and `k = LayaShortlist.DefaultShortlistK` (20):

```text
kept {k} of {n}:
  {label,-28} {score:0.0000}
  {label,-28} {score:0.0000}
  ...
answer: {choice} ({probability:0.0000})
```

Labels are printed in the order `ShortlistInfo.Labels` returns them — descending cosine
similarity, ties keeping the earlier label. `{probability}` is the winning choice's own
probability from the final (reduced-criteria) forward pass, i.e. `answer[answer.Choice]`, not
the shortlist's cosine score.

### Timings

Every timing (router construction, each `Predict` call, the shortlist call) is written to
**stderr only**, never stdout, so `diff`ing the stdout of two runs is never affected by them.

### Verified output

Run against the three exported checkpoints, this sample's stdout is:

```text
== 1. Language detection ==
english                script=latin      language=en   is_english=true
spanish_accent_stripped script=latin      language=es   is_english=false
french                 script=latin      language=fr   is_english=false
romanian               script=latin      language=ro   is_english=false
hindi                  script=devanagari language=-    is_english=false
arabic                 script=arabic     language=-    is_english=false
japanese               script=kana       language=-    is_english=false
dict_state             script=devanagari language=-    is_english=false

== 2. Routing decisions ==
auto_english               -> english  (English Latin text)
auto_hindi                 -> multilingual  (non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it)
auto_unknown_latin_romanian -> multilingual  (Latin script but language looks like 'ro', not English)
explicit_model_multilingual -> multilingual  (explicit model='multilingual')
explicit_lang_de           -> multilingual  (explicit lang='de')
workflow_opt_in_customer_service -> typed-decisions  (question ids match the 'customer_service' typed-decisions workflow)
workflow_opt_in_off_by_default -> english  (English Latin text)

== 3. Router predict ==
[en] routed to english: English Latin text
  department         billing
  urgency            1.4400
  churn_risk         0.8248
  refund_requested   0.8430
[hi] routed to multilingual: non-Latin script (devanagari, 74% of letters); the English checkpoint cannot read it
  department         billing
  urgency            1.9072
  churn_risk         0.1489
  refund_requested   0.9966

== 4. Email cleaning ==
--- cleaned ---
Hi team,

I was charged twice for the Pro plan this month and the second charge still has not been refunded. This is the third time I have written about it -- please resolve this today or I will need to cancel the account.
--- end ---
[en] routed to english: English Latin text
  category           billing
  is_spam            0.1318
  is_phishing        0.1345
  urgency            1.4932
  needs_reply        0.1545

== 5. Shortlist ==
kept 20 of 40:
  card_swallowed               0.5369
  compromised_card             0.5229
  cash_withdrawal_not_recognised 0.4926
  card_payment_not_recognised  0.4672
  automatic_top_up             0.4631
  card_linking                 0.4371
  atm_support                  0.3783
  direct_debit_payment_not_recognised 0.3643
  card_payment_wrong_exchange_rate 0.3472
  balance_not_updated_after_bank_transfer 0.3320
  card_not_working             0.3301
  declined_card_payment        0.3205
  card_payment_fee_charged     0.3154
  card_acceptance              0.2983
  exchange_rate                0.2915
  card_delivery_estimate       0.2862
  activate_my_card             0.2751
  balance_not_updated_after_cheque_or_cash_deposit 0.2737
  card_about_to_expire         0.2668
  country_support              0.2631
answer: card_swallowed (0.7885)
```

---

## Run it

> **`--model-root` takes the export root, not one checkpoint's folder.** Point it at the
> directory that *contains* `english/`, `multilingual/` and `typed-decisions/` (for example
> `onnx/`), not at `onnx/multilingual`. This is the opposite of [`Laya.Sample`](../Laya.Sample/README.md)
> and `Laya.Benchmark`, which take a single checkpoint's folder with `--model-dir` and reject
> `--model-root`.

Everything after the bare `--` is passed to the app; without it, `dotnet run` tries to read
`--model-root` itself. From the repository root:

```bash
dotnet run --project laya-dotnet/samples/Laya.Routing -- --model-root /path/to/onnx
```

Or from this sample's own directory (`laya-dotnet/samples/Laya.Routing`):

```powershell
dotnet run -- --model-root C:/path/to/onnx
dotnet run -- --model-root C:/path/to/onnx --preload
```

or set `LAYA_ONNX_ROOT` once and drop the flag — the resolution behaviour is exactly
`LayaRouter`'s own (`LAYA_ONNX_ROOT/<subfolder>`, then cache, then optional download; see
`ModelArtifacts.ResolveAsync`). The directory must contain one subdirectory per checkpoint
actually exercised by this sample: `english/`, `multilingual/`, and `typed-decisions/` is
resolved only far enough to decide it *would* route there in Section 2 (nothing under
`typed-decisions/` is ever loaded by this sample — no request routes there in Sections 3-5).

| Option | Effect |
|---|---|
| `--model-root <dir>` | Sets `LAYA_ONNX_ROOT` for this process only. Takes priority over an `LAYA_ONNX_ROOT` already in the environment. |
| `--download` | Allow the router to fetch a missing checkpoint from Hugging Face (`LayaOptions.AllowDownload`), with per-file progress on stderr. Inherits the same "no ONNX export published yet" limitation as `Laya.Sample`. |
| `--preload` | Build every checkpoint up front (`LayaRouterOptions.Preload`) instead of loading each lazily on first use. |
| `-h`, `--help` | Print usage and exit 0. |

| Exit code | Meaning |
|---|---|
| `0` | All five sections printed. |
| `1` | A checkpoint could not be resolved (bad `--model-root`/`LAYA_ONNX_ROOT`, or a genuinely missing artifact file). The resolver's own message, which lists every location it tried, is printed to stderr. |
| `2` | Bad arguments: `--model-root` with no value, or an unrecognised argument. |

## What this sample demonstrates, beyond `Laya.Sample`

| Part | API | Section |
|---|---|---|
| Script/language detection with no model at all | `LanguageDetection.Analyse` | 1 |
| Pure routing decisions, explicit and automatic | `LayaRouter.Route` | 2 |
| Multi-checkpoint routing with a shared LRU cache (`MaxLoaded=2`) | `LayaRouter.Predict` | 3 |
| Turning a raw email into a clean, structured state | `LayaEmail.CleanBody`, `LayaEmail.State` | 4 |
| A ready-made question set, run through the router | `LayaPresets.Email()` | 4 |
| Reducing a high-cardinality choice question before inference | `LayaShortlist.Predict` | 5 |

## The demo embedder

Section 5 needs *some* `LayaEmbedFunction` to rank 40 labels by similarity to the state. This
sample uses [`DemoHashingEmbedder.cs`](DemoHashingEmbedder.cs) — **demo only, replace with a
real embedding model.** It is a hashed character-trigram bag-of-features vector: deterministic,
dependency-free, and a pure function of its input, chosen only because it can be reimplemented
bit-for-bit with nothing but a string, UTF-8 bytes and 32-bit FNV-1a. It is a
direct copy of `laya-dotnet/tools/hashing_embedder.py` (also ported once more, for the test suite, at
`laya-dotnet/tests/Laya.Tests/HashingEmbedder.cs`) — this sample keeps its own copy rather than
referencing the test project, so it has no test-only dependency. Do not use it for anything but
this demo; pass a real bi-encoder (or Python's `embed_fn_from_agent`, which this SDK does not
port) in production.

## Parity fixture

Like `Laya.Sample`, this sample's exact inputs are recorded as golden data —
[`sample_inputs.json`](../../tests/Laya.Tests/golden/routing/sample_inputs.json) — generated
from `laya-dotnet/tools/routing_cases.py`'s `SAMPLE_*` constants by `laya-dotnet/tools/dump_routing_golden.py`. If you
change a state, question, or the banking criteria list in `Program.cs`, update
`laya-dotnet/tools/routing_cases.py` to match and regenerate, the same way `Laya.Sample`'s golden cases are
kept in sync (see that sample's README, "If you change the sample").

Section 3's English support email and question set are, byte-for-byte, the same inputs
`Laya.Sample` uses for `case_sample_app_english.json`, so its four answers (`billing`, `1.4400`,
`0.8248`, `0.8430`) and the shortlist's answer (`card_swallowed`, `0.7885`, matching
`case_shortlist_many_options.json` for the `english` checkpoint) can be checked directly against
those existing goldens. The Hindi email in this sample is intentionally **not** the same input as
`case_sample_app_hindi.json` — it carries a `subject` and `from` in addition to `body`, to
exercise the router's non-Latin path on a fuller state — so its answers are not expected to match
that golden and are not asserted anywhere; they are illustrative output only.
