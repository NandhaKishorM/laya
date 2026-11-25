# laya-routing

**The combined routing sample.** It exercises everything the base Rust SDK does not cover on its
own: language/script detection, `LayaRouter` routing decisions and predictions, email cleaning,
and the embedding shortlist.

## Output contract

**This is the load-bearing part of this file.** This sample's stdout format is a stable contract
(apart from timings, which never go to stdout — see below), so anyone changing it should keep this
section up to date.

- Flags: `--model-root <dir>` (default: the `LAYA_ONNX_ROOT` env var), `--download`, `--preload`,
  `-h`/`--help`.
- Exit codes: `0` ok, `1` a runtime or artifact error (bad model root, missing checkpoint file),
  `2` bad usage (unknown flag, a flag missing its value).
- Section headers, printed in this order: `== 1. Language detection ==`, `== 2. Routing decisions
  ==`, `== 3. Router predict ==`, `== 4. Email cleaning ==`, `== 5. Shortlist ==`. A single blank
  line separates each section's output from the next section's header (no blank line after the
  last section).
- All numbers are printed with exactly 4 decimals (`0.1234`), not fewer, not scientific notation.
- **Section 1** (language detection), one line per state:
  `{name:<22} script={script:<10} language={language or "-":<4} is_english={true|false}`
- **Section 2** (routing decisions, `Route` only, nothing loaded), one line per case:
  `{name:<26} -> {model subdir}  ({reason})` — two spaces before the parenthesised reason.
- **Section 3** (router predict), per email: a routed-to header, `[{lang}] routed to {subdir}:
  {reason}`, then one line per answered question: `  {id:<18} {value}`, where `value` is the
  choice label for a choice question, the score to 4 decimals for a score question, or the `P(true)`
  probability to 4 decimals for a noul question (**not** `yes`/`no`).
- **Section 4** (email cleaning): the cleaned body verbatim between a `--- cleaned ---` line and an
  `--- end ---` line, then the `LayaPresets.Email()` answers in the same routed-header-plus-lines
  format as section 3.
- **Section 5** (shortlist): `kept {k} of {n}:`, then one line per kept label in rank order —
  `  {label:<28} {score:.4}` — then `answer: {choice} ({probability:.4})`.
- Timings go to **stderr only**, never stdout, and are not part of the diff.

Sample inputs are copied verbatim from `laya-rust/tools/routing_cases.py`'s `SAMPLE_*` constants into
[`src/main.rs`](src/main.rs) — no runtime JSON is read, so both samples read literally the same
bytes, recorded once from Python as `laya-rust/tools/dump_routing_golden.py`'s `sample_inputs.json`.

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

### Ambiguities resolved while implementing this contract

The plan's contract text left two things unspecified:

1. **`{lang}` in section 3/4's routed-to header** is a short, fixed, caller-chosen tag per input
   (`"en"`, `"hi"`) — **not** derived from `RouteDecision.detection.language`. A non-Latin-script
   decision (the Hindi email here) never gets a `language` guess at all: routing is decided by
   script alone, so `detection.language` is `None`/`null` and there would be nothing to print.
2. **The shortlist question's id** (used nowhere in `laya-rust/tools/routing_cases.py`, which only records
   state/instructions/criteria, not a question id) is `"banking_intent"`.

## Run it

> **`--model-root` takes the export root, not one checkpoint's folder.** Point it at the
> directory that *contains* `english/`, `multilingual/` and `typed-decisions/` (for example
> `onnx/`), not at `onnx/multilingual`. This is the opposite of [`laya-sample`](../laya-sample/README.md)
> and `laya-benchmark`, which take a single checkpoint's folder with `--model-dir` and reject
> `--model-root`.

Everything after the bare `--` is passed to the app; without it, `cargo run` tries to read
`--model-root` itself. From the `laya-rust` directory:

```bash
cargo run --release -p laya-routing -- --model-root ../onnx
cargo run --release -p laya-routing -- --model-root ../onnx --preload
```

Or set the environment variable once and leave out the flag:

```powershell
$env:LAYA_ONNX_ROOT = "C:\path\to\onnx"
cargo run --release -p laya-routing
```

The root must contain one subdirectory per checkpoint used by the sample (`english/`,
`multilingual/`, `typed-decisions/`), each holding `model.onnx`, `model.onnx.data`,
`rl_agent_config.json` and `tokenizer/tokenizer.json` — the same layout `laya-sample` expects for a
single checkpoint. `LayaOptions` has no "root" field of its own (only a single-checkpoint
`model_directory`), so `--model-root` works by overriding the `LAYA_ONNX_ROOT` process environment
variable that `ModelArtifacts::resolve` already reads per checkpoint.

### Command-line options

| Option | Effect |
|---|---|
| `--model-root <dir>` | Root directory holding every checkpoint subdirectory. Overrides `LAYA_ONNX_ROOT` for this process. |
| `--download` | Fetch missing checkpoints from Hugging Face into the local cache, with per-file progress on stderr. |
| `--preload` | Build and load every checkpoint up front (`LayaRouterOptions::preload`) instead of on first use. |
| `-h`, `--help` | Print usage and exit 0. |

| Exit code | Meaning |
|---|---|
| `0` | All five sections printed. |
| `1` | A checkpoint's artifact could not be resolved, or another runtime error occurred. The resolver's message, which lists every location tried, is printed to stderr. |
| `2` | Bad arguments: a flag missing its value, or an unrecognized argument. |

See [laya-sample's README](../laya-sample/README.md#run-it) for the two path pitfalls that apply
here too (quoting Windows paths in a POSIX shell, and `LAYA_ONNX_ROOT` needing to be absolute).

`--download` has the same limitation as in `laya-sample`: it can't succeed today, because
`convaiinnovations/laya` publishes PyTorch weights, not an ONNX export.

## What each section demonstrates

1. **Language detection** — `LanguageDetection::analyse` over ~8 states (English, accent-stripped
   Spanish, French, Romanian, Hindi, Arabic, Japanese, and a `serde_json::json!` dict state),
   showing detected script, best-effort Latin-language guess, and whether the English checkpoint
   can be expected to read the text. No model is loaded for this section.
2. **Routing decisions** — `LayaRouter::route` alone (never `predict`), so nothing is loaded.
   Covers the full precedence ladder: script/language auto-detection, an explicit `model`, an
   explicit `lang`, and the opt-in typed-decisions workflow match (shown both with
   `auto_task_detection` on, where it decides the route, and off, where the match is still
   detected but does not win).
3. **Router predict** — the same four support questions (`department`, `urgency`, `churn_risk`,
   `refund_requested`) asked of an English and a Hindi support email through one `LayaRouter` with
   `max_loaded: 2`, so both checkpoints stay resident and neither request evicts the other.
4. **Email cleaning** — a raw email with quoted history, a signature, and a confidentiality
   disclaimer goes through `LayaEmail::clean_body` (printed verbatim) and `LayaEmail::state`, then
   `LayaPresets::email(None)` through the same router.
5. **Shortlist** — a 40-label banking-intent choice question is narrowed to the top 20 by cosine
   similarity (`LayaShortlist::predict`, `DEFAULT_SHORTLIST_K`), using the demo hashing embedder in
   [`hashing_embedder.rs`](src/hashing_embedder.rs) — **demo only, replace with a real embedding
   model**. It is a deterministic, dependency-free, hashed-character-trigram bag-of-features
   vector, ported bit-for-bit from `laya-rust/tools/hashing_embedder.py` (the same port used by the SDK's own
   test suite at `tests/common/hashing_embedder.rs`), chosen only because every language
   in this repo can reproduce it identically without a network call or a bi-encoder model.

## Code walkthrough

[`src/main.rs`](src/main.rs) follows the same shape as
[`laya-sample`](../laya-sample/src/main.rs): hand-parsed arguments (no argument-parsing crate,
because an unrecognized flag should fail loudly rather than be silently dropped), a `LayaRouter`
built once up front, then one function per section. `timed()` wraps each of the three sections
that actually call `predict`, printing wall-clock time to stderr only — the sample's stdout is a
stable contract, and wall-clock time never could be part of it.

## Troubleshooting

Same as [laya-sample's table](../laya-sample/README.md#troubleshooting), with `--model-root` /
`LAYA_ONNX_ROOT` in place of `--model-dir` / `LAYA_ONNX_DIR`. In particular:

| Symptom | Cause |
|---|---|
| `Laya model artifact (~1.3 GB) not found` under `LAYA_ONNX_ROOT env var  <root>\<checkpoint>` | `--model-root`/`LAYA_ONNX_ROOT` points at a directory that has no `<checkpoint>/model.onnx` subdirectory for the checkpoint being routed to. |
| Devanagari, Arabic, or CJK text prints as `?` or boxes | The console code page or font can't display it. The answers aren't affected. Run `chcp 65001` or use Windows Terminal. |
| First `predict` call takes noticeably longer | ORT session warm-up for that checkpoint. Pass `--preload` to pay this cost once, up front, for every checkpoint instead of on first use. |
