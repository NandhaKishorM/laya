# HTTP API

`laya-serve` exposes Laya over the TypeSafe Jev `/v1/systemone` wire protocol. A client written
against Jev -- `hs-jev`, `typesafe-sdk`, or your own -- can point its base URL at this server and
keep working: Laya's `predict()` output is already schema-compatible, and the server adds only the
HTTP surface: one decision route, a health probe, an optional bearer check and request limits.

A client that targets `laya-serve` rather than the Jev API exists for PHP:
[`marcreichel/laya-php`](https://github.com/marcreichel/laya-php) is a Composer SDK (PHP 8.4+)
that maps a class of enums and attributes onto questions and returns an instance, reads `GET /health`
for a deploy check, and ships a test fake so callers can unit-test without a running server.

```bash
pip install "laya[serve]"
laya-serve            # http://0.0.0.0:8000
```

The same entry point runs embedded in any ASGI server: `laya.serve.create_app()` builds the FastAPI
app, optionally with a `Router` you inject (`create_app(router)`) instead of one built from the
environment.

## Configuration

Everything is environment variables, so one image serves a laptop dev run and a systemd unit.

| env var | meaning | default |
|---|---|---|
| `LAYA_HOST` | bind address | `0.0.0.0` |
| `LAYA_PORT` | bind port | `8000` |
| `LAYA_ROOT_PATH` | public URL prefix when served behind a reverse proxy | empty |
| `LAYA_DEVICE` | torch device for every checkpoint | auto |
| `LAYA_PRELOAD` | build the checkpoints at startup, not lazily | `1` |
| `LAYA_MODELS` | comma list to preload (`english,multilingual,typed-decisions`); empty = all | all |
| `LAYA_THREADS` | cap torch intra-op threads on CPU; keep it <= physical cores -- oversubscribing logical cores is a large regression | torch default |
| `LAYA_AUTO_TASK` | auto-route to the typed-decisions checkpoint | `0` |
| `LAYA_IDLE_UNLOAD_SECONDS` | unload resident checkpoints after this many idle seconds; the next request loads its checkpoint again. Zero disables unloading | `0` |
| `LAYA_MAX_LOADED` | checkpoints kept resident at once; a cap below what routing chooses rebuilds one per switch, and `preload()` raises it to hold whatever it builds | `2` |
| `LAYA_MAX_TOKEN_BUDGET` | server-side ceiling on the per-request `max_len` and `head_max_len` overrides; a larger value is a `422`. Unparseable or non-positive input logs a warning and falls back | `8192` |
| `LAYA_DEFAULT_MODEL` | checkpoint a state with no language evidence falls back to; aliases such as `ml` resolve the way core resolves them, and an unresolvable name stops the server at startup | `multilingual` |
| `LAYA_EXTRA_MODELS` | JSON object `{name: source}` registering extra checkpoints beside the bundled ones: a Hub repo id or local checkpoint directory as a string, or a `["repo", "subfolder"]` pair. Names get the same `model=` pin a built-in does; a malformed value or bad name stops the server at startup | none |
| `LAYA_API_KEY` | if set, require `Authorization: Bearer <key>` | none |
| `LAYA_LOG_LEVEL` | uvicorn log level | `info` |
| `LAYA_MAX_CONCURRENT` | requests admitted past auth at once; excess gets `503` | `16` |
| `LAYA_MAX_BATCH_TOKENS` | tokens one `/v1/systemone/batch` FORWARD PASS may collate (`states` x questions x row width); a larger batch is split into several passes, not refused | `131072` |
| `LAYA_JEV_STRICT` | serve the strict Jev wire contract: no root `routing`, no per-answer `action` / `answer_confidence`, no `confidence` on noul answers, and `usage` reduced to `input_tokens` + `output_tokens`. For clients that validate the response against the Jev contract with no extra fields | `0` |

For a deployment published under a prefix such as `/laya`, set `LAYA_ROOT_PATH=/laya`.
FastAPI uses it when generating OpenAPI and Swagger UI URLs. Configure the reverse proxy to
strip `/laya` before forwarding requests to Laya; the app's routes remain `/health` and
`/v1/systemone` internally.

For containers, including CUDA and ARM64 images, see [Docker quickstart](docker.md).

For bursty local use, set `LAYA_IDLE_UNLOAD_SECONDS=300`. Inference and unload run on the same
worker, and the idle window starts again when a single or batch forward pass finishes, including
failed requests. The next prediction pays a cold load. Unloading releases model references and
device caches, including Metal; the process allocator may retain RAM pages, so process RSS need
not fall by the size of the checkpoint.

## Endpoints

### `GET /health`

Liveness is always open (no auth), and stays responsive during inference because the CPU-bound
forward pass runs on its own worker, not the event loop. The fields below liveness are not open on
a deployment that set `LAYA_API_KEY`: without the bearer, `/health` answers `{"status": "ok"}` and
nothing else, because the rest names resident checkpoints, their exact revision SHAs, the device
state and each checkpoint's last fallback reason, which quotes host hardware. A probe needs only
the 200, so a healthcheck is unaffected and a wrong bearer is still a 200 rather than a 401. With
no `LAYA_API_KEY` set, every caller gets the full payload shown here.

```json
{"status": "ok", "loaded": ["english", "multilingual"], "revisions": {"english": "...", "multilingual": "..."},
 "device": "cuda", "device_is_preference": false,
 "checkpoint_devices": {"english": "cuda", "multilingual": "cuda"},
 "cpu_fallbacks": {"english": {"count": 0, "last_reason": null}, "multilingual": {"count": 0, "last_reason": null}}}
```

One server's answer, so the blocks agree with each other: every key of `revisions`,
`checkpoint_devices` and `cpu_fallbacks` is a name in `loaded`. `tests/test_serve.py` holds this
sample to the handler that produces it, field by field.

With idle unloading enabled, authenticated health responses also include `idle_unload_seconds`
(the configured window) and `idle_seconds` (time since the last inference request or completion).
Health probes do not reset that clock. An empty `loaded` list is normal after an idle unload.

- `status` is `ok` whenever the process answers at all. It says nothing about the checkpoints.
- `loaded` lists the checkpoints resident in memory. It is empty until a request builds one, which is
  what `LAYA_PRELOAD=0` leaves the process doing.
- `revisions` is the artifact revision each resident checkpoint was loaded from, keyed by the same
  names as `loaded`, so a deployment can confirm what it is actually serving.
- `device` is the device a resident checkpoint really computes on, which is not always what
  `LAYA_DEVICE` asked for: a checkpoint that wants a GPU it cannot get falls back to CPU silently and
  still answers correctly. With nothing resident it is the configured preference instead.
- `device_is_preference` is `true` exactly while nothing is resident, and `false` as soon as the
  handler can measure. That is the difference between a server reporting its configuration and a
  server reporting where its work is: one that quietly lost its GPU says `false` with `device`
  `cpu`, rather than going on answering `cuda`.
- `checkpoint_devices` gives the measurement per checkpoint, keyed by the names in `loaded`;
  `device` is the first of those values.
- `cpu_fallbacks` counts, per resident checkpoint, the requests that exhausted GPU memory and were
  retried once on CPU: `count` since the process started, and `last_reason` carrying the error text
  of the latest one. The demotion is scoped to the request that failed, so a checkpoint built on CPU
  because the GPU was never available is not a fallback and counts `0` here -- that shows in `device`.

### `POST /v1/systemone`

One request carries a `state` and any number of questions over it:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice this month, I want my money back",
  "questions": {
    "queue":   {"type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "billing and refunds", "tech": "login and app issues",
                             "other": "everything else"}},
    "urgency": {"type": "score",  "instructions": "How urgent?",
                "criteria": ["calm", "firm", "angry", "furious"]}
  }
}'
```

| field | required | meaning |
|---|---|---|
| `state` | yes | text, email, ticket or JSON document to decide on; a missing or `null` state is a `400` |
| `questions` | yes | object keyed by question id; each question is `choice` / `score` / `noul` with `instructions` and `criteria` |
| `model` | no | names a checkpoint; a path or unpublished Hub id is a `422`, anything else is ignored (see below) |
| `task` | no | forces a checkpoint by workflow name instead of letting routing decide; an unknown name is a `422` naming it |
| `lang` | no | a language code (`de`, `en-US`) that skips detection when it names a language; a blank or unrecognised code falls through to detection |
| `lang_guess` | no | a language code from the client's own identifier, consulted after `lang` and before detection; any non-English code routes to the multilingual checkpoint |
| `max_len` | no | total token window for this request, capped by `LAYA_MAX_TOKEN_BUDGET` |
| `head_max_len` | no | token window the option prompt shares, same cap; see [Widening the Token Budget](langchain.md) for when a question needs it |
| `min_confidence` | no | abstention threshold: a number in `[0.0, 1.0]`, or a per-bucket map keyed by option-count bucket (`{"choice:3-5": 0.9, "default": 0.4}`) so the threshold can differ by option count; an answer whose `answer_confidence` falls below its own threshold comes back marked `low_confidence`, and the answer itself is kept |

`model`, `task`, `lang`, `lang_guess`, `max_len`, `head_max_len` and `min_confidence` are the
arguments `Router.predict` takes that a JSON body can state; each is forwarded only when the request
sends it, so an absent one leaves the deployment's own `Router(...)` setting in charge. The five
hook arguments `predict` also takes -- `hooks`, `on_predict_start`, `on_predict_end`,
`hooks_raise`, `hooks_timeout` -- are refused with a `422` rather than dropped: a hook is a callable
that runs inside the server process, and the last two say how the hooks a deployment installed
execute, so no value a caller sends has a meaning here. The same five are refused client-side by a
LangChain node with a `base_url` (`laya.integrations.langchain`), so a chain and a raw HTTP client
now get the same answer.

`model` is accepted so a Jev client can keep sending one. The public Hugging Face ids
(`convaiinnovations/laya-multilingual`, `convaiinnovations/laya-typed-decisions`), the checkpoint
names (`english`, `multilingual`, `typed-decisions`) and their aliases select a checkpoint, and so
does every name `LAYA_EXTRA_MODELS` registered on this server's Router -- including a name that
re-points a built-in, which is how a fine-tune is served under `english` without a code change.
`convaiinnovations/laya`, and any other value that is not a path or a Hub repo id -- including a
Jev id like `jev-1` -- means "let the router choose", and the response's `routing` block records
what was chosen and why. A value that looks like a filesystem path or an unpublished Hub id
(`/path/to/checkpoint`, `org/repo`, `~/ckpt`, `.\ckpt`) is a `422` on both `/v1/systemone` and
`/v1/systemone/batch`: this server cannot load it, and answering with another checkpoint would
hide that. The detail is the same `unknown model` text core raises, plus the reminder to omit
`model` to let the router choose.

### Response

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "queue": {"type": "choice", "choice": "billing",
              "probabilities": {"billing": 0.9519, "tech": 0.0327, "other": 0.0154},
              "confidence": 0.797, "answer_confidence": 0.9519,
              "action": {"act_probability": 1.0}, "x_jev_confidence": 0.9278},
    "urgency": {"type": "score", "score": 1.6994,
                "legend": {"0": "calm", "1": "firm", "2": "angry", "3": "furious"},
                "probabilities": {"0": 0.0249, "1": 0.4136, "2": 0.3985, "3": 0.1629},
                "confidence": 0.1925, "answer_confidence": 0.4136,
                "action": {"act_probability": 1.0}, "x_jev_confidence": 0.2507}
  },
  "usage": {"input_tokens": 83, "output_tokens": 0, "state_tokens": 12,
            "state_tokens_dropped": 0, "truncated": false, "truncated_questions": []},
  "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text",
              "detection": {"script": "latin", "script_profile": {"latin": 1.0}, "language": "en",
                            "is_english": true, "language_undecided": false, "diacritic_rate": 0.0,
                            "non_latin_fraction": 0.0, "mixed_segment": null},
              "workflow": null}
}
```

The sample is one answer this server gave, verbatim: the request above, the cached `english`
checkpoint on CPU. `answers` and `usage` are the keys Jev clients decode; `model` is the constant
name of the decision head, and the checkpoint that answered is in `routing`.

| answer type | keys |
|---|---|
| `choice` | `choice` (the argmax option), `probabilities` per option, `x_jev_confidence` |
| `score` | `score` (expected level index, may fall between levels), `probabilities` keyed `"0".. "k-1"`, `legend` mapping index to the level text, `x_jev_confidence` |
| `noul` | `noul`, the probability of the yes option |
| all | `type` (the discriminator, echoing the question's own type), `confidence`, `answer_confidence`, and `action.act_probability` |
| gate | `abstention`, `abstention_threshold` and `low_confidence`, written by the abstention gate -- see below |

The gate row is the abstention report (#361), and it is the only way a caller can see that the gate
it paid for ran. A request that sets `min_confidence` gets it; one that does not gets none of the
three keys. `abstention` is one of three states, written on **every** answer of a gated request:
`passed` (its confidence cleared the threshold), `abstained` (it fell below, and `low_confidence` is
`true` on exactly those answers), or `unevaluated` (the answer carried no usable confidence, so the
gate could not decide -- reporting that as a pass would be the same lie as reporting it as a flag).
`abstention_threshold` echoes the threshold those states were measured against, which is what makes a
batch run with per-class thresholds re-splittable after the fact. With `min_confidence` unset none of
the three keys appear on any answer: absence is the report, not a fourth state, and it is how a caller
tells an ungated run from a cleared gate. `min_confidence` of exactly `0.0` *was* set, so states are
reported, and nothing can fall below it, so every answer reads `passed` -- the echoed `0.0` is what
distinguishes that from a pass at a real threshold. The answer itself is kept in every state; the gate
marks, it does not drop.

`usage` reports what the forward pass was built from. How much of a state the model reads is a token
budget, not a character count, and the budget moves with `max_len`, `head_max_len` and every
question's own option prompt (#174), so these keys are the only place that fact is visible:

| `usage` key | meaning |
|---|---|
| `input_tokens` | non-pad tokens of the state's rows -- one row per question, so it grows with the questions rather than being a context length |
| `output_tokens` | always `0` -- the head answers in one pass, it generates nothing |
| `state_tokens` | tokens the whole serialized state needs |
| `state_tokens_dropped` | tokens of it at least one question did not get: the worst case over the questions, since each leaves the state a different room |
| `truncated` | `true` when that worst case dropped anything |
| `truncated_questions` | the ids of the questions whose own window was cut, `[]` when none |
| `options` | present only when some question's options no longer have a token span each: keyed by question id, with `total` (the options that question defines), `distinct` (the spans that reached the sequence) and `tokens_per_option` |

A truncated answer is still an answer -- the head decides on the evidence it was given -- but a
caller sizing states by character count cannot see the cut anywhere else in the response.

`routing` records which checkpoint answered and why:

| `routing` key | meaning |
|---|---|
| `model` | the checkpoint that answered: `english`, `multilingual` or `typed-decisions` |
| `repo` | where it was loaded from, as `repo` or `repo/subfolder`. The default server loads all three from the `convaiinnovations/laya` bundle, so `multilingual` and `typed-decisions` report `convaiinnovations/laya/multilingual` and `convaiinnovations/laya/typed-decisions` -- a readable id, not a Hub repo id to fetch with |
| `reason` | the sentence for the choice, naming the evidence it acted on |
| `detection` | `laya.lang.analyse()` on the state -- `script`, `script_profile`, `language`, `is_english`, `language_undecided`, `diacritic_rate`, `non_latin_fraction`, `mixed_segment` -- or `null` when the route decided before reading the text |
| `workflow` | the typed-decisions workflow the question ids match, or `null` |

`detection` is `null` on every path that decides without reading the state: one forced by `model` or
`task`, one answered by `lang` or `lang_guess`, or one that matched a typed-decisions workflow from
the question ids. A `lang_guess` leaves no key of its own -- the hint it acted on is named in
`reason`. The `model` and `task` branches report `workflow` as `null` too, because they answer
before the question ids are read.

### Confidence: three numbers, not interchangeable

- `answer_confidence` is the probability mass on the reported answer (`max(p)`). It is the
  quantity temperature scaling fits and the one this repo's ECE figures are computed on, so it
  carries the gating property the [Benchmarks and known limits](benchmarks.md) page relies on --
  but only for a checkpoint whose temperature fit has been validated on your traffic.
- `confidence` means something different per type: normalized entropy `1 - H(p)/log(k)` on
  `choice` and `score`, and `max(p_yes, p_no)` on `noul` (where it equals `answer_confidence`).
- `x_jev_confidence` is the `confidence` Jev would report for the same probabilities, on `choice`
  and `score` answers only (a Jev `noul` answer has no confidence). It uses TypeSafe's own
  formulas: `(p_max - 1/n)/(1 - 1/n)` for `choice`, and for `score` one minus the probability-weighted
  distance from the most likely level, scaled by the same distance for a uniform distribution.

Never compare them against one threshold. When porting from Jev, a threshold carried over from a
Jev deployment gates differently on Laya's entropy `confidence` (#302); read it against
`x_jev_confidence` instead. The `x_` prefix keeps the field clear of any name Jev adds later.

### Strict Jev contract: `LAYA_JEV_STRICT`

The payload above is the full Laya payload. The Jev contract a client may hold it to defines
less: three top-level fields (`model`, `answers`, `usage`), the contracted keys on each answer
and nothing else, and a `usage` of the two token counts. A client that validates the response
against that contract with no extra fields -- OpenClaw's TypeSafe provider plugin is one --
rejects the full payload, so `LAYA_JEV_STRICT=1` projects the response onto the contract before
answering, on both `/v1/systemone` and `/v1/systemone/batch`:

- the root keeps `model`, `answers` and `usage` only; `routing` is not sent;
- a `choice` answer keeps `type`, `choice`, `probabilities` and `confidence`;
- a `score` answer keeps `type`, `score`, `probabilities`, `confidence` and `legend`;
- a `noul` answer keeps `type` and `noul`;
- `usage` keeps `input_tokens` and `output_tokens`; the truncation facts and the collapsed-
  options ceiling are not sent;
- `x_jev_confidence` is not sent, like the other additions.

The projection keeps only the contracted keys and recomputes nothing: every value is the one the
result already carries, so the probabilities and scores a strict client reads are identical to
the ones the full payload reports. The default stays the full payload, and a deployment that
turns the flag on loses the truncation visibility `usage` provides -- a cut state is then
visible in the logs, not in the response. Score `criteria` should stay plain strings under the
strict contract: a strict client compares the returned `legend` against the criteria it sent,
and Laya renders a structured criterion with Python's JSON, which a JavaScript caller that
stringifies its own criteria may not match byte for byte.

Successful responses also carry `Server-Timing: inference;dur=<ms>` and `X-Inference-Time-Ms`.

## Limits

Request guardrails are checked before tokenization, so an oversized request costs the server
nothing but the bytes it read. Every size limit below is a `413`; the `detail` says which limit was
hit. The last row is the exception: the concurrency cap is an admission limit, not a size one, and
it answers `503` with `Retry-After: 1`.

| limit | value |
|---|---|
| request body | 2 MiB, enforced while streaming -- a chunked or understated `Content-Length` cannot bypass it |
| `state` | 50,000 characters of the text the model is given -- the string itself for a string state, `json.dumps(state, ensure_ascii=False)` for an object or array |
| questions per request | 64 |
| `states` per batch request | 64 |
| options per `choice` question | 100 |
| levels per `score` question | 32 |
| options across all questions | 512 |
| concurrent admitted requests (`503`, not `413`) | `LAYA_MAX_CONCURRENT` (16) |

`/v1/systemone/batch` is bounded differently, and not by a refusal. It tokenizes each state once per
question and collates every row into a single tensor, so the field caps multiply: 64 states of 64
questions is 4096 rows, which every other limit on this page permits. What a row costs is its width,
and `max_len` is itself a request field, so the cost of a batch is `states x questions x width`.

Rather than refuse a large batch, the endpoint **splits** it: when that product exceeds
`LAYA_MAX_BATCH_TOKENS` (131,072 by default) it chooses a `batch_size` so each forward pass stays
inside the budget, and `Router.predict_batch` runs the batch in several passes. Every state is still
answered and the response is unchanged. A request whose rows already fit is passed no `batch_size` at
all, so it behaves exactly as before -- which matters because batch shape can move floating-point
results. A `batch_size` the caller sends always wins: it asked for a shape.

At the default, 256 rows go through in one pass -- 64 states of 4 questions, or 8 of 32. Larger
batches are split, and raising `max_len` makes each pass narrower rather than costing 16 times the
work. What this does not bound is how long one request occupies the server; that is
`LAYA_MAX_CONCURRENT` and the single inference worker, and is already true of one `/v1/systemone`
request over a 50,000-character state.

The option caps are HTTP-only amplification guards; the model itself fits option tokens into a
`head_max_len=192` window, so a question inside the HTTP caps can still be refused as a `422` when
the option texts together exceed that budget. The [Evaluation harness](evals.md) runs the same
requests in-process without the HTTP layer.

## Errors

| status | when | body `detail` |
|---|---|---|
| `400` | body is not valid JSON, not an object, has no `questions`, `state` is missing or `null`, `questions` is not an object, or a string anywhere in the body holds an unpaired `\udXXX` surrogate escape | what is wrong |
| `401` | `LAYA_API_KEY` is set and the bearer token is missing or wrong | `invalid or missing bearer token` |
| `413` | any limit above | which limit and by how much |
| `422` | the question is well-formed JSON but invalid to Laya (unknown type, options over the head budget), or a request control (`lang`, `min_confidence`, a hook argument) is not in the form this endpoint accepts | names the question or the field and what to fix |
| `500` | inference failed for any other reason | `inference failed` -- always this string, so paths, weights and memory state never leak; the cause is in the server log |
| `503` | `LAYA_MAX_CONCURRENT` requests are already in flight | `server busy, try again later` |

The unpaired-surrogate `400` is the one that looks unusual. `\udXXX` with no pair is legal JSON, but
the character it names cannot be UTF-8 encoded, so the tokenizer raises `TypeError` on it -- the
caller's own string arriving as a server fault, with a traceback per request. Both decision routes
therefore walk the parsed body for lone surrogates and refuse one before it reaches inference. The
walk runs after the size checks, so an oversized body is still refused first and the character and
question limits bound what it can reach. A *paired* surrogate is one ordinary astral character by
the time the parser is done, so an emoji in a state is unaffected.

Over-cap load is refused, not queued: clients holding an admission slot while streaming a slow body
cannot starve `/health`, and a retry can take the slot a refused client left.

## Concurrency model

Inference is a synchronous torch call that takes hundreds of milliseconds to seconds on CPU, so it
never runs on the event loop: requests are handed to a single-worker executor, which means one
forward pass at a time -- the shape a single checkpoint on one device wants. Admission (the
`LAYA_MAX_CONCURRENT` semaphore) is checked before any body byte is read and held through
inference; the inference gate is joined only after the body is complete, so a slow client holds an
admission slot but never an inference slot.

## Not (yet) here

This server speaks one protocol on purpose. There is no OpenAI-compatible endpoint; run several
questions in one request instead, since they share a single forward pass per question set. The one
other route is `POST /v1/systemone/batch`, which answers one `questions` set over an array of
`states`. It has no section on this page yet -- its request shape is in the README's self-hosting
section -- and every check above applies to it as it does to `POST /v1/systemone`: the same shape
`400`s, the same unpaired-surrogate refusal, the same auth, admission, size limits, body-control
validation and `500` mapping. The `laya` CLI and MCP server cover local use -- see the
[README](https://github.com/NandhaKishorM/laya#readme).
