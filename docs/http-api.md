# HTTP API

`laya-serve` exposes Laya over the TypeSafe Jev `/v1/systemone` wire protocol. A client written
against Jev -- `hs-jev`, `typesafe-sdk`, or your own -- can point its base URL at this server and
keep working: Laya's `predict()` output is already schema-compatible, and the server adds only the
HTTP surface: one decision route, a health probe, an optional bearer check and request limits.

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
| `LAYA_DEFAULT_MODEL` | checkpoint a state with no language evidence falls back to; aliases such as `ml` resolve the way core resolves them, and an unresolvable name stops the server at startup | `english` |
| `LAYA_API_KEY` | if set, require `Authorization: Bearer <key>` | none |
| `LAYA_LOG_LEVEL` | uvicorn log level | `info` |
| `LAYA_MAX_CONCURRENT` | requests admitted past auth at once; excess gets `503` | `16` |

For a deployment published under a prefix such as `/laya`, set `LAYA_ROOT_PATH=/laya`.
FastAPI uses it when generating OpenAPI and Swagger UI URLs. Configure the reverse proxy to
strip `/laya` before forwarding requests to Laya; the app's routes remain `/health` and
`/v1/systemone` internally.

For containers, including CUDA and ARM64 images, see [Docker quickstart](docker.md).

## Endpoints

### `GET /health`

Always open (no auth), and stays responsive during inference because the CPU-bound forward pass
runs on its own worker, not the event loop.

```json
{"status": "ok", "loaded": ["english", "multilingual"], "revisions": {"english": "...", "multilingual": "..."},
 "device": "cuda", "device_is_preference": false,
 "checkpoint_devices": {"english": "cuda", "multilingual": "cuda"},
 "cpu_fallbacks": {"english": {"count": 0, "last_reason": null}, "multilingual": {"count": 0, "last_reason": null}}}
```

One server's answer, so the blocks agree with each other: every key of `revisions`,
`checkpoint_devices` and `cpu_fallbacks` is a name in `loaded`. `tests/test_serve.py` holds this
sample to the handler that produces it, field by field.

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
| `model` | no | names a checkpoint; anything else is ignored (see below) |
| `task` | no | forces a checkpoint by workflow name instead of letting routing decide; an unknown name is a `422` naming it |
| `lang` | no | a language code (`de`, `en-US`) that skips detection when it names a language; a blank or unrecognised code falls through to detection |
| `lang_guess` | no | a language code from the client's own identifier, consulted after `lang` and before detection; any non-English code routes to the multilingual checkpoint |
| `max_len` | no | total token window for this request, capped by `LAYA_MAX_TOKEN_BUDGET` |
| `head_max_len` | no | token window the option prompt shares, same cap; see [Widening the Token Budget](langchain.md) for when a question needs it |
| `min_confidence` | no | abstention threshold in `[0.0, 1.0]`; an answer whose `answer_confidence` falls below it comes back marked `low_confidence`, and the answer itself is kept |

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
names (`english`, `multilingual`, `typed-decisions`) and their aliases select a checkpoint; any
other value -- including a Jev id like `jev-1` -- means "let the router choose", and the response's
`routing` block records what was chosen and why.

### Response

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "queue": {"type": "choice", "choice": "billing",
              "probabilities": {"billing": 0.9281, "tech": 0.0412, "other": 0.0307},
              "confidence": 0.4534, "answer_confidence": 0.9281,
              "action": {"act_probability": 1.0}},
    "urgency": {"type": "score", "score": 2.6389,
                "legend": {"0": "calm", "1": "firm", "2": "angry", "3": "furious"},
                "probabilities": {"0": 0.0099, "1": 0.0713, "2": 0.536, "3": 0.3828},
                "confidence": 0.3542, "answer_confidence": 0.536,
                "action": {"act_probability": 1.0}}
  },
  "usage": {"input_tokens": 74, "output_tokens": 0},
  "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text",
              "detection": {"script": "latin", "language": "en", "is_english": true, "non_latin_fraction": 0.0}}
}
```

`answers` and `usage` are the keys Jev clients decode; `model` is the constant name of the decision
head, and the checkpoint that answered is in `routing` (`model`, `repo`, `reason`, and the
`detection` or `lang_guess` evidence behind it).

| answer type | keys |
|---|---|
| `choice` | `choice` (the argmax option), `probabilities` per option |
| `score` | `score` (expected level index, may fall between levels), `probabilities` keyed `"0".. "k-1"`, `legend` mapping index to the level text |
| `noul` | `noul`, the probability of the yes option |
| all | `confidence`, `answer_confidence`, and `action.act_probability` |

### Confidence: two numbers, not interchangeable

- `answer_confidence` is the probability mass on the reported answer (`max(p)`). It is the
  quantity temperature scaling fits and the one this repo's ECE figures are computed on, so it
  carries the gating property the [Benchmarks and known limits](benchmarks.md) page relies on --
  but only for a checkpoint whose temperature fit has been validated on your traffic.
- `confidence` means something different per type: normalized entropy `1 - H(p)/log(k)` on
  `choice` and `score`, and `max(p_yes, p_no)` on `noul` (where it equals `answer_confidence`).

Never compare the two against one threshold. Also note the difference when porting from Jev:
TypeSafe defines confidence as `(n*p_max - 1)/(n - 1)`, so a threshold carried over from a Jev
deployment gates differently on Laya's entropy value.

Successful responses also carry `Server-Timing: inference;dur=<ms>` and `X-Inference-Time-Ms`.

## Limits

Request guardrails are checked before tokenization, so an oversized request costs the server
nothing but the bytes it read. Every one of them is a `413`; the `detail` says which limit was hit.

| limit | value |
|---|---|
| request body | 2 MiB, enforced while streaming -- a chunked or understated `Content-Length` cannot bypass it |
| `state` | 50,000 characters of the text the model is given -- the string itself for a string state, `json.dumps(state, ensure_ascii=False)` for an object or array |
| questions per request | 64 |
| options per `choice` question | 100 |
| levels per `score` question | 32 |
| options across all questions | 512 |
| concurrent admitted requests | `LAYA_MAX_CONCURRENT` (16) |

The option caps are HTTP-only amplification guards; the model itself fits option tokens into a
`head_max_len=192` window, so a question inside the HTTP caps can still be refused as a `422` when
the option texts together exceed that budget. The [Evaluation harness](evals.md) runs the same
requests in-process without the HTTP layer.

## Errors

| status | when | body `detail` |
|---|---|---|
| `400` | body is not valid JSON, not an object, has no `questions`, `state` is missing or `null`, or `questions` is not an object | what is wrong |
| `401` | `LAYA_API_KEY` is set and the bearer token is missing or wrong | `invalid or missing bearer token` |
| `413` | any limit above | which limit and by how much |
| `422` | the question is well-formed JSON but invalid to Laya (unknown type, options over the head budget), or a request control (`lang`, `min_confidence`, a hook argument) is not in the form this endpoint accepts | names the question or the field and what to fix |
| `500` | inference failed for any other reason | `inference failed` -- always this string, so paths, weights and memory state never leak; the cause is in the server log |
| `503` | `LAYA_MAX_CONCURRENT` requests are already in flight | `server busy, try again later` |

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

This server speaks one protocol on purpose. There is no OpenAI-compatible endpoint and no batch
endpoint; run several questions in one request instead, since they share a single forward pass per
question set. The `laya` CLI and MCP server cover local use -- see the
[README](https://github.com/NandhaKishorM/laya#readme).
