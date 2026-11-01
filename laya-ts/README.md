# laya-ts

TypeScript inference for Laya (`Agent.predict`, `Router`, `lang`, `email`, `presets`, `shortlist`, `hooks`) on Node and the browser via split ONNX (`encoder.onnx` + `head.onnx`). ESM-only (`"type": "module"`); no CJS build — import from ESM or bundle.

## Export weights (once per checkpoint)

```bash
python laya-ts/scripts/export_onnx.py --model-dir <ckpt> --out-dir ./model
# writes encoder.onnx, head.onnx + copies tokenizer.json, rl_agent_config.json
# verifies torch vs ONNX match within 1e-4 (skip with --no-verify)
```

## Node (CPU/CUDA)

```ts
import { Agent, Router } from "laya-ts";

const agent = await Agent.load("./model"); // local dir, or ("convaiinnovations/laya", { subfolder: "multilingual" })
const router = new Router({ default: "english" }); // this example attaches only the English checkpoint
router.attach("english", agent);
const out = await router.predict({ body: "charged twice, refund please" }, {
  intent: { type: "choice", instructions: "What does the customer want?", criteria: { refund: "money back", other: "anything else" } },
});
console.log(out.answers.intent);
```

CUDA: `Agent.load("./model", { device: "cuda" })` (falls back to CPU with a warning).

`new Router()` uses the multilingual checkpoint for text whose language the detector cannot
identify, including text with no letters, matching Python 0.4.0. Identified English still routes
to the English checkpoint, and detected non-Latin scripts route to multilingual. Set
`new Router({ default: "english" })` to restore the earlier fallback; explicit `model` and `lang`
options still take precedence.

## Browser (WebGPU → WASM fallback)

```ts
import { Agent } from "laya-ts";

const agent = await Agent.load("https://example.com/models/laya"); // serves encoder.onnx, head.onnx, tokenizer.json, rl_agent_config.json
const out = await agent.predict("charged twice", {
  d: { type: "choice", instructions: "pick", criteria: { refund: "money back", other: "rest" } },
});
```

`onnxruntime-node` / `onnxruntime-web` are optional peer deps, imported lazily behind the provider you use.

## Long documents (`predictLong`)

`Agent.predictLong` and `Router.predictLong` scan overlapping token windows locally, in Node
or the browser. This lets a question consider a long ticket or conversation beyond the single
window used by `predict`. The Router chooses and loads a checkpoint once for the document.

```ts
const scanned = await router.predictLong(longTicket, {
  refund: { type: "noul", instructions: "Does the customer request a refund?" },
  action: { type: "choice", instructions: "Choose the next action", criteria: ["refund", "investigate"] },
}, { window: 256, stride: 128, batchSize: 4 });
console.log(scanned.answers.refund, scanned.usage.windows, scanned.routing);
```

The only aggregation mode is `"auto"`: `noul` takes the largest value; `choice` and `score`
take the answer with the highest `answer_confidence`. Ties retain the first result. The
winner keeps its metadata. Its confidence is a window score, not calibrated confidence for
the whole document; scanning more windows can change which answer wins.

The window is capped by the space left by the questions in the checkpoint's configured
sequence budget. Its default is `max(64, max_len - head_max_len - 8)`, then capped; stride
defaults to half that effective window. An explicit stride cannot leave gaps. `window`,
`stride`, and `batchSize` must be positive safe integers when supplied; null/undefined use
defaults. `maxLen`, `headMaxLen`, and `minConfidence` are deliberately absent from the long
options. Window sizing and aggregation have different meanings from single-pass truncation
and confidence gating. Ordinary prediction options and result types remain unchanged.

Windows follow Python 0.4.18's decode → batch prediction → re-encode path. The official
English and multilingual tokenizers support decoding. A custom encode-only tokenizer still
supports ordinary predictions and the short-input fast path; splitting requires `decode`
and throws if it is unavailable. Decoding and re-encoding can change the token count,
particularly for multilingual text, so normal truncation can still occur. An answer's
optional `window.token_start` / `token_end` identifies its **original candidate token slice**,
not a guarantee that exactly those tokens reached the model.

Agent hooks wrap one window batch; Router hooks wrap one document. Process defaults run
once. Reordering result objects preserves identifiable origins, while rewriting/reordering
states, replacing an answer object, or editing prediction fields removes window attribution.
In-place reliability/action annotations preserve it. An end hook may uniformly remove its own
temporary questions, but cannot silently remove a caller's question. Inconsistent answer
schemas or window counts fail. A skipping hook must return exactly one document result;
the result and usage are copied, a warning is emitted, and `usage.windows` is zero.
The warning also covers short-input and Router cache hits, as agreed in #1316; Python
0.4.18 itself only warns on the multi-window Agent skip path. Hook results are validated
against the TypeScript return contract, which also rejects malformed payloads Python accepts.

If a start hook changes questions or token budgets so planned windows no longer fit, the
scan is refused **after inference**. This check is outside hook error suppression and hook
timeouts; the forward-pass cost has already been incurred. The short path checks this too.
`LongResult` / `RoutedLongResult` have separate usage types: `truncated` is a boolean on the
short path and a numeric count on the multi-window path. Numeric usage is summed, dictionaries
are shallow-merged, other values come from the last result, and `output_tokens` stays zero.

Deterministic parity fixtures are recorded directly from Python `ONNXAgent.predict_long`
at v0.4.18, with source hashes, and replayed by TypeScript tests. The generator's `--check`
detects later reference behavior changes; these checks establish control-flow parity, not
model quality or a speed improvement.

To repeat the optional real-inference comparison with an English checkpoint and its split
ONNX export (weights are not downloaded by these scripts):

```bash
# From laya-ts/, with numpy, torch, transformers and onnxruntime installed in Python.
python scripts/predict_long_onnx_parity.py --python-root /path/to/laya-v0.4.18 \
  --checkpoint /path/to/english-checkpoint --bundle /path/to/split-onnx --output python-long.json
npm run build
node scripts/predict_long_onnx_parity.mjs python-long.json comparison.json
```

The Python harness calls `ONNXAgent.predict_long` with an adapter joining the same encoder
and head graphs used by TypeScript. It records source/artifact hashes and checks all three
answer types on short, overlapping long, and structured inputs. The Node harness verifies
artifact digests and compares actual token/marker/mask inputs, decoded answers, attribution,
and usage at absolute tolerance
`1e-4`, excluding the existing runtime model-name difference. It does not measure retrieval
quality, accuracy, or performance, and does not cover browser GPU numerical differences.

## Hooks (observe or shape every decision)

Port of the Python `laya.hooks` lifecycle. A hook is a `(ctx) => void` for `onPredictStart` /
`onPredictEnd`, or an object implementing any subset of `onPredictStart`, `onPredictEnd`,
`onRoute`, `onLoad`, `onEvict`, `onError`. A hook may be `async`: it is awaited, in order, before the
call continues, and a rejection follows `hooksRaise` like a thrown error. `onRoute` runs inside the
synchronous `route()`, so it is not awaited and a rejection there is only logged:

```ts
const tracer = {
  onPredictStart(ctx) { console.time(ctx.runId); },
  onPredictEnd(ctx) { console.timeEnd(ctx.runId); console.log(ctx.model, ctx.usage, ctx.elapsedMs); },
};
const router = new Router({ hooks: [tracer], hooksRaise: false }); // telemetry must not fail a request
await router.withHooks([auditHook], () => router.predict(state, questions)); // scoped install

// a start hook may rewrite ctx.states / ctx.questions, or serve a cached result:
const cache = { onPredictStart(ctx) { const hit = lookup(ctx.states[0]); if (hit) ctx.skip([hit]); } };
// an onRoute hook may replace ctx.decision (e.g. pin a checkpoint)

// subclass BaseHook to override only the events you need:
class MetricsHook extends BaseHook {
  onPredictEnd(ctx) { record(ctx.usage); }
}

// process-wide defaults run before installed and per-call hooks for every Agent/Router,
// so a tracer or metrics hook does not have to be threaded through every construction:
setDefaultHooks([new MetricsHook()]);  // addDefaultHook(...) appends; clearDefaultHooks() resets
```

### Option stability

`OptionStabilityHook` adds opt-in `reliability` diagnostics to choice answers. It expands
the questions with reorder and letter-rename probes and evaluates them through the existing
batch path. The original answer stays in place; internal probe answers and their usage keys
are removed before the result returns. Extra rows cost inference work even within one call.

```ts
import { Agent, OptionStabilityHook, calibrateReliability, decideReliability } from "laya-ts";

const agent = await Agent.load("./dist/laya");
agent.addHook(new OptionStabilityHook({
  accept: null, escalate: null, variants: "fast", seed: "evaluation-v1",
}));
const result = await agent.predict("Please approve the request", {
  action: { type: "choice", instructions: "Choose action", criteria: ["approve", "reject"] },
});
if (result.answers.action.type === "choice") {
  console.log(result.answers.action.reliability);
}

// Fit on your labelled examples, using only probed numeric soft_stability values.
// Keep these scores and cutoffs as numbers; round only a separate display string.
function fitAndReplay(scores: number[], correct: boolean[], savedScore: number) {
  const [accept, escalate] = calibrateReliability(scores, correct, { targetAccuracy: 0.9 });
  return decideReliability(savedScore, accept, escalate);
}
```

Both thresholds are required. `accept: null` disables categorical decisions;
`escalate: null` disables only `ESCALATE`. No calibrated defaults are recommended.
`ACCEPT` uses `soft_stability >= accept`; `ESCALATE` uses
`soft_stability < escalate`; otherwise the decision is `VERIFY`. An unprobed question has
null signals. The report keeps the unrounded mean used by the comparator, matching the
Python precision correction in #1289. Other diagnostic fractions retain Python's
four-decimal rounding. Historical rounded Python reports cannot recover the lost precision.

Calibration requires equally sized, nonempty score/boolean-label arrays, finite scores in
`[0, 1]`, and accuracy targets in `[0, 1]`. Invalid evidence is rejected rather than silently
dropping unmatched rows. The helpers match Python on valid inputs; they do not copy its
permissive handling of invalid calibration inputs.

Options use camelCase (`nShuffles`, `renameMaxOptions`); result fields match Python's
snake_case fields. `variants` accepts `"fast"`, `"full"`, or an array of variant names;
`probes` accepts `"auto"`, `"reorder"`, `"rename"`, `"all"`, or a probe-name array.
Renaming retains original label text for missing descriptions, including blank strings,
as in #1286, and preserves nonblank descriptions and caller `option_order`.

Explicit string seeds and the same ordered labels produce the same permutations as Python's
CRC32-seeded shuffle. JavaScript enumerates integer-like object keys numerically, so preserve
the intended label order when comparing inputs across languages.
They reproduce probe selection, not model outputs. The default seed uses the single string
state, or `"batch"` for multiple states. A single structured state uses the existing TS state
serializer; Python uses `repr`, so supply the same explicit seed for cross-language replay.
The shared fixtures record their Python version and source hashes. Numerical comparison is
for those decoded inputs; different inference backends or Python floating-point accumulation
versions can still produce different scores near a threshold.

The hook reads `agent.cfg.option_layout` to choose probes automatically: a declared
parallel layout selects rename-only probes; sequential selects both. This is probe selection,
not an implementation of parallel attention. The current local ONNX runtime supports the
sequential path; this hook adds no new checkpoint/backend support. It also does not add the
feature to the separate HTTP SDK.

Install the hook once at the Agent, Router, or process-default scope. `predictBatch` and
concurrent calls can share one instance; plans are scoped to each prediction context and
released on completion or error. A cache hit without probe answers does not gain a fabricated
reliability score. The hook's own methods are synchronous and work within the existing async
Agent/Router prediction lifecycle.

## Structured decisions (`decide`)

Turn a JSON schema into typed values in one call — the port of Python's `laya.structured`
(#280). Enum properties become choice questions, booleans become noul, bounded integers
become scores; anything the fixed-option model cannot answer (free strings, arrays, nested
objects, `$ref`) is rejected with a `SchemaError` naming the path:

```ts
import { Agent, decide } from "laya-ts";

const agent = await Agent.load("./dist/laya");
const values = await agent.decide(ticketText, {
  type: "object",
  properties: {
    department: { type: "string", enum: ["billing", "support", "sales"] },
    urgency: { type: "integer", minimum: 0, maximum: 2 },
    needs_human: { type: "boolean" },
  },
});
// { department: "billing", urgency: 2, needs_human: false }
```

`router.decide(...)` works the same way (routing options are forwarded to `predict`), and the
free `decide(runner, state, schema, opts)` accepts anything with a `predict` method. Pass
`{ returnDetails: true }` for per-field `confidence`, `answerConfidence` (calibrated `max(p)`),
and probabilities, or `{ questions }` instead of a schema to get raw answers.

For high throughput over multiple inputs, `agent.decideBatch(states, schema)` (and `router.decideBatch`,
or the free `decideBatch(runner, states, schema, opts)`) plans the schema once and evaluates every state
through shared forward passes on `predictBatch` (mirroring Python's `laya.decide_batch`).

Zod/TypeBox users can pass `z.toJSONSchema(Model)` —
any object with a `toJSONSchema()` method is accepted. `planFromJsonSchema`,
`questionsFromJsonSchema` and `answersToJson` expose the planning and projection steps.

## Truncation reporting

The state is clamped to whatever token room a question's head leaves, and that budget moves
with `max_len`, `head_max_len` and the rendered head — so `usage` reports it instead of letting
you guess from character counts (issue #174; mirrors Python #181):

```ts
const out = await agent.predict(longState, questions);
out.usage.truncated;             // true when any state token was dropped
out.usage.state_tokens;          // encoded length of the full state
out.usage.state_tokens_dropped;  // worst case across the questions
out.usage.truncated_questions;   // ids of the questions whose head left too little room
```

The fields are absent only where no state was ever encoded (empty question schema, or a
start hook that supplied the result).

### Collapsed options

The same budget also cuts the options themselves: each is capped at 48 tokens, and once they
overflow `head_max_len` all of them are re-capped at `max(4, (head_max_len - 16) // n)`. Two
options that share a prefix can come out of that cut as the *same* token span, so the question
can no longer name them apart while still answering normally — and an answer chosen from 42
distinguishable spans of 58 has an accuracy ceiling of 72% that nothing else in the response
mentions (issue #538; mirrors Python `laya.common.collapsed_options`):

```ts
out.usage.options;   // absent when every option kept a span of its own
out.usage.options?.intent;  // { total: 58, distinct: 42, tokens_per_option: 4 }
```

`total` is what the question defines, not the markers that reached the sequence, so a report
cannot read "43 of 43" about a question whose missing options never entered the input at all.

## Batching (many states, one call)

Port of `Agent.predict_batch` / `Router.route_batch` / `Router.predict_batch`. The throughput
path: states that share a question schema are collated into one shared forward pass (or one
per `batchSize` chunk) instead of one pass per state.

```ts
// Agent: same questions over many states; results align with `states` by index.
const results = await agent.predictBatch(states, questions, { batchSize: 32 });

// Router: heterogeneous requests — route first, then each checkpoint scores its requests
// in as few forward passes as possible. Results keep input order and carry `routing`.
const decisions = router.routeBatch(requests);          // validate + route, nothing loads
const routed = await router.predictBatch(requests, 32); // == router.predictMany(...)
// Per-call hooks are composed like predict: installed hooks first, then this list,
// then onPredictStart / onPredictEnd. null and [] add nothing. hooksRaise overrides
// the Router's policy for this call. The list runs once per request, including onRoute.
const traced = await router.predictBatch(requests, null, {
  hooks: [tracer],
  onPredictEnd(ctx) { console.log(ctx.usage); },
  hooksRaise: false,
});

// requests routed to the same checkpoint still split into separate batches when their
// question schemas differ (order-sensitively), when per-request start hooks set
// different ctx.maxLen / ctx.headMaxLen overrides, or — for an agent carrying
// lang_temperatures — when their languages differ: each request's effective language (an
// explicit lang, otherwise the detected non-English one) is forwarded to the agent, so
// the batched path scores exactly like predict. Router-level predict hooks run once per
// request: ctx.decision is set, ctx.skip() serves a cached result, and onPredictEnd runs
// per request even when the batch fails.
```

Python's `sort_by_length` grouping is not ported yet.

## Shortlist (many labels)

```ts
import { shortlistChoice, predictShortlist, embedFnFromAgent, cached } from "laya-ts";

const cachedEmbed = cached(embedFn);
const keep = await shortlistChoice(state, bigCriteriaDict, cachedEmbed, 20);
const out = await predictShortlist(agent, state, questions, cachedEmbed, 20);
// out.shortlist[qid] = { labels, scores, k, n, passthrough }
// embedFnFromAgent(agent) mean-pools the loaded encoder; a dedicated bi-encoder usually shortlists better.
// cached(embedFn) keeps an LRU cache across calls; cachedEmbed.cacheInfo() reports size/hits/misses.
```

### Tournament (without an embedder)

For a large tool or intent catalog, `predictTournament` lets the decision model narrow the
options itself. It ports Python's `predict_tournament` to the local Node/browser runtime:

```ts
import { predictTournament } from "laya-ts";

const out = await predictTournament(agent, state, questions);
console.log(out.answers.intent);
console.log(out.tournament.intent); // { labels: [...finalists], n: originalCount, rounds: 1 }

// Pick a group size and forward options to every prediction, including the final one.
const routed = await predictTournament(router, state, questions, 16, { model: "english" });
```

Each oversized choice is split into balanced groups of at most 16 labels. All groups from
all questions share one `predict` call per round; their winners advance until each choice
fits. Up to 256 labels therefore take one elimination call and one final call. Small choices
and non-choice questions reach the final call unchanged, and a request needing no
elimination makes exactly one call. The helper does not mutate the supplied questions.

The returned `tournament` metadata covers every choice, including `rounds: 0` for a
passthrough. Probabilities and confidences describe **only the finalists**; `usage`,
`routing`, and other prediction fields come from the final call, not all rounds combined.
Hooks run for every underlying call. Calibrate thresholds on the strategy you actually use;
a threshold fitted on the full catalog does not automatically transfer to a tournament.
Widening the head budget or using a dedicated shortlist embedder can still be a better fit.

A narrowed choice drops its original `option_order` with one warning per question, matching
Python. Passthrough definitions keep it. This does not add `option_order` handling to the
TypeScript `Agent`, whose prediction path does not currently apply that field.

For a reproducible local comparison, build the package and export a checkpoint as above,
then run from the repository root:

```bash
python laya-ts/scripts/tournament_parity.py --model-dir CHECKPOINT --onnx-dir ./model \
  --fixture ./tournament-python.json
node laya-ts/examples/tournament.mjs --model-dir ./model \
  --fixture ./tournament-python.json --report ./tournament-node.json
```

The Python script records fresh predictions for fixed inputs; Node independently runs
the same inputs through the split ONNX export. The check compares each round's questions,
winners, probabilities and final metadata, including a two-round case. Reports include
artifact hashes, call counts and warmed timings. These few cases check runtime parity;
they do not establish accuracy gains or production latency. Model files stay local.

## Per-language calibration (`lang_temperatures`)

Port of the Python `Agent(lang_temperatures=...)` knob. A language override replaces the
checkpoint's temperature for matching requests — keys normalise to the base subtag
(`de-AT` → `de`), an omitted `temperature` inherits the base one, and
`temperature_by_options` works per option-count bucket as usual:

```ts
const agent = await Agent.load("convaiinnovations/laya", {
  lang_temperatures: {
    de: { temperature: [1.2, 1.2, 1.2] },                 // fitted on German evals
    ja: { temperature_by_options: { "choice:11+": 1.4 } }, // buckets only, base temperature kept
  },
});
await agent.systemOne(state, questions, { lang: "de" });   // uses the German temperature
await router.predict(state, questions);                    // Router forwards the detected language
```

`Router.predict` forwards an explicit `lang` verbatim and otherwise the detected language
(never `"en"` — matching Python, where detection only names non-English languages), so an
override applies exactly to the requests it was fitted on.


## Example (repo root)

```bash
node laya-ts/examples/try-ml.mjs   # needs ./model-ml from the export step
node laya-ts/examples/snake.mjs --ticks 50   # autonomous snake demo, headless smoke (live TUI without --ticks)
```

## Web demo (browser, WebGPU → WASM)

No build step — serve the repo root over HTTP and open the page
(`localhost` counts as a secure context for WebGPU):

```bash
python -m http.server 8000   # run at the repo root
# open http://localhost:8000/laya-ts/examples/web/
```

The page loads `laya-ts/dist` (run `npm run build` inside `laya-ts/`
first), pulls `onnxruntime-web` from a pinned CDN import map, and
fetches the model from the URL in the box (default `../../../model-ml`,
i.e. the exported `./model-ml` at the repo root). First load transfers ~1.3GB and is
cached in CacheStorage afterwards; the encoder tries WebGPU and falls
back to WASM automatically. Chrome/Edge for WebGPU, any modern
browser for WASM.

## Packaging

ponytail: CJS/browser-field dual build + tsconfig tests-include deferred — Task 7 verified ESM-only; CJS needs second tsc config + export-map change, untested. Add when a CJS consumer or browser-field swap is requested.
