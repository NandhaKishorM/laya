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
const router = new Router();
router.attach("english", agent);
const out = await router.predict({ body: "charged twice, refund please" }, {
  intent: { type: "choice", instructions: "What does the customer want?", criteria: { refund: "money back", other: "anything else" } },
});
console.log(out.answers.intent);
```

CUDA: `Agent.load("./model", { device: "cuda" })` (falls back to CPU with a warning).

## Browser (WebGPU → WASM fallback)

```ts
import { Agent } from "laya-ts";

const agent = await Agent.load("https://example.com/models/laya"); // serves encoder.onnx, head.onnx, tokenizer.json, rl_agent_config.json
const out = await agent.predict("charged twice", {
  d: { type: "choice", instructions: "pick", criteria: { refund: "money back", other: "rest" } },
});
```

`onnxruntime-node` / `onnxruntime-web` are optional peer deps, imported lazily behind the provider you use.

## Hooks (observe or shape every decision)

Port of the Python `laya.hooks` lifecycle. A hook is a `(ctx) => void` for `onPredictStart` /
`onPredictEnd`, or an object implementing any subset of `onPredictStart`, `onPredictEnd`,
`onRoute`, `onLoad`, `onEvict`, `onError`:

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

## Shortlist (many labels)

```ts
import { shortlistChoice, predictShortlist, embedFnFromAgent } from "laya-ts";

const keep = await shortlistChoice(state, bigCriteriaDict, embedFn, 20);
const out = await predictShortlist(agent, state, questions, embedFn, 20);
// out.shortlist[qid] = { labels, scores, k, n, passthrough }
// embedFnFromAgent(agent) mean-pools the loaded encoder; a dedicated bi-encoder usually shortlists better.
```

## Example (repo root)

```bash
node laya-ts/examples/try-ml.mjs   # needs ./model-ml from the export step
node laya-ts/examples/snake.mjs --ticks 50   # autonomous snake demo, headless smoke (live TUI without --ticks)
```

## Packaging

ponytail: CJS/browser-field dual build + tsconfig tests-include deferred — Task 7 verified ESM-only; CJS needs second tsc config + export-map change, untested. Add when a CJS consumer or browser-field swap is requested.
