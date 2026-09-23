# @laya/typescript-sdk

A dependency-free HTTP client for Laya's typed decision engine. Supports ESM and
CommonJS on Node.js 22+, and browsers with `fetch`, `AbortController`, and
`structuredClone`. TypeScript consumers need TypeScript 5 or newer.

Inference runs in the companion Python server. JavaScript clients do not need
Python, PyTorch, or model weights installed; the server does. This package does
not run models directly in a browser or Node.js.

## Run from this repository

In a terminal at the repository root, start the server:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[server]'
laya-serve --host 127.0.0.1 --port 8000
```

In a second terminal, build the SDK and run the plain JavaScript example:

```sh
cd sdk/typescript
npm ci
npm run build
node examples/triage.mjs
```

The first prediction downloads and loads the selected Hugging Face checkpoint.
Allow sufficient memory and disk space; each model has hundreds of millions of
parameters. To load the English and multilingual checkpoints before accepting
requests, start with:

```sh
laya-serve --preload english multilingual --device cpu
# Use --device cuda for a suitable GPU installation.
```

The server normally keeps one model loaded. Alternating languages can repeatedly
reload checkpoints. `--preload` retains all named models; alternatively use
`--max-loaded 2`. The client timeout defaults to 120 seconds; the example uses
10 minutes for the first model download.

## Install in another JavaScript project

The npm package name is `@laya/typescript-sdk`. After the first release is
published, install it with:

```sh
npm install @laya/typescript-sdk
```

You can also install the locally built tarball before publication:

```sh
# From sdk/typescript:
npm pack

# From your app, substitute the actual checkout path:
npm install /path/to/laya/sdk/typescript/laya-typescript-sdk-0.1.0.tgz
```

```js
import { Laya, triageQuestions } from '@laya/typescript-sdk';

const laya = new Laya({ baseURL: 'http://127.0.0.1:8000' });
const result = await laya.predict(
  { message: 'I was charged twice. Please refund the duplicate.' },
  triageQuestions(),
);

console.log(result.answers.intent.choice);
console.log(result.answers.refund_requested.noul); // P(true)
console.log(result.routing.model);
```

CommonJS works too:

```js
const { Laya, triageQuestions } = require('@laya/typescript-sdk');
const laya = new Laya();
laya.predict({ message: 'Please refund my order' }, triageQuestions())
  .then(result => console.log(result.answers));
```

## Typed custom questions

Question IDs, primitive types, and choice labels are inferred. Use
`defineQuestions` to preserve literal types when storing a schema in a variable.
Inline schemas also infer automatically.

```ts
import { Laya, defineQuestions } from '@laya/typescript-sdk';

const laya = new Laya();
const questions = defineQuestions({
  department: {
    type: 'choice',
    instructions: 'Which department should handle this request?',
    criteria: {
      billing: 'invoices, payments, refunds',
      technical: 'bugs, outages, integrations',
    },
  },
  urgency: {
    type: 'score',
    instructions: 'How urgent is this request?',
    criteria: ['no time pressure', 'soon', 'blocking issue or deadline'],
  },
  refund: { type: 'noul', instructions: 'Does the user ask for a refund?' },
});

const result = await laya.predict({ body: 'Refund the duplicate charge today' }, questions);
result.answers.department.choice; // 'billing' | 'technical'
result.answers.urgency.score;      // number: expected rubric index, from 0 to 2
result.answers.refund.noul;        // number: P(true), from 0 to 1
result.answers.department.probabilities.billing;
result.answers.department.confidence;
result.answers.department.action.act_probability;
result.usage.input_tokens;
```

Choice criteria also accept an array of unique labels. Structured JSON values are
supported in instructions and descriptions, as in Python. State may be text, a
JSON object, an array (including chat turns), or null. Undefined values, dates,
non-finite numbers, BigInts and cycles are rejected locally instead of silently
changing during serialization. Empty questions or criteria are rejected.

Response fields retain Python's spelling (`input_tokens`, `act_probability`,
etc.). Scores use zero-based rubric levels. Confidence and accuracy have the
same calibration limits as the Python model; see the root README.

## Routing and presets

```js
const decision = await laya.route({ body: 'मुझे पैसे वापस चाहिए' });
console.log(decision.model, decision.reason); // no weights loaded by route()

await laya.predict(state, questions, { model: 'typed-decisions' });
await laya.predict(state, questions, { lang: 'de' });
await laya.predict(state, questions, { task: 'typed_decisions' });
await laya.health(); // liveness and loaded model names; not a model readiness check
```

The server retains Python's precedence: explicit model, explicit task, optional
workflow detection, explicit language, detected language, default model. Enable
automatic workflow detection with `laya-serve --auto-task-detection`.

Available preset functions: `triageQuestions()`, `emailQuestions(categories?)`,
`guardQuestions()`, `moderationQuestions()`, and `routerQuestions()`. Each returns
a fresh schema. Email categories can be customized:

```js
import { emailQuestions } from '@laya/typescript-sdk';
const questions = emailQuestions({ finance: 'payments', engineering: 'bugs' });
```

## Errors, cancellation, authentication

```js
import { Laya, LayaAPIError, LayaTimeoutError } from '@laya/typescript-sdk';

const client = new Laya({
  baseURL: 'http://127.0.0.1:8000',
  apiKey: process.env.LAYA_API_KEY,
  timeoutMs: 120_000,
});
const controller = new AbortController();
try {
  await client.predict(state, questions, { signal: controller.signal, timeoutMs: 300_000 });
} catch (error) {
  if (error instanceof LayaAPIError) console.error(error.status, error.code, error.details);
  else if (error instanceof LayaTimeoutError) console.error('Request timed out');
  else throw error;
}
```

`timeoutMs: 0` disables the deadline. Calling `controller.abort()` stops waiting;
inference already running on the server may still finish. Requests are never
automatically retried. Other errors extend `LayaError`:
`LayaValidationError`, `LayaConnectionError`, `LayaAbortError`, and
`LayaResponseError` (invalid JSON or incompatible response structure).

The client accepts custom `headers` and a Fetch-compatible `fetch` implementation.
Keep API keys on your application's backend. For browser use, put a same-origin
application proxy in front of a protected server, or configure an appropriate
authenticated gateway. A shared server key embedded in browser code is public.

The server defaults to localhost. Set `LAYA_API_KEY` in its environment to require
Bearer authentication on health, routing and prediction endpoints. `HF_TOKEN`,
if needed for checkpoint downloads, is a separate server-only credential.
`LAYA_CORS_ORIGINS=http://localhost:3000,https://your-app.example` enables the
specified browser origins. Cross-origin access is disabled by default; CORS is
not authentication. For remote deployment, configure TLS, rate limiting and
request size limits at your reverse proxy.

## Python service contract

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | None | `{status, version, loaded_models}` |
| `POST /v1/route` | `{state, questions?, model?, task?, lang?}` | Route decision, without inference |
| `POST /v1/predict` | `{state, questions, model?, task?, lang?}` | `{model, answers, usage, routing}` |

Errors use `{error: {code, message, details?}}`, with 401 for authentication,
422 for invalid input, and 500 for internal/model-loading failures. OpenAPI is
available at `/openapi.json` and interactive request documentation at `/docs`.

For custom checkpoint paths or an existing agent, configure Python's router:

```python
import uvicorn
from laya import Router
from laya.server import create_app

router = Router(models={"english": "/path/to/checkpoint"}, device="cpu")
router.preload(["english"])
app = create_app(router, api_key="your-server-key")
uvicorn.run(app, host="127.0.0.1", port=8000)
```

The adapter serializes predictions per process because `Agent` can mutate its
device during GPU fallback. Sync endpoints run in FastAPI's worker pool. Each
additional service process has its own models and memory footprint. There is no
cross-request batching or server-side cancellation in this version.

## Development

```sh
# In sdk/typescript:
npm ci
npm test
PYTHON=../../.venv/bin/python npm run test:integration
python3 scripts/sync_presets.py --check
```

Integration testing uses real local PyTorch inference with a tiny random test
checkpoint, compares answers against direct Python calls, and downloads no model
weights. It verifies transport and runtime compatibility, not pretrained model
accuracy. Run the repository's pretrained model tests separately when those
weights are available. Presets are generated from `laya/presets.py`; regenerate
with `python3 scripts/sync_presets.py` after editing that source.

## Publishing

Publish from `sdk/typescript` using an npm account with publishing rights in the
`@laya` scope. npm uses `@scope/package` names; `laya/typescript-sdk` without `@`
is not this registry package. See [npm's scoped package guide](https://docs.npmjs.com/creating-and-publishing-scoped-public-packages/).

```sh
npm login --registry=https://registry.npmjs.org
npm whoami --registry=https://registry.npmjs.org
npm ci
npm test
PYTHON=../../.venv/bin/python npm run test:integration
python3 scripts/sync_presets.py --check
npm publish --dry-run
npm publish
```

`publishConfig` sets public access and the official npm registry. Publishing runs
the build, type checks, and SDK tests through `prepublishOnly`; `prepack` rebuilds
the distributable. Only `dist/`, this README, the license, and package metadata
are included. Complete any authentication challenge npm presents locally.

The first version is `0.1.0`. For later releases, run
`npm version patch --no-git-tag-version` (or `minor` / `major`) here, then repeat
the checks and publishing commands. Keep SDK versions independent of the Python
package. npm does not allow republishing an already-used name/version pair.

After publication, verify the release and install it into a clean project:

```sh
npm view @laya/typescript-sdk version --registry=https://registry.npmjs.org
npm install @laya/typescript-sdk
```
