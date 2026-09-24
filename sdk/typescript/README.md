# @laya/typescript-sdk

A dependency-free HTTP client for Laya's Jev-compatible `/v1/systemone` endpoint. Supports ESM and
CommonJS on Node.js 22+, and browsers with `fetch`, `AbortController`, and
`structuredClone`. TypeScript consumers need TypeScript 5 or newer.

Inference runs in Laya's existing `laya.serve` server or the Jev API. JavaScript clients do not need
Python, PyTorch, or model weights installed; the server does. This package does
not run models directly in a browser or Node.js.

## Run from this repository

In a terminal at the repository root, start the server:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[serve]'
LAYA_HOST=127.0.0.1 LAYA_MODELS=english laya-serve
```

In a second terminal, build the SDK and run the plain JavaScript example:

```sh
cd sdk/typescript
npm ci
npm run build
node examples/triage.mjs
```

The server preloads checkpoints at startup by default. The command above loads
the English checkpoint; the router can load others when needed. To preload both
English and multilingual on CPU:

```sh
LAYA_HOST=127.0.0.1 LAYA_MODELS=english,multilingual LAYA_DEVICE=cpu laya-serve
# Set LAYA_DEVICE=cuda or mps for a suitable GPU installation.
```

Set `LAYA_PRELOAD=0` to defer loading until the first prediction. Initial loads
download weights and need sufficient memory and disk space. The client timeout
defaults to 120 seconds; the example allows 10 minutes for a cold load. Server
configuration uses environment variables, not command-line flags. See the
[server guide](../../README.md#self-hosting-http-server-jev-compatible).

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
console.log(result.routing?.model); // Laya-only metadata
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
result.answers.department.action?.act_probability; // Laya-only metadata
result.usage.input_tokens;
```

Choice criteria also accept an array of unique labels, serialized as an option-to-null
map for the shared protocol. Structured JSON values are
supported in instructions and descriptions, as in Python. State may be text, a
JSON object, an array (including chat turns), or null. Undefined values, dates,
non-finite numbers, BigInts and cycles are rejected locally instead of silently
changing during serialization. Empty questions or criteria are rejected.

Response fields retain Python's spelling (`input_tokens`, `act_probability`,
etc.). Scores use zero-based rubric levels. Confidence and accuracy have the
same calibration limits as the Python model; see the root README.

## Jev compatibility and model selection

The same `predict` method calls [TypeSafe's System One API](https://docs.typesafe.ai/api):

```js
const jev = new Laya({
  baseURL: 'https://api.typesafe.ai', // server root; do not append /v1
  apiKey: process.env.TYPESAFE_API_KEY,
});
const result = await jev.predict('Please refund the duplicate charge.', triageQuestions());
console.log(result.answers.refund_requested.noul);
```

The default request model is `jev-latest`. Jev uses that model alias; Laya ignores
unrecognized model IDs and chooses a local checkpoint automatically. This lets
the same prediction code switch servers through `baseURL` and credentials.
Set a client-wide `model` or override it per prediction to select a checkpoint:

```js
const laya = new Laya({ model: 'english' });
await laya.predict(state, questions, { model: 'typed-decisions' });
await laya.predict(state, questions, { model: 'multilingual' });
await jev.predict(state, questions, { model: 'jev-preview' });
```

Laya accepts its checkpoint aliases and published multilingual/typed-decisions
model IDs. Use a Jev model ID when calling Jev. The HTTP endpoint supports
`model`; it does not expose standalone `route()`, `task`, or `lang` overrides.
Laya still detects language automatically; set `LAYA_AUTO_TASK=1` on the server
to enable automatic workflow routing.

Both servers return `model`, `answers`, and `usage`. Laya additionally returns
`routing`, answer `action`, and Noul `confidence`. Those extensions are optional
in the SDK types and validated when present. Choice and Score confidence remains
required. `health()` is a Laya-only probe and is never called by `predict()`.

Jev compatibility is covered by fixtures based on its official API contract;
the automated suite does not call the hosted service. Jev also enforces its own
input limits (up to 255 Choice options and 2–10 Score levels). Use text/object/array
state for portability; null state and broader JSON scalar descriptions are Laya
extensions accepted by this SDK.

## Presets

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

Laya's server binds to `0.0.0.0` by default; the quickstart sets `LAYA_HOST=127.0.0.1`
for local use. Set `LAYA_API_KEY` in its environment to require Bearer
authentication for predictions. `/health` remains public. `HF_TOKEN`, if needed
for checkpoint downloads, is a separate server-only credential. Configure CORS
and TLS at your application proxy or gateway.

## Python service contract

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` (Laya only) | None | `{status, loaded, device}` |
| `POST /v1/systemone` | `{state, questions, model}` | `{model, answers, usage}` plus optional Laya `routing` |

Laya errors use FastAPI's `{detail: ...}` body: 400 for malformed requests, 401
for authentication, 413 for request limits, 422 for invalid questions, and 500
for inference failures. The SDK preserves the status and details in
`LayaAPIError`; string details become the error message. It also accepts
`{error: {code, message, details?}}` from compatible services. OpenAPI is
available at `/openapi.json` and interactive request documentation at `/docs`.

For custom checkpoint paths or an existing agent, configure Python's router:

```python
import uvicorn
from laya import Router
from laya.serve import create_app

router = Router(models={"english": "/path/to/checkpoint"}, device="cpu")
router.preload(["english"])
app = create_app(router)  # Reads LAYA_API_KEY from the environment.
uvicorn.run(app, host="127.0.0.1", port=8000)
```

This SDK uses the repository's existing server without changing its Python
implementation. The integration fixture injects a router into that same server.

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
