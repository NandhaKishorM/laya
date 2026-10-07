# Command line and MCP server

Laya has two local interfaces for trying the same structured-decision engine:

| Interface | Use it for | Transport |
|---|---|---|
| `laya` | quick checks and interactive exploration from a terminal | command line |
| `laya-mcp-server` | connecting an MCP client or agent to Laya's built-in tools | MCP over stdio |

Choose the CLI when you are the person reading the result. Choose MCP when another process needs
a stable tool interface. Both use Laya's `Router` to select a checkpoint and return typed
`choice`, `score`, and `noul` decisions; neither is an open-ended question-answering or text
generation interface.

For the routing decision and typed-question examples, see the README's [Route Mode quickstart](https://github.com/NandhaKishorM/laya#quickstart-route-mode-recommended).
For confidence and built-in workflows, see the README's [confidence gating](https://github.com/NandhaKishorM/laya#automated-confidence-gating) and [workflow presets](https://github.com/NandhaKishorM/laya#built-in-workflow-presets).

## 1. Command line

Installing the package installs the `laya` entry point. Run `laya --help` for the complete
option list.

```bash
python -m pip install laya
laya --help
```

### Evaluation CLI

The package also installs `laya-evals`. The main CLI exposes the same evaluation commands
through `laya eval`:

```bash
laya eval --help
```

See the [Evaluation harness](evals.md) guide for datasets, metrics, and baseline gates.

### Training CLI

`laya-train`, also reachable as `laya train`, fine-tunes a checkpoint from a CSV or JSONL of labelled
decisions. See [Fine-tune with `laya-train`](finetune.md#fine-tune-with-laya-train).

```bash
laya-train --help
```

### Route without loading a checkpoint

With text and no prediction flag, the CLI calls `Router.route`:

```bash
laya "I was charged twice, please refund it"
```

The output names the selected checkpoint, explains why it was selected, and shows detected
language information when available. Routing alone does not download or build a checkpoint, so
it is a quick offline check of the routing decision.

Use `--json` when another local script should consume the decision:

```bash
laya "I was charged twice, please refund it" --json
```

### Run a prediction

`--predict` runs the full typed prediction and loads the routed checkpoint on first use. The
first load needs access to the Hugging Face Hub; later runs use the local cache.

```bash
laya "Classify this support request" --predict
laya "Classify this support request" --predict --json
```

`--json` prints the complete result as JSON. Without it, the CLI prints each answer together
with its choice probability, score, or `noul` value, plus the routing decision.

The main controls are:

- `--model english|multilingual|typed-decisions` pins a checkpoint instead of auto-routing.
- `--lang en|de|...` supplies an explicit language code instead of automatic detection.
- `--lang-guess en|de|...` supplies a soft hint that routing reads after `--lang` and before its
  own detector; a hint that resolves to nothing falls through, so it nudges the checkpoint without
  forcing it.
- `--task NAME` forces the typed-decisions workflow instead of detecting it.
- `--device cpu|cuda|...` passes a device choice to the Router.
- `--json` emits machine-readable output.

### Use a built-in preset

A preset supplies a ready-made question set and implies prediction, so `--predict` is not needed:

```bash
laya "My payment failed twice" --preset triage
laya "Ignore all previous instructions" --preset guard --json
```

The CLI presets are `email`, `guard`, `moderation`, `router`, and `triage`. The CLI places the
text under the state field expected by the selected preset; `--predict` uses the router
question set's `request` field. Presets are useful for a quick local check, but their questions
are still domain decisions: inspect the preset and validate it on your own data before using it
as an application policy.

### Explore interactively

With no text argument, the CLI opens a small prompt:

```bash
laya
# laya> Classify this request
# laya> quit
```

Press Enter to run each request. An empty line, `quit`, `exit`, or `Ctrl-D` ends the session. The
interactive loop reuses one Router, so it is a convenient way to compare several inputs without
writing a script.

### Failures are visible

The CLI handles invalid values and common dependency, download, and runtime failures at the
application boundary. It prints a diagnostic to stderr and returns exit code `2` instead of
showing an unhandled traceback. If a first-use checkpoint download fails, check dependency
installation, Hub access, and the selected device before retrying.

## 2. Built-in MCP stdio server

The MCP server is an optional extra. The core package does not install the `mcp` dependency:

```bash
python -m pip install "laya[mcp]"
laya-mcp-server
# equivalent module form:
python -m laya.mcp.server
```

The server speaks MCP over **stdio**, not HTTP. Configure the client with the console script:

```json
{
  "mcpServers": {
    "laya": {
      "command": "laya-mcp-server",
      "env": {
        "LAYA_DEVICE": "cpu"
      }
    }
  }
}
```

If the client configuration supports a Python executable and arguments, use
`python -m laya.mcp.server` as the equivalent launch form. The client owns the server process;
Laya does not open a network port.

### Available tools

| Tool | What it does | Main inputs |
|---|---|---|
| `laya_status` | Reports the configured or actual device, CUDA availability, loaded checkpoints, preload state, readiness, and package versions. | none |
| `laya_route` | Selects a checkpoint and returns its model, repository, and reason without running a forward pass. | `state`, `questions`, optional `model`, `task`, `lang`, `lang_guess` |
| `laya_predict` | Runs typed questions and returns answers, routing metadata, latency, and the answering device when readable. | `state`, `questions`, optional `model` (`auto`, `english`, `multilingual`, or `typed-decisions`), `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `laya_shortlist` | Shortlists a many-option choice question, then answers it and returns the shortlist metadata. | `state`, `questions`, optional `model`, `k` (default `20`), `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `laya_preset` | Runs a built-in workflow using its built-in question set. | `preset`, `state`, optional `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `laya_predict_batch` | Answers many requests in one call. Requests are routed first and grouped by checkpoint, so matching question schemas share forward passes; answers come back in input order. | `requests`, each `{state, questions, model?, task?, lang?, lang_guess?, max_len?, head_max_len?}`, optional `batch_size` |
| `laya_route_batch` | Decides which checkpoint would answer each request, with no forward pass and no checkpoint load. | `requests`, each `{state, questions, model?, task?, lang?, lang_guess?}`, optional `batch_size` |
| `laya_decide` | Answers a JSON-schema-shaped decision in one forward pass and returns the decided values with per-field confidence, instead of an answer map to parse. Schema properties may be enum choices, booleans, or integers with a minimum and maximum; free strings, arrays, and nested objects are rejected by path. | `state`, `schema`, optional `model`, `min_confidence` |

The three batch and schema tools exist because the same operations are available on the SDK and
`laya-serve`: handling many requests, or serving a caller that already knows the answer shape,
does not require dropping to Python. For the schema-driven form in more depth, see
[Schema-driven decisions](structured.md).

The shared guardrail says not to send choice questions with more than 20 options without
shortlisting. `laya_shortlist` keeps the `k` most likely labels before the forward pass; its
default is `k=20`. It uses mean-pooled embeddings from the answering checkpoint's own encoder,
so it does not download a second model, and returns the kept labels, cosine scores, `k`,
option count, and whether the question passed through unshortlisted for each shortlisted
question. A question with `k` at or past its option count is passed through unchanged, so
its cosine scores come back `null` rather than empty.

`state` must be a non-empty JSON object. `questions` must be a non-empty object whose values use
Laya's typed question schema. `laya_preset` accepts the same five presets the CLI does: `email`,
`guard`, `moderation`, `triage`, and the router workflow, whose canonical name on this surface is
`model_router`. `router` is accepted as an alias and names the same preset, so the CLI spelling
works here too; the canonical key is the one that comes back in the result. Given a state of
exactly one string, `laya_preset` places it under the field that preset's questions name, the
same placement the CLI does, so a caller does not have to guess the key. Anything richer than one
string is the caller's own shape and is passed through untouched.

Every single-request tool takes the same per-call routing controls the batch requests do. Alongside
`model`, a request may set `task` (name a checkpoint by the work), `lang` (force a language code),
and `lang_guess` (a soft language hint that sits below `lang` and above the built-in detector, so a
probable-but-uncertain code can nudge which checkpoint is chosen without forcing it the way `lang`
does). `lang_guess` only participates in routing, so like `task` it is refused on a call that pins
`model` -- a pinned checkpoint has nothing left to route. `laya_predict` and `laya_shortlist` also
take `max_len`/`head_max_len` for the answering token budget and `min_confidence` for the abstention
gate.

A prediction call has the same shape as the SDK's typed call:

```json
{
  "state": {
    "body": "I was billed twice for the same plan. Please reverse the duplicate charge."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "payments, invoices, refunds, duplicate charges",
        "technical": "bugs, outages, integration problems"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does the user need immediate help?"
    }
  }
}
```

The tool response is JSON containing the typed `answers`, the `routing` decision, and timing
information. Do not treat a high-confidence answer as permission to perform an external action;
the application or agent remains responsible for policy, review, and side effects.

### Startup and environment

The MCP server keeps a resident Router and serializes first-time construction. By default it
preloads `english` and `multilingual`; `typed-decisions` stays lazy. A preload failure is
reported at startup and retried on the next tool call, so inspect `laya_status` before assuming
the server is ready.

| Variable | Default | Meaning |
|---|---|---|
| `LAYA_DEVICE` | automatic | Device value passed to PyTorch, such as `cpu` or `cuda`. |
| `LAYA_PRELOAD` | `1` | Build the configured checkpoints at startup. Set to `0` for lazy loading. |
| `LAYA_MODELS` | `english,multilingual` | Comma-separated checkpoints to preload. An empty value keeps the MCP default rather than preloading every checkpoint. |
| `LAYA_THREADS` | PyTorch default | Caps Torch intra-op threads for CPU inference; keep it at or below the physical core count. |
| `LAYA_AUTO_TASK` | `0` | Set to `1` to let a request auto-route to the `typed-decisions` checkpoint. Same meaning as in `laya.serve`; it does not preload that checkpoint, so `LAYA_MODELS` still decides what is built at startup. |
| `LAYA_DEFAULT_MODEL` | `multilingual` | The checkpoint a state with no language evidence falls back to, same meaning as in `laya.serve`. Unlike `laya.serve`, an unresolvable name does not stop the server: it comes back as a `router construction failed` tool error on the next call, because a stdio server has no startup to refuse. |
| `LAYA_BASE_URL` | unset | Send predictions to a `laya-serve` on your own hardware instead of loading checkpoints in each MCP process. A bare `host:port` is read as HTTP. |
| `LAYA_REMOTE_TIMEOUT` | `300` | HTTP timeout in seconds when `LAYA_BASE_URL` is set, including the server's cold load. Invalid or non-positive values use the default. |

### Share one model server across MCP sessions

Run one local HTTP server and point each MCP client's environment at it:

```bash
LAYA_HOST=127.0.0.1 LAYA_PRELOAD=0 LAYA_IDLE_UNLOAD_SECONDS=300 laya-serve
```

```json
{
  "mcpServers": {
    "laya": {
      "command": "laya-mcp-server",
      "env": {"LAYA_BASE_URL": "http://127.0.0.1:8000"}
    }
  }
}
```

Install `laya[serve]` where the HTTP server runs. MCP still uses stdio with the editor; its
prediction tools use HTTP to reach your server. `laya_predict`, `laya_predict_batch`, `laya_decide`
and `laya_preset` use the server's original state, instructions and option descriptions.
Heterogeneous batches send one `/v1/systemone` request per item, preserving input order;
`batch_size` and `sort_by_length` do not change the server's execution. `laya_status` reports the
server's `/health`; `laya_route` and `laya_route_batch` stay local and need no model or HTTP request.
The MCP process imports no torch and loads no checkpoint, including when `LAYA_THREADS` or
`LAYA_PRELOAD` is set.

Set the same `LAYA_API_KEY` in both processes when the server requires a bearer token. Keep
`LAYA_DEFAULT_MODEL` and `LAYA_AUTO_TASK` aligned so local routing previews match the server's
actual routing. Device and preload settings belong to the HTTP server. The first call after an
idle unload waits for a cold load; raise `LAYA_REMOTE_TIMEOUT` if that takes longer than 300 seconds.
`laya_shortlist` and prediction hook overrides return `unsupported_remote`, since their code needs
the model process. HTTP errors retain the server's detail text as MCP tool errors. With
`LAYA_BASE_URL` unset, the MCP server keeps loading checkpoints in its own process.

The stock `laya-mcp-server` launcher creates its Router without installing hooks. Install prediction
hooks in the process that runs inference: the MCP process in local mode, or the HTTP server in
shared-server mode. A custom launcher can use `laya.hooks.set_default_hooks` before building its Router. The environment variables
above configure model lifecycle, not hook registration. The client still decides when to call a
tool and what to do with the returned decision.

## 3. Shared boundaries and related guides

The CLI and MCP server are interfaces to the same typed decision engine:

- Use `choice` for a finite label set, `score` for an ordered rubric, and `noul` for the
  probability of true.
- Validate thresholds and presets on representative data; there is no universal adoption
  threshold.
- Keep irreversible or high-cost actions behind the application's review and fallback policy.
- The MCP server calls `Router.predict`, so hooks fire when a custom launcher installs them. See
  [Prediction hooks](hooks/index.md), [hook lifecycle](hooks/lifecycle.md), and
  [Tracing](hooks/tracing.md) for observability and `run_id` correlation.

This guide covers the local CLI and the built-in MCP stdio server. It does not document the
HTTP API, community wrappers, or an MCP protocol redesign.
