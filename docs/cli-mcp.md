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
| `laya_route` | Selects a checkpoint and returns its model, repository, and reason without running a forward pass. | `state`, `questions` |
| `laya_predict` | Runs typed questions and returns answers, routing metadata, latency, and the answering device when readable. | `state`, `questions`, optional `model` (`auto`, `english`, `multilingual`, or `typed-decisions`) |
| `laya_shortlist` | Shortlists a many-option choice question, then answers it and returns the shortlist metadata. | `state`, `questions`, optional `model`, optional `k` (default `20`) |
| `laya_preset` | Runs a built-in workflow using its built-in question set. | `preset`, `state` |

The shared guardrail says not to send choice questions with more than 20 options without
shortlisting. `laya_shortlist` keeps the `k` most likely labels before the forward pass; its
default is `k=20`. It uses mean-pooled embeddings from the answering checkpoint's own encoder,
so it does not download a second model, and returns the kept labels, cosine scores, `k`, and
option count for each shortlisted question.

`state` must be a non-empty JSON object. `questions` must be a non-empty object whose values use
Laya's typed question schema. `laya_preset` accepts `guard`, `moderation`, `triage`, and
`model_router`; the last name is the MCP spelling of the router workflow. The CLI's preset
names are listed separately above because the two entry points expose different preset aliases.

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

The stock `laya-mcp-server` launcher creates its Router without installing hooks. If you need
prediction hooks, use a custom launcher that installs them, for example with
`laya.hooks.set_default_hooks`, before the server builds its Router. The environment variables
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
