<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NandhaKishorM/laya/main/assets/logo-lockup-dark.png" />
    <img src="https://raw.githubusercontent.com/NandhaKishorM/laya/main/assets/logo-lockup.png" alt="Laya" width="330" />
  </picture>
</p>

**Multilingual, non-autoregressive System 1 decision engine.** Typed decisions over 100+ languages in a single forward pass — 33 ms — trained with reinforcement learning against strictly proper scoring rules (RLCD), with a router that picks the right checkpoint per request.

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15d4Yv__KHeHjshVb-6PRTfqVllxih2S3?usp=sharing)
[![PyPI version](https://img.shields.io/pypi/v/laya.svg)](https://pypi.org/project/laya/)
[![Docs](https://img.shields.io/badge/docs-online-2ea44f)](https://nandhakishorm.github.io/laya/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-convaiinnovations%2Flaya-blue)](https://huggingface.co/convaiinnovations/laya)
[![Multilingual](https://img.shields.io/badge/%F0%9F%A4%97%20Model-laya--multilingual-blue)](https://huggingface.co/convaiinnovations/laya-multilingual)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-laya--demo-orange)](https://huggingface.co/spaces/convaiinnovations/laya-demo)
[![Dev.to Article](https://img.shields.io/badge/dev.to-Read%20Article-0A0A0A?logo=devdotto&logoColor=white)](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-nandakishorm-FFDD00?logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/nandakishorm)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## Installation

```bash
python -m pip install laya
```

With [uv](https://docs.astral.sh/uv/), run `uv add laya` in a uv project or `uv pip install laya` in a virtual environment.

Python 3.10 or newer. Optional extras: `laya[serve]` (HTTP server), `laya[mcp]` (MCP server), `laya[langchain]` (LangChain and LangGraph), `laya[llamaindex]` (LlamaIndex selectors), `laya[crewai]` (CrewAI routing), `laya[onnx]` (ONNX Runtime), `laya[fast]` (TileLang GPU fast path), `laya[structured]` (pydantic models in `decide`). Step-by-step setup for each platform, CPU-only or GPU PyTorch builds, and troubleshooting are in [Installation details](#installation-details).

For TypeScript / Node.js / browser, see [`laya-ts/`](https://github.com/NandhaKishorM/laya/tree/main/laya-ts/). npm releases (`npm install laya-ts`) are published from this repository's `laya-ts-v*` release tags.

**Long documents.** `laya-multilingual` reads up to 8,192 tokens with `max_len=8192`. Measured accuracy and time by document length, reproducible with [`research/scripts/bench_long_context.py`](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/bench_long_context.py):

<p align="center">
  <img src="https://raw.githubusercontent.com/NandhaKishorM/laya/main/assets/long_context_8192.png" alt="laya-multilingual with max_len=8192: 16 to 18 of 20 requests correct with up to about 4,000 tokens of text before them, more variable beyond" width="100%" />
</p>

## Quickstart

> **Long documents: `laya-multilingual` reads up to 8,192 tokens.** It ships with a 1,024-token limit that cuts long documents off, so pass `max_len=8192` for them:
>
> ```python
> result = router.predict(long_document, questions, model="multilingual", max_len=8192)
> ```
>
> In the table above, 16 to 18 of 20 requests were answered correctly with up to about 4,000 tokens of text before them; beyond that results vary (8 to 17 of 20), so check long-document accuracy on your own data. Short inputs give identical answers with `max_len=8192`, and speed follows the input's real length, not the limit: short inputs are unchanged, and a 4,000-token input takes about 1.7 s on an Apple GPU. Name the checkpoint with `model="multilingual"`, since long mostly-English text would otherwise route to the English checkpoint.

```python
from laya import Router

router = Router()  # downloads a checkpoint on first use; Router(preload=True) loads all three up front

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}

result = router.predict(state, questions)
print(result["answers"]["department"]["choice"])  # billing
print(result["answers"]["churn_risk"]["noul"])    # probability the answer is yes
print(result["routing"]["model"])                 # english
```

The same call works in any of 100+ languages. The `Router` detects the script and language and sends non-English text to `laya-multilingual`:

```python
for text in ["मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।",
             "La aplicación se cierra cada vez que abro la configuración."]:
    r = router.predict(text, {"department": questions["department"]})
    print(r["routing"]["model"], r["answers"]["department"]["choice"])
# multilingual billing
# multilingual technical
```

From the command line, `laya "My payment failed twice" --preset triage` answers a ready-made question set. More in the [full quickstart](#quickstart-route-mode-recommended) and the [docs](https://nandhakishorm.github.io/laya/).

## Fine-tune for better accuracy

The shipped checkpoints work zero-shot, but fine-tuning on decisions from your own domain is where accuracy jumps. On the typed-decisions benchmark (2,000 decisions across four workflows), the fine-tuned `laya-typed-decisions` checkpoint scores **0.766** accuracy, against **0.362** for the base English checkpoint on the same decisions.

**[Fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)** (Kaggle 2xT4) and **[Apple Silicon script](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_mps.py)** (MPS / CPU): run the whole loop (build dataset, train with RLCD, calibrate temperatures, evaluate, export). Details in [Fine-Tuning](#fine-tuning).

## Documentation

**[nandhakishorm.github.io/laya](https://nandhakishorm.github.io/laya/)**: guides for [prediction hooks](https://nandhakishorm.github.io/laya/hooks/), [schema-driven decisions](https://nandhakishorm.github.io/laya/structured/), [Docker](https://nandhakishorm.github.io/laya/docker/) and [LangChain and LangGraph](https://nandhakishorm.github.io/laya/langchain/), plus a full [API reference](https://nandhakishorm.github.io/laya/reference/).

## What's new in 0.4.2

* **The TypeScript tree parses again.** An unterminated template literal in `hooks.ts` shipped
  in 0.4.1 and stopped 22 of 34 test files from loading (#1056).
* **Jev confidence alongside ours.** `/v1/systemone` adds `x_jev_confidence`, computed locally
  from the same probabilities, so a threshold carried over from Jev can be compared directly
  (#1066, #302). `confidence` is unchanged and the field is absent under `LAYA_JEV_STRICT`.
* **Shortlisting keeps working with an `option_order`.** A narrowed choice drops the order with
  a warning instead of failing the agent's permutation check (#1063).
* **A near-miss checkpoint name warns.** Registering `englsh` tells you it is one edit from
  `english`, and registers it anyway (#1055, #919).
* **Compose forwards four more knobs** that the runtime already read (#1053, #1054), and the
  Java and .NET docs now name the multilingual default 0.4.0 moved them to (#1069).
* **Java** gains `Router.predictBatch` and `Shortlist.predictTournament` (#1060, #1061).

Earlier releases are in the [GitHub releases](https://github.com/NandhaKishorM/laya/releases).

---


<p align="center">
  <img src="https://raw.githubusercontent.com/NandhaKishorM/laya/main/assets/laya_vs_jev_full.png" alt="Laya versus TypeSafe Jev: accuracy on shared public datasets, every application workflow, all 51 languages, speed, calibration, and the cost of not preloading" width="100%" />
</p>

Laya evaluates typed questions (`choice`, `score`, `noul`) over any state (text, email, ticket or JSON document) in **a single forward pass** — 33 ms for one question, 7.2 ms/question batched, measured on a T4. No text generation, so nothing to parse and nothing to hallucinate.

Three checkpoints, and a `Router` that picks between them per request:

| | encoder | params | context | use it for |
|---|---|---|---|---|
| [`laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 421M | 512 | English |
| [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 322M | 1024 (up to 8,192) | 100+ languages, 2x faster |
| [`laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |

## Where to go next

| document | what is in it |
|---|---|
| Quickstart below | the shortest path to a working call |
| [`examples/README.md`](https://github.com/NandhaKishorM/laya/blob/main/examples/README.md) | 41 runnable examples as an eight-stage learning path: first call → schema design → production service |
| [`LOCAL_SETUP.md`](https://github.com/NandhaKishorM/laya/blob/main/LOCAL_SETUP.md) | running this checkout on macOS/Intel: pinned stack, checkpoint manifest, the two portability fixes, every verification run and its result |
| [`BENCHMARKS.md`](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) | every benchmark run consolidated, all 51 languages, with the limits stated plainly |
| [`research/README.md`](https://github.com/NandhaKishorM/laya/blob/main/research/README.md) | the harnesses that produce those numbers, and how to re-run them |


## Installation details

Python 3.10 or newer. The dependencies set that floor: `huggingface_hub` 1.x, `transformers` 5.x and `torch` 2.14 all require 3.10.

**Optional PyTorch build selection:** If you need a CPU-only or GPU-specific PyTorch build, follow [PyTorch's installation guide](https://pytorch.org/get-started/locally/) after creating your virtual environment and before installing Laya. Replace `pip` or `pip3` in the selected command with the environment's Python executable followed by `-m pip`.

If you already use a virtual environment, install the PyPI release with:

```bash
python -m pip install laya
```

For a new environment, choose the commands for your platform below. Run them from your project directory; the explicit Python paths keep installation and verification in the same environment.

**macOS / Linux** (with Python 3.10 or newer):

On Debian/Ubuntu, the system Python may require `sudo apt install python3-venv` before creating a virtual environment. If `venv` reports that `ensurepip` is unavailable, install that package and retry.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install laya
.venv/bin/python -I -c "import laya; print(laya.__version__)"
```

**Windows PowerShell** (this example uses an installed Python 3.11):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install laya
.\.venv\Scripts\python.exe -I -c "import laya; print(laya.__version__)"
```

Both checks print the installed Laya version without loading a checkpoint. `-I` excludes the current directory from the import search path, so a local source copy cannot mask a missing installation. Keep using the same virtual environment's Python when running your application.

**Intel GPU (XPU)**

Install a supported Intel GPU driver first. For an XPU-enabled PyTorch build, install its wheel before Laya; the default PyPI wheel may be CPU-only. PyTorch's validated hardware and OS list is in the [Intel GPU guide](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html).

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/xpu
.\.venv\Scripts\python.exe -m pip install laya
.\.venv\Scripts\python.exe -c "import torch; print(torch.xpu.is_available())"
```

For a source checkout, replace `pip install laya` with `pip install -e .`. Laya automatically selects an available XPU when no device is specified; you can also request one explicitly with `device="xpu"` in `laya.load()` or `Router(device="xpu")`.

**Install from GitHub**

To use the development version instead of the PyPI release, create the virtual environment above and replace its installation command with the appropriate command below. Git must be installed.

```bash
# macOS / Linux
.venv/bin/python -m pip install "git+https://github.com/NandhaKishorM/laya.git"
```

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m pip install "git+https://github.com/NandhaKishorM/laya.git"
```

Run the same version check afterward. The GitHub version follows the repository's default branch and may differ from the published release.

**Install with uv**

[uv](https://docs.astral.sh/uv/) creates the virtual environment, downloads a matching Python if none is installed, and installs into it. The commands are the same on macOS, Linux and Windows PowerShell:

```bash
uv venv --python 3.12
uv pip install laya
```

`uv pip install` targets the `.venv` in the current directory without activating it, so run the version check for your platform above afterward. Extras and the GitHub version install the same way: `uv pip install "laya[serve]"`, `uv pip install "git+https://github.com/NandhaKishorM/laya.git"`. For a CPU-only or GPU-specific PyTorch build, add `--torch-backend=auto` to pick the build that matches the machine's GPU driver, or name one such as `--torch-backend=cpu`; [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) has the list.

If your application is a uv project, add Laya as a dependency instead:

```bash
uv add laya
uv run python -I -c "import laya; print(laya.__version__)"
```

`--torch-backend` applies to `uv pip` only; in a uv project, uv's PyTorch guide shows how to set the PyTorch index in `pyproject.toml`.

**Model setup and troubleshooting**

Continue with the [Router quickstart](#quickstart-route-mode-recommended) to run inference. Loading a Hub checkpoint requires access to Hugging Face on its first download; the quickstart's `Router(preload=True)` loads all three configured checkpoints at construction.

- **`ModuleNotFoundError: No module named 'laya'`:** run both installation and your script with the same virtual environment's Python executable shown above. In an editor, select that interpreter as well.
- **Missing `rl_agent_config.json`:** this file ships with a Laya checkpoint alongside `model.safetensors`; it is not a configuration file you need to create in the source repository. For a local model, pass the directory containing those checkpoint files.
- **Checkpoint downloads and cache:** `Router()` is lazy: the first prediction downloads only the
  selected checkpoint, then reuses the Hugging Face cache. Set `HF_HUB_CACHE` to move that cache.
  The English and typed-decisions checkpoints have about 421M parameters each; multilingual has
  about 322M (see the table above). Download sizes also depend on weight precision and files.
- **Memory limits:** `Router()` keeps up to two checkpoints resident by default (`max_loaded=2`),
  evicting the least recently used when needed. Use `Router(max_loaded=1)` to reduce resident
  memory, at the cost of reloads when switching checkpoints. `Router(preload=True)` loads all
  three up front and raises the resident limit to fit them; it is not an out-of-memory remedy.
- **Windows + Python 3.14:** the model-construction crash reported in
  [#123](https://github.com/NandhaKishorM/laya/issues/123) was fixed in Laya 0.3.7 by
  [#195](https://github.com/NandhaKishorM/laya/pull/195) and verified on the reporter's Windows 11
  setup without a manual patch. Upgrade older Laya installations with your environment's
  Python executable followed by `-m pip install -U laya`.

---

### Command line

Installing the package also installs a `laya` command for quick local testing, no script needed:

```bash
laya "I was charged twice, please refund"            # routing decision only; works offline, no download
laya "Refactor this service" --predict               # full answers (downloads the checkpoint on first use)
laya "Mein Konto wurde zweimal belastet" --lang de   # force a language instead of detecting it
laya "My payment failed twice" --lang-guess en        # a soft hint: nudge routing, still fall through to detection
laya "My payment failed twice" --model ml            # pin a checkpoint: names, aliases and casing all resolve as the SDK resolves them
laya "My payment failed twice" --preset triage       # answer a ready-made preset (triage, email, guard, moderation, router)
laya --batch tickets.txt --predict                   # score a file of requests, one per line, in one batch
cat tickets.txt | laya --batch - --predict --json    # stdin; one JSON line of answers per request
laya "Where is my card" --questions intents.json     # answer your own questions, written in a JSON file
laya "Refund my card" --predict --min-confidence 0.9 # mark an answer the model is unsure of
laya                                                 # interactive mode
```

`--lang` is decisive: a real code picks the checkpoint and skips detection. `--lang-guess` is the
soft sibling core checks in between, so a probable-but-uncertain language can nudge routing without
forcing it — and a hint that resolves to nothing falls through to the built-in detector, exactly the
`Router(lang_guess=...)` behaviour the library offers. Neither flag changes the routing decision's
precedence rules; `--lang-guess` just adds a hint those rules read before detection.

Routing alone never downloads a checkpoint, so it returns in milliseconds. `--predict` loads the routed checkpoint, which needs network access to the Hugging Face hub the first time; if a checkpoint cannot be downloaded, the CLI says so instead of crashing. `--batch` (with or without `--predict`) sends the whole file through `Router.predict_batch` in one process, so the requests share checkpoint loads and forward passes — measured 2.6x on 20 tickets vs looping `predict` one by one, with `--batch-size N` to bound the forward pass, `--sort-by-length` to group similarly sized requests inside it, and `--json` for JSONL output. Batch routing (`laya --batch FILE`, no `--predict`) likewise answers with `route_batch` in one pass, still without loading anything.

`--sort-by-length` reaches the length grouping `Agent.predict_batch` has done since #294, which
until now only the library call in [Batch Mode](#batch-mode-score-many-states-in-one-forward-pass)
could ask for. A forward pass pads every state in it to the longest one, so a two-line request
sharing a pass with a two-paragraph one spends most of its compute on padding; grouping by length
first puts comparable sizes together. Measured on 128 real Yelp reviews of 69-2,293 characters,
run as `laya --batch FILE --predict --model laya --batch-size 8` with and without the flag, four
interleaved rounds on an Apple-silicon Mac at the CLI's own device pick: 41.25 s median unsorted
against 29.23 s sorted, a **paired median of 1.42x** (1.39-1.44x across the four pairs), with all
128 decisions identical and still in input order. `research/` measures the same knob at 2.15x over
10,000 synthetic multilingual tickets. It needs `--batch-size N` with 1 < N < the number of
requests — a single pass over the whole file has no group to reorder — and the CLI says so on
stderr rather than printing a same-speed result and leaving you to notice.

`--questions` takes the same question dict the SDK takes, as JSON: either the mapping itself, or
`{"state_key": "body", "questions": {...}}` when the question's instructions name a field other than
`request`. It implies `--predict`, and a question set with many labels usually wants
`--head-max-len` with it: on 58 MASSIVE-INTENT labels written as one choice question, the English
checkpoint goes from 24/58 correct at its default 192-token option budget to 34/58 at
`--head-max-len 384`, for about 1.4x the per-request time on CPU. Widening it further costs the
accuracy back, because `max_len` then leaves fewer tokens for the request itself. [Honest
limits](#honest-limits) describes the same budget ceiling for a 77-option question.

Device selection is automatic, in this order: **CUDA → MPS → XPU → CPU**. Mixed precision is used on
CUDA; CPU and MPS run fp32. Override with `device=`, which is accepted by both entry points:

```python
agent = laya.load("convaiinnovations/laya", device="cpu")      # or "cuda", "mps", "xpu"
router = Router(preload=True, device="mps")
```

### macOS, including Intel Macs

PyTorch stops publishing macOS **x86_64** wheels at 2.2.2 (2.3 and later are Apple-silicon only),
while transformers 5.x requires torch >= 2.4 and torch 2.2.2 was built against NumPy 1.x. A plain
`pip install laya` on an Intel Mac therefore resolves a stack that cannot run — torch 2.2.2 with
transformers 5.x and NumPy 2.x, and torch fails to initialise NumPy 2, which `predict()` needs for
its final `.numpy()` conversion. Pin the stack:

```bash
pip install "numpy<2" "torch==2.2.2" "transformers==4.57.6" laya
```

`transformers` 4.57.x is the last 4.x line and the newest one that runs on torch 2.2.2. The
transformers-5 checkpoints load on it unchanged: `build_model` maps their `rope_parameters` onto
the RoPE attributes 4.x reads, which mmBERT needs because both of its bases are 160000 rather
than the 4.x defaults. Apple-silicon and CUDA machines need none of this pinning.

---

## Running from this checkout

`setup_laya.sh` stands the whole thing up in the checkout: an isolated `.venv` with the pinned
stack, the three checkpoints under `models/` (gitignored, ~2.3 GB), and a verification run. It is
idempotent, so re-running only fills in what is missing. The numbers below are from the machine it
was developed on — an Intel Mac Pro, no CUDA, AMD Radeon GPUs reachable through Metal (MPS); the
script itself is not Intel-specific. Commands are relative to the repository root.

```bash
./setup_laya.sh                                    # venv + pinned deps + checkpoints + verify
.venv/bin/python verify/laya_smoke_test.py --models ./models            # real weights, device auto -> MPS
.venv/bin/python verify/laya_smoke_test.py --models ./models --device cpu
.venv/bin/python verify/numerics_check.py          # RoPE bases, determinism, SDPA vs eager
.venv/bin/python verify/bench_devices.py           # CPU vs MPS, thread scaling
.venv/bin/python verify/edge_sweep.py              # edge cases: option budgets, truncation, router lifecycle
.venv/bin/python verify/checkpoints.py             # weights vs the recorded sha256 hashes
.venv/bin/python verify/soak_check.py              # drift, RSS and concurrency over 200 calls
.venv/bin/python tests/test_local_e2e.py ./models             # the repo's own e2e suite
./examples/run_all.sh                              # all 41 examples, pass/fail
```

Measured here, one `predict()` call answering 4 questions about a short email (median of 10,
after warm-up):

| checkpoint | CPU (28 threads) | MPS (Radeon Pro Vega II) |
|---|---|---|
| `laya` (421M) | 453 ms — 113 ms/question | **146 ms — 37 ms/question** |
| `laya-multilingual` (322M) | 192 ms — 48 ms/question | **121 ms — 30 ms/question** |
| `laya-typed-decisions` (421M) | 441 ms — 110 ms/question | **145 ms — 36 ms/question** |

MPS is 1.6-3x this 56-core CPU and matches the T4 reference quoted below (32.8 ms/question for
`laya-multilingual`). CPU-only work runs fastest around 16 threads (`OMP_NUM_THREADS=16`); the
default 28 already over-subscribes. The first MPS call pays ~13 s of Metal kernel compilation,
so warm up before timing. Individual runs move by a few percent with machine load.

To keep the router fully offline, point it at the local checkpoints rather than the hub repos:

```python
from laya import Router

router = Router(models={"english": "models/laya",
                        "multilingual": "models/laya-multilingual",
                        "typed-decisions": "models/laya-typed-decisions"},
                preload=True)
```

Full notes — version rationale, the two portability fixes this needed, and every verification
run — are in [`LOCAL_SETUP.md`](https://github.com/NandhaKishorM/laya/blob/main/LOCAL_SETUP.md).

---

## Try it locally: web GUI + JSON API

`examples/server.py` is a self-contained FastAPI app for testing Laya without writing any code:
a two-pane playground (edit the request as a form or as JSON, run it with Ctrl+Enter, read each
answer's full distribution and calibrated confidence, copy it as curl or Python), plus a plain
JSON API (`/predict`, `/predict/batch`) for scripting against.

```bash
pip install "laya[serve]"
python examples/server.py               # http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000` in a browser for the playground, or hit it directly:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "state": {"body": "We were billed twice for March. Please refund it today."},
  "questions": {
    "department": {"type": "choice",
                   "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds", "other": "everything else"}},
    "urgency": {"type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "critical"]}
  }
}' | python -m json.tool
```

`--no-preload` loads checkpoints lazily instead of all three up front; `--device cuda|cpu|mps`
pins the device. See `python examples/server.py --help` for the rest.

A request body may carry `model` to pin a checkpoint instead of letting the router choose, and it
takes exactly the spellings `laya --model` takes: names, aliases and casing all resolve through
`laya.router`. `GET /models` lists the checkpoints and the aliases alongside them.

Both `/predict` and `/predict/batch` also accept the four per-call controls `laya-serve`
forwards on `/v1/systemone`: `lang_guess` (a soft ISO hint that participates in routing, where
`lang` skips detection entirely), `max_len` and `head_max_len` (per-call token-budget overrides,
capped by `LAYA_MAX_TOKEN_BUDGET`), and `min_confidence` (core's abstention threshold; the answer
comes back with `abstention` and `abstention_threshold` fields naming whether it fired). An absent key means "inherit what the
Router was built with", so a caller never has to send them back as `null`; a body that names
`hooks`, `hooks_raise`, `hooks_timeout`, `on_predict_start`, or `on_predict_end` is refused with
422 rather than silently dropped, matching what `laya-serve` does for the same fields. On
`/predict/batch`, `min_confidence` reaches `Router.predict_batch` as a call argument rather than
a per-request key; the other three travel inside each request dict, so states in one batch may
name different budgets.

`/predict/batch` accepts two optional body fields that control the shape of the forward passes
without changing any answer: `batch_size` (states per pass; omit it and the whole batch is one
pass) and `sort_by_length` (group similarly sized states so each pass pads to a shorter maximum —
it needs a `batch_size` below the number of states, since one pass has nothing to reorder).
Measured on the running server: 64 real reviews of 91–2,293 characters, one choice question,
`batch_size: 8`, three interleaved rounds on MPS — the default one-pass shape 5.35s median against
3.04s sorted, a paired median of **1.77x** (min 1.76x), and 64/64 labels identical in input order.
`batch_size: 8` on its own was 4.95s, so the length grouping is what earns the second factor:
1.66x over the sized batch.

### JavaScript / TypeScript

[`laya-client`](https://github.com/NandhaKishorM/laya/blob/main/sdk/typescript/README.md) supports JavaScript and TypeScript
applications over HTTP to a self-hosted `laya-serve` `/v1/systemone` server. It
includes inferred answer types, all five question presets, ESM/CommonJS exports,
and cancellation.

```bash
# From this checkout, install and start the Python inference service:
pip install -e '.[serve]'
LAYA_HOST=127.0.0.1 LAYA_MODELS=english laya-serve

# In another terminal:
cd sdk/typescript
npm ci
npm run build
node examples/triage.mjs
```

The npm package is named `laya-client`. Use it when a JavaScript or TypeScript
application talks over HTTP to self-hosted Python `laya-serve`. Use `laya-ts`
when inference must run directly inside JavaScript through its local ONNX
runtime, without a Python server.
See [installation, examples, and publishing](https://github.com/NandhaKishorM/laya/blob/main/sdk/typescript/README.md) and the
[repository analysis](https://github.com/NandhaKishorM/laya/blob/main/docs/typescript-sdk.md) for architecture and scope.

---

## Quickstart: Route Mode (Recommended)

To try the Python SDK in a CPU container, see the
[Docker Compose quickstart](https://github.com/NandhaKishorM/laya/blob/main/docs/docker.md). It runs a sample request and keeps
downloaded models between runs.

Laya ships three checkpoints. The built-in **`Router`** is the recommended entry point: it evaluates any state in any language, automatically detects scripts and languages in sub-milliseconds, and dispatches to the optimal checkpoint in a single forward pass.

### Minimal example (30 seconds)

```python
from laya import Router

router = Router()  # downloads only the selected checkpoint on first use
result = router.predict(
    {"body": "We were billed twice. Please refund the duplicate."},
    {"billing": {"type": "noul", "instructions": "Does the user request a refund?"}},
)
print(result["answers"]["billing"]["noul"])  # P(true), a float from 0.0 to 1.0
print(result["routing"]["model"])            # which checkpoint answered, e.g. 'english'
```

Input is any state (`str`, `dict`, or `list` — text, email, ticket, JSON document) plus a dict of typed
questions (`choice` = pick one label, `score` = ordinal levels, `noul` = yes/no statement check).
Each `result["answers"][qid]` is an answer dict: `choice` is the selected label, `score` is the
expected ordinal level, and `noul` is P(true). `confidence` is a separate field in that dict;
`probabilities` is available for `choice` and `score`. `result["routing"]` explains which checkpoint was
picked and why. Values depend on the checkpoint and input. For ready-made question sets see
`laya.presets` (`triage_questions`, `moderation_questions`, `email_questions`, ...).

Full walkthrough:

```python
from laya import Router

# Preload checkpoints into memory for instant sub-35ms routing
router = Router(preload=True)

# 1. State in any language or schema
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

# 2. Define your typed questions
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts, plan upgrades",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

# 3. English state -> automatically routed to laya (ModernBERT-large, 39.5 ms)
res_en = router.predict(state, questions)
print("Department :", res_en["answers"]["department"]["choice"])  # -> billing (confidence: 0.94)
print("Routing    :", res_en["routing"]["model"])                 # -> english

# 4. Hindi state -> automatically routed to laya-multilingual (mmBERT-base, 32.8 ms)
res_hi = router.predict({"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, questions)
print("Department :", res_hi["answers"]["department"]["choice"])  # -> billing (confidence: 0.86)
print("Routing    :", res_hi["routing"]["model"])                 # -> multilingual

# 5. Explicit override when you want a specific checkpoint
res_td = router.predict(state, questions, model="typed-decisions")
```

Every result carries full routing metadata explaining why the choice was made:

```python
res_hi["routing"]
# {
#   'model': 'multilingual',
#   'repo': 'convaiinnovations/laya/multilingual',
#   'reason': 'non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it',
#   'detection': {'script': 'devanagari', 'script_profile': {'devanagari': 1.0},
#                 'language': None, 'is_english': False, 'language_undecided': True,
#                 'diacritic_rate': 0.0, 'non_latin_fraction': 1.0, 'mixed_segment': None},
#   'workflow': None
# }
```

All five keys are always present. `detection` is the
[`laya.lang.analyse`](#supplying-your-own-language-detection) result the decision was made from,
and is `None` whenever routing decided before detection ran: `model=`, `task=`, a detected
workflow, a decisive `lang=` and a resolving `lang_guess` all take precedence over it and
short-circuit it. `workflow` names the typed-decisions workflow whose question ids matched, and is
`None` when none did.

Inspect a routing decision without running any forward pass:

```python
router.route({"body": "Der Kunde wurde zweimal belastet"}, questions).reason
# "Latin script but language looks like 'de', not English"
```

Very short Latin-script text often carries nothing that identifies its language (`"Quero cancelar"`, `"Esqueci minha senha"`). Such text goes to `default`, which is `"multilingual"` since 0.4.0. Text the detector *can* place is unaffected: identified English still goes to the English checkpoint.

```python
router = Router()                                                   # default="multilingual"
router.route({"body": "Esqueci minha senha"}).model                 # -> multilingual
router.route({"body": "Please refund the duplicate charge"}).model  # -> english
```

If most of your traffic is English, set it back. The English checkpoint scores 0.820 on English against the multilingual checkpoint's 0.710, and the break-even is around 62% English traffic:

```python
router = Router(default="english")
```

Running one of the shipped servers rather than your own `Router`, the same setting is
`LAYA_DEFAULT_MODEL=english` — in the environment of `laya-serve`, the MCP server or the
`laya-serve` container ([docs/docker.md](https://github.com/NandhaKishorM/laya/blob/main/docs/docker.md)). It is the routing fallback only: text
the detector can place is routed on what it detects, whatever this is set to.

### Heterogeneous routed batches

If a lazy router receives an interleaved workload whose requests route to different checkpoints, calling `predict()` in a loop can still cause unnecessary checkpoint churn when the required checkpoints exceed the resident cache, for example with `max_loaded=1` or when `typed-decisions` is also used.

`Router.predict_batch()` routes the full workload first, groups requests by checkpoint, then groups requests with the same question schema within each checkpoint. Each compatible group is dispatched to `Agent.predict_batch()` so states can share forward passes, and results are restored to the original request order.

```python
requests = [
    {"state": "Please refund invoice 1", "questions": questions},
    {"state": "تم خصم المبلغ مرتين", "questions": questions},
    {"state": "Please refund invoice 2", "questions": questions},
]

results = Router(max_loaded=1).predict_batch(requests)
# results stay in input order while compatible requests are batched by checkpoint
```
Each item can independently set `model`, `task`, `lang`, `lang_guess`, or the token budget (`max_len`, `head_max_len`). Use `route_batch(requests)` when you only want the ordered routing decisions without loading any checkpoint. `predict_many` is an alias for `predict_batch`.

Requests are validated before model loading. Different requests may use different question schemas; requests sharing a checkpoint, question schema and token budget are passed together to `Agent.predict_batch()`.

* **Per-call hooks reach the batch path.** `Router.predict_batch` takes `hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise` and `hooks_timeout`, matching `predict`; `Router.route_batch` takes `hooks`, `hooks_raise` and `hooks_timeout`, since it fires only `on_route` and has no predict events for the convenience callables to bind to (#909). The batched dispatch also composes installed hooks properly, so a `set_default_hooks` default no longer fires on `predict` and silently vanishes on `predict_batch`.

[Prediction hooks](#prediction-hooks) installed on the `Router` run once per request, as they do for `predict()`, so a redaction hook rewrites every state before the model sees it. Requests that share a checkpoint run all their start hooks before their shared forward pass; see [`docs/hooks/lifecycle.md`](https://github.com/NandhaKishorM/laya/blob/main/docs/hooks/lifecycle.md#routerpredict_batch).

You can also bound the Agent-level forward-pass batch size, and forward the length grouping knob
to every group (see [Batch Mode](#batch-mode-score-many-states-in-one-forward-pass)):
```python
results = router.predict_batch(requests, batch_size=8, sort_by_length=True)
```

### Why Route: The Evidence

On a shared benchmark (17,416 questions, one T4 GPU, identical questions per model):

| Benchmark / Task | English (`laya`) | Multilingual (`laya-multilingual`) | `Router` (Routed) |
|---|---|---|---|
| MASSIVE intent, English | **0.783** | 0.657 | **0.783** |
| MASSIVE intent, 13 other languages | 0.306 | **0.451** | **0.451** |
| XNLI, English | **0.860** | 0.843 | **0.860** |
| XNLI, 14 other languages | 0.521 | **0.731** | **0.731** |
| Languages usable (>3x random) | 23 / 51 | 48 / 51 | **48 / 51** |
| Latency, 1 question (T4 GPU) | 39.5 ms | **32.8 ms** | **32.8 ms** |
| Latency, 10 questions batched | 158.6 ms | **72.3 ms** | **72.3 ms** |

The English checkpoint collapses on non-Latin scripts (Khmer scores **0.000 accuracy at 0.952 confidence**, the raw-temperature figure; **0.705** as served after the clamp). Because the model stays confident while being wrong, confidence gating cannot save you. `Router` detects the script in <0.5 ms pure Python before the forward pass.

### Production Preload & Memory

A cold checkpoint build costs seconds; language detection costs microseconds. The lazy default keeps **two** checkpoints resident — `english` and `multilingual`, the only two automatic routing chooses between — so a language flip costs detection only once each has been built. `max_loaded=1` rebuilds the checkpoint it just evicted on *every* switch (measured at a 7.4 s median reload on CPU and 10.3 s on T4), and traffic that only ever sees one language never builds the second, so the default costs a single-language deployment nothing.

For a server or production app, preload:

```python
# Every checkpoint resident in memory; language flips cost detection only (<1 ms)
router = Router(preload=True)
router = Router(preload=True, device="cuda")     # or "mps" (Apple/AMD GPU), "cpu" to force

# Or preload only the specific checkpoints you serve:
router.preload(["english", "multilingual"])

# If your app already built an agent, attach it to avoid duplicate VRAM:
router.attach("english", existing_agent)

# Manage resident memory (default keeps two hot: english + multilingual, LRU eviction)
router = Router(max_loaded=3)       # keep all three hot, e.g. with auto_task_detection
router = Router(max_loaded=1)       # memory-constrained host, reloads on every switch
router.unload()                     # free memory
```

| Deployment Mode | Per-Request Latency | Model Reloads |
|---|---|---|
| `Router()` (lazy, `max_loaded=2`) | detection only (<1 ms) on a switch, after each language's first load | 1 the first time a language appears |
| `Router(max_loaded=1)` | 7 to 10 s on every language switch | 1 per switch |
| `Router(preload=True)` | **32.8 ms (GPU) / 193–464 ms (CPU)** | **none** |

A rebuild still re-reads the checkpoint, but each checkpoint's tokenizer is parsed once per process
and reused by every `Agent` — including one the Router rebuilds after eviction. The multilingual
`tokenizer.json` alone is 34 MB / 256k vocab, several times the cost of applying its weights.
Preloading is still the right answer for a server: it removes the rebuild rather than making it
cheaper.

### Supplying Your Own Language Detection

Routing asks one question: *can the English checkpoint read this state?* The built-in detector answers it from the script and a function-word heuristic, and is deliberately dependency-free. That heuristic is best-effort on Latin-script languages it holds no word list for, so a short request can carry no usable signal:

```python
from laya.lang import analyse
analyse("Care este ora in Tokyo?")
# {'script': 'latin', 'script_profile': {'latin': 1.0}, 'language': 'en', 'is_english': True,
#  'language_undecided': False, 'diacritic_rate': 0.0, 'non_latin_fraction': 0.0,
#  'mixed_segment': None}                                     -> the English checkpoint
```

`is_english` is the key routing reads. `language_undecided` separates "read as English" from
"carried no usable signal, so the default applies", and `mixed_segment` names the line or field
that made a mostly-English state non-English (`None` when none did).

If you already run a language-identification model, hand routing the answer instead of relying on the heuristic. `lang_guess` takes a language code or a callable receiving the state, and is checked after an explicit `lang=` and before detection:

```python
# A code you already know
router.predict(state, questions, lang_guess="ro")

# A callable, e.g. wrapping fastText, CLD3 or a transformer LID
router.predict(state, questions, lang_guess=lambda s: my_lid(s))

# Or install one for every request on a server
router = Router(preload=True, lang_guess=my_lid)
```

The hint only decides *English or not*: a code whose primary subtag is `en`, `eng` or `english` routes to the English checkpoint, and every other code that names a language routes to the multilingual one. `"en_US"` and `"en_US.UTF-8"` are read as English, so `$LANG` can be passed straight through. Returning `None`, or a code that names no language, makes it abstain and the built-in detector decides as before — so a LID model that is unsure does not force a checkpoint. `C`, `POSIX` and `C.UTF-8` abstain, which matters because `C.UTF-8` is the default `$LANG` in the official Python image: passing it through no longer pins every request to the multilingual checkpoint, which is what it used to do. The ISO 639-2 special codes `und`, `zxx` and `mul` abstain for the same reason. An explicit `model=`, `task=` or `lang=` still wins, and the default path is unchanged.

---

## Self-Hosting: HTTP Server (Jev-compatible)

`laya.serve` exposes the `Router` over HTTP on the same `POST /v1/systemone`
wire protocol as TypeSafe's hosted Jev API. Laya's answer payload is already
schema-identical to what Jev returns (`choice`/`score`/`noul` answers and a
`{input_tokens, output_tokens}` usage block), so an existing Jev client — e.g.
the [`hs-jev`](https://github.com/getmissionctrl/hs-jev) Haskell client — just
needs its `baseUrl` repointed; nothing else changes. A PHP client that targets
`laya-serve` rather than Jev also exists: [`marcreichel/laya-php`](https://github.com/marcreichel/laya-php).

```bash
pip install "laya[serve]"          # adds fastapi + uvicorn + python-multipart
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve   # binds 0.0.0.0:8000, preloads all 3 checkpoints
```

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"body": "billed twice, refund please or we cancel"},
  "questions": {"dept": {"type": "choice", "instructions": "which team?",
                "criteria": {"billing": "refunds", "tech": "bugs"}}}
}'
```

To evaluate multiple states in a single call, send a `states` array to `POST /v1/systemone/batch` (capped at 64 states; a batch larger than `LAYA_MAX_BATCH_TOKENS` is split across several forward passes rather than refused):

```bash
curl -s localhost:8000/v1/systemone/batch -H 'content-type: application/json' -d '{
  "states": [
    {"body": "billed twice, refund please"},
    {"body": "cannot login, getting 500 error"}
  ],
  "questions": {"dept": {"type": "choice", "instructions": "which team?",
                "criteria": {"billing": "refunds", "tech": "bugs"}}}
}'
```

The endpoint shares forward passes via `Router.predict_batch` and returns an array of `results` in matching order along with aggregated `total_usage`.


Configuration is by environment variable: `LAYA_HOST`, `LAYA_PORT`,
`LAYA_DEVICE`, `LAYA_PRELOAD`, `LAYA_MODELS` (comma list to preload),
`LAYA_THREADS` (cap torch intra-op threads for CPU inference — keep at or below
physical cores), `LAYA_AUTO_TASK`, `LAYA_DEFAULT_MODEL` (the checkpoint a state with no
language evidence falls back to, `multilingual` by default; set it to `english` when most of
your traffic is English), `LAYA_MAX_LOADED` (checkpoints resident at
once, 2 by default; raise it to 3 when `LAYA_AUTO_TASK` makes a third one
reachable on demand, or the server rebuilds one every time routing switches),
`LAYA_IDLE_UNLOAD_SECONDS` (unload idle checkpoints; `0` disables it, `300` frees device memory
after five minutes and makes the next request pay a cold load), `LAYA_EXTRA_MODELS` (a JSON object
`{"my-checkpoint": "/path/to/it"}` registering extra checkpoints — Hub repo ids, local directories
or `["repo", "subfolder"]` pairs — beside the bundled ones), and `LAYA_API_KEY` (when set, clients must
send `Authorization: Bearer <key>`). A client's `model` field is honoured when it
names a Laya checkpoint (`english`/`multilingual`/`typed-decisions`, a `LAYA_EXTRA_MODELS` name,
or a published Hub id).
`jev-1` and `convaiinnovations/laya` still mean the router auto-selects by script/language.
A path or an unpublished Hub id (`/path/to/checkpoint`, `org/repo`) is a 422 on both
`/v1/systemone` and `/v1/systemone/batch` instead of being answered by another checkpoint.
The same body can also carry `task`,
`lang`, `lang_guess`, `min_confidence`, `max_len` and `head_max_len` — the controls
`Router.predict` takes that a JSON body can state — while the five hook arguments
(`hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`, `hooks_timeout`) are refused
with 422, because a hook is a callable that has to live where the server runs.

When publishing the server under a reverse-proxy prefix such as `/laya`, set
`LAYA_ROOT_PATH=/laya`. The proxy should strip that prefix before forwarding;
the setting controls FastAPI-generated URLs and leaves the internal routes unchanged.

Three things differ from Jev when you port a client:

* **Options per question.** A question's options share the checkpoint's option budget, `head_max_len` (192 tokens on `laya`, 256 on the other two), not Jev's cap of 255 options. In addition, the HTTP server (`laya.serve`) enforces an amplification guard of at most 100 choice options per question (`MAX_CHOICE_OPTIONS = 100`, rejected with 413 before inference). Once options overflow the token budget, around 20 options with a short description each, every option is trimmed to fit, so long or similar labels can reach the model reading the same ([Where Jev leads](#where-jev-leads)). Once they no longer fit the window at all, the library rejects the request with 422. For more candidates, narrow them first with `predict_shortlist` ([Honest limits](#honest-limits)).
* **Score levels.** Every level needs a description. A `null` level is rejected with 422 rather than scored and echoed back in `legend`.
* **`confidence`** on `choice` and `score` answers is 1 minus normalised entropy, a measure of how concentrated the distribution is, not Jev's `(n·p_max − 1)/(n − 1)`. A threshold carried over from Jev does not transfer. For one calibrated number on every question type, gate on `answer_confidence`, the probability of the reported answer. Both values shift with the number of options, so fit and validate any cutoff on held-out data at the option counts your workload uses ([#394](https://github.com/NandhaKishorM/laya/issues/394)).

### Nix / NixOS

This repo is a flake. On a machine with an NVIDIA GPU:

```bash
nix run .#laya-serve          # build (prebuilt CUDA torch, no compile) and serve
nix develop                   # dev shell: torch-bin, transformers, fastapi, pytest
```

For a NixOS host, import the module and enable the service:

```nix
# flake inputs:  laya.url = "github:<you>/laya";  # or path:/… on the same host
{
  imports = [ laya.nixosModules.default ];
  services.laya-serve = {
    enable = true;
    host = "0.0.0.0";           # or bind to the Tailscale/LAN address
    openFirewall = true;
    device = "cuda";
    models = [ "english" "multilingual" "typed-decisions" ];
    # apiKeyFile = config.age.secrets.laya-api-key.path;  # optional bearer auth
  };
}
```

The module runs a hardened `DynamicUser` systemd unit with CUDA device access,
caches weights under `/var/lib/laya-serve`, and reads the bearer token (if any)
via `LoadCredential` so it never enters the store.

---

## Single-Model Mode (Direct SDK)

If you only need a single checkpoint for a dedicated pipeline, you can load models directly:

```python
import laya

# 1. Load a specific checkpoint directly from the hub
agent = laya.load("convaiinnovations/laya")                           # English root
agent_ml = laya.load("convaiinnovations/laya", subfolder="multilingual") # 100+ languages
agent_td = laya.load("convaiinnovations/laya", subfolder="typed-decisions")

# 2. Run all questions in ONE single forward pass (~35 ms on GPU)
result = agent.predict(state, questions)
answers = result["answers"]

print("Department :", answers["department"]["choice"])   # -> billing (confidence: 0.94)
print("Urgency    :", answers["urgency"]["score"])        # -> 1.84 / 2.0
print("Churn Risk :", answers["churn_risk"]["noul"])       # -> 0.892 (89.2% probability)
```

Passing an empty question dictionary to `agent.predict(state, {})` or
`agent.system_one(state, {})` returns the standard response with `"answers": {}`
and `"usage": {"input_tokens": 0, "output_tokens": 0}`. The state is not tokenized
and no model forward pass runs.

### Batch Mode: score many states in one forward pass

`predict` handles one state per call, which leaves most of the GPU's batch dimension idle. When you
have a list of items to score against the *same* questions — a backlog of tickets, a table of rows,
a log slice — `predict_batch` packs them into shared forward passes:

```python
states = [{"body": t} for t in ticket_texts]           # a list of states

results = agent.predict_batch(states, questions)       # one forward pass for the whole list
# results[i] corresponds to states[i], with the same output shape as predict

# Bound peak memory when the list (or the texts) are large — chunk into passes of N:
results = agent.predict_batch(states, questions, batch_size=64)

# Reduce padding when input lengths vary; results still follow the original state order:
results = agent.predict_batch(states, questions, batch_size=64, sort_by_length=True)
```

`sort_by_length=True` groups states by their longest encoded question row, after truncation.
It looks ahead at most eight batches and reuses the encoded rows for sorting. This uses more temporary
CPU memory for tokenized inputs, and takes effect only when `1 < batch_size < len(states)`.
Benchmark it on your workload and backend: uniform lengths offer little benefit, and changed
batch shapes can cause small floating-point differences, including near decision thresholds.
Hooks still see states and final results in input order. The option is available on `Agent` and
`ONNXAgent`.

Results are aligned with `states` by index and identical in shape to `predict`. Changing batch
shapes can introduce floating-point differences on CPU and GPU; check decision thresholds on
your workload, particularly with mixed precision. Batching is a **GPU
throughput win** — on an RTX 5060 Ti, per-decision latency drops from ~10 ms one-by-one to ~1 ms
batched (measured ~9–10×). On CPU, increasing batch size alone may not speed up inference;
length grouping can help by reducing the padded work in a mixed-length workload. See the
[CPU measurements and reproduction commands](https://github.com/NandhaKishorM/laya/blob/main/research/README.md#length-batching).

`ONNXAgent.predict_batch(states, questions, batch_size=..., sort_by_length=...)` has the same
contract, backed by one ONNX Runtime session run per chunk, so an ONNX deployment gets the same
batch API, the same result shape, and the same length grouping. Measured on the English checkpoint
(flat fp32 export, Apple M1, 80 support tickets alternating short and ~8x-longer documents,
`batch_size=4`, best of 3): wall clock 53.7 s unsorted → 36.3 s sorted (**~1.48x**), with 0/80
decision changes and max probability drift 0.0 across queue/noul/score.

### Long documents: `predict_long`

`predict`/`system_one` truncate a state that exceeds `max_len` to a single window (the first, or
for a conversation list the last), silently dropping the rest. `predict_long` scans the whole state
in overlapping windows, scores them in shared forward passes (via `predict_batch`), and aggregates
per question:

```python
result = agent.predict_long(state, questions)              # windows the state, one result back
result = agent.predict_long(state, questions, window=256)  # smaller window isolates a localized span
result = agent.predict_long(state, questions, hooks=[AuditLog()])   # the scan, instrumented
```

- `noul` takes the strongest window (the statement holds if any window supports it).
- `choice` / `score` take the most-confident window — averaging over a long, mostly-neutral
  document lets the neutral majority out-vote the one window that saw the deciding span.
- A state that already fits one window is passed straight to `system_one` (identical output, plus
  `usage["windows"] = 1`). Without an explicit `window`, that is any state the questions leave
  room for, since `system_one` reads it whole. The key is total: `1` single window, `N` scanned windows, `0` a hook
  answered the document before the model read any of it.
- Hooks wrap the inference that answers the state, so on a scanned document `on_predict_start`
  fires once with `ctx.states` holding the decoded windows, not the state you passed in (it was
  tokenized to produce them). A start hook may replace that list: the answers are aggregated over
  whatever reached inference, and `usage["windows"]` counts those states. What a rewritten scan
  costs is the attribution — `answer["window"]` names a span of *your* document, so it is only
  reported when the scan reached inference unchanged. A hook that means to answer the document
  calls `ctx.skip(...)` instead: its result comes back with no `answer["window"]` and
  `usage["windows"]` at 0, because no window scored it.

A smaller `window` isolates a short deciding span better (it becomes a larger fraction of its
window); the default (`max(64, max_len - head_max_len - 8)` — 312 tokens on the English checkpoint
at its 512/192 defaults, 760 on the other two at 1,024/256) favors context and throughput. Either
way the window is capped at the room the questions leave for the state inside `max_len`, so a
window is never re-truncated on the way to the model and `token_start`/`token_end` describe the
span it read:
many options leave little room (on the English checkpoint, 2 options leave 483 state tokens and
100 leave 100), and a window past that room used to be cut short silently. Output shape matches
`predict`, with `usage["windows"]` added.

`ONNXAgent.predict_long(state, questions, window=..., stride=..., batch_size=...)` has the same
contract and the same aggregation rules, with the windows scored through `ONNXAgent.predict_batch`
— one ONNX Runtime session run for all of them, or one per chunk when `batch_size` bounds memory.

The returned probability is the deciding window's, **not a calibrated number for the whole
document** — a `noul` max drifts up with the window count even with no signal, and `choice` can land
on a confidently-neutral window when nothing is decisive. Each answer carries `answer["window"]`
(the deciding window's `index`, `token_start`/`token_end`, and `count`) so you can check the span
the answer actually came from:

```python
r = agent.predict_long(state, questions)
r["answers"]["refund"]["window"]   # {'index': 13, 'token_start': 4680, 'token_end': 5432, 'count': 14}
```

The same scan is reachable from the Router, which routes first and then windows the checkpoint it
picked — the same `model=`/`task=`/`lang=` hints, hooks and `routing` key as `predict`:

```python
result = router.predict_long(state, questions, model="multilingual")
```

---

## GPU Fast Path (TileLang)

`pip install laya[fast]` adds an optional forward built from fused [TileLang](https://github.com/tile-ai/tilelang)
kernels: GEMM + bias/activation epilogues, GEMM + GEGLU, residual + LayerNorm, in-place RoPE, and a
sliding-window flash attention that reads the packed QKV buffer directly. Weights stay resident in bf16
and every (batch, length) bucket is captured as a CUDA graph, so a one-question call no longer pays
~200 kernel launches from Python.

```python
agent = laya.load("convaiinnovations/laya", fast=True)   # or: agent.accelerate()
agent.predict(state, questions)                            # same API, same answers
```

Numerics: on a fixed set of 60 states the fast path stays within 0.046 of an fp32 forward and within 0.076 of the stock
bf16 path (max |Δp| ≤ 0.05 vs fp32 on both checkpoints, argmax agreement ≥ 47/48 per question type; every per-option
probability is in `benchmarks/results/parity_*.json`) — see `benchmarks/parity_fast.py` and [BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md#gpu-fast-path).
The fast path runs in the agent's autocast dtype at the time `accelerate()` is called: bf16 by default, fp16 if
`agent.dtype` is `torch.float16`, where it stays within 0.009 of fp32 on the same set ([BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md#fp16)).
Falls back to the stock forward on CPU/MPS or when `tilelang` is not installed; `agent.deaccelerate()`
restores it. Kernels compile once per shape bucket on first use (a few seconds, cached on disk).

### Warm-up before serving: `agent.warmup()`

`compile=True` now runs `agent.warmup()` before returning from load. This moves the batch and
single-row graph compiles into startup; startup can take tens of seconds. To defer the work:

```python
agent = laya.load("convaiinnovations/laya", compile=True, compile_warmup=False)
seconds = agent.warmup()         # manual entry point, also accepts shapes=[(rows, tokens, markers)]
```

If automatic warm-up fails, load emits a `RuntimeWarning` naming the underlying error and returns
with the `torch.compile` wrapper and compile settings intact. Later requests still use the compiled
model and propagate compilation failures; there is no automatic eager fallback. Explicit
`agent.warmup()` calls also propagate failures, even after automatic warm-up has failed.

Warm-up uses synthetic inputs without prediction hooks or prediction caches. It covers common batch
and single-row shapes, not every possible guard or CUDA graph shape. `compile=False` and `fast=True`
do not warm automatically. Manual warm-up still works on both paths.
On CUDA, compiled inference pads the masked end of each sequence to a multiple of eight tokens. This
avoids extra SDPA graph specialisations on PyTorch versions that distinguish lengths modulo eight;
reported token usage still counts the original, unpadded sequence.
Inductor caches compiled graphs under `TORCHINDUCTOR_CACHE_DIR` (by default in `/tmp`); point it at a
persistent directory to keep them across restarts. Alternatively, load with
`compile=True, compile_cache=True` to use `$XDG_CACHE_HOME/laya/torchinductor` (or
`~/.cache/laya/torchinductor`). This opt-in sets the process-wide environment variable only if
absent; an existing value always wins. It does nothing on eager or TileLang loads.

`compile_mode="reduce-overhead"` is a separate opt-in for CUDA graphs. With `compile=True`, it
can reduce launch overhead on repeated shapes, but each new shape needs recording and GPU memory
can grow. Laya copies outputs out of reusable graph buffers and serializes these CUDA forwards.
The default mode remains unchanged. See [the engineering notes](https://github.com/NandhaKishorM/laya/blob/main/docs/compile-and-fast-path.md)
for measurements and limitations.

---

## Automated Confidence Gating

Because Laya's probabilities are trained with strictly proper scoring rules (RLCD), confidence scores are statistically meaningful:

```python
dept = answers["department"]["choice"]
conf = answers["department"]["confidence"]

if conf >= THRESHOLD:               # refit and validate THRESHOLD on your own held-out data
    route_automatically(dept)       # above it: act, and sample the decisions you act on
else:
    escalate_to_human_agent(dept, reason=f"Low confidence ({conf:.2f})")
```

A threshold is a policy you choose from measured accuracy at that coverage on your data, not a property of the model. Both checkpoints are over-confident as shipped and `laya-multilingual` has no fitted temperatures at all, so fit them before relying on these numbers — see [Calibration](#calibration) below, and the [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) for the fitting loop itself. Then pick the point where the errors you accept are ones you can live with. Confidence orders decisions; it does not establish that a decision is correct.

A threshold also depends on the autocast dtype. On CUDA at compute capability 8 or above the runtime uses the checkpoint's `amp_dtype`, which is bf16 for all three shipped checkpoints. On the fixed set from `benchmarks/parity_fast.py` (60 states, 288 questions per checkpoint, RTX 2000 Ada) bf16 moves a probability by up to 0.073 against the fp32 forward and flips 3 of 864 argmaxes across the three checkpoints; fp16 stays within 0.019 and flips none, at the same latency. `LAYA_CUDA_AMP=fp16` selects fp16 and `LAYA_CUDA_AMP=bf16` selects bf16 (`LAYA_CPU_AMP=bf16` is the CPU counterpart). MPS autocasts in fp16 too, but the overhead dominates on a single small row, so there it engages only once a call reaches `mps_amp_min_rows` rows -- 5 by default, `LAYA_MPS_AMP_MIN_ROWS` to move it; a value that does not parse falls back to 5 and anything below 1 is clamped to 1. On MPS, `agent.dtype` is the autocast target, so it says float16 even when every call stays below the gate and runs in fp32; `agent.dtype_for(rows)` returns the precision a call with that many rows runs in. Fit and measure a threshold in the dtype you serve with.

### Opt-in abstention: `min_confidence`

`predict`, `predict_batch`, `system_one`, `decide` and `decide_batch` — on `Agent`, `Router` and `ONNXAgent` — take an opt-in `min_confidence`, off by default. It is a caller-side policy on top of the emitted confidence: every answer whose `answer_confidence` falls below the threshold is flagged `low_confidence: True`, with the raw answer, probabilities and confidence left intact for inspection.

```python
res = agent.predict(state, questions, min_confidence=0.85)
ans = res["answers"]["department"]

if ans.get("low_confidence"):        # answer_confidence < 0.85
    escalate_to_human_agent(ans["choice"], reason=f"Low confidence ({ans['answer_confidence']:.2f})")
else:
    route_automatically(ans["choice"])
```

The threshold reads `answer_confidence` (`max(p)`) — the calibrated quantity, invariant to the number of options — never the entropy `confidence`. With `decide(..., min_confidence=...)` a low-confidence field comes back as `None` in the schema output, while `return_details=True` keeps the answer and its confidence. [LangChain `LayaRouter`](https://github.com/NandhaKishorM/laya/blob/main/docs/langchain.md)'s `confidence_threshold` reads the same value: `answer_confidence` when the answer carries it, `confidence` otherwise. Left unset, `min_confidence` changes nothing.

A gate is a policy, and a policy whose application you cannot observe is not one. `low_confidence` is written only when the gate fires, so its absence cannot tell you "a gate ran and this answer cleared it" apart from "no gate ran at all" — you could neither compute an abstention rate nor prove the gate was in effect. **When you pass `min_confidence`,** every answer therefore also reports `abstention`:

| `abstention` | meaning |
|---|---|
| `passed` | a gate ran and this answer's `answer_confidence` cleared it |
| `abstained` | a gate ran and this answer's `answer_confidence` fell below it |
| `unevaluated` | a gate ran and this answer carried no usable confidence, so the gate could not decide |

`abstention_threshold` echoes the threshold whenever a gate ran, so a log can be re-split by the gate that produced it instead of by whatever the caller happened to remember passing. `unevaluated` is the case a boolean cannot express — reporting it as a pass would be as wrong as reporting it as a flag, so a NaN, a missing confidence or a `bool` lands in its own state. The raw answer, probabilities and confidence are untouched either way.

Left without `min_confidence`, the response is byte-for-byte what it was before any of this: no `abstention`, no `abstention_threshold`, no flag. That is deliberate — a field on every answer that only some calls populate is a schema change for callers who never asked to be gated. The presence of `abstention` is how you tell a gated run from an ungated one; a `min_confidence` of exactly `0.0` still counts as set, so it reports states, and `flag_low_confidence` treats `0.0` as a no-op because nothing can fall below it.

The same gate is a flag on the `laya` command:

```bash
laya "Is it the blue one or the green one" --preset triage --min-confidence 0.9
```

```text
intent      : other (p=0.992)
frustration : 1.06  [low-confidence]
churn_risk  : 0.158  [low-confidence]
```

Each marked answer gets `[low-confidence]` on its printed line and the answer itself is still
printed, so a run at a terminal reads the same way the dict does; `--json` carries the raw
`low_confidence` and `abstention` keys. `--batch FILE` applies one threshold to the whole file, which is the shape
`predict_batch` takes. Routing has no answer to gate, so `laya --min-confidence 0.9 "..."` without
`--predict`, `--preset` or `--questions` is refused rather than ignored.

---

## Prediction Hooks

Hooks observe or shape every decision without forking: audit logging, PII redaction before
inference, caching, metrics, confidence gating, routing overrides, and forwarding to an
external service. They are opt-in, and unset hooks are a no-op.

```python
import laya

def log(ctx):
    print(ctx.model, ctx.results[0]["answers"], ctx.elapsed_ms)

agent = laya.load("convaiinnovations/laya", on_predict_end=log)
agent.system_one("I was charged twice.", {"urgent": {"type": "noul", "instructions": "Urgent?"}})
```

A hook is a plain callable, or an object implementing any of `on_predict_start`,
`on_predict_end`, `on_route`, `on_load`, `on_evict`, `on_error`. A start hook can rewrite the
state/questions or `ctx.skip(...)` a cached answer; an end hook can rewrite the result. See
[**`docs/hooks/`**](https://github.com/NandhaKishorM/laya/blob/main/docs/hooks/index.md) and [`examples/hooks/`](https://github.com/NandhaKishorM/laya/tree/main/examples/hooks/).

---

## Schema-driven decisions

Describe the shape you want with a JSON schema or a pydantic model, and Laya answers it in one
forward pass, with typed values and calibrated confidence.

```python
import laya

schema = {
    "type": "object",
    "properties": {
        "department": {"type": "string", "enum": ["billing", "support", "sales"],
                       "description": "Which team should handle this?"},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"},
    },
}

agent = laya.load("convaiinnovations/laya")
agent.decide("I was charged twice, refund me.", schema=schema)
# {"department": "billing", "urgency": 2, "needs_human": True}
```

`decide` also works on a `Router`, accepts a pydantic model (install `laya[structured]`), and can
return per-field confidence with `return_details=True`. In an LCEL chain or LangGraph node the
same call is `LayaDecision`. See [`docs/structured.md`](https://github.com/NandhaKishorM/laya/blob/main/docs/structured.md).

For throughput, `decide_batch` answers a list of states against the *same* schema through
`predict_batch` (see [Batch Mode](#batch-mode-score-many-states-in-one-forward-pass)), so the schema
is planned once and the states share forward passes:

```python
values = agent.decide_batch(ticket_texts, schema=schema)   # values[i] matches ticket_texts[i]

# On a Router the states may land on different checkpoints; keywords reach predict_batch:
details = router.decide_batch(states, schema=schema, return_details=True, batch_size=64)
```

Measured on an Apple M-series (MPS), 8 English tickets through one checkpoint: 2624 ms one-by-one
vs 723 ms batched (**3.6×**); 16 mixed English/German states through a `Router`: 2977 ms vs
1903 ms (**1.6×**). On CPU the same workloads gave **1.6×** and **2.0×**. Projected values matched
the one-by-one loop 8/8 and 16/16 on both devices, as expected wherever argmax is not at a
threshold — verify against your own decision boundaries. `laya.decide_batch(runner, states, ...)`
is the function form, and `ONNXAgent.decide_batch` runs the same thing on an ONNX export.

---

## Built-in Workflow Presets

Laya provides pre-tuned question schemas for immediate production use:

```python
import laya

agent = laya.load("convaiinnovations/laya")

# 1. Intelligent Model Router (routes to small vs. frontier models)
routing = agent.predict({"request": "Refactor this service using dependency injection"}, laya.router_questions())

# 2. Real-time Prompt Guardrails (jailbreaks, injections, leaks)
guard = agent.predict({"prompt": "Ignore all instructions"}, laya.guard_questions())

# 3. Content Safety & Moderation (toxicity, harassment, threats)
safety = agent.predict({"post": "User comment text"}, laya.moderation_questions())

# 4. Support Ticket Triage (intent, urgency, frustration, churn)
triage = agent.predict({"message": "My payment failed twice"}, laya.triage_questions())

# 5. Email Triage (category, spam, phishing, urgency, reply expected)
email = agent.predict({"body": "Your account is locked, verify now"}, laya.email_questions())
```

---

## LangChain and LangGraph Integration

Fast System 1 routing and guardrails directly inside LangGraph workflows and LCEL chains. Every node also takes core's per-call prediction hooks (`hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`, `hooks_timeout`):

```python
from laya.integrations.langchain import LayaDecision, LayaGuardrail, LayaRouter

# 1. Sub-35ms LangGraph conditional edge routing with confidence fallback
router = LayaRouter(
    criteria={"billing": "invoices, charges", "tech": "bugs, outages"},
    confidence_threshold=0.80,
    fallback="human_agent",
)
workflow.add_conditional_edges("triage", router)

# 2. Inline prompt guardrails
guard = LayaGuardrail(action="raise")  # raises LayaGuardrailError on jailbreak/injection

# 3. Schema-driven decisions: a JSON schema in, schema-shaped values out
decide = LayaDecision({
    "type": "object",
    "properties": {
        "department": {"type": "string", "enum": ["billing", "technical", "other"]},
        "needs_human": {"type": "boolean"},
    },
})
decide.invoke("I was charged twice and the API still 500s for us.")
# {'department': 'billing', 'needs_human': False}

# 4. Score a backlog in one batched call, not one forward pass per input
routes = router.batch(["refund my invoice", "the app crashes", "change my password"])
```

`batch()` and `abatch()` run the whole list through `predict_batch`, so `chain.batch(...)`,
`RunnableParallel` and LangGraph map-reduce nodes get Laya's shared forward passes instead of
LangChain's default one-call-per-input thread pool -- which races on MPS, where concurrent torch
forwards abort the process. Measured on Apple M-series (medians of three): **2.2x** on a 16-ticket
routing batch, **2.2x** across 24 mixed-language tickets through a `Router`, **1.8x** on the guard
preset, with every route label and guardrail flag unchanged. On CPU the same workloads are
2.2-2.4x over the one-by-one loop and 1.1-1.5x over the thread pool.

See [**`docs/langchain.md`**](https://github.com/NandhaKishorM/laya/blob/main/docs/langchain.md) for the full guide, support ticket triage nodes, and remote HTTP server configuration.

---

## Decision Primitives

| Primitive | Output | Use Cases |
|---|---|---|
| **`choice`** | Top label, probabilities per option, confidence | Department routing, intent classification, topic categorization |
| **`score`** | Expected level on ordinal rubric, distribution, confidence | Frustration level, ticket urgency, harm severity |
| **`noul`** | Calibrated probability P(true) from 0.0 to 1.0 | Phishing detection, spam filtering, jailbreak detection, churn risk |

`noul` always scores two semantic slots in `[false, true]` order and returns the probability of
the second slot. For compatibility, those slots are shown to the model as `false` and `true` by
default. The optional `labels` mapping overrides only that model-facing text without changing the
returned meaning:

```python
question = {
    "type": "noul",
    "instructions": "Is this review positive?",
    "criteria": {
        "false": "the review is negative",
        "true": "the review is positive",
    },
    "labels": {
        "false": "B",
        "true": "A",
    },
}
```

The `labels` mapping is optional. It must contain exactly the string keys `false` and `true`,
whose values must be distinct non-empty strings. Mapping order does not matter, and the returned
`noul` value is still P(true). Label sensitivity varies by checkpoint and state, so validate any
override on your own data rather than treating `A`/`B` as a universal fix.

### Option order

Every question type also takes an optional `option_order`: a permutation of the option indices
saying which option goes in which slot. Slot `s` shows option `option_order[s]`. Probabilities
always come back keyed in your own option order, whatever order the model saw them in, so the
key is presentation only — it never changes what a returned label means.

```python
question = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "...", "technical": "...", "sales": "..."},
    "option_order": [2, 0, 1],     # slot 0 shows `sales`, slot 1 `billing`, slot 2 `technical`
}
```

It exists because where an option sits changes the answer. `research/eval/presentation_checks.py`
measures the English checkpoint on five options whose text is *identical*: the per-slot logits
centre at `[+1.68, +1.47, +0.11, -1.35, -1.91]`, a spread decided by position alone, and the
multilingual checkpoint leans the other way on `score` (see [#131](https://github.com/NandhaKishorM/laya/issues/131)).
The checkpoint-side fix is a position-balanced retrain; until then `option_order` lets a caller
average the effect out, by asking the same question under orders in which every option occupies
every slot equally often:

```python
k = len(question["criteria"])

# the k rotations go in as k questions, so they share one forward pass
rotations = {
    "q%d" % r: dict(question, option_order=[(i + r) % k for i in range(k)])
    for r in range(k)
}
answers = agent.predict(state, rotations)["answers"]

totals = {label: 0.0 for label in question["criteria"]}
for answer in answers.values():
    for label, value in answer["probabilities"].items():
        totals[label] += value / k

decision = max(totals, key=totals.get)
```

`predict` already scores every question in a request in one parallel forward pass, so this costs
`k` rows rather than `k` forward passes — sending the rotations as separate `predict` calls would
cost `k` of them. On 62 banking intents shortlisted to 20 over 49 states, averaging this way took
the share of answers that change with presentation order from 16.3% to 6.1%. Omitting
`option_order` keeps the canonical order and the behaviour every existing caller already has.

#### Order-invariant checkpoints (`option_layout: "parallel"`)

Averaging reduces the position effect; a checkpoint trained on the parallel option layout does not
have one. Every option starts at the same position id, the instruction's closing `[SEP]` and the
state continue after the longest option, and options cannot attend to each other (the local layers
keep their window, measured in position ids). The decision head has no positional encoding, so
reordering the options can only reorder the logits: `option_order` and any other permutation return
the same probabilities per label, up to float rounding. The idea is the one Parallel Context Windows
([Ratner et al., 2023](https://arxiv.org/abs/2212.10947)) uses for long contexts, applied to options.

The layout is a property of the weights, so it is set in the checkpoint's `rl_agent_config.json`
(`"option_layout": "parallel"`) and picked up by `laya.load`; every published checkpoint stays on the
default sequential layout. Train one with `laya.train`:

```python
from laya.train import TrainConfig, finetune

finetune("train.jsonl", "./laya_base", "./laya_parallel", TrainConfig(option_layout="parallel"))
```

Fine-tuned from `convaiinnovations/laya` on typed-decisions (4 epochs, same recipe and seed per arm,
one T4, 2,000 test decisions; each question also re-asked under a random `option_order`):

| layout | accuracy (listed order) | accuracy (shuffled) | ECE | same answer when shuffled |
|---|---|---|---|---|
| sequential | 0.779 | 0.767 | 0.153 | 91.7% |
| sequential + `shuffle_options` | 0.757 | 0.750 | 0.136 | 95.0% |
| parallel | 0.764 | 0.764 | 0.143 | 100% |

One seed per arm; the sequential arm moved by 0.008 between two runs of the same recipe. The
parallel layout needs transformers>=5 and runs on the eager backend only: the ONNX runtime, the
TileLang fast path and `compile=True` refuse such a checkpoint, and `backend="auto"` resolves to eager.

---

## MCP Server (Optional)

Laya can be exposed as an [MCP](https://modelcontextprotocol.io) stdio server, so any MCP
client (OpenClaw, Claude Desktop, Cursor, ...) can call typed decisions as tools
(`laya_predict`, `laya_predict_batch`, `laya_route`, `laya_route_batch`, `laya_decide`, `laya_shortlist`, `laya_preset`, `laya_status`) without writing glue code.
This is an **optional extra**: the core package has no `mcp` dependency.

```bash
pip install "laya[mcp]"
laya-mcp-server          # or: python -m laya.mcp.server
```

To share a `laya-serve` on your own hardware across editor sessions, set
`LAYA_BASE_URL=http://127.0.0.1:8000` in each MCP client's environment. Its prediction tools use
HTTP and import no torch; routing stays local. Set `LAYA_API_KEY` to the server's bearer key and
`LAYA_REMOTE_TIMEOUT` to the allowed HTTP wait (300 seconds by default). `laya_shortlist` is
unavailable in this mode. See [shared-server MCP setup](https://github.com/NandhaKishorM/laya/blob/main/docs/cli-mcp.md#share-one-model-server-across-mcp-sessions).

Example MCP client configuration (stdio transport):

```json
{
  "mcpServers": {
    "laya": {
      "command": "laya-mcp-server",
      "env": { "LAYA_DEVICE": "cpu" }
    }
  }
}
```

The environment variables follow the contract documented at the top of
[`laya/serve.py`](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py), so the same variable has one meaning across the
package:

| Variable | Default | Meaning |
|---|---|---|
| `LAYA_DEVICE` | (auto) | Same as `laya.serve`: the device type is lower-cased (`CUDA` → `cuda`, `CUDA:0` → `cuda:0`) and passed to torch, whose device parser is case-sensitive |
| `LAYA_PRELOAD` | `1` | Same as `laya.serve`: build the checkpoints at startup, not lazily |
| `LAYA_MODELS` | `english,multilingual` | Comma list to preload (serve contract). MCP difference: an empty value preloads `english,multilingual` so `typed-decisions` stays lazy; in `laya.serve` empty means every checkpoint |
| `LAYA_THREADS` | (torch default) | Same as `laya.serve`: cap torch intra-op threads for CPU inference; keep it at or below the physical core count |
| `LAYA_AUTO_TASK` | `0` | Same as `laya.serve`: `1` lets a request whose question ids match a typed-decisions workflow route to that checkpoint, which is then loaded on demand; it never joins the preload list |

The tools return structured JSON (answers with probabilities, routing metadata, device,
`latency_ms`). `laya_predict_batch` and `laya_route_batch` are the MCP form of
[`Router.predict_batch` / `route_batch`](#batch-mode-score-many-states-in-one-forward-pass):
one tool call takes an array of `{state, questions, model?, task?, lang?, lang_guess?, max_len?, head_max_len?}` requests, routes them
first, groups them by checkpoint, and shares forward passes between requests with the same
question schema, returning the answers in input order. `max_len` / `head_max_len` are per request
there, exactly as `Router.predict_batch` reads them, so one wide question can raise its own budget
without shrinking the rest of the batch to it. On 16 mixed-language tickets through
the tool functions themselves, one batch call beat 16 `laya_predict` calls by **2.1-2.3x on
MPS** (983-1082 ms -> 467-477 ms) and **~1.25x on CPU** (1861-2471 ms -> 1470-1911 ms), with
**0/16 decision flips** (choice label, rounded score, noul sign) against the loop. Prefer it
whenever a client has more than a few requests: each saved round trip is also an MCP
request/response. `laya_decide` is the MCP form of [`laya.decide`](#schema-driven-decisions): it
takes a JSON schema (enum choices, booleans, bounded integers) instead of hand-written
questions and returns the decided `values` projected onto that schema -- enum member, integer
level, boolean -- beside per-field `confidence` and `probabilities`, so a client that already
knows the answer shape never parses an answer map by hand. `laya_shortlist` is the MCP form of [`predict_shortlist`](#honest-limits):
it shortlists a many-option choice question to its `k` most likely labels by embedding
similarity (mean-pooled from the answering checkpoint's own encoder, so no extra model is
downloaded), answers in one forward pass, and returns per-question shortlist metadata
(kept labels, cosine scores, `k`, option count, and whether the question passed through
unshortlisted). The guardrails shown on every decision
tool point clients to `laya_shortlist` for >20-option choices. As with the SDK, use it for structured
decisions only; not for open Q&A or
text generation. Tests: `tests/test_mcp.py` (CI, no weights) and
`tests/test_mcp_local_e2e.py` (local, real weights and a real stdio handshake).

---

## Evaluation harness

`laya.evals` scores a labelled dataset and gates a build on it, so a quality change is a
reviewable diff instead of a hand-check. It is pure Python plus numpy, imports no torch, and
needs no weights until you point it at a checkpoint.

```bash
laya-evals validate research/evals/fixture.jsonl
laya-evals run data.jsonl --model english --min-accuracy 0.8 --max-ece 0.05 --slice language
laya-evals run data.jsonl --baseline baseline.json --tolerance choice_accuracy=0.02 --json report.json
```

`run` reports overall and per-slice metrics (`choice_accuracy`, `noul_accuracy`, `score_mae`,
`ece`, `mean_confidence`, latency) and exits non-zero on a threshold or baseline failure, so it
drops into CI unchanged. A weight-free job runs the metric and API tests on every PR, and a
scheduled workflow evaluates the English checkpoint against the committed baseline. See
[**`docs/evals.md`**](https://github.com/NandhaKishorM/laya/blob/main/docs/evals.md) for the dataset format and the gate.

---

## Benchmarks

Community diagnostics: [Chinese workplace decisions (Feishu-style)](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/README.md) · [中文说明](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/README.zh-CN.md). Includes frozen synthetic cases, archived paired Laya/Jev responses, and an offline audit; separate from the benchmark suites below. Also [Chinese short-command routing](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/zh_short_commands/README.md) · [中文说明](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/zh_short_commands/README.zh-CN.md): 18 frozen commands and a seven-rung ablation of the documented prompt guidance, which locates the accuracy loss on the four-question path rather than the six-option one.

**Full report: [`BENCHMARKS.md`](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)** — every run consolidated, languages and themes, with per-language detail for all 51 languages.

<p align="center">
  <img src="https://raw.githubusercontent.com/NandhaKishorM/laya/main/assets/laya_benchmark.png" alt="Per-language accuracy for both checkpoints across 51 languages" width="100%" />
</p>

All Laya numbers below are measured. Every model answered byte-identical questions
(fixed seed) in the same run. Reproduce with
[`research/scripts/laya_benchmark_colab.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/laya_benchmark_colab.ipynb) on a T4.

### Speed (Tesla T4, measured)

| questions per call | `laya` | `laya-multilingual` |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 5 | 84.5 ms | **40.1 ms** |
| 10 | 158.6 ms (15.9 ms/q) | **72.3 ms (7.2 ms/q)** |
| 50 | 771 ms | **337 ms (6.8 ms/q)** |

Batched throughput reaches 103-332 questions/sec on a single T4. For reference, TypeSafe Jev
has been independently measured at 236-276 ms p50
([AbdelStark](https://github.com/AbdelStark/jev-benchmarks),
[nibzard](https://github.com/nibzard/decision-model-benchmark)) -- Laya answers a single
question roughly **6-7x faster**.

### Laya (with routing) vs Jev

Every Laya figure is what `Router().predict(...)` actually returns — the checkpoint the router
selects for that input, not a hand-picked best of three. Jev figures are **third-party
published, never measured here** (no TypeSafe API access), so sample sizes and prompts differ.
The Laya column here comes from the Applications run (`research/scripts/bench_apps.py`, N=400 per task; `research/results/app_benchmark_results.json`), so AG News / DAIR Emotion read 0.950 / 0.595 here versus 0.947 / 0.573 in the committed T4 English suites (N=600) below.

| | Jev 1.13.0 | Laya (routed) | |
|---|---|---|---|
| typed-decisions, 2,000 decisions | 0.727 | **0.766** | +0.039 |
| AG News, 4 labels | 0.910 | **0.950** | +0.040 |
| DAIR Emotion, 6 labels | 0.480 | **0.595** | +0.115 |
| Banking77 (72 vs 77 labels) | **0.870** | 0.425 | Jev leads on >20 options |
| ECE *(lower better)* | 0.246 | **0.081** | 3× better (post-temperature) |
| p50 latency, 1 question | 236–276 ms | **32.8 ms** | 7.8× faster |
| Languages usable | *no published benchmark* | **48 of 51** | — |
| Weights | closed API | **Apache 2.0** | — |
| Cost | $0.042 / 1M tokens | **$0 self-hosted** | — |

On DAIR Emotion, Jev assigned **zero probability to the true label on 16% of examples** — a hard
failure for anything branching on confidence.

#### Where Jev leads

* **High-cardinality label spaces (>20 options at default settings):** On Banking77, Jev scores 0.870 (on 72 labels) while Laya scores 0.425 (on 77 labels at default 256-token head budget). This is an architectural token-budget constraint: options share a fixed `head_max_len` budget (192 tokens on English, 256 on multilingual), so 77 options receive only ~3 to 4 tokens per label, causing text to become indistinguishable. Jev supports up to 255 options out-of-the-box. While `laya-multilingual` supports 1,024 context (and up to 8,192 in the encoder) and you can raise `agent.cfg["head_max_len"] = 512` at runtime, Jev is currently better suited for 50+ options in a single prompt without tuning. `predict_shortlist` (see [Honest limits](#honest-limits)) keeps the top `k` labels with a caller-supplied embedding, then runs one forward pass on that shortlist.
* **Soft distribution matching:** On typed-decisions, while Laya achieves higher argmax accuracy (0.766 vs 0.727), Jev achieves higher soft accuracy (0.580 vs 0.471) against the teacher's full probability distributions.
* **Out-of-the-box raw calibration:** Before temperature scaling, the base checkpoint has higher raw ECE (0.213 vs 0.144). Laya achieves its 0.081 ECE after domain temperature fitting.

Full detail, including every workflow and all 51 languages: **[`BENCHMARKS.md`](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)**.

### typed-decisions, measured on all three checkpoints

400 cases, 2,000 decisions, four workflows.

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| **`laya-typed-decisions`** | **0.766** | 0.471 | **0.062** | 0.213 | **0.242** |
| `laya` | 0.362 | 0.331 | 0.315 | 0.174 | 0.694 |
| `laya-multilingual` | 0.352 | 0.328 | 0.463 | 0.314 | 0.760 |
| *Jev 1.13.0 (published)* | *0.727* | *0.580* | *0.148* | *0.144* | *0.391* |
| *teacher self-agreement ceiling* | *0.735* | | | | |
| *per-question majority class* | *0.461* | | | | |
| *random guess* | *0.318* | | | | |

The fine-tuned checkpoint beats Jev by 3.9 points and clears the teacher ceiling, with 2.4x
better Brier and 1.6x better score MAE. It wins on all four workflows: invoice processing
0.804, security incidents 0.766, customer service 0.764, agent-trace observability 0.730.
By primitive: `noul` 0.857, `choice` 0.733, `score` 0.723.

Two places it still trails Jev: **soft accuracy** (0.471 vs 0.580 — its argmax is better but
its distributions match the teacher less well) and **ECE** (0.213 vs 0.144), which temperature
fitting addresses.

**The base checkpoints sit below the majority-class baseline** (0.362 and 0.352 against 0.461).
All of the capability on this benchmark comes from fine-tuning. Base rows match the committed runs (`suites.typed_decisions` in `research/results/t4_colab_benchmark.json`: 0.3620 / 0.3515; CPU sweep `part_b`: 0.3615 for English). The `laya-typed-decisions` row and its four per-workflow scores are from the fine-tuning run and have no committed result file behind them yet.

### Multilingual (51 languages, MASSIVE intent, 20 options, random = 0.050)

| | `laya` | `laya-multilingual` |
|---|---|---|
| English | **0.783** | 0.657 |
| 13 other languages | 0.306 | **0.451** |
| XNLI, English | **0.860** | 0.843 |
| XNLI, 14 other languages | 0.521 | **0.731** |

Across all 51 languages the English checkpoint macro-averages **0.227** with macro ECE
**0.733** *(0.571 as served, after the temperature clamp)*, and only 23 of 51 languages clear
3x random. `laya-multilingual` reaches **0.401** on the same 5,100 cases, clearing 3x random in
48 of 51 — see [BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) for why those multilingual numbers are a re-run
rather than the committed file. Khmer scores **0.000 at 95.2% confidence** raw, **70.5%** as
served, on the English checkpoint. This is why
[`Router`](#quickstart-route-mode-recommended) exists: the model's own
confidence gives no warning, so the routing decision has to be made before the forward pass.

### English tasks

| task | `laya` | `laya-multilingual` | note |
|---|---|---|---|
| AG News | **0.947** | 0.937 | in training mix |
| BoolQ | **0.830** | 0.787 | in training mix |
| DAIR Emotion | **0.573** | 0.513 | held out |
| prompt-injections | **0.698** | 0.578 | held out, n=116 |
| SST-5 (ordinal) | 0.372 | 0.282 | held out |

### Calibration

Both checkpoints are over-confident as shipped. Refitting one temperature per (question type,
option count) on held-out data moves mean ECE **0.466 -> 0.081** (`laya`) and
**0.314 -> 0.106** (`laya-multilingual`). `laya-multilingual` ships with no fitted
temperatures at all, so fit them before relying on its probabilities.

At checkpoint load, numeric temperature entries are clamped to `[0.5, 5.0]`; invalid or
non-finite entries use the neutral fallback `1.0`. A runtime warning reports the affected
entries and applied values. Bucket-specific temperatures still take precedence over per-type
values, including when a bucket uses the fallback. Raw values remain available in
`agent.temperature_raw` and `agent.temperature_by_options_raw`. A fallback prevents a loading
failure; it does not establish calibrated confidence.

### Honest limits

* **The base checkpoints are near chance on typed-decisions zero-shot** -- 0.362 and 0.352
  against a 0.318 random baseline and a 0.461 majority-class baseline. The 0.766 figure comes
  from the checkpoint fine-tuned on that benchmark's own training split. Laya is a fast base to
  specialise, not a zero-shot decision engine.
* **Avoid boolean-word labels in `choice` questions.** Choice keys are rendered verbatim, and the
  current checkpoints can follow labels such as `true`/`false` or `yes`/`no` instead of the option
  descriptions. Use semantic labels or opaque labels such as `A`/`B`, and validate them on the
  checkpoint and states you serve.
* **Semantic `choice` labels do not make negation safe.** In the five cancellation examples from
  [#377](https://github.com/NandhaKishorM/laya/issues/377), a CPU run on Laya 0.3.20 with
  `no_action` / `cancel_account` keys selected `cancel_account` for all four negated requests on
  `laya` and two on `laya-multilingual`; one multilingual answer assigned it probability `0.9998`.
  The positive control passed on both checkpoints. These are narrow cancellation examples, not
  evidence that every negated state fails. Validate the exact checkpoint and wording you serve;
  using semantic keys alone does not avoid this failure.
* **High-cardinality choice questions and token budgets:** Sequences split into an option prompt budget (`head_max_len`) and the remaining document/state budget. **`head_max_len` is a cap, not the head length.** The renderer fills the option prompt budget only as far as the question and its option descriptions need, and `build_sequence` then sizes the state from the head it *actually built* (`room = max_len - head_len - 1`). The real room therefore depends on the question, not on the cap:
  * `laya` (English) defaults to 512 context (`head_max_len = 192`). The shipped department question above renders a **48-token** head, leaving **463 tokens for state**, not 320.
  * `laya-multilingual` and `laya-typed-decisions` default to 1,024 context (`head_max_len = 256`). The shipped question renders a **45-token** head, leaving **978 tokens for state**, not 768 (mmBERT-base encoder supports up to 8,192 with RoPE).
  `max_len - head_max_len` is **not a safe figure in either direction**, so size a state to `max_len - head_len - 1` for the question you are actually asking:
  * For a question that does not fill the cap it **understates** the room, as above — conservative, but it discards state the model would have read.
  * For a high-cardinality question it **overstates** it. `per = max(4, (head_max_len - 16) // k)` floors at 4 tokens per option and `head_ids` keeps a floor of 8, so past roughly `head_max_len / 4` options the head grows *beyond* the cap. A state sized to the cap-based figure is then **truncated**, not merely conservative:

    | | `k` options | head | real room | cap-based figure |
    |---|---|---|---|---|
    | `laya`, `max_len=512` | 77 | 319 | **192** | 320 |
    | `laya`, `max_len=512` | 100 | 411 | **100** | 320 |
    | `laya-multilingual`, `max_len=1024` | 100 | 411 | **612** | 768 |

    The crossover is at `k ≈ head_max_len / 4` and the gap widens from there; `MAX_CHOICE_OPTIONS = 100` in `laya/serve.py` puts the overstating regime within reach over HTTP.

  At default settings, a 77-option question like Banking77 allocates only `(256 - 16) // 77` ≈ 3–4 tokens per label, which causes accuracy to fall off sharply (0.425 vs Jev's 0.870). If evaluating 50+ options in a single question:
  1. Raise `agent.cfg["head_max_len"] = 512` and `agent.cfg["max_len"] = 1024` (or up to 2048 / 4096 / 8192) so every option has enough tokens to remain distinct. Both are also per-request: `predict(state, questions, head_max_len=512, max_len=1024)` widens one question without changing the agent for everyone else, and every LangChain node takes the same two arguments ([LangChain guide](https://github.com/NandhaKishorM/laya/blob/main/docs/langchain.md)). `laya --questions` takes the same two budgets as `--max-len` / `--head-max-len`.
  2. Or shortlist with embeddings and run one forward pass on the top `k` labels (`predict_shortlist`, example below). `predict` and `system_one` still score every criterion they are given.
  3. Or split the label set yourself into a coarse question and a fine question.
  4. Or let the decision model narrow the set itself: `laya.predict_tournament(agent, state, questions)` answers the labels in groups of at most 16, every group in the same forward pass, then asks the question again over the group winners, so up to 256 labels take two `predict` calls and no embedder. On the full test splits with the English checkpoint it took BANKING77 (77 labels) from 0.430 to 0.610, CLINC150 (150) from 0.625 to 0.876 and MASSIVE intent (60) from 0.515 to 0.569, with ECE between 0.06 and 0.14 instead of 0.30 to 0.37, at about twice the latency of one question ([`research/benchmarks/tournament`](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/tournament/README.md)). When the labels fit uncut in a raised budget, as MASSIVE's 60 do at `head_max_len=512`, raising it did better (0.622 against 0.590 on 500 rows). Probabilities on a tournament choice are over its finalists.

```python
import laya

questions = {
    "intent": {
        "type": "choice",
        "instructions": "Which banking intent is this?",
        "criteria": {
            "card_arrival": "where is my card",
            "transfer_fee": "fee charged on a transfer",
            # ...the rest of a large label set
        },
    }
}
result = laya.predict_shortlist(
    agent,
    {"text": "I was charged twice for a transfer"},
    questions,
    embed_fn=laya.embed_fn_from_agent(agent),  # or any callable: texts -> (n, dim)
    k=20,
)
result["shortlist"]["intent"]["labels"]  # the top 20 labels sent to the model
```

`embed_fn(texts)` returns one vector per string. `embed_fn_from_agent` mean-pools the encoder already loaded on the agent; the decision head runs in the following `predict` / `system_one` call. Probabilities on a shortlisted choice are over those `k` labels. When `k` is at least the number of labels, the original question is passed through and `embed_fn` is not called. `laya.shortlist_choice` returns bare labels; pass `return_scores=True` for the `(labels, scores)` pair in rank order (`None` when nothing was dropped), the same cosines `predict_shortlist` reports in its `shortlist` metadata.

Shortlisting the same option set on every request re-embeds option texts that do not change. Wrap the embedder once with `laya.cached_embed_fn(embed_fn)` and repeat calls embed only the new query text: lookups are exact string matches into an LRU of at most 4,096 entries (about `maxsize * dim * 4` bytes, so ~12 MB at the default with a 768-dim encoder), and texts missing from the cache are still embedded in one batched call. The wrapper's `cache_info()` reports hits and misses; call `cache_clear()` if the model behind `embed_fn` changes.

[Issue #102](https://github.com/NandhaKishorM/laya/issues/102) reports that a top-20 zero-shot shortlist moved a BANKING77 run from 54.3% to 60.8% on the reporter's setup. Those figures are the reporter's; this repository has not remeasured them.

* Ordinal `score` questions are the weakest primitive (SST-5 0.372).
* **`noul` can follow its option labels instead of the state, most strongly on `laya` (English).** `noul` renders its two options as `false:` / `true:` by default, and on the English checkpoint that label pair can dominate the answer, returning a confident "no" for clearly positive input (#156). Until a retrained checkpoint lands, check `noul` answers on your own data. You can override the model-facing pair while keeping the `noul` result as P(true):

  ```python
  {"type": "noul", "instructions": "Is this review positive?",
   "criteria": {"true": "the review is positive", "false": "the review is negative"},
   "labels": {"true": "A", "false": "B"}}
  ```

  A `noul` with no `criteria` renders one generic option pair for every state (`no, the statement
  does not hold` / `yes, the statement holds`). On `laya` that pair carries the whole decision, so
  it answers "no" whatever the state — give a `noul` criteria if you need it to discriminate. On
  `laya-multilingual` the criteria-less form does read the state.

  Label sensitivity varies by checkpoint and state, so validate the override on your own data. A
  two-option `choice` with neutral keys remains another workaround:

  ```python
  {"type": "choice", "instructions": "Is this review positive?",
   "criteria": {"A": "yes, the review is positive", "B": "no, the review is negative"}}
  ```

  `criteria` on a `noul` must be keyed `true`/`false` — those two keys *are* the option text the
  model reads, so any other key is rejected instead of being quietly replaced with the defaults.
  Before that check, `criteria: {"yes": ..., "no": ...}` was accepted, dropped, and answered
  against `false:` / `true:` anyway, which cost 2 of 3 clearly positive reviews on the English
  checkpoint ([#156](https://github.com/NandhaKishorM/laya/issues/156)). Use `labels` as above to
  change the wording without touching the option text.
* **`laya-multilingual` has a position bias on `score` questions** (#131): it rarely picks the first-listed level, in any language. For English score questions, route to `model="english"`, and for other languages validate score outputs on your own data before relying on them.
* **`action.act_probability` carries no usable signal yet** (#185). It reads 1.0 for almost every input, and its raw logits run against correctness (AUROC 0.30 on 396 labelled decisions). Gate on `confidence` instead, which reaches an AUROC of 0.77 on the same items.
* `laya` collapses outside English; `laya-multilingual` is weaker on English. Route, or pick
  deliberately.

---

## Community Tools

* **[omp-laya-judge](https://github.com/F0Rextasy/omp-laya-judge)**: an [oh-my-pi](https://github.com/can1357/oh-my-pi) plugin with a local System-1 judge MCP server and skill (`choice`/`bool`/`score`, 0 tokens, about 0.3 s on CPU), confidence-gated escalation, and reproducible quiz and Snake demos.
* **[laya-adk-toolkit](https://github.com/Ashfaqbs/laya-adk-toolkit)**: [Google ADK](https://google.github.io/adk-docs/) tools that let an agent call Laya's `classify`/`score`/`detect` typed decisions directly as tools, instead of asking an LLM to guess at structured output.
* **[laya-Ascend](https://github.com/zzhdbw/laya-Ascend)**: Laya on Huawei Ascend NPUs through `torch-npu`, with a CPU vs NPU benchmark (34x to 71x faster at batch size 1), a setup guide, and Snake and Tetris demos.
* **[laya-apple](https://github.com/tc3oliver/laya-apple)**: a correctness-validated Laya runtime for Apple silicon that uses the MLX GPU and the Apple Neural Engine, with automatic routing and concurrent heterogeneous serving.
* **[stuntd](https://github.com/bladedevoff/stuntd)**: runs Laya locally behind the Jev API (`POST /v1/systemone`, no key) and trains a head per decision on the frozen encoder from your own labelled rows, with a calibrated confidence threshold (a 12-label intent task: 89.5% zero-shot to 100% trained).

---

## Live Demo & Resources

* **Hugging Face Model:** [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
* **Interactive Web Demo:** [convaiinnovations/laya-demo](https://huggingface.co/spaces/convaiinnovations/laya-demo)
* **Engineering Writeup:** [Read the full story on Dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)

---

## Fine-Tuning

Fine-tune Laya on your own domain data. Two training options are provided:

* **[`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)**: runs on Kaggle's free 2xT4 GPUs (DDP) and does the whole loop: build the dataset, train with RLCD (proper-scoring-rule rewards, GRPO-style policy gradient), fit calibration temperatures, evaluate, and push the result to the Hub.
* **[`notebooks/laya_finetune_typed_decisions_mps.py`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_mps.py)**: standalone training script for Apple Silicon MacBooks (PyTorch MPS with CPU fallback). Replaces DDP with single-process training and configurable micro-batches and gradient accumulation.

For Apple Silicon (e.g. 16 GB MacBook), fine-tuning runs locally with unified memory:

```bash
python notebooks/laya_finetune_typed_decisions_mps.py \
  --micro-batch 1 \
  --grad-accum 32
```

The notebook and script enable gradient checkpointing on both the encoder and the decision head.
For custom training loops, `model.head_checkpointing = True` enables activation
checkpointing for the decision-head layers; enable the encoder's gradient checkpointing
separately. During gradient-enabled training, this reduces stored intermediate activations
by recomputing them during backward, trading extra computation for lower activation memory.
The head flag defaults to `False` and is bypassed in evaluation and under `torch.no_grad()`.

The notebook fits one `temperature` per type (`choice`, `score`, `noul`) and removes inherited
`temperature_by_options` from the exported config. Otherwise those old bucket values take
precedence at inference and silently mask the new fit. Existing checkpoints still honor
intentional bucket-specific temperatures, falling back to the corresponding per-type value
when a bucket is absent; the runtime's temperature clamp is unchanged.

This fixes configuration persistence, not measured model accuracy or calibration quality.
The notebook's calibration samples come from its training items; evaluate on separate held-out
data before claiming an improvement. Already published checkpoints are not rewritten.
Run the CPU-only regression checks with `python tests/test_calibration_persistence.py`
(synthetic configs and tiny local fixtures; no pretrained downloads or training).

Fine-tuning is where most of the value is. On the typed-decisions benchmark the base
checkpoints score near chance zero-shot (0.36 and 0.35 against a 0.318 random baseline),
while the fine-tuned checkpoint reaches **0.766** on the same 2,000 decisions -- above
TypeSafe Jev's published 0.727 and above the 0.735 teacher self-agreement ceiling. Treat Laya
as a fast base to specialise, not as a zero-shot decision engine.

Runtime on 2xT4 is roughly 4-5 hours for 4 epochs over ~30k questions.

### Worked example: a browser-agent decision head

[`docs/finetune_browser_agent.md`](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md) records a complete specialisation
on a single 16 GB GPU with no paid API: Laya as the operation/target decider for
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (same request format as
TypeSafe Jev). Element top-1 among ~45 candidates goes from 0.10 zero-shot to 0.66, real-task
success from 0 % to 62 % at 17-23 ms per step; weights, pipeline code and per-run results are on
the Hub at [cklxx/laya-browser](https://huggingface.co/cklxx/laya-browser). The write-up covers the
data recipe (reverse-generated goals, executed DONE states, Mind2Web, on-policy corrections), the
input-format change that mattered most, and the things that did not work.

---

## Support the Project

If Laya helps your research or products, consider supporting independent research:

<p align="left">
  <a href="https://www.buymeacoffee.com/nandakishorm" target="_blank">
    <img src="https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=nandakishorm&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" alt="Buy Me A Coffee" />
  </a>
</p>

---

## License

Apache 2.0. Developed by Convai Innovations.
