# Laya Models: Obtaining the ONNX Artifacts

The Laya .NET SDK ([README](https://github.com/NandhaKishorM/laya/blob/main/laya-dotnet/README.md)), a .NET port of the Laya Python SDK, runs
**ONNX** models through ONNX Runtime, not PyTorch weights, so no Python process is needed at runtime. This
guide explains how to get those ONNX artifacts: either convert them yourself
from the published PyTorch checkpoints (works today), or let the SDK download
them once they are published.

**Quick path:** create a venv, `pip install -e .` plus the exporter's
dependencies, run `python laya-dotnet/tools/export_onnx.py --model all`, and set `LAYA_ONNX_ROOT` to the
resulting `onnx/` folder. [Section 1](#1-export-the-onnx-model-step-by-step)
walks through each step.

The ONNX export is not on Hugging Face yet, so exporting it yourself is the
only way to get the artifacts today. Downloading
([section 6](#6-downloading-future-once-published)) will work without code
changes once they are published.

---

## 1. Export the ONNX model (step by step)

You need Python 3.9+ and about 2.5 GB free per checkpoint (6.7 GB for all
three, counting the Hugging Face download cache; see step 3). Run everything
from the **repository root** unless a step says otherwise.

### Step 1: Create a virtual environment and install

The dynamo-based ONNX exporter requires `torch >= 2.9`.
Tested versions (from the repo's `.venv`):

| Package | Tested version |
|---|---|
| `torch` | 2.14.0 |
| `onnx` | 1.23.0 |
| `onnxscript` | 0.7.2 |
| `onnxruntime` | 1.30.0 |
| `transformers` | 5.17.0 |
| `safetensors` | 0.8.0 |
| `huggingface_hub` | 1.32.0 |
| `numpy` | 2.5.3 |

The `laya` Python package itself must be importable, so the commands below
also install the repository with `pip install -e .`.

```powershell
# PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install "torch>=2.9" onnx onnxscript onnxruntime transformers safetensors huggingface_hub numpy
pip install -e .
```

```bash
# Bash / WSL
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.9" onnx onnxscript onnxruntime transformers safetensors huggingface_hub numpy
pip install -e .
```

### Step 2 (optional): Hugging Face authentication

The `convaiinnovations/laya` checkpoints are public and need no token. For
gated or private repos, supply a token:

```powershell
$env:HF_TOKEN = "hf_..."          # PowerShell — persists for the session
```

```bash
export HF_TOKEN="hf_..."           # Bash
```

Or pass `--token hf_...` directly to the script.

### Step 3 (optional): Disk space and HF\_HOME

The exporter calls `huggingface_hub.snapshot_download` to fetch the PyTorch
weights. If the default HuggingFace cache (`%LOCALAPPDATA%\HuggingFace` on
Windows, `~/.cache/huggingface` elsewhere) is on a volume without enough
space, redirect it with `HF_HOME`.

| Checkpoint | PyTorch weights (HF cache) | ONNX output | Total |
|---|---|---|---|
| `multilingual` | ~644 MB | ~1.2 GB | ~1.9 GB |
| `english` | ~842 MB | ~1.6 GB | ~2.5 GB |
| `typed-decisions` | ~842 MB | ~1.6 GB | ~2.5 GB |
| all three | ~2.3 GB | ~4.4 GB | ~6.7 GB |

```powershell
$env:HF_HOME = "C:\hf_cache"      # PowerShell
```

```bash
export HF_HOME=/d/hf_cache         # Bash
```

### Step 4: Export

The script downloads the PyTorch checkpoint from `convaiinnovations/laya`
(public, no token needed) and writes `onnx/<checkpoint>/` under the repository
root. The multilingual checkpoint takes about 9 minutes on CPU.

```powershell
# Export the multilingual checkpoint to onnx/multilingual/  (~9 min on CPU)
python laya-dotnet/tools/export_onnx.py --model multilingual

# Export and verify ONNX parity against PyTorch (~9 min + ~5 min verify)
python laya-dotnet/tools/export_onnx.py --model multilingual --verify

# Export typed-decisions with parity check (~30 min total)
python laya-dotnet/tools/export_onnx.py --model typed-decisions --verify

# Export all three checkpoints
python laya-dotnet/tools/export_onnx.py --model all

# Export to a custom directory (no --force needed; leaves onnx/ untouched)
python laya-dotnet/tools/export_onnx.py --model multilingual --out C:\tmp\laya-test --verify

# Re-export, overwriting an existing artifact
python laya-dotnet/tools/export_onnx.py --model multilingual --force

# Export a local checkpoint directory (must contain rl_agent_config.json)
python laya-dotnet/tools/export_onnx.py C:\path\to\checkpoint --out onnx\my-model
```

The script refuses to overwrite `<out>/<name>/model.onnx` if it already exists.
Pass `--force` to overwrite. The check happens before any download, so a
refusal wastes no time or bandwidth.

The `--verify` flag runs a self-check against the `fixtures.npz` reference
outputs and a 36-combination shape sweep (`6 seq_lens × 2 batch_sizes × 3
marker_counts`). Skip it if you just need the files quickly; run it before
publishing.

### Step 5: Point the SDK at the export

The quickest way — no code changes.

**Single checkpoint (legacy, multilingual only):**

```powershell
# PowerShell
$env:LAYA_ONNX_DIR = "C:\models\laya\multilingual"
```

```bash
# Bash
export LAYA_ONNX_DIR=/c/models/laya/multilingual
```

**All three checkpoints at once:**

If you export all three under a common parent directory (e.g. `onnx/multilingual`,
`onnx/english`, `onnx/typed-decisions`), point `LAYA_ONNX_ROOT` at the parent:

```powershell
# PowerShell
$env:LAYA_ONNX_ROOT = "C:\models\laya"
```

```bash
# Bash
export LAYA_ONNX_ROOT=/c/models/laya
```

The SDK appends the checkpoint subdirectory name automatically
(`multilingual`, `english`, or `typed-decisions`).

Use absolute paths. Relative paths depend on the working directory, which
is not reliable across test runners and hosts.

Or pass the checkpoint folder in code with `LayaOptions.ModelDirectory` /
`LayaEngine.FromDirectory(...)` (see [section 7](#7-using-the-artifact-in-code-and-tests)).

### Step 6: Check it works

Run the quickstart sample against the export, from the repository root:

```bash
# Multilingual (default)
dotnet run --project laya-dotnet/samples/Laya.Sample -- --model-dir onnx/multilingual

# English checkpoint
dotnet run --project laya-dotnet/samples/Laya.Sample -- --checkpoint english --model-dir onnx/english

# Typed-decisions checkpoint
dotnet run --project laya-dotnet/samples/Laya.Sample -- --checkpoint typed-decisions --model-dir onnx/typed-decisions
```

---

## 2. The three checkpoints

All three are published under the bundle repo `convaiinnovations/laya`; the
exporter downloads from there by default.

| Name | Bundle subfolder | Encoder | Params | max\_len | head\_max\_len | model.onnx | model.onnx.data |
|---|---|---|---|---|---|---|---|
| `english` | (root) | `answerdotai/ModernBERT-large` | 421 M | 512 | 192 | 3.2 MB | 1.57 GB |
| `multilingual` | `multilingual` | `jhu-clsp/mmBERT-base` | 322 M | 1024 | 256 | 2.5 MB | 1.20 GB |
| `typed-decisions` | `typed-decisions` | `answerdotai/ModernBERT-large` | 421 M | 1024 | 256 | 2.6 MB | 1.57 GB |

Each checkpoint also has a standalone repo (`convaiinnovations/laya-multilingual`,
etc.) that mirrors the same weights.

---

## 3. SDK support status

**All three checkpoints are fully supported by the .NET SDK.**

`HfTokenizer` resolves special-token ids from `tokenizer_config.json` +
`tokenizer.json` at load time, so it works for both the mmBERT-style tokens
(`<pad>=0`, `<bos>=2`, `<eos>=1`, `<unk>=3`, `<mask>=4`) and the ModernBERT
ByteLevel-BPE tokens (`[PAD]=50283`, `[CLS]=50281`, `[SEP]=50282`,
`[UNK]=50280`, `[MASK]=50284`). Select a checkpoint with `LayaOptions.Checkpoint`
or point `LayaOptions.ModelDirectory` / `LAYA_ONNX_DIR` at the artifact
directory directly.

```csharp
// explicit checkpoint selection
using var engine = LayaEngine.Create(new LayaOptions
{
    Checkpoint = LayaCheckpoint.English,
    AllowDownload = true,
});

// or point directly at the directory
using var engine = LayaEngine.FromDirectory(@"C:\models\laya\english");
```

---

## 4. Output layout

`export_onnx.py` (this SDK's exporter) writes the **fused** layout below to `<out>/<name>/`:

```
<out>/<name>/
    model.onnx              # ONNX graph (~2.5–3.2 MB, opset 18)
    model.onnx.data         # External weight sidecar (~1.2–1.6 GB)
    rl_agent_config.json    # Sequence lengths and temperature calibration
    tokenizer/
        tokenizer.json      # HuggingFace tokenizer (post_processor kept intact)
        tokenizer_config.json
    fixtures.npz            # PyTorch reference outputs (used by --verify; not needed by SDK)
```

**`model.onnx` and `model.onnx.data` must stay in the same directory.** ONNX
Runtime resolves the external-data file relative to the graph file, which is
also why the API takes a *directory* rather than a file. Moving `model.onnx`
alone produces a baffling "missing initializer" error.

**Tokenizer post\_processor:** `tokenizer.json` retains its `TemplateProcessing`
post\_processor. The SDK strips it at load time via
`HfTokenizer.EnsureStrippedTokenizer` and caches the result as
`tokenizer.nopost.json` alongside the original. That cache file is written by
the SDK, not by the exporter.

### Split layout (laya-ts exporter)

The `laya-ts` exporter produces a second, **split** layout instead of one fused graph — the
encoder and the RL head as two separate ONNX files, each with its own external-data sidecar:

```
<out>/<name>/
    encoder.onnx            # Encoder graph: (input_ids, attention_mask) -> last_hidden_state
    encoder.onnx.data       # Encoder weights (~1.2-1.6 GB)
    head.onnx               # RL head graph: (hidden_states, marker_pos, marker_mask,
                            #   qtype [B, 1], attention_mask) -> logits, act_logits
    head.onnx.data          # Head weights (~60-110 MB)
    rl_agent_config.json    # Same shape/config file as the fused layout
    tokenizer.json          # Sits directly in the directory, not under tokenizer/
```

Each `.data` file must stay next to its graph, for the same reason as `model.onnx.data`.

Note the split layout's `tokenizer.json` has no sibling `tokenizer_config.json` and no
`tokenizer/` subfolder. When `tokenizer_config.json` is absent, `HfTokenizer` takes CLS and SEP
from `tokenizer.json`'s own `post_processor` template (`[CLS] $A [SEP]`, or `<bos> $A <eos>` for
multilingual) and resolves PAD, MASK and UNK from `added_tokens`' conventional spellings
(`[PAD]`/`[MASK]`/`[UNK]` or `<pad>`/`<mask>`/`<unk>`). A spelling guess alone is not safe for
CLS/SEP: the multilingual vocabulary also contains `<s>`/`</s>`, which are not its CLS/SEP.

The .NET SDK auto-detects which layout a directory holds (see [section 5](#5-where-the-engine-looks));
callers do not choose a layout explicitly. If a directory somehow has both `model.onnx` and
`encoder.onnx`+`head.onnx`, the fused layout wins, since that is what this SDK's own exporter has
always produced.

In both layouts, `rl_agent_config.json` is where `MaxLen` (`max_len`) and `HeadMaxLen`
(`head_max_len`) come from — `LayaConfig.Load` reads the same file regardless of which ONNX
layout sits beside it, so a split export needs the same config file the fused one does.

---

## 5. Where the engine looks

The engine looks for the directory in this order:

1. `LayaOptions.ModelDirectory`
2. the `LAYA_ONNX_DIR` environment variable — maps to the **multilingual** checkpoint only (legacy)
3. `LAYA_ONNX_ROOT/<checkpoint-name>` — the recommended multi-checkpoint path; the SDK appends the
   subfolder name (`multilingual`, `english`, or `typed-decisions`)
4. the local cache, `<CacheDirectory>/<checkpoint-name>`. `CacheDirectory` defaults to
   `%LOCALAPPDATA%\laya\onnx` on Windows and `~/.cache/laya/onnx` elsewhere.
5. a download into that cache, only if `LayaOptions.AllowDownload = true`

If none of these has the artifact, resolution fails with a message that lists every location it
tried.

Once a directory is chosen, `ModelArtifacts.Resolve` decides fused vs. split by checking which
files are actually present, not by any option or environment variable:

- `model.onnx` + `model.onnx.data` → fused (`LayaArtifactLayout.Fused`)
- `encoder.onnx` + `head.onnx` → split (`LayaArtifactLayout.Split`)
- both present → fused wins
- neither present, or only one half of a layout (e.g. `encoder.onnx` with no `head.onnx`) →
  `FileNotFoundException` listing all four filenames, so a partial/interrupted export fails with
  the same message as an empty directory rather than a confusing null-session error later.

The tokenizer path is resolved the same way in both layouts: `tokenizer/tokenizer.json` (fused)
is tried before the root-level `tokenizer.json` (split); if a directory happens to have both, the
nested one wins.

---

## 6. Downloading (future, once published)

**The ONNX artifacts have not been uploaded to HuggingFace yet.** The
`convaiinnovations/laya` repository ships PyTorch weights
(`model.safetensors`), which the SDK cannot load. Until the ONNX files are
uploaded, `AllowDownload = true` fails immediately with a 404 and an explanatory
message.

Downloading is **opt-in**. Nothing touches the network unless you ask for it.
Once the files are published (see section 8), enabling the download requires only:

```csharp
using var engine = LayaEngine.Create(new LayaOptions
{
    AllowDownload = true,                  // fetch from Hugging Face into the cache, once
    HuggingFaceToken = null,               // null falls back to HF_TOKEN; "" disables auth
    DownloadProgress = new Progress<ArtifactDownloadProgress>(p =>
        Console.Error.WriteLine($"{p.FileName}: {p.BytesReceived}/{p.TotalBytes}")),
});
```

The SDK downloads five files into the per-user cache and skips the download on
subsequent runs if the directory is already complete:

- Windows: `%LOCALAPPDATA%\laya\onnx\multilingual`
- Linux / macOS: `~/.cache/laya/onnx/multilingual`

Override the cache root with `LayaOptions.CacheDirectory`.

---

## 7. Using the artifact in code and tests

### Create the engine in code

```csharp
// From a known directory (all checkpoints work)
using var engine = LayaEngine.FromDirectory(@"C:\models\laya\multilingual");

// English checkpoint via LayaOptions.Checkpoint
using var engine = LayaEngine.Create(new LayaOptions
{
    Checkpoint = LayaCheckpoint.English,
    ModelDirectory = @"C:\models\laya\english",
    IntraOpThreads = 4,   // pin in containers with a CPU limit
});

// Multilingual with thread config, no ModelDirectory (LAYA_ONNX_DIR / LAYA_ONNX_ROOT / cache)
using var engine = LayaEngine.Create(new LayaOptions
{
    IntraOpThreads = 4,
});
```

### Run the tests

The test suite is parameterized over all three checkpoints. Set `LAYA_ONNX_ROOT`
to the parent of the three checkpoint directories:

```powershell
# PowerShell — all three checkpoints
$env:LAYA_ONNX_ROOT = "C:\models\laya"
dotnet test --solution laya-dotnet/Laya.slnx
```

```bash
# Bash — all three checkpoints
LAYA_ONNX_ROOT=/c/models/laya dotnet test --solution laya-dotnet/Laya.slnx
```

The legacy `LAYA_ONNX_DIR` still works and maps to the multilingual checkpoint
only; the other two are found via walk-up or skipped gracefully.

Checkpoints whose artifacts are missing are skipped, not failed. The 646 tests that need no
ONNX artifact at all (Tier 1, plus the layout-detection and tokenizer-fallback fixtures) always
run; a much larger number of theory rows additionally run once `onnx/<checkpoint>` (fused) and/or
`onnx-split/<checkpoint>` (split) are present, since most of Tier 2 is parameterized per checkpoint,
per shape, and now per layout. That theory-row count is not a fixed constant — it grows every time
a checkpoint's golden-answer coverage or shape sweep grows, and again once split-layout artifacts
exist for a checkpoint — so treat any specific figure quoted here as a snapshot, not a target.
As of 0.1.0 the full count (all three checkpoints, both layouts, everything present) is 1368.
Re-derive it yourself with `dotnet test --solution laya-dotnet/Laya.slnx -c Release` (all
artifacts present) or with `-class-` exclusions for the model-backed classes (see
[README.md § Tests](https://github.com/NandhaKishorM/laya/blob/main/laya-dotnet/README.md#tests)) if you need the true count without loading any weights.

---

## 8. Publishing to HuggingFace (maintainers only)

**Requires write access to `convaiinnovations/laya`.**

The SDK downloader fetches five files per checkpoint from `convaiinnovations/laya`.
A maintainer must upload the following paths for each checkpoint:

| Checkpoint | HF path in repo |
|---|---|
| `multilingual` | `multilingual/<files>` |
| `english` | `<files>` (bundle root — no subfolder) |
| `typed-decisions` | `typed-decisions/<files>` |

The five files per checkpoint are:
`model.onnx`, `model.onnx.data`, `rl_agent_config.json`,
`tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`.

`onnx/` is gitignored (the weights exceed GitHub's 100 MB hard limit), so
upload must go through the HuggingFace Hub CLI or Python API directly.

Run from the repo root after completing `--verify`:

```bash
# huggingface-cli (pip install huggingface_hub)
huggingface-cli upload convaiinnovations/laya onnx/multilingual multilingual \
    --repo-type model --token hf_...
huggingface-cli upload convaiinnovations/laya onnx/english . \
    --repo-type model --token hf_...
huggingface-cli upload convaiinnovations/laya onnx/typed-decisions typed-decisions \
    --repo-type model --token hf_...
```

Once these files are live, `AllowDownload = true` with the matching checkpoint
works without any code changes.

The downloader handles either layout. It first sends a `HEAD` request for `model.onnx`; if that
exists it fetches the fused files above. Otherwise, if `encoder.onnx` and `head.onnx` both exist,
it fetches the split files (`encoder.onnx`, `head.onnx`, their `.data` sidecars when present,
`rl_agent_config.json` and `tokenizer.json`; see [section 4](#4-output-layout)). So whichever
layout gets uploaded to Hugging Face, `AllowDownload = true` works without code changes.

---

## 9. Troubleshooting

**404 on download (`AllowDownload`)**
The ONNX artifacts have not been published yet (see section 8). Export locally with
`laya-dotnet/tools/export_onnx.py` and point the SDK at the result via `LAYA_ONNX_DIR` or
`LayaOptions.ModelDirectory`.

**Low disk space during export**
Set `HF_HOME` to a volume with enough free space before running the exporter
(see the disk-space table in [section 1, step 3](#step-3-optional-disk-space-and-hf_home)). `--out` controls where the ONNX output lands.

**`model.onnx.data` not found / "missing initializer" in ONNX Runtime**
The sidecar must sit beside `model.onnx`. Copy both files together; the SDK
error message already reminds you of this: *"both model.onnx and model.onnx.data
must reside in the same directory."*

**"already exists, pass --force"**
`<out>/<name>/model.onnx` is present from a previous run. Use `--force` to
overwrite in place, or use `--out` to write to a separate directory.

**Wrong-checkpoint error (tokenizer/config mismatch)**
`tokenizer.json` and `tokenizer_config.json` are from different checkpoints
(e.g., the tokenizer from `english` with the config from `multilingual`). The
error message names the missing token. Keep the two files together as exported.

**"neither model.onnx nor encoder.onnx/head.onnx found" (or the message names only three of the
four filenames it tried)**
The directory has neither a complete fused export nor a complete split export — most often a
split export that only partly copied (`encoder.onnx` present, `head.onnx` missing, or vice
versa). `ModelArtifacts.Resolve` treats a half-split directory the same as an empty one rather
than guessing; copy the missing file, or re-run whichever exporter produced the directory.

**Split-layout tokenizer error naming a token role (e.g. "cls_token") instead of a filename**
The split layout has no `tokenizer_config.json`, so `HfTokenizer` reads CLS/SEP from the
`post_processor` template and the other roles from `added_tokens`' spelling instead (see
[section 4](#4-output-layout)). This error means the vocabulary uses a spelling the fallback does
not recognise; either add a `tokenizer_config.json` beside `tokenizer.json` (the fused layout's
approach), or update the fallback if this is a spelling worth supporting generally.

**Windows console encoding during export**
The dynamo exporter logs Unicode emoji (✅, etc.) that crash a non-UTF-8 Windows
console. The script reconfigures `stdout`/`stderr` to UTF-8 with replacement
automatically. Any unrecognised emoji appears as `?`; the export continues
normally.
