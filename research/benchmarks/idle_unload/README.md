# Idle unload and shared-server MCP measurements

This script uses a checkpoint already on disk, sets `HF_HUB_OFFLINE=1`, and runs real MPS
inference. It needs the `serve` and `mcp` extras on an Apple Silicon Mac. Memory is in MiB;
process RSS comes from `ps`, Metal values from torch. Run every mode in a fresh process.

Set `CHECKPOINT` to a local English checkpoint containing `model.safetensors`, `encoder/`, and
`tokenizer/`. The default idle window in these measurements is two seconds so the check is quick.

```bash
python research/benchmarks/idle_unload/benchmark.py --checkpoint "$CHECKPOINT" --mode idle-off
python research/benchmarks/idle_unload/benchmark.py --checkpoint "$CHECKPOINT" --mode idle-on
python research/benchmarks/idle_unload/benchmark.py --checkpoint "$CHECKPOINT" --mode mcp-local
```

For the remote measurement, run this server in another terminal, then run the client:

```bash
python research/benchmarks/idle_unload/benchmark.py --checkpoint "$CHECKPOINT" --mode serve --port 8000
```

```bash
python research/benchmarks/idle_unload/benchmark.py --checkpoint "$CHECKPOINT" --mode mcp-remote --base-url http://127.0.0.1:8000
```

Repeat the four measurements twice. The idle modes assert that residency follows the setting,
that the next request reloads successfully, and that answers match before and after unloading.
The MCP modes exercise status, routing, prediction, batch prediction, structured decisions and
presets; remote mode also checks that shortlisting refuses the call and torch stays unimported.
Compare answer maps and decided values across local and remote runs, excluding timing and device
metadata. Remote-client RSS excludes the one shared HTTP server, which still holds the model.

The baseline is the feature disabled on the same implementation and checkpoint. Results do not
claim that idle unloading returns all process RAM to the OS: allocators can retain pages even
after Metal allocations and model references are freed.

## Recorded results

`results.json` records two fresh-process runs on an Apple M4 Pro, Python 3.12, using the English
checkpoint at revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. Neither run downloaded weights.

| Measurement | Run 1 | Run 2 |
|---|---:|---:|
| Metal allocations, idle unloading disabled | 1610.387 MiB retained | 1610.387 MiB retained |
| Metal allocations, idle unloading enabled | 1610.387 → 0 MiB | 1610.387 → 0 MiB |
| Metal driver memory, idle unloading enabled | 2056.703 → 0.703 MiB | 2056.703 → 0.703 MiB |
| Warm request, idle unloading enabled | 188.836 ms | 193.897 ms |
| Request after idle unload | 727.361 ms | 836.043 ms |
| Local MCP process RSS after the tool sweep | 1074.453 MiB | 308.516 MiB |
| Remote MCP process RSS after the tool sweep | 71.453 MiB | 71.672 MiB |
| Torch imported by remote MCP | no | no |
| Local/remote answer parity | yes | yes |

Server RSS did not shrink consistently after unloading: it moved from 419.984 to 492.312 MiB in
the first enabled run and from 462.406 to 488.719 MiB in the second. Local MCP RSS also varied
substantially between processes. These are observed process snapshots, not a claim of a stable
percentage reduction or a total-system-memory measurement. Remote MCP RSS excludes the shared
server's model memory. The idle allocation reduction and answer parity reproduced in both runs.

`validation.json` records two complete runs of the shared CI/release suite list, the required lint
and compile gates, and strict documentation builds with no `griffe:` diagnostics. Run the same
checks with `python scripts/test_suites.py`, the commands in `AGENTS.md`, and
`zensical build --strict --clean`.
