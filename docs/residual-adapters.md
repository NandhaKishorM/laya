# Residual encoder adapters

Residual adapters are an optional, local PyTorch architecture for fine-tuning LAYA.
They add a trainable parallel bottleneck to selected ModernBERT encoder blocks:

```text
output = encoder_block(hidden) + scale * up(GELU(down(LayerNorm(hidden))))
```

The branch reads the block's input, not its output. Its up-projection starts at zero,
so attaching freshly initialized adapters preserves the original model's outputs.
Training is required to obtain a useful change; adapters are not a zero-shot quality upgrade.
Stock checkpoints and the public prediction interface remain unchanged.

## Configuration and loading

An adapted checkpoint records this additional field in `rl_agent_config.json`:

```json
{
  "residual_adapters": {
    "version": 1,
    "layers": [0, 1],
    "rank": 256,
    "scale": 8.0
  }
}
```

The layer indices must be unique and within the encoder's depth. Rank is a positive
integer; scale is finite and positive. Configuration versions and unknown fields are
checked, not silently ignored. Omitting the field (or setting it to `null`) selects
stock LAYA. The example selects two layers; use all layer indices to adapt every block.

Load a complete adapted bundle through the ordinary SDK:

```python
import laya

agent = laya.load("/path/to/adapted-checkpoint", device="cpu")
```

The bundle contains the complete `model.safetensors`, encoder configuration, tokenizer,
and LAYA configuration. Branch parameters live under `residual_adapters.branches.<layer>`.
The loader constructs them before strict weight loading. An adapter-only state dictionary
is not a complete LAYA checkpoint, and research wrappers using other key names require
an explicit conversion and parity check.

## Start from an existing LAYA checkpoint

This creates a separate trainable model while leaving the loaded agent unchanged:

```python
import copy
from laya.common import DecisionModel

base = agent.model
adapter_config = {
    "version": 1,
    "layers": list(range(len(base.encoder.layers))),
    "rank": 256,
    "scale": 8.0,
}
model = DecisionModel(
    copy.deepcopy(base.encoder),
    head_layers=agent.cfg["head_layers"],
    n_act=base.act_head[-1].out_features,
    residual_adapters=adapter_config,
)
missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
assert missing and all(k.startswith("residual_adapters.") for k in missing)
assert not unexpected
model.to(agent.device)
model.train()
```

Use an uncompiled, stock checkpoint as the starting agent. Construct the optimizer after
constructing the adapters. Normal supervised losses and encoder/head gradient checkpointing
work with the branches; use non-reentrant encoder checkpointing (`use_reentrant=False`).
Freezing `model.encoder` leaves branch parameters trainable because they are registered
separately. `detach_encoder=True` detaches the complete encoder result, including branches.
The architecture does not choose losses, learning rates, or freeze the action head for you.

The shared `laya.train.train_model` loop keeps adapter gradients when
`TrainConfig(freeze_encoder=True)` freezes the backbone. Adapters use `head_lr` alongside the
decision head; encoder parameters remain unchanged. This also applies when continuing an
adapted checkpoint through `laya-train --freeze-encoder`. Freeze the adapters explicitly too
if you intend to train only the decision head.

After training, save into a **new** checkpoint directory:

```python
import json
from pathlib import Path
from safetensors.torch import save_file

output = Path("adapted-checkpoint")
output.mkdir(exist_ok=False)
cfg = copy.deepcopy(agent.cfg)
cfg["residual_adapters"] = copy.deepcopy(model.residual_adapters.config)
# Inherited calibration is not validated for newly trained weights.
cfg["temperature"] = [1.0, 1.0, 1.0]
cfg.pop("temperature_by_options", None)
model.encoder.config.save_pretrained(output / "encoder")
agent.tok.save_pretrained(output / "tokenizer")
save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
          output / "model.safetensors")
(output / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
```

Re-fit any calibration/abstention policy on separate calibration data before relying on
confidence thresholds. Configuration policies beyond temperature also need review after
training. Reload the saved bundle and compare its outputs before deployment. This inference
bundle is not a resumable optimizer/scheduler/RNG training checkpoint.

## Ablation and supported execution

`model.set_residual_enabled(False)` bypasses the branches without removing parameters;
`True` re-enables them. This keeps the **current shared encoder weights**. If training changed
those weights, disabling branches does not recover the pre-training model. Toggle only
between requests/optimizer steps, never during an in-flight forward or backward.
This temporary switch is not serialized: reloading enables configured branches.

The supported path is eager PyTorch with padded ModernBERT inputs and the normal SDK
batching interface. CPU and CUDA are supported. `compile=True`, ONNX export/loading,
and the TileLang fast path are not qualified for adapters: compilation/ONNX raise;
`accelerate(strict=True)` raises, while non-strict acceleration warns and keeps eager inference.
`backend="auto"` selects eager without probing accelerated backends. An explicit
`set_backend("compile", strict=True)` raises; with `strict=False` it warns and falls back to eager.
The upstream parallel-option layout is supported on its required Transformers 5 runtime;
adapters preserve its option-reordering equivariance.
Do not bypass these checks with a custom exporter or a replaced encoder forward.

For width `d`, rank `r`, and `L` adapted blocks, the added parameter count is
`L * (2*d*r + 2*d)`. For 28 blocks of width 1,024 and rank 256 this is 14,737,408
parameters. Latency and training memory must be measured, not inferred from that count.

## Runnable checks

```bash
python -m pytest tests/test_residual_adapters.py -q
python benchmarks/residual_adapters.py --model /path/to/local-checkpoint \
    --device cuda:0 --iterations 20 --train-steps 2
```

The benchmark uses synthetic token inputs and the supplied local weights. It checks exact
zero-initialization parity, reports warmed median latency, parameter storage and CUDA peak allocated memory,
and optionally runs optimizer steps. It does not download models, evaluate decision quality,
or save checkpoints. Use held-out task data to decide whether trained adapters improve your task.
Parameter storage excludes activations, optimizer state and allocator overhead; it is not peak
inference or training memory. CUDA peak memory is reported as `null` on CPU.
