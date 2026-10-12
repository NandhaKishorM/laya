"""Optional parallel bottleneck adapters for the eager ModernBERT encoder."""
import math
from contextlib import nullcontext

import torch
import torch.nn as nn


def validate_config(config, depth):
    if not isinstance(config, dict) or set(config) != {"version", "layers", "rank", "scale"}:
        raise ValueError("residual_adapters requires version, layers, rank and scale")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("unsupported residual_adapters version; expected 1")
    layers = config["layers"]
    if (not isinstance(layers, list) or not layers or
            any(type(i) is not int or not 0 <= i < depth for i in layers) or
            len(set(layers)) != len(layers)):
        raise ValueError("residual_adapters layers must be unique encoder layer indices")
    rank, scale = config["rank"], config["scale"]
    if type(rank) is not int or rank < 1:
        raise ValueError("residual_adapters rank must be a positive integer")
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("residual_adapters scale must be finite and positive")
    return {"version": 1, "layers": sorted(layers), "rank": rank, "scale": float(scale)}


class _Branch(nn.Module):
    def __init__(self, width, rank, scale):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)
        self.scale = scale
        self.enabled = True

    def forward(self, hidden):
        return self.up(torch.nn.functional.gelu(self.down(self.norm(hidden)))) * self.scale

    def apply_to_output(self, module, args, kwargs, output):
        if not self.enabled:
            return output
        hidden = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("residual_adapters require padded, rank-three hidden states")
        delta = self(hidden)
        if isinstance(output, tuple):
            return (output[0] + delta,) + output[1:]
        return output + delta


class ResidualAdapters(nn.Module):
    def __init__(self, encoder, config, no_init=False):
        super().__init__()
        if getattr(encoder.config, "model_type", None) != "modernbert":
            raise ValueError("residual_adapters currently support ModernBERT encoders only")
        layers = getattr(encoder, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise ValueError("residual_adapters require an encoder.layers ModuleList")
        self.config = validate_config(config, len(layers))
        # Check the entire attachment before installing any hooks. Bound methods survive
        # deepcopy without closing over the original model or registering it twice.
        for index in self.config["layers"]:
            if any(isinstance(getattr(hook, "__self__", None), _Branch)
                   for hook in layers[index]._forward_hooks.values()):
                raise ValueError("residual_adapters already attached to encoder layer %d" % index)
        parameter = next(encoder.parameters())
        with torch.device("meta") if no_init else nullcontext():
            self.branches = nn.ModuleDict({
                str(index): _Branch(encoder.config.hidden_size, self.config["rank"], self.config["scale"])
                for index in self.config["layers"]
            })
        if no_init:
            self.to_empty(device=parameter.device)
        self.to(device=parameter.device, dtype=parameter.dtype)
        handles = []
        try:
            for index in self.config["layers"]:
                handles.append(layers[index].register_forward_hook(
                    self.branches[str(index)].apply_to_output, with_kwargs=True))
        except Exception:
            for handle in handles:
                handle.remove()
            raise

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise TypeError("residual adapter enabled must be a bool")
        for branch in self.branches.values():
            branch.enabled = enabled


def reject_unsupported_backend(model, backend):
    # The backend manager compiles a bound forward, while the legacy SDK compiles
    # the module. Inspect the owning model in either case.
    model = getattr(model, "__self__", model)
    if getattr(model, "residual_adapters", None) is not None:
        raise ValueError("residual_adapters do not support %s; use eager PyTorch inference" % backend)
