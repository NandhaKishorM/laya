"""Backend adapter for the current TileLang fast path."""
import torch

from .base import Backend, BackendUnavailable


class TileLangBackend(Backend):
    name = "tilelang"

    def __init__(self, agent, use_graphs=True, verbose=False):
        super().__init__(agent)
        self.use_graphs = use_graphs
        self.verbose = verbose
        self.fast = None

    def _install(self):
        agent = self.agent
        if agent.device.type != "cuda":
            raise BackendUnavailable("needs a CUDA device")
        if agent.dtype not in (torch.bfloat16, torch.float16):
            raise BackendUnavailable("the TileLang kernels need bf16 or fp16 and the agent runs %s" % (agent.dtype,))
        if getattr(getattr(self.model.encoder, "config", None), "model_type", None) != "modernbert":
            raise BackendUnavailable("the kernels implement a ModernBERT-family encoder, got %r"
                                     % getattr(self.model.encoder.config, "model_type", None))
        last = None
        for _attempt in range(2):  # tilelang's JIT cache has been seen to fail once, then succeed
            try:
                from ..fast import FastLaya
                self.fast = FastLaya(self.model, max_len=agent.cfg.get("max_len", 512),
                                     use_graphs=self.use_graphs, verbose=self.verbose, dtype=agent.dtype)
                break
            except Exception as e:  # tilelang missing / unsupported arch
                last = e
        if self.fast is None:
            raise BackendUnavailable(last)
        self.max_len = self.fast.max_len

    def _uninstall(self):
        self.fast = None

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        return self.fast.forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder)

    def warmup(self, shapes=None):
        return self.agent.warmup(shapes)
