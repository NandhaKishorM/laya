"""GPU output ownership and threaded replay check; no checkpoint required.

Run: python tests/test_compile_cuda.py (skips without CUDA).
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya.agent import Agent
from laya._compile import compile_model


def test_reduce_overhead_retains_outputs_across_replay_and_threads():
    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable")
        return

    class Model(torch.nn.Module):
        def forward(self, ids, mask, pos, marker_mask, qtype):
            values = ids.float().sum(1, keepdim=True)
            return values.sin(), values.cos()

    agent = Agent.__new__(Agent)
    agent.device = torch.device("cuda")
    agent.dtype = torch.float32
    agent.amp_enabled = False
    agent.tok = SimpleNamespace(pad_token_id=0)
    agent._compiled = True
    agent._reduce_overhead = True
    agent.model = compile_model(Model().cuda().eval(), mode="reduce-overhead")

    @torch.no_grad()
    def call(value):
        b = {"input_ids": torch.full((2, 16), value, dtype=torch.long),
             "attention_mask": torch.ones(2, 16, dtype=torch.long),
             "marker_pos": torch.ones(2, 3, dtype=torch.long),
             "marker_mask": torch.ones(2, 3, dtype=torch.bool),
             "qtype": torch.zeros(2, dtype=torch.long)}
        out = agent._infer(b)
        ref = Model()(*b.values())
        return out, ref

    # First call warms, second records, later calls replay with changing tensor contents.
    held = [call(value) for value in range(1, 6)]
    from torch._inductor.cudagraph_trees import get_manager
    manager = get_manager(0, create_if_none_exists=False)
    assert manager is not None and list(manager.get_roots()), "CUDA graph was not recorded"
    with ThreadPoolExecutor(max_workers=2) as pool:
        held.extend(pool.map(call, range(6, 12)))
    torch.cuda.synchronize()
    for out, ref in held:
        for actual, expected in zip(out, ref):
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
    print("CUDA graph capture, replay, retained outputs, and two-thread parity passed")


if __name__ == "__main__":
    test_reduce_overhead_retains_outputs_across_replay_and_threads()
