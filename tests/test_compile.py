"""`compile=True`: one graph across request shapes, and no torch setting left changed. No weights.

Dynamo runs with `backend="eager"` so the graph count is checked on CPU without inductor or a C
compiler; the guards and recompiles being tested are dynamo's, the same under every backend.

    python tests/test_compile.py      (or python -m pytest tests/test_compile.py)
"""
import os
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya._compile import compile_model, independent_dims  # noqa: E402
from laya.agent import Agent  # noqa: E402
from laya.common import DecisionModel  # noqa: E402

fx_config = torch.fx.experimental._config

# rows x tokens x markers; the first has rows == markers, which duck sizing would tie together
SHAPES = [(4, 40, 4), (4, 57, 4), (3, 70, 4), (5, 33, 2), (2, 90, 3), (8, 130, 5), (6, 61, 6), (7, 45, 3)]


def tiny_model():
    from transformers import AutoModel, ModernBertConfig

    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=2, global_attn_every_n_layers=2, local_attention=16,
                           max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                           cls_token_id=1, sep_token_id=2)
    return DecisionModel(AutoModel.from_config(cfg, attn_implementation="sdpa"), 1, 2).eval()


def batch(rows, tokens, markers):
    return {
        "input_ids": torch.randint(3, 100, (rows, tokens)),
        "attention_mask": torch.ones((rows, tokens), dtype=torch.long),
        "marker_pos": torch.arange(1, markers + 1).repeat(rows, 1),
        "marker_mask": torch.ones((rows, markers), dtype=torch.bool),
        "qtype": torch.zeros((rows,), dtype=torch.long),
    }


def compiled_agent(model):
    agent = Agent.__new__(Agent)
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.amp_enabled = False
    agent.model = compile_model(model, backend="eager")
    agent._compiled = True
    return agent


def graphs():
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


def test_one_graph_across_shapes_and_same_outputs():
    torch._dynamo.reset()
    model = tiny_model()
    agent = compiled_agent(model)
    before = fx_config.use_duck_shape
    start = graphs()
    with torch.no_grad():
        for shape in SHAPES:
            b = batch(*shape)
            logits, act = agent._infer(b)
            ref_logits, ref_act = model(*b.values())
            assert torch.allclose(logits, ref_logits, atol=1e-4), shape
            assert torch.allclose(act, ref_act, atol=1e-4), shape
    # stock torch.compile(model) builds 4 graphs for these shapes (static, then automatic dynamic,
    # then two duck-sizing recompiles); dynamic=True with independent dimensions builds one
    assert graphs() - start == 1, graphs() - start
    assert fx_config.use_duck_shape is before


def test_duck_shape_restored_after_errors_and_nesting():
    before = fx_config.use_duck_shape
    with independent_dims():
        assert fx_config.use_duck_shape is False
        with independent_dims():
            assert fx_config.use_duck_shape is False
        assert fx_config.use_duck_shape is False  # an inner exit does not restore early
    assert fx_config.use_duck_shape is before
    try:
        with independent_dims():
            raise RuntimeError("forward failed")
    except RuntimeError:
        pass
    assert fx_config.use_duck_shape is before


def test_duck_shape_restored_across_threads():
    before = fx_config.use_duck_shape
    inside, release = threading.Barrier(4), threading.Event()
    seen = []

    def call():
        with independent_dims():
            inside.wait()
            seen.append(fx_config.use_duck_shape)
            release.wait()

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    while len(seen) < 4:
        pass
    release.set()
    for t in threads:
        t.join()
    assert seen == [False] * 4
    assert fx_config.use_duck_shape is before


def test_eager_agent_leaves_the_setting_alone():
    agent = Agent.__new__(Agent)
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.amp_enabled = False
    seen = []

    class Spy(torch.nn.Module):
        def forward(self, ids, *rest):
            seen.append(fx_config.use_duck_shape)
            return torch.zeros(ids.shape[0], 2), torch.zeros(ids.shape[0], 2)

    agent.model = Spy()
    agent._infer(batch(2, 8, 2))
    assert seen == [fx_config.use_duck_shape]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all compile tests passed")
