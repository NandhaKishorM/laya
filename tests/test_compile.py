"""`compile=True`: one graph across request shapes, and no torch setting left changed. No weights.

Dynamo runs with `backend="eager"` so the graph count is checked on CPU without inductor or a C
compiler; the guards and recompiles being tested are dynamo's, the same under every backend.

    python tests/test_compile.py      (or python -m pytest tests/test_compile.py)
"""
import os
import sys
import threading
from types import SimpleNamespace

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya._compile import compile_model, independent_dims, _fx_config  # noqa: E402
from laya.agent import Agent, _pad_cuda_compile_batch  # noqa: E402
from laya.common import DecisionModel  # noqa: E402

# `torch.fx.experimental._config` only exists where duck sizing does; `laya._compile` degrades
# to a no-op without it, so the tests that observe the setting skip rather than assert on a
# knob this torch does not have.
fx_config = _fx_config()
requires_fx_config = pytest.mark.skipif(
    fx_config is None,
    reason="the compile path (duck sizing, and transformers' CPU compile support) needs a "
           "newer torch; it falls back to eager here")

# rows x tokens x markers; the first has rows == markers, which duck sizing would tie together
SHAPES = [(4, 40, 4), (4, 57, 4), (3, 70, 4), (5, 33, 2), (2, 90, 3), (8, 130, 5), (6, 61, 6), (7, 45, 3)]


def tiny_model():
    from transformers import AutoModel, ModernBertConfig

    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=2, global_attn_every_n_layers=2, local_attention=16,
                           max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                           cls_token_id=1, sep_token_id=2)
    encoder = AutoModel.from_config(cfg, attn_implementation="sdpa")
    # The CPU fixture checks Laya's dynamic dimensions, not ModernBERT's own
    # first-forward mutation of reference_compile ("auto" -> False on CPU).
    encoder.config.reference_compile = False
    return DecisionModel(encoder, 1, 2).eval()


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


def test_cuda_compile_padding_masks_only_the_new_tail():
    b = batch(2, 49, 3)
    padded = _pad_cuda_compile_batch(b, 7)
    assert padded["input_ids"].shape == (2, 56)
    assert padded["attention_mask"].shape == (2, 56)
    assert torch.equal(padded["input_ids"][:, :49], b["input_ids"])
    assert torch.equal(padded["attention_mask"][:, :49], b["attention_mask"])
    assert torch.all(padded["input_ids"][:, 49:] == 7)
    assert not padded["attention_mask"][:, 49:].any()
    assert padded["marker_pos"] is b["marker_pos"]
    assert padded["marker_mask"] is b["marker_mask"]
    assert padded["qtype"] is b["qtype"]
    assert b["input_ids"].shape == (2, 49)
    assert _pad_cuda_compile_batch(padded, 0) is padded


@requires_fx_config
def test_one_graph_across_shapes_and_same_outputs():
    torch._dynamo.reset()
    model = tiny_model()
    agent = compiled_agent(model)
    before = fx_config.use_duck_shape if fx_config is not None else None
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
    if fx_config is not None:
        assert fx_config.use_duck_shape is before


@requires_fx_config
def test_warmup_builds_every_graph_before_the_first_request():
    torch._dynamo.reset()
    agent = compiled_agent(tiny_model())
    agent.cfg = {"max_len": 512}
    agent.tok = SimpleNamespace(cls_token_id=1)
    start = graphs()
    assert agent.warmup() >= 0.0
    # the batch graph and torch's own single-row specialisation
    assert graphs() - start == 2, graphs() - start
    with torch.no_grad():
        for shape in SHAPES + [(1, 50, 4), (1, 200, 2)]:
            agent._infer(batch(*shape))
    assert graphs() - start == 2, graphs() - start


@requires_fx_config
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


@requires_fx_config
def test_duck_shape_restored_across_threads():
    before = fx_config.use_duck_shape
    inside, left, release = threading.Barrier(4), threading.Barrier(4), threading.Event()
    seen, after = [], []

    def call():
        with independent_dims():
            inside.wait()
            seen.append(fx_config.use_duck_shape)
            release.wait()
        left.wait()  # read once every call has returned, global or per-thread setting alike
        after.append(fx_config.use_duck_shape)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    while len(seen) < 4:
        pass
    release.set()
    for t in threads:
        t.join()
    assert seen == [False] * 4
    assert after == [before] * 4
    assert fx_config.use_duck_shape is before


@requires_fx_config
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


def test_constructor_warms_only_the_active_compiled_path():
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import laya.agent as module
    from laya import load

    model = tiny_model()
    events = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "rl_agent_config.json").write_text(json.dumps({"encoder": "unused", "head_layers": 1}))
        (root / "model.safetensors").touch()
        with patch.object(module, "build_model", side_effect=lambda *a, **kw: events.append("model") or model), \
             patch.object(module, "_load_tokenizer", return_value=SimpleNamespace(cls_token_id=1)), \
             patch("safetensors.torch.load_file", return_value=model.state_dict()), \
             patch.object(module, "compile_model", side_effect=lambda m, **kw: m) as compile_spy, \
             patch.object(module, "configure_cache", side_effect=lambda: events.append("cache")) as cache_spy, \
             patch.object(Agent, "accelerate"), \
             patch.object(Agent, "warmup", autospec=True) as warm:
            agent = load(directory, device="cpu", compile=True)
            warm.assert_called_once_with(agent)
            assert not agent.model.training and agent._compiled
            warm.reset_mock()
            load(directory, device="cpu", compile=True, compile_warmup=False)
            load(directory, device="cpu", compile=False)
            load(directory, device="cpu", compile=True, fast=True)
            warm.assert_not_called()
            assert compile_spy.call_count == 2
            agent.warmup()
            warm.assert_called_once_with(agent)
            cache_spy.assert_not_called()
            events.clear()
            load(directory, device="cpu", compile=True, compile_cache=True, compile_warmup=False)
            assert events == ["cache", "model"]
            cache_spy.assert_called_once_with()
            cache_spy.reset_mock()
            compile_spy.reset_mock()
            load(directory, device="cpu", compile=False, compile_cache=True)
            load(directory, device="cpu", compile=True, fast=True, compile_cache=True)
            compile_spy.assert_not_called()
            reduced = load(directory, device="cpu", compile=True, compile_warmup=False,
                           compile_mode="reduce-overhead")
            assert reduced._reduce_overhead
            assert compile_spy.call_args.kwargs == {"mode": "reduce-overhead"}
            compile_spy.reset_mock()
            load(directory, device="cpu", compile=False, compile_mode="reduce-overhead")
            load(directory, device="cpu", compile=True, fast=True, compile_mode="reduce-overhead")
            compile_spy.assert_not_called()
            try:
                load(directory, device="cpu", compile=True, compile_mode="typo")
            except ValueError as error:
                assert "compile_mode" in str(error)
            else:
                raise AssertionError("invalid mode accepted")
            cache_spy.assert_not_called()


@requires_fx_config
def test_missing_compiler_preserves_wrapper_and_warns_only_during_automatic_warmup():
    import json
    import tempfile
    import warnings
    from pathlib import Path
    from unittest.mock import patch
    from torch._inductor.exc import InvalidCxxCompiler
    from torch._dynamo.eval_frame import OptimizedModule
    import laya.agent as module
    from laya import load

    attempts = []

    def missing_compiler(graph, inputs, **kwargs):
        attempts.append(True)
        # Torch versions differ: some take the compiler, others read cpp.cxx.
        with torch._inductor.config.patch({"cpp.cxx": ("laya-missing-cxx",)}):
            try:
                error = InvalidCxxCompiler()
            except TypeError:
                error = InvalidCxxCompiler("laya-missing-cxx")
        raise error

    def compile_without_toolchain(model, **kwargs):
        return compile_model(model, backend=missing_compiler, **kwargs)

    torch._dynamo.reset()
    model = tiny_model()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "rl_agent_config.json").write_text(json.dumps({"encoder": "unused", "head_layers": 1}))
        (root / "model.safetensors").touch()
        with patch.object(module, "build_model", return_value=model), \
             patch.object(module, "_load_tokenizer", return_value=SimpleNamespace(cls_token_id=1)), \
             patch("safetensors.torch.load_file", return_value=model.state_dict()), \
             patch.object(module, "compile_model", side_effect=compile_without_toolchain):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                agent = load(directory, device="cpu", compile=True, compile_mode="reduce-overhead")
            assert attempts, "automatic warm-up never reached the compile backend"
            assert any(issubclass(w.category, RuntimeWarning)
                       and "InvalidCxxCompiler" in str(w.message)
                       and "laya-missing-cxx" in str(w.message)
                       and "keeping the compiled model" in str(w.message) for w in caught)
            wrapped = agent.model
            assert isinstance(wrapped, OptimizedModule) and wrapped._orig_mod is model
            assert agent._compiled and agent._reduce_overhead
            assert model.encoder.config.reference_compile is True
            for operation in (lambda: agent._infer(batch(2, 16, 3)), agent.warmup):
                calls = len(attempts)
                try:
                    operation()
                except Exception as error:
                    assert "InvalidCxxCompiler" in str(error) and "laya-missing-cxx" in str(error)
                else:
                    raise AssertionError("request or explicit warmup swallowed the compiler failure")
                assert len(attempts) > calls
                assert agent.model is wrapped and agent._compiled and agent._reduce_overhead
                assert model.encoder.config.reference_compile is True
            calls = len(attempts)

            explicit = load(directory, device="cpu", compile=True, compile_warmup=False)
            assert len(attempts) == calls
            try:
                explicit.warmup()
            except Exception as error:
                assert "InvalidCxxCompiler" in str(error) and "laya-missing-cxx" in str(error)
            else:
                raise AssertionError("explicit warmup swallowed the compiler failure")
            assert len(attempts) > calls and explicit._compiled
    torch._dynamo.reset()


def test_persistent_cache_is_opt_in_and_respects_the_environment():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from laya._compile import configure_cache

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Keep mixed separators in the inputs: Windows accepts them, and joining
        # child names need not preserve the spelling produced by pathlib.
        with patch.dict(os.environ, {"XDG_CACHE_HOME": directory + "/xdg"}, clear=True), \
             patch("os.path.expanduser", return_value=directory + "/.cache"), \
             patch("torch.compile") as compiler:
            compile_model(object())
            assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
            configure_cache()
            expected = root / "xdg" / "laya" / "torchinductor"
            assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == expected
            assert expected.is_dir()
            assert compiler.call_args.kwargs == {"dynamic": True}
            # A caller-selected directory always wins, including after repeated loads.
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = directory + "/explicit"
            assert Path(configure_cache()) == root / "explicit"
            assert Path(configure_cache()) == root / "explicit"
            assert (root / "explicit").is_dir()
            del os.environ["TORCHINDUCTOR_CACHE_DIR"]
            os.environ["XDG_CACHE_HOME"] = "relative-is-invalid"
            expected = root / ".cache" / "laya" / "torchinductor"
            assert Path(configure_cache()) == expected
            assert expected.is_dir()
            del os.environ["TORCHINDUCTOR_CACHE_DIR"]
            del os.environ["XDG_CACHE_HOME"]
            assert Path(configure_cache()) == expected
            assert expected.is_dir()


def test_cuda_graph_step_marks_each_forward_and_releases_after_failure():
    from unittest.mock import patch
    from laya._compile import cuda_graph_step

    with patch("torch.compiler.cudagraph_mark_step_begin") as mark:
        try:
            with cuda_graph_step():
                raise RuntimeError("forward failed")
        except RuntimeError:
            pass
        with cuda_graph_step():
            pass
        assert mark.call_count == 2
    with patch.object(torch, "compiler", None):
        try:
            with cuda_graph_step():
                pass
        except RuntimeError as error:
            assert "requires torch.compiler.cudagraph_mark_step_begin" in str(error)
        else:
            raise AssertionError("unsupported CUDA graph step API accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all compile tests passed")
