import os
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent

# Skip the test if onnx isn't installed
try:
    import onnxruntime
    from laya.onnx_agent import ONNXAgent
    from scripts.export_onnx import export_to_onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

import pytest

MODEL_ID = "convaiinnovations/laya"
INPUT_NAMES = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


@pytest.fixture(scope="module")
def exported():
    """One export shared by the tests below: (onnx path, the PyTorch agent it came from)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "laya.onnx")
        export_to_onnx(MODEL_ID, onnx_path)
        yield onnx_path, Agent(MODEL_ID, compile=False, device="cpu")


@pytest.mark.skipif(not HAS_ONNX, reason="onnx and onnxruntime are required")
def test_onnx_graph_runs_at_any_batch(exported):
    """The exported graph itself at batch 1, 3 and 7 with a padded row, against the PyTorch model.

    Runs before any ONNXAgent call, so an export that only works at batch 1 (#695: the trace
    specialised batch=1 and `act_logits` came out as a static (1, 2)) fails here, on the shape.
    """
    onnx_path, agent_pt = exported
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # The declared shapes: symbolic names, not integers baked in at trace time. Catches an export
    # that drops or renames a declaration even on a torch where the batch test below still passes.
    declared = {t.name: t.shape for t in session.get_inputs() + session.get_outputs()}
    assert declared == {
        "input_ids": ["batch_size", "seq_len"],
        "attention_mask": ["batch_size", "seq_len"],
        "marker_pos": ["batch_size", "num_markers"],
        "marker_mask": ["batch_size", "num_markers"],
        "qtype": ["batch_size"],
        "logits": ["batch_size", "num_markers"],
        "act_logits": ["batch_size", 2],
    }, declared
    for batch in (1, 3, 7):
        rng = np.random.default_rng(batch)
        seq_len = 53
        feed = {
            "input_ids": rng.integers(5, 1000, (batch, seq_len)),
            "attention_mask": np.ones((batch, seq_len), dtype=np.int64),
            "marker_pos": np.tile(np.array([[3, 9, 14, 20, 30]]), (batch, 1)),
            "marker_mask": np.ones((batch, 5), dtype=bool),
            "qtype": np.arange(batch, dtype=np.int64) % 3,
        }
        feed["attention_mask"][0, 40:] = 0
        logits, act_logits = session.run(["logits", "act_logits"], feed)
        assert logits.shape == (batch, 5) and act_logits.shape == (batch, 2), (logits.shape, act_logits.shape)
        with torch.no_grad():
            ref = agent_pt.model(*[torch.from_numpy(feed[n]) for n in INPUT_NAMES])
        np.testing.assert_allclose(logits, ref[0].numpy(), atol=1e-3)
        np.testing.assert_allclose(act_logits, ref[1].numpy(), atol=1e-2)


@pytest.mark.skipif(not HAS_ONNX, reason="onnx and onnxruntime are required")
def test_onnx_numerical_parity(exported):
    """Verify that PyTorch and ONNX agents produce identical outputs for all 3 question types."""
    onnx_path, agent_pt = exported
    agent_onnx = ONNXAgent(MODEL_ID, onnx_path)
    
    # 3. Define a state and all 3 question types
    state = {"request": "Refactor this service using dependency injection"}
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "What is the user asking to do?",
            "criteria": {
                "refactor": "code refactoring, rewriting",
                "bug_fix": "fixing bugs, issues",
                "feature": "adding new features"
            }
        },
        "complexity": {
            "type": "score",
            "instructions": "How complex is this request?",
            "criteria": ["trivial", "simple", "moderate", "complex", "very complex"]
        },
        "safety": {
            "type": "noul",
            "instructions": "Does this request involve any unsafe or harmful content?"
        },
    }
    
    # 4. Predict with both agents
    res_pt = agent_pt.predict(state, questions)
    res_onnx = agent_onnx.predict(state, questions)
    
    # 5. Assert parity for choice question
    assert res_pt["answers"]["intent"]["choice"] == res_onnx["answers"]["intent"]["choice"], \
        f"Choice mismatch: {res_pt['answers']['intent']['choice']} vs {res_onnx['answers']['intent']['choice']}"
    
    probs_pt = res_pt["answers"]["intent"]["probabilities"]
    probs_onnx = res_onnx["answers"]["intent"]["probabilities"]
    for k in probs_pt.keys():
        np.testing.assert_allclose(probs_pt[k], probs_onnx[k], atol=1e-3, rtol=1e-3)
    
    # 6. Assert parity for score question
    np.testing.assert_allclose(
        res_pt["answers"]["complexity"]["score"],
        res_onnx["answers"]["complexity"]["score"],
        atol=1e-3, rtol=1e-3,
    )
    
    probs_pt_s = res_pt["answers"]["complexity"]["probabilities"]
    probs_onnx_s = res_onnx["answers"]["complexity"]["probabilities"]
    for k in probs_pt_s.keys():
        np.testing.assert_allclose(probs_pt_s[k], probs_onnx_s[k], atol=1e-3, rtol=1e-3)
    
    # 7. Assert parity for noul question
    np.testing.assert_allclose(
        res_pt["answers"]["safety"]["noul"],
        res_onnx["answers"]["safety"]["noul"],
        atol=1e-3, rtol=1e-3,
    )
    
    # 8. The act head: its output was a static (1, 2) when the export traced at batch 1
    # (#695), and nothing above reads it.
    for qid in questions:
        np.testing.assert_allclose(
            res_pt["answers"][qid]["action"]["act_probability"],
            res_onnx["answers"][qid]["action"]["act_probability"],
            atol=5e-3,
        )

    print("ONNX numerical parity test passed for all 3 question types!")
