import os
import sys
import tempfile
import numpy as np

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

@pytest.mark.skipif(not HAS_ONNX, reason="onnx and onnxruntime are required")
def test_onnx_numerical_parity():
    """Verify that PyTorch and ONNX agents produce identical outputs for all 3 question types."""
    model_id = "convaiinnovations/laya"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "laya.onnx")
        
        # 1. Export to ONNX
        export_to_onnx(model_id, onnx_path)
        
        # 2. Load both PyTorch and ONNX Agents
        agent_pt = Agent(model_id, compile=False, device="cpu")
        agent_onnx = ONNXAgent(model_id, onnx_path)
        
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
        
        print("ONNX numerical parity test passed for all 3 question types!")


@pytest.mark.skipif(not HAS_ONNX, reason="onnx and onnxruntime are required")
def test_onnx_batch_shapes_and_act_probability():
    """The exported graph must run at more than one batch size, and `act_probability` must match.

    Two gaps this closes. The graph used to be exported from a batch-1 example, which baked
    batch=1 into the decision head's attention intermediates: it answered one row and failed
    on the second with `Add` broadcasting seq_len against seq_len * batch. Feeding the graph
    directly at batch 1, 2 and 3 catches that without going through `ONNXAgent`, whose
    collation happens to always produce one row per question.

    And `act_probability` is the one field the parity test above never compares, so a
    wrong-shaped `act_logits` would pass the suite even with the `Add` fixed.
    """
    import onnx

    model_id = "convaiinnovations/laya"

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "laya.onnx")
        export_to_onnx(model_id, onnx_path)

        # 1. Both batch axes the export declares must survive to the graph, not just the inputs.
        graph = onnx.load(onnx_path, load_external_data=False).graph
        declared = {
            vi.name: [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]
            for vi in list(graph.input) + list(graph.output)
        }
        for name in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype",
                     "logits", "act_logits"):
            assert isinstance(declared[name][0], str),                 f"{name} has a static batch axis {declared[name]!r}; the export froze the batch"

        # 2. Feed the graph directly. seq_len and markers are held fixed so the only thing
        #    varying is the batch, which is what regressed.
        seq_len, markers = 53, 5
        for batch in (1, 2, 3):
            feed = {
                "input_ids": np.random.randint(1, 1000, (batch, seq_len)).astype(np.int64),
                "attention_mask": np.ones((batch, seq_len), dtype=np.int64),
                "marker_pos": np.tile(np.arange(markers, dtype=np.int64), (batch, 1)),
                "marker_mask": np.ones((batch, markers), dtype=bool),
                "qtype": np.zeros((batch,), dtype=np.int64),
            }
            session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            logits, act_logits = session.run(["logits", "act_logits"], feed)
            assert logits.shape == (batch, markers),                 f"batch {batch}: logits {logits.shape}, expected {(batch, markers)}"
            assert act_logits.shape[0] == batch,                 f"batch {batch}: act_logits {act_logits.shape}, expected batch {batch}"

        # 3. `act_probability` parity, on every question type.
        agent_pt = Agent(model_id, compile=False, device="cpu")
        agent_onnx = ONNXAgent(model_id, onnx_path)

        state = {"request": "Refactor this service using dependency injection"}
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "What is the user asking to do?",
                "criteria": {"refactor": "code refactoring", "bug_fix": "fixing bugs"},
            },
            "complexity": {
                "type": "score",
                "instructions": "How complex is this request?",
                "criteria": ["trivial", "simple", "moderate", "complex", "very complex"],
            },
            "safety": {
                "type": "noul",
                "instructions": "Does this request involve any unsafe or harmful content?",
            },
        }

        res_pt = agent_pt.predict(state, questions)
        res_onnx = agent_onnx.predict(state, questions)

        for qid in questions:
            np.testing.assert_allclose(
                res_pt["answers"][qid]["action"]["act_probability"],
                res_onnx["answers"][qid]["action"]["act_probability"],
                atol=1e-3, rtol=1e-3,
                err_msg=f"act_probability mismatch on {qid}",
            )

        print("ONNX batch-shape and act_probability test passed!")
