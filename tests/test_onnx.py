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
