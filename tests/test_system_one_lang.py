from typing import Any, Dict, List, Union
import pytest
from laya.agent import Agent

class DummyAgent(Agent):
    def __init__(self):
        # Skip the whole HuggingFace initialization for testing the plumbing
        self.device = "cpu"
        self.temperature_by_options = {}
        self.temperature = [1.0, 1.0, 1.0]
        self.lang_temperatures = {}
        self.cfg = {"max_len": 512, "head_max_len": 192}

    def predict_batch(self, states: List[Union[str, dict, list]], questions: Dict[str, Dict[str, Any]],
                      batch_size: int = None, lang: str = None, **kwargs) -> List[Dict[str, Any]]:
        # Mock predict_batch to just return the lang it was given so we can test the pass-through
        return [{"lang_passed_down": lang}]


def test_system_one_lang_parameter_passthrough():
    agent = DummyAgent()
    questions = {"q1": {"type": "noul", "instructions": "test"}}
    
    # 1. Test without lang
    res_no_lang = agent.system_one("test state", questions)
    assert res_no_lang.get("lang_passed_down") is None

    # 2. Test with lang
    res_with_lang = agent.system_one("test state", questions, lang="zh-CN")
    assert res_with_lang.get("lang_passed_down") == "zh-CN"
