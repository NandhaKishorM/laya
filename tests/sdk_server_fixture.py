"""Serve a tiny offline checkpoint for the JavaScript integration test, never for production."""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import uvicorn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402
from laya import Router, load  # noqa: E402
from laya.common import DecisionModel  # noqa: E402
from laya.serve import create_app  # noqa: E402


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="laya-sdk-test-") as directory:
        root = Path(directory)
        cfg = BertConfig(vocab_size=6, hidden_size=64, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=128)
        cfg.save_pretrained(root / "encoder")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5},
                unk_token="[UNK]")),
            pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
            sep_token="[SEP]", mask_token="[MASK]",
        )
        tokenizer.save_pretrained(root / "tokenizer")
        save_file(DecisionModel(BertModel(cfg), head_layers=1).state_dict(), root / "model.safetensors")
        (root / "rl_agent_config.json").write_text(json.dumps({
            "encoder": "unused/offline", "head_layers": 1, "act_costs": {"act": 0},
            "max_len": 128, "head_max_len": 64,
        }))
        agent = load(str(root), device="cpu")
        questions = {
            "team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": {"meaning": "money"}, "technical": None}},
            "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "medium", "high"]},
            "refund": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "yes", "false": "no"}},
            "single": {"type": "choice", "instructions": "Pick", "criteria": ["only"]},
        }
        state = {"text": "hello", "unicode": "नमस्ते", "metadata": {"count": 0, "flag": False}}
        expected = agent.predict(state, questions)
        router = Router()
        # Both routing branches reach the same tiny fixture; this tests transport,
        # model invocation and numeric parity, not pretrained model quality.
        for name in ["english", "multilingual", "typed-decisions"]:
            router.attach(name, agent)
        os.environ["LAYA_API_KEY"] = "integration-test"
        app = create_app(router)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            print("LAYA_TEST_SERVER=" + json.dumps({
                "baseURL": "http://127.0.0.1:%d" % listener.getsockname()[1],
                "questions": questions, "state": state, "expected": expected,
            }), flush=True)
            uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[listener])


if __name__ == "__main__":
    main()
