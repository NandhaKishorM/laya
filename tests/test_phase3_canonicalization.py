"""Phase 3 and Phase 4 canonicalization regression tests.

Verifies:
1. Choice permutation invariance (end-to-end and representation-level).
2. Two-order regression test (wire2 vs wire3).
3. Mixed scalar label handling and native type preservation.
4. Description binding invariant.
5. Score order preservation (rubric levels 0..N-1 preserved).
6. Noul fixed semantics preservation ([false, true] preserved).
7. List input normalization and canonicalization.
8. Shortlist ranking semantics vs model representation boundary.
9. Agent and ONNXAgent normalization parity.
"""
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya import load  # noqa: E402
from laya.agent import Agent  # noqa: E402
from laya.common import (  # noqa: E402
    DecisionModel,
    canonicalize_choice_criteria,
    render_options,
)
from laya.onnx_agent import ONNXAgent  # noqa: E402
from laya.shortlist import predict_shortlist  # noqa: E402


def build_tiny_repo(repo: Path):
    """Create a minimal valid checkpoint for offline CPU testing."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    repo.mkdir(parents=True, exist_ok=True)
    config = BertConfig(
        vocab_size=16,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
    )
    config.save_pretrained(repo / "encoder")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5},
                unk_token="[UNK]",
            )
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(repo / "tokenizer")
    model = DecisionModel(BertModel(config), head_layers=0)
    save_file(model.state_dict(), repo / "model.safetensors")
    (repo / "rl_agent_config.json").write_text(
        json.dumps({
            "encoder": "unused/offline",
            "head_layers": 0,
            "act_costs": {"act": 0},
            "max_len": 64,
            "head_max_len": 32,
        })
    )


class CanonicalizationFunctionTests(unittest.TestCase):
    """Test unit behavior of canonicalize_choice_criteria."""

    def test_permutation_determinism(self):
        c1 = {"security": "sec desc", "cloud": "cloud desc", "database": "db desc"}
        c2 = {"cloud": "cloud desc", "database": "db desc", "security": "sec desc"}
        c3 = {"database": "db desc", "security": "sec desc", "cloud": "cloud desc"}

        res1 = canonicalize_choice_criteria(c1)
        res2 = canonicalize_choice_criteria(c2)
        res3 = canonicalize_choice_criteria(c3)

        self.assertEqual(list(res1.keys()), ["cloud", "database", "security"])
        self.assertEqual(list(res2.keys()), ["cloud", "database", "security"])
        self.assertEqual(list(res3.keys()), ["cloud", "database", "security"])
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)

    def test_description_binding(self):
        criteria = {"cloud": "cloud description", "database": "database description"}
        canonical = canonicalize_choice_criteria(criteria)
        self.assertEqual(canonical["cloud"], "cloud description")
        self.assertEqual(canonical["database"], "database description")

    def test_mixed_scalar_labels(self):
        criteria = {0: "zero", "2": "two", 3.0: "three", True: "true"}
        canonical = canonicalize_choice_criteria(criteria)
        # Check no TypeError and all native types preserved
        keys = list(canonical.keys())
        self.assertEqual(len(keys), 4)
        for k in keys:
            self.assertIn(k, criteria)
            self.assertEqual(canonical[k], criteria[k])

    def test_list_input_normalization(self):
        criteria = ["security", "cloud", "database"]
        canonical = canonicalize_choice_criteria(criteria)
        self.assertEqual(list(canonical.keys()), ["cloud", "database", "security"])
        self.assertEqual(canonical, {"cloud": None, "database": None, "security": None})


class NormalizationBoundaryTests(unittest.TestCase):
    """Test Agent._to_internal and ONNXAgent._to_internal."""

    def test_agent_and_onnx_agent_parity(self):
        qdef = {
            "type": "choice",
            "instructions": "Route request",
            "criteria": {"security": "sec desc", "cloud": "cloud desc"},
        }
        internal_pt = Agent._to_internal(qdef)
        internal_onnx = ONNXAgent._to_internal(qdef)
        self.assertEqual(internal_pt, internal_onnx)
        self.assertEqual(list(internal_pt["crit"].keys()), ["cloud", "security"])

    def test_score_order_is_preserved(self):
        qdef = {
            "type": "score",
            "instructions": "Rate severity",
            "criteria": ["critical", "degraded", "normal"],
        }
        internal = Agent._to_internal(qdef)
        self.assertEqual(internal["crit"], ["critical", "degraded", "normal"])

        qdef_rev = {
            "type": "score",
            "instructions": "Rate severity",
            "criteria": ["normal", "degraded", "critical"],
        }
        internal_rev = Agent._to_internal(qdef_rev)
        self.assertEqual(internal_rev["crit"], ["normal", "degraded", "critical"])

    def test_noul_semantics_preserved(self):
        qdef = {
            "type": "noul",
            "instructions": "Is spam?",
            "criteria": {"true": "is spam", "false": "not spam"},
        }
        internal = Agent._to_internal(qdef)
        rendered = render_options(internal)
        self.assertEqual(len(rendered), 2)
        self.assertTrue(rendered[0].startswith("false: not spam"))
        self.assertTrue(rendered[1].startswith("true: is spam"))

    def test_caller_criteria_dict_not_mutated(self):
        original = {"b": "desc b", "a": "desc a"}
        copy_keys = list(original.keys())
        qdef = {"type": "choice", "instructions": "Select", "criteria": original}
        Agent._to_internal(qdef)
        self.assertEqual(list(original.keys()), copy_keys)


class ShortlistBoundaryTests(unittest.TestCase):
    """Test shortlist ranking semantics vs model representation boundary."""

    def test_shortlist_ranking_order_preserved_in_metadata(self):
        class MockAgent:
            def __init__(self):
                self.called_with = None

            def predict(self, state, questions, **kwargs):
                self.called_with = questions
                # Mock return
                return {
                    "answers": {
                        "intent": {
                            "type": "choice",
                            "choice": "beta",
                            "confidence": 0.8,
                            "probabilities": {"alpha": 0.2, "beta": 0.8},
                        }
                    }
                }

        agent = MockAgent()
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "Which topic?",
                "criteria": {"alpha": "topic A", "beta": "topic B", "gamma": "topic C"},
            }
        }
        # Fake embed fn: beta > alpha > gamma
        embeddings = {
            "Which topic?\nstate": [1.0, 0.0],
            "beta: topic B": [1.0, 0.0],
            "alpha: topic A": [0.5, 0.5],
            "gamma: topic C": [0.0, 1.0],
        }

        def mock_embed(texts):
            return [embeddings.get(t, [0.0, 0.0]) for t in texts]

        res = predict_shortlist(agent, "state", questions, mock_embed, k=2)

        # 1. Shortlist ranking order in metadata must be ["beta", "alpha"]
        self.assertEqual(res["shortlist"]["intent"]["labels"], ["beta", "alpha"])
        self.assertTrue(
            res["shortlist"]["intent"]["scores"][0] > res["shortlist"]["intent"]["scores"][1]
        )

        # 2. When agent converts to internal representation, it canonicalizes to ['alpha', 'beta']
        internal = Agent._to_internal(agent.called_with["intent"])
        self.assertEqual(list(internal["crit"].keys()), ["alpha", "beta"])


class EndToEndPermutationInvarianceTests(unittest.TestCase):
    """End-to-end tests with tiny model running inference on CPU."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        build_tiny_repo(Path(cls.tmp.name) / "repo")
        cls.agent = load(str(Path(cls.tmp.name) / "repo"), device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_choice_permutation_invariance_all_orderings(self):
        labels = ["alpha", "beta", "gamma", "delta"]
        descriptions = {
            "alpha": "first option description",
            "beta": "second option description",
            "gamma": "third option description",
            "delta": "fourth option description",
        }
        state = "hello world request"

        baseline_res = None
        for perm in itertools.permutations(labels):
            crit = {k: descriptions[k] for k in perm}
            q = {"topic": {"type": "choice", "instructions": "Select topic", "criteria": crit}}
            res = self.agent.predict(state, q)
            ans = res["answers"]["topic"]

            if baseline_res is None:
                baseline_res = ans
            else:
                self.assertEqual(ans["choice"], baseline_res["choice"])
                self.assertAlmostEqual(ans["confidence"], baseline_res["confidence"], places=5)
                for label in labels:
                    self.assertAlmostEqual(
                        ans["probabilities"][label],
                        baseline_res["probabilities"][label],
                        places=5,
                        msg=f"Probability mismatch for label {label}",
                    )

    def test_two_order_wire2_wire3_regression(self):
        wire2_criteria = {
            "cloud_and_infrastructure": "Cloud infrastructure and DevOps tools",
            "communication": "Chat, email, and messaging services",
            "data_and_databases": "Data storage, databases, and pipelines",
            "developer_tools": "SDKs, IDEs, and developer utilities",
            "other": "Miscellaneous tools and services",
            "search_and_retrieval": "Search engines and retrieval systems",
            "utility_and_system": "System-level tools and OS utilities",
        }
        wire3_criteria = {
            "developer_tools": "SDKs, IDEs, and developer utilities",
            "utility_and_system": "System-level tools and OS utilities",
            "search_and_retrieval": "Search engines and retrieval systems",
            "cloud_and_infrastructure": "Cloud infrastructure and DevOps tools",
            "other": "Miscellaneous tools and services",
            "communication": "Chat, email, and messaging services",
            "data_and_databases": "Data storage, databases, and pipelines",
        }
        state = "Troubleshooting an issue with my local database connection"
        q_wire2 = {"cat": {"type": "choice", "instructions": "Category?", "criteria": wire2_criteria}}
        q_wire3 = {"cat": {"type": "choice", "instructions": "Category?", "criteria": wire3_criteria}}

        res2 = self.agent.predict(state, q_wire2)["answers"]["cat"]
        res3 = self.agent.predict(state, q_wire3)["answers"]["cat"]

        self.assertEqual(res2["choice"], res3["choice"])
        self.assertAlmostEqual(res2["confidence"], res3["confidence"], places=5)
        for k in wire2_criteria:
            self.assertAlmostEqual(res2["probabilities"][k], res3["probabilities"][k], places=5)

    def test_mixed_scalar_labels_e2e(self):
        criteria = {1: "level 1", "2": "level 2", 3.5: "level 3.5"}
        state = "hello"
        q = {"scale": {"type": "choice", "instructions": "Pick scale", "criteria": criteria}}
        res = self.agent.predict(state, q)["answers"]["scale"]

        self.assertIn(res["choice"], criteria)
        # Check original scalar types preserved in probabilities dict
        self.assertEqual(set(res["probabilities"].keys()), {1, "2", 3.5})
    def test_metamorphic_option_order_flip_rate_zero(self):
        from research.eval.metamorphic import evaluate
        state = "hello world request"
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "Select intent",
                "criteria": {"alpha": "first option", "beta": "second option", "gamma": "third option"},
            }
        }
        cases = [(state, questions)]

        def score(batch_cases):
            vectors = []
            for s, q_dict in batch_cases:
                res = self.agent.predict(s, q_dict)
                qid = next(iter(q_dict))
                probs = res["answers"][qid]["probabilities"]
                presented_keys = list(q_dict[qid]["criteria"].keys())
                vectors.append([probs[k] for k in presented_keys])
            return vectors

        report = evaluate(cases, score, gold_indices=None, seed=42)["report"]
        agreement = report["option_order"]["semantic_agreement_rate"]
        flip_rate = 1.0 - agreement
        self.assertEqual(flip_rate, 0.0)
        self.assertEqual(report["option_order"]["max_probability_drift"], 0.0)

if __name__ == "__main__":
    unittest.main()
