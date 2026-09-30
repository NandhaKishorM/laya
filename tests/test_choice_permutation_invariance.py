"""Permanent regression tests for choice criteria permutation invariance (Phase 4).

Explicit test coverage for STEP 7 invariants:
- Test 1: Choice permutation invariance across many permutations.
- Test 2: Two-order wire2 vs wire3 regression test.
- Test 3: Mixed scalar labels (native type preservation, no TypeError).
- Test 4: Description binding invariant.
- Test 5: Score order preservation.
- Test 6: Noul semantics preservation.
- Test 7: List input expansion and canonicalization.
- Test 8: Shortlist ranking semantics vs model representation.
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


class ChoicePermutationInvarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        build_tiny_repo(Path(cls.tmp.name) / "repo")
        cls.agent = load(str(Path(cls.tmp.name) / "repo"), device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_1_choice_permutation_invariance(self):
        """Test 1: Given [A, B, C, D, E], all permutations produce identical predictions and probabilities."""
        labels = ["A", "B", "C", "D", "E"]
        criteria_base = {
            "A": "Option Alpha description",
            "B": "Option Beta description",
            "C": "Option Gamma description",
            "D": "Option Delta description",
            "E": "Option Epsilon description",
        }
        state = "Select an appropriate option"

        baseline_choice = None
        baseline_probs = None
        baseline_conf = None

        # Sample 20 permutations
        perms = list(itertools.permutations(labels))[:20]
        for perm in perms:
            crit = {k: criteria_base[k] for k in perm}
            q = {"topic": {"type": "choice", "instructions": "Select option", "criteria": crit}}
            res = self.agent.predict(state, q)
            ans = res["answers"]["topic"]

            if baseline_choice is None:
                baseline_choice = ans["choice"]
                baseline_probs = ans["probabilities"]
                baseline_conf = ans["confidence"]
            else:
                self.assertEqual(ans["choice"], baseline_choice)
                self.assertAlmostEqual(ans["confidence"], baseline_conf, places=5)
                for label in labels:
                    self.assertAlmostEqual(
                        ans["probabilities"][label],
                        baseline_probs[label],
                        places=5,
                        msg=f"Mismatch for label {label}",
                    )

    def test_2_two_order_regression(self):
        """Test 2: Direct regression reproducing wire2 and wire3 orderings."""
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

    def test_3_mixed_scalar_labels(self):
        """Test 3: Mixed scalar labels preserve types and avoid TypeError."""
        criteria = {0: "level 0", "2": "level 2", 3.0: "level 3.0", True: "level true"}
        canonical = canonicalize_choice_criteria(criteria)
        self.assertEqual(len(canonical), 4)

        # In Agent.predict, ensure decoder preserves scalar types
        state = "test mixed scalar"
        q = {"scale": {"type": "choice", "instructions": "Pick scale", "criteria": criteria}}
        ans = self.agent.predict(state, q)["answers"]["scale"]
        self.assertIn(ans["choice"], criteria)
        self.assertEqual(set(ans["probabilities"].keys()), set(criteria.keys()))

    def test_4_description_binding(self):
        """Test 4: Descriptions remain bound to their respective criterion."""
        criteria = {
            "cloud": "cloud description",
            "database": "database description",
        }
        canonical = canonicalize_choice_criteria(criteria)
        self.assertEqual(canonical["cloud"], "cloud description")
        self.assertEqual(canonical["database"], "database description")

    def test_5_score_order_preservation(self):
        """Test 5: Score criteria order is strictly preserved."""
        score_crit = ["level 0: normal", "level 1: degraded", "level 2: critical"]
        qdef = {"type": "score", "instructions": "Rate", "criteria": score_crit}
        internal = Agent._to_internal(qdef)
        self.assertEqual(internal["crit"], score_crit)

        opts = render_options(internal)
        self.assertTrue("level 0" in opts[0])
        self.assertTrue("level 1" in opts[1])
        self.assertTrue("level 2" in opts[2])

    def test_6_noul_preservation(self):
        """Test 6: Noul semantics and ordering [false, true] are strictly preserved."""
        qdef = {"type": "noul", "instructions": "Is spam?", "criteria": {"true": "yes", "false": "no"}}
        internal = Agent._to_internal(qdef)
        opts = render_options(internal)
        self.assertEqual(len(opts), 2)
        self.assertTrue(opts[0].startswith("false"))
        self.assertTrue(opts[1].startswith("true"))

    def test_7_list_input(self):
        """Test 7: List input is normalized to dict and canonicalized."""
        qdef = {"type": "choice", "instructions": "Select", "criteria": ["cloud", "database", "security"]}
        internal = Agent._to_internal(qdef)
        self.assertEqual(list(internal["crit"].keys()), ["cloud", "database", "security"])
        self.assertEqual(internal["crit"], {"cloud": None, "database": None, "security": None})

    def test_8_shortlist_behavior(self):
        """Test 8: Shortlist ranking semantics are preserved in metadata while model uses canonical order."""
        class MockAgent:
            def __init__(self):
                self.called_with = None

            def predict(self, state, questions, **kwargs):
                self.called_with = questions
                return {
                    "answers": {
                        "intent": {
                            "type": "choice",
                            "choice": "beta",
                            "confidence": 0.9,
                            "probabilities": {"alpha": 0.1, "beta": 0.9},
                        }
                    }
                }

        agent = MockAgent()
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "Which topic?",
                "criteria": {"gamma": "topic C", "beta": "topic B", "alpha": "topic A"},
            }
        }

        # Ranking: beta (1.0), alpha (0.5), gamma (0.0)
        embeddings = {
            "Which topic?\nstate": [1.0, 0.0],
            "beta: topic B": [1.0, 0.0],
            "alpha: topic A": [0.5, 0.5],
            "gamma: topic C": [0.0, 1.0],
        }

        def mock_embed(texts):
            return [embeddings.get(t, [0.0, 0.0]) for t in texts]

        res = predict_shortlist(agent, "state", questions, mock_embed, k=2)

        # Ranking order is beta, alpha
        self.assertEqual(res["shortlist"]["intent"]["labels"], ["beta", "alpha"])
        self.assertTrue(
            res["shortlist"]["intent"]["scores"][0] > res["shortlist"]["intent"]["scores"][1]
        )

        # Internal model representation canonicalizes the shortlisted set to alpha, beta
        internal = Agent._to_internal(agent.called_with["intent"])
        self.assertEqual(list(internal["crit"].keys()), ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
