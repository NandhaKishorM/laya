"""P0 #2/#3: lifting the option ceiling and covering documents longer than the window."""
import pytest

from conftest import FakeAgent, choice, noul, score
from laya import patterns
from laya.errors import InvalidCriteriaError

NOUL_Q = {"type": "noul", "instructions": "spam?"}


def pick(name, conf=0.9):
    """Stub selecting `name`, or the group/branch leading to it, else the first option.

    Stage-1 options are group labels whose descriptions list their members, so matching on
    the description is what lets the stub follow the target down into the right group.
    """
    def inner(state, questions):
        q = next(iter(questions.values()))
        crit = q["criteria"]
        opts = list(crit)
        winner = None
        if name in crit:
            winner = name
        else:
            for opt, desc in crit.items():
                if isinstance(desc, str) and name in desc:
                    winner = opt
                    break
        winner = winner or opts[0]
        return {next(iter(questions)): choice(winner, {o: 1.0 / len(opts) for o in opts}, conf=conf)}
    return inner


class TestTwoStageChoice:
    def test_keeps_per_call_options_small(self):
        """The point of the pattern: 256 options, but no single call sees them all."""
        options = {"opt_%03d" % i: "desc %d" % i for i in range(256)}
        agent = FakeAgent(pick("opt_200"))
        patterns.two_stage_choice(agent, "s", "pick", options, group_size=16)
        for _, questions in agent.calls:
            assert len(next(iter(questions.values()))["criteria"]) <= 16

    def test_finds_option_in_a_later_group(self):
        options = {"opt_%03d" % i: "d" for i in range(64)}
        agent = FakeAgent(pick("opt_050"))
        out = patterns.two_stage_choice(agent, "s", "pick", options, group_size=16)
        assert out["choice"] == "opt_050"

    def test_uses_two_calls(self):
        options = {"o%d" % i: "d" for i in range(64)}
        agent = FakeAgent(pick("o5"))
        patterns.two_stage_choice(agent, "s", "pick", options, group_size=8)
        assert len(agent.calls) == 2

    def test_confidence_is_joint(self):
        options = {"o%d" % i: "d" for i in range(32)}
        agent = FakeAgent(pick("o1", conf=0.8))
        out = patterns.two_stage_choice(agent, "s", "pick", options, group_size=8)
        assert out["confidence"] == pytest.approx(0.64, abs=1e-4), "0.8 * 0.8"

    def test_single_group_asks_directly(self):
        options = {"a": "x", "b": "y"}
        agent = FakeAgent(pick("b"))
        out = patterns.two_stage_choice(agent, "s", "pick", options, group_size=16)
        assert out["choice"] == "b"
        assert len(agent.calls) == 1, "no need to narrow a single group"

    def test_custom_grouping(self):
        options = {"cat_a": "d", "cat_b": "d", "dog_a": "d", "dog_b": "d"}
        agent = FakeAgent(pick("dog_a"))
        out = patterns.two_stage_choice(
            agent, "s", "pick", options, group_of=lambda name, crit: name.split("_")[0]
        )
        assert out["group"] == "dog" and out["choice"] == "dog_a"

    def test_rejects_empty_options(self):
        with pytest.raises(InvalidCriteriaError):
            patterns.two_stage_choice(FakeAgent(), "s", "pick", {})

    def test_rejects_tiny_group_size(self):
        with pytest.raises(ValueError):
            patterns.two_stage_choice(FakeAgent(), "s", "pick", {"a": "b"}, group_size=1)


class TestTaxonomyChoice:
    def test_walks_to_a_leaf(self):
        tree = {"tech": {"bug": "a bug", "outage": "an outage"}, "billing": {"refund": "refund"}}
        agent = FakeAgent(pick("outage"))
        out = patterns.taxonomy_choice(agent, "s", "classify", tree)
        assert out["path"] == ["tech", "outage"]
        assert out["choice"] == "outage"

    def test_joint_confidence_across_levels(self):
        tree = {"a": {"a1": "x", "a2": "y"}}
        agent = FakeAgent(pick("a1", conf=0.9))
        out = patterns.taxonomy_choice(agent, "s", "classify", tree)
        assert out["confidence"] == pytest.approx(0.81, abs=1e-4)

    def test_flat_tree_is_one_level(self):
        agent = FakeAgent(pick("b"))
        out = patterns.taxonomy_choice(agent, "s", "classify", {"a": "x", "b": "y"})
        assert out["path"] == ["b"] and len(agent.calls) == 1

    def test_depth_is_bounded(self):
        deep = cur = {}
        node = deep
        for i in range(20):
            nxt = {}
            node["level%d" % i] = nxt
            node = nxt
        node["leaf"] = "end"
        agent = FakeAgent(pick("nonexistent"))
        out = patterns.taxonomy_choice(agent, "s", "classify", deep, max_depth=3)
        assert len(out["path"]) <= 3

    def test_rejects_empty_tree(self):
        with pytest.raises(InvalidCriteriaError):
            patterns.taxonomy_choice(FakeAgent(), "s", "classify", {})


class TestChunkState:
    def test_short_text_is_one_chunk(self):
        assert patterns.chunk_state("hello", chunk_chars=100) == ["hello"]

    def test_long_text_splits(self):
        assert len(patterns.chunk_state("x" * 5000, chunk_chars=1000, overlap=0)) == 5

    def test_chunks_respect_size(self):
        for c in patterns.chunk_state("y" * 5000, chunk_chars=700, overlap=50):
            assert len(c) <= 700

    def test_overlap_preserves_boundary_context(self):
        text = "".join(str(i % 10) for i in range(3000))
        chunks = patterns.chunk_state(text, chunk_chars=1000, overlap=100)
        assert chunks[0][-100:] == chunks[1][:100]

    def test_full_coverage(self):
        text = "".join(str(i % 10) for i in range(2500))
        chunks = patterns.chunk_state(text, chunk_chars=600, overlap=50)
        assert "".join(dict.fromkeys(chunks)) and text[-10:] in chunks[-1]

    def test_rejects_overlap_past_chunk(self):
        """Only matters once the text is long enough to actually need splitting."""
        with pytest.raises(ValueError, match="cannot advance"):
            patterns.chunk_state("x" * 100, chunk_chars=10, overlap=10)

    def test_short_text_ignores_overlap(self):
        """Default overlap must not error on text smaller than the chunk size."""
        assert patterns.chunk_state("hi", chunk_chars=50) == ["hi"]

    def test_empty_text(self):
        assert patterns.chunk_state("") == [""]


class TestMapReduce:
    def test_max_takes_strongest_signal(self):
        vals = iter([0.1, 0.95, 0.2])
        agent = FakeAgent(lambda s, q: {"spam": noul(next(vals))})
        out = patterns.map_reduce(agent, "x" * 3000, {"spam": NOUL_Q}, chunk_chars=1000, overlap=0)
        assert out["answers"]["spam"]["noul"] == pytest.approx(0.95)

    def test_mean_averages(self):
        vals = iter([0.0, 1.0])
        agent = FakeAgent(lambda s, q: {"spam": noul(next(vals))})
        out = patterns.map_reduce(
            agent, "x" * 2000, {"spam": NOUL_Q}, chunk_chars=1000, overlap=0, reduce="mean"
        )
        assert out["answers"]["spam"]["noul"] == pytest.approx(0.5)

    def test_reports_chunk_count(self):
        agent = FakeAgent(lambda s, q: {"spam": noul(0.5)})
        out = patterns.map_reduce(agent, "x" * 3000, {"spam": NOUL_Q}, chunk_chars=1000, overlap=0)
        assert out["chunks"] == 3 and len(out["per_chunk"]) == 3

    def test_score_reduction_keeps_legend(self):
        agent = FakeAgent(lambda s, q: {"sev": score(2.0)})
        out = patterns.map_reduce(
            agent, "x" * 2000, {"sev": {"type": "score", "instructions": "i", "criteria": ["a", "b"]}},
            chunk_chars=1000, overlap=0,
        )
        assert "legend" in out["answers"]["sev"]

    def test_choice_reduces_by_confidence(self):
        confs = iter([0.4, 0.95])
        agent = FakeAgent(lambda s, q: {"cat": choice("winner" if (c := next(confs)) > 0.9 else "loser",
                                                      {"winner": 1.0}, conf=c)})
        out = patterns.map_reduce(
            agent, "x" * 2000, {"cat": {"type": "choice", "instructions": "i", "criteria": {"a": None}}},
            chunk_chars=1000, overlap=0,
        )
        assert out["answers"]["cat"]["choice"] == "winner"

    def test_rejects_bad_reduce(self):
        with pytest.raises(ValueError):
            patterns.map_reduce(FakeAgent(), "text", {"x": NOUL_Q}, reduce="median")


class TestTwoStageGroupDescriptions:
    """Regressions from the GPU eval: group descriptions used to crowd out the state."""

    def test_group_description_stays_short(self):
        """With 400 options, listing every member left zero room for the state."""
        options = {"option_with_a_long_name_%03d" % i: "d" for i in range(400)}
        agent = FakeAgent(pick("option_with_a_long_name_200"))
        patterns.two_stage_choice(agent, "s", "pick", options, group_size=16)
        for _, questions in agent.calls:
            for desc in next(iter(questions.values()))["criteria"].values():
                assert desc is None or len(desc) < 200, "descriptions must stay bounded"

    def test_many_groups_recurse_instead_of_overflowing(self):
        """400 options / 16 per group = 25 groups, more than fit one head."""
        options = {"opt_%03d" % i: "d" for i in range(400)}
        agent = FakeAgent(pick("opt_200"))
        patterns.two_stage_choice(agent, "s", "pick", options, group_size=16)
        for _, questions in agent.calls:
            assert len(next(iter(questions.values()))["criteria"]) <= 16

    def test_semantic_grouping_preserved(self):
        """group_of is the accuracy-preserving path; it must keep working."""
        options = {}
        for fam in ("billing", "outage", "shipping"):
            for i in range(8):
                options["%s_%d" % (fam, i)] = "%s issue %d" % (fam, i)
        agent = FakeAgent(pick("outage_3"))
        out = patterns.two_stage_choice(
            agent, "s", "pick", options, group_of=lambda n, c: n.split("_")[0]
        )
        assert out["group"] == "outage" and out["choice"] == "outage_3"
