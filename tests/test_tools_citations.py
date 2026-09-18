"""P2 completion: function calling (select_tool) and citation checking."""
import pytest

from conftest import FakeAgent, choice, noul
from laya import patterns
from laya.errors import InvalidCriteriaError, QuestionError

TOOLS = {
    "search_web": "look something up on the internet",
    "send_email": "compose and send an email",
    "run_query": "run a SQL query against the warehouse",
}


def pick_tool(name, conf=0.9):
    def inner(state, questions):
        q = next(iter(questions.values()))
        opts = list(q["criteria"])
        winner = name if name in opts else opts[0]
        return {next(iter(questions)): choice(winner, {o: 1.0 / len(opts) for o in opts}, conf=conf)}
    return inner


class TestSelectTool:
    def test_picks_the_named_tool(self):
        agent = FakeAgent(pick_tool("run_query"))
        out = patterns.select_tool(agent, "how many users signed up?", TOOLS)
        assert out["tool"] == "run_query" and out["should_call"]

    def test_none_option_is_offered(self):
        agent = FakeAgent(pick_tool("search_web"))
        patterns.select_tool(agent, "hello", TOOLS)
        crit = next(iter(agent.calls[0][1].values()))["criteria"]
        assert "no_tool" in crit, "model must be able to decline"

    def test_declining_returns_no_tool(self):
        agent = FakeAgent(pick_tool("no_tool"))
        out = patterns.select_tool(agent, "just chatting", TOOLS)
        assert out["tool"] is None and out["declined"] and not out["should_call"]

    def test_none_option_can_be_disabled(self):
        agent = FakeAgent(pick_tool("search_web"))
        patterns.select_tool(agent, "x", TOOLS, none_option=None)
        crit = next(iter(agent.calls[0][1].values()))["criteria"]
        assert set(crit) == set(TOOLS)

    def test_low_confidence_blocks_the_call(self):
        agent = FakeAgent(pick_tool("send_email", conf=0.3))
        out = patterns.select_tool(agent, "x", TOOLS, min_confidence=0.8)
        assert out["tool"] == "send_email"
        assert not out["should_call"], "must not fire a tool on a coin flip"

    def test_schema_is_returned_untouched(self):
        schema = {"description": "run a SQL query", "parameters": {"sql": {"type": "string"}}}
        agent = FakeAgent(pick_tool("run_query"))
        out = patterns.select_tool(agent, "x", {"run_query": schema})
        assert out["schema"] == schema

    def test_schema_description_is_used_as_criterion(self):
        schema = {"description": "run a SQL query", "parameters": {}}
        agent = FakeAgent(pick_tool("run_query"))
        patterns.select_tool(agent, "x", {"run_query": schema})
        crit = next(iter(agent.calls[0][1].values()))["criteria"]
        assert crit["run_query"] == "run a SQL query"

    def test_many_tools_route_through_two_stage(self):
        many = {"tool_%03d" % i: "does thing %d" % i for i in range(200)}
        agent = FakeAgent(pick_tool("tool_100"))
        patterns.select_tool(agent, "x", many)
        for _, questions in agent.calls:
            assert len(next(iter(questions.values()))["criteria"]) <= 96

    def test_rejects_empty_tools(self):
        with pytest.raises(InvalidCriteriaError):
            patterns.select_tool(FakeAgent(), "x", {})

    def test_rejects_none_option_collision(self):
        with pytest.raises(InvalidCriteriaError, match="collides"):
            patterns.select_tool(FakeAgent(), "x", {"no_tool": "a real tool"})


def cite(support, contradict, relevant=None, conf=0.9):
    """Stub the three questions check_citation asks. Relevance defaults to tracking support."""
    def inner(state, questions):
        # A source that supports OR contradicts a claim is by definition on-topic.
        rel = relevant if relevant is not None else max(support, contradict)
        return {
            "supported": noul(support, conf=conf),
            "contradicted": noul(contradict),
            "relevant": noul(rel),
        }
    return inner


class TestCheckCitation:
    def test_supported(self):
        agent = FakeAgent(cite(0.95, 0.02))
        out = patterns.check_citation(agent, "The sky is blue.", "The sky appears blue.")
        assert out["verdict"] == "supported"

    def test_contradicted(self):
        agent = FakeAgent(cite(0.05, 0.93))
        out = patterns.check_citation(agent, "Revenue rose.", "Revenue fell by 12%.")
        assert out["verdict"] == "contradicted"

    def test_unsupported_is_distinct_from_contradicted(self):
        """A source that simply doesn't mention the claim is a different failure."""
        agent = FakeAgent(cite(0.1, 0.1, relevant=0.1))
        out = patterns.check_citation(agent, "Revenue rose.", "The cafeteria menu changed.")
        assert out["verdict"] == "unsupported"

    def test_contradiction_wins_when_both_are_high(self):
        agent = FakeAgent(cite(0.6, 0.9))
        assert patterns.check_citation(agent, "c", "s")["verdict"] == "contradicted"

    def test_support_wins_when_it_dominates(self):
        agent = FakeAgent(cite(0.9, 0.6))
        assert patterns.check_citation(agent, "c", "s")["verdict"] == "supported"

    def test_threshold_is_respected(self):
        agent = FakeAgent(cite(0.6, 0.0))
        assert patterns.check_citation(agent, "c", "s", threshold=0.5)["verdict"] == "supported"
        assert patterns.check_citation(agent, "c", "s", threshold=0.8)["verdict"] == "unsupported"

    def test_claim_and_source_both_reach_the_model(self):
        """Inline premise/hypothesis text, not a JSON dict -- see the docstring on why."""
        agent = FakeAgent(cite(0.9, 0.0))
        patterns.check_citation(agent, "my claim", "my source")
        state = agent.calls[0][0]
        assert isinstance(state, str)
        assert "my claim" in state and "my source" in state

    def test_offtopic_source_is_not_called_contradicted(self):
        """The checkpoint scores unrelated sources high on contradiction; relevance gates it."""
        agent = FakeAgent(cite(0.0, 0.8, relevant=0.0))
        assert patterns.check_citation(agent, "c", "s")["verdict"] == "unsupported"

    def test_decisive_contradiction_survives_low_relevance(self):
        """A very strong contradiction is not downgraded by a weak relevance signal."""
        agent = FakeAgent(cite(0.02, 0.97, relevant=0.02))
        assert patterns.check_citation(agent, "c", "s")["verdict"] == "contradicted"

    def test_rejects_empty_claim(self):
        with pytest.raises(QuestionError):
            patterns.check_citation(FakeAgent(), "   ", "source")

    def test_rejects_bad_threshold(self):
        with pytest.raises(ValueError):
            patterns.check_citation(FakeAgent(), "c", "s", threshold=2.0)


class TestCheckCitations:
    def test_flags_only_the_failures(self):
        vals = iter([(0.95, 0.0), (0.1, 0.1), (0.9, 0.0)])

        def inner(state, questions):
            s, c = next(vals)
            return {"supported": noul(s), "contradicted": noul(c), "relevant": noul(s)}

        out = patterns.check_citations(FakeAgent(inner), [("a", "x"), ("b", "y"), ("c", "z")])
        assert out["unsupported"] == [1]
        assert not out["all_supported"]
        assert out["n_checked"] == 3

    def test_all_supported(self):
        agent = FakeAgent(cite(0.95, 0.0))
        out = patterns.check_citations(agent, [("a", "x"), ("b", "y")])
        assert out["all_supported"] and out["unsupported"] == []

    def test_empty_input(self):
        out = patterns.check_citations(FakeAgent(), [])
        assert out["n_checked"] == 0 and out["all_supported"]
