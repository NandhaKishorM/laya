"""Laya System 1 decision engine: Tool and Function Selection Quickstart.

Demonstrates:
1. Non-autoregressive tool selection for AI agent workflows (replaces 1-2s LLM tool calling).
2. Direct execution via .call() with function arguments.
3. Direct answer detection when no tool is needed (allow_direct_answer=True).
4. Calibrated confidence threshold fallback gating for ambiguous requests.
5. LangChain LCEL Runnable integration with pipe (|) operator.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.integrations.tools import (
    LayaLowConfidenceError,
    LayaToolSelector,
)


# Optional lightweight demo agent for running without local model checkpoints
class _DemoAgent:
    def predict(self, state, questions, **kwargs):
        text = str(state).lower()
        q_id = next(iter(questions.keys()))

        words = set(text.replace("?", "").replace(",", "").replace("!", "").split())
        if any(w in words for w in ("hello", "hi", "greetings")) or "how are you" in text:
            chosen = "direct_answer"
            conf = 0.98
        elif any(w in words for w in ("search", "weather", "news", "president", "tokyo", "paris")):
            chosen = "tool_0"
            conf = 0.95
        elif any(w in words for w in ("sql", "database", "orders", "users", "postgres", "query", "select")):
            chosen = "tool_2"
            conf = 0.92
        elif any(w in words for w in ("tax", "calculate", "mortgage", "+", "multiply", "math", "sum")):
            chosen = "tool_1"
            conf = 0.94
        else:
            chosen = "tool_0"
            conf = 0.40  # Low confidence on ambiguous input

        return {
            "model": "laya-demo",
            "answers": {
                q_id: {
                    "type": "choice",
                    "choice": chosen,
                    "confidence": conf,
                    "answer_confidence": conf,
                    "probabilities": {chosen: conf},
                }
            },
        }


# Define tool candidates
def search_web(query: str) -> str:
    """Search Google or Bing for real-time news and websites."""
    return f"[Web Results] Found 10 results for query: {query}"


def calculate_tax(amount: float, rate: float = 0.08) -> float:
    """Calculate sales and state taxes on purchase amount."""
    return amount * rate


def query_database(sql: str) -> str:
    """Execute read-only SQL queries against Postgres customer database."""
    return f"[Postgres] Returned 5 rows for: {sql}"


def general_reasoning_fallback(prompt: str) -> str:
    """Fallback handler for ambiguous or complex reasoning requests."""
    return f"[Fallback LLM] Processing complex request: {prompt}"


def main():
    print("=" * 70)
    print(" Laya System 1: Tool and Function Selection")
    print("=" * 70)

    tools = [search_web, calculate_tax, query_database]
    agent = _DemoAgent()

    # 1. Standard Tool Selection
    print("\n[1] Single-Pass Tool Selection:")
    selector = LayaToolSelector(tools=tools, agent=agent)

    queries = [
        "What is the current weather forecast in Tokyo?",
        "Calculate the 8% tax on a $250 purchase",
        "SELECT * FROM orders WHERE status = 'shipped'",
    ]

    for q in queries:
        decision = selector.select(q)
        print(f" Query:      {q}")
        print(f" -> Tool:    {decision.tool_name}")
        print(f" -> Conf:    {decision.confidence:.3f}")
        print(f" -> Reason:  {decision.reason}\n")

    # 2. Direct Execution via .call()
    print("[2] Direct Tool Execution via selector.call():")
    tax_value = selector.call("Calculate sales tax on $500", tools=None, amount=500.0, rate=0.08)
    print(f" Calculated Tax: ${tax_value:.2f}\n")

    # 3. Direct Answer Detection (No Tool Required)
    print("[3] Direct Conversational Answer Detection:")
    selector_direct = LayaToolSelector(tools=tools, allow_direct_answer=True, agent=agent)
    decision = selector_direct.select("Hello! Can you help me today?")
    if decision.is_direct_answer:
        print(" -> Detected conversational query: Answer directly without calling tools!\n")

    # 4. Confidence Threshold & Fallbacks
    print("[4] Confidence Gating & Fallback Dispatch:")
    ambiguous_query = "Summarize the philosophical differences between Kant and Hegel"

    # With fallback tool
    selector_fb = LayaToolSelector(
        tools=tools,
        confidence_threshold=0.80,
        fallback_tool=general_reasoning_fallback,
        agent=agent,
    )
    decision = selector_fb.select(ambiguous_query)
    print(f" Query: {ambiguous_query}")
    print(f" -> Selected:   {decision.tool_name}")
    print(f" -> Confidence: {decision.confidence:.3f} (< 0.80 threshold)")
    print(f" -> Fallback:   {decision.is_fallback}\n")

    # Without fallback tool: raises LayaLowConfidenceError
    selector_gate = LayaToolSelector(tools=tools, confidence_threshold=0.80, agent=agent)
    try:
        selector_gate.select(ambiguous_query)
    except LayaLowConfidenceError as e:
        print(f" -> Successfully caught LayaLowConfidenceError: {e}\n")

    # 5. LangChain LCEL Composition
    print("[5] LangChain LCEL Runnable Composition:")
    chain = selector.as_runnable() | (lambda d: f"Chain routed to: {d.tool_name}")
    routed_output = chain.invoke("Search for the latest tech news")
    print(f" -> {routed_output}\n")

    print("=" * 70)
    print(" Laya Tool Selection Quickstart completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
