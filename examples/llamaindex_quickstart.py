"""Laya System 1 decision engine: LlamaIndex Integration Quickstart.

Demonstrates:
1. Sub-35ms single-choice tool routing in LlamaIndex with confidence fallback gating.
2. Multi-choice routing across multiple RAG query engines for composite questions.
3. Direct high-performance query dispatch with LayaQueryRouter.
4. Edge/Serverless deployment against remote laya-serve HTTP instances.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.integrations.llamaindex import (
    LayaMultiSelector,
    LayaQueryRouter,
    LayaSingleSelector,
    QueryBundle,
    ToolMetadata,
)

# Optional mock agent for environments without local GPU / PyTorch weights:
class _DemoAgent:
    def predict(self, state, questions, **kwargs):
        text = str(state).lower()
        q_id = next(iter(questions.keys()))
        criteria = questions[q_id].get("criteria", {})
        
        # Route database/orders queries to sql_database or database
        if any(w in text for w in ("order", "orders", "account", "database", "balance", "user")):
            chosen = "choice_1" if "choice_1" in criteria else "database"
            probs = {"choice_0": 0.08, "choice_1": 0.88, "choice_2": 0.04}
        elif "both" in text or "compare" in text or "security" in text:
            chosen = "choice_0" if "choice_0" in criteria else "docs"
            probs = {"choice_0": 0.52, "choice_1": 0.44, "choice_2": 0.04}
        else:
            chosen = "choice_0" if "choice_0" in criteria else "docs"
            probs = {"choice_0": 0.91, "choice_1": 0.06, "choice_2": 0.03}
        return {
            "model": "laya-demo",
            "answers": {
                q_id: {
                    "choice": chosen,
                    "answer_confidence": probs.get(chosen, 0.90),
                    "confidence": probs.get(chosen, 0.90),
                    "probabilities": probs,
                }
            },
        }

_agent = None
try:
    from laya import Router
    r = Router()
    # Test if models/dependencies are present
    r.predict("test", {"test": {"type": "choice", "instructions": "test", "criteria": {"a": "a"}}})
    _agent = r
except Exception:
    _agent = _DemoAgent()

# =====================================================================
# 1. Sub-35ms Single-Choice RAG Selector (Replaces LLMSingleSelector)
# =====================================================================
# Evaluates query against candidate query engine tools in ~33 ms without
# burning tokens or waiting 1-2 seconds for autoregressive LLM completion.

tools = [
    ToolMetadata(
        name="vector_documentation",
        description="Semantic search over technical user documentation and API guides.",
    ),
    ToolMetadata(
        name="sql_database",
        description="Structured SQL database containing customer accounts, billing, and orders.",
    ),
    ToolMetadata(
        name="summary_index",
        description="High-level summaries and quarterly reports of company financial performance.",
    ),
]

# Configure selector with confidence threshold fallback
selector = LayaSingleSelector(
    confidence_threshold=0.80,
    fallback_index=0,  # Fallback to vector documentation if uncertain
    agent=_agent,
)

query = "Can you show me the total number of orders placed by customer #1042 last month?"
print(f"--- 1. Single-Choice Selector ---")
print(f"Query: {query}")

result = selector.select(tools, query)
selected_tool = tools[result.selections[0].index]
print(f"Selected Tool: {selected_tool.name} (Index: {result.selections[0].index})")
print(f"Reason: {result.selections[0].reason}")


# =====================================================================
# 2. Multi-Choice Selector for Composite Questions (Replaces LLMMultiSelector)
# =====================================================================
# Identifies which subset of candidate tools are relevant for complex queries.

multi_selector = LayaMultiSelector(
    probability_threshold=0.25,
    max_outputs=2,
    agent=_agent,
)

composite_query = (
    "How does the new authentication architecture documented in the guides impact "
    "our customer account database security settings?"
)
print(f"\n--- 2. Multi-Choice Selector ---")
print(f"Composite Query: {composite_query}")

multi_result = multi_selector.select(tools, composite_query)
print(f"Selected {len(multi_result.selections)} relevant tools:")
for sel in multi_result.selections:
    print(f"  -> Tool: {tools[sel.index].name} | {sel.reason}")


# =====================================================================
# 3. Direct High-Performance Query Dispatch with LayaQueryRouter
# =====================================================================
# Dispatches incoming queries directly to backend query engines or functions.

class DummyQueryEngine:
    def __init__(self, name: str):
        self.name = name

    def query(self, query_str: str) -> str:
        return f"[{self.name}] Synthesized response for: {query_str}"


engines = {
    "docs": DummyQueryEngine("DocsEngine"),
    "database": DummyQueryEngine("SQLEngine"),
    "finance": DummyQueryEngine("FinanceEngine"),
}

router = LayaQueryRouter(
    query_engines=engines,
    descriptions={
        "docs": "Documentation and setup tutorials",
        "database": "User transactions and database queries",
        "finance": "Quarterly earnings and financial statements",
    },
    confidence_threshold=0.75,
    fallback_key="docs",
    agent=_agent,
)

db_query = "What is the account balance for user Alice?"
routed_key = router.route(db_query)
response = router.query(db_query)
print(f"\n--- 3. LayaQueryRouter Direct Dispatch ---")
print(f"Query: {db_query}")
print(f"Routed Engine Key: -> {routed_key}")
print(f"Engine Response: {response}")


# =====================================================================
# 4. Remote HTTP Server Execution (Zero Heavy Dependencies)
# =====================================================================
# Connects to your self-hosted `laya-serve` instance without requiring PyTorch on edge clients.
remote_selector = LayaSingleSelector(
    base_url="http://localhost:8080",
    confidence_threshold=0.85,
    fallback_index=0,
)
print(f"\n--- 4. Remote HTTP Server Deployment ---")
print("Remote selector configured with base_url='http://localhost:8080' via standard urllib.")
