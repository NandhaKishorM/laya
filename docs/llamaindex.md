# LlamaIndex Integration

Laya provides sub-35ms, non-autoregressive decision components for **LlamaIndex** RAG pipelines, `RouterQueryEngine`, and tool selection (single-question latency measured at **32.8 ms** with `laya-multilingual` and **39.5 ms** with `laya` on a Tesla T4 GPU; 193–464 ms on CPU):

* **`LayaSingleSelector`**: Sub-35ms single-choice selector replacing `LLMSingleSelector` for `RouterQueryEngine`.
* **`LayaMultiSelector`**: Multi-choice selector replacing `LLMMultiSelector` for composite queries spanning multiple data sources.
* **`LayaQueryRouter`**: Standalone query dispatcher routing incoming requests directly to target query engines or callables.

Supports both **local in-process inference** (`Agent` or `Router`) and **remote HTTP inference** against your own `laya-serve` instance without requiring PyTorch on edge clients.

---

## Installation

```bash
pip install "laya[llamaindex]"
```

---

## 1. Single-Choice Routing with `RouterQueryEngine`

In LlamaIndex, `RouterQueryEngine` uses a selector to decide which underlying query engine or tool should answer a question. Autoregressive LLM selectors (`LLMSingleSelector`) take 1,000–2,000 ms generating text. `LayaSingleSelector` evaluates candidate tools in **~33 ms** without token generation:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from laya.integrations.llamaindex import LayaSingleSelector

# Define query engine tools
docs_tool = QueryEngineTool(
    query_engine=vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_documentation",
        description="Semantic search over technical user documentation and API guides.",
    ),
)
sql_tool = QueryEngineTool(
    query_engine=sql_index.as_query_engine(),
    metadata=ToolMetadata(
        name="sql_database",
        description="Structured SQL database containing customer accounts, billing, and orders.",
    ),
)

# Initialize Laya sub-35ms selector with confidence fallback
selector = LayaSingleSelector(
    confidence_threshold=0.80,   # If confidence < 0.80, fall back to index 0
    fallback_index=0,
)

router_engine = RouterQueryEngine(
    selector=selector,
    query_engine_tools=[docs_tool, sql_tool],
)

response = router_engine.query("What is the shipping address for order #4912?")
print(response)
```

---

## 2. Multi-Choice Selection for Composite Queries

For queries that require synthesis across multiple indexes (e.g. comparing documentation specs with transactional database records), `LayaMultiSelector` evaluates candidate relevance and returns multiple selected tools:

```python
from laya.integrations.llamaindex import LayaMultiSelector

multi_selector = LayaMultiSelector(
    probability_threshold=0.25,  # Select all tools with probability >= 0.25
    max_outputs=2,
)

tools = [docs_tool.metadata, sql_tool.metadata, summary_tool.metadata]
result = multi_selector.select(
    tools,
    "How does the database security policy compare with our published compliance guide?"
)

for sel in result.selections:
    print(f"Tool: {tools[sel.index].name} | {sel.reason}")
```

---

## 3. Direct Query Dispatch with `LayaQueryRouter`

For direct routing without the overhead of `RouterQueryEngine`, `LayaQueryRouter` routes queries directly to dictionary-registered engines:

```python
from laya.integrations.llamaindex import LayaQueryRouter

router = LayaQueryRouter(
    query_engines={
        "vector": vector_query_engine,
        "sql": sql_query_engine,
        "summary": summary_query_engine,
    },
    descriptions={
        "vector": "Semantic search over product documentation and guides",
        "sql": "Structured SQL queries for user accounts and transactions",
        "summary": "Quarterly reports and high-level business summaries",
    },
    confidence_threshold=0.75,
    fallback_key="vector",
)

# Route and execute in one call:
response = router.query("How many active subscriptions were renewed in Q3?")
print(response)
```

Both synchronous `query()` and asynchronous `aquery()` are supported.

---

## 4. Confidence Threshold Gating

Like Laya's LangChain integration, `LayaSingleSelector` and `LayaQueryRouter` read calibrated `answer_confidence` (`max(p)`):

- **Automatic Fallback:** Specify `fallback_index` (or `fallback_key`) to seamlessly divert uncertain queries to a safe default engine.
- **Strict Guarding:** Set `raise_on_low_confidence=True` on `LayaSingleSelector` to raise `LayaLowConfidenceError` when input is ambiguous, allowing caller escalation.

---

## 5. Remote HTTP Deployments

For serverless RAG, edge environments, or environments without local GPUs:

```python
from laya.integrations.llamaindex import LayaSingleSelector

selector = LayaSingleSelector(
    base_url="http://laya-serve.internal:8080",
    confidence_threshold=0.85,
    fallback_index=0,
)
```

The remote client uses Python's standard library `urllib` with zero heavy dependencies, preventing cross-origin credential forwarding and matching the `/v1/systemone` specification.
