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

---

## 6. Per-call decision controls

`LayaSingleSelector`, `LayaMultiSelector` and `LayaQueryRouter` take the same per-call arguments the
core API does: the two token budgets (`max_len`, `head_max_len`), the language and abstention
controls (`lang`, `min_confidence`), and the five prediction-hook
arguments (`hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`, `hooks_timeout`). They are
per selector, so a wide routing step can be given room while the rest of the pipeline keeps the
checkpoint's defaults.

A choice question's options share the checkpoint's *option* budget -- `head_max_len`, 192 tokens on
`laya` -- and every candidate contributes its name and description, so past roughly 20 tools the
descriptions start reaching the model as the same text.

```python
selector = LayaSingleSelector(
    instructions="Which tool or query engine is best suited to answer this query?",
    max_len=1024,          # total window
    head_max_len=512,      # tokens shared by the option prompt
)
result = selector.select(tools, query)     # tools: 59 descriptions
```

Measured on `laya` (Apple silicon, one forward pass per query, scored on the chosen tool) with a
59-tool roster built from the MASSIVE en intent labels and one utterance per label, so ground truth
is exact. Each cell is how many of the 59 queries reached their own tool; both repeats gave the
same count.

| 59-tool roster | Default budget | `max_len=1024, head_max_len=384` | `…, head_max_len=512` |
|---|---|---|---|
| Queries on their own tool | 2/59 | 8/59 | 15/59 |
| Median ms per query | 160 | 172 | 184 |

Absolute accuracy is not the claim here: the checkpoint is not a MASSIVE classifier, and 59 similar
labels are a stress shape. The claim is the direction and the price -- a roster the default budget
collapses to near-nothing is readable, and at this size the window costs little time. With fewer
than about 20 options the labels already fit and widening can move answers the wrong way, which is
why both arguments are opt-in per selector. See the [LangChain
integration](langchain.md#7-widening-the-token-budget-for-many-options) for that measured cliff.

**Hooks run on the local path only.** A selector with a `base_url` and `hooks=[...]` raises
`ValueError` rather than reporting a success whose cache never ran -- a hook is a Python callable
that runs inside `predict`, and no wire format carries it. Install hooks in the process that runs
inference. The two budgets do travel to a remote node, in the request body, up to its
`LAYA_MAX_TOKEN_BUDGET` ceiling; a larger value comes back as a 422.

### Language and abstention

`lang` pins the language the query is routed and answered in -- selecting the answering
checkpoint's per-language calibration instead of relying on built-in detection -- and
`min_confidence` is core's abstention gate: a decision under it comes back as an abstention rather
than a forced selection. Both are read by `Agent.predict` and `Router.predict` alike and accepted by
`laya-serve` in the request body, so a selector forwards them on the local and the remote path. An
unset one is omitted, not sent as `None`, so it cannot shadow the deployment's own default;
`min_confidence=0.0` and `lang=""` are real values and are forwarded as given.

```python
selector = LayaSingleSelector(
    instructions="Which tool or query engine is best suited to answer this query?",
    lang="de",             # answer German queries in German
    min_confidence=0.3,    # abstain when no tool clears a 0.3 confidence
)
```
