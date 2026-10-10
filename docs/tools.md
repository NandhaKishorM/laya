# Tool & Function Selection

Laya provides non-autoregressive tool routing and function selection for AI agent workflows across **LangChain**, **CrewAI**, **LlamaIndex**, and vanilla agent architectures (single-question latency measured at **32.8 ms** with `laya-multilingual` and **39.5 ms** with `laya` on a Tesla T4 GPU; **p50 125.6 ms** on Apple Silicon; **193–464 ms** on CPU):

* **`LayaToolSelector`**: Tool routing component evaluating user queries and agent states against candidate tools in a single forward pass (~33–40 ms on Tesla T4 GPU, ~88–464 ms on CPU/Apple Silicon).
* **`ToolRouteDecision`**: Structured routing result carrying the selected tool, tool index, calibrated confidence, probability distribution across tools, and fallback status.

Replaces slow, token-generating LLM tool-calling (1,000–2,500 ms) with deterministic, typed decisions executed in a single forward pass with **zero token generation cost**.

Supports both **local in-process inference** (`Agent` or `Router`) and **remote HTTP inference** against your own `laya-serve` instance without requiring PyTorch on edge clients.

---

## Installation

```bash
pip install laya
```

---

## 1. Single-Pass Tool Selection

Autoregressive LLMs (such as GPT-4o or Claude 3.5 Sonnet) take 1–2 seconds generating reasoning tokens just to choose which tool to call. `LayaToolSelector` evaluates candidate tools against the query in a single forward pass (~33–40 ms on Tesla T4 GPU, ~88–126 ms on Apple Silicon, and 193–464 ms on CPU):

```python
from laya.integrations.tools import LayaToolSelector

def search_web(query: str):
    """Search Google or Bing for real-time news and websites."""
    return f"Search results for: {query}"

def calculate_mortgage(principal: float, rate: float, years: int):
    """Calculate monthly mortgage amortization payments and interest."""
    return "Monthly payment: $2,450"

def query_database(sql: str):
    """Execute read-only SQL queries against Postgres customer database."""
    return "Rows returned: 12"

tools = [search_web, calculate_mortgage, query_database]

# Initialize tool selector
selector = LayaToolSelector(tools=tools)

# Select the matching tool:
decision = selector.select("What is the current mortgage rate on a 30-year fixed loan?")
print(f"Selected Tool: {decision.tool_name}")
print(f"Confidence:    {decision.confidence:.3f}")
print(f"Probabilities: {decision.probabilities}")
```

Both synchronous `select()` and non-blocking asynchronous `aselect()` are supported.

---

## 2. Direct Execution (`call`)

`call()` selects the candidate tool and immediately executes it with the supplied arguments:

```python
result = selector.call("Calculate payments for 500k at 6.5% for 30 years",
                       principal=500000, rate=0.065, years=30)
print(result)  # "Monthly payment: $2,450"
```

If the tool is a LangChain tool with `.run()`, `call()` automatically invokes its execution method.

---

## 3. Direct Answer Detection (No Tool Needed)

In conversational agent pipelines, many user queries do not require external tool invocation (e.g. greetings, general conversational questions). Setting `allow_direct_answer=True` enables Laya to recognize when to answer conversationally without calling any tool:

```python
selector = LayaToolSelector(
    tools=[search_web, query_database],
    allow_direct_answer=True,
)

decision = selector.select("Hello! How can you help me today?")
if decision.is_direct_answer:
    print("No tool needed: respond conversationally.")
else:
    print(f"Call tool: {decision.tool_name}")
```

---

## 4. Calibrated Confidence Gating & Fallbacks

`LayaToolSelector` uses Laya's calibrated confidence scores to gate routing decisions. When an ambiguous or out-of-domain query is presented, the selector avoids bad tool invocations by falling back to a general agent or raising `LayaLowConfidenceError`:

```python
from laya.integrations._errors import LayaLowConfidenceError

def general_fallback_llm(prompt: str):
    """Fallback handler for ambiguous or novel requests."""
    return "Dispatching to general reasoning LLM..."

selector = LayaToolSelector(
    tools=[search_web, calculate_mortgage],
    confidence_threshold=0.85,
    fallback_tool=general_fallback_llm,
)

decision = selector.select("Compose a haiku about autumn leaves.")
if decision.is_fallback:
    print(f"Ambiguous request ({decision.confidence:.2f} < 0.85). Routed to fallback handler.")
```

If `fallback_tool` is omitted, requests below threshold raise `LayaLowConfidenceError`:

```python
try:
    decision = selector.select("Unclear request")
except LayaLowConfidenceError as e:
    print(f"Abstained: confidence {e.confidence:.3f} below threshold {e.threshold}")
```

---

## 5. LangChain LCEL Integration

`LayaToolSelector` exports an LCEL-compatible Runnable via `.as_runnable()`, allowing it to be composed seamlessly into LangGraph nodes or LangChain pipelines:

```python
from laya.integrations.tools import LayaToolSelector

selector = LayaToolSelector(tools=[search_web, query_database])

# Compose with pipe operator |
chain = selector.as_runnable() | (lambda decision: decision.tool_name)
tool_to_call = chain.invoke("Look up order #45920 in the database")
print(f"Dispatching to: {tool_to_call}")
```

---

## 6. Remote HTTP Mode (`laya-serve`)

In serverless or microservice architectures, `LayaToolSelector` connects directly to a self-hosted `laya-serve` cluster using standard library `urllib` (no external HTTP client required):

```python
selector = LayaToolSelector(
    tools=[search_web, query_database],
    base_url="http://laya-cluster.internal:8000",
    api_key="your-cluster-secret",
    model="laya-multilingual",
)

decision = selector.select("Search for latest updates")
```

---

## 7. Supported Tool Formats

`LayaToolSelector` automatically parses candidate tools from all standard formats:
* **Python functions**: extracts `__name__` and first line of docstring `__doc__`.
* **LangChain & CrewAI tools**: reads `.name` and `.description`.
* **LlamaIndex tools**: reads `tool.metadata.name` and `tool.metadata.description`.
* **JSON Schema / OpenAI specs**: parses `{"type": "function", "function": {"name": ..., "description": ...}}`.
* **Dictionaries**: reads `{"name": ..., "description": ...}`.
* **Plain strings**: parses `"tool_name: tool description"`.
