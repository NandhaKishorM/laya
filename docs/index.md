# Laya

Multilingual, non-autoregressive System 1 decision engine: typed `choice`, `score` and `noul`
decisions over any state, in a single forward pass.

The [README](https://github.com/NandhaKishorM/laya#readme) is the main guide. It covers
installation, the `Router` quickstart, the HTTP server, calibration, benchmarks and known limits.
These guides cover individual topics:

- [Docker quickstart](docker.md): run the SDK in a container, on CPU or an NVIDIA GPU.
- [LangChain & LangGraph](langchain.md): Laya as router, guardrail, triage and evaluator
  components, in process or over HTTP.
- [Fine-tuning example: browser agent](finetune_browser_agent.md): specialising Laya as the
  decision head of a browser agent, end to end.
