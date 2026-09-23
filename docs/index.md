# Laya

Multilingual, non-autoregressive System 1 decision engine: typed `choice`, `score` and `noul`
decisions over any state, in a single forward pass.

The [README](https://github.com/NandhaKishorM/laya#readme) is the main guide. It covers
installation, the `Router` quickstart, the HTTP server, calibration, benchmarks and known limits.
These guides cover individual topics:

- [Docker quickstart](docker.md): run the SDK in a container, on CPU or an NVIDIA GPU.
- [ARM64 and DGX Spark containers](docker-platforms.md): builds for ARM64 CPU hosts and for
  DGX Spark with CUDA 13.0.
- [LangChain & LangGraph](langchain.md): Laya as router, guardrail, triage and evaluator
  components, in process or over HTTP.
- [Fine-tuning example: browser agent](finetune_browser_agent.md): specialising Laya as the
  decision head of a browser agent, end to end.
- [Prediction hooks](hooks/index.md): observe or shape every decision without forking Laya.
