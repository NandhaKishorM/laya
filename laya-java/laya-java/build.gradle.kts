// The runtime. One dependency, deliberately: ONNX Runtime. `tokenizer.json` and
// `rl_agent_config.json` need a JSON reader, but a JSON dependency in the core forces a version on
// every consumer, and two known schemas do not justify that -- see the internal reader in
// `com.convaiinnovations.laya.json`.
dependencies {
    api("com.microsoft.onnxruntime:onnxruntime:1.20.0")
}
