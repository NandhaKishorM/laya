# Agent

`laya.Agent` loads one checkpoint and answers typed questions about a state. `laya.load` is
a shortcut for `Agent(...)`, and `laya.RLAgent` is an alias of `Agent`. `ONNXAgent` runs an
exported ONNX model on CPU; import it from `laya.onnx_agent`.

::: laya.agent.Agent

::: laya.agent.load

::: laya.onnx_agent.ONNXAgent
