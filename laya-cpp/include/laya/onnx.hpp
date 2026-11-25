#pragma once
// The ONNX Runtime backend (laya-go/onnx). Linking laya_onnx pulls in ONNX Runtime; the core `laya`
// library never does, and this header includes no ONNX Runtime header.
#include <cstdint>
#include <memory>
#include <string>
#include "laya/agent.hpp"
#include "laya/infer.hpp"
#include "laya/result.hpp"

namespace laya {
namespace onnx {

struct Options {
  // 0 leaves ONNX Runtime's default (one intra-op pool sized to the cores). Every session owns its
  // own pool and sessions do not share one, so many sessions in one process should set this. There
  // is no inter-op option: the graph runs sequentially, where it would have no effect.
  int intra_op_threads = 0;  // at most kMaxThreads
};
constexpr int kMaxThreads = 4096;

// The version of the ONNX Runtime library loaded, e.g. "1.30.0".
std::string RuntimeVersion();

// A Session over one fused laya.onnx (+ laya.onnx.data beside it). Checks the graph's inputs and
// outputs at once. Run is thread-safe and stops in flight when its CancelToken fires (kCancelled).
// Close is idempotent, refuses new runs, terminates the runs in flight and returns once they have
// left. A Session or Agent may be held in a static, closed or not. No exception leaves this API.
// The first Open sets ORT_DISABLE_TELEMETRY=1 unless it is already set: the runtime's telemetry
// thread can abort the process at exit. A program that builds its own OrtEnv first must set it too.
Result<std::unique_ptr<infer::Session>> Open(const std::string& graph_path, const Options& options = {});

}  // namespace onnx

namespace agent {
// checkpoint_dir holds rl_agent_config.json and the tokenizer; graph_path is the fused laya.onnx.
Result<std::unique_ptr<Agent>> OpenOnnx(const std::string& checkpoint_dir, const std::string& graph_path,
                                        const onnx::Options& options = {});
}  // namespace agent

}  // namespace laya
