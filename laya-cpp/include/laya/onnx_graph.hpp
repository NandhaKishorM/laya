#pragma once
// The ONNX backend's checks that need no ONNX Runtime (laya-go/onnx/graph.go): what a graph must
// declare, where a graph path may point, and how a run's outputs become an infer::Output.
#include <cstdint>
#include <string>
#include <vector>
#include "laya/infer.hpp"
#include "laya/result.hpp"

namespace laya {
namespace onnx {

struct TensorInfo {
  std::string name;
  std::string dtype;  // "int64", "bool", "float32", or whatever else the graph declares
  std::vector<int64_t> dims;  // -1 for a symbolic or unknown dimension
};

// "" when the graph's own metadata is what the agent feeds: five inputs (int64, int64, int64, bool,
// int64) and two float32 outputs, matching by name, no extras, dynamic batch dimension, and a fixed
// 2 columns for act_logits. Else the first problem.
std::string CheckGraph(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>& outputs);

// "" when path names a file that may be opened as the fused graph; refuses the split
// encoder.onnx + head.onnx layout with a message that says how to re-export.
std::string CheckPath(const std::string& path);

struct OutTensor {
  std::vector<int64_t> shape;
  const float* data = nullptr;
  int64_t count = 0;  // elements at data
};

// Copies the two outputs after checking their shapes against b, then Output::Check. NaN anywhere is
// an error: a NaN would decode to a silent first-max answer.
Result<infer::Output> CollectOutput(const infer::Batch& b, const OutTensor& logits, const OutTensor& act_logits);

}  // namespace onnx
}  // namespace laya
