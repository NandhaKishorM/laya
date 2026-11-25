#include "laya/onnx_graph.hpp"

#include <cmath>
#include <filesystem>
#include <set>

namespace laya {
namespace onnx {
namespace {

const char* const kReexport = "re-export with scripts/export_onnx.py";
const std::string kSplitProblem = std::string("the split encoder.onnx + head.onnx layout is not supported; ") + kReexport +
                                  " to write one fused laya.onnx";

struct Spec {
  const char* name;
  const char* dtype;
  size_t rank;
  int64_t fixed_cols;  // 0: every dimension after the batch is dynamic
};

const Spec kInputs[] = {{"input_ids", "int64", 2, 0},
                        {"attention_mask", "int64", 2, 0},
                        {"marker_pos", "int64", 2, 0},
                        {"marker_mask", "bool", 2, 0},
                        {"qtype", "int64", 1, 0}};
const Spec kOutputs[] = {{"logits", "float32", 2, 0}, {"act_logits", "float32", 2, 2}};

template <size_t N>
std::string CheckSide(const char* kind, const std::vector<TensorInfo>& got, const Spec (&want)[N]) {
  for (const Spec& w : want) {
    const TensorInfo* g = nullptr;
    for (const TensorInfo& t : got) {
      if (t.name == w.name) {
        g = &t;
        break;
      }
    }
    const std::string q = std::string(kind) + " \"" + w.name + "\"";
    if (g == nullptr) return std::string("missing ") + q + "; " + kReexport;
    if (g->dtype != w.dtype) return q + " is " + g->dtype + ", want " + w.dtype + "; " + kReexport;
    if (g->dims.size() != w.rank) {
      return q + " has rank " + std::to_string(g->dims.size()) + ", want " + std::to_string(w.rank) + "; " + kReexport;
    }
    for (size_t i = 0; i < g->dims.size(); i++) {
      const int64_t d = g->dims[i];
      if (d == -1 || (i > 0 && w.fixed_cols > 0 && d == w.fixed_cols)) continue;
      return q + " declares " + (i == 0 ? "a fixed batch dimension " : "a fixed size ") + std::to_string(d) + "; " + kReexport;
    }
  }
  std::set<std::string> known;
  for (const Spec& w : want) known.insert(w.name);
  for (const TensorInfo& t : got) {
    if (known.count(t.name) == 0) {
      return std::string("unexpected ") + kind + " \"" + t.name + "\"; the agent feeds only " + std::to_string(sizeof(kInputs) / sizeof(kInputs[0])) +
             " inputs and " + std::to_string(sizeof(kOutputs) / sizeof(kOutputs[0])) + " outputs; " + kReexport;
    }
  }
  if (got.size() != N) return std::string("duplicate ") + kind + " name; " + kReexport;
  return "";
}

bool IsSplit(const std::filesystem::path& dir) {
  std::error_code ec;
  for (const char* n : {"encoder.onnx", "head.onnx"}) {
    if (!std::filesystem::is_regular_file(dir / n, ec)) return false;
  }
  return true;
}

std::string Shape(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); i++) out += (i ? " " : "") + std::to_string(s[i]);
  return out + "]";
}

}  // namespace

std::string CheckGraph(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>& outputs) {
  std::string p = CheckSide("input", inputs, kInputs);
  if (p.empty()) p = CheckSide("output", outputs, kOutputs);
  return p;
}

std::string CheckPath(const std::string& path) {
  namespace fs = std::filesystem;
  if (path.empty()) return "graph path is empty";
  std::error_code ec;
  const fs::path p(path);
  const fs::file_status st = fs::status(p, ec);
  if (!ec && fs::is_directory(st)) {
    return IsSplit(p) ? kSplitProblem : "graph " + path + " is a directory; pass the path of laya.onnx";
  }
  if (!ec && fs::is_regular_file(st)) {
    const std::string base = p.filename().string();
    if ((base == "encoder.onnx" || base == "head.onnx") && IsSplit(p.parent_path())) return kSplitProblem;
    return "";
  }
  const fs::path dir = p.has_parent_path() ? p.parent_path() : fs::path(".");
  if (st.type() == fs::file_type::not_found && IsSplit(dir)) return "graph " + path + ": no such file; " + kSplitProblem;
  if (st.type() == fs::file_type::not_found) return "graph " + path + ": no such file";
  return "graph " + path + ": " + (ec ? ec.message() : std::string("not a regular file"));
}

Result<infer::Output> CollectOutput(const infer::Batch& b, const OutTensor& logits, const OutTensor& act) {
  struct Col {
    const char* name;
    const OutTensor& t;
    int64_t cols;
  };
  for (const Col& c : {Col{"logits", logits, b.markers}, Col{"act_logits", act, 2}}) {
    if (c.t.shape.size() != 2 || c.t.shape[0] != b.rows || c.t.shape[1] != c.cols) {
      return Result<infer::Output>::Fail(std::string("onnx: output ") + c.name + " has shape " + Shape(c.t.shape) + ", want [" +
                                         std::to_string(b.rows) + " " + std::to_string(c.cols) + "]");
    }
    // rows and cols are bounded by Batch::Validate, so this product cannot overflow.
    if (c.t.count != b.rows * c.cols || (c.t.count > 0 && c.t.data == nullptr)) {
      return Result<infer::Output>::Fail(std::string("onnx: output ") + c.name + " holds " + std::to_string(c.t.count) + " values");
    }
    for (int64_t i = 0; i < c.t.count; i++) {
      if (!std::isfinite(c.t.data[i])) return Result<infer::Output>::Fail(std::string("onnx: output ") + c.name + " contains NaN or Inf");
    }
  }
  infer::Output o;
  o.logits.assign(logits.data, logits.data + logits.count);
  o.act_logits.assign(act.data, act.data + act.count);
  const std::string bad = o.Check(b);
  if (!bad.empty()) return Result<infer::Output>::Fail("onnx: " + bad);
  return Result<infer::Output>::Ok(std::move(o));
}

}  // namespace onnx
}  // namespace laya
