#pragma once
// The seam between the agent and a model runtime (laya-go/infer). A Session takes one collated
// batch of the graph's five inputs and returns its two outputs; the ONNX backend (C5) implements it.
#include <cstdint>
#include <string>
#include <vector>

#include "laya/cancel.hpp"
#include "laya/result.hpp"

namespace laya {
namespace infer {

// The graph's tensor names, as an error names them and as a native session binds them.
extern const char* const kInputIds;
extern const char* const kAttentionMask;
extern const char* const kMarkerPos;
extern const char* const kMarkerMask;
extern const char* const kQType;
extern const char* const kLogits;
extern const char* const kActLogits;

// The most elements any one tensor of a batch may hold: a hostile row count or length is refused
// before anything is allocated.
constexpr int64_t kMaxTensorElements = int64_t(1) << 24;

// Each input flattened row-major. A value type: copy or move it freely.
struct Batch {
  std::vector<int64_t> input_ids;       // [rows*length]; padding is the pad id
  std::vector<int64_t> attention_mask;  // [rows*length]; 1 for real tokens
  std::vector<int64_t> marker_pos;      // [rows*markers]
  std::vector<uint8_t> marker_mask;     // [rows*markers]; 1 for real markers
  std::vector<int64_t> qtype;           // [rows]; 0, 1 or 2
  int64_t rows = 0, length = 0, markers = 0;

  // Empty when the batch is well formed, else the first problem: negative or oversize dimensions,
  // fewer than 2 marker slots (the exported head takes topk(2)), a slice whose length does not match,
  // a marker outside [0, length), a negative token id, a question type outside 0..2.
  std::string Validate() const;
};

// Row-major: logits [rows*markers], act_logits [rows*2].
struct Output {
  std::vector<float> logits, act_logits;
  std::string Check(const Batch& b) const;  // empty when both hold the lengths b calls for
};

// What a Session returns from Run when the cancel token fired and it stopped early.
extern const char* const kCancelled;
bool IsCancelled(const std::string& error);

// Run may be called from several threads at once. Close is idempotent and never runs concurrently
// with Run (the agent guarantees it). A Session is owned through std::unique_ptr, so it is moved,
// never copied.
class Session {
 public:
  Session() = default;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  virtual ~Session() = default;
  // Must not modify or retain b. Should return early with kCancelled when cancel fires. No Session
  // method may throw; the agent catches anyway and reports the exception as an error.
  virtual Result<Output> Run(const Batch& b, const CancelToken& cancel) = 0;
  // Releases the runtime; "" on success, else the error. Run after Close fails.
  virtual std::string Close() = 0;
};

}  // namespace infer
}  // namespace laya
