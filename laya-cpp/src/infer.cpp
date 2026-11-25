#include "laya/infer.hpp"

namespace laya {
namespace infer {

const char* const kInputIds = "input_ids";
const char* const kAttentionMask = "attention_mask";
const char* const kMarkerPos = "marker_pos";
const char* const kMarkerMask = "marker_mask";
const char* const kQType = "qtype";
const char* const kLogits = "logits";
const char* const kActLogits = "act_logits";

const char* const kCancelled = "infer: cancelled";
bool IsCancelled(const std::string& e) { return e == kCancelled; }

namespace {
std::string Shape(const char* name, size_t got, int64_t want) {
  return std::string("infer: ") + name + " has " + std::to_string(got) + " values, want " + std::to_string(want);
}
}  // namespace

std::string Batch::Validate() const {
  if (rows < 0 || length < 0 || markers < 0) return "infer: batch has negative dimensions";
  if (rows > 0) {
    if (length == 0) return "infer: batch has rows of 0 tokens";
    if (markers < 2) return "infer: batch has " + std::to_string(markers) + " marker slots, the graph needs at least 2";
    if (length > kMaxTensorElements / rows || markers > kMaxTensorElements / rows) {
      return "infer: batch dimensions " + std::to_string(rows) + " x " + std::to_string(length) + " x " +
             std::to_string(markers) + " are too large";
    }
  }
  const int64_t rl = rows * length, rm = rows * markers;
  struct Check {
    const char* name;
    size_t got;
    int64_t want;
  };
  const Check checks[] = {{kInputIds, input_ids.size(), rl},
                          {kAttentionMask, attention_mask.size(), rl},
                          {kMarkerPos, marker_pos.size(), rm},
                          {kMarkerMask, marker_mask.size(), rm},
                          {kQType, qtype.size(), rows}};
  for (const Check& c : checks) {
    if (int64_t(c.got) != c.want) return Shape(c.name, c.got, c.want);
  }
  for (size_t i = 0; i < marker_pos.size(); i++) {
    if (marker_pos[i] < 0 || marker_pos[i] >= length) {
      return "infer: marker_pos[" + std::to_string(i) + "] is " + std::to_string(marker_pos[i]) + ", outside [0, " +
             std::to_string(length) + ")";
    }
  }
  for (size_t i = 0; i < input_ids.size(); i++) {
    if (input_ids[i] < 0) return "infer: input_ids[" + std::to_string(i) + "] is negative, token ids must not be";
  }
  for (size_t i = 0; i < qtype.size(); i++) {
    if (qtype[i] < 0 || qtype[i] > 2) return "infer: qtype row " + std::to_string(i) + " is not 0, 1 or 2";
  }
  return "";
}

std::string Output::Check(const Batch& b) const {
  if (b.rows < 0 || b.markers < 0 || (b.rows > 0 && (b.markers > kMaxTensorElements / b.rows || b.rows > kMaxTensorElements))) {
    return "infer: batch dimensions " + std::to_string(b.rows) + " x " + std::to_string(b.markers) + " are out of range";
  }
  if (int64_t(logits.size()) != b.rows * b.markers) return Shape(kLogits, logits.size(), b.rows * b.markers);
  if (int64_t(act_logits.size()) != b.rows * 2) return Shape(kActLogits, act_logits.size(), b.rows * 2);
  return "";
}

}  // namespace infer
}  // namespace laya
