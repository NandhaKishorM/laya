#pragma once
// Pads built sequences into the rectangular tensors the graph takes (laya-go/internal/collate).
// One row per question, not per state: four questions about one document are four rows that share
// a state. Rows are padded to the longest sequence and markers to the most options.
#include <cstdint>
#include <vector>

#include "laya/infer.hpp"
#include "laya/result.hpp"
#include "laya/schema.hpp"

namespace laya {
namespace collate {

// The fewest option slots the graph accepts: one-option rows get a masked second slot that
// decoding never reads.
constexpr int64_t kMinMarkerSlots = 2;

struct Row {
  std::vector<int32_t> ids;
  std::vector<int64_t> markers;
  schema::Type type = schema::Type::Invalid;
};

// pad_id is fed to the encoder under a zero mask, so it must be a real vocabulary id. Refuses an
// empty list, a row with no ids, a row with an invalid type, a marker outside its ids, a negative
// id, and a batch that would exceed infer::kMaxTensorElements.
Result<infer::Batch> Collate(const std::vector<Row>& rows, int32_t pad_id);

// How many states each session run takes when one run must not exceed infer::kMaxTensorElements in
// any tensor. Each state contributes rows_per_state rows of length[i] tokens and width[i] markers,
// taken in order; a state that alone exceeds the cap still gets a run (Collate refuses it).
std::vector<size_t> PlanRuns(size_t rows_per_state, const std::vector<int64_t>& length, const std::vector<int64_t>& width);

}  // namespace collate
}  // namespace laya
