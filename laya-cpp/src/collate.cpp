#include "laya/collate.hpp"

#include <algorithm>

namespace laya {
namespace collate {

Result<infer::Batch> Collate(const std::vector<Row>& rows, int32_t pad_id) {
  using R = Result<infer::Batch>;
  if (rows.empty()) return R::Fail("collate: nothing to collate");
  if (pad_id < 0) return R::Fail("collate: the pad id is negative");
  int64_t length = 0, width = kMinMarkerSlots;
  for (size_t i = 0; i < rows.size(); i++) {
    const Row& r = rows[i];
    const std::string at = "collate: row " + std::to_string(i);
    if (r.ids.empty()) return R::Fail(at + " has no token ids");
    if (schema::Code(r.type) < 0) return R::Fail(at + " has an invalid question type");
    for (int32_t id : r.ids) {
      if (id < 0) return R::Fail(at + " has a negative token id");
    }
    for (int64_t m : r.markers) {
      if (m < 0 || m >= int64_t(r.ids.size())) {
        return R::Fail(at + " has a marker at " + std::to_string(m) + " outside its " + std::to_string(r.ids.size()) + " ids");
      }
    }
    length = std::max(length, int64_t(r.ids.size()));
    width = std::max(width, int64_t(r.markers.size()));
  }
  const int64_t n = int64_t(rows.size());
  if (length > infer::kMaxTensorElements / n || width > infer::kMaxTensorElements / n) {
    return R::Fail("collate: " + std::to_string(n) + " rows of " + std::to_string(length) + " tokens and " +
                   std::to_string(width) + " markers are too large");
  }
  infer::Batch b;
  b.rows = n;
  b.length = length;
  b.markers = width;
  b.input_ids.assign(size_t(n * length), int64_t(pad_id));
  b.attention_mask.assign(size_t(n * length), 0);
  b.marker_pos.assign(size_t(n * width), 0);
  b.marker_mask.assign(size_t(n * width), 0);
  b.qtype.assign(size_t(n), 0);
  for (size_t i = 0; i < rows.size(); i++) {
    const Row& r = rows[i];
    const size_t base = i * size_t(length);
    for (size_t j = 0; j < r.ids.size(); j++) {
      b.input_ids[base + j] = r.ids[j];
      b.attention_mask[base + j] = 1;
    }
    for (size_t j = 0; j < r.markers.size(); j++) {
      b.marker_pos[i * size_t(width) + j] = r.markers[j];
      b.marker_mask[i * size_t(width) + j] = 1;
    }
    b.qtype[i] = schema::Code(r.type);
  }
  return R::Ok(std::move(b));
}

std::vector<size_t> PlanRuns(size_t rows_per_state, const std::vector<int64_t>& length, const std::vector<int64_t>& width) {
  std::vector<size_t> runs;
  const int64_t per = int64_t(rows_per_state);
  size_t count = 0;
  int64_t longest = 0, widest = kMinMarkerSlots;
  for (size_t i = 0; i < length.size(); i++) {
    const int64_t l = std::max(longest, length[i]), w = std::max(widest, width[i]);
    const int64_t rows = int64_t(count + 1) * per;
    if (count > 0 && (l > infer::kMaxTensorElements / rows || w > infer::kMaxTensorElements / rows)) {
      runs.push_back(count);
      count = 0;
      longest = 0, widest = kMinMarkerSlots;
    }
    longest = std::max(longest, length[i]);
    widest = std::max(widest, width[i]);
    count++;
  }
  if (count > 0) runs.push_back(count);
  return runs;
}

}  // namespace collate
}  // namespace laya
