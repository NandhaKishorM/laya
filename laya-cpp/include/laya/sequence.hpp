#pragma once
// The model's input sequence (laya-go/sequence): head, state, markers, and the window plan.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "laya/pyvalue.hpp"
#include "laya/result.hpp"
#include "laya/schema.hpp"
#include "laya/tokenizer.hpp"

namespace laya {
namespace sequence {

constexpr int64_t kOptionTokenCap = 48;
constexpr int64_t kMinOptionBudget = 16;
constexpr int64_t kMinTokensPerOption = 4;
constexpr int64_t kMinInstructionTokens = 8;

struct Budget {
  int64_t max_len = 0;
  int64_t head_max_len = 0;
  int64_t DefaultWindow() const;  // max(64, max_len - head_max_len - 8)
};

enum class Keep { Start, End };

struct State {
  std::vector<int32_t> ids;
  Keep keep = Keep::Start;
};

// serialize_state: a string as is, anything else as json.dumps(ensure_ascii=False).
Result<std::string> SerializeState(const pyvalue::Value& state);
// A list keeps its end when it is too long, anything else its start.
Result<State> EncodeState(const tokenizer::Tokenizer& tok, const pyvalue::Value& state);

struct Stats {
  int64_t options = 0;
  int64_t options_distinct = 0;
  std::optional<int64_t> tokens_per_option;
  bool Collapsed() const { return options_distinct < options; }
};

struct Truncation {
  int64_t state_tokens = 0, state_tokens_used = 0, state_tokens_dropped = 0;
  bool truncated = false;
  bool operator==(const Truncation& o) const {
    return state_tokens == o.state_tokens && state_tokens_used == o.state_tokens_used &&
           state_tokens_dropped == o.state_tokens_dropped && truncated == o.truncated;
  }
};

struct Head {
  std::vector<int32_t> ids;
  std::vector<int64_t> markers;
  Stats stats;
};

struct Sequence {
  std::vector<int32_t> ids;
  std::vector<int64_t> markers;
  Stats stats;
  Truncation truncation;
};

// Some option markers fell beyond max_len. Message(id) is the agent's refusal text.
struct MarkersError {
  int64_t kept = 0, total = 0, max_len = 0, head_max_len = 0;
  std::string Message(const std::string& id = "") const;
};

Result<Head> BuildHead(const tokenizer::Tokenizer& tok, const schema::Question& q, int64_t head_max_len);
// markers_error, when not null, is filled iff the failure is the marker refusal.
Result<Sequence> Build(const tokenizer::Tokenizer& tok, const schema::Question& q, const State& state,
                       const Budget& b, MarkersError* markers_error = nullptr);
Result<int64_t> StateRoom(const tokenizer::Tokenizer& tok, const schema::Question& q, const Budget& b);

struct WindowRequest {
  int64_t window = 0;  // <= 0: the default
  int64_t stride = 0;  // <= 0: half a window
};

struct WindowPlan {
  int64_t window = 0, stride = 0, room = 0;
  std::vector<std::string> warnings;
  bool Clamped() const { return !warnings.empty(); }
};

enum class PlanError { None, NoRoom, StrideGap, Other };

Result<WindowPlan> PlanWindows(const tokenizer::Tokenizer& tok, const std::vector<schema::Question>& questions,
                               const Budget& b, const WindowRequest& req, PlanError* kind = nullptr);

// window_batch_cap: ok=false is Python's None (do not cap). batch_size <= 0 is unset.
struct Cap {
  int64_t cap = 0;
  bool ok = false;
};
Cap BatchCap(int64_t n_windows, int64_t window, int64_t config_window, int64_t batch_size);

}  // namespace sequence
}  // namespace laya
