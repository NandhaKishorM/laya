#pragma once
// The agent core (laya-go/agent): Open a checkpoint directory over a model Session, then Predict or
// PredictBatch states against a question set. Hooks, the confidence gate, predict_long, calibration
// and digests are later changes.
//
// Errors are returned as Result<T>; no exception crosses this API, and an exception thrown by a
// Session is caught and reported as an error. An Agent is safe for concurrent use: any number of
// threads may call Predict while another calls Close.
//
// Deferred to later changes: per-language temperatures (CallOptions has no `lang` until then), the
// confidence gate, predict_long, hooks, calibration and digests.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "laya/cancel.hpp"
#include "laya/decode.hpp"
#include "laya/infer.hpp"
#include "laya/pyvalue.hpp"
#include "laya/result.hpp"
#include "laya/schema.hpp"
#include "laya/sequence.hpp"
#include "laya/tokenizer.hpp"

namespace laya {
namespace agent {

// The text of the errors a caller may want to tell apart; test with IsClosed and IsCancelled.
extern const char* const kClosed;
extern const char* const kCancelled;
bool IsClosed(const std::string& error);
bool IsCancelled(const std::string& error);

// Bounds on what one call may be given.
constexpr int64_t kMaxStates = int64_t(1) << 20;
constexpr int64_t kMaxSequenceLength = int64_t(1) << 20;  // max_len and head_max_len

// rl_agent_config.json, as far as the agent reads it.
struct Config {
  int64_t max_len = 512;
  int64_t head_max_len = 192;
  std::string option_layout = "sequential";
  std::vector<std::string> warnings;  // e.g. temperatures clamped into range
};

// Per-call settings. Zero values mean the defaults.
struct CallOptions {
  // States per session run. <= 0 runs all states at once, split only where one run would exceed
  // infer::kMaxTensorElements in any tensor. A positive value is used as given; a run over the cap fails.
  int64_t batch_size = 0;
  bool sort_by_length = false;  // group similar lengths per run; results stay in input order
  int64_t max_len = 0;          // > 0 overrides the checkpoint's
  int64_t head_max_len = 0;     // > 0 overrides the checkpoint's
  CancelToken cancel;           // checked between states and runs, and passed to the Session
};

class Agent {
 public:
  // checkpoint_dir holds rl_agent_config.json and the tokenizer. Takes ownership of the session; on
  // failure the session is closed and destroyed.
  static Result<std::unique_ptr<Agent>> Open(const std::string& checkpoint_dir, std::unique_ptr<infer::Session> session);
  // From files already read: the config text and a loaded tokenizer.
  static Result<std::unique_ptr<Agent>> OpenParts(std::string_view config_json, tokenizer::Tokenizer tok,
                                                  std::unique_ptr<infer::Session> session);
  Agent(const Agent&) = delete;
  Agent& operator=(const Agent&) = delete;
  ~Agent();

  const Config& config() const { return cfg_; }

  // A state is a string, a list or an object (what the Python agent accepts).
  Result<schema::Prediction> Predict(const pyvalue::Value& state, const schema::Questions& qs,
                                     const CallOptions& opts = {}) const;
  // One prediction per state, in input order whatever the batching and sorting.
  Result<std::vector<schema::Prediction>> PredictBatch(const std::vector<pyvalue::Value>& states,
                                                       const schema::Questions& qs,
                                                       const CallOptions& opts = {}) const;
  // Refuses new calls at once and waits for each call in flight to return; a call with runs left is
  // cut short with kClosed at its next run. Then closes the session exactly once, outside the lock;
  // concurrent and later Closes return the same result. Later calls fail with kClosed. Returns the
  // session's close error, "" on success. Must not be called from inside a Session (it would wait
  // for itself).
  std::string Close();

 private:
  Agent() = default;
  using Encoded = std::vector<std::vector<sequence::Sequence>>;  // [state][question]
  Result<std::vector<schema::Prediction>> Chunks(const std::vector<pyvalue::Value>& states, const schema::Questions& qs,
                                                 const CallOptions& c) const;
  Result<std::vector<sequence::Sequence>> Encode(const pyvalue::Value& state, const schema::Questions& qs, const CallOptions& c) const;
  Result<std::vector<schema::Prediction>> Run(const schema::Questions& qs, const Encoded& encoded,
                                              const std::vector<size_t>& idx, const CallOptions& c) const;
  bool Enter() const;
  void Leave() const;

  tokenizer::Tokenizer tok_;
  int32_t pad_id_ = 0;
  Config cfg_;
  decode::Temperatures temps_;
  std::unique_ptr<infer::Session> sess_;

  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  mutable int64_t inflight_ = 0;
  bool shut_ = false;
  bool close_started_ = false;
  bool close_done_ = false;
  std::string close_err_;
  std::atomic<bool> closed_{false};
};

}  // namespace agent
}  // namespace laya
