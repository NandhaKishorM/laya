#pragma once
// A stand-in for the model runtime: its outputs are a function of the integers it is fed (exact in
// float32), as the Python generator's _plan_session. Records every run. Safe for concurrent Run.
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "laya/infer.hpp"

namespace laya_test {

struct RunRecord {
  int64_t rows = 0, length = 0, markers = 0;
  std::vector<int64_t> id_sums, tokens, marker_sums, qtypes;
};

class StubSession : public laya::infer::Session {
 public:
  // Shared with the test so counts survive the agent that owns the session.
  struct State {
    std::mutex mu;
    std::vector<RunRecord> runs;
    int closes = 0;
    bool destroyed = false;
    bool closed = false;
    std::function<void()> on_run;                       // called at the start of every Run
    std::function<void(laya::infer::Output*)> tamper;   // may break the output
    std::string fail;                                   // non-empty: Run fails with this
    bool ignore_cancel = false;                         // Run does not look at the token
    std::string close_error;                            // what Close returns
  };
  explicit StubSession(std::shared_ptr<State> s = std::make_shared<State>()) : st_(std::move(s)) {}
  ~StubSession() override {
    std::lock_guard<std::mutex> l(st_->mu);
    st_->destroyed = true;
  }
  std::shared_ptr<State> state() const { return st_; }

  laya::Result<laya::infer::Output> Run(const laya::infer::Batch& b, const laya::CancelToken& cancel) override {
    using R = laya::Result<laya::infer::Output>;
    std::function<void()> hook;
    {
      std::lock_guard<std::mutex> l(st_->mu);
      if (st_->closed) return R::Fail("infer: session is closed");
      hook = st_->on_run;
    }
    if (hook) hook();
    if (!st_->ignore_cancel && cancel.cancelled()) return R::Fail(laya::infer::kCancelled);
    RunRecord rec;
    rec.rows = b.rows, rec.length = b.length, rec.markers = b.markers;
    laya::infer::Output out;
    for (int64_t r = 0; r < b.rows; r++) {
      int64_t sum = 0, tokens = 0, msum = 0;
      for (int64_t i = 0; i < b.length; i++) {
        if (b.attention_mask[size_t(r * b.length + i)] == 1) {
          sum += b.input_ids[size_t(r * b.length + i)];
          tokens++;
        }
      }
      for (int64_t j = 0; j < b.markers; j++) {
        if (b.marker_mask[size_t(r * b.markers + j)]) msum += b.marker_pos[size_t(r * b.markers + j)];
        out.logits.push_back(float(((sum + 31 * j) % 97 - 48) / 8.0));
      }
      for (int64_t c = 0; c < 2; c++) out.act_logits.push_back(float(((sum + 17 * c) % 97 - 48) / 8.0));
      rec.id_sums.push_back(sum), rec.tokens.push_back(tokens), rec.marker_sums.push_back(msum);
      rec.qtypes.push_back(b.qtype[size_t(r)]);
    }
    std::lock_guard<std::mutex> l(st_->mu);
    st_->runs.push_back(std::move(rec));
    if (!st_->fail.empty()) return R::Fail(st_->fail);
    if (st_->tamper) st_->tamper(&out);
    return R::Ok(std::move(out));
  }
  std::string Close() override {
    std::lock_guard<std::mutex> l(st_->mu);
    st_->closes++;
    st_->closed = true;
    return st_->close_error;
  }

 private:
  std::shared_ptr<State> st_;
};

}  // namespace laya_test
