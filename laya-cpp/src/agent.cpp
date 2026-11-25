#include "laya/agent.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <sstream>

#include "laya/collate.hpp"
#include "laya/unicodedata.hpp"

namespace laya {
namespace agent {

const char* const kClosed = "agent: agent is closed";
const char* const kCancelled = "agent: call cancelled";

bool IsClosed(const std::string& e) { return e == kClosed; }
bool IsCancelled(const std::string& e) { return e == kCancelled; }

namespace {

using pyvalue::Value;
using Kind = Value::Kind;
constexpr int64_t kLookaheadBatches = 8;
const char* const kConfigFile = "rl_agent_config.json";

std::string Num(double d) {
  std::string s = pyvalue::Repr(Value::Float(d));
  if (s.size() > 2 && s.compare(s.size() - 2, 2, ".0") == 0) s.resize(s.size() - 2);
  return s;
}

// Whether the raw value is exactly what clamp_temperature returned.
bool SameNumber(const Value& raw, double applied) {
  switch (raw.kind()) {
    case Kind::String: {
      auto f = decode::PyFloat(raw.as_string());
      return f && *f == applied;
    }
    case Kind::Int: return double(raw.as_int()) == applied;
    case Kind::Float: return raw.as_float() == applied;
    default: return false;  // a bool is no number to clamp_temperature
  }
}

Result<int64_t> LengthField(const Value& obj, const char* key, int64_t dflt) {
  using R = Result<int64_t>;
  const Value* v = obj.Find(key);
  if (v == nullptr) return R::Ok(dflt);
  const std::string name = std::string("agent: ") + kConfigFile + ": " + key + ": ";
  if (v->kind() == Kind::BigInt && v->big_int().rfind('-', 0) != 0) return R::Fail(name + "too large for an int");
  if (v->kind() != Kind::Int || v->as_int() <= 0) return R::Fail(name + "want a positive integer, got " + pyvalue::Repr(*v));
  if (v->as_int() > kMaxSequenceLength) return R::Fail(name + "too large (the limit is " + std::to_string(kMaxSequenceLength) + ")");
  return R::Ok(v->as_int());
}


}  // namespace

namespace {

// Closes the session on every failure path of Open; empty once the agent owns it.
struct SessionGuard {
  std::unique_ptr<infer::Session>& s;
  ~SessionGuard() {
    if (!s) return;
    try {
      s->Close();
    } catch (...) {
    }
  }
};

}  // namespace

Result<std::unique_ptr<Agent>> Agent::Open(const std::string& dir, std::unique_ptr<infer::Session> session) {
  using R = Result<std::unique_ptr<Agent>>;
  if (!session) return R::Fail("agent: Open needs a session");
  SessionGuard guard{session};
  std::ifstream in(dir + "/" + kConfigFile, std::ios::binary);
  if (!in) return R::Fail(std::string("agent: ") + kConfigFile + ": cannot read " + dir + "/" + kConfigFile);
  std::stringstream ss;
  ss << in.rdbuf();
  auto tok = tokenizer::Tokenizer::LoadDir(dir);
  if (!tok.ok()) return R::Fail("agent: tokenizer: " + tok.error());
  return OpenParts(ss.str(), std::move(tok).value(), std::move(session));
}

Result<std::unique_ptr<Agent>> Agent::OpenParts(std::string_view config_json, tokenizer::Tokenizer tok,
                                                std::unique_ptr<infer::Session> session) {
  using R = Result<std::unique_ptr<Agent>>;
  if (!session) return R::Fail("agent: Open needs a session");
  SessionGuard guard{session};
  const std::string where = std::string("agent: ") + kConfigFile + ": ";
  auto parsed = pyvalue::Parse(config_json);
  if (!parsed.ok()) return R::Fail(where + parsed.error());
  const Value& obj = parsed.value();
  if (obj.kind() != Kind::Object) return R::Fail(where + "want a JSON object, got " + pyvalue::TypeName(obj));
  std::unique_ptr<Agent> a(new Agent());
  if (const Value* l = obj.Find("option_layout")) {
    if (l->kind() != Kind::String || l->as_string() != "sequential") {
      return R::Fail(where + "option_layout: option_layout is not supported by the ONNX runtime: " + pyvalue::Repr(*l));
    }
  }
  auto max_len = LengthField(obj, "max_len", 512);
  if (!max_len.ok()) return R::Fail(max_len.error());
  auto head_max_len = LengthField(obj, "head_max_len", 192);
  if (!head_max_len.ok()) return R::Fail(head_max_len.error());
  a->cfg_.max_len = max_len.value();
  a->cfg_.head_max_len = head_max_len.value();

  const Value* by_type = obj.Find("temperature");
  if (by_type != nullptr && by_type->kind() != Kind::Array) {
    return R::Fail(where + "temperature: want a list of 3 numbers, got " + pyvalue::Repr(*by_type));
  }
  const Value* by_options = obj.Find("temperature_by_options");
  if (by_options != nullptr && by_options->kind() != Kind::Object) {
    return R::Fail(where + "temperature_by_options: want an object of bucket to number, got " + pyvalue::Repr(*by_options));
  }
  auto temps = decode::Temperatures::Of(by_type, by_options);
  if (!temps.ok()) return R::Fail(where + "temperature: " + temps.error());
  a->temps_ = std::move(temps).value();
  std::vector<std::string> rejected;
  if (by_options != nullptr) {
    for (size_t i = 0; i < by_options->size(); i++) {
      const double applied = a->temps_.by_options().at(by_options->key(i));
      if (!SameNumber(by_options->value(i), applied)) {
        rejected.push_back(by_options->key(i) + "=" + pyvalue::Repr(by_options->value(i)) + " -> " + Num(applied));
      }
    }
  }
  if (by_type != nullptr) {
    for (size_t i = 0; i < by_type->size(); i++) {
      if (!SameNumber(by_type->items()[i], a->temps_.by_type()[i])) {
        rejected.push_back("temperature[" + std::to_string(i) + "]=" + pyvalue::Repr(by_type->items()[i]) + " -> " + Num(a->temps_.by_type()[i]));
      }
    }
  }
  if (!rejected.empty()) {
    std::string joined;
    for (size_t i = 0; i < rejected.size(); i++) joined += (i ? ", " : "") + rejected[i];
    a->cfg_.warnings.push_back("this checkpoint ships temperatures outside [0.5, 5] which would distort confidence; clamping " +
                               joined + ". Treat confidence from the affected buckets as uncalibrated.");
  }

  const std::optional<int32_t> pad = tok.Pad(), cls = tok.Cls(), sep = tok.Sep(), mask = tok.Mask();
  if (!pad) return R::Fail("agent: missing special token: no pad token");
  if (!cls) return R::Fail("agent: missing special token: no CLS token");
  if (!sep) return R::Fail("agent: missing special token: no SEP token");
  if (!mask) return R::Fail("agent: missing special token: no mask token");
  a->pad_id_ = *pad;
  a->tok_ = std::move(tok);
  a->sess_ = std::move(session);
  return R::Ok(std::move(a));
}

Agent::~Agent() { Close(); }

bool Agent::Enter() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (shut_ || !sess_) return false;
  inflight_++;
  return true;
}

void Agent::Leave() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (--inflight_ == 0) cv_.notify_all();
}

std::string Agent::Close() {
  std::unique_lock<std::mutex> lock(mu_);
  shut_ = true;
  closed_.store(true);
  cv_.wait(lock, [&] { return inflight_ == 0; });
  if (close_started_) {
    cv_.wait(lock, [&] { return close_done_; });
    return close_err_;
  }
  close_started_ = true;
  lock.unlock();
  std::string err;
  try {
    if (sess_) err = sess_->Close();
  } catch (const std::exception& e) {
    err = std::string("agent: session Close threw: ") + e.what();
  } catch (...) {
    err = "agent: session Close threw";
  }
  lock.lock();
  close_err_ = err;
  close_done_ = true;
  cv_.notify_all();
  return close_err_;
}

Result<schema::Prediction> Agent::Predict(const Value& state, const schema::Questions& qs, const CallOptions& opts) const {
  using R = Result<schema::Prediction>;
  auto r = PredictBatch(std::vector<Value>{state}, qs, opts);
  if (!r.ok()) {
    // The top-level error only: drop the "agent: state 0: " prefix a one-state call has no use for.
    std::string e = r.error();
    const std::string prefix = "agent: state 0: ";
    if (e.compare(0, prefix.size(), prefix) == 0) e = "agent: " + e.substr(prefix.size());
    return R::Fail(e);
  }
  return R::Ok(std::move(r).value()[0]);
}

Result<std::vector<schema::Prediction>> Agent::PredictBatch(const std::vector<Value>& states, const schema::Questions& qs,
                                                            const CallOptions& opts) const {
  using R = Result<std::vector<schema::Prediction>>;
  if (opts.cancel.cancelled()) return R::Fail(kCancelled);
  if (int64_t(states.size()) > kMaxStates) {
    return R::Fail("agent: " + std::to_string(states.size()) + " states in one call, the limit is " + std::to_string(kMaxStates));
  }
  if (opts.max_len > kMaxSequenceLength || opts.head_max_len > kMaxSequenceLength) {
    return R::Fail("agent: max_len and head_max_len are limited to " + std::to_string(kMaxSequenceLength));
  }
  if (!Enter()) return R::Fail(kClosed);
  struct Leaver {
    const Agent* a;
    ~Leaver() { a->Leave(); }
  } leaver{this};
  for (size_t i = 0; i < states.size(); i++) {
    if (states[i].is_null()) return R::Fail("agent: state " + std::to_string(i) + ": state is null; pass a string, an object or a list");
  }
  if (states.empty()) return R::Ok({});
  if (qs.size() == 0) {
    std::vector<schema::Prediction> preds(states.size());
    for (auto& p : preds) p.usage.bare = true;  // what the Python ONNX agent returns with no questions
    return R::Ok(std::move(preds));
  }
  return Chunks(states, qs, opts);
}

Result<std::vector<sequence::Sequence>> Agent::Encode(const Value& state, const schema::Questions& qs, const CallOptions& c) const {
  using R = Result<std::vector<sequence::Sequence>>;
  auto st = sequence::EncodeState(tok_, state);
  if (!st.ok()) return R::Fail(st.error());
  sequence::Budget budget{cfg_.max_len, cfg_.head_max_len};
  if (c.max_len > 0) budget.max_len = c.max_len;
  if (c.head_max_len > 0) budget.head_max_len = c.head_max_len;
  std::vector<sequence::Sequence> seqs;
  seqs.reserve(qs.size());
  for (size_t i = 0; i < qs.size(); i++) {
    sequence::MarkersError me;
    auto s = sequence::Build(tok_, qs.at(i).question, st.value(), budget, &me);
    if (!s.ok()) return R::Fail(me.max_len != 0 ? me.Message(qs.at(i).id) : s.error());
    seqs.push_back(std::move(s).value());
  }
  return R::Ok(std::move(seqs));
}

Result<std::vector<schema::Prediction>> Agent::Chunks(const std::vector<Value>& states, const schema::Questions& qs,
                                                      const CallOptions& c) const {
  using R = Result<std::vector<schema::Prediction>>;
  const size_t n = states.size();
  size_t chunk = n;
  if (c.batch_size > 0) chunk = std::min(size_t(c.batch_size), n);
  const bool reorder = c.sort_by_length && 1 < chunk && chunk < n;
  const size_t window = reorder ? chunk * size_t(kLookaheadBatches) : chunk;
  std::vector<schema::Prediction> results(n);
  for (size_t start = 0; start < n; start += window) {
    const size_t len = std::min(window, n - start);
    Encoded encoded(len);
    for (size_t i = 0; i < len; i++) {
      if (c.cancel.cancelled()) return R::Fail(kCancelled);
      auto e = Encode(states[start + i], qs, c);
      if (!e.ok()) return R::Fail("agent: state " + std::to_string(start + i) + ": " + e.error());
      encoded[i] = std::move(e).value();
    }
    std::vector<size_t> order(len);
    for (size_t i = 0; i < len; i++) order[i] = i;
    if (reorder) {
      std::vector<size_t> longest(len, 0);
      for (size_t i = 0; i < len; i++) {
        for (const auto& s : encoded[i]) longest[i] = std::max(longest[i], s.ids.size());
      }
      std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return longest[x] < longest[y]; });
    }
    std::vector<size_t> sizes;  // states per run
    if (c.batch_size > 0) {
      for (size_t off = 0; off < len; off += chunk) sizes.push_back(std::min(chunk, len - off));
    } else {
      std::vector<int64_t> length(len), width(len);
      for (size_t i = 0; i < len; i++) {
        for (const auto& s : encoded[order[i]]) {
          length[i] = std::max(length[i], int64_t(s.ids.size()));
          width[i] = std::max(width[i], int64_t(s.markers.size()));
        }
      }
      sizes = collate::PlanRuns(qs.size(), length, width);
    }
    size_t off = 0;
    for (size_t take : sizes) {
      if (c.cancel.cancelled()) return R::Fail(kCancelled);
      if (closed_.load()) return R::Fail(kClosed);
      std::vector<size_t> idx(order.begin() + std::ptrdiff_t(off), order.begin() + std::ptrdiff_t(off + take));
      off += take;
      auto preds = Run(qs, encoded, idx, c);
      if (!preds.ok()) return R::Fail(preds.error());
      for (size_t j = 0; j < idx.size(); j++) results[start + idx[j]] = std::move(preds.value()[j]);
    }
  }
  return R::Ok(std::move(results));
}

namespace {

schema::Usage UsageOf(const schema::Questions& qs, const std::vector<sequence::Sequence>& seqs, int64_t input_tokens) {
  schema::Usage u;
  u.input_tokens = input_tokens;
  u.state_tokens = seqs[0].truncation.state_tokens;
  for (size_t i = 0; i < seqs.size(); i++) {
    const std::string& id = qs.at(i).id;
    const sequence::Sequence& s = seqs[i];
    u.state_tokens_dropped = std::max(u.state_tokens_dropped, s.truncation.state_tokens_dropped);
    if (s.truncation.truncated) u.truncated_questions.push_back(id);
    if (s.stats.Collapsed()) {
      schema::CollapsedQuestion c;
      c.id = id;
      c.options.total = s.stats.options;
      c.options.distinct = s.stats.options_distinct;
      c.options.tokens_per_option = s.stats.tokens_per_option;
      u.collapsed_options.push_back(std::move(c));
    }
  }
  u.truncated = u.state_tokens_dropped > 0;
  return u;
}

}  // namespace

Result<std::vector<schema::Prediction>> Agent::Run(const schema::Questions& qs, const Encoded& encoded,
                                                   const std::vector<size_t>& idx, const CallOptions& c) const {
  using R = Result<std::vector<schema::Prediction>>;
  std::vector<collate::Row> rows;
  rows.reserve(idx.size() * qs.size());
  for (size_t i : idx) {
    for (size_t r = 0; r < encoded[i].size(); r++) {
      rows.push_back({encoded[i][r].ids, encoded[i][r].markers, qs.at(r).question.type()});
    }
  }
  auto batch = collate::Collate(rows, pad_id_);
  if (!batch.ok()) return R::Fail("agent: " + batch.error());
  const infer::Batch& b = batch.value();
  auto fail = [&](const std::string& why) {
    return R::Fail("agent: running " + std::to_string(b.rows) + " rows of " + std::to_string(b.length) + " tokens with " +
                   std::to_string(b.markers) + " marker slots: " + why);
  };
  if (std::string bad = b.Validate(); !bad.empty()) return fail(bad);
  Result<infer::Output> out = Result<infer::Output>::Fail("");
  try {
    out = sess_->Run(b, c.cancel);
  } catch (const std::exception& e) {
    return fail(std::string("session threw: ") + e.what());
  } catch (...) {
    return fail("session threw");
  }
  if (!out.ok()) return infer::IsCancelled(out.error()) ? R::Fail(kCancelled) : fail(out.error());
  if (std::string bad = out.value().Check(b); !bad.empty()) return fail(bad);
  const infer::Output& o = out.value();

  std::vector<schema::Prediction> preds(idx.size());
  size_t row = 0;
  for (size_t j = 0; j < idx.size(); j++) {
    const auto& seqs = encoded[idx[j]];
    int64_t input_tokens = 0;
    for (size_t r = 0; r < seqs.size(); r++) {
      const size_t at = row + r;
      for (int64_t m = 0; m < b.length; m++) input_tokens += b.attention_mask[at * size_t(b.length) + size_t(m)];
      const auto& e = qs.at(r);
      auto act = decode::ActionProbabilities({o.act_logits[at * 2], o.act_logits[at * 2 + 1]});
      if (!act.ok()) return fail("action logits of question " + e.id + ": " + act.error());
      decode::Row dr;
      dr.logits.assign(o.logits.begin() + std::ptrdiff_t(at * size_t(b.markers)),
                       o.logits.begin() + std::ptrdiff_t((at + 1) * size_t(b.markers)));
      dr.k = int64_t(seqs[r].markers.size());
      dr.act = act.value()[0];
      auto ans = decode::Decode(e.question, dr, "", &temps_);
      if (!ans.ok()) return fail("question " + e.id + ": " + ans.error());
      preds[j].answers.push_back({e.id, std::move(ans).value()});
    }
    preds[j].usage = UsageOf(qs, seqs, input_tokens);
    row += seqs.size();
  }
  return R::Ok(std::move(preds));
}

}  // namespace agent
}  // namespace laya
