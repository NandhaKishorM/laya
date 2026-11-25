#include "laya/sequence.hpp"

#include <algorithm>
#include <cstdio>
#include <set>

#include "cnum.hpp"
#include "laya/pyjson.hpp"

namespace laya {
namespace sequence {

using schema::Question;
using tokenizer::Tokenizer;

namespace {

std::string N(int64_t i) { return std::to_string(i); }

std::string ReplaceAll(std::string s, const std::string& from, const std::string& to) {
  if (from.empty()) return s;
  std::string out;
  for (size_t at = 0;;) {
    size_t hit = s.find(from, at);
    if (hit == std::string::npos) {
      out.append(s, at, std::string::npos);
      return out;
    }
    out.append(s, at, hit - at);
    out += to;
    at = hit + from.size();
  }
}

struct Specials {
  int32_t cls = 0, sep = 0, mask = 0;
  std::string mask_token;
};

Result<Specials> GetSpecials(const Tokenizer& tok) {
  Specials sp;
  auto cls = tok.Cls(), sep = tok.Sep(), mask = tok.Mask();
  sp.mask_token = tok.MaskToken();
  const char* what = "tokenizer names no required special token";
  if (!cls) return Result<Specials>::Fail(std::string(what) + ": no CLS token");
  if (!sep) return Result<Specials>::Fail(std::string(what) + ": no SEP token");
  if (!mask || sp.mask_token.empty()) return Result<Specials>::Fail(std::string(what) + ": no mask token");
  sp.cls = *cls;
  sp.sep = *sep;
  sp.mask = *mask;
  return Result<Specials>::Ok(std::move(sp));
}

int64_t FloorDiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if (a % b != 0 && a < 0) q--;
  return q;
}

bool CheckBudget(const Budget& b, std::string* err) {
  if (b.max_len <= 0 || b.head_max_len <= 0) {
    *err = "sequence: budget needs a positive MaxLen and HeadMaxLen, got " + N(b.max_len) + " and " + N(b.head_max_len);
    return false;
  }
  return true;
}

}  // namespace

int64_t Budget::DefaultWindow() const { return std::max<int64_t>(64, max_len - head_max_len - 8); }

Result<std::string> SerializeState(const pyvalue::Value& state) {
  if (state.kind() == pyvalue::Value::Kind::String) return Result<std::string>::Ok(state.as_string());
  auto text = pyvalue::Dumps(state);
  if (!text.ok()) return Result<std::string>::Fail("state cannot be serialized as laya serializes it: " + text.error());
  return text;
}

Result<State> EncodeState(const Tokenizer& tok, const pyvalue::Value& state) {
  auto sp = GetSpecials(tok);
  if (!sp.ok()) return Result<State>::Fail(sp.error());
  auto text = SerializeState(state);
  if (!text.ok()) return Result<State>::Fail(text.error());
  State out;
  out.keep = state.kind() == pyvalue::Value::Kind::Array ? Keep::End : Keep::Start;
  out.ids = tok.Encode(ReplaceAll(text.value(), sp.value().mask_token, " "));
  return Result<State>::Ok(std::move(out));
}

std::string MarkersError::Message(const std::string& id) const {
  std::string prefix = id.empty() ? "" : "question " + pyjson::Repr(id) + ": ";
  return prefix + "only " + N(kept) + " of its " + N(total) + " option markers fit in max_len=" + N(max_len) +
         " with head_max_len=" + N(head_max_len) +
         " spent on the question; lower head_max_len, raise max_len, or use fewer options";
}

Result<Head> BuildHead(const Tokenizer& tok, const Question& q, int64_t head_max_len) {
  auto spr = GetSpecials(tok);
  if (!spr.ok()) return Result<Head>::Fail(spr.error());
  const Specials& sp = spr.value();
  if (q.type() == schema::Type::Invalid || q.NumOptions() == 0) {
    return Result<Head>::Fail(
        "sequence: the zero Question has no head; build one with Question::Choice, ChoiceOf, Score or Noul");
  }
  std::vector<std::string> opts = q.RenderedOptions();
  std::vector<int> order = q.OptionOrder();
  if (order.empty()) {
    for (size_t i = 0; i < opts.size(); i++) order.push_back(int(i));
  }
  std::string ins = ReplaceAll(q.instructions(), sp.mask_token, " ");
  std::vector<int32_t> head_ids = tok.Encode(std::string(schema::Name(q.type())) + " question: " + ins);
  std::vector<std::vector<int32_t>> spans(order.size());
  int64_t total = 0;
  for (size_t slot = 0; slot < order.size(); slot++) {
    std::string text = " " + ReplaceAll(opts[size_t(order[slot])], sp.mask_token, " ");
    spans[slot].push_back(sp.mask);
    std::vector<int32_t> body = tok.Encode(text, kOptionTokenCap);
    spans[slot].insert(spans[slot].end(), body.begin(), body.end());
    total += int64_t(spans[slot].size());
  }
  int64_t budget = head_max_len - total;
  std::optional<int64_t> per_option;
  if (budget < kMinOptionBudget) {
    int64_t per = std::max(kMinTokensPerOption,
                           FloorDiv(head_max_len - kMinOptionBudget, std::max<int64_t>(1, int64_t(spans.size()))));
    per_option = per;
    total = 0;
    for (auto& span : spans) {
      if (int64_t(span.size()) > per) span.resize(size_t(per));
      total += int64_t(span.size());
    }
    budget = head_max_len - total;
  }
  int64_t keep = std::max(kMinInstructionTokens, budget);
  if (keep < int64_t(head_ids.size())) head_ids.resize(size_t(keep));

  Head h;
  h.ids.push_back(sp.cls);
  h.ids.insert(h.ids.end(), head_ids.begin(), head_ids.end());
  h.ids.push_back(sp.sep);
  std::set<std::vector<int32_t>> distinct;
  for (const auto& span : spans) {
    h.markers.push_back(int64_t(h.ids.size()));
    h.ids.insert(h.ids.end(), span.begin(), span.end());
    distinct.insert(span);
  }
  h.ids.push_back(sp.sep);
  h.stats = {int64_t(spans.size()), int64_t(distinct.size()), per_option};
  return Result<Head>::Ok(std::move(h));
}

Result<Sequence> Build(const Tokenizer& tok, const Question& q, const State& state, const Budget& b,
                       MarkersError* markers_error) {
  std::string err;
  if (!CheckBudget(b, &err)) return Result<Sequence>::Fail(err);
  auto head = BuildHead(tok, q, b.head_max_len);
  if (!head.ok()) return Result<Sequence>::Fail(head.error());
  const Head& h = head.value();
  int32_t sep = *tok.Sep();  // BuildHead found it
  int64_t room = std::max<int64_t>(0, b.max_len - int64_t(h.ids.size()) - 1);
  int64_t have = int64_t(state.ids.size());
  int64_t used = std::min(have, room);
  Sequence s;
  s.ids = h.ids;
  size_t from = state.keep == Keep::End ? size_t(have - used) : 0;
  s.ids.insert(s.ids.end(), state.ids.begin() + std::ptrdiff_t(from), state.ids.begin() + std::ptrdiff_t(from + size_t(used)));
  s.ids.push_back(sep);
  if (int64_t(s.ids.size()) > b.max_len) s.ids.resize(size_t(b.max_len));
  for (int64_t m : h.markers) {
    if (m < b.max_len) s.markers.push_back(m);
  }
  if (s.markers.size() != h.markers.size()) {
    MarkersError e{int64_t(s.markers.size()), int64_t(h.markers.size()), b.max_len, b.head_max_len};
    if (markers_error != nullptr) *markers_error = e;
    return Result<Sequence>::Fail(e.Message());
  }
  s.stats = h.stats;
  s.truncation = {have, used, have - used, used < have};
  return Result<Sequence>::Ok(std::move(s));
}

Result<int64_t> StateRoom(const Tokenizer& tok, const Question& q, const Budget& b) {
  std::string err;
  if (!CheckBudget(b, &err)) return Result<int64_t>::Fail(err);
  auto head = BuildHead(tok, q, b.head_max_len);
  if (!head.ok()) return Result<int64_t>::Fail(head.error());
  return Result<int64_t>::Ok(std::max<int64_t>(0, b.max_len - int64_t(head.value().ids.size()) - 1));  // -1: closing [SEP]
}

namespace {
constexpr int64_t kClampWarnRatio = 2;
constexpr int64_t kBatchBlowup = 2;

std::string Ratio1(double x) { return cnum::Fixed(x, 1); }
}  // namespace

Result<WindowPlan> PlanWindows(const Tokenizer& tok, const std::vector<Question>& questions, const Budget& b,
                               const WindowRequest& req, PlanError* kind) {
  if (kind != nullptr) *kind = PlanError::Other;
  std::string err;
  if (!CheckBudget(b, &err)) return Result<WindowPlan>::Fail(err);
  std::vector<int64_t> rooms;
  for (const auto& q : questions) {
    auto r = StateRoom(tok, q, b);
    if (!r.ok()) return Result<WindowPlan>::Fail(r.error());
    rooms.push_back(r.value());
  }
  int64_t requested = req.window > 0 ? req.window : b.DefaultWindow();
  int64_t size = requested;
  int64_t room = rooms.empty() ? size : *std::min_element(rooms.begin(), rooms.end());
  if (room <= 0) {
    if (kind != nullptr) *kind = PlanError::NoRoom;
    return Result<WindowPlan>::Fail(
        "predict_long: the questions' options fill the whole sequence (max_len=" + N(b.max_len) + ", head_max_len=" +
        N(b.head_max_len) + "), leaving no room for the state; no window can carry any of it. A label set this large "
        "is what laya.shortlist.predict_shortlist is for");
  }
  WindowPlan plan;
  if (size > room) {
    if (req.window > 0) {
      plan.warnings.push_back("laya: predict_long: window=" + N(size) + " is wider than the " + N(room) +
                              " state tokens these questions leave inside max_len=" + N(b.max_len) +
                              ", so every window would be truncated to " + N(room) +
                              " on the way to the model; scanning with window=" + N(room) + " instead");
    } else if (size / kClampWarnRatio >= room) {  // size >= room * kClampWarnRatio, without the overflow
      plan.warnings.push_back("laya: predict_long: these questions leave only " + N(room) + " of max_len=" +
                              N(b.max_len) + " for the state (their heads take the rest), so the scan window is "
                              "capped " + N(size) + " -> " + N(room) + " and roughly " +
                              Ratio1(double(size) / double(room)) + "x as many windows -- each a full forward pass -- "
                              "are needed to read the document. Fewer or shorter options, a larger max_len, or "
                              "laya.shortlist.predict_shortlist for a large label set will all cost less than scanning "
                              "at this width");
    }
    size = room;
  }
  int64_t step = req.stride > 0 ? req.stride : std::max<int64_t>(1, size / 2);
  if (step > size) {
    if (size < requested && req.stride > 0 && req.stride <= requested) {
      plan.warnings.push_back("laya: predict_long: stride=" + N(step) + " was a 50% step for the window=" +
                              N(requested) + " you asked for, but the window was reduced to " + N(size) +
                              " to fit the room these questions leave; scanning with stride=" +
                              N(std::max<int64_t>(1, size / 2)) + " instead");
      step = std::max<int64_t>(1, size / 2);
    } else {
      if (kind != nullptr) *kind = PlanError::StrideGap;
      return Result<WindowPlan>::Fail(
          "predict_long: stride=" + N(step) + " steps past the " + N(size) + "-token window" +
          (size < requested ? " these questions leave room for" : "") + ", so " + N(step - size) +
          " tokens between every pair of windows would be read by no window at all; pass stride <= " + N(size));
    }
  }
  plan.window = size;
  plan.stride = step;
  plan.room = room;
  if (kind != nullptr) *kind = PlanError::None;
  return Result<WindowPlan>::Ok(std::move(plan));
}

Cap BatchCap(int64_t n_windows, int64_t window, int64_t config_window, int64_t batch_size) {
  if (batch_size > 0) return {batch_size, true};
  if (window >= config_window) return {0, false};  // not capped: one pass, exactly as before
#if defined(__SIZEOF_INT128__)
  __extension__ typedef __int128 Wide;
#else
  typedef long long Wide;  // no 128-bit integer: callers beyond 2^31 windows are out of contract
#endif
  Wide num = Wide(n_windows) * Wide(std::max<int64_t>(1, FloorDiv(window, 2)));
  Wide den = Wide(std::max<int64_t>(1, FloorDiv(config_window, 2)));
  Wide unclamped = num > 0 ? (num + den - 1) / den : 1;  // ceil(num / den), at least 1
  if (unclamped < 1) unclamped = 1;
  if (Wide(n_windows) <= unclamped * kBatchBlowup) return {0, false};  // a mild cap: keep the single shared pass
  return {int64_t(unclamped), true};  // below n_windows, so it fits
}

}  // namespace sequence
}  // namespace laya
