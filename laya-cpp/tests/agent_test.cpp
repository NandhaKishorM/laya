#include "laya/agent.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "checkpoint.hpp"
#include "golden.hpp"
#include "laya/collate.hpp"
#include "stub_session.hpp"

namespace {

using laya::pyvalue::Value;
namespace ag = laya::agent;
namespace sc = laya::schema;
namespace in = laya::infer;
using laya_test::RunRecord;
using laya_test::StubSession;
using Kind = Value::Kind;

constexpr const char* kModels[] = {"english", "multilingual"};
const char* const kConfig = R"({"max_len":512,"head_max_len":192,"option_layout":"sequential"})";

std::string Root() {
  const char* r = std::getenv("LAYA_CHECKPOINTS");
  return r ? r : "";
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// ---- collate and infer: no checkpoint needed ----

laya::collate::Row Seq(std::vector<int32_t> ids, std::vector<int64_t> markers, sc::Type t = sc::Type::Choice) {
  return {std::move(ids), std::move(markers), t};
}

TEST(Collate, PadsToTheLongestAndWidestAndMasks) {
  auto b = laya::collate::Collate({Seq({5, 6, 7}, {0, 2, 1}), Seq({9}, {0}, sc::Type::Noul)}, 99);
  ASSERT_TRUE(b.ok()) << b.error();
  const in::Batch& x = b.value();
  EXPECT_EQ(x.rows, 2);
  EXPECT_EQ(x.length, 3);
  EXPECT_EQ(x.markers, 3);
  EXPECT_EQ(x.input_ids, (std::vector<int64_t>{5, 6, 7, 9, 99, 99}));
  EXPECT_EQ(x.attention_mask, (std::vector<int64_t>{1, 1, 1, 1, 0, 0}));
  EXPECT_EQ(x.marker_pos, (std::vector<int64_t>{0, 2, 1, 0, 0, 0}));
  EXPECT_EQ(x.marker_mask, (std::vector<uint8_t>{1, 1, 1, 1, 0, 0}));
  EXPECT_EQ(x.qtype, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(x.Validate(), "");
}

TEST(Collate, OneOptionRowsGetAMaskedSecondSlot) {
  auto b = laya::collate::Collate({Seq({1, 2}, {1})}, 0);
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(b.value().markers, 2);
  EXPECT_EQ(b.value().marker_mask, (std::vector<uint8_t>{1, 0}));
}

TEST(Collate, RefusesBadRows) {
  using laya::collate::Collate;
  EXPECT_FALSE(Collate({}, 0).ok());
  EXPECT_FALSE(Collate({Seq({}, {})}, 0).ok());
  EXPECT_FALSE(Collate({Seq({1}, {1})}, 0).ok());                       // marker at the end
  EXPECT_FALSE(Collate({Seq({1, 2}, {-1})}, 0).ok());
  EXPECT_FALSE(Collate({Seq({1, -2}, {0})}, 0).ok());
  EXPECT_FALSE(Collate({Seq({1}, {0}, sc::Type::Invalid)}, 0).ok());
  EXPECT_FALSE(Collate({Seq({1}, {0})}, -1).ok());
}

TEST(Collate, RefusesAHostileSize) {
  // 2 rows of 2^23+1 tokens hold more than kMaxTensorElements: refused before the tensors are allocated.
  laya::collate::Row big = Seq(std::vector<int32_t>((size_t(1) << 23) + 1, 1), {0});
  std::vector<laya::collate::Row> rows(2, big);
  auto b = laya::collate::Collate(rows, 0);
  EXPECT_FALSE(b.ok());
}

TEST(Infer, ValidateCatchesEachShape) {
  in::Batch ok = laya::collate::Collate({Seq({1, 2}, {0, 1})}, 0).value();
  EXPECT_EQ(ok.Validate(), "");
  auto bad = [&](auto mutate) {
    in::Batch b = ok;
    mutate(b);
    return b.Validate();
  };
  EXPECT_NE(bad([](in::Batch& b) { b.input_ids.pop_back(); }), "");
  EXPECT_NE(bad([](in::Batch& b) { b.marker_mask.push_back(0); }), "");
  EXPECT_NE(bad([](in::Batch& b) { b.marker_pos[0] = 2; }), "");
  EXPECT_NE(bad([](in::Batch& b) { b.input_ids[0] = -1; }), "");
  EXPECT_NE(bad([](in::Batch& b) { b.qtype[0] = 3; }), "");
  EXPECT_NE(bad([](in::Batch& b) { b.rows = -1; }), "");
  EXPECT_NE(bad([](in::Batch& b) { b.markers = 1; }), "");
  EXPECT_NE(bad([](in::Batch& b) { b.rows = int64_t(1) << 62; }), "");
}

TEST(Infer, OutputCheckDoesNotOverflowOnAHostileBatch) {
  in::Batch b;
  b.rows = int64_t(1) << 40;
  b.markers = int64_t(1) << 40;
  in::Output o;
  EXPECT_NE(o.Check(b), "");
  b.rows = -1;
  EXPECT_NE(o.Check(b), "");
}

TEST(Collate, PlanRunsSplitsOnlyAtTheTensorCap) {
  using laya::collate::PlanRuns;
  // 32769 rows of 512 tokens: one row over 2^24 elements.
  EXPECT_EQ(PlanRuns(1, std::vector<int64_t>(32768, 512), std::vector<int64_t>(32768, 2)), (std::vector<size_t>{32768}));
  EXPECT_EQ(PlanRuns(1, std::vector<int64_t>(32769, 512), std::vector<int64_t>(32769, 2)), (std::vector<size_t>{32768, 1}));
  // three rows per state: 10922 states fit, the 10923rd starts a new run
  EXPECT_EQ(PlanRuns(3, std::vector<int64_t>(10923, 512), std::vector<int64_t>(10923, 2)), (std::vector<size_t>{10922, 1}));
  // the marker width splits too
  EXPECT_EQ(PlanRuns(1, std::vector<int64_t>(3, 4), std::vector<int64_t>(3, 6000000)), (std::vector<size_t>{2, 1}));
  // one state over the cap alone is still one run (collate refuses it)
  EXPECT_EQ(PlanRuns(1, {int64_t(1) << 25}, {2}), (std::vector<size_t>{1}));
  EXPECT_TRUE(PlanRuns(1, {}, {}).empty());
}

TEST(Infer, OutputCheck) {
  in::Batch b = laya::collate::Collate({Seq({1, 2}, {0, 1})}, 0).value();
  in::Output o;
  o.logits = {0, 0};
  o.act_logits = {0, 0};
  EXPECT_EQ(o.Check(b), "");
  o.logits.pop_back();
  EXPECT_NE(o.Check(b), "");
}

TEST(CancelToken, CopiesShareAFlagAndADefaultFires) {
  laya::CancelToken none;
  EXPECT_FALSE(none.cancelled());
  none.Cancel();
  EXPECT_TRUE(none.cancelled());
  laya::CancelToken a = laya::CancelToken::Make(), b = a;
  EXPECT_FALSE(b.cancelled());
  a.Cancel();
  EXPECT_TRUE(b.cancelled());
}

// ---- the agent on the stub session, tokenizer-backed ----

struct Rig {
  std::unique_ptr<ag::Agent> agent;
  std::shared_ptr<StubSession::State> st;
};

Rig OpenRig(const char* model, const char* config = kConfig) {
  Rig r;
  const laya::tokenizer::Tokenizer* tok = laya_test::LoadCheckpoint(model);
  if (tok == nullptr) return r;
  auto s = std::make_unique<StubSession>();
  r.st = s->state();
  auto a = ag::Agent::OpenParts(config, *tok, std::move(s));
  EXPECT_TRUE(a.ok()) << (a.ok() ? "" : a.error());
  if (a.ok()) r.agent = std::move(a).value();
  return r;
}

#define LAYA_RIG_OPEN(var, open_expr)                                                      \
  Rig var = open_expr;                                                                     \
  if (!var.agent) {                                                                        \
    if (laya_test::RequireGoldens()) ADD_FAILURE() << "no checkpoint";                     \
    else GTEST_SKIP() << "LAYA_CHECKPOINTS unset (local run)";                             \
    return;                                                                                \
  }
#define LAYA_RIG(var, model) LAYA_RIG_OPEN(var, OpenRig(model))
#define LAYA_RIG_CFG(var, model, cfg) LAYA_RIG_OPEN(var, OpenRig(model, cfg))

sc::Questions Mix() {
  return sc::Questions::Parse(R"({
    "intent":{"type":"choice","instructions":"What does the customer want?","criteria":{"refund":"asks for money back","status":"asks where the order is","other":"anything else"}},
    "urgency":{"type":"score","instructions":"How urgent is it?","criteria":["not at all","slightly","moderately","very","critical"]},
    "angry":{"type":"noul","instructions":"Is the customer angry?"}})").value();
}

std::vector<Value> States(int n) {
  std::vector<Value> out;
  for (int i = 0; i < n; i++) {
    std::string s = "state";
    for (int j = 0; j < (i * 7) % 23; j++) s += " alpha bravo";
    out.push_back(Value::String(s + " " + std::to_string(i)));
  }
  return out;
}

std::vector<std::string> Jsons(const std::vector<sc::Prediction>& p) {
  std::vector<std::string> out;
  for (const auto& x : p) out.push_back(x.ToJson().value());
  return out;
}

// Which rows each run holds, compared with ONNXAgent._infer_batch on the same states; numbers within
// the tolerance Python's float32 needs, the chosen label only when it was decisive.
struct Tally {
  int64_t numbers = 0, differing = 0, failures = 0, ties = 0, choices = 0;
  std::string where;
  void Fail(const std::string& m) {
    if (failures++ < 10) ADD_FAILURE() << where << ": " << m;
  }
  void Compare(const std::string& path, const std::string& key, const Value& got, const Value& want) {
    auto num = [](const Value& v) { return v.kind() == Kind::Int || v.kind() == Kind::Float; };
    if (want.kind() == Kind::Object) {
      bool same = got.kind() == Kind::Object && got.size() == want.size();
      for (size_t i = 0; same && i < want.size(); i++) same = got.key(i) == want.key(i);
      if (!same) return Fail(path + ": keys differ: " + laya::pyvalue::Repr(got) + " vs " + laya::pyvalue::Repr(want));
      for (size_t i = 0; i < want.size(); i++) Compare(path + "." + want.key(i), want.key(i), got.value(i), want.value(i));
    } else if (want.kind() == Kind::Array) {
      if (got.kind() != Kind::Array || got.size() != want.size()) return Fail(path + ": array differs");
      for (size_t i = 0; i < want.size(); i++) Compare(path + "[]", key, got.items()[i], want.items()[i]);
    } else if (num(want)) {
      if (!num(got)) return Fail(path + ": not a number");
      const double tol = key == "score" ? 8e-4 : 2e-4, d = std::fabs(laya_test::Num(got) - laya_test::Num(want));
      numbers++;
      if (laya::pyvalue::Repr(got) != laya::pyvalue::Repr(want)) differing++;
      if (d > tol) Fail(path + ": off by " + std::to_string(d));
    } else if (laya::pyvalue::Repr(got) != laya::pyvalue::Repr(want)) {
      Fail(path + ": " + laya::pyvalue::Repr(got) + " vs " + laya::pyvalue::Repr(want));
    }
  }
  void Answer(const std::string& path, const Value& got, const Value& want) {
    if (const Value* t = want.Find("type"); t != nullptr && t->as_string() == "choice") {
      std::vector<double> ps;
      const Value* probs = want.Find("probabilities");
      for (size_t i = 0; i < probs->size(); i++) ps.push_back(laya_test::Num(probs->value(i)));
      std::sort(ps.rbegin(), ps.rend());
      if (ps.size() > 1 && ps[0] - ps[1] <= 1e-3) {
        ties++;
        auto trim = [](const Value& o) {
          Value r = Value::Object();
          for (size_t i = 0; i < o.size(); i++) {
            if (o.key(i) != "choice") r.Set(o.key(i), o.value(i));
          }
          return r;
        };
        return Compare(path, "", trim(got), trim(want));
      }
      choices++;
    }
    Compare(path, "", got, want);
  }
};

RunRecord RunFrom(const Value& v) {
  auto ints = [&](const char* k) {
    std::vector<int64_t> out;
    for (const Value& x : v.Find(k)->items()) out.push_back(x.as_int());
    return out;
  };
  RunRecord r;
  r.rows = v.Find("rows")->as_int(), r.length = v.Find("length")->as_int(), r.markers = v.Find("markers")->as_int();
  r.id_sums = ints("id_sums"), r.tokens = ints("tokens"), r.marker_sums = ints("marker_sums"), r.qtypes = ints("qtypes");
  return r;
}

bool SameRun(const RunRecord& a, const RunRecord& b) {
  return a.rows == b.rows && a.length == b.length && a.markers == b.markers && a.id_sums == b.id_sums &&
         a.tokens == b.tokens && a.marker_sums == b.marker_sums && a.qtypes == b.qtypes;
}

TEST(AgentGolden, PredictPlan) {
  for (const char* model : kModels) {
    LAYA_GOLDEN_LIST(golden, "go_extra.json", "predict_plan");
    const Value* cases = golden->Find(model)->Find("cases");
    LAYA_EXPECT_ROWS(cases, 24);
    const Value* defs = golden->Find("questions");
    const std::string cfg = ReadFile(Root() + "/" + model + "/rl_agent_config.json");
    LAYA_RIG_CFG(rig, model, cfg.c_str());
    Tally t;
    int64_t runs = 0;
    for (const Value& rec : cases->items()) {
      t.where = std::string(model) + "/" + rec.Find("name")->as_string();
      auto qs = sc::Questions::Parse(laya::pyvalue::Dumps(*defs->Find(rec.Find("questions")->as_string())).value());
      ASSERT_TRUE(qs.ok()) << qs.error();
      ag::CallOptions o;
      if (!rec.Find("batch_size")->is_null()) o.batch_size = rec.Find("batch_size")->as_int();
      o.sort_by_length = rec.Find("sort")->as_bool();
      // The checkpoints set no language temperatures, so the recorded lang changes nothing.
      rig.st->runs.clear();
      auto preds = rig.agent->PredictBatch(rec.Find("states")->items(), qs.value(), o);
      ASSERT_TRUE(preds.ok()) << t.where << ": " << preds.error();
      const Value* want_runs = rec.Find("runs");
      if (rig.st->runs.size() != want_runs->size()) {
        t.Fail(std::to_string(rig.st->runs.size()) + " runs, want " + std::to_string(want_runs->size()));
        continue;
      }
      runs += int64_t(rig.st->runs.size());
      for (size_t i = 0; i < want_runs->size(); i++) {
        if (!SameRun(rig.st->runs[i], RunFrom(want_runs->items()[i]))) t.Fail("run " + std::to_string(i) + " differs");
      }
      const Value* want = rec.Find("results");
      ASSERT_EQ(preds.value().size(), want->size()) << t.where;
      for (size_t i = 0; i < want->size(); i++) {
        const Value got = laya::pyvalue::Parse(preds.value()[i].ToJson().value()).value();
        const Value& w = want->items()[i];
        const std::string path = t.where + "/state" + std::to_string(i);
        t.Compare(path + ".model", "model", *got.Find("model"), *w.Find("model"));
        if (laya::pyvalue::Dumps(*got.Find("usage")).value() != laya::pyvalue::Dumps(*w.Find("usage")).value()) {
          t.Fail(path + " usage\n got " + laya::pyvalue::Dumps(*got.Find("usage")).value() + "\nwant " +
                 laya::pyvalue::Dumps(*w.Find("usage")).value());
        }
        const Value* ga = got.Find("answers");
        const Value* wa = w.Find("answers");
        std::vector<std::string> want_ids;
        for (const Value& id : w.Find("answer_ids")->items()) want_ids.push_back(id.as_string());
        std::vector<std::string> got_ids;
        for (size_t k = 0; k < ga->size(); k++) got_ids.push_back(ga->key(k));
        if (got_ids != want_ids) {
          t.Fail(path + " answer ids differ");
          continue;
        }
        for (size_t k = 0; k < wa->size(); k++) t.Answer(path + "." + wa->key(k), ga->value(k), wa->value(k));
      }
    }
    std::printf("%s: %lld runs, %lld numbers, %lld differ at all, %lld choices checked, %lld near-ties skipped\n", model,
                (long long)runs, (long long)t.numbers, (long long)t.differing, (long long)t.choices, (long long)t.ties);
    EXPECT_EQ(t.failures, 0);
    EXPECT_GE(t.numbers, 2000);
    EXPECT_GE(runs, 80);
    EXPECT_LE(double(t.differing), 0.05 * double(t.numbers));
  }
}

// ---- behavior on the stub ----

TEST(Agent, ResultsComeBackInInputOrderWhateverTheBatching) {
  for (const char* model : kModels) {
    LAYA_RIG(rig, model);
    const auto qs = Mix();
    const auto states = States(19);
    auto whole = rig.agent->PredictBatch(states, qs);
    ASSERT_TRUE(whole.ok()) << whole.error();
    for (int64_t batch : {1, 2, 5, 19, 50}) {
      for (bool sort : {false, true}) {
        ag::CallOptions o;
        o.batch_size = batch;
        o.sort_by_length = sort;
        auto p = rig.agent->PredictBatch(states, qs, o);
        ASSERT_TRUE(p.ok()) << p.error();
        EXPECT_EQ(Jsons(p.value()), Jsons(whole.value())) << model << " batch " << batch << " sort " << sort;
      }
    }
    for (size_t i : {size_t(0), size_t(7), size_t(18)}) {  // a single Predict agrees with its slot
      auto one = rig.agent->Predict(states[i], qs);
      ASSERT_TRUE(one.ok());
      EXPECT_EQ(one.value().ToJson().value(), whole.value()[i].ToJson().value());
    }
  }
}

TEST(Agent, BatchSizeSetsTheNumberOfRunsAndSortGroupsByLength) {
  LAYA_RIG(rig, "english");
  const auto qs = Mix();
  const auto states = States(10);
  struct Case {
    int64_t batch;
    size_t runs;
  };
  for (Case c : {Case{0, 1}, Case{-3, 1}, Case{1, 10}, Case{3, 4}, Case{10, 1}, Case{99, 1}}) {
    ag::CallOptions o;
    o.batch_size = c.batch;
    rig.st->runs.clear();
    ASSERT_TRUE(rig.agent->PredictBatch(states, qs, o).ok());
    EXPECT_EQ(rig.st->runs.size(), c.runs) << "batch " << c.batch;
    for (const RunRecord& r : rig.st->runs) EXPECT_EQ(r.rows % 3, 0);  // three questions per state
  }
  // Sorted, chunks hold neighbors in length: the longest row of each run never shrinks run to run.
  ag::CallOptions o;
  o.batch_size = 3;
  o.sort_by_length = true;
  rig.st->runs.clear();
  ASSERT_TRUE(rig.agent->PredictBatch(states, qs, o).ok());
  int64_t prev = 0;
  for (const RunRecord& r : rig.st->runs) {
    EXPECT_GE(r.length, prev);
    prev = r.length;
  }
  EXPECT_GT(rig.st->runs.back().length, rig.st->runs.front().length);
}

TEST(Agent, UsageCountsTheMaskedTokens) {
  LAYA_RIG(rig, "english");
  const auto qs = Mix();
  rig.st->runs.clear();
  auto p = rig.agent->Predict(Value::String("alpha bravo charlie"), qs);
  ASSERT_TRUE(p.ok());
  int64_t tokens = 0;
  for (int64_t t : rig.st->runs[0].tokens) tokens += t;
  EXPECT_EQ(p.value().usage.input_tokens, tokens);
  EXPECT_GT(tokens, 0);
}

TEST(Agent, EmptyInputs) {
  LAYA_RIG(rig, "english");
  auto none = rig.agent->PredictBatch({}, Mix());
  ASSERT_TRUE(none.ok());
  EXPECT_TRUE(none.value().empty());
  auto noq = rig.agent->PredictBatch(States(2), sc::Questions());
  ASSERT_TRUE(noq.ok());
  EXPECT_EQ(noq.value().size(), 2u);
  EXPECT_TRUE(noq.value()[0].answers.empty());
  EXPECT_TRUE(rig.st->runs.empty());
}

TEST(Agent, RefusesANullStateAndNamesIt) {
  LAYA_RIG(rig, "english");
  auto states = States(3);
  states[2] = Value::Null();
  auto p = rig.agent->PredictBatch(states, Mix());
  ASSERT_FALSE(p.ok());
  EXPECT_NE(p.error().find("agent: state 2:"), std::string::npos) << p.error();
  auto one = rig.agent->Predict(Value::Null(), Mix());
  ASSERT_FALSE(one.ok());
  EXPECT_EQ(one.error().find("state 0"), std::string::npos) << one.error();
  EXPECT_TRUE(rig.st->runs.empty());
}

TEST(Agent, RefusesHostileCallSettings) {
  LAYA_RIG(rig, "english");
  ag::CallOptions o;
  o.max_len = ag::kMaxSequenceLength + 1;
  EXPECT_FALSE(rig.agent->Predict(Value::String("x"), Mix(), o).ok());
  o = {};
  o.head_max_len = int64_t(1) << 40;
  EXPECT_FALSE(rig.agent->Predict(Value::String("x"), Mix(), o).ok());
  EXPECT_TRUE(rig.st->runs.empty());
}

TEST(Agent, SessionFailuresNameTheRun) {
  LAYA_RIG(rig, "english");
  rig.st->fail = "boom";
  auto p = rig.agent->Predict(Value::String("alpha"), Mix());
  ASSERT_FALSE(p.ok());
  EXPECT_NE(p.error().find("agent: running 3 rows of"), std::string::npos) << p.error();
  EXPECT_NE(p.error().find("boom"), std::string::npos);
  rig.st->fail.clear();
  rig.st->tamper = [](in::Output* o) { o->logits.pop_back(); };
  EXPECT_FALSE(rig.agent->Predict(Value::String("alpha"), Mix()).ok());
  rig.st->tamper = [](in::Output* o) { o->logits[0] = NAN; };
  EXPECT_FALSE(rig.agent->Predict(Value::String("alpha"), Mix()).ok());
  rig.st->tamper = [](in::Output* o) { o->act_logits[1] = INFINITY; };
  EXPECT_FALSE(rig.agent->Predict(Value::String("alpha"), Mix()).ok());
}

TEST(Agent, CancelBeforeTheCallRunsNothing) {
  LAYA_RIG(rig, "english");
  ag::CallOptions o;
  o.cancel = laya::CancelToken::Make();
  o.cancel.Cancel();
  auto p = rig.agent->PredictBatch(States(4), Mix(), o);
  ASSERT_FALSE(p.ok());
  EXPECT_TRUE(ag::IsCancelled(p.error()));
  EXPECT_TRUE(rig.st->runs.empty());
}

TEST(Agent, CancelBetweenRunsStopsTheNextOne) {
  LAYA_RIG(rig, "english");
  ag::CallOptions o;
  o.batch_size = 1;
  o.cancel = laya::CancelToken::Make();
  rig.st->on_run = [c = o.cancel] { c.Cancel(); };  // the first run cancels the call
  auto p = rig.agent->PredictBatch(States(5), Mix(), o);
  ASSERT_FALSE(p.ok());
  EXPECT_TRUE(ag::IsCancelled(p.error())) << p.error();
  EXPECT_EQ(rig.st->runs.size(), 0u);  // the stub sees the flag and fails that run
}

TEST(Agent, CloseIsIdempotentAndRefusesLaterCalls) {
  LAYA_RIG(rig, "english");
  EXPECT_EQ(rig.agent->Close(), "");
  EXPECT_EQ(rig.agent->Close(), "");
  EXPECT_EQ(rig.st->closes, 1);
  auto p = rig.agent->Predict(Value::String("x"), Mix());
  ASSERT_FALSE(p.ok());
  EXPECT_TRUE(ag::IsClosed(p.error()));
}

TEST(Agent, DestroyingClosesTheSession) {
  LAYA_RIG(rig, "english");
  auto st = rig.st;
  rig.agent.reset();
  EXPECT_EQ(st->closes, 1);
  EXPECT_TRUE(st->destroyed);
}

TEST(Agent, ConcurrentPredictsAgreeWithASerialOne) {
  LAYA_RIG(rig, "multilingual");
  const auto qs = Mix();
  const auto states = States(12);
  const auto want = Jsons(rig.agent->PredictBatch(states, qs).value());
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; t++) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 10; i++) {
        ag::CallOptions o;
        o.batch_size = 1 + (t + i) % 5;
        o.sort_by_length = (t + i) % 2 == 0;
        auto p = rig.agent->PredictBatch(states, qs, o);
        if (!p.ok() || Jsons(p.value()) != want) bad++;
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(bad, 0);
}

TEST(Agent, CloseWaitsForCallsInFlightAndNoneStartsAfter) {
  LAYA_RIG(rig, "english");
  const auto qs = Mix();
  const auto states = States(6);
  std::atomic<int> ok{0}, closed{0}, other{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 6; t++) {
    threads.emplace_back([&] {
      for (int i = 0; i < 20; i++) {
        ag::CallOptions o;
        o.batch_size = 1;
        auto p = rig.agent->PredictBatch(states, qs, o);
        if (p.ok()) ok++;
        else if (ag::IsClosed(p.error())) closed++;
        else other++;  // a Run must never meet a closed session
      }
    });
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  rig.agent->Close();
  for (auto& th : threads) th.join();
  EXPECT_EQ(other, 0);
  EXPECT_EQ(rig.st->closes, 1);
  EXPECT_GT(ok + closed, 0);
}

// ---- review round 1 ----

class ThrowingSession : public in::Session {
 public:
  ThrowingSession(bool run, bool close, int* closes) : run_(run), close_(close), closes_(closes) {}
  laya::Result<in::Output> Run(const in::Batch&, const laya::CancelToken&) override {
    if (run_) throw std::runtime_error("run blew up");
    return laya::Result<in::Output>::Fail("unreachable");
  }
  std::string Close() override {
    (*closes_)++;
    if (close_) throw std::runtime_error("close blew up");
    return "";
  }

 private:
  bool run_, close_;
  int* closes_;
};

Rig OpenWith(const char* model, std::unique_ptr<in::Session> s) {
  Rig r;
  const laya::tokenizer::Tokenizer* tok = laya_test::LoadCheckpoint(model);
  if (tok == nullptr) return r;
  auto a = ag::Agent::OpenParts(kConfig, *tok, std::move(s));
  EXPECT_TRUE(a.ok());
  if (a.ok()) r.agent = std::move(a).value();
  return r;
}

TEST(Agent, ASessionThatThrowsIsAnErrorNotATerminate) {
  int closes = 0;
  LAYA_RIG_OPEN(rig, OpenWith("english", std::make_unique<ThrowingSession>(true, true, &closes)));
  auto p = rig.agent->Predict(Value::String("alpha"), Mix());
  ASSERT_FALSE(p.ok());
  EXPECT_NE(p.error().find("run blew up"), std::string::npos) << p.error();
  const std::string first = rig.agent->Close();
  EXPECT_NE(first.find("close blew up"), std::string::npos) << first;
  EXPECT_EQ(rig.agent->Close(), first);  // the second Close reports the same error
  EXPECT_EQ(closes, 1);
  rig.agent.reset();  // the destructor closes again without terminating
  EXPECT_EQ(closes, 1);
}

TEST(Agent, DestroyingWithAThrowingCloseDoesNotTerminate) {
  int closes = 0;
  {
    LAYA_RIG_OPEN(rig, OpenWith("english", std::make_unique<ThrowingSession>(false, true, &closes)));
  }
  EXPECT_EQ(closes, 1);
}

TEST(Agent, CloseReturnsTheSessionErrorToEveryCaller) {
  LAYA_RIG(rig, "english");
  rig.st->close_error = "disk gone";
  std::string a, b;
  std::thread t1([&] { a = rig.agent->Close(); }), t2([&] { b = rig.agent->Close(); });
  t1.join();
  t2.join();
  EXPECT_EQ(a, "disk gone");
  EXPECT_EQ(b, "disk gone");
  EXPECT_EQ(rig.st->closes, 1);
}

TEST(Agent, NoQuestionsUsageIsTheBareTwoCounters) {
  LAYA_RIG(rig, "english");
  auto p = rig.agent->Predict(Value::String("x"), sc::Questions());
  ASSERT_TRUE(p.ok());
  // The JSON the Python ONNX agent returns for an empty question set.
  EXPECT_EQ(p.value().ToJson().value(), R"({"model":"laya-rl-agent-onnx","answers":{},"usage":{"input_tokens":0,"output_tokens":0}})");
}

TEST(Agent, CancelledCallsReportCancelledEvenWithNothingToDo) {
  LAYA_RIG(rig, "english");
  ag::CallOptions o;
  o.cancel.Cancel();  // a default token fires too
  auto none = rig.agent->PredictBatch({}, Mix(), o);
  ASSERT_FALSE(none.ok());
  EXPECT_TRUE(ag::IsCancelled(none.error()));
  auto noq = rig.agent->PredictBatch(States(2), sc::Questions(), o);
  ASSERT_FALSE(noq.ok());
  EXPECT_TRUE(ag::IsCancelled(noq.error()));
}

TEST(Agent, ASessionThatIgnoresCancelIsStoppedBeforeTheNextRun) {
  LAYA_RIG(rig, "english");
  rig.st->ignore_cancel = true;
  ag::CallOptions o;
  o.batch_size = 2;
  o.sort_by_length = true;
  o.cancel = laya::CancelToken::Make();
  rig.st->on_run = [c = o.cancel] { c.Cancel(); };
  auto p = rig.agent->PredictBatch(States(6), Mix(), o);
  ASSERT_FALSE(p.ok());
  EXPECT_TRUE(ag::IsCancelled(p.error())) << p.error();
  EXPECT_EQ(rig.st->runs.size(), 1u);
}

// The last run of window one cancels; window two opens with a state that cannot be encoded, and the
// call must report the cancel, not that state's error.
TEST(Agent, CancelIsCheckedBeforeEncodingTheNextWindow) {
  LAYA_RIG(rig, "english");
  auto states = States(8);
  Value deep = Value::String("x");
  for (int i = 0; i < laya::pyvalue::kMaxDepth + 8; i++) deep = Value::Array({std::move(deep)});  // too deep to serialize
  states.push_back(std::move(deep));
  ag::CallOptions o;
  o.batch_size = 1;
  o.sort_by_length = true;  // windows of eight states
  o.cancel = laya::CancelToken::Make();
  rig.st->ignore_cancel = true;
  ASSERT_FALSE(rig.agent->Predict(states.back(), Mix()).ok());  // it does fail to encode
  rig.st->runs.clear();
  int runs = 0;
  rig.st->on_run = [&, c = o.cancel] {  // the last run of window one cancels
    if (++runs == 8) c.Cancel();
  };
  auto p = rig.agent->PredictBatch(states, Mix(), o);
  ASSERT_FALSE(p.ok());
  EXPECT_TRUE(ag::IsCancelled(p.error())) << p.error();
}

TEST(Agent, ClosedAgentRefusesEmptyAndQuestionlessCalls) {
  LAYA_RIG(rig, "english");
  rig.agent->Close();
  auto a = rig.agent->PredictBatch({}, Mix());
  ASSERT_FALSE(a.ok());
  EXPECT_TRUE(ag::IsClosed(a.error()));
  auto b = rig.agent->PredictBatch(States(2), sc::Questions());
  ASSERT_FALSE(b.ok());
  EXPECT_TRUE(ag::IsClosed(b.error()));
}

TEST(Agent, ASessionErrorSurvivesAFiredToken) {
  LAYA_RIG(rig, "english");
  rig.st->ignore_cancel = true;
  rig.st->fail = "disk on fire";
  ag::CallOptions o;
  o.cancel = laya::CancelToken::Make();
  rig.st->on_run = [c = o.cancel] { c.Cancel(); };
  auto p = rig.agent->Predict(Value::String("alpha"), Mix(), o);
  ASSERT_FALSE(p.ok());
  EXPECT_FALSE(ag::IsCancelled(p.error()));
  EXPECT_NE(p.error().find("disk on fire"), std::string::npos) << p.error();
}

// Close does not wait for a call to finish its runs: it waits for the run in flight and the call
// then fails with kClosed at the next one.
TEST(Agent, CloseCutsAnInFlightCallShortAtItsNextRun) {
  LAYA_RIG(rig, "english");
  std::mutex mu;
  std::condition_variable cv;
  bool in_run = false, release = false;
  rig.st->on_run = [&] {
    std::unique_lock<std::mutex> l(mu);
    if (in_run) return;  // only the first run is gated
    in_run = true;
    cv.notify_all();
    cv.wait(l, [&] { return release; });
  };
  ag::CallOptions o;
  o.batch_size = 1;
  laya::Result<std::vector<sc::Prediction>> got = laya::Result<std::vector<sc::Prediction>>::Fail("unset");
  std::thread call([&] { got = rig.agent->PredictBatch(States(4), Mix(), o); });
  {
    std::unique_lock<std::mutex> l(mu);
    cv.wait(l, [&] { return in_run; });
  }
  std::thread closer([&] { rig.agent->Close(); });
  while (rig.agent->PredictBatch({}, Mix()).ok()) std::this_thread::yield();  // until Close has begun
  {
    std::lock_guard<std::mutex> l(mu);
    release = true;
  }
  cv.notify_all();
  call.join();
  closer.join();
  ASSERT_FALSE(got.ok());
  EXPECT_TRUE(ag::IsClosed(got.error())) << got.error();
  EXPECT_EQ(rig.st->runs.size(), 1u);
  EXPECT_EQ(rig.st->closes, 1);
}

// ---- Open ----

TEST(AgentOpen, ReadsTheCheckpointDirectory) {
  for (const char* model : kModels) {
    if (Root().empty()) {
      if (laya_test::RequireGoldens()) ADD_FAILURE() << "LAYA_CHECKPOINTS unset";
      GTEST_SKIP();
    }
    auto s = std::make_unique<StubSession>();
    auto st = s->state();
    auto a = ag::Agent::Open(Root() + "/" + model, std::move(s));
    ASSERT_TRUE(a.ok()) << a.error();
    EXPECT_EQ(a.value()->config().option_layout, "sequential");
    EXPECT_GT(a.value()->config().max_len, a.value()->config().head_max_len);
    EXPECT_TRUE(a.value()->Predict(Value::String("alpha bravo"), Mix()).ok());
  }
}

TEST(AgentOpen, FailureDestroysTheSession) {
  auto s = std::make_unique<StubSession>();
  auto st = s->state();
  auto a = ag::Agent::Open("/nonexistent/checkpoint", std::move(s));
  EXPECT_FALSE(a.ok());
  EXPECT_TRUE(st->destroyed);
  EXPECT_EQ(st->closes, 1);  // a failed Open still closes the session
  EXPECT_FALSE(ag::Agent::Open("/x", nullptr).ok());
  auto s2 = std::make_unique<StubSession>();
  auto st2 = s2->state();
  const laya::tokenizer::Tokenizer* tok = laya_test::LoadCheckpoint("english");
  if (tok != nullptr) {
    EXPECT_FALSE(ag::Agent::OpenParts("[]", *tok, std::move(s2)).ok());
    EXPECT_EQ(st2->closes, 1);
  }
}

TEST(AgentOpen, RefusesABadConfig) {
  for (const char* model : kModels) {
    const laya::tokenizer::Tokenizer* tok = laya_test::LoadCheckpoint(model);
    if (tok == nullptr) GTEST_SKIP() << "LAYA_CHECKPOINTS unset (local run)";
    for (const char* cfg : {"[]", "{", R"({"option_layout":"parallel"})", R"({"option_layout":1})", R"({"max_len":0})",
                            R"({"max_len":-5})", R"({"max_len":true})", R"({"max_len":512.0})", R"({"max_len":"512"})",
                            R"({"head_max_len":99999999999999999999})", R"({"max_len":2000000})",
                            R"({"temperature":[1,2]})", R"({"temperature":"x"})", R"({"temperature_by_options":[1]})"}) {
      auto a = ag::Agent::OpenParts(cfg, *tok, std::make_unique<StubSession>());
      EXPECT_FALSE(a.ok()) << cfg;
    }
    auto ok = ag::Agent::OpenParts("{}", *tok, std::make_unique<StubSession>());
    ASSERT_TRUE(ok.ok());
    EXPECT_EQ(ok.value()->config().max_len, 512);
    EXPECT_EQ(ok.value()->config().head_max_len, 192);
    break;
  }
}

TEST(AgentOpen, WarnsWhenTemperaturesAreClamped) {
  LAYA_RIG_CFG(rig, "english", R"({"temperature":[0.1,"9",1.5],"temperature_by_options":{"choice:2":0.3,"noul:2":2}})");
  ASSERT_EQ(rig.agent->config().warnings.size(), 1u);
  const std::string& w = rig.agent->config().warnings[0];
  EXPECT_NE(w.find("choice:2=0.3 -> 0.5"), std::string::npos) << w;
  EXPECT_NE(w.find("temperature[0]=0.1 -> 0.5"), std::string::npos) << w;
  EXPECT_NE(w.find("temperature[1]='9' -> 5"), std::string::npos) << w;
  EXPECT_EQ(w.find("noul:2"), std::string::npos);
  LAYA_RIG_CFG(fine, "english", R"({"temperature":[1,2,3]})");
  EXPECT_TRUE(fine.agent->config().warnings.empty());
}

}  // namespace
