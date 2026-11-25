// Tests of the ONNX backend. The pure checks run anywhere; the tests that load a graph need
// LAYA_ONNX_GRAPH (multilingual), LAYA_ONNX_GRAPH_ENGLISH and LAYA_CHECKPOINTS, and with
// LAYA_REQUIRE_GOLDENS=1 (the onnx lane) a missing one FAILS instead of skipping.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

#include "golden.hpp"
#include "laya/onnx.hpp"
#include "laya/onnx_graph.hpp"

namespace {

namespace fs = std::filesystem;
using laya::Result;
using laya::infer::Batch;
using laya::onnx::OutTensor;
using laya::onnx::TensorInfo;
using laya::pyvalue::Value;
using Kind = Value::Kind;

// ---------------------------------------------------------------- graph checks (no runtime needed)

std::vector<TensorInfo> GoodInputs() {
  return {{"input_ids", "int64", {-1, -1}},
          {"attention_mask", "int64", {-1, -1}},
          {"marker_pos", "int64", {-1, -1}},
          {"marker_mask", "bool", {-1, -1}},
          {"qtype", "int64", {-1}}};
}
std::vector<TensorInfo> GoodOutputs() { return {{"logits", "float32", {-1, -1}}, {"act_logits", "float32", {-1, 2}}}; }

TEST(OnnxGraph, TheExportedShapeIsAccepted) { EXPECT_EQ(laya::onnx::CheckGraph(GoodInputs(), GoodOutputs()), ""); }

TEST(OnnxGraph, InputOrderInTheGraphDoesNotMatter) {
  auto in = GoodInputs();
  std::swap(in[0], in[4]);
  std::swap(in[1], in[3]);
  EXPECT_EQ(laya::onnx::CheckGraph(in, GoodOutputs()), "");
}

TEST(OnnxGraph, EveryInputIsHeldToItsName) {
  for (size_t i = 0; i < 5; i++) {
    auto in = GoodInputs();
    in[i].name = "renamed";
    const std::string e = laya::onnx::CheckGraph(in, GoodOutputs());
    EXPECT_NE(e.find("missing input"), std::string::npos) << i << ": " << e;
  }
}

TEST(OnnxGraph, EveryInputIsHeldToItsDtype) {
  for (size_t i = 0; i < 5; i++) {
    auto in = GoodInputs();
    in[i].dtype = in[i].dtype == "int64" ? "float32" : "int64";
    const std::string e = laya::onnx::CheckGraph(in, GoodOutputs());
    EXPECT_NE(e.find(in[i].name), std::string::npos) << e;
    EXPECT_NE(e.find("want"), std::string::npos) << e;
  }
}

TEST(OnnxGraph, TheMaskMustBeBoolNotInt64) {
  auto in = GoodInputs();
  in[3].dtype = "int64";
  EXPECT_NE(laya::onnx::CheckGraph(in, GoodOutputs()).find("is int64, want bool"), std::string::npos);
}

TEST(OnnxGraph, RankIsChecked) {
  auto in = GoodInputs();
  in[4].dims = {-1, -1};
  EXPECT_NE(laya::onnx::CheckGraph(in, GoodOutputs()).find("rank 2, want 1"), std::string::npos);
  in = GoodInputs();
  in[0].dims = {-1};
  EXPECT_NE(laya::onnx::CheckGraph(in, GoodOutputs()).find("rank 1, want 2"), std::string::npos);
}

TEST(OnnxGraph, AFixedBatchDimensionIsRefused) {
  auto out = GoodOutputs();
  out[1].dims = {1, 2};  // the layout of onnx/full-multilingual: act_logits traced at batch 1
  EXPECT_NE(laya::onnx::CheckGraph(GoodInputs(), out).find("fixed batch dimension 1"), std::string::npos);
  auto in = GoodInputs();
  in[0].dims = {4, -1};
  EXPECT_NE(laya::onnx::CheckGraph(in, GoodOutputs()).find("fixed batch dimension 4"), std::string::npos);
}

TEST(OnnxGraph, ActLogitsHasExactlyTwoColumns) {
  auto out = GoodOutputs();
  out[1].dims = {-1, 3};
  EXPECT_NE(laya::onnx::CheckGraph(GoodInputs(), out).find("fixed size 3"), std::string::npos);
  out[1].dims = {-1, -1};
  EXPECT_EQ(laya::onnx::CheckGraph(GoodInputs(), out), "");
}

TEST(OnnxGraph, ExtrasAndDuplicatesAreRefused) {
  auto in = GoodInputs();
  in.push_back({"extra", "int64", {-1}});
  EXPECT_NE(laya::onnx::CheckGraph(in, GoodOutputs()).find("unexpected input \"extra\""), std::string::npos);
  auto out = GoodOutputs();
  out.push_back({"logits", "float32", {-1, -1}});
  EXPECT_NE(laya::onnx::CheckGraph(GoodInputs(), out).find("duplicate output"), std::string::npos);
  EXPECT_NE(laya::onnx::CheckGraph({}, GoodOutputs()).find("missing input"), std::string::npos);
}

TEST(OnnxGraph, AnOutputDtypeIsChecked) {
  auto out = GoodOutputs();
  out[0].dtype = "int64";
  EXPECT_NE(laya::onnx::CheckGraph(GoodInputs(), out).find("output \"logits\" is int64"), std::string::npos);
}

// ---------------------------------------------------------------- paths

class TempDir {
 public:
  TempDir() {
    static std::atomic<int> n{0};
    path_ = fs::temp_directory_path() / ("laya_onnx_test_" + std::to_string(::getpid()) + "_" + std::to_string(n++));
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::string File(const std::string& name, const std::string& body = "x") const {
    std::ofstream(path_ / name, std::ios::binary) << body;
    return (path_ / name).string();
  }
  std::string str() const { return path_.string(); }

 private:
  fs::path path_;
};

TEST(OnnxPath, MissingAndDirectoryAndEmptyAreErrors) {
  TempDir d;
  EXPECT_NE(laya::onnx::CheckPath(d.str() + "/nope.onnx").find("no such file"), std::string::npos);
  EXPECT_NE(laya::onnx::CheckPath(d.str()).find("is a directory"), std::string::npos);
  EXPECT_NE(laya::onnx::CheckPath("").find("empty"), std::string::npos);
  EXPECT_EQ(laya::onnx::CheckPath(d.File("laya.onnx")), "");
}

TEST(OnnxPath, TheSplitLayoutIsRefusedWhicheverPathIsGiven) {
  TempDir d;
  const std::string enc = d.File("encoder.onnx");
  const std::string head = d.File("head.onnx");
  for (const std::string& p : {enc, head, d.str(), d.str() + "/laya.onnx"}) {
    const std::string e = laya::onnx::CheckPath(p);
    EXPECT_NE(e.find("split encoder.onnx + head.onnx layout is not supported"), std::string::npos) << p << ": " << e;
    EXPECT_NE(e.find("export_onnx.py"), std::string::npos) << e;
  }
}

TEST(OnnxPath, OneHalfAloneIsNotTheSplitLayout) {
  TempDir d;
  const std::string enc = d.File("encoder.onnx");
  EXPECT_EQ(laya::onnx::CheckPath(enc), "");
}

// ---------------------------------------------------------------- collecting outputs

Batch Tiny(int64_t rows, int64_t markers) {
  Batch b;
  b.rows = rows;
  b.length = 3;
  b.markers = markers;
  b.input_ids.assign(size_t(rows * 3), 1);
  b.attention_mask.assign(size_t(rows * 3), 1);
  b.marker_pos.assign(size_t(rows * markers), 1);
  b.marker_mask.assign(size_t(rows * markers), 1);
  b.qtype.assign(size_t(rows), 0);
  return b;
}

OutTensor Tensor(const std::vector<float>& v, std::vector<int64_t> shape) {
  OutTensor t;
  t.shape = std::move(shape);
  t.data = v.data();
  t.count = int64_t(v.size());
  return t;
}

TEST(OnnxCollect, ACorrectPairIsCopied) {
  const Batch b = Tiny(2, 3);
  const std::vector<float> l = {1, 2, 3, 4, 5, 6}, a = {7, 8, 9, 10};
  auto r = laya::onnx::CollectOutput(b, Tensor(l, {2, 3}), Tensor(a, {2, 2}));
  ASSERT_TRUE(r.ok()) << r.error();
  EXPECT_EQ(r.value().logits, l);
  EXPECT_EQ(r.value().act_logits, a);
}

TEST(OnnxCollect, ASlicedOutputIsTheOutputOfTheRightRows) {
  const Batch b = Tiny(2, 3);
  const std::vector<float> l = {1, 2, 3, 4, 5, 6}, a = {7, 8, 9, 10};
  auto r = laya::onnx::CollectOutput(b, Tensor(l, {2, 3}), Tensor(a, {2, 2}));
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().logits[3], 4.0f);  // row 1, column 0
  EXPECT_EQ(r.value().act_logits[2], 9.0f);
}

TEST(OnnxCollect, ANaNIsAnErrorNotAnAnswer) {
  const Batch b = Tiny(1, 3);
  std::vector<float> l = {1, std::numeric_limits<float>::quiet_NaN(), 3}, a = {1, 2};
  auto r = laya::onnx::CollectOutput(b, Tensor(l, {1, 3}), Tensor(a, {1, 2}));
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().find("NaN"), std::string::npos);
  l[1] = 0;
  a[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(laya::onnx::CollectOutput(b, Tensor(l, {1, 3}), Tensor(a, {1, 2})).ok());
}

TEST(OnnxCollect, AnInfinityIsAnErrorNotAnAnswer) {
  const Batch b = Tiny(1, 3);
  const float inf = std::numeric_limits<float>::infinity();
  for (float bad : {inf, -inf}) {
    std::vector<float> l = {1, bad, 3}, a = {1, 2};
    auto r = laya::onnx::CollectOutput(b, Tensor(l, {1, 3}), Tensor(a, {1, 2}));
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().find("Inf"), std::string::npos);
    l[1] = 0;
    a[1] = bad;
    EXPECT_FALSE(laya::onnx::CollectOutput(b, Tensor(l, {1, 3}), Tensor(a, {1, 2})).ok());
  }
}

TEST(OnnxCollect, WrongShapesAreErrors) {
  const Batch b = Tiny(2, 3);
  const std::vector<float> l(6, 1.f), a(4, 1.f), small(2, 1.f);
  EXPECT_FALSE(laya::onnx::CollectOutput(b, Tensor(l, {3, 2}), Tensor(a, {2, 2})).ok());
  EXPECT_FALSE(laya::onnx::CollectOutput(b, Tensor(l, {2, 3}), Tensor(a, {4, 1})).ok());
  EXPECT_FALSE(laya::onnx::CollectOutput(b, Tensor(l, {6}), Tensor(a, {2, 2})).ok());
  EXPECT_FALSE(laya::onnx::CollectOutput(b, Tensor(l, {2, 3, 1}), Tensor(a, {2, 2})).ok());
  EXPECT_FALSE(laya::onnx::CollectOutput(b, Tensor(small, {2, 3}), Tensor(a, {2, 2})).ok());  // shape says 6, holds 2
}

// ---------------------------------------------------------------- options

TEST(OnnxOpen, ThreadCountsAreRange_Checked) {
  laya::onnx::Options o;
  o.intra_op_threads = -1;
  EXPECT_FALSE(laya::onnx::Open("/nonexistent/laya.onnx", o).ok());
  o.intra_op_threads = laya::onnx::kMaxThreads + 1;
  const auto r = laya::onnx::Open("/nonexistent/laya.onnx", o);
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().find("intra_op_threads"), std::string::npos);
}

TEST(OnnxOpen, AMissingGraphIsAnError) {
  const auto r = laya::onnx::Open("/nonexistent/dir/laya.onnx");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().find("no such file"), std::string::npos);
}

TEST(OnnxOpen, AGarbageFileIsAnErrorNotAnException) {
  TempDir d;
  const auto r = laya::onnx::Open(d.File("laya.onnx", "this is not a protobuf \x01\x02\xff"));
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().find("cannot open"), std::string::npos) << r.error();
}

TEST(OnnxOpen, AnEmptyFileIsAnError) {
  TempDir d;
  EXPECT_FALSE(laya::onnx::Open(d.File("laya.onnx", "")).ok());
}

// ONNX Runtime refuses external data that leaves the graph's directory (`..` or an absolute path);
// this pins that, with a real file sitting where the graph points.
#include "onnx_hostile_graphs.inc"
TEST(OnnxOpen, ExternalDataOutsideTheGraphDirectoryIsRefused) {
  TempDir d;
  fs::create_directories(d.str() + "/sub");
  {
    std::ofstream(d.str() + "/secret.bin", std::ios::binary) << std::string(8, '\0');
  }
  const struct {
    const char* name;
    const char* bytes;
    size_t size;
  } cases[] = {{"dotdot", kDotdotGraph, kDotdotGraphSize}, {"absolute", kAbsGraph, kAbsGraphSize}};
  for (const auto& c : cases) {
    const std::string path = d.str() + "/sub/" + c.name + ".onnx";
    std::ofstream(path, std::ios::binary).write(c.bytes, std::streamsize(c.size));
    const auto r = laya::onnx::Open(path);
    ASSERT_FALSE(r.ok()) << c.name << " opened";
    EXPECT_NE(r.error().find("cannot open"), std::string::npos) << r.error();
  }
}

TEST(OnnxOpen, TheSplitLayoutIsRefusedByOpen) {
  TempDir d;
  d.File("encoder.onnx");
  d.File("head.onnx");
  const auto r = laya::onnx::Open(d.str() + "/encoder.onnx");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().find("split encoder.onnx + head.onnx layout"), std::string::npos) << r.error();
}

TEST(OnnxRuntime, TheLoadedLibraryIsThePinnedVersion) { EXPECT_EQ(laya::onnx::RuntimeVersion(), LAYA_ORT_VERSION); }

// ---------------------------------------------------------------- real graphs

struct Rig {
  const char* checkpoint;    // directory name under LAYA_CHECKPOINTS
  const char* graph_env;     // names the exported laya.onnx
  const char* golden;        // predict golden recorded from that graph
};
const Rig kRigs[] = {{"multilingual", "LAYA_ONNX_GRAPH", "predict.json"},
                     {"english", "LAYA_ONNX_GRAPH_ENGLISH", "predict-english.json"}};

// "" and a recorded failure (or skip) when the rig is not there.
std::string Need(const char* var) {
  const char* v = std::getenv(var);
  if (v != nullptr && *v != 0) return v;
  if (laya_test::RequireGoldens()) ADD_FAILURE() << var << " is unset but LAYA_REQUIRE_GOLDENS=1";
  return "";
}

#define NEED_RIG(rig, graph, dir)                                                                   \
  const std::string graph = Need(rig.graph_env);                                                    \
  const std::string root = Need("LAYA_CHECKPOINTS");                                                \
  if (graph.empty() || root.empty()) {                                                              \
    if (!HasFailure()) GTEST_SKIP() << "no ONNX rig (local run)";                                   \
    return;                                                                                         \
  }                                                                                                 \
  const std::string dir = root + "/" + rig.checkpoint

std::mutex g_agents_mu;
std::map<std::string, std::unique_ptr<laya::agent::Agent>> g_agents;

// Closes the shared agents after the last test, so Close on a live agent is checked once per run.
// Holding them open at exit is also fine (AStaticSessionAndAgentExitCleanly).
class CloseAgents : public ::testing::Environment {
 public:
  void TearDown() override {
    std::lock_guard<std::mutex> lock(g_agents_mu);
    for (auto& kv : g_agents) EXPECT_EQ(kv.second->Close(), "");
    g_agents.clear();
  }
};
const ::testing::Environment* const g_close_agents = ::testing::AddGlobalTestEnvironment(new CloseAgents);

// One agent per checkpoint for the whole process; the tests only read it.
laya::agent::Agent* Shared(const Rig& rig, const std::string& graph, const std::string& dir) {
  auto& agents = g_agents;
  std::lock_guard<std::mutex> lock(g_agents_mu);
  auto it = agents.find(rig.checkpoint);
  if (it != agents.end()) return it->second.get();
  auto a = laya::agent::OpenOnnx(dir, graph);
  if (!a.ok()) {
    ADD_FAILURE() << "open " << rig.checkpoint << ": " << a.error();
    return nullptr;
  }
  return (agents[rig.checkpoint] = std::move(a).value()).get();
}

constexpr double kTolerance = 2e-4;       // probabilities, confidences, act_probability, noul
constexpr double kScoreTolerance = 8e-4;  // score, the expectation over the probabilities
constexpr double kDecisiveGap = 1e-3;     // the chosen label is compared when the top two differ by more
constexpr double kMaxDiffering = 0.05;    // of all compared numbers may differ at all

struct Tally {
  int numbers = 0, differing = 0, labels = 0, ties = 0, label_diffs = 0, failures = 0;
  double worst = 0;
};

void Compare(const std::string& path, const std::string& key, const Value& got, const Value& want, Tally& t) {
  if (want.kind() == Kind::Object) {
    if (got.kind() != Kind::Object || got.size() != want.size()) {
      t.failures++;
      ADD_FAILURE() << path << ": object shape differs";
      return;
    }
    for (size_t i = 0; i < want.size(); i++) {
      const Value* g = got.Find(want.key(i));
      if (g == nullptr || got.key(i) != want.key(i)) {
        t.failures++;
        ADD_FAILURE() << path << ": key " << want.key(i) << " missing or out of order";
        continue;
      }
      Compare(path + "." + want.key(i), want.key(i), *g, want.value(i), t);
    }
  } else if (want.kind() == Kind::Array) {
    if (got.kind() != Kind::Array || got.size() != want.items().size()) {
      t.failures++;
      ADD_FAILURE() << path << ": array shape differs";
      return;
    }
    for (size_t i = 0; i < want.items().size(); i++) Compare(path + "[]", key, got.items()[i], want.items()[i], t);
  } else if (want.kind() == Kind::Int || want.kind() == Kind::Float) {
    if (got.kind() != Kind::Int && got.kind() != Kind::Float) {
      t.failures++;
      ADD_FAILURE() << path << ": not a number";
      return;
    }
    const double g = laya_test::Num(got), w = laya_test::Num(want), d = std::fabs(g - w);
    t.numbers++;
    if (d > 0) t.differing++;
    t.worst = std::max(t.worst, d);
    if (d > (key == "score" ? kScoreTolerance : kTolerance)) {
      t.failures++;
      ADD_FAILURE() << path << ": got " << g << " want " << w << " (off by " << d << ")";
    }
  } else if (got.kind() != want.kind() || laya::pyvalue::Repr(got) != laya::pyvalue::Repr(want)) {
    t.failures++;
    ADD_FAILURE() << path << ": " << laya::pyvalue::Repr(got) << " want " << laya::pyvalue::Repr(want);
  }
}

Value Without(const Value& obj, const std::string& drop) {
  Value out = Value::Object();
  for (size_t i = 0; i < obj.size(); i++) {
    if (obj.key(i) != drop) out.Set(obj.key(i), obj.value(i));
  }
  return out;
}

void CompareAnswer(const std::string& path, const Value& got, const Value& want, Tally& t) {
  const Value* type = want.Find("type");
  if (type != nullptr && type->as_string() == "choice") {
    std::vector<double> ps;
    for (const Value& p : want.Find("probabilities")->items()) ps.push_back(laya_test::Num(p));
    std::sort(ps.rbegin(), ps.rend());
    if (ps.size() > 1 && ps[0] - ps[1] <= kDecisiveGap) {
      t.ties++;
      Compare(path, "", Without(got, "choice"), Without(want, "choice"), t);
      return;
    }
    t.labels++;
    const Value* g = got.Find("choice");
    if (g == nullptr || laya::pyvalue::Repr(*g) != laya::pyvalue::Repr(*want.Find("choice"))) t.label_diffs++;
  }
  Compare(path, "", got, want, t);
}

void ComparePrediction(const std::string& path, const laya::schema::Prediction& p, const Value& want, Tally& t) {
  auto json = p.ToJson();
  ASSERT_TRUE(json.ok()) << json.error();
  auto parsed = laya::pyvalue::Parse(json.value());
  ASSERT_TRUE(parsed.ok()) << parsed.error();
  const Value& got = parsed.value();
  Compare(path + ".model", "model", *got.Find("model"), *want.Find("model"), t);
  Compare(path + ".usage", "usage", *got.Find("usage"), *want.Find("usage"), t);
  const Value& ga = *got.Find("answers");
  const Value& wa = *want.Find("answers");
  ASSERT_EQ(ga.size(), wa.size()) << path;
  for (size_t i = 0; i < wa.size(); i++) {
    ASSERT_EQ(ga.key(i), wa.key(i)) << path;
    CompareAnswer(path + "." + wa.key(i), ga.value(i), wa.value(i), t);
  }
}

laya::schema::Questions QuestionsOf(const Value& v) {
  auto text = laya::pyvalue::Dumps(v);
  EXPECT_TRUE(text.ok());
  auto qs = laya::schema::Questions::Parse(text.value());
  EXPECT_TRUE(qs.ok()) << qs.error();
  return qs.ok() ? qs.value() : laya::schema::Questions::Of({}).value();
}

void Finish(const char* name, const Tally& t, int min_numbers) {
  std::printf("[ tally ] %s: %d numbers, %d differ (worst %.2e), %d decisive labels (%d differ), %d near-ties\n", name, t.numbers,
              t.differing, t.worst, t.labels, t.label_diffs, t.ties);
  EXPECT_GE(t.numbers, min_numbers) << "too few numbers were compared";
  EXPECT_LE(double(t.differing), kMaxDiffering * t.numbers) << "more of the numbers differ at all than Python's CPU noise explains";
  EXPECT_EQ(t.label_diffs, 0) << "decisive labels differ from Python";
}

class OnnxRig : public ::testing::TestWithParam<int> {};
INSTANTIATE_TEST_SUITE_P(Checkpoints, OnnxRig, ::testing::Values(0, 1), [](const auto& i) { return std::string(kRigs[i.param].checkpoint); });

TEST_P(OnnxRig, SinglePredictionsMatchPython) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  LAYA_GOLDEN_LIST(singles, rig.golden, "single");
  LAYA_EXPECT_ROWS(singles, 6);
  laya::agent::Agent* a = Shared(rig, graph, dir);
  ASSERT_NE(a, nullptr);
  Tally t;
  for (size_t i = 0; i < singles->size(); i++) {
    const Value& row = singles->value(i);
    const std::string name = singles->key(i);
    auto pred = a->Predict(*row.Find("state"), QuestionsOf(*row.Find("questions")));
    ASSERT_TRUE(pred.ok()) << name << ": " << pred.error();
    ComparePrediction(name, pred.value(), row, t);
  }
  Finish("single", t, 60);
}

TEST_P(OnnxRig, BatchPredictionsMatchPythonInInputOrder) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  LAYA_GOLDEN_LIST(batches, rig.golden, "batch");
  LAYA_EXPECT_ROWS(batches, 3);
  laya::agent::Agent* a = Shared(rig, graph, dir);
  ASSERT_NE(a, nullptr);
  Tally t;
  for (const Value& rec : batches->items()) {
    laya::agent::CallOptions opts;
    if (!rec.Find("batch_size")->is_null()) opts.batch_size = rec.Find("batch_size")->as_int();
    opts.sort_by_length = rec.Find("sort_by_length")->kind() == Kind::Bool && rec.Find("sort_by_length")->as_bool();
    const std::vector<Value>& states = rec.Find("states")->items();
    auto preds = a->PredictBatch(states, QuestionsOf(*rec.Find("questions")), opts);
    ASSERT_TRUE(preds.ok()) << preds.error();
    const std::vector<Value>& want = rec.Find("results")->items();
    ASSERT_EQ(preds.value().size(), want.size());
    ASSERT_EQ(want.size(), 9u);
    for (size_t i = 0; i < want.size(); i++) ComparePrediction("batch/state" + std::to_string(i), preds.value()[i], want[i], t);
  }
  Finish("batch", t, 240);
}

TEST_P(OnnxRig, BatchOrderFollowsTheInputWhateverTheSortAndBatchSize) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  LAYA_GOLDEN_LIST(batches, rig.golden, "batch");
  laya::agent::Agent* a = Shared(rig, graph, dir);
  ASSERT_NE(a, nullptr);
  const Value& rec = batches->items()[0];
  std::vector<Value> states = rec.Find("states")->items();
  const auto qs = QuestionsOf(*rec.Find("questions"));
  auto fwd = a->PredictBatch(states, qs);
  ASSERT_TRUE(fwd.ok()) << fwd.error();
  std::vector<Value> rev(states.rbegin(), states.rend());
  laya::agent::CallOptions o;
  o.batch_size = 2;
  o.sort_by_length = true;
  auto back = a->PredictBatch(rev, qs, o);
  ASSERT_TRUE(back.ok()) << back.error();
  ASSERT_EQ(fwd.value().size(), back.value().size());
  const size_t n = states.size();
  for (size_t i = 0; i < n; i++) {
    auto x = fwd.value()[i].ToJson(), y = back.value()[n - 1 - i].ToJson();
    ASSERT_TRUE(x.ok() && y.ok());
    auto px = laya::pyvalue::Parse(x.value()), py = laya::pyvalue::Parse(y.value());
    Tally t;
    Compare("state" + std::to_string(i), "", py.value(), px.value(), t);  // within the numeric tolerances
    EXPECT_EQ(t.failures, 0) << "state " << i << " changed with its position in the batch";
  }
}

TEST_P(OnnxRig, ConcurrentPredictAgreesWithASerialCall) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  LAYA_GOLDEN_LIST(singles, rig.golden, "single");
  laya::agent::Agent* a = Shared(rig, graph, dir);
  ASSERT_NE(a, nullptr);
  const size_t n = singles->size();
  std::vector<laya::schema::Questions> qs;
  std::vector<std::string> serial;
  for (size_t i = 0; i < n; i++) {
    qs.push_back(QuestionsOf(*singles->value(i).Find("questions")));
    auto p = a->Predict(*singles->value(i).Find("state"), qs.back());
    ASSERT_TRUE(p.ok()) << p.error();
    serial.push_back(p.value().ToJson().value());
  }
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  for (int th = 0; th < 8; th++) {
    threads.emplace_back([&, th] {
      for (int round = 0; round < 3; round++) {
        for (size_t k = 0; k < n; k++) {
          const size_t i = (k + size_t(th)) % n;
          auto p = a->Predict(*singles->value(i).Find("state"), qs[i]);
          if (!p.ok() || p.value().ToJson().value() != serial[i]) bad++;
        }
      }
    });
  }
  for (auto& t : threads) t.join();
  EXPECT_EQ(bad.load(), 0);
}

TEST_P(OnnxRig, ACancelledCallStopsBeforeTheRuntime) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  LAYA_GOLDEN_LIST(singles, rig.golden, "single");
  laya::agent::Agent* a = Shared(rig, graph, dir);
  ASSERT_NE(a, nullptr);
  laya::agent::CallOptions o;
  o.cancel.Cancel();
  auto p = a->Predict(*singles->value(0).Find("state"), QuestionsOf(*singles->value(0).Find("questions")), o);
  ASSERT_FALSE(p.ok());
  EXPECT_TRUE(laya::agent::IsCancelled(p.error())) << p.error();
}

TEST_P(OnnxRig, SessionRunIsGuardedAgainstHostileBatches) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  auto s = laya::onnx::Open(graph);
  ASSERT_TRUE(s.ok()) << s.error();
  laya::infer::Session& session = *s.value();
  const laya::CancelToken live;

  auto ok = session.Run(Tiny(2, 3), live);
  ASSERT_TRUE(ok.ok()) << ok.error();
  EXPECT_EQ(ok.value().logits.size(), 6u);
  EXPECT_EQ(ok.value().act_logits.size(), 4u);

  Batch wrong_rank = Tiny(2, 3);
  wrong_rank.input_ids.pop_back();  // a flattened slice that no longer matches rows*length
  EXPECT_FALSE(session.Run(wrong_rank, live).ok());

  Batch huge = Tiny(1, 3);
  huge.rows = int64_t(1) << 40;  // far past the cap: refused before any allocation
  EXPECT_FALSE(session.Run(huge, live).ok());
  huge = Tiny(1, 3);
  huge.length = int64_t(1) << 40;
  EXPECT_FALSE(session.Run(huge, live).ok());
  Batch neg = Tiny(1, 3);
  neg.length = -1;
  EXPECT_FALSE(session.Run(neg, live).ok());

  Batch one_marker = Tiny(1, 1);  // the exported head takes topk(2)
  EXPECT_FALSE(session.Run(one_marker, live).ok());
  Batch oob = Tiny(1, 3);
  oob.marker_pos[0] = 99;
  EXPECT_FALSE(session.Run(oob, live).ok());
  Batch mask2 = Tiny(1, 3);
  mask2.marker_mask[1] = 2;  // not a bool
  EXPECT_FALSE(session.Run(mask2, live).ok());
  Batch bad_q = Tiny(1, 3);
  bad_q.qtype[0] = 7;
  EXPECT_FALSE(session.Run(bad_q, live).ok());
  Batch empty;
  EXPECT_FALSE(session.Run(empty, live).ok());

  // Passes every check of ours and fails inside ONNX Runtime (a token id past the vocabulary): the
  // failure must come back as a Result, not an exception.
  Batch past_vocab = Tiny(1, 3);
  past_vocab.input_ids[0] = 1000000000;
  const auto pv = session.Run(past_vocab, live);
  ASSERT_FALSE(pv.ok());
  EXPECT_EQ(pv.error().rfind("onnx: run:", 0), 0u) << pv.error();

  laya::CancelToken fired;
  fired.Cancel();
  auto c = session.Run(Tiny(1, 3), fired);
  ASSERT_FALSE(c.ok());
  EXPECT_TRUE(laya::infer::IsCancelled(c.error()));

  // The session still works after every refusal.
  EXPECT_TRUE(session.Run(Tiny(1, 3), live).ok());
}

TEST_P(OnnxRig, CloseIsIdempotentAndRunAfterCloseIsAnError) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  auto s = laya::onnx::Open(graph);
  ASSERT_TRUE(s.ok()) << s.error();
  EXPECT_EQ(s.value()->Close(), "");
  EXPECT_EQ(s.value()->Close(), "");
  auto r = s.value()->Run(Tiny(1, 3), laya::CancelToken());
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().find("closed"), std::string::npos) << r.error();
}

// A batch that keeps the runtime busy for seconds.
Batch Long(int64_t rows, int64_t length) {
  Batch b = Tiny(rows, 3);
  b.length = length;
  b.input_ids.assign(size_t(rows * length), 5);
  b.attention_mask.assign(size_t(rows * length), 1);
  return b;
}
using Clock = std::chrono::steady_clock;
double Since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
constexpr int64_t kLongRows = 16, kLongLength = 192;

// Seconds one uncancelled long run takes; the tests below only mean something if it is long.
double LongRunSeconds(laya::infer::Session& s) {
  const auto t = Clock::now();
  const auto r = s.Run(Long(kLongRows, kLongLength), laya::CancelToken());
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error());
  return Since(t);
}

TEST_P(OnnxRig, ACancelDuringARunStopsTheRun) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  auto s = laya::onnx::Open(graph);
  ASSERT_TRUE(s.ok()) << s.error();
  const double full = LongRunSeconds(*s.value());
  ASSERT_GT(full, 0.8) << "the long batch is too short to tell a mid-run cancel from a late one";
  laya::CancelToken token;
  std::thread canceller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    token.Cancel();
  });
  const auto t = Clock::now();
  const auto r = s.value()->Run(Long(kLongRows, kLongLength), token);
  const double took = Since(t);
  canceller.join();
  ASSERT_FALSE(r.ok());
  EXPECT_TRUE(laya::infer::IsCancelled(r.error())) << r.error();
  EXPECT_LT(took, full / 3) << "cancelled run took " << took << " s of " << full << " s";
  EXPECT_TRUE(s.value()->Run(Tiny(1, 3), laya::CancelToken()).ok());  // the session is still usable
}

TEST_P(OnnxRig, ACancelThatLandsAfterTheRunStillReportsCancelled) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  auto s = laya::onnx::Open(graph);
  ASSERT_TRUE(s.ok()) << s.error();
  // The token fires inside the post-run window: the run is tiny, so a watcher cannot beat it
  // reliably; either outcome is allowed only if it is cancelled or a complete correct result.
  for (int i = 0; i < 50; i++) {
    laya::CancelToken token;
    std::thread canceller([&] { token.Cancel(); });
    const auto r = s.value()->Run(Tiny(2, 3), token);
    canceller.join();
    if (!r.ok()) EXPECT_TRUE(laya::infer::IsCancelled(r.error())) << r.error();
    else EXPECT_EQ(r.value().logits.size(), 6u);
  }
}

TEST_P(OnnxRig, CloseInterruptsARunInFlight) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  auto s = laya::onnx::Open(graph);
  ASSERT_TRUE(s.ok()) << s.error();
  const double full = LongRunSeconds(*s.value());
  ASSERT_GT(full, 0.8);
  std::atomic<bool> failed{false};
  std::thread runner([&] { failed = !s.value()->Run(Long(kLongRows, kLongLength), laya::CancelToken()).ok(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const auto t = Clock::now();
  EXPECT_EQ(s.value()->Close(), "");
  const double took = Since(t);
  runner.join();
  EXPECT_TRUE(failed.load());
  EXPECT_LT(took, full / 3) << "Close took " << took << " s of a " << full << " s run";
}

TEST_P(OnnxRig, CloseReturnsPromptlyUnderSustainedRunTraffic) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  auto s = laya::onnx::Open(graph);
  ASSERT_TRUE(s.ok()) << s.error();
  const auto t0 = Clock::now();
  ASSERT_TRUE(s.value()->Run(Long(4, 128), laya::CancelToken()).ok());
  const double one = Since(t0);  // one such run alone
  ASSERT_GT(one, 0.15);
  std::atomic<bool> stop{false};
  std::atomic<int> ran{0}, refused{0}, other{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; i++) {
    threads.emplace_back([&] {
      while (!stop.load()) {
        const auto r = s.value()->Run(Long(4, 128), laya::CancelToken());
        if (r.ok()) ran++;
        else if (r.error().find("closed") != std::string::npos || r.error().find("run:") != std::string::npos) refused++;
        else other++;
      }
    });
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  const auto t = Clock::now();
  EXPECT_EQ(s.value()->Close(), "");
  const double took = Since(t);
  stop = true;
  for (auto& th : threads) th.join();
  EXPECT_GT(ran.load() + refused.load(), 0);  // every thread ended on a Result, none on a hang
  EXPECT_EQ(other.load(), 0);
  EXPECT_LT(took, one * 2) << "Close took " << took << " s with one run alone taking " << one << " s";
}

// A user program may hold a Session and an Agent in statics; it must exit cleanly, closed or not.
TEST_P(OnnxRig, AStaticSessionAndAgentExitCleanly) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  // The child gets no variable that turns the runtime's telemetry off, so it exits with the uploader
  // thread running unless Open opts out (the abort this test guards against).
  const std::set<std::string> drop = {"ORT_DISABLE_TELEMETRY", "ORT_RUNNING_UNIT_TESTS", "CI", "TF_BUILD",
                                      "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "TRAVIS", "JENKINS_URL",
                                      "CODEBUILD_BUILD_ID", "BUILDKITE", "TEAMCITY_VERSION", "APPVEYOR",
                                      "BITBUCKET_BUILD_NUMBER", "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI"};
  std::vector<const char*> envp;
  for (char** e = environ; *e != nullptr; e++) {
    const char* eq = std::strchr(*e, '=');
    if (eq == nullptr || drop.count(std::string(*e, eq - *e)) == 0) envp.push_back(*e);
  }
  envp.push_back(nullptr);
  for (const char* mode : {"", "close"}) {
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      const char* argv[] = {LAYA_STATIC_HOLDER, graph.c_str(), dir.c_str(), mode, nullptr};
      if (*mode == 0) argv[3] = nullptr;
      execve(LAYA_STATIC_HOLDER, const_cast<char* const*>(argv), const_cast<char* const*>(envp.data()));
      _exit(127);
    }
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status)) << "mode '" << mode << "' died by signal " << (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    EXPECT_EQ(WEXITSTATUS(status), 0) << "mode '" << mode << "'";
  }
}

TEST_P(OnnxRig, SessionsOpenAndCloseWhileOthersRun) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  auto first = laya::onnx::Open(graph);
  ASSERT_TRUE(first.ok()) << first.error();
  std::atomic<int> bad{0};
  std::thread worker([&] {
    for (int i = 0; i < 20; i++) {
      if (!first.value()->Run(Tiny(2, 3), laya::CancelToken()).ok()) bad++;
    }
  });
  for (int i = 0; i < 3; i++) {
    laya::onnx::Options o;
    o.intra_op_threads = 1;
    auto other = laya::onnx::Open(graph, o);  // a second session, dropped without Close
    if (!other.ok() || !other.value()->Run(Tiny(1, 3), laya::CancelToken()).ok()) bad++;
  }
  worker.join();
  EXPECT_EQ(bad.load(), 0);
}

TEST_P(OnnxRig, OpenOnnxWithAWrongCheckpointDirIsAnError) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  TempDir empty;
  const auto a = laya::agent::OpenOnnx(empty.str(), graph);
  EXPECT_FALSE(a.ok());
}

TEST_P(OnnxRig, TheGraphWithoutItsDataFileIsAnError) {
  const Rig& rig = kRigs[GetParam()];
  NEED_RIG(rig, graph, dir);
  TempDir d;
  fs::copy_file(graph, d.str() + "/laya.onnx");  // the .data file stays behind
  const auto r = laya::onnx::Open(d.str() + "/laya.onnx");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().find("cannot open"), std::string::npos) << r.error();
}

}  // namespace
