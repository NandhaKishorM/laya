#include "laya/decode.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include "golden.hpp"
#include "question_fixture.hpp"

namespace {

using laya::pyvalue::Value;
namespace dc = laya::decode;
namespace sc = laya::schema;
using Kind = Value::Kind;
using laya_test::GoldenQuestion;

constexpr const char* kModels[] = {"english", "multilingual"};

float F32(const Value& v) { return float(laya_test::Num(v)); }

std::vector<float> Floats(const Value& list) {
  std::vector<float> out;
  for (const Value& v : list.items()) out.push_back(F32(v));
  return out;
}

dc::Temperatures TempsOf(const Value& config) {
  auto t = dc::Temperatures::Of(config.Find("temperature"), config.Find("temperature_by_options"),
                                config.Find("lang_temperatures_raw"));
  EXPECT_TRUE(t.ok()) << (t.ok() ? "" : t.error());
  if (!t.ok()) std::abort();
  return std::move(t).value();
}

std::string Compact(const std::string& json) {  // normalize through the parser: one spelling
  auto v = laya::pyvalue::Parse(json);
  return v.ok() ? laya::pyvalue::Dumps(v.value()).value() : "unparsable: " + json;
}

dc::Temperatures Uniform(double t) {
  Value a = Value::Array({Value::Float(t), Value::Float(t), Value::Float(t)});
  return dc::Temperatures::Of(&a, nullptr).value();
}

// decode.json: Agent._decode_answers on logit rows aimed at every bucket, a tie, a uniform row,
// overflowing magnitudes, the option permutation and the language override. Exact, key order included.
TEST(DecodeGolden, DecodeJson) {
  for (const char* name : kModels) {
    LAYA_GOLDEN_LIST(entry, "decode.json", name);
    const Value* cases = entry->Find("cases");
    LAYA_EXPECT_ROWS(cases, 20);
    const dc::Temperatures temps = TempsOf(*entry->Find("config"));
    for (size_t i = 0; i < cases->size(); i++) {
      const Value& rec = cases->value(i);
      const std::string where = std::string(name) + "/" + cases->key(i);
      sc::Question q = GoldenQuestion(*rec.Find("question"), *rec.Find("option_order"));
      dc::Row row;
      row.logits = Floats(*rec.Find("logits"));
      row.k = int64_t(row.logits.size());
      for (int64_t w = int64_t(row.logits.size()); w < rec.Find("logits_block_width")->as_int(); w++) {
        row.logits.push_back(99);  // columns past k must not be read
      }
      row.act = Floats(*rec.Find("act"))[0];
      const Value* lang = rec.Find("lang");
      auto a = dc::Decode(q, row, lang->is_null() ? "" : lang->as_string(), &temps);
      ASSERT_TRUE(a.ok()) << where << ": " << a.error();
      EXPECT_EQ(Compact(a.value().ToJson().value()), laya::pyvalue::Dumps(*rec.Find("answer")).value()) << where;
    }
  }
}

// The checkpoints' own configs, clamped, must be the tables decode.json was recorded with.
TEST(DecodeGolden, TemperaturesOfTheCheckpointConfigs) {
  for (const char* name : kModels) {
    LAYA_GOLDEN_LIST(seq, "sequences.json", name, "config");
    LAYA_GOLDEN_LIST(rec, "decode.json", name, "config");
    auto t = dc::Temperatures::Of(seq->Find("temperature"), seq->Find("temperature_by_options"));
    ASSERT_TRUE(t.ok());
    const Value* by_type = rec->Find("temperature");
    for (size_t i = 0; i < 3; i++) EXPECT_EQ(t.value().by_type()[i], laya_test::Num(by_type->items()[i])) << name << i;
    const Value* by_options = rec->Find("temperature_by_options");
    EXPECT_EQ(t.value().by_options().size(), by_options->size()) << name;
    for (size_t i = 0; i < by_options->size(); i++) {
      EXPECT_EQ(t.value().by_options().at(by_options->key(i)), laya_test::Num(by_options->value(i))) << by_options->key(i);
    }
  }
}

TEST(DecodeGolden, Round4) {
  LAYA_GOLDEN_LIST(g, "go_extra.json", "round4");
  const Value* values = g->Find("values");
  const Value* rounded = g->Find("rounded");
  LAYA_EXPECT_ROWS(values, 5605);
  ASSERT_EQ(rounded->size(), values->size());
  int bad = 0;
  for (size_t i = 0; i < values->size(); i++) {
    const double got = dc::Round4(laya_test::Num(values->items()[i]));
    const double want = laya_test::Num(rounded->items()[i]);
    if (std::memcmp(&got, &want, sizeof got) != 0 && !(std::isnan(got) && std::isnan(want))) {
      if (bad++ < 5) ADD_FAILURE() << "round4(" << laya_test::Num(values->items()[i]) << ") = " << got << ", Python " << want;
    }
  }
  EXPECT_EQ(bad, 0);
}

// Round4 reads and writes through "%.4f"; a comma-decimal locale must not change it.
TEST(Round4, IgnoresACommaDecimalLocale) {
  laya_test::CommaLocale comma;
  if (!comma.ok()) GTEST_SKIP() << "no comma-decimal locale installed";
  EXPECT_EQ(dc::Round4(0.12345), 0.1235);
  EXPECT_EQ(dc::Round4(0.00005), 0.0001);
  EXPECT_EQ(dc::Round4(2.5), 2.5);
  EXPECT_EQ(dc::PyFloat("1.5").value(), 1.5);
}

TEST(Round4, HalfEvenOnTheBinaryValue) {
  EXPECT_EQ(dc::Round4(0.00005), 0.0001);   // the double is a hair above the tie
  EXPECT_EQ(dc::Round4(0.00015), 0.0001);   // a hair below
  EXPECT_EQ(dc::Round4(0.03125), 0.0312);   // an exact tie: even
  EXPECT_EQ(dc::Round4(0.09375), 0.0938);
  EXPECT_TRUE(std::signbit(dc::Round4(-0.00001)));
  EXPECT_TRUE(std::isnan(dc::Round4(NAN)));
  EXPECT_TRUE(std::isinf(dc::Round4(INFINITY)));
}

// 5,000 random, tied and near-tied rows against Python's bytes. Python computes exp in float32,
// whose last bits differ between CPUs; two kinds of difference are tolerated and counted: a number
// one unit in the fourth decimal on a rounding boundary, and a choice between options whose
// quotients are within kNearTieBand of the largest. Anything else fails.
constexpr double kNearTieBand = 1e-6;

struct Diff {
  std::string path;
  double delta = -1;  // -1: not a number
  std::string got, want;
};

void DiffJson(const std::string& path, const Value& got, const Value& want, std::vector<Diff>* out) {
  auto num = [](const Value& v) { return v.kind() == Kind::Int || v.kind() == Kind::Float; };
  if (want.kind() == Kind::Object) {
    bool same = got.kind() == Kind::Object && got.size() == want.size();
    for (size_t i = 0; same && i < want.size(); i++) same = got.key(i) == want.key(i);
    if (!same) {
      out->push_back({path, -1, laya::pyvalue::Repr(got), laya::pyvalue::Repr(want)});
      return;
    }
    for (size_t i = 0; i < want.size(); i++) DiffJson(path + "/" + want.key(i), got.value(i), want.value(i), out);
  } else if (num(want) && num(got)) {
    if (laya::pyvalue::Repr(got) != laya::pyvalue::Repr(want)) {
      out->push_back({path, std::fabs(laya_test::Num(got) - laya_test::Num(want)), laya::pyvalue::Repr(got), laya::pyvalue::Repr(want)});
    }
  } else if (laya::pyvalue::Repr(got) != laya::pyvalue::Repr(want)) {
    out->push_back({path, -1, laya::pyvalue::Repr(got), laya::pyvalue::Repr(want)});
  }
}

bool NearTied(const std::vector<float>& logits, const std::vector<int>& order, const dc::Temperatures& t, sc::Type kind,
              int go_idx, int py_idx) {
  const double temp = t.For(kind, int64_t(logits.size()), "");
  std::vector<float> z(logits.size());
  float top = -INFINITY;
  for (size_t i = 0; i < z.size(); i++) top = std::max(top, z[i] = logits[i] / float(temp));
  if (!order.empty()) {
    std::vector<float> canonical(z.size());
    for (size_t slot = 0; slot < order.size(); slot++) canonical[size_t(order[slot])] = z[slot];
    z = canonical;
  }
  for (int i : {go_idx, py_idx}) {
    if (i < 0 || size_t(i) >= z.size() || double(top) - double(z[size_t(i)]) > kNearTieBand) return false;
  }
  return true;
}

sc::Question FuzzQuestion(const std::string& kind, const Value& crit, const Value& order) {
  Value g = Value::Object();
  g.Set("t", Value::String(kind));
  g.Set("ins", Value::String("i"));
  Value labels = Value::Object();
  if (kind == "choice") {
    for (const Value& l : crit.items()) labels.Set(l.as_string(), Value::Null());
    g.Set("crit", labels);
  } else {
    g.Set("crit", crit);
  }
  return GoldenQuestion(g, order);
}

TEST(DecodeGolden, AnswersFuzz) {
  LAYA_GOLDEN_LIST(cases, "go_extra.json", "answers_fuzz", "cases");
  LAYA_EXPECT_ROWS(cases, 5000);
  int identical = 0, near_tie = 0, rounded = 0, bad = 0;
  for (size_t i = 0; i < cases->size(); i++) {
    const Value& rec = cases->items()[i];
    const std::string kind = rec.Find("type")->as_string();
    const Value& order_v = *rec.Find("option_order");
    std::vector<int> order;
    for (const Value& o : order_v.items()) order.push_back(int(o.as_int()));
    sc::Question q = FuzzQuestion(kind, *rec.Find("criteria"), order_v);
    std::vector<float> logits;
    for (const Value& s : rec.Find("logits")->items()) logits.push_back(float(dc::PyFloat(s.as_string()).value()));
    const Value* temp_v = rec.Find("temperature");
    const dc::Temperatures temps = Uniform(laya_test::Num(*temp_v));
    dc::Row row;
    row.logits = logits;
    row.k = int64_t(logits.size());
    row.act = 0.25f;
    auto a = dc::Decode(q, row, "", &temps);
    ASSERT_TRUE(a.ok()) << i << ": " << a.error();
    const std::string got = a.value().ToJson().value(), want = rec.Find("answer_text")->as_string();
    if (got == want) {
      identical++;
      continue;
    }
    std::vector<Diff> diffs;
    DiffJson("", laya::pyvalue::Parse(got).value(), laya::pyvalue::Parse(want).value(), &diffs);
    bool ok = true, saw_choice = false, saw_round = false;
    for (const Diff& d : diffs) {
      if (d.path == "/choice" && kind == "choice") {
        saw_choice = true;
        ok = ok && NearTied(logits, order, temps, sc::Type::Choice, std::atoi(d.got.c_str() + 2), std::atoi(d.want.c_str() + 2));
      } else if (d.delta > 0 && d.delta < 1.5e-4) {
        saw_round = true;
      } else {
        ok = false;
      }
    }
    if (!ok) {
      if (bad++ < 5) ADD_FAILURE() << "case " << i << " (" << kind << ")\n go: " << got << "\n py: " << want;
    } else if (saw_choice) {
      near_tie++;
    } else if (saw_round) {
      rounded++;
    }
  }
  std::printf("answers_fuzz: %d identical, %d near-tie choices, %d one-in-the-fourth-decimal, %d bad\n", identical, near_tie, rounded, bad);
  EXPECT_EQ(bad, 0);
  EXPECT_LE(rounded, 25) << "the tolerance is hiding more than float32 arithmetic";
}


sc::Question Pick(int n) {
  std::vector<std::string> labels;
  for (int i = 0; i < n; i++) labels.push_back("o" + std::to_string(i));
  return sc::Question::ChoiceOf("pick", labels).value();
}

dc::Row RowOf(std::vector<float> logits) {
  dc::Row r;
  r.k = int64_t(logits.size());
  r.logits = std::move(logits);
  r.act = 0.5f;
  return r;
}

TEST(Bucket, EdgesByOptionCount) {
  const sc::Type c = sc::Type::Choice;
  EXPECT_EQ(dc::Bucket(c, 1), "choice:2");
  EXPECT_EQ(dc::Bucket(c, 2), "choice:2");
  EXPECT_EQ(dc::Bucket(c, 3), "choice:3-5");
  EXPECT_EQ(dc::Bucket(c, 5), "choice:3-5");
  EXPECT_EQ(dc::Bucket(c, 6), "choice:6-10");
  EXPECT_EQ(dc::Bucket(c, 10), "choice:6-10");
  EXPECT_EQ(dc::Bucket(c, 11), "choice:11+");
  EXPECT_EQ(dc::Bucket(sc::Type::Score, 4), "score:3-5");
}

// The bucket table beats the type table, and a language override beats both.
TEST(Temperatures, BucketThenTypeThenLanguage) {
  auto js = laya::pyvalue::Parse(R"({"t":[1.5,2,3],"o":{"choice:3-5":4},"l":{"zh-Hans":{"temperature":[2,2,2],"temperature_by_options":{"choice:2":2.5}},"yo":null}})").value();
  auto t = dc::Temperatures::Of(js.Find("t"), js.Find("o"), js.Find("l")).value();
  bool ov = true;
  EXPECT_EQ(t.For(sc::Type::Choice, 4, "", &ov), 4.0);
  EXPECT_FALSE(ov);
  EXPECT_EQ(t.For(sc::Type::Choice, 7, ""), 1.5);
  EXPECT_EQ(t.For(sc::Type::Score, 7, ""), 2.0);
  EXPECT_EQ(t.For(sc::Type::Choice, 2, "ZH-Hant", &ov), 2.5);
  EXPECT_TRUE(ov);
  EXPECT_EQ(t.For(sc::Type::Choice, 4, "zh"), 2.0);
  EXPECT_EQ(t.For(sc::Type::Choice, 4, "yo-NG"), 1.5);  // null inherits the checkpoint's own
  EXPECT_EQ(t.For(sc::Type::Choice, 4, "xx"), 4.0);
}

TEST(Temperatures, RefusesWhatPythonRefuses) {
  auto of = [](const char* json) {
    auto v = laya::pyvalue::Parse(json).value();
    return dc::Temperatures::Of(&v, nullptr);
  };
  EXPECT_FALSE(of("[1,2]").ok());
  EXPECT_FALSE(of("[1,2,3,4]").ok());
  EXPECT_FALSE(of(R"("abc")").ok());
  auto v = laya::pyvalue::Parse(R"({"zh":5})").value();
  EXPECT_FALSE(dc::Temperatures::Of(nullptr, nullptr, &v).ok());
}

TEST(Clamp, BoundsAndNonNumbers) {
  auto c = [](const char* j) { return dc::Clamp(laya::pyvalue::Parse(j).value()); };
  EXPECT_EQ(c("0.1"), 0.5);
  EXPECT_EQ(c("9"), 5.0);
  EXPECT_EQ(c("2"), 2.0);
  EXPECT_EQ(c("true"), 1.0);
  EXPECT_EQ(c("null"), 1.0);
  EXPECT_EQ(c("[2]"), 1.0);
  EXPECT_EQ(c("NaN"), 1.0);
  EXPECT_EQ(c("Infinity"), 1.0);
  EXPECT_EQ(c(R"(" 1_0.5 ")"), 5.0);
  EXPECT_EQ(c(R"("1__0")"), 1.0);
  EXPECT_EQ(c(R"("0x10")"), 1.0);
  EXPECT_EQ(c(R"("1e400")"), 1.0);  // out of range is infinite
  EXPECT_EQ(c("100000000000000000000000"), 5.0);
}

TEST(PyFloat, Spellings) {
  EXPECT_EQ(dc::PyFloat("1.5").value(), 1.5);
  EXPECT_EQ(dc::PyFloat(".5").value(), 0.5);
  EXPECT_EQ(dc::PyFloat("5.").value(), 5.0);
  EXPECT_EQ(dc::PyFloat("-1e2").value(), -100.0);
  EXPECT_EQ(dc::PyFloat("\xD9\xA1.\xD9\xA5").value(), 1.5);  // Arabic-Indic digits
  EXPECT_TRUE(std::isinf(dc::PyFloat("-Infinity").value()));
  EXPECT_TRUE(std::isnan(dc::PyFloat("nan").value()));
  for (const char* bad : {"", ".", "e5", "1e", "1.5x", "_1", "1_", "1_.5", "0x1p3", "--1", "1 2"}) {
    EXPECT_FALSE(dc::PyFloat(bad).has_value()) << bad;
  }
}

// Python divides in float32, so two adjacent float32 logits can become one quotient: an exact tie
// that numpy.argmax gives to the lower index. The first pair fails if the quotient is kept in
// double; the second, whose temperature is not a float32, also fails if it is divided in double and rounded.
TEST(Decode, QuotientTieFallsToTheFirst) {
  struct Case {
    double temperature;
    uint32_t lower, upper;
  };
  for (const Case& c : {Case{3.0, 1069547521u, 1069547522u}, Case{1.3, 1072020843u, 1072020844u}}) {
    float lo, hi;
    std::memcpy(&lo, &c.lower, 4);
    std::memcpy(&hi, &c.upper, 4);
    ASSERT_TRUE(lo < hi && lo / float(c.temperature) == hi / float(c.temperature));
    auto t = Uniform(c.temperature);
    auto a = dc::Decode(Pick(2), RowOf({lo, hi}), "", &t);
    ASSERT_TRUE(a.ok());
    EXPECT_EQ(a.value().choice.Key(), "o0") << c.temperature;
  }
}

// numpy keeps exp and the sum in float32, so a logit gap below float32 resolution is an exact tie.
// Expected values were computed once from ONNXAgent._decode_answers with lo = float32(1e-6),
// hi = nextafter(lo, 1), T = 1: [lo,hi], [hi,lo], [lo,hi,hi] and [lo,hi] with option_order [1,0] all
// give p equal (0.5 / 0.3333) and choice "a" (the first option).
TEST(Decode, SubUlpGapIsAFloat32TieForTheChoice) {
  const float lo = 1e-6f, hi = std::nextafter(lo, 1.0f);
  auto t = Uniform(1);
  auto key = [&](int n, std::vector<float> v) { return dc::Decode(Pick(n), RowOf(v), "", &t).value().choice.Key(); };
  EXPECT_EQ(key(2, {lo, hi}), "o0");
  EXPECT_EQ(key(2, {hi, lo}), "o0");
  EXPECT_EQ(key(3, {lo, hi, hi}), "o0");
  EXPECT_EQ(key(3, {lo, 5.0f, hi}), "o1");  // a real gap still decides
  EXPECT_EQ(key(3, {-3.0f, lo, hi}), "o1");
  auto permuted = Pick(2).WithOptionOrder({1, 0}).value();
  auto a = dc::Decode(permuted, RowOf({lo, hi}), "", &t).value();
  EXPECT_EQ(a.choice.Key(), "o0");
  EXPECT_EQ(a.probabilities[0].p, 0.5);
}

TEST(Decode, FirstMaximumWinsAnExactTie) {
  auto t = Uniform(1);
  EXPECT_EQ(dc::Decode(Pick(3), RowOf({1, 1, 1}), "", &t).value().choice.Key(), "o0");
  EXPECT_EQ(dc::Decode(Pick(3), RowOf({0, 2, 2}), "", &t).value().choice.Key(), "o1");
}

TEST(Decode, LargeLogitsDoNotOverflow) {
  auto t = Uniform(1);
  auto a = dc::Decode(Pick(2), RowOf({800, 0}), "", &t);
  ASSERT_TRUE(a.ok());
  EXPECT_EQ(a.value().probabilities[0].p, 1.0);
  EXPECT_EQ(a.value().probabilities[1].p, 0.0);
}

TEST(Decode, RefusesWhatDoesNotFit) {
  auto t = Uniform(1);
  EXPECT_FALSE(dc::Decode(sc::Question(), RowOf({1, 2}), "", &t).ok());
  EXPECT_FALSE(dc::Decode(Pick(3), RowOf({1, 2}), "", &t).ok());           // k differs from the options
  dc::Row wide = RowOf({1, 2, 3});
  wide.k = 2;
  EXPECT_FALSE(dc::Decode(Pick(3), wide, "", &t).ok());
  dc::Row narrow = RowOf({1, 2});
  narrow.k = 3;
  EXPECT_FALSE(dc::Decode(Pick(3), narrow, "", &t).ok());                   // fewer logits than options
  EXPECT_FALSE(dc::Decode(Pick(2), RowOf({1, NAN}), "", &t).ok());
  EXPECT_FALSE(dc::Decode(Pick(2), RowOf({1, INFINITY}), "", &t).ok());
  dc::Row bad_act = RowOf({1, 2});
  bad_act.act = NAN;
  EXPECT_FALSE(dc::Decode(Pick(2), bad_act, "", &t).ok());
  auto small = Uniform(0.5);
  EXPECT_FALSE(dc::Decode(Pick(2), RowOf({3e38f, 0}), "", &small).ok());   // overflows float32 once divided
  EXPECT_FALSE(dc::Decode(sc::Question::Noul("n").value(), RowOf({1}), "", &t).ok());
}

TEST(Decode, PadColumnsAreNeverRead) {
  auto t = Uniform(1);
  dc::Row r = RowOf({1, 2});
  r.logits.push_back(NAN);  // past k
  EXPECT_TRUE(dc::Decode(Pick(2), r, "", &t).ok());
}

TEST(Decode, OptionOrderIsUndone) {
  auto t = Uniform(1);
  sc::Question q = Pick(3).WithOptionOrder({2, 0, 1}).value();  // slot 0 holds option 2, ...
  auto a = dc::Decode(q, RowOf({9, 0, 0}), "", &t);
  ASSERT_TRUE(a.ok());
  EXPECT_EQ(a.value().choice.Key(), "o2");
}

TEST(Temperatures, SamePrimarySubtagLastWins) {
  auto js = laya::pyvalue::Parse(R"({"zh-CN":{"temperature":[2,2,2]},"zh-TW":{"temperature":[4,4,4]}})").value();
  auto t = dc::Temperatures::Of(nullptr, nullptr, &js).value();
  EXPECT_EQ(t.For(sc::Type::Choice, 2, "zh", nullptr), 4.0);
}

TEST(Decode, BinningReplacesTheConfidenceOnlyWithoutALanguageOverride) {
  auto js = laya::pyvalue::Parse(R"({"l":{"zh":null}})").value();
  auto t = dc::Temperatures::Of(nullptr, nullptr, js.Find("l")).value();
  std::map<std::string, dc::Bins> bins;
  bins["choice:2"].values = {0.25, 0.75};
  t.SetBinning(bins);
  auto plain = dc::Decode(Pick(2), RowOf({4, 0}), "", &t).value();
  EXPECT_EQ(plain.answer_confidence, 0.75);  // 0.98 falls in the top of two bins
  auto zh = dc::Decode(Pick(2), RowOf({4, 0}), "zh", &t).value();
  EXPECT_NE(zh.answer_confidence, 0.75);
  EXPECT_EQ(dc::ApplyBinning(1.0, "choice:2", t.binning()), 0.75);  // clamped into the last bin
  EXPECT_EQ(dc::ApplyBinning(0.3, "absent", t.binning()), 0.3);
}

TEST(ActionProbabilities, SoftmaxAndRefusals) {
  auto p = dc::ActionProbabilities({0, 0});
  ASSERT_TRUE(p.ok());
  EXPECT_EQ(p.value()[0], 0.5f);
  EXPECT_FALSE(dc::ActionProbabilities({}).ok());
  EXPECT_FALSE(dc::ActionProbabilities({1, NAN}).ok());
  EXPECT_TRUE(dc::ActionProbabilities({800, 0}).ok());
}

}  // namespace
