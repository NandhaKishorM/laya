#include "laya/schema.hpp"

#include <gtest/gtest.h>

#include "golden.hpp"
#include "laya/pyjson.hpp"

namespace {

using laya::pyvalue::Value;
namespace sc = laya::schema;
using laya_test::At;
using Kind = Value::Kind;

std::string Dump(const Value& v) {
  auto r = laya::pyvalue::Dumps(v);
  return r.ok() ? r.value() : "<" + r.error() + ">";
}

std::vector<std::string> Strings(const Value& list) {
  std::vector<std::string> out;
  for (const Value& v : list.items()) out.push_back(v.as_string());
  return out;
}

void CheckQuestion(const sc::Question& q, const Value& rec, const std::string& what) {
  EXPECT_EQ(std::string(sc::Name(q.type())), rec.Find("type")->as_string()) << what;
  EXPECT_EQ(q.instructions(), rec.Find("instructions")->as_string()) << what;
  EXPECT_EQ(q.RenderedOptions(), Strings(*rec.Find("options"))) << what;
  EXPECT_EQ(q.Labels(), Strings(*rec.Find("labels"))) << what;
  if (const Value* lg = rec.Find("legend"); lg != nullptr && !lg->is_null()) {
    EXPECT_EQ(q.Legend(), Strings(*lg)) << what;
  }
  std::vector<int> want;
  if (const Value* o = rec.Find("option_order"); o != nullptr && !o->is_null()) {
    for (const Value& v : o->items()) want.push_back(int(v.as_int()));
  }
  EXPECT_EQ(q.OptionOrder(), want) << what;
}

class QuestionRow : public ::testing::TestWithParam<int> {};
TEST_P(QuestionRow, MatchesAgentCheckQuestion) {
  LAYA_GOLDEN_LIST(rows, "go_extra.json", "questions", "cases");
  LAYA_EXPECT_ROWS(rows, 140);
  for (size_t i = size_t(GetParam()) * laya_test::kChunk; i < rows->size() && i < size_t(GetParam() + 1) * laya_test::kChunk; i++) {
    const Value& rec = rows->value(i);
    const std::string name = rec.Find("case")->as_string();
    const std::string id = rec.Find("id")->as_string();
    Value doc = Value::Object();
    doc.Set(id, *rec.Find("definition"));
    auto qs = sc::Questions::Parse(Dump(doc));
    if (const Value* want = rec.Find("error")) {
      ASSERT_FALSE(qs.ok()) << name << ": accepted; Python refuses with " << want->as_string();
      EXPECT_EQ(qs.error(), want->as_string()) << name;
      continue;
    }
    ASSERT_TRUE(qs.ok()) << name << ": " << qs.error();
    const sc::Question* q = qs.value().Get(id);
    ASSERT_NE(q, nullptr) << name;
    ASSERT_EQ(qs.value().size(), 1u) << name;
    CheckQuestion(*q, rec, name);
    auto out = qs.value().ToJson();
    ASSERT_TRUE(out.ok()) << name << ": " << out.error();
    auto again = sc::Questions::Parse(out.value());
    ASSERT_TRUE(again.ok()) << name << ": own output " << out.value() << ": " << again.error();
    CheckQuestion(*again.value().Get(id), rec, name + " (round trip)");
    EXPECT_EQ(again.value().ToJson().value(), out.value()) << name << ": marshal is not stable";
  }
}
INSTANTIATE_TEST_SUITE_P(Golden, QuestionRow, ::testing::ValuesIn(laya_test::Chunks(140)));

TEST(Questions, RefusalAndAcceptanceCounts) {
  LAYA_GOLDEN_LIST(rows, "go_extra.json", "questions", "cases");
  int refused = 0, accepted = 0;
  for (const Value& rec : rows->items()) (rec.Find("error") ? refused : accepted)++;
  EXPECT_EQ(refused, 93);
  EXPECT_EQ(accepted, 47);
}

// The typed constructors refuse with the wire reader's words.
TEST(Builders, ReasonsMatchPython) {
  LAYA_GOLDEN_LIST(rows, "go_extra.json", "questions", "cases");
  std::map<std::string, std::string> gold;
  const std::string prefix = "question 'q': ";
  for (const Value& rec : rows->items()) {
    if (const Value* e = rec.Find("error")) gold[rec.Find("case")->as_string()] = e->as_string().substr(prefix.size());
  }
  auto two = sc::Question::ChoiceOf("Pick.", {"a", "b"}).value();
  auto noul = sc::Question::Noul("Holds?").value();
  auto spec = [](const char* f, const char* t) {
    sc::NoulSpec s;
    s.false_label = f;
    s.true_label = t;
    return s;
  };
  struct Row {
    const char* gold;
    std::string got;
  };
  auto err = [](const auto& r) { return r.ok() ? std::string("<accepted>") : r.error(); };
  std::vector<Row> cases = {
      {"ins-blank", err(sc::Question::ChoiceOf(" \t\n", {"a", "b"}))},
      {"ins-nbsp", err(sc::Question::Score(" \xc2\x85", {Value::String("a")}))},
      {"ins-ideographic-space", err(sc::Question::Noul("\xe3\x80\x80"))},
      {"ins-empty", err(sc::Question::Choice("", {{sc::Label::Text("a"), Value()}}))},
      {"choice-empty-list", err(sc::Question::Choice("Pick.", {}))},
      {"choice-list-repeat", err(sc::Question::ChoiceOf("Pick.", {"a", "b", "a"}))},
      {"score-empty", err(sc::Question::Score("Rate.", {}))},
      {"score-null-second", err(sc::Question::Score("Rate.", {Value::String("a"), Value(), Value::String("c")}))},
      {"labels-same", err(sc::Question::Noul("Holds?", spec("a", "a")))},
      {"labels-same-after-strip", err(sc::Question::Noul("Holds?", spec("a", " a ")))},
      {"labels-missing-true", err(sc::Question::Noul("Holds?", spec("a", "")))},
      {"labels-blank-value", err(sc::Question::Noul("Holds?", spec(" ", "x")))},
      {"order-short", err(two.WithOptionOrder({0}))},
      {"order-long", err(two.WithOptionOrder({0, 1, 2}))},
      {"order-repeat", err(two.WithOptionOrder({0, 0}))},
      {"order-range", err(two.WithOptionOrder({0, 2}))},
      {"order-negative", err(two.WithOptionOrder({-1, 0}))},
      {"order-empty", err(two.WithOptionOrder({}))},
      {"order-noul-wrong-size", err(noul.WithOptionOrder({0, 1, 2}))},
  };
  for (const Row& c : cases) {
    auto it = gold.find(c.gold);
    ASSERT_NE(it, gold.end()) << c.gold;
    EXPECT_EQ(c.got, it->second) << c.gold;
  }
}

TEST(Builders, OptionOrderAndSubset) {
  auto q = sc::Question::ChoiceOf("Pick.", {"a", "b", "c"}).value();
  auto o = q.WithOptionOrder({2, 0, 1});
  ASSERT_TRUE(o.ok());
  EXPECT_EQ(o.value().OptionOrder(), (std::vector<int>{2, 0, 1}));
  EXPECT_EQ(o.value().RenderedOptions(), q.RenderedOptions());  // rendering is label order; the order applies later
  EXPECT_TRUE(o.value().WithoutOptionOrder().OptionOrder().empty());
  EXPECT_TRUE(q.OptionOrder().empty());  // a question is immutable
  auto s = o.value().Subset({2, 0});
  ASSERT_TRUE(s.ok());
  EXPECT_EQ(s.value().Labels(), (std::vector<std::string>{"c", "a"}));
  EXPECT_TRUE(s.value().OptionOrder().empty());
  EXPECT_FALSE(q.Subset({}).ok());
  EXPECT_FALSE(q.Subset({0, 0}).ok());
  EXPECT_FALSE(q.Subset({3}).ok());
  EXPECT_FALSE(sc::Question().WithOptionOrder({0}).ok());
  EXPECT_FALSE(sc::Question().ToJson().ok());
}

TEST(Labels, PythonDictKeying) {
  using L = sc::Label;
  EXPECT_EQ(L::OfInt(1).EqKey(), L::OfFloat(1.0).EqKey());
  EXPECT_EQ(L::OfInt(1).EqKey(), L::OfBool(true).EqKey());
  EXPECT_NE(L::OfInt(1).EqKey(), L::Text("1").EqKey());
  EXPECT_NE(L::OfFloat(0.5).EqKey(), L::OfInt(0).EqKey());
  EXPECT_EQ(L::OfFloat(1e22).EqKey(), "n:10000000000000000000000");
  EXPECT_EQ(L::OfFloat(-0.0).EqKey(), L::OfInt(0).EqKey());
  EXPECT_EQ(L::OfBool(true).String(), "True");
  EXPECT_EQ(L::OfFloat(1.0 / 0.0).String(), "inf");
  EXPECT_EQ(L::OfFloat(1.0 / 0.0).Key(), "Infinity");
  EXPECT_FALSE(L::Of(Value()).ok());
  EXPECT_FALSE(L::Of(Value::Array()).ok());
  // 1, 1.0 and true are one dict key, so a question may not hold two of them.
  auto q = sc::Question::Choice("Pick.", {{L::OfInt(1), Value()}, {L::OfFloat(1.0), Value()}});
  ASSERT_FALSE(q.ok());
  EXPECT_EQ(q.error(),
            "choice label 1 (1.0) repeats label 0; the labels are the answer keys, so every option needs its own (1, "
            "1.0 and True are one key)");
}

// strtod follows the process locale: under de_DE "1.5" read as 1, so 1.5 and 1 looked like one key.
TEST(Labels, EqKeyIgnoresACommaDecimalLocale) {
  laya_test::CommaLocale comma;
  if (!comma.ok()) GTEST_SKIP() << "no comma-decimal locale installed";
  using L = sc::Label;
  EXPECT_NE(L::OfFloat(1.5).EqKey(), L::OfInt(1).EqKey());
  EXPECT_EQ(L::OfFloat(1.5).EqKey(), "f:1.5");
  EXPECT_EQ(L::OfFloat(2.0).EqKey(), L::OfInt(2).EqKey());
  EXPECT_EQ(L::OfFloat(1e22).EqKey(), "n:10000000000000000000000");
  EXPECT_TRUE(sc::Question::Choice("Pick.", {{L::OfInt(1), Value()}, {L::OfFloat(1.5), Value()}}).ok());
}

TEST(Questions, Duplicates) {
  auto q = sc::Question::Noul("Holds?").value();
  EXPECT_FALSE(sc::Questions::Of({{"a", q}, {"a", q}}).ok());
  EXPECT_FALSE(sc::Questions::Of({{" ", q}}).ok());
  EXPECT_FALSE(sc::Questions::Of({{"a", sc::Question()}}).ok());
  EXPECT_EQ(sc::Questions::Of({{"b", q}, {"a", q}}).value().Ids(), (std::vector<std::string>{"b", "a"}));
}

// Answer marshals as Python writes it: parse the Python text, write it back, and get it byte for byte.
class AnswerRow : public ::testing::TestWithParam<int> {};
TEST_P(AnswerRow, RoundTripsPythonText) {
  LAYA_GOLDEN_LIST(rows, "go_extra.json", "label_answers", "cases");
  LAYA_EXPECT_ROWS(rows, 500);
  for (size_t i = size_t(GetParam()) * laya_test::kChunk; i < rows->size() && i < size_t(GetParam() + 1) * laya_test::kChunk; i++) {
    const std::string text = rows->value(i).Find("answer_text")->as_string();
    auto v = laya::pyvalue::Parse(text);
    ASSERT_TRUE(v.ok()) << i;
    auto a = sc::Answer::FromJson(v.value());
    ASSERT_TRUE(a.ok()) << i << ": " << a.error();
    auto out = a.value().ToJson();
    ASSERT_TRUE(out.ok()) << i << ": " << out.error();
    // Python writes 1 and "1" as two keys of one object; a parsed object keeps one. Those rows are
    // the ones with fewer probabilities than options and are not byte-comparable after a parse.
    const Value* opts = rows->value(i).Find("options");
    if (a.value().type == sc::Type::Choice && a.value().probabilities.size() < opts->size()) continue;
    EXPECT_EQ(out.value(), text) << "row " << i;
  }
}
INSTANTIATE_TEST_SUITE_P(Golden, AnswerRow, ::testing::ValuesIn(laya_test::Chunks(500)));

TEST(Answer, GateFieldsAndRefusals) {
  sc::Answer a;
  a.type = sc::Type::Noul;
  a.noul = 0.5;
  a.confidence = 0.25;
  a.answer_confidence = 0.75;
  a.act_probability = 1;
  a.window = sc::Window{1, 10, 20, 3};
  a.low_confidence = true;
  a.abstention = "abstained";
  a.abstention_threshold = 0.5;
  const char* want =
      "{\"type\":\"noul\",\"noul\":0.5,\"confidence\":0.25,\"answer_confidence\":0.75,\"action\":{\"act_probability\":1.0},"
      "\"window\":{\"index\":1,\"token_start\":10,\"token_end\":20,\"count\":3},\"low_confidence\":true,"
      "\"abstention\":\"abstained\",\"abstention_threshold\":0.5}";
  EXPECT_EQ(a.ToJson().value(), want);
  auto back = sc::Answer::FromJson(laya::pyvalue::Parse(want).value());
  ASSERT_TRUE(back.ok()) << back.error();
  EXPECT_EQ(back.value().ToJson().value(), want);
  a.noul = 0.0 / 0.0;  // NaN is not JSON
  EXPECT_FALSE(a.ToJson().ok());
  a.noul = 0.5;
  a.abstention.clear();  // the gate fields come together
  auto lone = a.ToJson();
  ASSERT_TRUE(lone.ok());
  EXPECT_FALSE(sc::Answer::FromJson(laya::pyvalue::Parse(lone.value()).value()).ok());
  EXPECT_FALSE(sc::Answer().ToJson().ok());
}

TEST(Prediction, UsageShapes) {
  sc::Prediction p;
  p.usage.input_tokens = 10;
  p.usage.output_tokens = 2;
  p.usage.truncated_questions = {"q"};
  p.usage.truncated = true;
  p.usage.collapsed_options = {{"q", {3, 1, 4}}, {"r", {2, 2, std::nullopt}}};
  EXPECT_EQ(p.ToJson().value(),
            "{\"model\":\"laya-rl-agent-onnx\",\"answers\":{},\"usage\":{\"input_tokens\":10,\"output_tokens\":2,"
            "\"state_tokens\":0,\"state_tokens_dropped\":0,\"truncated\":true,\"truncated_questions\":[\"q\"],"
            "\"options\":{\"q\":{\"total\":3,\"distinct\":1,\"tokens_per_option\":4},\"r\":{\"total\":2,\"distinct\":2,"
            "\"tokens_per_option\":null}}}}");
  sc::Usage scan;
  scan.input_tokens = 5;
  scan.output_tokens = 1;
  scan.windows = 3;
  EXPECT_EQ(scan.ToJson().value(), "{\"input_tokens\":5,\"output_tokens\":1,\"windows\":3}");
  scan.truncated_windows = 2;
  EXPECT_EQ(scan.ToJson().value(),
            "{\"input_tokens\":5,\"output_tokens\":1,\"state_tokens\":0,\"state_tokens_dropped\":0,\"truncated\":2,"
            "\"truncated_questions\":[],\"windows\":3}");
}

}  // namespace
