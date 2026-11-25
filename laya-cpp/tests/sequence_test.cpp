#include "laya/sequence.hpp"

#include <gtest/gtest.h>

#include "checkpoint.hpp"
#include "golden.hpp"

namespace {

using laya::pyvalue::Value;
namespace sc = laya::schema;
namespace sq = laya::sequence;
using laya_test::At;
using Kind = Value::Kind;

const char* const kNames[] = {"english", "multilingual"};

sc::Question GoldenQuestion(const Value& g, const Value& order) {
  const std::string kind = g.Find("t")->as_string();
  const std::string ins = g.Find("ins")->as_string();
  const Value* crit = g.Find("crit");
  laya::Result<sc::Question> q = laya::Result<sc::Question>::Fail("unset");
  if (kind == "choice") {
    std::vector<sc::Option> opts;
    for (size_t i = 0; i < crit->size(); i++) opts.push_back({sc::Label::Text(crit->key(i)), crit->value(i)});
    q = sc::Question::Choice(ins, opts);
  } else if (kind == "score") {
    q = sc::Question::Score(ins, crit->items());
  } else {
    sc::NoulSpec spec;
    if (crit != nullptr && crit->kind() == Kind::Object) {
      if (const Value* t = crit->Find("true")) spec.true_text = *t;
      if (const Value* f = crit->Find("false")) spec.false_text = *f;
    }
    if (const Value* l = g.Find("labels"); l != nullptr && !l->is_null()) {
      spec.false_label = l->Find("false")->as_string();
      spec.true_label = l->Find("true")->as_string();
    }
    q = sc::Question::Noul(ins, spec);
  }
  EXPECT_TRUE(q.ok()) << (q.ok() ? "" : q.error());
  if (!q.ok()) std::abort();
  if (!order.is_null()) {
    std::vector<int> o;
    for (const Value& v : order.items()) o.push_back(int(v.as_int()));
    q = q.value().WithOptionOrder(o);
    EXPECT_TRUE(q.ok());
    if (!q.ok()) std::abort();
  }
  return q.value();
}

std::vector<int32_t> Ids32(const Value& list) {
  std::vector<int32_t> out;
  for (const Value& v : list.items()) out.push_back(int32_t(v.as_int()));
  return out;
}
std::vector<int64_t> Ids64(const Value& list) {
  std::vector<int64_t> out;
  for (const Value& v : list.items()) out.push_back(v.as_int());
  return out;
}

// The generator's _window_question: options of four words each.
sc::Question WindowQuestion(int options) {
  std::vector<sc::Option> opts;
  for (int i = 0; i < options; i++) {
    std::string text;
    for (int w = 0; w < 4; w++) text += (w ? " " : "") + std::string("criterion") + std::to_string(i) + " word" + std::to_string(w);
    char label[16];
    std::snprintf(label, sizeof label, "opt%03d", i);
    opts.push_back({sc::Label::Text(label), Value::String(text)});
  }
  return sc::Question::Choice("Pick one.", opts).value();
}

class SequenceRow : public ::testing::TestWithParam<int> {};
TEST_P(SequenceRow, MatchesBuildSequence) {
  const char* name = kNames[GetParam() / 26];
  LAYA_CHECKPOINT_TOK(tok, name);
  LAYA_GOLDEN_LIST(cases, "sequences.json", name, "cases");
  LAYA_EXPECT_ROWS(cases, 26);
  const size_t at = size_t(GetParam() % 26);
  const std::string label = std::string(name) + "/" + cases->key(at);
  const Value& rec = cases->value(at);
  sc::Question q = GoldenQuestion(*rec.Find("question"), *rec.Find("option_order"));
  auto state = sq::EncodeState(*tok, *rec.Find("state"));
  ASSERT_TRUE(state.ok()) << label << ": " << state.error();
  if (rec.Find("truncate_left")->as_bool()) state.value().keep = sq::Keep::End;
  sq::Budget b{rec.Find("max_len")->as_int(), rec.Find("head_max_len")->as_int()};
  sq::MarkersError dropped;
  auto seq = sq::Build(*tok, q, state.value(), b, &dropped);
  const auto want_markers = Ids64(*rec.Find("markers"));
  const int64_t option_count = rec.Find("stats")->Find("options")->as_int();
  if (!seq.ok() && dropped.total != 0) {
    // build_sequence returns the markers that fit and the agent refuses; Build does both.
    EXPECT_EQ(dropped.kept, int64_t(want_markers.size())) << label;
    EXPECT_EQ(dropped.total, option_count) << label;
    EXPECT_LT(int64_t(want_markers.size()), option_count) << label;
    return;
  }
  ASSERT_TRUE(seq.ok()) << label << ": " << seq.error();
  ASSERT_EQ(int64_t(want_markers.size()), option_count) << label << ": Python dropped markers and Build did not refuse";
  EXPECT_EQ(seq.value().ids, Ids32(*rec.Find("ids"))) << label;
  EXPECT_EQ(seq.value().markers, want_markers) << label;
  const Value& stats = *rec.Find("stats");
  EXPECT_EQ(seq.value().stats.options, stats.Find("options")->as_int()) << label;
  EXPECT_EQ(seq.value().stats.options_distinct, stats.Find("options_distinct")->as_int()) << label;
  const Value* tpo = stats.Find("tokens_per_option");
  EXPECT_EQ(seq.value().stats.tokens_per_option.has_value(), !tpo->is_null()) << label;
  if (!tpo->is_null() && seq.value().stats.tokens_per_option) {
    EXPECT_EQ(*seq.value().stats.tokens_per_option, tpo->as_int()) << label;
  }
  const Value& tr = *rec.Find("truncation");
  sq::Truncation want{tr.Find("state_tokens")->as_int(), tr.Find("state_tokens_used")->as_int(),
                      tr.Find("state_tokens_dropped")->as_int(), tr.Find("truncated")->as_bool()};
  EXPECT_TRUE(seq.value().truncation == want) << label;
}
INSTANTIATE_TEST_SUITE_P(Golden, SequenceRow, ::testing::ValuesIn(laya_test::Rows(52)));

class Checkpoint : public ::testing::TestWithParam<int> {};

TEST_P(Checkpoint, SpecialIdsMatchPython) {
  const char* name = kNames[GetParam()];
  LAYA_CHECKPOINT_TOK(tok, name);
  LAYA_GOLDEN_LIST(ids, "sequences.json", name, "special_ids");
  EXPECT_EQ(int64_t(*tok->Cls()), ids->Find("cls")->as_int());
  EXPECT_EQ(int64_t(*tok->Sep()), ids->Find("sep")->as_int());
  EXPECT_EQ(int64_t(*tok->Mask()), ids->Find("mask")->as_int());
  EXPECT_EQ(int64_t(*tok->Pad()), ids->Find("pad")->as_int());
  EXPECT_EQ(tok->MaskToken(), ids->Find("mask_token")->as_string());
}

// The same questions and budgets succeed or fail together with Agent._encode_state, with the same text.
TEST_P(Checkpoint, MarkerRefusalsMatchPython) {
  const char* name = kNames[GetParam()];
  LAYA_CHECKPOINT_TOK(tok, name);
  LAYA_GOLDEN_LIST(cases, "go_extra.json", "marker_refusals", name, "cases");
  LAYA_EXPECT_ROWS(cases, 42);
  int refused = 0;
  for (const Value& rec : cases->items()) {
    const int options = int(rec.Find("options")->as_int()), words = int(rec.Find("words")->as_int());
    sq::Budget b{rec.Find("max_len")->as_int(), rec.Find("head_max_len")->as_int()};
    const std::string label = std::string(name) + "/" + std::to_string(options) + " options x " + std::to_string(words) +
                              " words/" + std::to_string(b.max_len) + "," + std::to_string(b.head_max_len);
    std::vector<sc::Option> opts;
    for (int i = 0; i < options; i++) {
      std::string text;
      for (int j = 0; j < words; j++) text += (j ? " " : "") + std::string("w") + std::to_string(i) + "_" + std::to_string(j);
      char l[16];
      std::snprintf(l, sizeof l, "opt%03d", i);
      opts.push_back({sc::Label::Text(l), Value::String(text)});
    }
    auto q = sc::Question::Choice("Pick one.", opts);
    ASSERT_TRUE(q.ok()) << label;
    auto st = sq::EncodeState(*tok, Value::String("state text"));
    ASSERT_TRUE(st.ok()) << label;
    sq::MarkersError me;
    auto seq = sq::Build(*tok, q.value(), st.value(), b, &me);
    const Value* want = rec.Find("error");
    if (want->is_null()) {
      EXPECT_TRUE(seq.ok()) << label << ": refused: " << seq.error();
    } else {
      refused++;
      ASSERT_FALSE(seq.ok()) << label << ": not refused; Python says " << want->as_string();
      ASSERT_NE(me.total, 0) << label << ": " << seq.error();
      EXPECT_EQ(me.Message("it's"), want->as_string()) << label;
    }
  }
  EXPECT_GE(refused, 5) << name;
}

TEST_P(Checkpoint, WindowPlanMatchesPython) {
  const char* name = kNames[GetParam()];
  LAYA_CHECKPOINT_TOK(tok, name);
  LAYA_GOLDEN_LIST(entry, "window_plan.json", "by", name);
  const int64_t max_len = entry->Find("default_max_len")->as_int(), head_max = entry->Find("default_head_max_len")->as_int();
  const Value& rooms = *entry->Find("state_room");
  ASSERT_EQ(rooms.size(), 8u);
  for (const Value& rec : rooms.items()) {
    const int options = int(rec.Find("options")->as_int());
    sc::Question q = WindowQuestion(options);
    auto room = sq::StateRoom(*tok, q, {max_len, head_max});
    ASSERT_TRUE(room.ok());
    EXPECT_EQ(room.value(), rec.Find("room")->as_int()) << name << " room " << options;
    auto tight = sq::StateRoom(*tok, q, {256, 128});
    ASSERT_TRUE(tight.ok());
    EXPECT_EQ(tight.value(), rec.Find("room_tight")->as_int()) << name << " room_tight " << options;
  }
  const Value& cases = *entry->Find("cases");
  ASSERT_EQ(cases.size(), 24u);
  int refusals = 0;
  for (const Value& rec : cases.items()) {
    const std::string label = std::string(name) + "/" + rec.Find("case")->as_string();
    std::vector<sc::Question> qs;
    for (const Value& n : rec.Find("options")->items()) qs.push_back(WindowQuestion(int(n.as_int())));
    sq::Budget b{rec.Find("max_len")->as_int(), rec.Find("head_max_len")->as_int()};
    sq::WindowRequest req;
    if (const Value* w = rec.Find("window"); !w->is_null()) req.window = w->as_int();
    if (const Value* s = rec.Find("stride"); !s->is_null()) req.stride = s->as_int();
    sq::PlanError kind = sq::PlanError::None;
    auto plan = sq::PlanWindows(*tok, qs, b, req, &kind);
    if (const Value* e = rec.Find("error")) {
      refusals++;
      ASSERT_FALSE(plan.ok()) << label << ": Python refuses with " << e->as_string();
      EXPECT_EQ(plan.error(), e->as_string()) << label;
      const bool no_room = e->as_string().find("fill the whole sequence") != std::string::npos;
      EXPECT_EQ(kind, no_room ? sq::PlanError::NoRoom : sq::PlanError::StrideGap) << label;
      continue;
    }
    ASSERT_TRUE(plan.ok()) << label << ": " << plan.error();
    EXPECT_EQ(kind, sq::PlanError::None) << label;
    const Value& res = *rec.Find("result");
    EXPECT_EQ(plan.value().window, res.Find("window")->as_int()) << label;
    EXPECT_EQ(plan.value().stride, res.Find("stride")->as_int()) << label;
    EXPECT_EQ(plan.value().room, res.Find("room")->as_int()) << label;
    std::vector<std::string> warnings;
    for (const Value& w : rec.Find("warnings")->items()) warnings.push_back(w.as_string());
    EXPECT_EQ(plan.value().warnings, warnings) << label;
    EXPECT_EQ(plan.value().Clamped(), !warnings.empty()) << label;
  }
  EXPECT_GE(refusals, 1) << name;
}
INSTANTIATE_TEST_SUITE_P(Shipped, Checkpoint, ::testing::Values(0, 1));

// A budget of exactly kMinOptionBudget left for the instruction leaves the option spans whole;
// one token less cuts them.
TEST(BuildHead, OptionBudgetBoundary) {
  LAYA_CHECKPOINT_TOK(tok, "english");
  std::vector<sc::Option> opts;
  const char* texts[] = {"a", "a much longer description of the second option that has many words in it", "b c"};
  for (int i = 0; i < 3; i++) opts.push_back({sc::Label::Text("o" + std::to_string(i)), Value::String(texts[i])});
  auto q = sc::Question::Choice("Pick.", opts).value();
  auto whole = sq::BuildHead(*tok, q, 10000).value();
  const int64_t spans = int64_t(whole.ids.size()) - 1 - whole.markers[0];
  auto at = sq::BuildHead(*tok, q, spans + sq::kMinOptionBudget).value();
  EXPECT_FALSE(at.stats.tokens_per_option.has_value());
  EXPECT_EQ(at.ids.size(), whole.ids.size());
  auto below = sq::BuildHead(*tok, q, spans + sq::kMinOptionBudget - 1).value();
  EXPECT_TRUE(below.stats.tokens_per_option.has_value());
}

// A list keeps its end when it is too long; a string or object keeps its start.
TEST(EncodeState, KeepSide) {
  LAYA_CHECKPOINT_TOK(tok, "english");
  Value list = Value::Array();
  list.Push(Value::String("x"));
  Value obj = Value::Object();
  obj.Set("k", Value::String("v"));
  EXPECT_EQ(sq::EncodeState(*tok, list).value().keep, sq::Keep::End);
  EXPECT_EQ(sq::EncodeState(*tok, obj).value().keep, sq::Keep::Start);
  EXPECT_EQ(sq::EncodeState(*tok, Value::String("x")).value().keep, sq::Keep::Start);
  EXPECT_EQ(sq::EncodeState(*tok, Value::Int(5)).value().keep, sq::Keep::Start);
}

// A default window of exactly twice the room is the ratio branch of the clamp warning.
TEST(PlanWindows, ClampRatioBoundary) {
  LAYA_CHECKPOINT_TOK(tok, "english");
  bool found = false;
  for (int n = 1; n <= 40 && !found; n++) {
    auto q = WindowQuestion(n);
    for (int64_t head = 32; head <= 192 && !found; head += 8) {
      // room = max_len - len(head) - 1 and the default window = max_len - head_max_len - 8, so
      // default == 2 * room at max_len = 2 * len(head) - head_max_len - 6.
      const int64_t len = int64_t(sq::BuildHead(*tok, q, head).value().ids.size());
      sq::Budget b{2 * len - head - 6, head};
      if (b.max_len < 1 || b.DefaultWindow() <= 64) continue;
      auto room = sq::StateRoom(*tok, q, b);
      if (!room.ok() || room.value() <= 0 || b.DefaultWindow() != 2 * room.value()) continue;
      found = true;
      auto plan = sq::PlanWindows(*tok, {q}, b, {});
      ASSERT_TRUE(plan.ok());
      ASSERT_EQ(plan.value().warnings.size(), 1u);
      EXPECT_NE(plan.value().warnings[0].find("is capped"), std::string::npos) << plan.value().warnings[0];
      EXPECT_EQ(plan.value().window, room.value());
    }
  }
  EXPECT_TRUE(found) << "no budget with a default window of twice the room";
}

// snprintf follows the process locale: under de_DE the ratio read "2,5x".
TEST(PlanWindows, ClampRatioIgnoresACommaDecimalLocale) {
  LAYA_CHECKPOINT_TOK(tok, "english");
  laya_test::CommaLocale comma;
  if (!comma.ok()) GTEST_SKIP() << "no comma-decimal locale installed";
  bool found = false;
  for (int n = 1; n <= 40 && !found; n++) {
    auto q = WindowQuestion(n);
    for (int64_t head = 32; head <= 192 && !found; head += 8) {
      // room = max_len - len(head) - 1 and the default window = max_len - head_max_len - 8, so the
      // default is 2.5 * room at 3 * max_len = 5 * len(head) - 2 * head_max_len - 11.
      const int64_t len = int64_t(sq::BuildHead(*tok, q, head).value().ids.size());
      const int64_t top = 5 * len - 2 * head - 11;
      if (top <= 0 || top % 3 != 0) continue;
      sq::Budget b{top / 3, head};
      auto room = sq::StateRoom(*tok, q, b);
      if (!room.ok() || room.value() <= 0 || 2 * b.DefaultWindow() != 5 * room.value()) continue;
      auto plan = sq::PlanWindows(*tok, {q}, b, {});
      if (!plan.ok() || plan.value().warnings.size() != 1) continue;
      found = true;
      EXPECT_NE(plan.value().warnings[0].find("roughly 2.5x as many windows"), std::string::npos) << plan.value().warnings[0];
    }
  }
  EXPECT_TRUE(found) << "no budget with a default window of 2.5 times the room";
}

TEST(BatchCap, MatchesPythonGrid) {
  LAYA_GOLDEN_LIST(grid, "window_plan.json", "batch_cap");
  LAYA_EXPECT_ROWS(grid, 810);
  int capped = 0;
  for (size_t i = 0; i < grid->size(); i++) {
    const Value& rec = grid->value(i);
    const int64_t n = rec.Find("n_windows")->as_int(), w = rec.Find("window")->as_int(), cfg = rec.Find("config_budget")->as_int();
    const Value* bs = rec.Find("batch_size");
    const int64_t batch = bs->is_null() ? 0 : bs->as_int();
    const Value* want = rec.Find("cap");
    sq::Cap got = sq::BatchCap(n, w, cfg, batch);
    EXPECT_EQ(got.ok, !want->is_null()) << "row " << i;
    if (!want->is_null()) {
      EXPECT_EQ(got.cap, want->as_int()) << "row " << i;
    }
    if (!want->is_null() && batch <= 0) capped++;
  }
  EXPECT_GT(capped, 0);
}

TEST(BatchCap, HugeInputsDoNotOverflow) {
  const int64_t big = INT64_MAX;
  sq::Cap c = sq::BatchCap(big, big - 1, big, 0);  // a one-token-narrower window: a mild cap, so none
  EXPECT_FALSE(c.ok);
  sq::Cap d = sq::BatchCap(big, 2, big, 0);
  EXPECT_TRUE(d.ok);
  EXPECT_LT(d.cap, big);
}

TEST(Budget, RefusesNonPositive) {
  LAYA_CHECKPOINT_TOK(tok, "english");
  auto q = sc::Question::Noul("Holds?").value();
  EXPECT_FALSE(sq::StateRoom(*tok, q, {0, 8}).ok());
  EXPECT_FALSE(sq::StateRoom(*tok, q, {512, -1}).ok());
  EXPECT_FALSE(sq::Build(*tok, q, {}, {0, 0}).ok());
  EXPECT_FALSE(sq::BuildHead(*tok, sc::Question(), 64).ok());
  EXPECT_EQ((sq::Budget{512, 192}).DefaultWindow(), 312);
  EXPECT_EQ((sq::Budget{100, 192}).DefaultWindow(), 64);
}

// States: json.loads then serialize_state, on 2,000 random JSON texts.
class StateRow : public ::testing::TestWithParam<int> {};
TEST_P(StateRow, SerializeMatchesPython) {
  LAYA_GOLDEN_LIST(rows, "go_extra.json", "states", "cases");
  LAYA_EXPECT_ROWS(rows, 2000);
  for (size_t i = size_t(GetParam()) * laya_test::kChunk; i < rows->size() && i < size_t(GetParam() + 1) * laya_test::kChunk; i++) {
    const Value& rec = rows->value(i);
    const std::string text = rec.Find("text")->as_string();
    auto v = laya::pyvalue::Parse(text);
    ASSERT_TRUE(v.ok()) << "state " << i << ": " << v.error();
    auto s = sq::SerializeState(v.value());
    ASSERT_TRUE(s.ok()) << "state " << i << ": " << s.error();
    EXPECT_EQ(s.value(), rec.Find("serialized")->as_string()) << "state " << i;
    EXPECT_EQ(v.value().kind() == Kind::Array, rec.Find("list")->as_bool()) << "state " << i;
  }
}
INSTANTIATE_TEST_SUITE_P(Golden, StateRow, ::testing::ValuesIn(laya_test::Chunks(2000)));

TEST(States, PythonRejects) {
  LAYA_GOLDEN_LIST(rejects, "go_extra.json", "states", "python_rejects");
  for (const Value& t : rejects->items()) {
    EXPECT_FALSE(laya::pyvalue::Parse(t.as_string()).ok()) << "accepted " << t.as_string().substr(0, 60);
  }
}

}  // namespace
