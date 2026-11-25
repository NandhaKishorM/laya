#include <cmath>

#include "golden.hpp"
#include "laya/lang.hpp"
#include "laya/sha256.hpp"
#include "laya/unicodedata.hpp"

using laya_test::At;
using laya_test::Value;
namespace lg = laya::lang;
namespace pv = laya::pyvalue;

namespace {
// The state a golden row describes.
pv::Value StateOf(const Value& enc) {
  const std::string& kind = enc.Find("kind")->as_string();
  const Value& v = *enc.Find("value");
  if (kind == "bytes") {
    std::string b;
    for (size_t i = 0; i < v.size(); i++) b.push_back(char(v.value(i).as_int()));
    return laya::unicodedata::ValidUtf8(b) ? pv::Value::String(b) : pv::Value::Null();  // undecodable bytes read as nothing
  }
  return v;
}
void ExpectProfile(const Value& want, const lg::Profile& got, const std::string& what) {
  ASSERT_EQ(want.size(), got.size()) << what;
  for (size_t i = 0; i < got.size(); i++) {
    EXPECT_EQ(want.key(i), got[i].first) << what;
    EXPECT_EQ(laya_test::Num(want.value(i)), got[i].second) << what << " " << got[i].first;
  }
}
}  // namespace

class DetectRow : public ::testing::TestWithParam<int> {};
TEST_P(DetectRow, MatchesPython) {
  LAYA_GOLDEN_LIST(rows, "lang_detect.json", "cases");
  LAYA_EXPECT_ROWS(rows, 106);
  const Value& row = rows->value(size_t(GetParam()));
  const std::string name = row.Find("name")->as_string();
  pv::Value state = StateOf(*row.Find("state"));
  const std::string text = row.Find("state_text")->as_string();
  EXPECT_EQ(lg::StateText(state), text) << name;
  EXPECT_EQ(lg::DetectScript(text), row.Find("detect_script")->as_string()) << name;
  ExpectProfile(*row.Find("script_profile"), lg::ScriptProfile(text), name);
  const Value& lp = *row.Find("latin_profile");
  lg::LatinProfile got = lg::LatinProfileOf(text);
  auto wl = lp.Find("language");
  EXPECT_EQ(got.language, wl->is_null() ? "" : wl->as_string()) << name;
  EXPECT_EQ(got.english_hits, lp.Find("english_hits")->as_int()) << name;
  EXPECT_EQ(got.diacritic_rate, laya_test::Num(*lp.Find("diacritic_rate"))) << name;
  EXPECT_EQ(got.looks_non_english, lp.Find("looks_non_english")->as_bool()) << name;
  const Value& an = *row.Find("analyse");
  lg::Analysis a = lg::Analyze(state);
  EXPECT_EQ(a.script, an.Find("script")->as_string()) << name;
  ExpectProfile(*an.Find("script_profile"), a.script_profile, name);
  auto al = an.Find("language");
  EXPECT_EQ(a.language, al->is_null() ? "" : al->as_string()) << name;
  EXPECT_EQ(a.english, an.Find("is_english")->as_bool()) << name;
  EXPECT_EQ(a.diacritic_rate, laya_test::Num(*an.Find("diacritic_rate"))) << name;
  EXPECT_EQ(a.non_latin_fraction, laya_test::Num(*an.Find("non_latin_fraction"))) << name;
  auto ms = an.Find("mixed_segment");
  EXPECT_EQ(a.mixed_segment, ms->is_null() ? "" : ms->as_string()) << name;
  EXPECT_EQ(lg::IsEnglish(state), row.Find("is_english")->as_bool()) << name;
  EXPECT_EQ(lg::DetectLanguage(state), a.language) << name;
}
INSTANTIATE_TEST_SUITE_P(Golden, DetectRow, ::testing::ValuesIn(laya_test::Rows(106)));

TEST(Lang, ScriptOfMatchesPythonEverywhere) {
  LAYA_GOLDEN_LIST(want, "lang_detect.json", "unicode_digests", "script_of");
  laya::Sha256 h;
  for (char32_t r = 0; r < 0x110000; r++) h.Update(lg::ScriptOf(r) + ";");
  EXPECT_EQ(laya::ToHex(h.Finish()), want->as_string());
}

TEST(Lang, StateShapesMatchPython) {
  LAYA_GOLDEN_LIST(shapes, "go_extra.json", "state_shapes");
  const std::string german = shapes->Find("string")->Find("state_text")->as_string();
  auto S = [](const std::string& s) { return pv::Value::String(s); };
  auto L = [](std::vector<pv::Value> v) { return pv::Value::Array(std::move(v)); };
  auto O = [](const char* k, pv::Value v) {
    pv::Value o = pv::Value::Object();
    o.Set(k, std::move(v));
    return o;
  };
  pv::Value seven = S(german), six = S(german);
  for (int i = 0; i < 7; i++) seven = L({seven});
  for (int i = 0; i < 6; i++) six = L({six});
  std::vector<std::pair<std::string, pv::Value>> built = {
      {"string", S(german)},
      {"map_of_slices", O("a", L({S(german)}))},
      {"slice_of_maps", L({O("a", S(german))})},
      {"map_of_objects", O("a", O("b", S(german)))},
      {"slice_of_slices", L({L({S(german)})})},
      {"tuple", L({S(german)})},
      {"seven_deep", seven},
      {"six_deep", six},
      {"numbers_and_none", L({pv::Value::Int(1), pv::Value::Float(2.5), pv::Value::Null(), pv::Value::Bool(true)})},
      {"mixed", O("k", L({S(german), pv::Value::Int(3), O("x", pv::Value::Null())}))},
      {"bytes", S(german)},
      {"invalid_bytes", pv::Value::Null()}};
  for (const auto& b : built) {
    const Value* want = shapes->Find(b.first);
    ASSERT_NE(want, nullptr) << b.first;
    lg::Analysis a = lg::Analyze(b.second);
    EXPECT_EQ(lg::StateText(b.second), want->Find("state_text")->as_string()) << b.first;
    EXPECT_EQ(a.english, want->Find("is_english")->as_bool()) << b.first;
    auto wl = want->Find("language");
    EXPECT_EQ(a.language, wl->is_null() ? "" : wl->as_string()) << b.first;
    EXPECT_EQ(a.script, want->Find("script")->as_string()) << b.first;
  }
}

TEST(Lang, InvalidUtf8MatchesPython) {
  LAYA_GOLDEN_LIST(fam, "go_extra.json", "invalid_utf8");
  const Value& hex = *fam->Find("hex");
  const Value& script = *fam->Find("script");
  const Value& text = *fam->Find("state_text");
  ASSERT_EQ(hex.size(), 3000u);
  for (size_t i = 0; i < hex.size(); i++) {
    const std::string& h = hex.value(i).as_string();
    std::string bytes;
    for (size_t k = 0; k + 1 < h.size(); k += 2) bytes.push_back(char(std::stoi(h.substr(k, 2), nullptr, 16)));
    EXPECT_EQ(lg::DetectScript(bytes), script.value(i).as_string()) << "row " << i;
    EXPECT_EQ(lg::StateText(pv::Value::String(bytes)), text.value(i).as_string()) << "row " << i;
  }
}

// Boundary cases (ties, exact thresholds, 3 vs 4 words, the Latin range edge) with expectations
// recorded from laya.lang by laya-e2e/cpp-lane/fuzz-c1/gen_boundary.py. They are source text, not a
// golden, so they run in every build and pin the mutants the parity rows alone do not.
TEST(Lang, BoundaryCasesMatchPython) {
  struct Row { const char* text; const char* script; const char* language; bool english; double rate; double nl; };
  static const Row rows[] = {
#include "lang_boundary.inc"
  };
  for (const Row& r : rows) {
    lg::Analysis a = lg::Analyze(std::string_view(r.text));
    EXPECT_EQ(a.script, r.script) << r.text;
    EXPECT_EQ(a.language, r.language) << r.text;
    EXPECT_EQ(a.english, r.english) << r.text;
    EXPECT_EQ(a.diacritic_rate, r.rate) << r.text;
    EXPECT_EQ(a.non_latin_fraction, r.nl) << r.text;
  }
}
