#include <gtest/gtest-spi.h>

#include <chrono>
#include <clocale>
#include <cstdlib>

#include "golden.hpp"
#include "laya/pyjson.hpp"
#include "laya/pyvalue.hpp"
#include "laya/sha256.hpp"
#include "laya/unicodedata.hpp"

namespace ud = laya::unicodedata;
namespace pv = laya::pyvalue;

// ---- all-code-point digests recorded from CPython (fuzz-c1/gen_unicode_digests.py) -------------
namespace {
struct DigestRow { const char* name; const char* sha; };
const DigestRow kDigests[] = {
#include "unicode_digests.inc"
};
const char* Want(const std::string& name) {
  for (const auto& d : kDigests) {
    if (name == d.name) return d.sha;
  }
  return "";
}
template <class F>
std::string Digest(F fn) {
  laya::Sha256 h;
  for (char32_t cp = 0; cp < 0x110000; cp++) {
    if (cp >= 0xD800 && cp <= 0xDFFF) continue;
    h.Update(fn(cp));
  }
  return laya::ToHex(h.Finish());
}
std::string One(char32_t cp) {
  std::string s;
  ud::AppendRune(s, cp);
  return s;
}
}  // namespace

TEST(UnicodeDigests, PrintableTitleSigmaFieldsDecimal) {
  EXPECT_EQ(Digest([](char32_t c) { return std::string(ud::IsPrintable(c) ? "1" : "0"); }), Want("printable"));
  // IsCased (upper, lower or titlecase) decides whether "A"+ch is all uppercase.
  EXPECT_EQ(Digest([](char32_t c) { return std::string(ud::IsUpperString("A" + One(c)) ? "1" : "0"); }), Want("title_cased"));
  // The final-sigma context sets (cased, case-ignorable), probed exactly as gen_tables.py does.
  EXPECT_EQ(Digest([](char32_t c) {
              std::string l = ud::Lower("A" + One(c) + "\xCE\xA3");
              return std::string(l.compare(l.size() - 2, 2, "\xCF\x82") == 0 ? "1" : "0");
            }), Want("sigma_final"));
  EXPECT_EQ(Digest([](char32_t c) {
              std::string l = ud::Lower("A\xCE\xA3" + One(c));
              return std::string(l.compare(1, 2, "\xCF\x82") == 0 ? "1" : "0");
            }), Want("sigma_next"));
  EXPECT_EQ(Digest([](char32_t c) { return std::to_string(ud::Fields("a" + One(c) + "b").size()) + ";"; }), Want("fields"));
  EXPECT_EQ(Digest([](char32_t c) {
              std::string r = ud::DecimalASCII(One(c));
              return ud::IsDigit(c) ? r : std::string("-");  // an Nd digit becomes its ASCII digit
            }), Want("decimal"));
}

TEST(UnicodeDigests, PythonBoundaryRows) {
  EXPECT_EQ(ud::Fields("\n\xF0\x9F\x98\x80\x1c").size(), 1u);   // "\n😀\x1c".split(): U+001C is whitespace
  EXPECT_EQ(ud::Fields("a\xE2\x80\xA8" "b\xC2\x85" "c").size(), 3u);
  EXPECT_FALSE(ud::IsUpperString("A\xC7\x85"));                  // "Aǅ".isupper() is False (titlecase)
  EXPECT_EQ(ud::DecimalASCII("\xF0\x9D\x9F\x8E\xF0\x9D\x9F\xBF"), "09");  // U+1D7CE..U+1D7FF cycle 0-9
  EXPECT_EQ(ud::DecimalASCII("\xF0\x9D\x9F\x98\xF0\x9D\x9F\x9A"), "02");  // U+1D7D8 starts the next cycle
}

// ---- locale independence -------------------------------------------------------------------
TEST(Locale, NumbersIgnoreACommaDecimalLocale) {
  const char* candidates[] = {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8", "de_DE", "de_DE.ISO8859-1"};
  std::string saved = std::setlocale(LC_ALL, nullptr);
  bool set = false;
  for (const char* c : candidates) {
    if (std::setlocale(LC_ALL, c) != nullptr && *std::localeconv()->decimal_point == ',') {
      set = true;
      break;
    }
  }
  if (!set) {
    std::setlocale(LC_ALL, saved.c_str());
    GTEST_SKIP() << "no comma-decimal locale installed";
  }
  auto v = pv::Parse("[1.5, 0.30000000000000004, 1e-7, 123456.789]");
  std::string dumped = v.ok() ? pv::Dumps(v.value()).value() : "";
  std::string r4 = laya::pyjson::ReprFloat(laya::pyjson::Round4(0.12345));
  std::string pct = laya::pyjson::Percent0(0.125);
  double r4b = laya::pyjson::Round4(2.00015);
  std::setlocale(LC_ALL, saved.c_str());
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(dumped, "[1.5, 0.30000000000000004, 1e-07, 123456.789]");
  EXPECT_EQ(r4, "0.1235");
  EXPECT_EQ(pct, "12");  // 12.5 rounds half to even
  EXPECT_EQ(r4b, 2.0002);
}

// ---- sha256 after Finish -------------------------------------------------------------------
TEST(Sha256, FinishIsIdempotentAndUpdateAfterFinishIsRefused) {
  laya::Sha256 h;
  EXPECT_TRUE(h.Update("abc"));
  auto first = h.Finish();
  EXPECT_EQ(laya::ToHex(first), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(h.Finish(), first);  // not a digest of the padding
  EXPECT_FALSE(h.Update("d"));
  EXPECT_EQ(h.Finish(), first);
}

// ---- Value::Set / Find scale ---------------------------------------------------------------
TEST(PyValue, ObjectsWithManyKeysParseInLinearTime) {
  const int n = 200000;
  std::string text = "{";
  for (int i = 0; i < n; i++) text += (i ? ", \"k" : "\"k") + std::to_string(i) + "\": " + std::to_string(i);
  for (int i = 0; i < n; i++) text += ", \"k" + std::to_string(i % 1000) + "\": 0";  // repeated keys
  text += "}";
  auto t0 = std::chrono::steady_clock::now();
  auto v = pv::Parse(text);
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v.value().size(), size_t(n));
  ASSERT_NE(v.value().Find("k199999"), nullptr);
  EXPECT_EQ(v.value().Find("k5")->as_int(), 0);       // last value wins
  EXPECT_EQ(v.value().key(5), "k5");                  // first position kept
  EXPECT_LT(secs, 10.0) << "quadratic key handling";  // linear takes well under a second
}

// ---- a missing golden key fails under LAYA_REQUIRE_GOLDENS=1 -----------------------------------
namespace {
void UsesMissingKey() { LAYA_GOLDEN_LIST(x, "lang_detect.json", "no_such_key"); (void)x; }
}  // namespace

TEST(Golden, MissingKeyFailsWhenRequired) {
  const char* dir = std::getenv("LAYA_GOLDENS");
  if (dir == nullptr || !*dir) {
    if (laya_test::RequireGoldens()) FAIL() << "LAYA_GOLDENS unset";
    GTEST_SKIP() << "LAYA_GOLDENS unset (local run)";
  }
  const char* old = std::getenv("LAYA_REQUIRE_GOLDENS");
  std::string saved = old ? old : "";
  setenv("LAYA_REQUIRE_GOLDENS", "1", 1);
  EXPECT_FATAL_FAILURE(UsesMissingKey(), "golden key path missing");
  if (old) setenv("LAYA_REQUIRE_GOLDENS", saved.c_str(), 1);
  else unsetenv("LAYA_REQUIRE_GOLDENS");
}
