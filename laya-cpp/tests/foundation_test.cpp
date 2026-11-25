#include <cstring>

#include "golden.hpp"
#include "laya/pyjson.hpp"
#include "laya/pyvalue.hpp"
#include "laya/sha256.hpp"
#include "laya/unicodedata.hpp"

using laya_test::At;
using laya_test::Value;
namespace ud = laya::unicodedata;
namespace pv = laya::pyvalue;

// ---- sha256: FIPS 180-4 / NIST vectors and Python hashlib goldens --------------------------------
TEST(Sha256, FipsAndHashlibVectors) {
  EXPECT_EQ(laya::Sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(laya::Sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(laya::Sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(laya::Sha256Hex(std::string(1000000, 'a')),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  EXPECT_EQ(laya::Sha256Hex(std::string(55, 'a')),  // padding edge: 55, 56, 63, 64 byte messages
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
  EXPECT_EQ(laya::Sha256Hex(std::string(56, 'a')),
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
  EXPECT_EQ(laya::Sha256Hex(std::string(63, 'a')),
            "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34");
  EXPECT_EQ(laya::Sha256Hex(std::string(64, 'a')),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
}

TEST(Sha256, IncrementalEqualsOneShot) {
  std::string data(777, 'x');
  for (size_t i = 0; i < data.size(); i++) data[i] = char(i * 31);
  laya::Sha256 h;
  for (size_t i = 0; i < data.size(); i += 13) h.Update(data.data() + i, std::min<size_t>(13, data.size() - i));
  EXPECT_EQ(laya::ToHex(h.Finish()), laya::Sha256Hex(data));
}

// ---- unicodedata: whole-Unicode digests against CPython ------------------------------------------
namespace {
std::string PredicateDigest(bool (*pred)(char32_t)) {
  std::string bits;
  for (char32_t r = 0; r < 0x110000; r++) bits.push_back(pred(r) ? '1' : '0');
  return laya::Sha256Hex(bits);
}
}  // namespace

TEST(Unicode, VersionAndPredicatesMatchCPythonEverywhere) {
  LAYA_GOLDEN_LIST(d, "lang_detect.json", "unicode_digests");
  EXPECT_STREQ(ud::kVersion, At(laya_test::Load("lang_detect.json", true), {"unicode_version"})->as_string().c_str());
  struct P { const char* name; bool (*fn)(char32_t); };
  const P preds[] = {{"alpha", ud::IsAlpha}, {"word", ud::IsWord},   {"combining", ud::IsCombining},
                     {"digit", ud::IsDigit}, {"upper", ud::IsUpper}, {"lower", ud::IsLower},
                     {"space", ud::IsSpace}, {"mark", ud::IsMark},   {"initial", ud::IsInitial}};
  for (const P& p : preds) {
    ASSERT_NE(d->Find(p.name), nullptr) << p.name;
    EXPECT_EQ(PredicateDigest(p.fn), d->Find(p.name)->as_string()) << p.name;
  }
}

TEST(Unicode, LowerMatchesCPythonEverywhere) {
  LAYA_GOLDEN_LIST(d, "lang_detect.json", "unicode_digests", "python_lower");
  laya::Sha256 h;
  for (char32_t r = 0; r < 0x110000; r++) {
    if (r >= 0xD800 && r <= 0xDFFF) continue;
    std::string one;
    ud::AppendRune(one, r);
    std::string low = ud::Lower(one), line = std::to_string(r) + ":";
    for (size_t i = 0; i < low.size();) {
      ud::Decoded x = ud::DecodeRune(std::string_view(low).substr(i));
      if (i > 0) line += ",";
      line += std::to_string(x.rune);
      i += x.width;
    }
    line += ";";
    h.Update(line);
  }
  EXPECT_EQ(laya::ToHex(h.Finish()), d->as_string());
}

TEST(Unicode, FinalSigmaAndPythonLowerRows) {
  LAYA_GOLDEN_LIST(rows, "go_extra.json", "python_lower", "cases");
  for (size_t i = 0; i < rows->size(); i++) {
    const Value& c = rows->value(i);
    EXPECT_EQ(ud::Lower(c.Find("text")->as_string()), c.Find("lower")->as_string()) << "row " << i;
  }
}

TEST(Unicode, InvalidUtf8MatchesPython) {
  LAYA_GOLDEN_LIST(fam, "go_extra.json", "invalid_utf8");
  const Value& hex = *fam->Find("hex");
  const Value& decoded = *fam->Find("decoded");
  ASSERT_EQ(hex.size(), 3000u);
  for (size_t i = 0; i < hex.size(); i++) {
    const std::string& h = hex.value(i).as_string();
    std::string bytes;
    for (size_t k = 0; k + 1 < h.size(); k += 2) bytes.push_back(char(std::stoi(h.substr(k, 2), nullptr, 16)));
    EXPECT_EQ(ud::Lossy(bytes), decoded.value(i).as_string()) << "row " << i << " " << h;
  }
}

TEST(Unicode, StripFieldsDecimal) {
  EXPECT_EQ(ud::Strip("  a b\t\xe3\x80\x80"), "a b");
  EXPECT_EQ(ud::StripNumeric("\x1f 12 "), "\x1f 12");
  EXPECT_EQ(ud::Fields(" a  b\n").size(), 2u);
  EXPECT_EQ(ud::DecimalASCII("\xd9\xa1\xd9\xa2"), "12");
  EXPECT_TRUE(ud::IsBlank(" \t"));
  EXPECT_TRUE(ud::IsUpperString("AB1"));
  EXPECT_FALSE(ud::IsUpperString("Ab"));
}

// ---- pyjson: float repr, str repr, percent, round4 -----------------------------------------------
class DoublesChunk : public ::testing::TestWithParam<int> {};
TEST_P(DoublesChunk, ReprFloatMatchesCPython) {
  LAYA_GOLDEN_LIST(rows, "python_json.json", "doubles");
  LAYA_EXPECT_ROWS(rows, 533);
  for (size_t i = size_t(GetParam()) * laya_test::kChunk; i < std::min(rows->size(), (size_t(GetParam()) + 1) * laya_test::kChunk); i++) {
    uint64_t bits = laya_test::BitsOf(*rows->value(i).Find("bits"));
    double v;
    std::memcpy(&v, &bits, 8);
    EXPECT_EQ(laya::pyjson::ReprFloat(v), rows->value(i).Find("repr")->as_string()) << "bits " << bits;
  }
}
INSTANTIATE_TEST_SUITE_P(Golden, DoublesChunk, ::testing::ValuesIn(laya_test::Chunks(533)));

TEST(PyJson, ReprFloatNotationBoundaries) {  // fixed for exponents in [-4, 16), scientific outside
  EXPECT_EQ(laya::pyjson::ReprFloat(1e15), "1000000000000000.0");
  EXPECT_EQ(laya::pyjson::ReprFloat(1e16), "1e+16");
  EXPECT_EQ(laya::pyjson::ReprFloat(0.0001), "0.0001");
  EXPECT_EQ(laya::pyjson::ReprFloat(0.00001), "1e-05");
  EXPECT_EQ(laya::pyjson::ReprFloat(1.5e300), "1.5e+300");
  EXPECT_EQ(laya::pyjson::ReprFloat(0.30000000000000004), "0.30000000000000004");
}

TEST(PyJson, ReprStringsDigestPercentRound4) {
  LAYA_GOLDEN_LIST(rs, "python_json.json", "repr_strings");
  for (size_t i = 0; i < rs->size(); i++) {
    EXPECT_EQ(laya::pyjson::Repr(rs->value(i).value(0).as_string()), rs->value(i).value(1).as_string());
  }
  laya::Sha256 h;  // Repr("a"+chr(r)+"b") over every non-surrogate code point
  for (char32_t r = 0; r < 0x110000; r++) {
    if (r >= 0xD800 && r <= 0xDFFF) continue;
    std::string s = "a";
    ud::AppendRune(s, r);
    s += "b";
    h.Update(laya::pyjson::Repr(s));
  }
  EXPECT_EQ(laya::ToHex(h.Finish()), At(laya_test::Load("python_json.json", true), {"repr_digest"})->as_string());
  const Value* pc = At(laya_test::Load("python_json.json", true), {"percent0"});
  for (size_t i = 0; i < pc->size(); i++) {
    EXPECT_EQ(laya::pyjson::Percent0(laya_test::Num(pc->value(i).value(0))), pc->value(i).value(1).as_string());
  }
  const Value* groups = At(laya_test::Load("python_json.json", true), {"round4"});
  ASSERT_NE(groups->Find("halfway"), nullptr);
  for (size_t g = 0; g < groups->size(); g++) {
    const Value& rows = groups->value(g);
    for (size_t i = 0; i < rows.size(); i++) {
      uint64_t bits = laya_test::BitsOf(*rows.value(i).Find("bits"));
      double v;
      std::memcpy(&v, &bits, 8);
      EXPECT_EQ(laya::pyjson::Round4(v), laya_test::Num(*rows.value(i).Find("r4"))) << groups->key(g) << " " << v;
    }
  }
}

class Round4Chunk : public ::testing::TestWithParam<int> {};
TEST_P(Round4Chunk, HalfEvenMatchesPython) {
  LAYA_GOLDEN_LIST(vals, "go_extra.json", "round4", "values");
  LAYA_EXPECT_ROWS(vals, 5605);
  const Value* want = At(laya_test::Load("go_extra.json", true), {"round4", "rounded"});
  for (size_t i = size_t(GetParam()) * laya_test::kChunk; i < std::min(vals->size(), (size_t(GetParam()) + 1) * laya_test::kChunk); i++) {
    double got = laya::pyjson::Round4(laya_test::Num(vals->value(i))), w = laya_test::Num(want->value(i));
    EXPECT_EQ(std::memcmp(&got, &w, 8), 0) << "row " << i << " " << got << " vs " << w;
  }
}
INSTANTIATE_TEST_SUITE_P(Golden, Round4Chunk, ::testing::ValuesIn(laya_test::Chunks(5605)));

// ---- pyvalue: json.loads / json.dumps ------------------------------------------------------------
class StatesChunk : public ::testing::TestWithParam<int> {};
TEST_P(StatesChunk, ParseAndDumpsMatchPython) {
  LAYA_GOLDEN_LIST(cases, "go_extra.json", "states", "cases");
  LAYA_EXPECT_ROWS(cases, 2000);
  for (size_t i = size_t(GetParam()) * laya_test::kChunk; i < std::min(cases->size(), (size_t(GetParam()) + 1) * laya_test::kChunk); i++) {
    const std::string& text = cases->value(i).Find("text")->as_string();
    auto v = pv::Parse(text);
    ASSERT_TRUE(v.ok()) << i << " " << v.error();
    std::string got;
    if (v.value().kind() == pv::Value::Kind::String) {
      got = v.value().as_string();
    } else {
      auto d = pv::Dumps(v.value());
      ASSERT_TRUE(d.ok()) << i << " " << d.error();
      got = d.value();
    }
    EXPECT_EQ(got, cases->value(i).Find("serialized")->as_string()) << "row " << i;
    EXPECT_EQ(v.value().kind() == pv::Value::Kind::Array, cases->value(i).Find("list")->as_bool()) << i;
  }
}
INSTANTIATE_TEST_SUITE_P(Golden, StatesChunk, ::testing::ValuesIn(laya_test::Chunks(2000)));

TEST(PyValue, PythonRejectsAndLoneSurrogates) {
  LAYA_GOLDEN_LIST(bad, "go_extra.json", "states", "python_rejects");
  for (size_t i = 0; i < bad->size(); i++) EXPECT_FALSE(pv::Parse(bad->value(i).as_string()).ok()) << bad->value(i).as_string().substr(0, 40);
  const Value* lone = At(laya_test::Load("go_extra.json", true), {"states", "go_refuses"});
  for (size_t i = 0; i < lone->size(); i++) {
    auto r = pv::Parse(lone->value(i).as_string());
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().find("lone surrogate"), std::string::npos) << r.error();
  }
}

TEST(PyValue, DuplicateKeysNaNDepthBigIntRepr) {
  auto d = pv::Parse("{\"a\": 1, \"b\": 2, \"a\": 3}");
  ASSERT_TRUE(d.ok());
  EXPECT_EQ(pv::Dumps(d.value()).value(), "{\"a\": 3, \"b\": 2}");  // first position, last value
  auto n = pv::Parse("[NaN, Infinity, -Infinity, 2, 2.0, 1e400, -0, 12345678901234567890123]");
  ASSERT_TRUE(n.ok());
  EXPECT_EQ(pv::Dumps(n.value()).value(), "[NaN, Infinity, -Infinity, 2, 2.0, Infinity, 0, 12345678901234567890123]");
  EXPECT_EQ(pv::Repr(n.value()), "[nan, inf, -inf, 2, 2.0, inf, 0, 12345678901234567890123]");
  std::string deep(600, '[');
  deep += std::string(600, ']');
  EXPECT_FALSE(pv::Parse(deep).ok());
  // The outermost value is depth 0, so kMaxDepth + 1 brackets parse and kMaxDepth + 2 do not.
  std::string ok(pv::kMaxDepth + 1, '[');
  ok += std::string(pv::kMaxDepth + 1, ']');
  EXPECT_TRUE(pv::Parse(ok).ok());
  std::string over(pv::kMaxDepth + 2, '[');
  over += std::string(pv::kMaxDepth + 2, ']');
  EXPECT_FALSE(pv::Parse(over).ok());
  EXPECT_FALSE(pv::Parse(std::string(4301, '9')).ok());
  EXPECT_TRUE(pv::Parse(std::string(4300, '9')).ok());
  EXPECT_FALSE(pv::Parse("\"\\ud800\"").ok());
  EXPECT_FALSE(pv::Parse("\xff").ok());
  auto o = pv::Parse("{\"k\": [\"it's\", null, true]}");
  EXPECT_EQ(pv::Repr(o.value()), "{'k': [\"it's\", None, True]}");
  EXPECT_EQ(pv::TypeName(o.value()), "dict");
}
