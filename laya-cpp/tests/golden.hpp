#pragma once
// Golden loading for the parity tests. The goldens are regenerated from the Python reference at
// the commit under test (CI) or by the lane script (local) into $LAYA_GOLDENS; none is committed.
// With LAYA_REQUIRE_GOLDENS=1 (CI) a missing golden is a test FAILURE, never a skip.
#include <gtest/gtest.h>

#include <clocale>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "laya/pyvalue.hpp"

namespace laya_test {

using laya::pyvalue::Value;

// The parsed golden file, or nullptr. report=false is for instantiation time, outside any test.
inline const Value* Load(const std::string& file, bool report) {
  static std::map<std::string, Value> cache;
  auto it = cache.find(file);
  if (it != cache.end()) return &it->second;
  const char* dir = std::getenv("LAYA_GOLDENS");
  const char* require = std::getenv("LAYA_REQUIRE_GOLDENS");
  if (dir == nullptr || *dir == 0) {
    if (report && require != nullptr && std::string(require) == "1") {
      ADD_FAILURE() << "LAYA_GOLDENS is unset but LAYA_REQUIRE_GOLDENS=1";
    }
    return nullptr;
  }
  std::ifstream in(std::string(dir) + "/" + file, std::ios::binary);
  if (!in) {
    if (report) ADD_FAILURE() << "cannot open golden " << dir << "/" << file;
    return nullptr;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  auto parsed = laya::pyvalue::Parse(ss.str());
  if (!parsed.ok()) {
    if (report) ADD_FAILURE() << file << ": " << parsed.error();
    return nullptr;
  }
  return &(cache[file] = std::move(parsed).value());
}

#define LAYA_EXPECT_ROWS(list, n) \
  ASSERT_EQ((list)->size(), size_t(n)) << "the golden changed shape; update the row count in the test"

// Walks a path of object keys; nullptr when anything is missing.
inline const Value* At(const Value* v, std::initializer_list<const char*> path) {
  for (const char* k : path) {
    if (v == nullptr) return nullptr;
    v = v->Find(k);
  }
  return v;
}

inline bool RequireGoldens() {
  const char* r = std::getenv("LAYA_REQUIRE_GOLDENS");
  return r != nullptr && std::string(r) == "1";
}

// A test body's view of one golden list. Absent goldens skip locally; with LAYA_REQUIRE_GOLDENS=1 a
// missing directory, file or key path FAILS, so a renamed key cannot turn a parity test into a skip.
#define LAYA_GOLDEN_LIST(var, file, ...)                                                          \
  const laya::pyvalue::Value* var = laya_test::At(laya_test::Load(file, true), {__VA_ARGS__});    \
  if (var == nullptr) {                                                                           \
    if (laya_test::RequireGoldens()) GTEST_FAIL() << "golden key path missing in " << file;       \
    if (!::testing::Test::HasFailure()) GTEST_SKIP() << "LAYA_GOLDENS unset (local run)";                          \
    return;                                                                                       \
  }

constexpr size_t kChunk = 100;  // rows per test case in the large families
// Parameter lists are built from fixed row counts, never from the golden: loading a 7.6 MB file at
// registration made `--gtest_list_tests` take seconds and tripped ctest's discovery timeout. Each
// test asserts the golden still holds exactly that many rows (as laya-go's tests do), so a changed
// golden fails loudly instead of leaving rows untested.
inline std::vector<int> Chunks(size_t rows) {
  std::vector<int> out;
  for (size_t i = 0; i * kChunk < rows; i++) out.push_back(int(i));
  return out;
}
inline std::vector<int> Rows(size_t rows) {
  std::vector<int> out;
  for (size_t i = 0; i < rows; i++) out.push_back(int(i));
  return out;
}
inline uint64_t BitsOf(const Value& v) {
  return v.kind() == Value::Kind::Int ? uint64_t(v.as_int()) : std::strtoull(v.big_int().c_str(), nullptr, 10);
}
inline double Num(const Value& v) {
  return v.kind() == Value::Kind::Int ? double(v.as_int()) : v.as_float();
}

// Switches the process to a comma-decimal locale for a scope; ok() is false when none is installed.
class CommaLocale {
 public:
  CommaLocale() : saved_(std::setlocale(LC_ALL, nullptr)) {
    for (const char* c : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8", "de_DE", "de_DE.ISO8859-1"}) {
      if (std::setlocale(LC_ALL, c) != nullptr && *std::localeconv()->decimal_point == ',') {
        ok_ = true;
        return;
      }
    }
    std::setlocale(LC_ALL, saved_.c_str());
  }
  ~CommaLocale() { std::setlocale(LC_ALL, saved_.c_str()); }
  CommaLocale(const CommaLocale&) = delete;
  CommaLocale& operator=(const CommaLocale&) = delete;
  bool ok() const { return ok_; }

 private:
  std::string saved_;
  bool ok_ = false;
};

}  // namespace laya_test
