#pragma once
// A shipped checkpoint's tokenizer, loaded once. Without LAYA_CHECKPOINTS the test is skipped
// locally and FAILS under LAYA_REQUIRE_GOLDENS=1 (CI), so a missing checkpoint cannot pass as green.
#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

#include "laya/tokenizer.hpp"

namespace laya_test {

inline const laya::tokenizer::Tokenizer* LoadCheckpoint(const std::string& name) {
  static std::mutex mu;
  static std::map<std::string, laya::tokenizer::Tokenizer> loaded;
  std::lock_guard<std::mutex> lock(mu);
  auto it = loaded.find(name);
  if (it != loaded.end()) return &it->second;
  const char* root = std::getenv("LAYA_CHECKPOINTS");
  if (root == nullptr || *root == 0) return nullptr;
  auto t = laya::tokenizer::Tokenizer::LoadDir(std::string(root) + "/" + name);
  if (!t.ok()) {
    ADD_FAILURE() << "load the " << name << " tokenizer: " << t.error();
    return nullptr;
  }
  return &loaded.emplace(name, std::move(t).value()).first->second;
}

#define LAYA_CHECKPOINT_TOK(var, name)                                                     \
  const laya::tokenizer::Tokenizer* var = laya_test::LoadCheckpoint(name);                 \
  if (var == nullptr) {                                                                    \
    const char* req = std::getenv("LAYA_REQUIRE_GOLDENS");                                 \
    if (req != nullptr && std::string(req) == "1" && !HasFailure()) {                      \
      ADD_FAILURE() << "LAYA_CHECKPOINTS is unset but LAYA_REQUIRE_GOLDENS=1";             \
    } else if (!HasFailure()) {                                                            \
      GTEST_SKIP() << "LAYA_CHECKPOINTS unset (local run)";                                \
    }                                                                                      \
    return;                                                                                \
  }

}  // namespace laya_test
