#pragma once
// A reader and runner for Hugging Face tokenizer.json files. It reproduces the ids the reference
// `tokenizers` crate (0.23.2) gives for the BPE tokenizers laya ships: the normalizer, the
// pre-tokenizer, the added-token splitter, BPE with its byte fallback, and the decoder chain.
//
// Anything the loader does not implement is refused with an error that begins with kUnsupported
// rather than approximated: an approximated tokenizer changes ids in a way no answer-level test
// localises. A loaded Tokenizer is immutable and safe for concurrent use. Strings are UTF-8; text
// that is not valid UTF-8 is read as bytes.decode("utf-8", "replace") would deliver it.
//
// The Unicode behaviour that decides ids (letter and number classes, word characters beside a
// single_word token, lowercase, normalization) is compiled in from the reference, see
// laya-go/scripts/gen_tables.py --lang cpp, so ids do not depend on the C++ library's Unicode.
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "laya/result.hpp"

namespace laya {
namespace tokenizer {

// Every error that refuses a feature this library does not implement starts with this text.
extern const char* const kUnsupported;
bool IsUnsupported(const std::string& error);

struct Impl;

class Tokenizer {
 public:
  Tokenizer() = default;  // empty, only so Result<Tokenizer> can hold one; obtain a usable one from Parse or LoadDir
  // tokenizer.json bytes and, optionally, tokenizer_config.json bytes (names the CLS, SEP, MASK
  // and PAD tokens).
  static Result<Tokenizer> Parse(std::string_view tokenizer_json,
                                 std::optional<std::string_view> config_json = std::nullopt);
  // <dir>/tokenizer/tokenizer.json when that exists, else <dir>/tokenizer.json, with the
  // tokenizer_config.json beside whichever is used (it may be absent).
  static Result<Tokenizer> LoadDir(const std::string& dir);

  // Token ids of text. limit < 0 means no limit, otherwise only the first limit ids are made.
  std::vector<int32_t> Encode(std::string_view text, int64_t limit = -1) const;
  // Ids the vocabulary does not hold are skipped, as the reference skips them; with
  // skip_special_tokens the added tokens marked special are skipped too.
  std::string Decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const;

  size_t VocabSize() const;     // the model vocabulary plus added tokens outside it
  int32_t HighestId() const;
  std::optional<int32_t> TokenToId(std::string_view token) const;
  std::optional<std::string> IdToToken(int32_t id) const;
  std::optional<int32_t> SpecialId(std::string_view content) const;
  std::map<std::string, int32_t> SpecialTokens() const;
  std::optional<int32_t> Cls() const;
  std::optional<int32_t> Sep() const;
  std::optional<int32_t> Mask() const;
  std::optional<int32_t> Pad() const;
  std::string MaskToken() const;  // "" when none is configured

  // Test seams: the stages of Encode, so a stage can be compared with the reference on its own.
  // Pieces of text whose rescan and heap merges differ (must be 0); counts the pieces compared.
  size_t CountMergeDisagreements(std::string_view text, size_t* pieces_checked) const;
  std::string NormalizeForTest(std::string_view text) const;
  std::vector<std::string> PreTokenizeForTest(std::string_view normalized) const;

 private:
  explicit Tokenizer(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
};

// Stages exposed for the golden tests that sweep them without a tokenizer.json.
enum class Form { NFC, NFD, NFKC, NFKD };
std::string Normalize(std::string_view text, Form form);
std::string Lowercase(std::string_view text);  // the crate's, no final-sigma rule

// A normalizer or pre-tokenizer built from its tokenizer.json section, e.g. {"type":"Lowercase"}.
Result<std::string> RunNormalizerSpec(std::string_view spec_json, std::string_view input);
Result<std::vector<std::string>> RunPreTokenizerSpec(std::string_view spec_json, std::string_view input);

}  // namespace tokenizer
}  // namespace laya
