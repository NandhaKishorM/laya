#pragma once
// The values laya's Python code handles as dicts, lists, strs, ints and floats, read and written
// as json.loads / json.dumps(ensure_ascii=False) do: object order and number spelling kept, NaN
// and Infinity accepted, a repeated key keeps its first position and takes its last value.
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "laya/result.hpp"

namespace laya {
namespace pyvalue {

constexpr int kMaxDepth = 512;        // nesting bound for Parse and Dumps
constexpr int kMaxIntDigits = 4300;   // sys.int_info.default_max_str_digits

class Value {
 public:
  enum class Kind { Null, Bool, Int, BigInt, Float, String, Array, Object };

  Value() = default;
  static Value Null() { return Value(); }
  static Value Bool(bool b);
  static Value Int(int64_t i);
  static Value BigInt(std::string decimal);  // an int literal beyond int64: sign and digits
  static Value Float(double d);
  static Value String(std::string s);
  static Value Array(std::vector<Value> items = {});
  static Value Object();  // empty; use Set to add members

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::Null; }
  bool as_bool() const { return i_ != 0; }
  int64_t as_int() const { return i_; }
  const std::string& big_int() const { return s_; }
  double as_float() const { return d_; }
  const std::string& as_string() const { return s_; }
  const std::vector<Value>& items() const { return vals_; }  // Array elements or Object values

  // Object members in insertion order.
  size_t size() const { return vals_.size(); }
  const std::string& key(size_t i) const { return keys_[i]; }
  const Value& value(size_t i) const { return vals_[i]; }
  const Value* Find(std::string_view key) const;  // nullptr when absent
  // Python dict assignment: a new key goes last, an existing key keeps its place.
  void Set(std::string key, Value v);
  void Push(Value v) { vals_.push_back(std::move(v)); }

 private:
  Kind kind_ = Kind::Null;
  int64_t i_ = 0;
  double d_ = 0;
  std::string s_;
  std::vector<std::string> keys_;
  std::unordered_map<std::string, size_t> index_;  // key -> position, so Set and Find are O(1)
  std::vector<Value> vals_;
};

// json.loads of one document of UTF-8 text, as json.loads(str) reads it: a leading byte order mark
// is refused (json.loads(bytes) would skip it; strip it first to get that behavior). The nesting
// bound kMaxDepth is laya-go's, not Python's: CPython's RecursionError limit depends on the
// build and on what is nested (a wrapped value costs more than an empty array), so the two agree
// on ordinary documents and may differ right around the bound. Refused: invalid UTF-8, a lone surrogate escape, a raw
// control character in a string, an int literal over 4300 digits, nesting beyond kMaxDepth.
Result<Value> Parse(std::string_view text);
// json.dumps(v, ensure_ascii=False) with the default ", " and ": " separators. Fails on a string
// that is not valid UTF-8 and on nesting beyond kMaxDepth.
Result<std::string> Dumps(const Value& v);
// type(v).__name__, repr(v) and str(v) of a decoded value.
std::string TypeName(const Value& v);
std::string Repr(const Value& v);
std::string Str(const Value& v);

}  // namespace pyvalue
}  // namespace laya
