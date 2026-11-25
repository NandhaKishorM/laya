#include "laya/pyvalue.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include "cnum.hpp"
#include "laya/pyjson.hpp"
#include "laya/unicodedata.hpp"

namespace laya {
namespace pyvalue {

Value Value::Bool(bool b) {
  Value v;
  v.kind_ = Kind::Bool;
  v.i_ = b ? 1 : 0;
  return v;
}
Value Value::Int(int64_t i) {
  Value v;
  v.kind_ = Kind::Int;
  v.i_ = i;
  return v;
}
Value Value::BigInt(std::string decimal) {
  Value v;
  v.kind_ = Kind::BigInt;
  v.s_ = std::move(decimal);
  return v;
}
Value Value::Float(double d) {
  Value v;
  v.kind_ = Kind::Float;
  v.d_ = d;
  return v;
}
Value Value::String(std::string s) {
  Value v;
  v.kind_ = Kind::String;
  v.s_ = std::move(s);
  return v;
}
Value Value::Array(std::vector<Value> items) {
  Value v;
  v.kind_ = Kind::Array;
  v.vals_ = std::move(items);
  return v;
}
Value Value::Object() {
  Value v;
  v.kind_ = Kind::Object;
  return v;
}

const Value* Value::Find(std::string_view k) const {
  auto it = index_.find(std::string(k));
  return it == index_.end() ? nullptr : &vals_[it->second];
}

void Value::Set(std::string k, Value v) {
  auto it = index_.find(k);
  if (it != index_.end()) {
    vals_[it->second] = std::move(v);
    return;
  }
  index_.emplace(k, keys_.size());
  keys_.push_back(std::move(k));
  vals_.push_back(std::move(v));
}

namespace {

class Parser {
 public:
  explicit Parser(std::string_view s) : s_(s) {}

  Result<Value> Document() {
    if (!unicodedata::ValidUtf8(s_)) return Result<Value>::Fail("JSON text is not valid UTF-8");
    Space();
    Result<Value> v = Val(0);
    if (!v.ok()) return v;
    Space();
    if (i_ < s_.size()) return Fail("data after the JSON value");
    return v;
  }

 private:
  std::string_view s_;
  size_t i_ = 0;

  Result<Value> Fail(const char* msg) {
    return Result<Value>::Fail(std::string("JSON: ") + msg + " at offset " + std::to_string(i_));
  }
  void Space() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) i_++;
  }
  bool Literal(std::string_view w) {
    if (s_.substr(i_, w.size()) == w) {
      i_ += w.size();
      return true;
    }
    return false;
  }

  Result<Value> Val(int depth) {
    if (depth > kMaxDepth) {
      return Result<Value>::Fail("JSON nested deeper than " + std::to_string(kMaxDepth) + " levels");
    }
    if (i_ >= s_.size()) return Fail("unexpected end of text");
    const char c = s_[i_];
    if (c == '{') {
      i_++;
      return Obj(depth + 1);
    }
    if (c == '[') {
      i_++;
      return Arr(depth + 1);
    }
    if (c == '"') {
      i_++;
      std::string out;
      if (!Str(out)) return Fail(bad_);
      return Result<Value>::Ok(Value::String(std::move(out)));
    }
    if (Literal("true")) return Result<Value>::Ok(Value::Bool(true));
    if (Literal("false")) return Result<Value>::Ok(Value::Bool(false));
    if (Literal("null")) return Result<Value>::Ok(Value::Null());
    if (Literal("NaN")) return Result<Value>::Ok(Value::Float(std::nan("")));
    if (Literal("Infinity")) return Result<Value>::Ok(Value::Float(HUGE_VAL));
    if (Literal("-Infinity")) return Result<Value>::Ok(Value::Float(-HUGE_VAL));
    if (c == '-' || (c >= '0' && c <= '9')) return Num();
    return Fail("unexpected character");
  }

  Result<Value> Arr(int depth) {
    Value list = Value::Array();
    Space();
    if (i_ < s_.size() && s_[i_] == ']') {
      i_++;
      return Result<Value>::Ok(std::move(list));
    }
    for (;;) {
      Space();
      Result<Value> v = Val(depth);
      if (!v.ok()) return v;
      list.Push(std::move(v).value());
      Space();
      if (i_ >= s_.size()) return Fail("unterminated array");
      if (s_[i_] == ',') {
        i_++;
      } else if (s_[i_] == ']') {
        i_++;
        return Result<Value>::Ok(std::move(list));
      } else {
        return Fail("expected , or ]");
      }
    }
  }

  Result<Value> Obj(int depth) {
    Value obj = Value::Object();
    Space();
    if (i_ < s_.size() && s_[i_] == '}') {
      i_++;
      return Result<Value>::Ok(std::move(obj));
    }
    for (;;) {
      Space();
      if (i_ >= s_.size() || s_[i_] != '"') return Fail("expected a string key");
      i_++;
      std::string key;
      if (!Str(key)) return Fail(bad_);
      Space();
      if (i_ >= s_.size() || s_[i_] != ':') return Fail("expected :");
      i_++;
      Space();
      Result<Value> v = Val(depth);
      if (!v.ok()) return v;
      obj.Set(std::move(key), std::move(v).value());  // first position, last value
      Space();
      if (i_ >= s_.size()) return Fail("unterminated object");
      if (s_[i_] == ',') {
        i_++;
      } else if (s_[i_] == '}') {
        i_++;
        return Result<Value>::Ok(std::move(obj));
      } else {
        return Fail("expected , or }");
      }
    }
  }

  // Reads a string whose opening quote was consumed. On failure err_ names the problem.
  bool Str(std::string& out) {
    while (i_ < s_.size()) {
      const unsigned char c = static_cast<unsigned char>(s_[i_]);
      if (c == '"') {
        i_++;
        return true;
      }
      if (c < 0x20) return Bad("control character in a string");
      if (c != '\\') {
        out.push_back(char(c));
        i_++;
        continue;
      }
      i_++;
      if (i_ >= s_.size()) break;
      const char esc = s_[i_++];
      switch (esc) {
        case '"': case '\\': case '/': out.push_back(esc); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          uint32_t r;
          if (!Hex4(r)) return false;
          if (r >= 0xD800 && r < 0xDC00) {  // a high surrogate needs its low half
            if (s_.substr(i_, 2) != "\\u") return Bad("lone surrogate escape");
            i_ += 2;
            uint32_t lo;
            if (!Hex4(lo)) return false;
            if (lo < 0xDC00 || lo >= 0xE000) return Bad("lone surrogate escape");
            r = 0x10000 + ((r - 0xD800) << 10) + (lo - 0xDC00);
          } else if (r >= 0xDC00 && r < 0xE000) {
            return Bad("lone surrogate escape");
          }
          unicodedata::AppendRune(out, r);
          break;
        }
        default: return Bad("invalid escape");
      }
    }
    return Bad("unterminated string");
  }

  bool Bad(const char* msg) {
    bad_ = msg;
    return false;
  }

  bool Hex4(uint32_t& r) {
    if (i_ + 4 > s_.size()) return Bad("truncated \\u escape");
    r = 0;
    for (int k = 0; k < 4; k++) {
      char c = s_[i_ + k];
      int d;
      if (c >= '0' && c <= '9') d = c - '0';
      else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
      else return Bad("invalid \\u escape");
      r = r * 16 + uint32_t(d);
    }
    i_ += 4;
    return true;
  }

  Result<Value> Num() {
    const size_t start = i_;
    if (s_[i_] == '-') i_++;
    auto digits = [this]() {
      size_t n = 0;
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') {
        i_++;
        n++;
      }
      return n;
    };
    const size_t int_start = i_;
    if (i_ < s_.size() && s_[i_] == '0') {
      i_++;
    } else if (digits() == 0) {
      return Fail("invalid number");
    }
    const size_t int_digits = i_ - int_start;
    bool is_int = true;
    if (i_ < s_.size() && s_[i_] == '.') {
      i_++;
      if (digits() == 0) return Fail("invalid number");
      is_int = false;
    }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      i_++;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) i_++;
      if (digits() == 0) return Fail("invalid number");
      is_int = false;
    }
    const std::string text(s_.substr(start, i_ - start));
    if (is_int) {
      if (int_digits > size_t(kMaxIntDigits)) return Fail("int literal exceeds the limit of 4300 digits");
      errno = 0;
      long long n = std::strtoll(text.c_str(), nullptr, 10);
      if (errno == 0) return Result<Value>::Ok(Value::Int(n));
      return Result<Value>::Ok(Value::BigInt(text));
    }
    // Out of range is +-Inf, as float("1e400") is in Python.
    return Result<Value>::Ok(Value::Float(cnum::ToDouble(text)));
  }

 public:
  const char* bad_ = "string";
};

}  // namespace

Result<Value> Parse(std::string_view text) {
  Parser p(text);
  return p.Document();
}

namespace {

bool WriteString(std::string& b, const std::string& s) {
  if (!unicodedata::ValidUtf8(s)) return false;
  b.push_back('"');
  for (char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': b += "\\\""; break;
      case '\\': b += "\\\\"; break;
      case '\n': b += "\\n"; break;
      case '\r': b += "\\r"; break;
      case '\t': b += "\\t"; break;
      case '\b': b += "\\b"; break;
      case '\f': b += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          b += buf;
        } else {
          b.push_back(ch);
        }
    }
  }
  b.push_back('"');
  return true;
}

std::string FloatStr(double f) {
  if (std::isnan(f)) return "nan";
  if (std::isinf(f)) return f > 0 ? "inf" : "-inf";
  return pyjson::ReprFloat(f);
}

Result<bool> Write(std::string& b, const Value& v, int depth) {
  if (depth > kMaxDepth) {
    return Result<bool>::Fail("value nested deeper than " + std::to_string(kMaxDepth) + " levels");
  }
  switch (v.kind()) {
    case Value::Kind::Null: b += "null"; break;
    case Value::Kind::Bool: b += v.as_bool() ? "true" : "false"; break;
    case Value::Kind::Int: b += std::to_string(v.as_int()); break;
    case Value::Kind::BigInt: b += v.big_int(); break;
    case Value::Kind::Float: b += pyjson::ReprFloat(v.as_float()); break;
    case Value::Kind::String:
      if (!WriteString(b, v.as_string())) {
        return Result<bool>::Fail("string is not valid UTF-8, which a Python str always is");
      }
      break;
    case Value::Kind::Array: {
      b.push_back('[');
      for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) b += ", ";
        Result<bool> r = Write(b, v.value(i), depth + 1);
        if (!r.ok()) return r;
      }
      b.push_back(']');
      break;
    }
    case Value::Kind::Object: {
      b.push_back('{');
      for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) b += ", ";
        if (!WriteString(b, v.key(i))) {
          return Result<bool>::Fail("string is not valid UTF-8, which a Python str always is");
        }
        b += ": ";
        Result<bool> r = Write(b, v.value(i), depth + 1);
        if (!r.ok()) return r;
      }
      b.push_back('}');
      break;
    }
  }
  return Result<bool>::Ok(true);
}

void WriteRepr(std::string& b, const Value& v) {
  switch (v.kind()) {
    case Value::Kind::Null: b += "None"; break;
    case Value::Kind::Bool: b += v.as_bool() ? "True" : "False"; break;
    case Value::Kind::String: b += pyjson::Repr(v.as_string()); break;
    case Value::Kind::Int:
    case Value::Kind::BigInt:
    case Value::Kind::Float: b += Str(v); break;
    case Value::Kind::Array:
      b.push_back('[');
      for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) b += ", ";
        WriteRepr(b, v.value(i));
      }
      b.push_back(']');
      break;
    case Value::Kind::Object:
      b.push_back('{');
      for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) b += ", ";
        b += pyjson::Repr(v.key(i));
        b += ": ";
        WriteRepr(b, v.value(i));
      }
      b.push_back('}');
      break;
  }
}

}  // namespace

Result<std::string> Dumps(const Value& v) {
  std::string out;
  Result<bool> r = Write(out, v, 0);
  if (!r.ok()) return Result<std::string>::Fail(r.error());
  return Result<std::string>::Ok(std::move(out));
}

std::string TypeName(const Value& v) {
  switch (v.kind()) {
    case Value::Kind::Null: return "NoneType";
    case Value::Kind::Bool: return "bool";
    case Value::Kind::Int:
    case Value::Kind::BigInt: return "int";
    case Value::Kind::Float: return "float";
    case Value::Kind::String: return "str";
    case Value::Kind::Array: return "list";
    case Value::Kind::Object: return "dict";
  }
  return "";
}

std::string Repr(const Value& v) {
  std::string out;
  WriteRepr(out, v);
  return out;
}

std::string Str(const Value& v) {
  switch (v.kind()) {
    case Value::Kind::String: return v.as_string();
    case Value::Kind::Int: return std::to_string(v.as_int());
    case Value::Kind::BigInt: return v.big_int();
    case Value::Kind::Float: return FloatStr(v.as_float());
    default: return Repr(v);
  }
}

}  // namespace pyvalue
}  // namespace laya
