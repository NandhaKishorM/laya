#include "laya/pyjson.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cnum.hpp"
#include "laya/unicodedata.hpp"

namespace laya {
namespace pyjson {

namespace {
std::string Fixed(const std::string& digits, int exp) {
  if (exp < 0) return "0." + std::string(size_t(-exp - 1), '0') + digits;
  size_t whole = size_t(exp) + 1;
  if (digits.size() <= whole) return digits + std::string(whole - digits.size(), '0') + ".0";
  return digits.substr(0, whole) + "." + digits.substr(whole);
}

std::string Hex(char32_t r, int width, char marker) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "\\%c%0*x", marker, width, static_cast<unsigned>(r));
  return buf;
}
}  // namespace

std::string ReprFloat(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  std::string sign;
  if (std::signbit(v)) {
    sign = "-";
    v = -v;
  }
  if (v == 0) return sign + "0.0";
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);  // shortest d.ddde+XX
  std::string text(buf, res.ptr);
  size_t e = text.find('e');
  std::string mantissa = text.substr(0, e);
  int exp = std::atoi(text.c_str() + e + 1);
  std::string digits;
  for (char c : mantissa) {
    if (c != '.') digits.push_back(c);
  }
  if (exp >= -4 && exp < 16) return sign + Fixed(digits, exp);
  std::string out = sign;
  out.push_back(digits[0]);
  if (digits.size() > 1) {
    out.push_back('.');
    out += digits.substr(1);
  }
  out.push_back('e');
  out.push_back(exp < 0 ? '-' : '+');
  if (exp < 0) exp = -exp;
  if (exp < 10) out.push_back('0');
  out += std::to_string(exp);
  return out;
}

std::string Repr(std::string_view s) {
  bool has_single = s.find('\'') != std::string_view::npos;
  bool has_double = s.find('"') != std::string_view::npos;
  const char quote = (has_single && !has_double) ? '"' : '\'';
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back(quote);
  for (size_t i = 0; i < s.size();) {
    unicodedata::Decoded d = unicodedata::DecodeRune(s.substr(i));
    i += d.width;
    const char32_t r = d.rune;
    if (r == char32_t(quote) || r == '\\') {
      out.push_back('\\');
      out.push_back(char(r));
    } else if (r == '\n') {
      out += "\\n";
    } else if (r == '\r') {
      out += "\\r";
    } else if (r == '\t') {
      out += "\\t";
    } else if (unicodedata::IsPrintable(r)) {
      unicodedata::AppendRune(out, r);
    } else if (r < 0x100) {
      out += Hex(r, 2, 'x');
    } else if (r < 0x10000) {
      out += Hex(r, 4, 'u');
    } else {
      out += Hex(r, 8, 'U');
    }
  }
  out.push_back(quote);
  return out;
}

std::string Percent0(double fraction) {
  double scaled = 100 * fraction;
  if (std::isnan(scaled)) return "nan";
  if (std::isinf(scaled)) return scaled > 0 ? "inf" : "-inf";
  return cnum::Fixed(scaled, 0);
}

double Round4(double v) {
  if (std::isnan(v) || std::isinf(v)) return v;
  return cnum::ToDouble(cnum::Fixed(v, 4));
}

}  // namespace pyjson
}  // namespace laya
