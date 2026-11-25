#pragma once
// Locale-independent number text. strtod and snprintf follow the process locale (a comma decimal
// point under de_DE), which silently truncates or corrupts what Python writes with a dot; these
// helpers translate the decimal point and never touch the global locale.
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace laya {
namespace cnum {

inline std::string DecimalPoint() {
  const lconv* lc = std::localeconv();
  return (lc && lc->decimal_point && *lc->decimal_point) ? lc->decimal_point : ".";
}
// strtod over text using '.', whatever the locale. `point` is the locale's decimal point.
inline double ToDouble(const std::string& text, const std::string& point) {
  std::string t;
  for (char c : text) {
    if (c == '.') t += point;
    else t.push_back(c);
  }
  return std::strtod(t.c_str(), nullptr);
}
inline double ToDouble(const std::string& text) { return ToDouble(text, DecimalPoint()); }
// "%.<digits>f" with a '.' decimal point. `point` is the locale's decimal point.
inline std::string Fixed(double v, int digits, const std::string& point) {
  char buf[400];
  std::snprintf(buf, sizeof buf, "%.*f", digits, v);
  std::string s = buf;
  size_t p = s.find(point);
  if (point != "." && p != std::string::npos) s.replace(p, point.size(), ".");
  return s;
}
inline std::string Fixed(double v, int digits) { return Fixed(v, digits, DecimalPoint()); }

}  // namespace cnum
}  // namespace laya
