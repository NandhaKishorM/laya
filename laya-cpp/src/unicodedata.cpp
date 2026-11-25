#include "laya/unicodedata.hpp"

#include <cstddef>

namespace laya {
namespace unicodedata {
namespace {
#include "unicodedata_tables.inc"

template <size_t N>
bool Contains(const uint32_t (&pairs)[N], char32_t r) {
  size_t lo = 0, hi = N / 2;  // half-open over pairs
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (r < pairs[mid * 2]) {
      hi = mid;
    } else if (r > pairs[mid * 2 + 1]) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

template <size_t N>
bool Lookup(const uint64_t (&mask)[2], const uint32_t (&pairs)[N], char32_t r) {
  if (r < 0x80) return ((mask[r >> 6] >> (r & 63)) & 1) != 0;
  return Contains(pairs, r);
}

bool IsCased(char32_t r) { return IsUpper(r) || IsLower(r) || Contains(k_title, r); }

constexpr char32_t kCapitalSigma = 0x03A3, kSmallSigma = 0x03C3, kFinalSigma = 0x03C2;

bool SigmaIsFinal(std::string_view s, size_t i) {
  bool before = false;
  for (size_t j = i; j > 0;) {
    Decoded d = DecodeLastRune(s.substr(0, j));
    j -= d.width;
    if (Contains(k_case_ignorable, d.rune)) continue;
    before = Contains(k_cased, d.rune);
    break;
  }
  if (!before) return false;
  for (size_t j = i + 2; j < s.size();) {  // U+03A3 is two bytes in UTF-8
    Decoded d = DecodeRune(s.substr(j));
    j += d.width;
    if (Contains(k_case_ignorable, d.rune)) continue;
    return !Contains(k_cased, d.rune);
  }
  return true;
}

size_t InvalidPrefixLen(std::string_view b) {
  uint8_t lo = 0x80, hi = 0xBF;
  size_t need;
  uint8_t c = uint8_t(b[0]);
  if (c >= 0xC2 && c <= 0xDF) {
    need = 1;
  } else if (c == 0xE0) {
    need = 2;
    lo = 0xA0;
  } else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) {
    need = 2;
  } else if (c == 0xED) {
    need = 2;
    hi = 0x9F;
  } else if (c == 0xF0) {
    need = 3;
    lo = 0x90;
  } else if (c >= 0xF1 && c <= 0xF3) {
    need = 3;
  } else if (c == 0xF4) {
    need = 3;
    hi = 0x8F;
  } else {
    return 1;
  }
  size_t n = 1;
  for (; n <= need && n < b.size(); n++) {
    uint8_t x = uint8_t(b[n]);
    if (x < lo || x > hi) break;
    lo = 0x80;
    hi = 0xBF;
  }
  return n;
}

}  // namespace

const char* const kVersion = kUnicodeVersion;

bool IsAlpha(char32_t r) { return Lookup(k_alpha_ascii, k_alpha, r); }
bool IsWord(char32_t r) { return Lookup(k_word_ascii, k_word, r); }
bool IsCombining(char32_t r) { return Lookup(k_combining_ascii, k_combining, r); }
bool IsDigit(char32_t r) { return Lookup(k_digit_ascii, k_digit, r); }
bool IsUpper(char32_t r) { return Lookup(k_upper_ascii, k_upper, r); }
bool IsLower(char32_t r) { return Lookup(k_lower_ascii, k_lower, r); }
bool IsSpace(char32_t r) { return Lookup(k_space_ascii, k_space, r); }
bool IsMark(char32_t r) { return Lookup(k_mark_ascii, k_mark, r); }
bool IsInitial(char32_t r) { return Lookup(k_initial_ascii, k_initial, r); }
bool IsPrintable(char32_t r) { return Lookup(k_printable_ascii, k_printable, r); }
bool IsPythonWord(char32_t r) { return r == '_' || IsWord(r) || IsDigit(r); }
bool IsWordOrDigit(char32_t r) { return IsWord(r) || IsDigit(r); }

bool IsUpperString(std::string_view s) {
  bool saw = false;
  for (size_t i = 0; i < s.size();) {
    Decoded d = DecodeRune(s.substr(i));
    i += d.width;
    if (!IsCased(d.rune)) continue;
    if (!IsUpper(d.rune)) return false;
    saw = true;
  }
  return saw;
}

Decoded DecodeRune(std::string_view s) {
  const uint8_t c = uint8_t(s[0]);
  if (c < 0x80) return {c, 1};
  size_t need;
  uint8_t lo = 0x80, hi = 0xBF;
  char32_t r;
  if (c >= 0xC2 && c <= 0xDF) {
    need = 1;
    r = c & 0x1F;
  } else if (c >= 0xE0 && c <= 0xEF) {
    need = 2;
    r = c & 0x0F;
    if (c == 0xE0) lo = 0xA0;
    if (c == 0xED) hi = 0x9F;
  } else if (c >= 0xF0 && c <= 0xF4) {
    need = 3;
    r = c & 0x07;
    if (c == 0xF0) lo = 0x90;
    if (c == 0xF4) hi = 0x8F;
  } else {
    return {kReplacement, 1};
  }
  if (s.size() < need + 1) return {kReplacement, 1};
  for (size_t i = 1; i <= need; i++) {
    uint8_t x = uint8_t(s[i]);
    if (x < lo || x > hi) return {kReplacement, 1};
    lo = 0x80;
    hi = 0xBF;
    r = (r << 6) | (x & 0x3F);
  }
  return {r, need + 1};
}

Decoded DecodeLastRune(std::string_view s) {
  const size_t end = s.size();
  if (uint8_t(s[end - 1]) < 0x80) return {uint8_t(s[end - 1]), 1};
  const size_t lim = end > 4 ? end - 4 : 0;
  size_t start = end - 1;
  while (start > lim && (uint8_t(s[start]) & 0xC0) == 0x80) start--;  // back to a start byte
  Decoded d = DecodeRune(s.substr(start, end - start));
  if (start + d.width != end) return {kReplacement, 1};
  return d;
}

bool ValidUtf8(std::string_view s) {
  for (size_t i = 0; i < s.size();) {
    if (uint8_t(s[i]) < 0x80) {
      i++;
      continue;
    }
    Decoded d = DecodeRune(s.substr(i));
    if (d.rune == kReplacement && d.width == 1) return false;
    i += d.width;
  }
  return true;
}

void AppendRune(std::string& out, char32_t r) {
  if (r > 0x10FFFF || (r >= 0xD800 && r <= 0xDFFF)) r = kReplacement;
  if (r < 0x80) {
    out.push_back(char(r));
  } else if (r < 0x800) {
    out.push_back(char(0xC0 | (r >> 6)));
    out.push_back(char(0x80 | (r & 0x3F)));
  } else if (r < 0x10000) {
    out.push_back(char(0xE0 | (r >> 12)));
    out.push_back(char(0x80 | ((r >> 6) & 0x3F)));
    out.push_back(char(0x80 | (r & 0x3F)));
  } else {
    out.push_back(char(0xF0 | (r >> 18)));
    out.push_back(char(0x80 | ((r >> 12) & 0x3F)));
    out.push_back(char(0x80 | ((r >> 6) & 0x3F)));
    out.push_back(char(0x80 | (r & 0x3F)));
  }
}

size_t RuneCount(std::string_view s) {
  size_t n = 0;
  for (size_t i = 0; i < s.size(); n++) i += DecodeRune(s.substr(i)).width;
  return n;
}

char32_t LowerRune(char32_t r) {
  if (r < 0x80) return (r >= 'A' && r <= 'Z') ? r + 32 : r;
  size_t lo = 0, hi = sizeof(k_lower_from) / sizeof(k_lower_from[0]);
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (r < k_lower_from[mid]) {
      hi = mid;
    } else if (r > k_lower_from[mid]) {
      lo = mid + 1;
    } else {
      return k_lower_to[mid];
    }
  }
  return r;
}

std::string Lower(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    Decoded d = DecodeRune(s.substr(i));
    if (d.rune == kCapitalSigma) {
      AppendRune(out, SigmaIsFinal(s, i) ? kFinalSigma : kSmallSigma);
    } else {
      bool multi = false;
      for (const LowerMulti& m : k_lower_multi) {
        if (m.from == d.rune) {
          for (uint32_t k = 0; k < m.count; k++) AppendRune(out, m.to[k]);
          multi = true;
          break;
        }
      }
      if (!multi) AppendRune(out, LowerRune(d.rune));
    }
    i += d.width;
  }
  return out;
}

std::string Lossy(std::string_view b) {
  std::string out;
  out.reserve(b.size() + 8);
  for (size_t i = 0; i < b.size();) {
    Decoded d = DecodeRune(b.substr(i));
    if (!(d.rune == kReplacement && d.width == 1)) {
      out.append(b.substr(i, d.width));
      i += d.width;
      continue;
    }
    AppendRune(out, kReplacement);
    i += InvalidPrefixLen(b.substr(i));
  }
  return out;
}

std::string ToValid(std::string_view s) { return Lossy(s); }

std::string DecimalASCII(std::string_view s) {
  std::string out;
  for (size_t i = 0; i < s.size();) {
    Decoded d = DecodeRune(s.substr(i));
    i += d.width;
    char32_t r = d.rune;
    if (r >= 0x80 && Contains(k_digit, r)) {
      size_t lo = 0, hi = sizeof(k_digit) / sizeof(k_digit[0]) / 2 - 1;
      while (lo < hi) {  // last range starting at or before r
        size_t mid = (lo + hi + 1) / 2;
        if (k_digit[mid * 2] <= r) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }
      r = '0' + (r - k_digit[lo * 2]) % 10;
    }
    AppendRune(out, r);
  }
  return out;
}

namespace {
std::string TrimBy(std::string_view s, bool numeric) {
  auto trims = [numeric](char32_t r) { return IsSpace(r) && !(numeric && r >= 0x1c && r <= 0x1f); };
  size_t b = 0, e = s.size();
  while (b < e) {
    Decoded d = DecodeRune(s.substr(b, e - b));
    if (!trims(d.rune)) break;
    b += d.width;
  }
  while (e > b) {
    Decoded d = DecodeLastRune(s.substr(b, e - b));
    if (!trims(d.rune)) break;
    e -= d.width;
  }
  return std::string(s.substr(b, e - b));
}
}  // namespace

std::string Strip(std::string_view s) { return TrimBy(s, false); }
std::string StripNumeric(std::string_view s) { return TrimBy(s, true); }

bool IsBlank(std::string_view s) {
  for (size_t i = 0; i < s.size();) {
    Decoded d = DecodeRune(s.substr(i));
    if (!IsSpace(d.rune)) return false;
    i += d.width;
  }
  return true;
}

std::vector<std::string> Fields(std::string_view s) {
  std::vector<std::string> out;
  bool in = false;
  size_t start = 0;
  for (size_t i = 0; i < s.size();) {
    Decoded d = DecodeRune(s.substr(i));
    if (IsSpace(d.rune)) {
      if (in) {
        out.emplace_back(s.substr(start, i - start));
        in = false;
      }
    } else if (!in) {
      in = true;
      start = i;
    }
    i += d.width;
  }
  if (in) out.emplace_back(s.substr(start));
  return out;
}

}  // namespace unicodedata
}  // namespace laya
