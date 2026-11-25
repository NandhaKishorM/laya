// Port of laya-go/tokenizer. Internal failures are a private exception caught at the API boundary
// (Parse and the *Spec helpers); no exception crosses the public API.
#include "laya/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "laya/pyvalue.hpp"
#include "laya/unicodedata.hpp"

namespace laya {
namespace tokenizer {

const char* const kUnsupported = "unsupported tokenizer component: ";
bool IsUnsupported(const std::string& e) { return e.rfind(kUnsupported, 0) == 0; }

namespace {
#include "tokenizer_tables.inc"

using pyvalue::Value;
namespace ud = unicodedata;

struct Fail {
  std::string msg;
};
[[noreturn]] void Refuse(const std::string& m) { throw Fail{m}; }
[[noreturn]] void Unsupported(const std::string& m) { throw Fail{std::string(kUnsupported) + m}; }

template <class T, size_t N>
constexpr size_t Len(const T (&)[N]) {
  return N;
}

// ------------------------------------------------------------------------------------------
// Character classes: the reference's, from the generated tables.
// ------------------------------------------------------------------------------------------
bool IsWhiteSpace(char32_t r) {
  switch (r) {
    case '\t': case '\n': case 0x0B: case '\f': case '\r': case ' ': case 0x85: case 0xA0:
    case 0x1680: case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
      return true;
  }
  return r >= 0x2000 && r <= 0x200A;
}

template <size_t N>
bool InRanges(const uint32_t (&pairs)[N], char32_t r) {
  long lo = 0, hi = long(N / 2) - 1;
  while (lo <= hi) {
    long mid = (lo + hi) / 2;
    if (r < pairs[mid * 2]) hi = mid - 1;
    else if (r > pairs[mid * 2 + 1]) lo = mid + 1;
    else return true;
  }
  return false;
}
bool IsLetter(char32_t r) {
  if (r < 0x80) return 'a' <= (r | 0x20) && (r | 0x20) <= 'z';
  return InRanges(k_letter, r);
}
bool IsNumber(char32_t r) {
  if (r < 0x80) return '0' <= r && r <= '9';
  return InRanges(k_number, r);
}
bool IsWordChar(char32_t r) { return InRanges(k_word_char, r); }

// ------------------------------------------------------------------------------------------
// Lowercase (per character, no final sigma) and normalization (UAX #15) from the crate's tables.
// ------------------------------------------------------------------------------------------
void LowerOf(char32_t r, std::string& out) {
  if (r < 0x80) {
    out.push_back(char((r >= 'A' && r <= 'Z') ? r + 32 : r));
    return;
  }
  for (const auto& m : k_lower_multi) {
    if (m.from == r) {
      for (uint32_t i = 0; i < m.count; i++) ud::AppendRune(out, m.to[i]);
      return;
    }
  }
  const uint32_t* end = k_lower_from + Len(k_lower_from);
  const uint32_t* at = std::lower_bound(k_lower_from, end, uint32_t(r));
  if (at != end && *at == r) {
    ud::AppendRune(out, k_lower_to[at - k_lower_from]);
    return;
  }
  ud::AppendRune(out, r);
}

std::vector<char32_t> Runes(std::string_view s) {
  std::vector<char32_t> out;
  out.reserve(s.size());
  while (!s.empty()) {
    auto d = ud::DecodeRune(s);
    out.push_back(d.rune);
    s.remove_prefix(d.width);
  }
  return out;
}
std::string FromRunes(const std::vector<char32_t>& rs) {
  std::string out;
  out.reserve(rs.size());
  for (char32_t r : rs) ud::AppendRune(out, r);
  return out;
}

constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount,
                   kSCount = kLCount * kNCount;

bool LookupPool(const uint32_t* keys, size_t nkeys, const uint32_t* offsets, const uint32_t* pool,
                char32_t r, const uint32_t** begin, const uint32_t** end) {
  const uint32_t* at = std::lower_bound(keys, keys + nkeys, uint32_t(r));
  if (at == keys + nkeys || *at != r) return false;
  size_t i = size_t(at - keys);
  *begin = pool + offsets[i];
  *end = pool + offsets[i + 1];
  return true;
}

void Decompose(std::vector<char32_t>& dst, char32_t r, bool compat) {
  if (r < 0xA0 || (r < 0xC0 && !compat)) {
    dst.push_back(r);
    return;
  }
  if (r >= kSBase && r < kSBase + kSCount) {
    char32_t s = r - kSBase;
    dst.push_back(kLBase + s / kNCount);
    dst.push_back(kVBase + (s % kNCount) / kTCount);
    if (s % kTCount != 0) dst.push_back(kTBase + s % kTCount);
    return;
  }
  const uint32_t *b, *e;
  if (compat && LookupPool(k_compat_keys, Len(k_compat_keys), k_compat_offsets, k_compat_pool, r, &b, &e)) {
    dst.insert(dst.end(), b, e);
    return;
  }
  if (LookupPool(k_canon_keys, Len(k_canon_keys), k_canon_offsets, k_canon_pool, r, &b, &e)) {
    dst.insert(dst.end(), b, e);
    return;
  }
  dst.push_back(r);
}

uint8_t CombiningClass(char32_t r) {
  if (r < 0x300) return 0;
  size_t n = Len(k_ccc_ranges) / 3, lo = 0, hi = n;
  while (lo < hi) {  // first triple whose hi >= r
    size_t mid = (lo + hi) / 2;
    if (k_ccc_ranges[mid * 3 + 1] >= r) hi = mid;
    else lo = mid + 1;
  }
  if (lo < n && k_ccc_ranges[lo * 3] <= r) return uint8_t(k_ccc_ranges[lo * 3 + 2]);
  return 0;
}

void Reorder(std::vector<char32_t>& rs) {
  for (size_t i = 0; i < rs.size();) {
    if (CombiningClass(rs[i]) == 0) {
      i++;
      continue;
    }
    size_t j = i;
    while (j < rs.size() && CombiningClass(rs[j]) != 0) j++;
    std::stable_sort(rs.begin() + long(i), rs.begin() + long(j),
                     [](char32_t a, char32_t b) { return CombiningClass(a) < CombiningClass(b); });
    i = j;
  }
}

bool ComposePair(char32_t first, char32_t second, char32_t* out) {
  if (first >= kLBase && first < kLBase + kLCount && second >= kVBase && second < kVBase + kVCount) {
    *out = kSBase + ((first - kLBase) * kVCount + (second - kVBase)) * kTCount;
    return true;
  }
  if (first >= kSBase && first < kSBase + kSCount && (first - kSBase) % kTCount == 0 &&
      second > kTBase && second < kTBase + kTCount) {
    *out = first + (second - kTBase);
    return true;
  }
  uint64_t key = (uint64_t(first) << 32) | second;
  const uint64_t* end = k_compose_keys + Len(k_compose_keys);
  const uint64_t* at = std::lower_bound(k_compose_keys, end, key);
  if (at != end && *at == key) {
    *out = k_compose_values[at - k_compose_keys];
    return true;
  }
  return false;
}

void ComposeRunes(std::vector<char32_t>& rs) {
  std::vector<char32_t> out;
  out.reserve(rs.size());
  long starter = -1;
  uint8_t last = 0;
  for (char32_t r : rs) {
    uint8_t cc = CombiningClass(r);
    char32_t c;
    if (starter >= 0 && (last < cc || last == 0) && ComposePair(out[size_t(starter)], r, &c)) {
      out[size_t(starter)] = c;
      continue;
    }
    if (cc == 0) {
      starter = long(out.size());
      last = 0;
    } else {
      last = cc;
    }
    out.push_back(r);
  }
  rs.swap(out);
}

std::string NormalizeImpl(std::string_view s, Form f) {
  bool compat = f == Form::NFKC || f == Form::NFKD, compose = f == Form::NFC || f == Form::NFKC;
  bool ascii = true, below = true;
  for (char32_t r : Runes(s)) {
    if (r >= 0x80) ascii = false;
    if (r >= 0x300) {
      below = false;
      break;
    }
  }
  // ASCII has no decomposition in any form, and nothing below U+0300 combines with what precedes.
  if (ascii || (below && compose && !compat)) return std::string(s);
  std::vector<char32_t> rs;
  for (char32_t r : Runes(s)) Decompose(rs, r, compat);
  Reorder(rs);
  if (compose) ComposeRunes(rs);
  return FromRunes(rs);
}

std::string LowercaseImpl(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  while (!s.empty()) {
    auto d = ud::DecodeRune(s);
    LowerOf(d.rune, out);
    s.remove_prefix(d.width);
  }
  return out;
}

std::string StripImpl(std::string_view s, bool left, bool right) {
  size_t a = 0, b = s.size();
  if (left) {
    while (a < b) {
      auto d = ud::DecodeRune(s.substr(a, b - a));
      if (!IsWhiteSpace(d.rune)) break;
      a += d.width;
    }
  }
  if (right) {
    while (b > a) {
      auto d = ud::DecodeLastRune(s.substr(a, b - a));
      if (!IsWhiteSpace(d.rune)) break;
      b -= d.width;
    }
  }
  return std::string(s.substr(a, b - a));
}

// ------------------------------------------------------------------------------------------
// A small regex for the Replace normalizer. The reference reads \s \d \w \b \p as Unicode, and a
// stand-in regex engine would read them its own way, so those are refused (as laya-go does); what
// is left is literals, classes of explicit ranges, ., ^, $, groups, | and the greedy and lazy
// repeats. Matching is a backtracker that remembers failed (pc, position) pairs, so it is linear.
// ------------------------------------------------------------------------------------------
class Regex {
 public:
  explicit Regex(const std::string& pattern) { Compile(pattern); }

  // Replaces every non-overlapping match with the literal content.
  std::string ReplaceAll(const std::string& text, const std::string& content) const {
    std::string out;
    size_t copied = 0, from = 0;
    Memo memo(prog_.size(), text.size() + 1);
    while (from <= text.size()) {
      size_t ms, me;
      if (!Find(text, from, &ms, &me, memo)) break;
      out.append(text, copied, ms - copied);
      out += content;
      copied = me;
      from = me;  // matches are never empty: Compile refuses a pattern that can match nothing
    }
    out.append(text, copied, std::string::npos);
    return out;
  }
  bool MatchesEmpty() const {
    size_t s, e;
    Memo memo(prog_.size(), 1);
    return Find("", 0, &s, &e, memo);
  }

 private:
  enum Op { kChar, kAny, kSet, kSplit, kJmp, kBol, kEol, kMatch };
  struct Inst {
    Op op;
    char32_t c = 0;
    int x = 0, y = 0;
  };
  struct Set {
    std::vector<std::pair<char32_t, char32_t>> ranges;
    bool negate = false;
  };
  std::vector<Inst> prog_;
  std::vector<Set> sets_;

  static constexpr size_t kMaxProgram = 100000;  // instructions after repeats are expanded
  static constexpr int kMaxDepth = 256;          // nested groups
  static constexpr size_t kDenseBits = size_t(1) << 28;
  int depth_ = 0;

  // The (instruction, position) states already tried. Dense while the table is small, a hash set above
  // that. One table serves a whole ReplaceAll: a state that failed still fails for any later start, and
  // the states of a match lie at or before its end, which no later (non-empty) match revisits.
  class Memo {
   public:
    Memo(size_t insts, size_t positions) : insts_(insts), dense_(insts * positions <= kDenseBits) {
      if (dense_) bits_.assign((insts * positions + 63) / 64, 0);
    }
    bool Visit(int pc, size_t pos) {  // true the first time
      size_t key = pos * insts_ + size_t(pc);
      if (!dense_) return sparse_.insert(key).second;
      uint64_t& w = bits_[key / 64];
      uint64_t bit = uint64_t(1) << (key % 64);
      if (w & bit) return false;
      w |= bit;
      return true;
    }

   private:
    size_t insts_;
    bool dense_;
    std::vector<uint64_t> bits_;
    std::unordered_set<size_t> sparse_;
  };

  // --- parser: pattern -> instructions, emitted directly from a recursive descent -------------
  std::vector<char32_t> p_;
  size_t at_ = 0;
  bool icase_ = false;

  bool More() const { return at_ < p_.size(); }
  char32_t Peek() const { return p_[at_]; }
  int Emit(Op op, char32_t c = 0, int x = 0, int y = 0) {
    if (prog_.size() >= kMaxProgram) Refuse("a Replace regex that compiles to more than 100000 instructions");
    prog_.push_back(Inst{op, c, x, y});
    return int(prog_.size()) - 1;
  }

  void Compile(const std::string& pattern) {
    p_ = Runes(pattern);
    if (p_.size() >= 4 && p_[0] == '(' && p_[1] == '?' && p_[2] == 'i' && p_[3] == ')') {
      for (char32_t r : p_) {
        if (r >= 0x80) Unsupported("a case-insensitive Replace regex with a non-ASCII character is not implemented");
      }
      icase_ = true;
      at_ = 4;
    }
    Alternation();
    if (More()) Refuse("a Replace normalizer's regex: unexpected " + std::string(1, char(Peek())));
    Emit(kMatch);
    if (MatchesEmpty()) {
      Unsupported("this Replace regex can match the empty string, whose replacement rules differ between engines");
    }
  }

  // Each branch starts with a placeholder split (a plain jump for the last one), so no instruction is
  // ever inserted and every target stays valid.
  void Alternation() {
    std::vector<int> jumps;
    while (true) {
      int split = Emit(kSplit);
      Concatenation();
      if (More() && Peek() == '|') {
        at_++;
        jumps.push_back(Emit(kJmp));  // the finished branch jumps over the rest
        prog_[size_t(split)].x = split + 1;
        prog_[size_t(split)].y = int(prog_.size());
        continue;
      }
      prog_[size_t(split)] = Inst{kJmp, 0, split + 1, 0};
      break;
    }
    for (int j : jumps) prog_[size_t(j)].x = int(prog_.size());
  }

  void Concatenation() {
    while (More() && Peek() != '|' && Peek() != ')') Repeat();
  }

  void Repeat() {
    int start = int(prog_.size());
    Atom();
    int end = int(prog_.size());
    while (More()) {
      char32_t c = Peek();
      int lo, hi;  // hi < 0: unbounded
      if (c == '*') { lo = 0; hi = -1; at_++; }
      else if (c == '+') { lo = 1; hi = -1; at_++; }
      else if (c == '?') { lo = 0; hi = 1; at_++; }
      else if (c == '{') {
        size_t save = at_;
        if (!Counts(&lo, &hi)) { at_ = save; break; }
      } else break;
      bool lazy = More() && Peek() == '?';
      if (lazy) at_++;
      // Oniguruma (the crate's engine) ends a repeat whose body matched nothing; a backtracker
      // that remembers failed states would end it differently, so such a repeat is refused.
      if (!(lo == 0 && hi == 1) && Nullable(start, end)) {
        Unsupported("a Replace regex repeats a group that can match the empty string, whose repeat semantics differ between engines");
      }
      std::vector<Inst> body(prog_.begin() + start, prog_.begin() + end);
      for (Inst& in : body) {
        if (in.op == kSplit || in.op == kJmp) {
          in.x -= start;
          if (in.op == kSplit) in.y -= start;
        }
      }
      prog_.resize(size_t(start));
      if (lo > 1000 || hi > 1000) Refuse("a Replace regex repeat count above 1000");
      auto copy = [&]() {
        if (prog_.size() + body.size() > kMaxProgram) Refuse("a Replace regex that compiles to more than 100000 instructions");
        int base = int(prog_.size());
        for (Inst in : body) {
          if (in.op == kSplit || in.op == kJmp) {
            in.x += base;
            if (in.op == kSplit) in.y += base;
          }
          prog_.push_back(in);
        }
      };
      for (int i = 0; i < lo; i++) copy();
      if (hi < 0) {  // L: split(body, out); body; jmp L
        int l = int(prog_.size());
        int s = Emit(kSplit);
        copy();
        Emit(kJmp, 0, l);
        int out = int(prog_.size());
        prog_[size_t(s)].x = lazy ? out : s + 1;
        prog_[size_t(s)].y = lazy ? s + 1 : out;
      } else {
        std::vector<int> splits;
        for (int i = lo; i < hi; i++) {
          splits.push_back(Emit(kSplit));
          copy();
        }
        int out = int(prog_.size());
        for (int s : splits) {
          prog_[size_t(s)].x = lazy ? out : s + 1;
          prog_[size_t(s)].y = lazy ? s + 1 : out;
        }
      }
      end = int(prog_.size());
    }
  }

  // Can the instructions [start, end) reach `end` without consuming a character (anchors pass)?
  bool Nullable(int start, int end) const {
    std::vector<char> seen(size_t(end - start) + 1, 0);
    std::vector<int> todo{start};
    while (!todo.empty()) {
      int pc = todo.back();
      todo.pop_back();
      if (pc == end) return true;
      if (pc < start || pc > end || seen[size_t(pc - start)]) continue;
      seen[size_t(pc - start)] = 1;
      const Inst& in = prog_[size_t(pc)];
      if (in.op == kJmp) todo.push_back(in.x);
      else if (in.op == kSplit) { todo.push_back(in.x); todo.push_back(in.y); }
      else if (in.op == kBol || in.op == kEol) todo.push_back(pc + 1);
    }
    return false;
  }

  bool Counts(int* lo, int* hi) {  // at '{'; false when it is not a repeat count (then a literal)
    at_++;
    auto number = [&](int* v) {
      if (!More() || Peek() < '0' || Peek() > '9') return false;
      long n = 0;
      while (More() && Peek() >= '0' && Peek() <= '9') {
        n = n * 10 + long(Peek() - '0');
        if (n > 100000) n = 100000;
        at_++;
      }
      *v = int(n);
      return true;
    };
    if (!number(lo)) return false;
    *hi = *lo;
    if (More() && Peek() == ',') {
      at_++;
      if (More() && Peek() == '}') *hi = -1;
      else if (!number(hi)) return false;
    }
    if (!More() || Peek() != '}') return false;
    at_++;
    if (*hi >= 0 && *hi < *lo) Refuse("a Replace regex repeat count with its bounds reversed");
    return true;
  }

  void Atom() {
    char32_t c = Peek();
    if (c == '(') {
      at_++;
      if (More() && Peek() == '?') {
        if (at_ + 1 < p_.size() && p_[at_ + 1] == ':') at_ += 2;
        else Unsupported("this Replace regex group syntax (?...) is not implemented");
      }
      if (++depth_ > kMaxDepth) Refuse("a Replace regex nested more than 256 groups deep");
      Alternation();
      depth_--;
      if (!More() || Peek() != ')') Refuse("a Replace normalizer's regex: missing )");
      at_++;
    } else if (c == '[') {
      Class();
    } else if (c == '.') {
      at_++;
      Emit(kAny);
    } else if (c == '^') {
      at_++;
      Emit(kBol);
    } else if (c == '$') {
      at_++;
      Emit(kEol);
    } else if (c == '*' || c == '+' || c == '?') {
      Refuse("a Replace normalizer's regex: nothing to repeat");
    } else if (c == '\\') {
      Literal(Escape());
    } else {
      at_++;
      Literal(c);
    }
  }

  char32_t Escape() {  // at '\\'
    at_++;
    if (!More()) Refuse("a Replace normalizer's regex ends in a backslash");
    char32_t c = Peek();
    at_++;
    switch (c) {
      case 't': return '\t';
      case 'n': return '\n';
      case 'r': return '\r';
      case 'f': return '\f';
      case 'v': return 0x0B;
      default: break;
    }
    if (c < 0x80 && !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) return c;
    Unsupported("the Replace regex escape \\" + std::string(1, char(c < 0x80 ? c : '?')) + " is not implemented");
  }

  // Oniguruma folds with full Unicode rules: K (Kelvin) is k, the long s and the sharp s are s/ss, and
  // the ligatures ff, fi, fl, ffi, ffl, st fold to letter pairs. An ASCII-only fold would differ.
  void FoldTrap(char32_t lo, char32_t hi) const {
    if (!icase_) return;
    for (char32_t t : {U'k', U'K', U's', U'S', U'f', U'F'}) {
      if (lo <= t && t <= hi) Unsupported("a case-insensitive Replace regex with k, s or f, whose Unicode case folding is not implemented");
    }
  }

  void Literal(char32_t c) {
    FoldTrap(c, c);
    if (icase_ && ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
      Set s;
      s.ranges = {{c | 0x20, c | 0x20}, {c & ~char32_t(0x20), c & ~char32_t(0x20)}};
      sets_.push_back(s);
      Emit(kSet, 0, int(sets_.size()) - 1);
    } else {
      Emit(kChar, c);
    }
  }

  void Class() {  // at '['
    at_++;
    Set s;
    if (More() && Peek() == '^') {
      s.negate = true;
      at_++;
    }
    bool first = true;
    while (true) {
      if (!More()) Refuse("a Replace normalizer's regex: missing ]");
      char32_t c = Peek();
      if (c == ']' && !first) { at_++; break; }
      first = false;
      if (c == '[') Unsupported("a nested [ or a POSIX class in a Replace regex is not implemented");
      char32_t lo = c == '\\' ? Escape() : (at_++, c);
      char32_t hi = lo;
      if (More() && Peek() == '-' && at_ + 1 < p_.size() && p_[at_ + 1] != ']') {
        at_++;
        char32_t h = Peek();
        if (h == '[') Unsupported("a nested [ in a Replace regex is not implemented");
        hi = h == '\\' ? Escape() : (at_++, h);
        if (hi < lo) Refuse("a Replace normalizer's regex: a range with its ends reversed");
      }
      FoldTrap(lo, hi);
      s.ranges.push_back({lo, hi});
    }
    if (icase_) {
      auto more = s.ranges;
      for (auto& r : s.ranges) {
        char32_t a = std::max<char32_t>(r.first, 'a'), b = std::min<char32_t>(r.second, 'z');
        if (a <= b) more.push_back({a & ~char32_t(0x20), b & ~char32_t(0x20)});
        a = std::max<char32_t>(r.first, 'A');
        b = std::min<char32_t>(r.second, 'Z');
        if (a <= b) more.push_back({a | 0x20, b | 0x20});
      }
      s.ranges = more;
    }
    sets_.push_back(s);
    Emit(kSet, 0, int(sets_.size()) - 1);
  }

  // --- matcher --------------------------------------------------------------------------------
  bool Find(const std::string& text, size_t from, size_t* ms, size_t* me, Memo& seen) const {
    std::vector<std::pair<int, size_t>> stack;
    for (size_t start = from; start <= text.size();) {
      stack.assign(1, {0, start});
      while (!stack.empty()) {
        auto [pc, pos] = stack.back();
        stack.pop_back();
        while (true) {
          if (!seen.Visit(pc, pos)) break;
          const Inst& in = prog_[size_t(pc)];
          if (in.op == kMatch) {
            *ms = start;
            *me = pos;
            return true;
          }
          if (in.op == kJmp) { pc = in.x; continue; }
          if (in.op == kSplit) {
            stack.push_back({in.y, pos});
            pc = in.x;
            continue;
          }
          if (in.op == kBol) { if (pos != 0 && text[pos - 1] != '\n') break; pc++; continue; }
          if (in.op == kEol) { if (pos != text.size() && text[pos] != '\n') break; pc++; continue; }
          if (pos >= text.size()) break;
          auto d = ud::DecodeRune(std::string_view(text).substr(pos));
          bool ok;
          if (in.op == kChar) ok = d.rune == in.c;
          else if (in.op == kAny) ok = d.rune != '\n';
          else {
            const Set& s = sets_[size_t(in.x)];
            bool hit = false;
            for (auto& r : s.ranges) hit = hit || (d.rune >= r.first && d.rune <= r.second);
            ok = hit != s.negate;
          }
          if (!ok) break;
          pos += d.width;
          pc++;
        }
      }
      start += start < text.size() ? ud::DecodeRune(std::string_view(text).substr(start)).width : 1;
    }
    return false;
  }
};

// ------------------------------------------------------------------------------------------
// Component sections of tokenizer.json.
// ------------------------------------------------------------------------------------------
using Normalizer = std::function<std::string(const std::string&)>;

const Value* Member(const Value& node, const char* key) {
  const Value* v = node.Find(key);
  return (v == nullptr || v->is_null()) ? nullptr : v;
}
std::string Str(const Value& n, const char* key, const std::string& def) {
  const Value* v = Member(n, key);
  if (v == nullptr) return def;
  if (v->kind() != Value::Kind::String) Refuse(std::string("\"") + key + "\" must be a string, found " + pyvalue::TypeName(*v));
  return v->as_string();
}
bool Bool(const Value& n, const char* key, bool def) {
  const Value* v = Member(n, key);
  if (v == nullptr) return def;
  if (v->kind() != Value::Kind::Bool) Refuse(std::string("\"") + key + "\" must be a boolean, found " + pyvalue::TypeName(*v));
  return v->as_bool();
}
int Int(const Value& n, const char* key, int def) {
  const Value* v = Member(n, key);
  if (v == nullptr) return def;
  if (v->kind() == Value::Kind::Int && v->as_int() >= -1000000000 && v->as_int() <= 1000000000) return int(v->as_int());
  if (v->kind() == Value::Kind::Float && v->as_float() >= -1e9 && v->as_float() <= 1e9 && v->as_float() == double(int(v->as_float()))) {
    return int(v->as_float());
  }
  Refuse(std::string("\"") + key + "\" must be an integer");
}
const Value& Object(const Value& n, const char* key) {
  const Value* v = n.Find(key);
  if (v == nullptr || v->kind() != Value::Kind::Object) Refuse(std::string("\"") + key + "\" must be an object");
  return *v;
}
const Value& Array(const Value& n, const char* key) {
  const Value* v = n.Find(key);
  if (v == nullptr || v->kind() != Value::Kind::Array) Refuse(std::string("\"") + key + "\" must be an array");
  return *v;
}
std::string TypeOf(const Value& n, const char* what) {
  if (n.kind() != Value::Kind::Object) Refuse(std::string(what) + " must be an object or null");
  std::string kind = Str(n, "type", "");
  if (kind.empty()) Refuse(std::string("a ") + what + " must declare a type");
  return kind;
}

bool DialectDiffers(const std::string& expr, char* which) {
  for (size_t i = 0; i + 1 < expr.size(); i++) {
    if (expr[i] != '\\') continue;
    i++;
    if (std::string("sSdDwWbBpP").find(expr[i]) != std::string::npos) {
      *which = expr[i];
      return true;
    }
  }
  return false;
}

std::string ReplaceAllLiteral(const std::string& s, const std::string& from, const std::string& to) {
  std::string out;
  size_t at = 0;
  while (true) {
    size_t hit = s.find(from, at);
    if (hit == std::string::npos) break;
    out.append(s, at, hit - at);
    out += to;
    at = hit + from.size();
  }
  out.append(s, at, std::string::npos);
  return out;
}

Normalizer ParseNormalizer(const Value* n, bool* may_drop = nullptr) {
  if (n == nullptr || n->is_null()) return [](const std::string& s) { return s; };
  std::string kind = TypeOf(*n, "normalizer");
  if (kind == "NFC") return [](const std::string& s) { return NormalizeImpl(s, Form::NFC); };
  if (kind == "NFD") return [](const std::string& s) { return NormalizeImpl(s, Form::NFD); };
  if (kind == "NFKC") return [](const std::string& s) { return NormalizeImpl(s, Form::NFKC); };
  if (kind == "NFKD") return [](const std::string& s) { return NormalizeImpl(s, Form::NFKD); };
  if (kind == "Lowercase") return [](const std::string& s) { return LowercaseImpl(s); };
  if (kind == "Prepend") {
    std::string prefix = Str(*n, "prepend", "");
    return [prefix](const std::string& s) { return s.empty() ? s : prefix + s; };  // non-empty text only
  }
  if (kind == "Strip") {
    bool l = Bool(*n, "strip_left", true), r = Bool(*n, "strip_right", true);
    if (l && may_drop != nullptr) *may_drop = true;
    return [l, r](const std::string& s) { return StripImpl(s, l, r); };
  }
  if (kind == "Replace") {
    if (may_drop != nullptr) *may_drop = true;
    const Value& pattern = Object(*n, "pattern");
    const Value* content = n->Find("content");
    if (content == nullptr || content->kind() != Value::Kind::String) Refuse("a Replace normalizer must declare content");
    std::string to = content->as_string();
    if (const Value* lit = pattern.Find("String"); lit != nullptr && lit->kind() == Value::Kind::String) {
      std::string from = lit->as_string();
      if (from.empty()) Refuse("a Replace normalizer cannot match the empty string");
      return [from, to](const std::string& s) { return ReplaceAllLiteral(s, from, to); };
    }
    const Value* rx = pattern.Find("Regex");
    if (rx == nullptr || rx->kind() != Value::Kind::String) {
      Refuse("a Replace normalizer's pattern must be {\"String\": ...} or {\"Regex\": ...}");
    }
    char which;
    if (DialectDiffers(rx->as_string(), &which)) {
      Unsupported(std::string("the Replace regex uses \\") + which +
                  ", whose meaning differs between the reference's regex engine (Unicode-aware) and "
                  "an ASCII one, so it would normalize differently; refusing to load");
    }
    auto re = std::make_shared<Regex>(rx->as_string());
    return [re, to](const std::string& s) { return re->ReplaceAll(s, to); };
  }
  if (kind == "Sequence") {
    std::vector<Normalizer> steps;
    for (const Value& child : Array(*n, "normalizers").items()) {
      if (child.kind() != Value::Kind::Object) Refuse("a normalizer in a Sequence must be an object");
      steps.push_back(ParseNormalizer(&child, may_drop));
    }
    return [steps](const std::string& s) {
      std::string cur = s;
      for (const auto& step : steps) cur = step(cur);
      return cur;
    };
  }
  Unsupported("the normalizer \"" + kind + "\" is not implemented; refusing to load rather than produce ids "
              "that are wrong in a way no answer-level test would localise");
}

// The byte-level stand-in alphabet: printable ASCII and Latin-1 bytes map to themselves, the other
// 68 to U+0100 upward.
struct Alphabet {
  char32_t to_rune[256];
  int16_t to_byte[0x144];
  Alphabet() {
    std::fill(std::begin(to_byte), std::end(to_byte), int16_t(-1));
    int lifted = 0;
    for (int b = 0; b < 256; b++) {
      if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) to_rune[b] = char32_t(b);
      else to_rune[b] = char32_t(256 + lifted++);
      to_byte[to_rune[b]] = int16_t(b);
    }
  }
  int ByteFor(char32_t r) const { return r >= 0x144 ? -1 : to_byte[r]; }
  std::string Encode(std::string_view piece) const {
    std::string out;
    out.reserve(piece.size() + piece.size() / 4);
    for (unsigned char c : piece) ud::AppendRune(out, to_rune[c]);
    return out;
  }
};
const Alphabet& TheAlphabet() {
  static const Alphabet a;
  return a;
}

class PreTokenizer {
 public:
  virtual ~PreTokenizer() = default;
  // `origin`: the text starts where the input did (not after an added token).
  virtual void Split(std::string_view text, std::vector<std::string>& out, bool origin) const = 0;
  virtual bool PrependsFirstOnly() const { return false; }
};

class WholeText : public PreTokenizer {
 public:
  void Split(std::string_view text, std::vector<std::string>& out, bool) const override {
    if (!text.empty()) out.emplace_back(text);
  }
};

class ByteLevel : public PreTokenizer {
 public:
  ByteLevel(bool prefix, bool regex) : add_prefix_space_(prefix), use_regex_(regex) {}
  void Split(std::string_view text, std::vector<std::string>& out, bool) const override {
    if (text.empty()) return;
    std::string source(text);
    if (add_prefix_space_ && source[0] != ' ') source = " " + source;
    if (!use_regex_) {
      out.push_back(TheAlphabet().Encode(source));
      return;
    }
    for (size_t at = 0; at < source.size();) {
      size_t end = Cut(source, at);
      out.push_back(TheAlphabet().Encode(std::string_view(source).substr(at, end - at)));
      at = end;
    }
  }

 private:
  // The GPT-2 pattern  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  // as a scanner: the pattern has a lookahead, and no standard regex has it with Unicode classes.
  static bool InClass(char32_t r, int branch) {
    if (branch == 0) return IsLetter(r);
    if (branch == 1) return IsNumber(r);
    return !IsWhiteSpace(r) && !IsLetter(r) && !IsNumber(r);
  }
  static size_t RunEnd(const std::string& s, size_t at, int branch) {
    while (at < s.size()) {
      auto d = ud::DecodeRune(std::string_view(s).substr(at));
      if (!InClass(d.rune, branch)) break;
      at += d.width;
    }
    return at;
  }
  static size_t Cut(const std::string& s, size_t from) {
    static const char* const kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    std::string_view rest = std::string_view(s).substr(from);
    for (const char* c : kContractions) {
      std::string_view cv(c);
      if (rest.substr(0, cv.size()) == cv) return from + cv.size();
    }
    auto first = ud::DecodeRune(rest);
    for (int branch = 0; branch < 3; branch++) {
      if (first.rune == ' ') {
        size_t after = from + 1;
        if (after < s.size() && InClass(ud::DecodeRune(std::string_view(s).substr(after)).rune, branch)) {
          return RunEnd(s, after, branch);
        }
      } else if (InClass(first.rune, branch)) {
        return RunEnd(s, from, branch);
      }
    }
    if (IsWhiteSpace(first.rune)) {
      size_t end = from;
      while (end < s.size()) {
        auto d = ud::DecodeRune(std::string_view(s).substr(end));
        if (!IsWhiteSpace(d.rune)) break;
        end += d.width;
      }
      if (end >= s.size()) return end;
      // \s+(?!\S) gives the last white-space character back to the next piece.
      size_t w = ud::DecodeLastRune(std::string_view(s).substr(0, end)).width;
      if (end - w > from) return end - w;
      return end;  // a single space: \s+(?!\S) fails and \s+ takes it
    }
    return from + first.width;
  }
  bool add_prefix_space_, use_regex_;
};

class Metaspace : public PreTokenizer {
 public:
  Metaspace(std::string replacement, bool prepend, bool first_only, bool split)
      : replacement_(std::move(replacement)), prepend_(prepend), first_only_(first_only), split_(split) {}
  bool PrependsFirstOnly() const override { return prepend_ && first_only_; }
  void Split(std::string_view text, std::vector<std::string>& out, bool origin) const override {
    if (text.empty()) return;
    std::string source = ReplaceAllLiteral(std::string(text), " ", replacement_);
    if (prepend_ && (origin || !first_only_) && source.compare(0, replacement_.size(), replacement_) != 0) source = replacement_ + source;
    if (!split_) {
      out.push_back(source);
      return;
    }
    size_t start = 0;
    size_t from = ud::DecodeRune(source).width;  // a marker at the very start does not split
    while (true) {
      size_t next = source.find(replacement_, from);
      if (next == std::string::npos) break;
      out.push_back(source.substr(start, next - start));
      start = next;
      from = next + replacement_.size();
    }
    if (start < source.size()) out.push_back(source.substr(start));
  }

 private:
  std::string replacement_;
  bool prepend_, first_only_, split_;
};

class SequencePre : public PreTokenizer {
 public:
  std::vector<std::unique_ptr<PreTokenizer>> steps;
  bool PrependsFirstOnly() const override {
    for (const auto& step : steps) {
      if (step->PrependsFirstOnly()) return true;
    }
    return false;
  }
  void Split(std::string_view text, std::vector<std::string>& out, bool origin) const override {
    std::vector<std::string> current{std::string(text)};
    for (const auto& step : steps) {
      std::vector<std::string> next;
      // Every pre-tokenizer here keeps the text whole, so only the first piece is still at the start.
      for (size_t i = 0; i < current.size(); i++) step->Split(current[i], next, origin && i == 0);
      current.swap(next);
    }
    out.insert(out.end(), current.begin(), current.end());
  }
};

std::unique_ptr<PreTokenizer> ParsePreTokenizer(const Value* n) {
  if (n == nullptr || n->is_null()) return std::make_unique<WholeText>();
  std::string kind = TypeOf(*n, "pre_tokenizer");
  if (kind == "ByteLevel") return std::make_unique<ByteLevel>(Bool(*n, "add_prefix_space", true), Bool(*n, "use_regex", true));
  if (kind == "Metaspace") {
    std::string replacement = Str(*n, "replacement", "\xE2\x96\x81");
    if (ud::RuneCount(replacement) != 1) Refuse("a Metaspace replacement must be exactly one code point");
    std::string scheme = Str(*n, "prepend_scheme", "always");
    bool split = Bool(*n, "split", true);
    // The legacy flag must agree with the scheme, as in the crate (true with "never" is accepted).
    if (!Bool(*n, "add_prefix_space", true) && scheme != "never") {
      Refuse("a Metaspace add_prefix_space does not match its declared prepend_scheme");
    }
    if (scheme == "always") return std::make_unique<Metaspace>(replacement, true, false, split);
    if (scheme == "first") return std::make_unique<Metaspace>(replacement, true, true, split);
    if (scheme == "never") return std::make_unique<Metaspace>(replacement, false, false, split);
    Refuse("unknown Metaspace prepend_scheme \"" + scheme + "\"");
  }
  if (kind == "Sequence") {
    auto seq = std::make_unique<SequencePre>();
    for (const Value& child : Array(*n, "pretokenizers").items()) {
      if (child.kind() != Value::Kind::Object) Refuse("a pre-tokenizer in a Sequence must be an object");
      seq->steps.push_back(ParsePreTokenizer(&child));
    }
    return seq;
  }
  Unsupported("the pre-tokenizer \"" + kind + "\" is not implemented; refusing to load rather than produce ids "
              "that are wrong in a way no answer-level test would localise");
}

// ------------------------------------------------------------------------------------------
// Decoders.
// ------------------------------------------------------------------------------------------
class Decoder {
 public:
  virtual ~Decoder() = default;
  virtual void Transform(std::vector<std::string>& tokens) const = 0;
};

class ByteLevelDecoder : public Decoder {
 public:
  // A token holding a character outside the alphabet is taken as plain text; the whole is then
  // decoded as lossy UTF-8.
  void Transform(std::vector<std::string>& tokens) const override {
    std::string raw;
    for (const auto& token : tokens) {
      std::string mapped;
      bool ok = true;
      std::string_view rest(token);
      while (!rest.empty()) {
        auto d = ud::DecodeRune(rest);
        int b = TheAlphabet().ByteFor(d.rune);
        if (b < 0) {
          ok = false;
          break;
        }
        mapped.push_back(char(b));
        rest.remove_prefix(d.width);
      }
      raw += ok ? mapped : token;
    }
    tokens.assign(1, ud::Lossy(raw));
  }
};

class ReplaceDecoder : public Decoder {
 public:
  ReplaceDecoder(std::string p, std::string c) : pattern_(std::move(p)), content_(std::move(c)) {}
  void Transform(std::vector<std::string>& tokens) const override {
    for (auto& t : tokens) t = ReplaceAllLiteral(t, pattern_, content_);
  }

 private:
  std::string pattern_, content_;
};

int HexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}
// "<0xNN>", read as Rust's from_str_radix does: lower-case digits, and a '+' on the high digit
// leaves only the low digit.
bool ByteToken(const std::string& t, unsigned char* out) {
  if (t.size() != 6 || t[0] != '<' || t[1] != '0' || t[2] != 'x' || t[5] != '>') return false;
  if (t[3] == '+') {
    int l = HexValue(t[4]);
    *out = (unsigned char)l;
    return l >= 0;
  }
  int h = HexValue(t[3]), l = HexValue(t[4]);
  if (h < 0 || l < 0) return false;
  *out = (unsigned char)(h << 4 | l);
  return true;
}

class ByteFallbackDecoder : public Decoder {
 public:
  void Transform(std::vector<std::string>& tokens) const override {
    std::vector<std::string> out;
    std::string run;
    auto flush = [&]() {
      if (run.empty()) return;
      if (ud::ValidUtf8(run)) out.push_back(run);
      else for (size_t i = 0; i < run.size(); i++) out.push_back("\xEF\xBF\xBD");
      run.clear();
    };
    for (const auto& token : tokens) {
      unsigned char b;
      if (ByteToken(token, &b)) {
        run.push_back(char(b));
        continue;
      }
      flush();
      out.push_back(token);
    }
    flush();
    tokens.swap(out);
  }
};

class FuseDecoder : public Decoder {
 public:
  void Transform(std::vector<std::string>& tokens) const override {
    std::string all;
    for (const auto& t : tokens) all += t;
    tokens.assign(1, all);
  }
};

class StripDecoder : public Decoder {
 public:
  StripDecoder(std::string c, int start, int stop) : c_(std::move(c)), start_(start), stop_(stop) {}
  void Transform(std::vector<std::string>& tokens) const override {
    for (auto& t : tokens) {
      for (int n = 0; n < start_ && t.compare(0, c_.size(), c_) == 0; n++) t.erase(0, c_.size());
      for (int n = 0; n < stop_ && t.size() >= c_.size() && t.compare(t.size() - c_.size(), c_.size(), c_) == 0; n++) {
        t.erase(t.size() - c_.size());
      }
    }
  }

 private:
  std::string c_;
  int start_, stop_;
};

class SequenceDecoder : public Decoder {
 public:
  std::vector<std::unique_ptr<Decoder>> steps;
  void Transform(std::vector<std::string>& tokens) const override {
    for (const auto& s : steps) s->Transform(tokens);
  }
};

std::unique_ptr<Decoder> ParseDecoder(const Value* n) {
  if (n == nullptr || n->is_null()) return nullptr;
  std::string kind = TypeOf(*n, "decoder");
  if (kind == "ByteLevel") return std::make_unique<ByteLevelDecoder>();
  if (kind == "ByteFallback") return std::make_unique<ByteFallbackDecoder>();
  if (kind == "Fuse") return std::make_unique<FuseDecoder>();
  if (kind == "Replace") {
    const Value* p = n->Find("pattern");
    std::string literal;
    if (p != nullptr && p->kind() == Value::Kind::Object) {
      const Value* s = p->Find("String");
      if (s == nullptr || s->kind() != Value::Kind::String) {
        if (p->Find("Regex") != nullptr) {
          Unsupported("only a literal String pattern is implemented in a Replace decoder; this checkpoint declares a Regex one");
        }
        Refuse("a Replace decoder needs a String pattern");
      }
      literal = s->as_string();
    } else if (p != nullptr && p->kind() == Value::Kind::String) {
      literal = p->as_string();
    } else {
      Refuse("a Replace decoder needs a String pattern");
    }
    if (literal.empty()) Refuse("a Replace decoder cannot match the empty string");
    return std::make_unique<ReplaceDecoder>(literal, Str(*n, "content", ""));
  }
  if (kind == "Strip") {
    std::string c = Str(*n, "content", " ");
    if (ud::RuneCount(c) != 1) Refuse("a Strip decoder's content must be one character");
    return std::make_unique<StripDecoder>(c, Int(*n, "start", 0), Int(*n, "stop", 0));
  }
  if (kind == "Sequence") {
    auto seq = std::make_unique<SequenceDecoder>();
    const Value* steps = n->Find("decoders");
    if (steps == nullptr || steps->kind() != Value::Kind::Array) Refuse("a Sequence decoder needs a decoders array");
    for (const Value& step : steps->items()) {
      if (step.kind() != Value::Kind::Object) Refuse("every Sequence decoder step must be an object");
      seq->steps.push_back(ParseDecoder(&step));
    }
    return seq;
  }
  Unsupported("the decoder \"" + kind + "\" is not implemented (ByteLevel, Replace, ByteFallback, Fuse, Strip and "
              "Sequence are); refusing to load rather than decode it wrongly");
}

// ------------------------------------------------------------------------------------------
// Added tokens.
// ------------------------------------------------------------------------------------------
struct AddedToken {
  std::string content;
  int32_t id;
  bool special, lstrip, rstrip, single_word, normalized;
};

struct Span {
  std::string text;
  int32_t id = 0;
  bool is_token = false;
  size_t offset = 0;  // byte offset of the span in the text that was cut
};

// The added tokens that start with one character, longest first (the reference's leftmost-longest
// match). `second` records which bytes follow that character in any of them, so the common miss is
// rejected without trying each token.
struct Bucket {
  std::vector<const AddedToken*> tokens;
  bool second[256] = {};
  bool whole = false;  // a token is that single character
  void Add(const AddedToken* t, size_t first_width) {
    tokens.push_back(t);
    if (t->content.size() == first_width) whole = true;
    else second[(unsigned char)t->content[first_width]] = true;
  }
};

class AddedVocab {
 public:
  bool empty = true;
  AddedVocab() = default;
  explicit AddedVocab(const std::vector<AddedToken>& tokens) : empty(tokens.empty()), store_(tokens) {
    for (const auto& t : store_) {
      if (t.content.empty()) Refuse("an added token cannot have empty content");
      auto d = ud::DecodeRune(t.content);
      if (d.rune < 0x80) ascii_[d.rune].Add(&t, d.width);
      else other_[d.rune].Add(&t, d.width);
    }
    auto longest_first = [](Bucket& b) {
      std::sort(b.tokens.begin(), b.tokens.end(), [](const AddedToken* x, const AddedToken* y) {
        if (x->content.size() != y->content.size()) return x->content.size() > y->content.size();
        return x->content < y->content;
      });
    };
    for (auto& b : ascii_) longest_first(b);
    for (auto& kv : other_) longest_first(kv.second);
  }

  // Cuts text around its added tokens; lstrip and rstrip tokens absorb the white space beside them.
  std::vector<Span> Split(const std::string& text) const {
    std::vector<Span> out;
    if (text.empty()) return out;
    if (empty) {
      out.push_back(Span{text, 0, false, 0});
      return out;
    }
    size_t plain_from = 0;
    for (size_t at = 0; at < text.size();) {
      auto d = ud::DecodeRune(std::string_view(text).substr(at));
      size_t skip = 0;
      const AddedToken* hit = LongestAt(text, at, d.rune, d.width, &skip);
      if (skip != 0) {  // the crate skips a longest hit that fails single_word, shorter tokens inside it too
        at += skip;
        continue;
      }
      if (hit == nullptr) {
        at += d.width;
        continue;
      }
      size_t start = at, stop = at + hit->content.size();
      if (hit->lstrip) {
        while (start > plain_from) {
          auto p = ud::DecodeLastRune(std::string_view(text).substr(0, start));
          if (!IsWhiteSpace(p.rune)) break;
          start -= p.width;
        }
      }
      if (hit->rstrip) {
        while (stop < text.size()) {
          auto nx = ud::DecodeRune(std::string_view(text).substr(stop));
          if (!IsWhiteSpace(nx.rune)) break;
          stop += nx.width;
        }
      }
      if (start > plain_from) out.push_back(Span{text.substr(plain_from, start - plain_from), 0, false, plain_from});
      out.push_back(Span{hit->content, hit->id, true, start});
      plain_from = at = stop;
    }
    if (plain_from < text.size()) out.push_back(Span{text.substr(plain_from), 0, false, plain_from});
    return out;
  }

 private:
  const Bucket* BucketOf(char32_t r) const {
    if (r < 0x80) return &ascii_[r];
    auto it = other_.find(r);
    return it == other_.end() ? nullptr : &it->second;
  }
  // `skip` is set to the length of a longest hit that fails single_word.
  const AddedToken* LongestAt(const std::string& text, size_t at, char32_t first, size_t width, size_t* skip) const {
    const Bucket* b = BucketOf(first);
    if (b == nullptr || b->tokens.empty()) return nullptr;
    if (!b->whole) {
      size_t next = at + width;
      if (next >= text.size() || !b->second[(unsigned char)text[next]]) return nullptr;
    }
    for (const AddedToken* t : b->tokens) {
      if (text.compare(at, t->content.size(), t->content) != 0) continue;
      if (t->single_word && !StandsAlone(text, at, at + t->content.size())) {
        *skip = t->content.size();
        return nullptr;
      }
      return t;
    }
    return nullptr;
  }
  static bool StandsAlone(const std::string& text, size_t start, size_t stop) {
    if (start > 0 && IsWordChar(ud::DecodeLastRune(std::string_view(text).substr(0, start)).rune)) return false;
    if (stop < text.size() && IsWordChar(ud::DecodeRune(std::string_view(text).substr(stop)).rune)) return false;
    return true;
  }
  std::vector<AddedToken> store_;  // the buckets point into it; a moved vector keeps its buffer
  Bucket ascii_[128];
  std::unordered_map<char32_t, Bucket> other_;
};

// ------------------------------------------------------------------------------------------
// BPE with byte fallback.
// ------------------------------------------------------------------------------------------
constexpr size_t kHeapMinSymbols = 32;
constexpr int32_t kNoMerge = 1 << 30;
constexpr char32_t kSmallRunes = 0x180;

struct MergeRule {
  int32_t rank = kNoMerge;
  int32_t id = -1;
};
struct Symbol {
  int32_t start, end, id;  // id -1: a character the vocabulary lacks
};

uint64_t PairKey(int32_t l, int32_t r) { return (uint64_t(uint32_t(l)) << 32) | uint32_t(r); }

class Bpe {
 public:
  Bpe(const std::unordered_map<std::string, int32_t>& vocab, const std::vector<std::pair<std::string, std::string>>& merges,
      const std::string* unk, bool fuse_unk, bool ignore_merges, bool byte_fallback)
      : vocab_(&vocab), fuse_unk_(fuse_unk), ignore_merges_(ignore_merges) {
    for (char32_t r = 0; r < kSmallRunes; r++) {
      std::string s;
      ud::AppendRune(s, r);
      auto it = vocab.find(s);
      small_[r] = it == vocab.end() ? -1 : it->second;
    }
    if (unk != nullptr) {
      auto it = vocab.find(*unk);
      if (it == vocab.end()) {
        Refuse("the tokenizer names " + *unk + " as its unknown token, but neither the vocabulary nor the added-token "
               "table has an entry for it");
      }
      unk_id_ = it->second;
      has_unk_ = true;
    }
    merges_.reserve(merges.size());
    for (size_t rank = 0; rank < merges.size(); rank++) {
      const auto& pair = merges[rank];
      auto l = vocab.find(pair.first), r = vocab.find(pair.second);
      if (l == vocab.end() || r == vocab.end()) {
        Refuse("merge at rank " + std::to_string(rank) + " uses the symbol \"" + (l == vocab.end() ? pair.first : pair.second) +
               "\", which the vocabulary does not contain; merging on ids would be unsound");
      }
      auto merged = vocab.find(pair.first + pair.second);
      if (merged == vocab.end()) {
        Refuse("merge at rank " + std::to_string(rank) + " produces \"" + pair.first + pair.second +
               "\", which the vocabulary does not contain");
      }
      // The crate builds byte-fallback and unknown tokens before merging; this port applies them after.
      for (const std::string* name : {&pair.first, &pair.second, &merged->first}) {
        unsigned char b;
        if ((byte_fallback && ByteToken(*name, &b)) || (unk != nullptr && *name == *unk)) {
          Unsupported("merge at rank " + std::to_string(rank) + " names the byte or unknown token \"" + *name + "\"");
        }
      }
      merges_.emplace(PairKey(l->second, r->second), MergeRule{int32_t(rank), merged->second});  // the lowest rank wins
    }
    if (byte_fallback) {
      has_bytes_ = true;
      for (int v = 0; v < 256; v++) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02X>", v);
        auto it = vocab.find(name);
        byte_ids_[v] = it == vocab.end() ? -1 : it->second;
      }
    }
  }

  bool has_unk() const { return has_unk_; }
  bool has_byte_fallback() const { return has_bytes_; }

  void EncodePiece(const std::string& piece, std::vector<int32_t>& out) const {
    if (piece.empty()) return;
    if (ignore_merges_) {
      auto it = vocab_->find(piece);
      if (it != vocab_->end()) {
        out.push_back(it->second);
        return;
      }
    }
    std::vector<Symbol> symbols = Initial(piece);
    if (symbols.size() >= kHeapMinSymbols) MergeWithHeap(symbols);
    else MergeByRescan(symbols);
    Emit(piece, symbols, out);
  }

  std::vector<Symbol> Initial(const std::string& piece) const {
    std::vector<Symbol> symbols;
  symbols.reserve(piece.size());
  for (size_t i = 0; i < piece.size();) {
    auto d = ud::DecodeRune(std::string_view(piece).substr(i));
    int32_t id;
    if (d.rune < kSmallRunes && d.rune != ud::kReplacement) {
      id = small_[d.rune];
    } else {
      auto it = vocab_->find(piece.substr(i, d.width));
      id = it == vocab_->end() ? -1 : it->second;
    }
    symbols.push_back(Symbol{int32_t(i), int32_t(i + d.width), id});
    i += d.width;
  }
    return symbols;
  }

  // The two merge strategies must give one answer; the test seam compares them on a piece.
  bool StrategiesAgree(const std::string& piece) const {
    std::vector<Symbol> a = Initial(piece), b = a;
    if (a.empty()) return true;
    MergeByRescan(a);
    MergeWithHeap(b);
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i].start != b[i].start || a[i].end != b[i].end || a[i].id != b[i].id) return false;
    }
    return true;
  }

  // Both strategies are public for the test that they agree.
  void MergeByRescan(std::vector<Symbol>& symbols) const {
    std::vector<MergeRule> rules(symbols.size());
    auto pair = [&](size_t i) { rules[i] = Rule(symbols[i], symbols[i + 1]); };
    for (size_t i = 0; i + 1 < symbols.size(); i++) pair(i);
    while (symbols.size() > 1) {
      int32_t best_rank = kNoMerge;
      long best_at = -1;
      for (size_t i = 0; i + 1 < symbols.size(); i++) {
        if (rules[i].rank < best_rank) {
          best_rank = rules[i].rank;
          best_at = long(i);
        }
      }
      if (best_at < 0) return;
      size_t b = size_t(best_at);
      symbols[b] = Symbol{symbols[b].start, symbols[b + 1].end, rules[b].id};
      symbols.erase(symbols.begin() + long(b) + 1);
      rules.erase(rules.begin() + long(b) + 1);
      if (b + 1 < symbols.size()) pair(b);
      if (b > 0) pair(b - 1);
    }
  }

  void MergeWithHeap(std::vector<Symbol>& symbols) const {
    struct Item {
      int32_t rank, slot, version;
      bool operator>(const Item& o) const { return rank != o.rank ? rank > o.rank : slot > o.slot; }
    };
    size_t n = symbols.size();
    std::vector<int32_t> next(n), prev(n), version(n, 0);
    std::vector<char> dead(n, 0);
    for (size_t i = 0; i < n; i++) {
      prev[i] = int32_t(i) - 1;
      next[i] = int32_t(i) + 1;
    }
    next[n - 1] = -1;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    auto offer = [&](int32_t slot) {
      int32_t right = next[size_t(slot)];
      if (right < 0) return;
      MergeRule r = Rule(symbols[size_t(slot)], symbols[size_t(right)]);
      if (r.rank != kNoMerge) queue.push(Item{r.rank, slot, version[size_t(slot)]});
    };
    for (size_t i = 0; i < n; i++) offer(int32_t(i));
    while (!queue.empty()) {
      Item item = queue.top();
      queue.pop();
      size_t slot = size_t(item.slot);
      if (dead[slot] || version[slot] != item.version) continue;
      int32_t right = next[slot];
      if (right < 0 || dead[size_t(right)]) continue;
      MergeRule r = Rule(symbols[slot], symbols[size_t(right)]);
      if (r.rank == kNoMerge) continue;
      symbols[slot] = Symbol{symbols[slot].start, symbols[size_t(right)].end, r.id};
      dead[size_t(right)] = 1;
      next[slot] = next[size_t(right)];
      if (next[size_t(right)] >= 0) prev[size_t(next[size_t(right)])] = int32_t(slot);
      version[slot]++;
      offer(int32_t(slot));
      if (prev[slot] >= 0) {
        version[size_t(prev[slot])]++;
        offer(prev[slot]);
      }
    }
    std::vector<Symbol> out;
    for (int32_t i = 0; i >= 0 && size_t(i) < n; i = next[size_t(i)]) {
      if (!dead[size_t(i)]) out.push_back(symbols[size_t(i)]);
    }
    symbols.swap(out);
  }

 private:
  MergeRule Rule(const Symbol& l, const Symbol& r) const {
    if (l.id < 0 || r.id < 0) return MergeRule{};
    auto it = merges_.find(PairKey(l.id, r.id));
    return it == merges_.end() ? MergeRule{} : it->second;
  }

  // A symbol outside the vocabulary becomes its UTF-8 bytes when the checkpoint has byte fallback
  // and every byte resolves, else the unknown token (one per run with fuse_unk), else it is dropped.
  void Emit(const std::string& piece, const std::vector<Symbol>& symbols, std::vector<int32_t>& out) const {
    bool unknown_pending = false;
    for (const Symbol& s : symbols) {
      if (s.id >= 0) {
        unknown_pending = false;
        out.push_back(s.id);
        continue;
      }
      if (has_bytes_) {
        bool resolves = true;
        for (int32_t i = s.start; i < s.end; i++) resolves = resolves && byte_ids_[(unsigned char)piece[size_t(i)]] >= 0;
        if (resolves) {
          for (int32_t i = s.start; i < s.end; i++) out.push_back(byte_ids_[(unsigned char)piece[size_t(i)]]);
          unknown_pending = false;
          continue;
        }
      }
      if (!has_unk_) continue;
      if (fuse_unk_ && unknown_pending) continue;
      out.push_back(unk_id_);
      unknown_pending = true;
    }
  }

  int32_t small_[kSmallRunes];
  const std::unordered_map<std::string, int32_t>* vocab_;
  std::unordered_map<uint64_t, MergeRule> merges_;
  int32_t unk_id_ = 0;
  bool has_unk_ = false, fuse_unk_, ignore_merges_, has_bytes_ = false;
  int32_t byte_ids_[256];
};

}  // namespace

// ------------------------------------------------------------------------------------------
// Tokenizer.
// ------------------------------------------------------------------------------------------
struct Impl {
  Normalizer normalizer;
  std::unique_ptr<PreTokenizer> pre;
  std::unique_ptr<Decoder> decoder;
  AddedVocab raw_added, normal_added;
  std::unique_ptr<Bpe> bpe;
  std::unordered_map<std::string, int32_t> ids;
  std::vector<std::string> tokens;
  std::map<std::string, int32_t> specials;
  std::unordered_set<int32_t> is_special;
  std::map<std::string, int32_t> roles;
  std::string mask_token;
  bool byte_level_only = false;
};

namespace {

void ReadMerges(const Value& model, std::vector<std::pair<std::string, std::string>>& out) {
  const Value* merges = model.Find("merges");
  if (merges == nullptr || merges->is_null()) return;
  if (merges->kind() != Value::Kind::Array) Refuse("model.merges must be an array");
  out.reserve(merges->size());
  for (const Value& m : merges->items()) {
    if (m.kind() == Value::Kind::String) {
      const std::string& j = m.as_string();
      bool found = false;
      for (size_t i = 1; i + 1 < j.size(); i++) {
        if (j[i] == ' ') {
          out.emplace_back(j.substr(0, i), j.substr(i + 1));
          found = true;
          break;
        }
      }
      if (!found) Refuse("a space-joined merge must have two non-empty halves, got \"" + j + "\"");
    } else if (m.kind() == Value::Kind::Array && m.size() == 2 && m.items()[0].kind() == Value::Kind::String &&
               m.items()[1].kind() == Value::Kind::String) {
      out.emplace_back(m.items()[0].as_string(), m.items()[1].as_string());
    } else {
      Refuse("a merge must be two strings or one space-joined string");
    }
  }
}

std::shared_ptr<const Impl> Build(std::string_view tokenizer_json, const Value* config) {
  auto parsed = pyvalue::Parse(tokenizer_json);
  if (!parsed.ok()) Refuse("tokenizer.json: " + parsed.error());
  const Value& doc = parsed.value();
  if (doc.kind() != Value::Kind::Object) Refuse("tokenizer.json must be a JSON object");
  const Value* model_v = doc.Find("model");
  if (model_v == nullptr || model_v->kind() != Value::Kind::Object) Refuse("model must be an object");
  const Value& model = *model_v;
  std::string type = Str(model, "type", "");
  if (type != "BPE") {
    Unsupported("this library implements the BPE tokenizer model; the checkpoint declares \"" + type +
                "\". Refusing to load rather than mis-tokenize silently");
  }
  auto impl = std::make_shared<Impl>();
  bool normalizer_may_drop = false;
  impl->normalizer = ParseNormalizer(Member(doc, "normalizer"), &normalizer_may_drop);
  impl->pre = ParsePreTokenizer(Member(doc, "pre_tokenizer"));
  if (normalizer_may_drop && impl->pre->PrependsFirstOnly()) {
    // The crate decides "first" by the original offset of the normalized text, which this port does not track.
    Unsupported("a Metaspace prepend_scheme \"first\" after a normalizer that can delete leading text (Strip, Replace)");
  }
  impl->decoder = ParseDecoder(Member(doc, "decoder"));
  impl->byte_level_only = dynamic_cast<const ByteLevel*>(impl->pre.get()) != nullptr;

  // Added tokens.
  std::vector<AddedToken> added;
  if (const Value* at = Member(doc, "added_tokens")) {
    if (at->kind() != Value::Kind::Array) Refuse("added_tokens must be an array");
    std::map<int32_t, std::string> seen;
    for (const Value& r : at->items()) {
      if (r.kind() != Value::Kind::Object) Refuse("an added_tokens entry must be an object");
      std::string content = Str(r, "content", "");
      if (content.empty()) Refuse("an added_tokens entry must declare content");
      const Value* id = Member(r, "id");
      if (id == nullptr || id->kind() != Value::Kind::Int || id->as_int() < 0 || id->as_int() > INT32_MAX) {
        Refuse("added token \"" + content + "\" has no non-negative id");
      }
      int32_t i = int32_t(id->as_int());
      auto clash = seen.find(i);
      if (clash != seen.end() && clash->second != content) {
        Refuse("added token id " + std::to_string(i) + " is claimed by both \"" + clash->second + "\" and \"" + content + "\"");
      }
      seen[i] = content;
      for (const char* flag : {"special", "lstrip", "rstrip", "single_word", "normalized"}) {
        const Value* f = r.Find(flag);
        if (f == nullptr || f->kind() != Value::Kind::Bool) {
          Refuse("added token \"" + content + "\" must declare " + flag + " as a boolean");
        }
      }
      added.push_back(AddedToken{content, i, Bool(r, "special", false), Bool(r, "lstrip", false), Bool(r, "rstrip", false),
                                 Bool(r, "single_word", false), Bool(r, "normalized", false)});
    }
  }
  const Value* vocab_v = Member(model, "vocab");
  if (vocab_v == nullptr || vocab_v->kind() != Value::Kind::Object) Refuse("model.vocab must be an object");
  auto& vocab = impl->ids;
  vocab.reserve(vocab_v->size() + added.size());
  for (size_t i = 0; i < vocab_v->size(); i++) {
    const Value& v = vocab_v->value(i);
    if (v.kind() != Value::Kind::Int || v.as_int() > INT32_MAX) Refuse("the vocabulary holds an id that is not an integer");
    if (v.as_int() < 0) Refuse("the vocabulary holds a negative id " + std::to_string(v.as_int()));
    vocab[vocab_v->key(i)] = int32_t(v.as_int());
  }
  {
    int32_t next = int32_t(vocab.size());
    for (auto& tok : added) {
      auto known = vocab.find(tok.content);
      int32_t expect = known != vocab.end() ? known->second : next++;
      if (known == vocab.end()) vocab[tok.content] = expect;
      if (tok.id != expect) {
        Refuse("added token \"" + tok.content + "\" declares id " + std::to_string(tok.id) + ", but the reference gives it " +
               std::to_string(expect) + " (its vocabulary id, or the next free id in file order)");
      }
    }
  }
  for (const auto& tok : added) {
    if (tok.special) {
      impl->is_special.insert(tok.id);
      impl->specials[tok.content] = tok.id;
    }
    vocab[tok.content] = tok.id;  // the added-token id wins over the vocabulary's, as in the reference
  }
  int32_t highest = 0;
  for (const auto& kv : vocab) highest = std::max(highest, kv.second);
  // Ids may have gaps, but a dense table sized by a hostile id would be gigabytes.
  if (int64_t(highest) >= int64_t(vocab.size()) * 4 + (1 << 16)) {
    Refuse("the highest token id " + std::to_string(highest) + " is far beyond the " + std::to_string(vocab.size()) +
           " tokens; ids this sparse are refused rather than allocated");
  }
  impl->tokens.assign(size_t(highest) + 1, std::string());
  for (const auto& kv : vocab) {
    std::string& owner = impl->tokens[size_t(kv.second)];
    if (!owner.empty() && owner != kv.first) {
      Refuse("the vocabulary gives id " + std::to_string(kv.second) + " to both \"" + owner + "\" and \"" + kv.first + "\"");
    }
    owner = kv.first;
  }

  std::vector<std::pair<std::string, std::string>> merges;
  ReadMerges(model, merges);
  for (const char* key : {"end_of_word_suffix", "continuing_subword_prefix"}) {
    const Value* v = Member(model, key);
    if (v != nullptr && v->kind() == Value::Kind::String && !v->as_string().empty()) {
      Unsupported(std::string("model.") + key + " is not implemented");
    }
  }
  if (Member(model, "dropout") != nullptr) Unsupported("model.dropout is not implemented");
  std::string unk;
  const Value* unk_v = Member(model, "unk_token");
  if (unk_v != nullptr && unk_v->kind() != Value::Kind::String) Refuse("model.unk_token must be a string or null");
  if (unk_v != nullptr) unk = unk_v->as_string();
  impl->bpe = std::make_unique<Bpe>(vocab, merges, unk_v != nullptr ? &unk : nullptr, Bool(model, "fuse_unk", false),
                                    Bool(model, "ignore_merges", false), Bool(model, "byte_fallback", false));

  // Refuse a tokenizer that could drop text without trace.
  if (!impl->bpe->has_unk() && !impl->bpe->has_byte_fallback()) {
    if (!impl->byte_level_only) {
      Unsupported("this checkpoint declares no unk_token and no byte_fallback, and its pre-tokenizer is not byte-level, "
                  "so a character outside the vocabulary would be dropped without trace. Refusing to load");
    }
    int missing = 0, reachable = 0;
    std::string first;
    for (int b = 0; b < 256; b++) {
      if (b == 0xC0 || b == 0xC1 || b >= 0xF5) continue;  // bytes UTF-8 never contains
      reachable++;
      std::string symbol;
      ud::AppendRune(symbol, TheAlphabet().to_rune[b]);
      if (vocab.find(symbol) == vocab.end()) {
        if (missing == 0) first = symbol;
        missing++;
      }
    }
    if (missing > 0) {
      Refuse("this checkpoint declares no unk_token and no byte_fallback, and " + std::to_string(missing) + " of the " +
             std::to_string(reachable) + " byte-level alphabet symbols that text can reach are absent from its vocabulary "
             "(first: \"" + first + "\"), so some inputs would lose text silently. Refusing to load");
    }
  }

  std::vector<AddedToken> on_raw, on_normalized;
  for (const auto& tok : added) (tok.normalized ? on_normalized : on_raw).push_back(tok);
  impl->raw_added = AddedVocab(on_raw);
  impl->normal_added = AddedVocab(on_normalized);

  // Roles named by tokenizer_config.json.
  if (config != nullptr) {
    static const char* const kRoles[] = {"cls_token", "sep_token", "mask_token", "pad_token", "unk_token"};
    for (const char* key : kRoles) {
      const Value* v = Member(*config, key);
      if (v == nullptr) continue;
      std::string content;
      if (v->kind() == Value::Kind::String) {
        content = v->as_string();
      } else if (v->kind() == Value::Kind::Object && v->Find("content") != nullptr &&
                 v->Find("content")->kind() == Value::Kind::String) {
        content = v->Find("content")->as_string();
      } else {
        Refuse(std::string(key) + " in tokenizer_config.json must be a string or an object with a content field");
      }
      auto it = vocab.find(content);
      if (it == vocab.end()) {
        Refuse("tokenizer_config.json names " + content + " as " + key +
               ", but neither the vocabulary nor the added-token table has it");
      }
      impl->roles[key] = it->second;
      if (std::string(key) == "mask_token") impl->mask_token = content;
    }
  }
  return impl;
}

}  // namespace

Result<Tokenizer> Tokenizer::Parse(std::string_view tokenizer_json, std::optional<std::string_view> config_json) {
  try {
    Value config;
    const Value* cp = nullptr;
    if (config_json.has_value()) {
      auto c = pyvalue::Parse(*config_json);
      if (!c.ok() || c.value().kind() != Value::Kind::Object) {
        return Result<Tokenizer>::Fail("tokenizer_config.json must be a JSON object" + (c.ok() ? std::string() : ": " + c.error()));
      }
      config = std::move(c).value();
      cp = &config;
    }
    return Result<Tokenizer>::Ok(Tokenizer(Build(tokenizer_json, cp)));
  } catch (const Fail& f) {
    return Result<Tokenizer>::Fail(f.msg);
  } catch (const std::bad_alloc&) {
    return Result<Tokenizer>::Fail("out of memory loading the tokenizer");
  }
}

Result<Tokenizer> Tokenizer::LoadDir(const std::string& dir) {
  auto read = [](const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    return true;
  };
  std::string where = dir, json, config;
  if (std::ifstream(dir + "/tokenizer/tokenizer.json").good()) where = dir + "/tokenizer";
  if (!read(where + "/tokenizer.json", &json)) return Result<Tokenizer>::Fail("cannot read " + where + "/tokenizer.json");
  bool has_config = read(where + "/tokenizer_config.json", &config);
  return Parse(json, has_config ? std::optional<std::string_view>(config) : std::nullopt);
}

std::vector<int32_t> Tokenizer::Encode(std::string_view text_in, int64_t limit) const {
  std::vector<int32_t> out;
  if (limit == 0) return out;
  const Impl& t = *impl_;
  std::string text = ud::ToValid(text_in);  // a Python str is always valid
  out.reserve(size_t(limit > 0 && limit < int64_t(text.size() / 3 + 16) ? limit : int64_t(text.size() / 3 + 16)));
  auto full = [&]() { return limit >= 0 && int64_t(out.size()) >= limit; };
  auto plain = [&](const std::string& s, bool origin) {
    std::vector<std::string> pieces;
    t.pre->Split(s, pieces, origin);
    for (const auto& piece : pieces) {
      if (full()) return;
      t.bpe->EncodePiece(piece, out);
    }
  };
  for (const Span& sp : t.raw_added.Split(text)) {
    if (full()) break;
    if (sp.is_token) {
      out.push_back(sp.id);
      continue;
    }
    std::string normalized = t.normalizer(sp.text);
    bool origin = sp.offset == 0;  // only the text at the start of the input is "first"
    if (t.normal_added.empty) {
      plain(normalized, origin);
      continue;
    }
    for (const Span& inner : t.normal_added.Split(normalized)) {
      if (full()) break;
      if (inner.is_token) out.push_back(inner.id);
      else plain(inner.text, origin && inner.offset == 0);
    }
  }
  if (limit >= 0 && int64_t(out.size()) > limit) out.resize(size_t(limit));
  return out;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
  const Impl& t = *impl_;
  std::vector<std::string> pieces;
  pieces.reserve(ids.size());
  for (int32_t id : ids) {
    if (skip_special_tokens && t.is_special.count(id) != 0) continue;
    if (id < 0 || size_t(id) >= t.tokens.size() || t.tokens[size_t(id)].empty()) continue;
    pieces.push_back(t.tokens[size_t(id)]);
  }
  std::string out;
  if (t.decoder == nullptr) {  // with no decoder section the reference joins tokens with a space
    for (size_t i = 0; i < pieces.size(); i++) out += (i ? " " : "") + pieces[i];
    return out;
  }
  t.decoder->Transform(pieces);
  for (const auto& p : pieces) out += p;
  return out;
}

size_t Tokenizer::VocabSize() const { return impl_->ids.size(); }
int32_t Tokenizer::HighestId() const { return int32_t(impl_->tokens.size()) - 1; }
std::optional<int32_t> Tokenizer::TokenToId(std::string_view token) const {
  auto it = impl_->ids.find(std::string(token));
  if (it == impl_->ids.end()) return std::nullopt;
  return it->second;
}
std::optional<std::string> Tokenizer::IdToToken(int32_t id) const {
  if (id < 0 || size_t(id) >= impl_->tokens.size() || impl_->tokens[size_t(id)].empty()) return std::nullopt;
  return impl_->tokens[size_t(id)];
}
std::optional<int32_t> Tokenizer::SpecialId(std::string_view content) const {
  auto it = impl_->specials.find(std::string(content));
  if (it == impl_->specials.end()) return std::nullopt;
  return it->second;
}
std::map<std::string, int32_t> Tokenizer::SpecialTokens() const { return impl_->specials; }
static std::optional<int32_t> Role(const Impl& i, const char* key) {
  auto it = i.roles.find(key);
  if (it == i.roles.end()) return std::nullopt;
  return it->second;
}
std::optional<int32_t> Tokenizer::Cls() const { return Role(*impl_, "cls_token"); }
std::optional<int32_t> Tokenizer::Sep() const { return Role(*impl_, "sep_token"); }
std::optional<int32_t> Tokenizer::Mask() const { return Role(*impl_, "mask_token"); }
std::optional<int32_t> Tokenizer::Pad() const { return Role(*impl_, "pad_token"); }
std::string Tokenizer::MaskToken() const { return impl_->mask_token; }

size_t Tokenizer::CountMergeDisagreements(std::string_view text, size_t* pieces_checked) const {
  size_t bad = 0;
  std::vector<std::string> pieces;
  impl_->pre->Split(impl_->normalizer(ud::ToValid(text)), pieces, true);
  for (const auto& p : pieces) {
    if (!impl_->bpe->StrategiesAgree(p)) bad++;
    if (pieces_checked != nullptr) (*pieces_checked)++;
  }
  return bad;
}

std::string Tokenizer::NormalizeForTest(std::string_view text) const { return impl_->normalizer(ud::ToValid(text)); }
std::vector<std::string> Tokenizer::PreTokenizeForTest(std::string_view normalized) const {
  std::vector<std::string> out;
  impl_->pre->Split(normalized, out, true);
  return out;
}

std::string Normalize(std::string_view text, Form form) { return NormalizeImpl(text, form); }
std::string Lowercase(std::string_view text) { return LowercaseImpl(text); }

Result<std::string> RunNormalizerSpec(std::string_view spec_json, std::string_view input) {
  try {
    auto v = pyvalue::Parse(spec_json);
    if (!v.ok()) return Result<std::string>::Fail(v.error());
    return Result<std::string>::Ok(ParseNormalizer(&v.value())(std::string(input)));
  } catch (const Fail& f) {
    return Result<std::string>::Fail(f.msg);
  }
}

Result<std::vector<std::string>> RunPreTokenizerSpec(std::string_view spec_json, std::string_view input) {
  try {
    auto v = pyvalue::Parse(spec_json);
    if (!v.ok()) return Result<std::vector<std::string>>::Fail(v.error());
    std::vector<std::string> out;
    ParsePreTokenizer(&v.value())->Split(input, out, true);
    return Result<std::vector<std::string>>::Ok(out);
  } catch (const Fail& f) {
    return Result<std::vector<std::string>>::Fail(f.msg);
  }
}

}  // namespace tokenizer
}  // namespace laya
