#include "laya/lang.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "laya/pyjson.hpp"
#include "laya/unicodedata.hpp"

namespace laya {
namespace lang {
namespace ud = unicodedata;
namespace {
#include "lang_tables.inc"

using Set = std::unordered_set<std::string>;
constexpr size_t kScripts = sizeof(k_script_names) / sizeof(k_script_names[0]);
constexpr int kShoutedMinStopword = 4, kShoutedMinWord = 5;
constexpr int kLatinBelowCounts = 0x02B0, kLatinBelowScriptOf = 0x0250;

template <size_t N>
Set MakeSet(const char* const (&w)[N]) {
  return Set(w, w + N);
}
struct Tables {
  Set short_swedish = MakeSet(k_short_swedish), shared = MakeSet(k_shared), nordic = MakeSet(k_nordic_overlap),
      en_only = MakeSet(k_en_only), en_collision = MakeSet(k_en_collision), openers{"kan", "jag", "vi"};
  std::vector<std::pair<std::string, Set>> stops;
  Tables() {
    for (const StopSpec& s : k_stop_lists) stops.emplace_back(s.code, Set(s.words, s.words + s.count));
  }
  const Set* Stop(const std::string& code) const {
    for (const auto& s : stops) {
      if (s.first == code) return &s.second;
    }
    return nullptr;
  }
};
const Tables& T() {
  static const Tables t;
  return t;
}

template <class F>
void Each(std::string_view s, F fn) {  // fn(byte offset, rune, width)
  for (size_t i = 0; i < s.size();) {
    ud::Decoded d = ud::DecodeRune(s.substr(i));
    fn(i, d.rune, d.width);
    i += d.width;
  }
}
bool Has(const Set& s, const std::string& w) { return s.count(w) != 0; }
bool Intersects(const Set& a, const Set& b) {
  for (const auto& w : a) {
    if (b.count(w)) return true;
  }
  return false;
}
bool IsDiacritic(char32_t r) { return std::binary_search(std::begin(k_non_en_diacritics), std::end(k_non_en_diacritics), uint32_t(r)); }
int CountAlpha(std::string_view t) {
  int n = 0;
  Each(t, [&](size_t, char32_t r, size_t) { n += ud::IsAlpha(r); });
  return n;
}
bool HasDiacritic(const std::string& w) {
  bool h = false;
  Each(w, [&](size_t, char32_t r, size_t) { h = h || IsDiacritic(r); });
  return h;
}
std::vector<std::string> Words(std::string_view text) {
  std::vector<std::string> out;
  long start = -1;
  Each(text, [&](size_t i, char32_t r, size_t) {
    if (ud::IsWord(r)) {
      if (start < 0) start = long(i);
    } else if (start >= 0) {
      out.emplace_back(text.substr(size_t(start), i - size_t(start)));
      start = -1;
    }
  });
  if (start >= 0) out.emplace_back(text.substr(size_t(start)));
  return out;
}
bool IsIdentChar(char32_t r) { return r == '-' || ud::IsPythonWord(r); }
size_t RunEnd(std::string_view t, size_t from) {
  size_t i = from;
  while (i < t.size()) {
    ud::Decoded d = ud::DecodeRune(t.substr(i));
    if (!IsIdentChar(d.rune)) break;
    i += d.width;
  }
  return i;
}
long MatchIdentifier(std::string_view t, size_t from) {  // [\w-]*(?:[.@][\w-]+)+
  size_t i = RunEnd(t, from);
  int groups = 0;
  while (i < t.size() && (t[i] == '.' || t[i] == '@')) {
    size_t after = RunEnd(t, i + 1);
    if (after == i + 1) break;
    i = after;
    groups++;
  }
  return groups == 0 ? -1 : long(i);
}
std::string SubstituteIdentifiers(std::string_view t) {
  std::string out;
  size_t copied = 0;
  bool replaced = false;
  for (size_t i = 0; i < t.size();) {
    size_t w = ud::DecodeRune(t.substr(i)).width;
    if (i > 0 && IsIdentChar(ud::DecodeLastRune(t.substr(0, i)).rune)) {
      i += w;
      continue;
    }
    long end = MatchIdentifier(t, i);
    if (end < 0) {
      i += w;
      continue;
    }
    replaced = true;
    out.append(t.substr(copied, i - copied));
    out.push_back(' ');
    i = copied = size_t(end);
  }
  if (!replaced) return std::string(t);
  out.append(t.substr(copied));
  return out;
}
bool HasCodeLine(std::string_view t) {
  for (size_t i = 0; i < t.size(); i++) {
    char c = t[i];
    if (c == '=' || c == ';' || c == '{' || c == '}' || c == '[' || c == ']') return true;
    if (c == '(' && i > 0 && ud::IsPythonWord(ud::DecodeLastRune(t.substr(0, i)).rune)) return true;
  }
  return false;
}
bool HasJoined(std::string_view tok) {
  bool found = false;
  Each(tok, [&](size_t i, char32_t r, size_t w) {
    if (found || (r != '.' && r != '_' && r != '/' && r != '\\')) return;
    (void)w;
    if (i == 0 || i + 1 >= tok.size()) return;
    if (ud::IsWordOrDigit(ud::DecodeLastRune(tok.substr(0, i)).rune) && ud::IsWordOrDigit(ud::DecodeRune(tok.substr(i + 1)).rune)) found = true;
  });
  return found;
}
template <class Keep, class Fn>
void EachWordRun(std::string_view text, Keep keep, Fn fn) {
  long start = -1;
  size_t last = 0;
  Each(text, [&](size_t i, char32_t r, size_t) {
    if (ud::IsWord(r)) {
      if (start < 0) {
        if (i > last) keep(text.substr(last, i - last));
        start = long(i);
      }
    } else if (start >= 0) {
      fn(text.substr(size_t(start), i - size_t(start)));
      start = -1;
      last = i;
    }
  });
  if (start >= 0) {
    fn(text.substr(size_t(start)));
    return;
  }
  if (last < text.size()) keep(text.substr(last));
}
std::string BlankUpperRuns(std::string_view text) {
  std::string out;
  EachWordRun(text, [&](std::string_view gap) { out.append(gap); }, [&](std::string_view run) {
    if (ud::RuneCount(run) >= 2 && ud::IsUpperString(run)) out.push_back(' ');
    else out.append(run);
  });
  return out;
}
std::vector<std::string> UpperLetterRuns(std::string_view text) {
  std::vector<std::string> runs;
  EachWordRun(text, [](std::string_view) {}, [&](std::string_view run) {
    if (ud::RuneCount(run) >= 2 && ud::IsUpperString(run)) runs.emplace_back(run);
  });
  return runs;
}
bool EnglishRescued(const std::vector<std::string>& ws, double rate) {
  if (rate >= k_english_rescue_diacritic_rate) return false;
  Set uniq(ws.begin(), ws.end());
  int eo = 0, wd = 0;
  for (const auto& w : uniq) {
    eo += Has(T().en_only, w);
    wd += HasDiacritic(w);
  }
  return eo >= 2 && wd <= 1;
}
bool HasOwnEvidence(const Set& ws, const std::string& language) {
  const Set* stop = T().Stop(language);
  if (!stop) return false;
  for (const auto& w : ws) {
    if (Has(*stop, w) && !Has(T().shared, w)) return true;
  }
  return false;
}
bool Contains(const std::vector<std::string>& ws, const char* t) { return std::find(ws.begin(), ws.end(), t) != ws.end(); }
std::string Head(std::string_view s, long count) {
  if (count <= 0) return "";
  size_t i = 0;
  for (; i < s.size() && count > 0; count--) i += ud::DecodeRune(s.substr(i)).width;
  return std::string(s.substr(0, i));
}
std::vector<std::string> SplitNewline(const std::string& s) {
  std::vector<std::string> out;
  size_t p = 0;
  for (;;) {
    size_t q = s.find('\n', p);
    if (q == std::string::npos) {
      out.push_back(s.substr(p));
      return out;
    }
    out.push_back(s.substr(p, q - p));
    p = q + 1;
  }
}

// ---- scripts -------------------------------------------------------------------------------
bool IsLatinRange(char32_t r, char32_t below) {
  return r < below || (r >= 0x1E00 && r <= 0x1EFF) || (r >= 0xFF21 && r <= 0xFF3A) || (r >= 0xFF41 && r <= 0xFF5A);
}
int LookupScript(char32_t r) {
  if (r > k_script_max_code_point) return -1;
  size_t lo = 0, hi = sizeof(k_script_lookup_index);
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (r < k_script_lookup[mid * 2]) hi = mid;
    else if (r > k_script_lookup[mid * 2 + 1]) lo = mid + 1;
    else return k_script_lookup_index[mid];
  }
  return -1;
}
int ScriptIndexOf(char32_t r) { return IsLatinRange(r, kLatinBelowScriptOf) ? -1 : LookupScript(r); }

using Counts = std::vector<std::pair<std::string, int>>;
Counts ScriptCounts(std::string_view text) {  // Latin comes last, so a named script wins a tie
  int tally[kScripts + 1] = {};
  size_t order[kScripts + 1];
  size_t distinct = 0;
  int latin = 0;
  Each(text, [&](size_t, char32_t r, size_t) {
    if (!ud::IsAlpha(r)) return;
    if (IsLatinRange(r, kLatinBelowCounts)) {
      latin++;
      return;
    }
    int slot = LookupScript(r);
    size_t sl = slot < 0 ? kScripts : size_t(slot);
    if (tally[sl] == 0) order[distinct++] = sl;
    tally[sl]++;
  });
  Counts counts;
  for (size_t k = 0; k < distinct; k++) counts.emplace_back(order[k] == kScripts ? "other" : k_script_names[order[k]], tally[order[k]]);
  counts.emplace_back("latin", latin);
  return counts;
}
std::string ScriptFromCounts(const Counts& c) {
  std::string best;
  int bn = 0;
  for (const auto& x : c) {
    if (x.second > bn) {
      best = x.first;
      bn = x.second;
    }
  }
  return bn == 0 ? "unknown" : best;
}
Profile ProfileFromCounts(const Counts& c) {
  int total = 0;
  for (const auto& x : c) total += x.second;
  Profile p;
  if (total == 0) return p;
  for (const auto& x : c) {
    if (x.first == "latin" && x.second != 0) p.emplace_back(x.first, double(x.second) / double(total));
  }
  for (const auto& x : c) {
    if (x.first != "latin" && x.second != 0) p.emplace_back(x.first, double(x.second) / double(total));
  }
  return p;
}
double Get(const Profile& p, const char* s) {
  for (const auto& x : p) {
    if (x.first == s) return x.second;
  }
  return 0;
}
bool NonLatinCarries(double nl, int letters) {
  return nl >= k_non_latin_fraction || (nl >= k_non_latin_min_fraction && letters >= k_non_latin_min_letters);
}
size_t NonLatinWordCount(std::string_view text) {
  size_t n = 0;
  std::vector<char32_t> cur;
  int script = -1;
  auto emit = [&]() {
    if (cur.size() >= 2 && !ud::IsUpper(cur[0])) n++;
    cur.clear();
  };
  Each(text, [&](size_t, char32_t r, size_t) {
    if (ud::IsCombining(r)) return;
    int here = ScriptIndexOf(r);
    if (here >= 0 && here == script) {
      cur.push_back(r);
      return;
    }
    emit();
    if (here >= 0) cur.push_back(r);
    script = here;
  });
  emit();
  return n;
}

// ---- language ------------------------------------------------------------------------------
bool Shouted(std::string_view text) {
  bool upper = false, lower = false;
  Each(text, [&](size_t, char32_t r, size_t) {
    if (lower) return;
    if (ud::IsLower(r)) lower = true;
    else if (ud::IsUpper(r)) upper = true;
  });
  return !lower && upper;
}
bool ShoutedEvidence(const std::vector<std::string>& tokens, const std::string& language, double rate) {
  bool has_word = false;
  for (const auto& t : tokens) has_word = has_word || ud::RuneCount(t) >= size_t(kShoutedMinWord);
  if (!has_word) return false;
  if (rate >= k_non_en_diacritic_rate) return true;
  const Set* stop = T().Stop(language);
  if (!stop) return false;
  for (const auto& t : tokens) {
    std::string low = ud::Lower(t);
    if (ud::RuneCount(low) >= size_t(kShoutedMinStopword) && Has(*stop, low)) return true;
  }
  return false;
}
std::string NamedProseLanguage(const std::string& segment) {
  if (ud::IsBlank(segment) || HasCodeLine(segment)) return "";
  std::string prose;
  bool first = true;
  for (const auto& tok : ud::Fields(segment)) {
    if (HasJoined(tok)) continue;
    if (!first) prose.push_back(' ');
    prose += tok;
    first = false;
  }
  bool all_caps = Shouted(prose);
  if (!all_caps) prose = BlankUpperRuns(prose);
  auto tokens = Words(prose);
  if (tokens.size() < 4) return "";
  LatinProfile p = LatinProfileOf(prose);
  if (p.language.empty() || p.language == "en") return "";
  if (all_caps && !ShoutedEvidence(tokens, p.language, p.diacritic_rate)) return "";
  const Set* stop = T().Stop(p.language);
  if (!stop) return "";
  Set distinct;
  for (const auto& t : tokens) {
    std::string low = ud::Lower(t);
    if (Has(*stop, low)) distinct.insert(low);
  }
  return distinct.size() < 2 ? "" : p.language;
}

std::string DominantNonLatin(const Profile& p) {
  std::string best;
  double share = 0;
  for (const auto& s : p) {
    if (s.first == "latin") continue;
    if (best.empty() || s.second > share) {
      best = s.first;
      share = s.second;
    }
  }
  return best;
}

Analysis AnalyzeText(const std::string& text) {
  Counts counts = ScriptCounts(text);
  Profile profile = ProfileFromCounts(counts);
  std::string script = ScriptFromCounts(counts);
  double nl = 0;
  if (!profile.empty()) nl = pyjson::Round4(1.0 - Get(profile, "latin"));
  int nl_letters = int(std::nearbyint(nl * double(CountAlpha(text))));
  if (script == "latin" && NonLatinWordCount(text) > 0 && NonLatinCarries(nl, nl_letters)) script = DominantNonLatin(profile);
  Analysis a;
  a.script_profile = profile;
  if (script == "unknown") {
    a.script = "unknown";
    a.english = true;
    return a;
  }
  a.script = script;
  if (script != "latin") {
    a.non_latin_fraction = nl;
    return a;
  }
  LatinProfile latin = LatinProfileOf(text);
  std::string language = latin.language;
  if (!language.empty() && language != "en" && !NonLatinCarries(nl, nl_letters)) {
    if (Shouted(text)) {
      if (!ShoutedEvidence(Words(text), language, latin.diacritic_rate)) language = "";
    } else {
      auto caps = UpperLetterRuns(text);
      bool all_acronyms = !caps.empty();
      for (const auto& run : caps) {
        if (ud::RuneCount(run) >= size_t(kShoutedMinWord)) {
          all_acronyms = false;
          break;
        }
      }
      if (all_acronyms) {
        std::string blanked = BlankUpperRuns(text);
        if (blanked != text) language = LatinProfileOf(blanked).language;
      }
    }
  }
  a.language = language;
  a.english = language == "en" || (language.empty() && !latin.looks_non_english);
  a.diacritic_rate = pyjson::Round4(latin.diacritic_rate);
  a.non_latin_fraction = nl;
  return a;
}

bool LeafNonEnglish(const std::string& leaf, Analysis& best_out) {
  int best_letters = -1;
  Analysis best;
  for (const auto& line : SplitNewline(leaf)) {
    if (ud::RuneCount(line) < 7) continue;
    std::string sample = Head(line, kMaxChars);
    if (ud::IsBlank(sample) || HasCodeLine(sample)) continue;
    Analysis d = AnalyzeText(sample);
    if (d.english) continue;
    bool named = !d.language.empty() && d.language != "en" && !NamedProseLanguage(sample).empty();
    if (!named) {
      if (d.script != "latin" && d.script != "unknown") {
        if (NonLatinWordCount(sample) == 0 || CountAlpha(sample) < k_non_latin_min_letters) continue;
      } else if (!(d.diacritic_rate >= k_non_en_diacritic_rate && Words(sample).size() >= 4)) {
        continue;
      }
    }
    int letters = CountAlpha(sample);
    if (letters > best_letters) {
      best_letters = letters;
      best = d;
    }
  }
  best_out = best;
  return best_letters >= 0;
}

// ---- state ---------------------------------------------------------------------------------
struct Collector {
  std::vector<std::string> out;
  bool limited, done;
  long budget;
  Collector(long limit, bool lim) : limited(lim), done(lim && limit <= 0), budget(limit) {}
  void Leaf(const std::string& s) {
    if (done) return;
    if (!limited) {
      out.push_back(s);
      return;
    }
    long n = long(ud::RuneCount(s));
    if (n > budget) {
      out.push_back(Head(s, budget));
      done = true;
      return;
    }
    out.push_back(s);
    budget -= n + 1;
    if (budget <= 0) done = true;
  }
};
void Collect(const pyvalue::Value& v, int depth, Collector& c) {
  if (depth > kMaxDepth || c.done) return;
  switch (v.kind()) {
    case pyvalue::Value::Kind::String: c.Leaf(ud::ToValid(v.as_string())); break;
    case pyvalue::Value::Kind::Array:
    case pyvalue::Value::Kind::Object:
      for (size_t i = 0; i < v.size() && !c.done; i++) Collect(v.value(i), depth + 1, c);
      break;
    default: break;
  }
}
std::string JoinLeaves(const std::vector<std::string>& leaves, long max_chars) {
  Collector c(max_chars, true);
  for (const auto& l : leaves) c.Leaf(l);
  std::string joined;
  for (size_t i = 0; i < c.out.size(); i++) {
    if (i) joined.push_back(' ');
    joined += c.out[i];
  }
  return Head(joined, max_chars);
}
}  // namespace

std::string ScriptOf(char32_t r) {
  int i = ScriptIndexOf(r);
  return i < 0 ? "" : k_script_names[i];
}

std::string DetectScript(std::string_view text) { return ScriptFromCounts(ScriptCounts(ud::ToValid(text))); }
Profile ScriptProfile(std::string_view text) { return ProfileFromCounts(ScriptCounts(ud::ToValid(text))); }

LatinProfile LatinProfileOf(std::string_view raw) {
  const std::string text = ud::ToValid(raw);
  std::string sub = SubstituteIdentifiers(text);
  for (size_t p = sub.find("\xC4\xB0"); p != std::string::npos; p = sub.find("\xC4\xB0", p + 1)) sub.replace(p, 2, "i");
  std::vector<std::string> ws = Words(ud::Lower(sub));
  std::string lowered = ud::Lower(text);
  int diacritics = 0;
  Each(lowered, [&](size_t, char32_t r, size_t) { diacritics += IsDiacritic(r); });
  size_t n = std::max<size_t>(ud::RuneCount(lowered), 1);
  double rate = double(diacritics) / double(n);
  bool non_english = rate >= k_non_en_diacritic_rate;
  Set uniq(ws.begin(), ws.end());
  bool nordic = Intersects(uniq, T().nordic) && !Intersects(uniq, T().en_only);
  LatinProfile out;
  out.diacritic_rate = rate;
  if (ws.size() > 1 && ws.size() < 4 && Intersects(uniq, T().short_swedish)) {
    out.language = "sv";
    out.looks_non_english = non_english;
    return out;
  }
  if (ws.size() < 4) {
    out.looks_non_english = non_english || nordic;
    return out;
  }
  std::unordered_map<std::string, int> counts;
  for (const auto& w : ws) counts[w]++;
  const auto& stops = T().stops;
  std::vector<int> scores(stops.size());
  int english = 0;
  for (size_t i = 0; i < stops.size(); i++) {
    int score = 0;
    for (const auto& kv : counts) {
      if (Has(stops[i].second, kv.first)) score += Has(T().en_collision, kv.first) ? 1 : kv.second;
    }
    scores[i] = score;
    if (stops[i].first == "en") english = score;
  }
  std::string best_language;
  int best = 0;
  bool seen = false;
  for (size_t i = 0; i < stops.size(); i++) {
    const std::string& code = stops[i].first;
    if (code == "en" || !HasOwnEvidence(uniq, code)) continue;
    if (!seen || scores[i] > best) {  // strictly greater: the first listed language wins a tie
      best_language = code;
      best = scores[i];
      seen = true;
    }
  }
  std::string language;
  if (seen && best >= std::max(2, english + 2)) {
    language = best_language;
  } else if (best_language == "sv" && Contains(ws, "inte") && Contains(ws, "kan") && Has(T().openers, ws[0]) && english <= 1) {
    language = best_language;
  } else if (seen && non_english && best >= std::max(2, english)) {
    language = best_language;
  } else if (english != 0 && (!non_english || EnglishRescued(ws, rate))) {
    language = "en";
  }
  out.language = language;
  out.english_hits = english;
  out.looks_non_english = non_english || (language.empty() && nordic);
  return out;
}

std::string StateText(const pyvalue::Value& state) {
  Collector c(kMaxChars, true);
  Collect(state, 0, c);
  return JoinLeaves(c.out, kMaxChars);
}

Analysis Analyze(const pyvalue::Value& state) {
  Collector all(0, false);
  Collect(state, 0, all);
  const std::vector<std::string>& leaves = all.out;
  Analysis result = AnalyzeText(JoinLeaves(leaves, kMaxChars));
  if (result.script == "latin" && result.english) {
    bool multiline = false;
    for (const auto& l : leaves) multiline = multiline || l.find('\n') != std::string::npos;
    if (leaves.size() > 1 || multiline) {
      long seen = 0;
      bool found = false;
      for (const auto& leaf : leaves) {
        for (const auto& line : SplitNewline(leaf)) {
          if (seen >= kMaxChars) goto done_segment;
          std::string seg = Head(line, kMaxChars - seen);
          seen += long(ud::RuneCount(seg));
          std::string lg = NamedProseLanguage(seg);
          if (!lg.empty()) {
            result.language = lg;
            result.english = false;
            result.mixed_segment = ud::Strip(seg);
            found = true;
            goto done_segment;
          }
        }
      }
    done_segment:
      (void)found;
    }
  }
  const bool scalar = state.kind() == pyvalue::Value::Kind::String || state.is_null();
  if (scalar || !result.english) return result;
  long best_letters = -1;
  Analysis best;
  for (const auto& leaf : leaves) {
    Analysis d;
    if (!LeafNonEnglish(leaf, d)) continue;
    long letters = CountAlpha(Head(leaf, kMaxChars));
    if (letters > best_letters) {
      best_letters = letters;
      best = d;
    }
  }
  if (best_letters < 0) return result;
  result.language = best.language;
  result.english = false;
  return result;
}

Analysis Analyze(std::string_view text) { return Analyze(pyvalue::Value::String(std::string(text))); }
std::string DetectLanguage(const pyvalue::Value& s) { return Analyze(s).language; }
bool IsEnglish(const pyvalue::Value& s) { return Analyze(s).english; }

}  // namespace lang
}  // namespace laya
