#pragma once
// Script and language detection, a port of laya/lang.py (via laya-go/lang). A state is a
// pyvalue::Value: a string, or lists and objects of them nested up to kMaxDepth levels; numbers,
// booleans and null contribute no text. Objects are read in insertion order, as a Python dict is.
// Every function is safe for concurrent use: the tables are immutable after first use.
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "laya/pyvalue.hpp"

namespace laya {
namespace lang {

constexpr int kMaxChars = 4000;  // characters of a state that are read
constexpr int kMaxDepth = 6;     // container nesting that is read

// Script names are lang.py's strings: "greek", ..., "latin", "other", "unknown".
using Profile = std::vector<std::pair<std::string, double>>;  // Latin first, then first appearance

struct LatinProfile {
  std::string language;  // "" when undecided
  int english_hits = 0;
  double diacritic_rate = 0;
  bool looks_non_english = false;
};

struct Analysis {
  std::string script;
  Profile script_profile;
  std::string language;  // "" when none could be named
  bool english = false;
  double diacritic_rate = 0;
  double non_latin_fraction = 0;
  std::string mixed_segment;
};

// The named non-Latin script of one letter, or "" when none claims it.
std::string ScriptOf(char32_t r);
std::string DetectScript(std::string_view text);
Profile ScriptProfile(std::string_view text);
LatinProfile LatinProfileOf(std::string_view text);
std::string StateText(const pyvalue::Value& state);  // leaves joined by a space, at most kMaxChars
Analysis Analyze(const pyvalue::Value& state);
Analysis Analyze(std::string_view text);             // a state that is one string
std::string DetectLanguage(const pyvalue::Value& state);
bool IsEnglish(const pyvalue::Value& state);

}  // namespace lang
}  // namespace laya
