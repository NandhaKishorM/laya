// A tiny line protocol over the library for the differential fuzz (laya-e2e/cpp-lane/fuzz-c1):
// each stdin line is "<kind> <hex bytes>", each stdout line the hex of the answer.
//   json  -> "ok <hex of json.dumps(json.loads(x))>" or "err"
//   lower -> hex of str.lower()          sha -> hex digest
//   lang  -> "<script>|<language>|<english>|<diacritic>|<non_latin>|<hex mixed_segment>"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include "laya/lang.hpp"
#include "laya/pyjson.hpp"
#include "laya/pyvalue.hpp"
#include "laya/sha256.hpp"
#include "laya/unicodedata.hpp"

namespace {
std::string FromHex(const std::string& h) {
  std::string out;
  for (size_t i = 0; i + 1 < h.size(); i += 2) out.push_back(char(std::stoi(h.substr(i, 2), nullptr, 16)));
  return out;
}
std::string Hex(const std::string& s) {
  static const char* d = "0123456789abcdef";
  std::string o;
  for (unsigned char c : s) {
    o.push_back(d[c >> 4]);
    o.push_back(d[c & 15]);
  }
  return o;
}
}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::string line;
  while (std::getline(std::cin, line)) {
    size_t sp = line.find(' ');
    std::string kind = line.substr(0, sp), data = FromHex(sp == std::string::npos ? "" : line.substr(sp + 1));
    if (kind == "json") {
      auto v = laya::pyvalue::Parse(data);
      auto d = v.ok() ? laya::pyvalue::Dumps(v.value()) : laya::Result<std::string>::Fail("");
      std::cout << (d.ok() ? "ok " + Hex(d.value()) : std::string("err")) << "\n";
    } else if (kind == "lower") {
      std::cout << Hex(laya::unicodedata::Lower(data)) << "\n";
    } else if (kind == "sha") {
      std::cout << laya::Sha256Hex(data) << "\n";
    } else if (kind == "lang") {
      laya::lang::Analysis a = laya::lang::Analyze(std::string_view(data));
      std::cout << a.script << "|" << a.language << "|" << (a.english ? 1 : 0) << "|" << laya::pyjson::ReprFloat(a.diacritic_rate)
                << "|" << laya::pyjson::ReprFloat(a.non_latin_fraction) << "|" << Hex(a.mixed_segment) << "\n";
    } else if (kind == "strip") {
      std::cout << Hex(laya::unicodedata::Strip(data)) << "\n";
    } else if (kind == "stripn") {
      std::cout << Hex(laya::unicodedata::StripNumeric(data)) << "\n";
    } else if (kind == "fields") {
      for (const auto& f : laya::unicodedata::Fields(data)) std::cout << Hex(f) << ",";
      std::cout << "\n";
    } else if (kind == "dec") {
      std::cout << Hex(laya::unicodedata::DecimalASCII(data)) << "\n";
    } else if (kind == "isup") {
      std::cout << (laya::unicodedata::IsUpperString(data) ? "1" : "0") << "\n";
    } else if (kind == "repr") {
      std::cout << Hex(laya::pyjson::Repr(data)) << "\n";
    } else if (kind == "float") {  // the payload here is decimal bits, not hex
      uint64_t bits = std::strtoull(line.substr(sp + 1).c_str(), nullptr, 10);
      double v;
      std::memcpy(&v, &bits, 8);
      std::cout << laya::pyjson::ReprFloat(v) << "|" << laya::pyjson::ReprFloat(laya::pyjson::Round4(v)) << "|"
                << laya::pyjson::Percent0(v) << "\n";
    } else {
      std::cout << "?\n";
    }
  }
  return 0;
}
