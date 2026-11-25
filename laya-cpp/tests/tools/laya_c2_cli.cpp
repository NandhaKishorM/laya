// A tiny line protocol over laya::tokenizer for the differential fuzz (laya-e2e/cpp-lane/fuzz-c2):
// argv[1] is the checkpoints directory; each stdin line is "<kind> <args...>" (bytes are hex), each
// stdout line the answer.
//   enc <name> <hex>            -> ids, space separated
//   dec <name> <skip 0|1> <ids> -> hex of the text (ids comma separated, "-" for none)
//   norm <NFC|NFD|NFKC|NFKD> <hex> -> hex       lower <hex> -> hex
//   re <hex pattern> <hex content> <hex input> -> "refused" | "unsupported" | "ok <hex>"
#include <iostream>
#include <map>
#include <sstream>

#include "laya/tokenizer.hpp"

namespace {
std::string FromHex(const std::string& h) {
  std::string out;
  if (h == "-") return out;  // "-" stands for the empty string, so fields never vanish from the line
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
// JSON string literal of raw bytes holding valid UTF-8 (the driver sends valid text only).
std::string Quote(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') o += std::string("\\") + char(c);
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += char(c);
  }
  return o + "\"";
}
}  // namespace

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  if (argc < 2) return 2;
  std::map<std::string, laya::tokenizer::Tokenizer> toks;
  for (const char* name : {"english", "multilingual"}) {
    auto t = laya::tokenizer::Tokenizer::LoadDir(std::string(argv[1]) + "/" + name);
    if (!t.ok()) { std::cerr << t.error() << "\n"; return 2; }
    toks.emplace(name, std::move(t).value());
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string kind;
    in >> kind;
    if (kind == "enc") {
      std::string name, h;
      in >> name >> h;
      std::string out;
      for (int32_t id : toks[name].Encode(FromHex(h))) out += std::to_string(id) + " ";
      std::cout << out << "\n";
    } else if (kind == "dec") {
      std::string name, ids;
      int skip;
      in >> name >> skip >> ids;
      std::vector<int32_t> v;
      std::istringstream is(ids == "-" ? "" : ids);
      std::string tok;
      while (std::getline(is, tok, ',')) v.push_back(int32_t(std::stol(tok)));
      std::cout << Hex(toks[name].Decode(v, skip != 0)) << "\n";
    } else if (kind == "norm") {
      std::string form, h;
      in >> form >> h;
      auto f = form == "NFC" ? laya::tokenizer::Form::NFC : form == "NFD" ? laya::tokenizer::Form::NFD
               : form == "NFKC" ? laya::tokenizer::Form::NFKC : laya::tokenizer::Form::NFKD;
      std::cout << Hex(laya::tokenizer::Normalize(FromHex(h), f)) << "\n";
    } else if (kind == "lower") {
      std::string h;
      in >> h;
      std::cout << Hex(laya::tokenizer::Lowercase(FromHex(h))) << "\n";
    } else if (kind == "re") {
      std::string p, c, x;
      in >> p >> c >> x;
      std::string spec = "{\"type\":\"Replace\",\"pattern\":{\"Regex\":" + Quote(FromHex(p)) + "},\"content\":" + Quote(FromHex(c)) + "}";
      auto r = laya::tokenizer::RunNormalizerSpec(spec, FromHex(x));
      if (r.ok()) std::cout << "ok " << Hex(r.value()) << "\n";
      else std::cout << (laya::tokenizer::IsUnsupported(r.error()) ? "unsupported" : "refused") << "\n";
    } else {
      std::cout << "bad\n";
    }
  }
  return 0;
}
