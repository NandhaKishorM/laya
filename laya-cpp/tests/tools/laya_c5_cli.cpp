// A line protocol over laya::agent::OpenOnnx for the differential fuzz (laya-e2e/cpp-lane/fuzz-c5).
// argv: graph checkpoint_dir. Bytes travel as hex.
//   pred <hex json state> <hex json questions> -> "err <hex>" | "ok <hex compact prediction json>"
#include <iostream>
#include <sstream>

#include "laya/onnx.hpp"
#include "laya/pyvalue.hpp"
#include "laya/schema.hpp"

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
std::string Err(const std::string& e) { return "err " + Hex(e); }

std::string Pred(const laya::agent::Agent& a, const std::string& state, const std::string& questions) {
  auto sv = laya::pyvalue::Parse(FromHex(state));
  if (!sv.ok()) return Err(sv.error());
  auto qs = laya::schema::Questions::Parse(FromHex(questions));
  if (!qs.ok()) return Err(qs.error());
  auto p = a.Predict(sv.value(), qs.value());
  if (!p.ok()) return Err(p.error());
  auto js = p.value().ToJson();
  if (!js.ok()) return Err(js.error());
  return "ok " + Hex(js.value());
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return 64;
  auto agent = laya::agent::OpenOnnx(argv[2], argv[1]);
  if (!agent.ok()) {
    std::cerr << agent.error() << std::endl;
    return 2;
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    std::stringstream ss(line);
    std::string cmd, state, questions;
    ss >> cmd >> state >> questions;
    std::cout << (cmd == "pred" ? Pred(*agent.value(), state, questions) : Err("unknown command")) << std::endl;
  }
  return 0;
}
