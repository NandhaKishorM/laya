// A line protocol over laya::decode for the differential fuzz (laya-e2e/cpp-lane/fuzz-c4); bytes travel as hex.
//   dec <hex question wire def> <order|-> <hex json {"logits","act","temperature","tbo"}>
//        -> "err <hex>" | "ok <hex compact answer json>"
#include <iostream>
#include <sstream>

#include "laya/decode.hpp"
#include "laya/pyvalue.hpp"
#include "laya/schema.hpp"

namespace {
using laya::pyvalue::Value;
namespace dc = laya::decode;
namespace sc = laya::schema;

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

double Num(const Value& v) { return v.kind() == Value::Kind::Int ? double(v.as_int()) : v.as_float(); }

std::string Dec(const std::string& def, const std::string& order, const std::string& arg) {
  auto wire = laya::pyvalue::Parse(FromHex(def));
  if (!wire.ok()) return Err(wire.error());
  auto q = sc::Question::FromWire(wire.value());
  if (!q.ok()) return Err(q.error());
  sc::Question question = q.value();
  if (order != "-") {
    std::vector<int> o;
    std::stringstream ss(order);
    for (std::string part; std::getline(ss, part, ',');) o.push_back(std::stoi(part));
    auto w = question.WithOptionOrder(o);
    if (!w.ok()) return Err(w.error());
    question = w.value();
  }
  auto a = laya::pyvalue::Parse(FromHex(arg));
  if (!a.ok()) return Err(a.error());
  auto temps = dc::Temperatures::Of(a.value().Find("temperature"), a.value().Find("tbo"));
  if (!temps.ok()) return Err(temps.error());
  dc::Row row;
  for (const Value& v : a.value().Find("logits")->items()) row.logits.push_back(float(Num(v)));
  row.k = int64_t(row.logits.size());
  row.act = float(Num(*a.value().Find("act")));
  auto ans = dc::Decode(question, row, "", &temps.value());
  if (!ans.ok()) return Err(ans.error());
  auto js = ans.value().ToJson();
  if (!js.ok()) return Err(js.error());
  auto back = laya::pyvalue::Parse(js.value());
  return "ok " + Hex(laya::pyvalue::Dumps(back.value()).value());
}
}  // namespace

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::stringstream ss(line);
    std::string cmd, def, order, arg;
    ss >> cmd >> def >> order >> arg;
    std::cout << (cmd == "dec" ? Dec(def, order, arg) : Err("unknown command")) << std::endl;
  }
  return 0;
}
