// A tiny line protocol over laya::schema and laya::sequence for the differential fuzz
// (laya-e2e/cpp-lane/fuzz-c3): argv[1] is the checkpoints directory; bytes travel as hex ("-" is empty).
//   q <hex def>                                       -> "err <hex>" | "ok <hex json {type, ins, opts, labels}>"
//   seq <name> <max> <head> <left 0|1> <order|-> <hex def> <hex state json>
//        -> "err <hex>" | "mk <kept> <total>" | "ok ids|markers|opts|distinct|tpo|st|su|sd|trunc"
//   plan <name> <max> <head> <window|-> <stride|-> <n,n,...|-> -> "err <hex>" | "ok w s r <hex warning>..."
//   cap <n> <window> <config> <batch|-> -> "none" | cap
#include <iostream>
#include <map>
#include <sstream>

#include "laya/pyjson.hpp"
#include "laya/pyvalue.hpp"
#include "laya/schema.hpp"
#include "laya/sequence.hpp"

namespace {
using laya::pyvalue::Value;
namespace sc = laya::schema;
namespace sq = laya::sequence;

std::string FromHex(const std::string& h) {
  std::string out;
  if (h == "-") return out;
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
  return o.empty() ? "-" : o;
}
template <class T>
std::string Join(const std::vector<T>& v) {
  std::string o;
  for (size_t i = 0; i < v.size(); i++) o += (i ? "," : "") + std::to_string(v[i]);
  return o.empty() ? "-" : o;
}
std::vector<int> Ints(const std::string& s) {
  std::vector<int> out;
  if (s == "-") return out;
  std::stringstream ss(s);
  for (std::string part; std::getline(ss, part, ',');) out.push_back(std::stoi(part));
  return out;
}
int64_t Opt(const std::string& s) { return s == "-" ? 0 : std::stoll(s); }

laya::Result<sc::Question> Make(const std::string& def_json, const std::string& id = "q") {
  auto v = laya::pyvalue::Parse(def_json);
  if (!v.ok()) return laya::Result<sc::Question>::Fail("parse: " + v.error());
  auto q = sc::Question::FromWire(v.value());
  if (!q.ok()) return laya::Result<sc::Question>::Fail("question " + laya::pyjson::Repr(id) + ": " + q.error());
  return q;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  std::map<std::string, laya::tokenizer::Tokenizer> toks;
  for (const char* n : {"english", "multilingual"}) {
    auto t = laya::tokenizer::Tokenizer::LoadDir(std::string(argv[1]) + "/" + n);
    if (!t.ok()) { std::cerr << t.error() << "\n"; return 2; }
    toks.emplace(n, std::move(t).value());
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    std::stringstream ss(line);
    std::string kind;
    ss >> kind;
    if (kind == "q") {
      std::string h;
      ss >> h;
      auto q = Make(FromHex(h));
      if (!q.ok()) { std::cout << "err " << Hex(q.error()) << "\n"; continue; }
      Value o = Value::Object();
      o.Set("type", Value::String(std::string(sc::Name(q.value().type()))));
      o.Set("ins", Value::String(q.value().instructions()));
      Value opts = Value::Array(), labels = Value::Array();
      for (const auto& s : q.value().RenderedOptions()) opts.Push(Value::String(s));
      for (const auto& s : q.value().Labels()) labels.Push(Value::String(s));
      o.Set("opts", opts);
      o.Set("labels", labels);
      std::cout << "ok " << Hex(laya::pyvalue::Dumps(o).value()) << "\n";
    } else if (kind == "seq") {
      std::string name, order, def, state;
      int64_t mx, hd;
      int left;
      ss >> name >> mx >> hd >> left >> order >> def >> state;
      auto q = Make(FromHex(def));
      if (!q.ok()) { std::cout << "err " << Hex(q.error()) << "\n"; continue; }
      sc::Question question = q.value();
      if (order != "-") {
        auto w = question.WithOptionOrder(Ints(order));
        if (!w.ok()) { std::cout << "err " << Hex(w.error()) << "\n"; continue; }
        question = w.value();
      }
      auto sv = laya::pyvalue::Parse(FromHex(state));
      if (!sv.ok()) { std::cout << "err " << Hex("state: " + sv.error()) << "\n"; continue; }
      const auto& tok = toks.at(name);
      auto st = sq::EncodeState(tok, sv.value());
      if (!st.ok()) { std::cout << "err " << Hex(st.error()) << "\n"; continue; }
      if (left) st.value().keep = sq::Keep::End;
      sq::MarkersError me;
      auto s = sq::Build(tok, question, st.value(), {mx, hd}, &me);
      if (!s.ok()) {
        if (me.total != 0) std::cout << "mk " << me.kept << " " << me.total << "\n";
        else std::cout << "err " << Hex(s.error()) << "\n";
        continue;
      }
      const auto& r = s.value();
      std::cout << "ok " << Join(r.ids) << "|" << Join(r.markers) << "|" << r.stats.options << "|" << r.stats.options_distinct
                << "|" << (r.stats.tokens_per_option ? std::to_string(*r.stats.tokens_per_option) : "N") << "|"
                << r.truncation.state_tokens << "|" << r.truncation.state_tokens_used << "|" << r.truncation.state_tokens_dropped
                << "|" << (r.truncation.truncated ? 1 : 0) << "\n";
    } else if (kind == "plan") {
      std::string name, w, st, counts;
      int64_t mx, hd;
      ss >> name >> mx >> hd >> w >> st >> counts;
      std::vector<sc::Question> qs;
      for (int n : Ints(counts)) {
        std::vector<sc::Option> opts;
        for (int i = 0; i < n; i++) {
          std::string t;
          for (int k = 0; k < 4; k++) t += (k ? " " : "") + std::string("criterion") + std::to_string(i) + " word" + std::to_string(k);
          char l[16];
          std::snprintf(l, sizeof l, "opt%03d", i);
          opts.push_back({sc::Label::Text(l), Value::String(t)});
        }
        qs.push_back(sc::Question::Choice("Pick one.", opts).value());
      }
      auto p = sq::PlanWindows(toks.at(name), qs, {mx, hd}, {Opt(w), Opt(st)});
      if (!p.ok()) { std::cout << "err " << Hex(p.error()) << "\n"; continue; }
      std::cout << "ok " << p.value().window << " " << p.value().stride << " " << p.value().room;
      for (const auto& x : p.value().warnings) std::cout << " " << Hex(x);
      std::cout << "\n";
    } else if (kind == "cap") {
      std::string n, w, c, b;
      ss >> n >> w >> c >> b;
      auto r = sq::BatchCap(std::stoll(n), std::stoll(w), std::stoll(c), Opt(b));
      if (r.ok) std::cout << r.cap << "\n"; else std::cout << "none\n";
    } else {
      std::cout << "bad\n";
    }
  }
  return 0;
}
