#pragma once
// A schema::Question from the golden's internal form {"t", "ins", "crit", "labels"}, with an option order.
#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

#include "laya/schema.hpp"

namespace laya_test {

using laya::pyvalue::Value;
namespace sc = laya::schema;
using Kind = Value::Kind;

inline sc::Question GoldenQuestion(const Value& g, const Value& order) {
  const std::string kind = g.Find("t")->as_string();
  const std::string ins = g.Find("ins")->as_string();
  const Value* crit = g.Find("crit");
  laya::Result<sc::Question> q = laya::Result<sc::Question>::Fail("unset");
  if (kind == "choice") {
    std::vector<sc::Option> opts;
    for (size_t i = 0; i < crit->size(); i++) opts.push_back({sc::Label::Text(crit->key(i)), crit->value(i)});
    q = sc::Question::Choice(ins, opts);
  } else if (kind == "score") {
    q = sc::Question::Score(ins, crit->items());
  } else {
    sc::NoulSpec spec;
    if (crit != nullptr && crit->kind() == Kind::Object) {
      if (const Value* t = crit->Find("true")) spec.true_text = *t;
      if (const Value* f = crit->Find("false")) spec.false_text = *f;
    }
    if (const Value* l = g.Find("labels"); l != nullptr && !l->is_null()) {
      spec.false_label = l->Find("false")->as_string();
      spec.true_label = l->Find("true")->as_string();
    }
    q = sc::Question::Noul(ins, spec);
  }
  EXPECT_TRUE(q.ok()) << (q.ok() ? "" : q.error());
  if (!q.ok()) std::abort();
  if (!order.is_null()) {
    std::vector<int> o;
    for (const Value& v : order.items()) o.push_back(int(v.as_int()));
    q = q.value().WithOptionOrder(o);
    EXPECT_TRUE(q.ok());
    if (!q.ok()) std::abort();
  }
  return q.value();
}

}  // namespace laya_test
