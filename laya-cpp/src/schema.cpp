#include "laya/schema.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "cnum.hpp"
#include "laya/pyjson.hpp"
#include "laya/unicodedata.hpp"

namespace laya {
namespace schema {

namespace ud = unicodedata;
using Kind = Value::Kind;

std::string_view Name(Type t) {
  switch (t) {
    case Type::Choice: return "choice";
    case Type::Score: return "score";
    case Type::Noul: return "noul";
    case Type::Invalid: break;
  }
  return "";
}

int Code(Type t) {
  switch (t) {
    case Type::Choice: return 0;
    case Type::Score: return 1;
    case Type::Noul: return 2;
    case Type::Invalid: break;
  }
  return -1;
}

std::string JsonString(std::string_view s) {
  auto r = pyvalue::Dumps(Value::String(ud::ValidUtf8(s) ? std::string(s) : ud::Lossy(s)));
  return r.ok() ? r.value() : "\"\"";
}

namespace {

// Reasons laya's Agent._check_question gives, word for word.
constexpr const char* kNoInstructions = "no 'instructions'; add the text the model should answer";
constexpr const char* kNilInstructions = "'instructions' must not be None; add the text the model should answer";
constexpr const char* kBlankIns = "'instructions' must not be empty; add the text the model should answer";
constexpr const char* kChoiceShape =
    "a choice question takes 'criteria' as a dict of label -> description, or a list of labels";
constexpr const char* kChoiceEmpty = "a choice question needs at least one criterion";
constexpr const char* kScoreShape = "a score question takes 'criteria' as a list of level descriptions, index 0 first";
constexpr const char* kScoreEmpty = "a score question needs at least one level";
constexpr const char* kNoulShape =
    "a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it";
constexpr const char* kLabelsOnlyNoul = "'labels' is only supported for noul questions";
constexpr const char* kNoulLabels = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";
constexpr const char* kDefaultFalse = "no, the statement does not hold";
constexpr const char* kDefaultTrue = "yes, the statement holds";

std::string N(int64_t i) { return std::to_string(i); }

std::string NoulKeysReason(const std::string& got) {
  return "a noul question takes 'criteria' keyed only 'true'/'false' (either or both, and omitted is fine), got " + got +
         ". Those keys are the option texts the model reads; any other key was silently dropped and replaced with "
         "the defaults. If you want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set "
         "'labels' instead.";
}

std::string OrderReason(size_t n, const std::string& got) {
  return "'option_order' must be a permutation of range(" + N(int64_t(n)) +
         ") -- one slot per option, each option once -- got " + got;
}

std::string LabelRepeatReason(size_t i, const std::string& repr, size_t j) {
  return "choice label " + N(int64_t(i)) + " (" + repr + ") repeats label " + N(int64_t(j)) +
         "; the labels are the answer keys, so every option needs its own (1, 1.0 and True are one key)";
}

std::string ScoreNullReason(size_t i) {
  return "score level " + N(int64_t(i)) + " is null; give every level a description, index 0 first";
}

// An integer-valued double as exact decimal digits (no sign for zero).
std::string ExactInteger(double d) {
  if (d == 0) return "0";
  if (std::fabs(d) < 9.0e18) return N(int64_t(d));
  int exp = 0;
  double m = std::frexp(std::fabs(d), &exp);                  // d = m * 2^exp, 0.5 <= m < 1
  uint64_t mant = uint64_t(std::ldexp(m, 53));                // 53-bit integer
  exp -= 53;                                                  // d = mant * 2^exp, exp >= 0 here
  std::vector<int> digits;                                    // little endian, base 10
  for (uint64_t v = mant; v > 0; v /= 10) digits.push_back(int(v % 10));
  for (int i = 0; i < exp; i++) {
    int carry = 0;
    for (int& x : digits) {
      int v = x * 2 + carry;
      x = v % 10;
      carry = v / 10;
    }
    if (carry) digits.push_back(carry);
  }
  std::string out = d < 0 ? "-" : "";
  for (size_t i = digits.size(); i-- > 0;) out.push_back(char('0' + digits[i]));
  return out;
}

Criterion NewCriterion(const Value& v, std::string* err) {
  if (v.is_null()) return {};
  if (v.kind() == Kind::String) {
    if (v.as_string().empty()) return {};
    return {true, v.as_string(), JsonString(v.as_string())};
  }
  auto text = pyvalue::Dumps(v);
  if (!text.ok()) {
    *err = text.error();
    return {};
  }
  return {true, text.value(), text.value()};
}

std::string CheckInstructions(const std::string& ins) {
  if (ud::IsBlank(ins)) return kBlankIns;
  if (!ud::ValidUtf8(ins)) return "'instructions' is not valid UTF-8; a Python str always is";
  return "";
}

std::string CheckDistinct(const std::vector<Label>& labels) {
  std::map<std::string, size_t> first;
  for (size_t i = 0; i < labels.size(); i++) {
    auto ins = first.emplace(labels[i].EqKey(), i);
    if (!ins.second) return LabelRepeatReason(i, labels[i].Repr(), ins.first->second);
  }
  return "";
}

template <class T>
Result<T> Refuse(const std::string& reason) {
  return Result<T>::Fail(reason);
}

}  // namespace

// ---- Label -------------------------------------------------------------------------------------

Label Label::Text(std::string s) {
  Label l;
  l.text_ = std::move(s);
  return l;
}
Label Label::OfBool(bool b) {
  Label l;
  l.kind_ = Kind::Bool;
  l.text_ = b ? "true" : "false";
  return l;
}
Label Label::OfInt(int64_t i) {
  Label l;
  l.kind_ = Kind::Int;
  l.text_ = std::to_string(i);
  return l;
}
Label Label::OfFloat(double d) {
  Label l;
  l.kind_ = Kind::Float;
  l.text_ = pyjson::ReprFloat(d);
  return l;
}

Result<Label> Label::Of(const Value& v) {
  switch (v.kind()) {
    case pyvalue::Value::Kind::String: return Result<Label>::Ok(Text(v.as_string()));
    case pyvalue::Value::Kind::Bool: return Result<Label>::Ok(OfBool(v.as_bool()));
    case pyvalue::Value::Kind::Int: return Result<Label>::Ok(OfInt(v.as_int()));
    case pyvalue::Value::Kind::BigInt: {
      Label l;
      l.kind_ = Kind::Int;
      l.text_ = v.big_int();
      return Result<Label>::Ok(l);
    }
    case pyvalue::Value::Kind::Float: return Result<Label>::Ok(OfFloat(v.as_float()));
    default: break;
  }
  return Result<Label>::Fail("schema: " + pyvalue::TypeName(v) + " is not a label; use a string, bool or number");
}

std::string Label::String() const {
  if (kind_ == Kind::Bool) return text_ == "true" ? "True" : "False";
  if (kind_ == Kind::Float) {
    if (text_ == "NaN") return "nan";
    if (text_ == "Infinity") return "inf";
    if (text_ == "-Infinity") return "-inf";
  }
  return text_;
}

std::string Label::Repr() const { return kind_ == Kind::Text ? pyjson::Repr(text_) : String(); }

std::string Label::EqKey() const {
  switch (kind_) {
    case Kind::Text: return "s:" + text_;
    case Kind::Bool: return text_ == "true" ? "n:1" : "n:0";
    case Kind::Int: return "n:" + text_;
    case Kind::Float: break;
  }
  if (text_ == "NaN" || text_ == "Infinity" || text_ == "-Infinity") return "n:" + text_;
  double d = cnum::ToDouble(text_);  // text_ is Python-spelled with a dot, whatever the process locale
  if (d == std::trunc(d)) return "n:" + ExactInteger(d);
  return "f:" + text_;  // shortest repr is unique per double
}

std::string Label::Json() const { return kind_ == Kind::Text ? JsonString(text_) : text_; }

// ---- Question ----------------------------------------------------------------------------------

Result<Question> Question::Choice(std::string instructions, const std::vector<Option>& options) {
  if (auto r = CheckInstructions(instructions); !r.empty()) return Refuse<Question>(r);
  if (options.empty()) return Refuse<Question>(kChoiceEmpty);
  std::vector<Label> labels;
  for (size_t i = 0; i < options.size(); i++) {
    const Label& l = options[i].label;
    if (l.kind() == Label::Kind::Text && !ud::ValidUtf8(l.Key())) {
      return Refuse<Question>("choice label " + N(int64_t(i)) + " is not valid UTF-8; a Python str always is");
    }
    labels.push_back(l);
  }
  if (auto r = CheckDistinct(labels); !r.empty()) return Refuse<Question>(r);
  Question q;
  q.typ_ = Type::Choice;
  q.instructions_ = std::move(instructions);
  for (size_t i = 0; i < options.size(); i++) {
    std::string err;
    Criterion c = NewCriterion(options[i].criterion, &err);
    if (!err.empty()) {
      return Refuse<Question>("choice option " + labels[i].Repr() + ": criterion cannot be rendered: " + err);
    }
    q.options_.push_back({labels[i], std::move(c)});
  }
  return Result<Question>::Ok(std::move(q));
}

Result<Question> Question::ChoiceOf(std::string instructions, const std::vector<std::string>& labels) {
  std::vector<Option> options;
  for (const auto& l : labels) options.push_back({Label::Text(l), Value()});
  return Choice(std::move(instructions), options);
}

Result<Question> Question::Score(std::string instructions, const std::vector<Value>& levels) {
  if (auto r = CheckInstructions(instructions); !r.empty()) return Refuse<Question>(r);
  if (levels.empty()) return Refuse<Question>(kScoreEmpty);
  Question q;
  q.typ_ = Type::Score;
  q.instructions_ = std::move(instructions);
  for (size_t i = 0; i < levels.size(); i++) {
    if (levels[i].is_null()) return Refuse<Question>(ScoreNullReason(i));
    std::string err;
    Criterion c = NewCriterion(levels[i], &err);
    if (!err.empty()) return Refuse<Question>("score level " + N(int64_t(i)) + " cannot be rendered: " + err);
    if (!c.set) c = {true, "", "\"\""};  // the empty string is a level: it renders as "level i: "
    q.levels_.push_back(std::move(c));
  }
  return Result<Question>::Ok(std::move(q));
}

Result<Question> Question::Noul(std::string instructions, const NoulSpec& spec) {
  if (auto r = CheckInstructions(instructions); !r.empty()) return Refuse<Question>(r);
  Question q;
  q.typ_ = Type::Noul;
  q.instructions_ = std::move(instructions);
  q.false_label_ = "false";
  q.true_label_ = "true";
  const Value* sides[2] = {&spec.false_text, &spec.true_text};
  const char* names[2] = {"false", "true"};
  for (int i = 0; i < 2; i++) {
    std::string err;
    q.noul_[i] = NewCriterion(*sides[i], &err);
    if (!err.empty()) return Refuse<Question>(std::string("noul ") + names[i] + " description cannot be rendered: " + err);
  }
  if (!spec.true_label.empty() || !spec.false_label.empty()) {
    std::string f = ud::Strip(spec.false_label), t = ud::Strip(spec.true_label);
    if (f.empty() || t.empty() || f == t || !ud::ValidUtf8(f) || !ud::ValidUtf8(t)) return Refuse<Question>(kNoulLabels);
    q.false_label_ = f;
    q.true_label_ = t;
    q.custom_labels_ = true;
  }
  return Result<Question>::Ok(std::move(q));
}

size_t Question::NumOptions() const {
  switch (typ_) {
    case Type::Choice: return options_.size();
    case Type::Score: return levels_.size();
    case Type::Noul: return 2;
    case Type::Invalid: break;
  }
  return 0;
}

Result<Question> Question::WithOptionOrder(const std::vector<int>& order) const {
  if (typ_ == Type::Invalid) {
    return Refuse<Question>("the zero Question has no options to order; build one with Choice, ChoiceOf, Score, Noul");
  }
  Question q = *this;
  size_t n = NumOptions();
  std::vector<bool> seen(n, false);
  bool ok = order.size() == n;
  for (size_t i = 0; ok && i < order.size(); i++) {
    int v = order[i];
    if (v < 0 || size_t(v) >= n || seen[size_t(v)]) ok = false;
    else seen[size_t(v)] = true;
  }
  if (!ok) {
    std::string s = "[";
    for (size_t i = 0; i < order.size(); i++) s += (i ? ", " : "") + N(order[i]);
    return Refuse<Question>(OrderReason(n, s + "]"));
  }
  q.order_ = order;
  return Result<Question>::Ok(std::move(q));
}

std::vector<std::string> Question::Labels() const {
  std::vector<std::string> out;
  switch (typ_) {
    case Type::Choice: for (const auto& o : options_) out.push_back(o.label.String()); break;
    case Type::Score: for (size_t i = 0; i < levels_.size(); i++) out.push_back(N(int64_t(i))); break;
    case Type::Noul: out = {false_label_, true_label_}; break;
    case Type::Invalid: break;
  }
  return out;
}

std::vector<Label> Question::AnswerLabels() const {
  std::vector<Label> out;
  switch (typ_) {
    case Type::Choice: for (const auto& o : options_) out.push_back(o.label); break;
    case Type::Score: for (size_t i = 0; i < levels_.size(); i++) out.push_back(Label::Text(N(int64_t(i)))); break;
    case Type::Noul: out = {Label::Text(false_label_), Label::Text(true_label_)}; break;
    case Type::Invalid: break;
  }
  return out;
}

std::vector<std::string> Question::RenderedOptions() const {
  std::vector<std::string> out;
  switch (typ_) {
    case Type::Choice:
      for (const auto& o : options_) out.push_back(o.crit.set ? o.label.String() + ": " + o.crit.text : o.label.String());
      break;
    case Type::Score:
      for (size_t i = 0; i < levels_.size(); i++) out.push_back("level " + N(int64_t(i)) + ": " + levels_[i].text);
      break;
    case Type::Noul:
      out.push_back(false_label_ + ": " + (noul_[0].set ? noul_[0].text : std::string(kDefaultFalse)));
      out.push_back(true_label_ + ": " + (noul_[1].set ? noul_[1].text : std::string(kDefaultTrue)));
      break;
    case Type::Invalid: break;
  }
  return out;
}

std::vector<std::string> Question::Legend() const {
  std::vector<std::string> out;
  if (typ_ == Type::Score) for (const auto& l : levels_) out.push_back(l.text);
  return out;
}

Result<std::string> Question::ToJson() const {
  if (typ_ == Type::Invalid) return Refuse<std::string>("schema: cannot marshal the zero Question");
  std::string b = "{\"type\":\"" + std::string(Name(typ_)) + "\",\"instructions\":" + JsonString(instructions_);
  switch (typ_) {
    case Type::Choice: {
      bool described = false;
      for (const auto& o : options_) described = described || o.crit.set;
      if (!described) {
        b += ",\"criteria\":[";
        for (size_t i = 0; i < options_.size(); i++) b += (i ? "," : "") + options_[i].label.Json();
        b += "]";
      } else {
        b += ",\"criteria\":{";
        for (size_t i = 0; i < options_.size(); i++) {
          b += (i ? "," : "") + JsonString(options_[i].label.Key()) + ":" + (options_[i].crit.set ? options_[i].crit.raw : "null");
        }
        b += "}";
      }
      break;
    }
    case Type::Score:
      b += ",\"criteria\":[";
      for (size_t i = 0; i < levels_.size(); i++) b += (i ? "," : "") + levels_[i].raw;
      b += "]";
      break;
    case Type::Noul:
      if (noul_[0].set || noul_[1].set) {
        b += ",\"criteria\":{";
        const char* sep = "";
        const char* keys[2] = {"false", "true"};
        for (int i = 0; i < 2; i++) {
          if (!noul_[i].set) continue;
          b += std::string(sep) + "\"" + keys[i] + "\":" + noul_[i].raw;
          sep = ",";
        }
        b += "}";
      }
      if (custom_labels_) {
        b += ",\"labels\":{\"false\":" + JsonString(false_label_) + ",\"true\":" + JsonString(true_label_) + "}";
      }
      break;
    case Type::Invalid: break;
  }
  if (!order_.empty()) {
    b += ",\"option_order\":[";
    for (size_t i = 0; i < order_.size(); i++) b += (i ? "," : "") + N(order_[i]);
    b += "]";
  }
  b += "}";
  return Result<std::string>::Ok(std::move(b));
}

Question Question::WithoutOptionOrder() const {
  Question q = *this;
  q.order_.clear();
  return q;
}

Result<Question> Question::Subset(const std::vector<int>& indices) const {
  if (typ_ != Type::Choice) {
    return Refuse<Question>("only a choice question has options to subset, not " + std::string(Name(typ_)));
  }
  if (indices.empty()) return Refuse<Question>(kChoiceEmpty);
  std::vector<bool> seen(options_.size(), false);
  Question q;
  q.typ_ = Type::Choice;
  q.instructions_ = instructions_;
  for (size_t i = 0; i < indices.size(); i++) {
    int ix = indices[i];
    if (ix < 0 || size_t(ix) >= options_.size() || seen[size_t(ix)]) {
      return Refuse<Question>("subset index " + N(ix) + " at position " + N(int64_t(i)) + " is out of range or repeated");
    }
    seen[size_t(ix)] = true;
    q.options_.push_back(options_[size_t(ix)]);
  }
  return Result<Question>::Ok(std::move(q));
}

// ---- wire form ---------------------------------------------------------------------------------

namespace {

Result<std::string> WireInstructions(const Value& def) {
  const Value* ins = def.Find("instructions");
  if (ins == nullptr) return Refuse<std::string>(kNoInstructions);
  switch (ins->kind()) {
    case Kind::Null: return Refuse<std::string>(kNilInstructions);
    case Kind::String: {
      if (auto r = CheckInstructions(ins->as_string()); !r.empty()) return Refuse<std::string>(r);
      return Result<std::string>::Ok(ins->as_string());
    }
    case Kind::Array:
    case Kind::Object:
      if (ins->size() == 0) return Refuse<std::string>(kBlankIns);
      break;
    default: break;
  }
  auto text = pyvalue::Dumps(*ins);  // _to_internal: json.dumps(ins, ensure_ascii=False)
  if (!text.ok()) return Refuse<std::string>("'instructions' cannot be written as JSON: " + text.error());
  return text;
}

Result<std::vector<ChoiceOption>> WireChoice(const Value* crit) {
  using R = Result<std::vector<ChoiceOption>>;
  std::vector<Label> labels;
  std::vector<Value> values;
  if (crit != nullptr && crit->kind() == Kind::Object) {
    for (size_t i = 0; i < crit->size(); i++) {
      labels.push_back(Label::Text(crit->key(i)));
      values.push_back(crit->value(i));
    }
  } else if (crit != nullptr && crit->kind() == Kind::Array) {
    for (size_t i = 0; i < crit->size(); i++) {
      const Value& l = crit->value(i);
      switch (l.kind()) {
        case Kind::String: case Kind::Bool: case Kind::Int: case Kind::BigInt: case Kind::Float: break;
        case Kind::Null:
          return R::Fail("choice label " + N(int64_t(i)) +
                         " is null; a label is rendered as option text and used as the answer key, so it must be a "
                         "string, number or bool -- a null label renders as the text \"None\" while its answer key is "
                         "\"null\"");
        default:
          return R::Fail("choice label " + N(int64_t(i)) + " is a " + pyvalue::TypeName(l) +
                         "; a label is rendered as option text and used as the answer key, so it must be a scalar (a "
                         "string, number or bool), got " + pyvalue::Repr(l));
      }
      labels.push_back(Label::Of(l).value());
      values.push_back(Value());
    }
    if (auto r = CheckDistinct(labels); !r.empty()) return R::Fail(r);
  } else {
    return R::Fail(kChoiceShape);
  }
  if (labels.empty()) return R::Fail(kChoiceEmpty);
  std::vector<ChoiceOption> opts;
  for (size_t i = 0; i < labels.size(); i++) {
    std::string err;
    Criterion c = NewCriterion(values[i], &err);
    if (!err.empty()) return R::Fail("choice option " + labels[i].Repr() + ": criterion cannot be rendered: " + err);
    opts.push_back({labels[i], std::move(c)});
  }
  return R::Ok(std::move(opts));
}

Result<std::vector<Criterion>> WireScore(const Value* crit) {
  using R = Result<std::vector<Criterion>>;
  if (crit == nullptr || crit->kind() != Kind::Array) return R::Fail(kScoreShape);
  if (crit->size() == 0) return R::Fail(kScoreEmpty);
  std::vector<Criterion> out;
  for (size_t i = 0; i < crit->size(); i++) {
    const Value& v = crit->value(i);
    if (v.is_null()) return R::Fail(ScoreNullReason(i));
    std::string err;
    Criterion c = NewCriterion(v, &err);
    if (!err.empty()) return R::Fail("score level " + N(int64_t(i)) + " cannot be rendered: " + err);
    if (!c.set) c = {true, "", "\"\""};
    out.push_back(std::move(c));
  }
  return R::Ok(std::move(out));
}

bool WireNoul(const Value* crit, Criterion (&out)[2], std::string* reason) {
  if (crit == nullptr || crit->is_null()) return true;
  if (crit->kind() != Kind::Object) {
    *reason = kNoulShape;
    return false;
  }
  std::map<std::string, Value> sides;  // _to_internal: {str(k).lower(): v}; a later key wins
  for (size_t i = 0; i < crit->size(); i++) sides[ud::Lower(crit->key(i))] = crit->value(i);
  bool unknown = false;
  Value keys = Value::Array();
  for (const auto& kv : sides) {
    unknown = unknown || (kv.first != "true" && kv.first != "false");
    keys.Push(Value::String(kv.first));
  }
  if (unknown) {
    *reason = NoulKeysReason(pyvalue::Repr(keys));
    return false;
  }
  const char* names[2] = {"false", "true"};
  for (int i = 0; i < 2; i++) {
    auto it = sides.find(names[i]);
    std::string err;
    out[i] = NewCriterion(it == sides.end() ? Value() : it->second, &err);
    if (!err.empty()) {
      *reason = std::string("noul ") + names[i] + " description cannot be rendered: " + err;
      return false;
    }
  }
  return true;
}

}  // namespace

Result<Question> Question::FromWire(const Value& def) {
  if (def.kind() != Kind::Object) return Refuse<Question>("definition must be a dict, got " + pyvalue::TypeName(def));
  const Value* tv = def.Find("type");
  Type typ = Type::Invalid;
  if (tv != nullptr && tv->kind() == Kind::String) {
    const std::string& s = tv->as_string();
    typ = s == "choice" ? Type::Choice : s == "score" ? Type::Score : s == "noul" ? Type::Noul : Type::Invalid;
  }
  if (typ == Type::Invalid) {
    return Refuse<Question>("unknown type " + pyvalue::Repr(tv == nullptr ? Value() : *tv) +
                            "; use one of ['choice', 'noul', 'score']");
  }
  auto ins = WireInstructions(def);
  if (!ins.ok()) return Refuse<Question>(ins.error());
  Question q;
  q.typ_ = typ;
  q.instructions_ = ins.value();
  const Value* crit = def.Find("criteria");
  if (typ == Type::Choice) {
    auto r = WireChoice(crit);
    if (!r.ok()) return Refuse<Question>(r.error());
    q.options_ = std::move(r).value();
  } else if (typ == Type::Score) {
    auto r = WireScore(crit);
    if (!r.ok()) return Refuse<Question>(r.error());
    q.levels_ = std::move(r).value();
  } else {
    std::string reason;
    if (!WireNoul(crit, q.noul_, &reason)) return Refuse<Question>(reason);
    q.false_label_ = "false";
    q.true_label_ = "true";
  }
  if (const Value* order = def.Find("option_order")) {
    size_t n = q.NumOptions();
    bool good = order->kind() == Kind::Array && order->size() == n;
    std::vector<int> ints;
    std::vector<bool> seen(n, false);
    for (size_t i = 0; good && i < n; i++) {
      const Value& v = order->value(i);  // only true ints (not bools or floats) count
      if (v.kind() != Kind::Int || v.as_int() < 0 || uint64_t(v.as_int()) >= n || seen[size_t(v.as_int())]) {
        good = false;
      } else {
        seen[size_t(v.as_int())] = true;
        ints.push_back(int(v.as_int()));
      }
    }
    if (!good) return Refuse<Question>(OrderReason(n, pyvalue::Repr(*order)));
    q.order_ = std::move(ints);
  }
  if (const Value* labels = def.Find("labels")) {
    if (typ != Type::Noul) return Refuse<Question>(kLabelsOnlyNoul);
    if (!labels->is_null()) {  // _resolve_noul_labels(None) is the default pair
      if (labels->kind() != Kind::Object || labels->size() != 2) return Refuse<Question>(kNoulLabels);
      std::string f, t;
      bool have_f = false, have_t = false;
      for (size_t i = 0; i < 2; i++) {
        bool is_str = labels->value(i).kind() == Kind::String;
        const std::string s = is_str ? labels->value(i).as_string() : "";
        if (labels->key(i) == "false") { f = s; have_f = is_str; }
        else if (labels->key(i) == "true") { t = s; have_t = is_str; }
        else return Refuse<Question>(kNoulLabels);
      }
      f = ud::Strip(f);
      t = ud::Strip(t);
      if (!have_f || !have_t || f.empty() || t.empty() || f == t) return Refuse<Question>(kNoulLabels);
      q.false_label_ = f;
      q.true_label_ = t;
      q.custom_labels_ = true;
    }
  }
  return Result<Question>::Ok(std::move(q));
}

// ---- Questions ---------------------------------------------------------------------------------

namespace {
Result<Questions> IdRefusal(const std::string& id, const std::string& reason) {
  return Result<Questions>::Fail(id.empty() ? reason : "question " + pyjson::Repr(id) + ": " + reason);
}
std::string CheckId(const std::string& id) {
  if (ud::IsBlank(id)) return "question id must be a non-empty string, got " + pyjson::Repr(id);
  return "";
}
}  // namespace

Result<Questions> Questions::Of(const std::vector<Entry>& entries) {
  Questions qs;
  for (const auto& e : entries) {
    if (auto r = CheckId(e.id); !r.empty()) return Result<Questions>::Fail(r);
    if (e.question.type() == Type::Invalid) {
      return IdRefusal(e.id, "zero Question; build one with Choice, ChoiceOf, Score, Noul");
    }
    if (qs.index_.count(e.id)) {
      return IdRefusal(e.id, "id is given twice; answers are keyed by id, so every question needs its own");
    }
    qs.index_[e.id] = qs.entries_.size();
    qs.entries_.push_back(e);
  }
  return Result<Questions>::Ok(std::move(qs));
}

Result<Questions> Questions::Parse(std::string_view json) {
  auto v = pyvalue::Parse(json);
  if (!v.ok()) return Result<Questions>::Fail("schema: questions: " + v.error());
  const Value& root = v.value();
  if (root.kind() != Kind::Object) {
    return Result<Questions>::Fail("schema: questions must be a JSON object, got " + pyvalue::TypeName(root));
  }
  Questions qs;
  for (size_t i = 0; i < root.size(); i++) {
    const std::string& id = root.key(i);
    if (auto r = CheckId(id); !r.empty()) return Result<Questions>::Fail(r);
    auto q = Question::FromWire(root.value(i));
    if (!q.ok()) return IdRefusal(id, q.error());
    qs.index_[id] = qs.entries_.size();
    qs.entries_.push_back({id, std::move(q).value()});
  }
  return Result<Questions>::Ok(std::move(qs));
}

const Question* Questions::Get(std::string_view id) const {
  auto it = index_.find(id);
  return it == index_.end() ? nullptr : &entries_[it->second].question;
}

std::vector<std::string> Questions::Ids() const {
  std::vector<std::string> out;
  for (const auto& e : entries_) out.push_back(e.id);
  return out;
}

Result<std::string> Questions::ToJson() const {
  std::string b = "{";
  for (size_t i = 0; i < entries_.size(); i++) {
    auto q = entries_[i].question.ToJson();
    if (!q.ok()) return q;
    b += (i ? "," : "") + JsonString(entries_[i].id) + ":" + q.value();
  }
  return Result<std::string>::Ok(b + "}");
}

// ---- answers and prediction --------------------------------------------------------------------

namespace {

Result<std::string> Float(double f) {
  if (std::isnan(f) || std::isinf(f)) return Refuse<std::string>("schema: cannot marshal " + pyjson::ReprFloat(f) + " as JSON");
  return Result<std::string>::Ok(pyjson::ReprFloat(f));
}

#define LAYA_TRY_APPEND(out, expr) \
  do {                             \
    auto r_ = (expr);              \
    if (!r_.ok()) return r_;       \
    (out) += r_.value();           \
  } while (0)

Result<std::string> Probabilities(const std::vector<Probability>& ps) {
  std::string b = "{";
  for (size_t i = 0; i < ps.size(); i++) {
    b += (i ? "," : "") + JsonString(ps[i].label.Key()) + ":";
    LAYA_TRY_APPEND(b, Float(ps[i].p));
  }
  return Result<std::string>::Ok(b + "}");
}

}  // namespace

Result<std::string> Answer::ToJson() const {
  std::string b;
  switch (type) {
    case Type::Choice:
      b = "{\"type\":\"choice\",\"choice\":" + choice.Json() + ",\"probabilities\":";
      LAYA_TRY_APPEND(b, Probabilities(probabilities));
      break;
    case Type::Score:
      b = "{\"type\":\"score\",\"score\":";
      LAYA_TRY_APPEND(b, Float(score));
      b += ",\"legend\":{";
      for (size_t i = 0; i < legend.size(); i++) b += (i ? ",\"" : "\"") + N(int64_t(i)) + "\":" + JsonString(legend[i]);
      b += "},\"probabilities\":";
      LAYA_TRY_APPEND(b, Probabilities(probabilities));
      break;
    case Type::Noul:
      b = "{\"type\":\"noul\",\"noul\":";
      LAYA_TRY_APPEND(b, Float(noul));
      break;
    case Type::Invalid: return Refuse<std::string>("schema: cannot marshal an answer of no type");
  }
  b += ",\"confidence\":";
  LAYA_TRY_APPEND(b, Float(confidence));
  b += ",\"answer_confidence\":";
  LAYA_TRY_APPEND(b, Float(answer_confidence));
  b += ",\"action\":{\"act_probability\":";
  LAYA_TRY_APPEND(b, Float(act_probability));
  b += "}";
  if (window) {
    b += ",\"window\":{\"index\":" + N(window->index) + ",\"token_start\":" + N(window->token_start) +
         ",\"token_end\":" + N(window->token_end) + ",\"count\":" + N(window->count) + "}";
  }
  if (low_confidence) b += ",\"low_confidence\":true";
  if (!abstention.empty()) b += ",\"abstention\":" + JsonString(abstention);
  if (abstention_threshold) {
    b += ",\"abstention_threshold\":";
    LAYA_TRY_APPEND(b, Float(*abstention_threshold));
  }
  return Result<std::string>::Ok(b + "}");
}

namespace {

Result<double> Number(const Value& v, const std::string& what) {
  switch (v.kind()) {
    case Kind::Int: return Result<double>::Ok(double(v.as_int()));
    case Kind::BigInt: return Result<double>::Ok(std::strtod(v.big_int().c_str(), nullptr));
    case Kind::Float: return Result<double>::Ok(v.as_float());
    default: break;
  }
  return Result<double>::Fail("schema: \"" + what + "\" must be a number, got " + pyvalue::TypeName(v));
}

Result<const Value*> Field(const Value& obj, const char* key) {
  const Value* v = obj.Find(key);
  if (v == nullptr) return Result<const Value*>::Fail(std::string("schema: answer has no \"") + key + "\"");
  return Result<const Value*>::Ok(v);
}

Result<double> FieldNumber(const Value& obj, const char* key) {
  auto v = Field(obj, key);
  if (!v.ok()) return Result<double>::Fail(v.error());
  return Number(*v.value(), key);
}

Result<std::vector<Probability>> ProbsFrom(const Value& v) {
  using R = Result<std::vector<Probability>>;
  if (v.kind() != Kind::Object) return R::Fail("schema: probabilities must be an object, got " + pyvalue::TypeName(v));
  std::vector<Probability> out;
  for (size_t i = 0; i < v.size(); i++) {
    auto p = Number(v.value(i), v.key(i));
    if (!p.ok()) return R::Fail(p.error());
    out.push_back({Label::Text(v.key(i)), p.value()});
  }
  return R::Ok(std::move(out));
}

}  // namespace

Result<Answer> Answer::FromJson(const Value& v) {
  using R = Result<Answer>;
  if (v.kind() != Kind::Object) return R::Fail("schema: an answer is a JSON object, got " + pyvalue::TypeName(v));
  Answer a;
  auto tv = Field(v, "type");
  if (!tv.ok()) return R::Fail(tv.error());
  if (tv.value()->kind() != Kind::String) {
    return R::Fail("schema: answer \"type\" must be a string, got " + pyvalue::TypeName(*tv.value()));
  }
  const std::string& ts = tv.value()->as_string();
  a.type = ts == "choice" ? Type::Choice : ts == "score" ? Type::Score : ts == "noul" ? Type::Noul : Type::Invalid;
  auto conf = FieldNumber(v, "confidence");
  if (!conf.ok()) return R::Fail(conf.error());
  a.confidence = conf.value();
  auto ac = FieldNumber(v, "answer_confidence");
  if (!ac.ok()) return R::Fail(ac.error());
  a.answer_confidence = ac.value();
  auto act = Field(v, "action");
  if (!act.ok()) return R::Fail(act.error());
  if (act.value()->kind() != Kind::Object) return R::Fail("schema: answer \"action\" must be an object");
  auto ap = FieldNumber(*act.value(), "act_probability");
  if (!ap.ok()) return R::Fail(ap.error());
  a.act_probability = ap.value();
  if (const Value* w = v.Find("window")) {
    if (w->kind() != Kind::Object) return R::Fail("schema: answer \"window\" must be an object");
    int64_t f[4];
    const char* keys[4] = {"index", "token_start", "token_end", "count"};
    for (int i = 0; i < 4; i++) {
      auto n = FieldNumber(*w, keys[i]);
      if (!n.ok() || n.value() != std::trunc(n.value()) || std::fabs(n.value()) > 9.0e15) {
        return R::Fail(std::string("schema: answer window \"") + keys[i] + "\" must be an integer");
      }
      f[i] = int64_t(n.value());
    }
    if (f[3] < 1 || f[0] < 0 || f[0] >= f[3] || f[1] < 0 || f[2] < f[1]) return R::Fail("schema: answer \"window\" is out of range");
    a.window = Window{f[0], f[1], f[2], f[3]};
  }
  if (const Value* lc = v.Find("low_confidence")) {
    if (lc->kind() != Kind::Bool || !lc->as_bool()) return R::Fail("schema: answer \"low_confidence\" must be true when present");
    a.low_confidence = true;
  }
  if (const Value* ab = v.Find("abstention")) {
    if (ab->kind() != Kind::String || (ab->as_string() != "passed" && ab->as_string() != "abstained" && ab->as_string() != "unevaluated")) {
      return R::Fail("schema: answer \"abstention\" must be one of \"passed\", \"abstained\", \"unevaluated\", got " + pyvalue::Repr(*ab));
    }
    a.abstention = ab->as_string();
  }
  if (v.Find("abstention_threshold") != nullptr) {
    auto t = FieldNumber(v, "abstention_threshold");
    if (!t.ok()) return R::Fail(t.error());
    a.abstention_threshold = t.value();
  }
  if (a.abstention.empty() != !a.abstention_threshold.has_value()) {
    return R::Fail("schema: answer \"abstention\" and \"abstention_threshold\" come together");
  }
  if (a.type == Type::Choice) {
    auto c = Field(v, "choice");
    if (!c.ok()) return R::Fail(c.error());
    auto l = Label::Of(*c.value());
    if (!l.ok()) return R::Fail("schema: answer \"choice\" must be a string, bool or number, got " + pyvalue::TypeName(*c.value()));
    a.choice = l.value();
  } else if (a.type == Type::Score) {
    auto s = FieldNumber(v, "score");
    if (!s.ok()) return R::Fail(s.error());
    a.score = s.value();
    auto lg = Field(v, "legend");
    if (!lg.ok()) return R::Fail(lg.error());
    const Value& lo = *lg.value();
    if (lo.kind() != Kind::Object) return R::Fail("schema: answer \"legend\" must be an object");
    for (size_t i = 0; i < lo.size(); i++) {
      if (lo.value(i).kind() != Kind::String || lo.key(i) != N(int64_t(i))) {
        return R::Fail("schema: answer \"legend\" must map \"0\", \"1\", ... to text, got key " + lo.key(i));
      }
      a.legend.push_back(lo.value(i).as_string());
    }
  } else if (a.type == Type::Noul) {
    auto n = FieldNumber(v, "noul");
    if (!n.ok()) return R::Fail(n.error());
    a.noul = n.value();
  } else {
    return R::Fail("schema: unknown answer type " + pyjson::Repr(ts));
  }
  if (a.type != Type::Noul) {
    auto p = Field(v, "probabilities");
    if (!p.ok()) return R::Fail(p.error());
    auto ps = ProbsFrom(*p.value());
    if (!ps.ok()) return R::Fail(ps.error());
    a.probabilities = std::move(ps).value();
  }
  return R::Ok(std::move(a));
}

Result<std::string> Usage::ToJson() const {
  if (bare) return Result<std::string>::Ok("{\"input_tokens\":" + N(input_tokens) + ",\"output_tokens\":" + N(output_tokens) + "}");
  bool scanned = windows > 1;
  if (scanned && state_tokens == 0 && state_tokens_dropped == 0 && truncated_windows == 0 &&
      truncated_questions.empty() && collapsed_options.empty()) {
    return Result<std::string>::Ok("{\"input_tokens\":" + N(input_tokens) + ",\"output_tokens\":" + N(output_tokens) +
                                   ",\"windows\":" + N(windows) + "}");
  }
  std::string b = "{\"input_tokens\":" + N(input_tokens) + ",\"output_tokens\":" + N(output_tokens) +
                  ",\"state_tokens\":" + N(state_tokens) + ",\"state_tokens_dropped\":" + N(state_tokens_dropped) +
                  ",\"truncated\":" + (scanned ? N(truncated_windows) : std::string(truncated ? "true" : "false")) +
                  ",\"truncated_questions\":[";
  for (size_t i = 0; i < truncated_questions.size(); i++) b += (i ? "," : "") + JsonString(truncated_questions[i]);
  b += "]";
  if (!collapsed_options.empty()) {
    b += ",\"options\":{";
    for (size_t i = 0; i < collapsed_options.size(); i++) {
      const auto& c = collapsed_options[i];
      b += (i ? "," : "") + JsonString(c.id) + ":{\"total\":" + N(c.options.total) + ",\"distinct\":" +
           N(c.options.distinct) + ",\"tokens_per_option\":" +
           (c.options.tokens_per_option ? N(*c.options.tokens_per_option) : std::string("null")) + "}";
    }
    b += "}";
  }
  if (windows > 0) b += ",\"windows\":" + N(windows);
  return Result<std::string>::Ok(b + "}");
}

Result<std::string> Prediction::ToJson() const {
  std::string b = "{\"model\":" + JsonString(model) + ",\"answers\":{";
  for (size_t i = 0; i < answers.size(); i++) {
    auto a = answers[i].answer.ToJson();
    if (!a.ok()) return a;
    b += (i ? "," : "") + JsonString(answers[i].id) + ":" + a.value();
  }
  b += "},\"usage\":";
  LAYA_TRY_APPEND(b, usage.ToJson());
  return Result<std::string>::Ok(b + "}");
}

}  // namespace schema
}  // namespace laya
