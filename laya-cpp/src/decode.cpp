#include "laya/decode.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "cnum.hpp"
#include "laya/unicodedata.hpp"

namespace laya {
namespace decode {

namespace {

using pyvalue::Value;
namespace ud = unicodedata;
using Kind = Value::Kind;

constexpr double kInf = std::numeric_limits<double>::infinity();

bool Finite(double x) { return std::isfinite(x); }

std::string Shape(const std::string& why) { return "decode: logits do not fit the question: " + why; }

bool IsDigit(char c) { return c >= '0' && c <= '9'; }

std::string PrimarySubtag(std::string_view lang) {
  return ud::Lower(lang.substr(0, lang.find('-')));
}

// A list of exactly three clamped values, from a Value list.
bool ThreeOf(const Value& list, std::array<double, 3>* out) {
  if (list.kind() != Kind::Array || list.size() != 3) return false;
  for (size_t i = 0; i < 3; i++) (*out)[i] = Clamp(list.items()[i]);
  return true;
}

bool Truthy(const Value* v) {
  if (v == nullptr) return false;
  switch (v->kind()) {
    case Kind::Null: return false;
    case Kind::Bool: return v->as_bool();
    case Kind::Int: return v->as_int() != 0;
    case Kind::Float: return v->as_float() != 0;
    case Kind::String: return !v->as_string().empty();
    case Kind::Array: case Kind::Object: return v->size() != 0;
    default: return true;
  }
}

}  // namespace

double Round4(double v) {
  if (!Finite(v)) return v;
  return cnum::ToDouble(cnum::Fixed(v, 4));
}

std::optional<double> PyFloat(std::string_view in) {
  std::string s = ud::DecimalASCII(ud::StripNumeric(std::string(in)));
  if (s.find_first_of("xXpP") != std::string::npos) return std::nullopt;
  if (s.find('_') != std::string::npos) {
    std::string t;
    for (size_t i = 0; i < s.size(); i++) {
      if (s[i] == '_') {
        if (i == 0 || i + 1 == s.size() || !IsDigit(s[i - 1]) || !IsDigit(s[i + 1])) return std::nullopt;
        continue;
      }
      t.push_back(s[i]);
    }
    s = t;
  }
  size_t i = 0;
  bool neg = false;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) neg = s[i++] == '-';
  std::string rest = s.substr(i), low;
  for (char c : rest) low.push_back(c >= 'A' && c <= 'Z' ? char(c + 32) : c);
  if (low == "nan") return std::numeric_limits<double>::quiet_NaN();
  if (low == "inf" || low == "infinity") return neg ? -kInf : kInf;
  // [digits][.digits][e[+-]digits] with at least one mantissa digit.
  size_t j = 0, digits = 0;
  while (j < rest.size() && IsDigit(rest[j])) j++, digits++;
  if (j < rest.size() && rest[j] == '.') {
    j++;
    while (j < rest.size() && IsDigit(rest[j])) j++, digits++;
  }
  if (digits == 0) return std::nullopt;
  if (j < rest.size() && (rest[j] == 'e' || rest[j] == 'E')) {
    j++;
    if (j < rest.size() && (rest[j] == '+' || rest[j] == '-')) j++;
    size_t e = 0;
    while (j < rest.size() && IsDigit(rest[j])) j++, e++;
    if (e == 0) return std::nullopt;
  }
  if (j != rest.size()) return std::nullopt;
  double v = cnum::ToDouble(rest);  // out of range is infinite, as in Python
  return neg ? -v : v;
}

double Clamp(const Value& v) {
  double f = 0;
  switch (v.kind()) {
    case Kind::Int: f = double(v.as_int()); break;
    case Kind::BigInt: f = cnum::ToDouble(v.big_int()); break;
    case Kind::Float: f = v.as_float(); break;
    case Kind::String: {
      auto p = PyFloat(v.as_string());
      if (!p) return 1;
      f = *p;
      break;
    }
    default: return 1;
  }
  if (!Finite(f)) return 1;
  return std::min(kTempMax, std::max(kTempMin, f));
}

std::string Bucket(schema::Type t, int64_t k) {
  std::string name(schema::Name(t));
  if (k <= 2) return name + ":2";
  if (k <= 5) return name + ":3-5";
  if (k <= 10) return name + ":6-10";
  return name + ":11+";
}

double ApplyBinning(double confidence, const std::string& bucket, const std::map<std::string, Bins>& m) {
  auto it = m.find(bucket);
  if (it == m.end() || it->second.values.empty()) return confidence;
  const auto& v = it->second.values;
  const int64_t n = int64_t(v.size());
  int64_t idx = int64_t(confidence * double(n));
  return v[size_t(std::min(n - 1, std::max<int64_t>(0, idx)))];
}

Result<Temperatures> Temperatures::Of(const Value* by_type, const Value* by_options, const Value* by_language) {
  using R = Result<Temperatures>;
  Temperatures t;
  std::array<double, 3> base{1, 1, 1};
  if (by_type != nullptr && !by_type->is_null()) {
    if (!ThreeOf(*by_type, &base)) return R::Fail("decode: temperature must list 3 values, one per question type");
  }
  t.by_type_ = base;
  if (by_options != nullptr && by_options->kind() == Kind::Object) {
    for (size_t i = 0; i < by_options->size(); i++) t.by_options_[by_options->key(i)] = Clamp(by_options->value(i));
  }
  if (Truthy(by_language)) {
    if (by_language->kind() != Kind::Object) return R::Fail("decode: lang_temperatures must be a mapping of tag -> overrides");
    for (size_t i = 0; i < by_language->size(); i++) {
      const std::string& lang = by_language->key(i);
      const Value& raw = by_language->value(i);
      if (!raw.is_null() && raw.kind() != Kind::Object) {
        return R::Fail("decode: language override '" + lang + "' must be a mapping, got " + pyvalue::TypeName(raw));
      }
      LanguageTemperatures l;
      l.by_type = base;  // null inherits the checkpoint's own
      if (raw.kind() == Kind::Object) {
        if (const Value* tr = raw.Find("temperature"); tr != nullptr && !tr->is_null()) {
          if (!ThreeOf(*tr, &l.by_type)) {
            return R::Fail("decode: language override '" + lang + "' temperature must be a list of 3 floats");
          }
        }
        if (const Value* bo = raw.Find("temperature_by_options"); Truthy(bo)) {
          if (bo->kind() != Kind::Object) {
            return R::Fail("decode: language override '" + lang + "' temperature_by_options must be a mapping of bucket -> float");
          }
          for (size_t j = 0; j < bo->size(); j++) l.by_options[bo->key(j)] = Clamp(bo->value(j));
        }
      }
      t.by_language_[PrimarySubtag(lang)] = std::move(l);  // the last tag with this subtag wins
    }
  }
  return R::Ok(std::move(t));
}

double Temperatures::For(schema::Type typ, int64_t k, std::string_view lang, bool* override_used) const {
  if (override_used) *override_used = false;
  if (schema::Code(typ) < 0) return 1;
  const std::string bucket = Bucket(typ, k);
  auto it = by_options_.find(bucket);
  double temperature = it != by_options_.end() ? it->second : by_type_[size_t(schema::Code(typ))];
  if (!lang.empty()) {
    auto l = by_language_.find(PrimarySubtag(lang));
    if (l != by_language_.end()) {
      auto lo = l->second.by_options.find(bucket);
      if (override_used) *override_used = true;
      return lo != l->second.by_options.end() ? lo->second : l->second.by_type[size_t(schema::Code(typ))];
    }
  }
  return temperature;
}

namespace {

// numpy's float32 sum of a contiguous row (pairwise, eight accumulators from eight values up).
float PairwiseSum(const float* a, size_t n) {
  if (n < 8) {
    float r = 0;
    for (size_t i = 0; i < n; i++) r += a[i];
    return r;
  }
  if (n <= 128) {
    float r[8];
    for (size_t j = 0; j < 8; j++) r[j] = a[j];
    size_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (size_t j = 0; j < 8; j++) r[j] += a[i + j];
    }
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  size_t half = n / 2;
  half -= half % 8;
  return PairwiseSum(a, half) + PairwiseSum(a + half, n - half);
}

std::vector<schema::Probability> Probabilities(const std::vector<schema::Label>& labels, const std::vector<double>& p) {
  std::vector<schema::Probability> out;
  out.reserve(p.size());
  for (size_t i = 0; i < p.size(); i++) out.push_back({labels[i], Round4(p[i])});
  return out;
}

double AnswerConfidence(const std::vector<double>& p) {
  if (p.empty()) return 1;
  double top = p[0];
  for (size_t i = 1; i < p.size(); i++) top = std::max(top, p[i]);
  return std::min(1.0, std::max(0.0, top));
}

// 1 - H(p)/log(k), clipped to [0, 1]; 1 for fewer than two options. Probabilities are clipped to
// [1e-12, 1] inside the log only.
double EntropyConfidence(const std::vector<double>& p) {
  if (p.size() < 2) return 1;
  double h = 0;
  for (double v : p) h -= v * std::log(std::min(1.0, std::max(1e-12, v)));
  return std::min(1.0, std::max(0.0, 1 - h / std::log(double(p.size()))));
}

}  // namespace

Result<schema::Answer> Decode(const schema::Question& q, const Row& row, std::string_view lang, const Temperatures* t) {
  using R = Result<schema::Answer>;
  const schema::Type typ = q.type();
  if (typ == schema::Type::Invalid) return R::Fail(Shape("the zero Question has no answer"));
  const int64_t k = row.k;
  if (k != int64_t(q.NumOptions())) {
    return R::Fail(Shape(std::to_string(k) + " option markers for a question of " + std::to_string(q.NumOptions()) + " options"));
  }
  if (int64_t(row.logits.size()) < k) {
    return R::Fail(Shape(std::to_string(row.logits.size()) + " logits for " + std::to_string(k) + " options"));
  }
  for (int64_t i = 0; i < k; i++) {
    if (!Finite(double(row.logits[size_t(i)]))) return R::Fail(Shape("non-finite logit"));
  }
  if (!Finite(double(row.act))) return R::Fail(Shape("non-finite action probability"));

  bool override_used = false;
  const double temperature = t ? t->For(typ, k, lang, &override_used) : 1.0;
  // softmax(logits[:k] / temperature), the maximum subtracted so a large logit cannot overflow.
  std::vector<float> z;
  z.resize(size_t(k));
  float peak = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = row.logits[i] / float(temperature);  // float32 division, as numpy does it
    if (std::isinf(z[i])) return R::Fail(Shape("a logit over the temperature is not finite"));
    peak = std::max(peak, z[i]);
  }
  std::vector<double> p;
  p.resize(z.size());
  double sum = 0;
  for (size_t i = 0; i < z.size(); i++) {
    p[i] = std::exp(double(z[i]) - double(peak));
    sum += p[i];
  }
  for (double& v : p) v /= sum;
  // The choice is made on float32 probabilities as numpy computes them: a gap below float32 resolution
  // is an exact tie that argmax gives to the lower index. Reported numbers stay double.
  std::vector<float> pf(z.size());
  for (size_t i = 0; i < z.size(); i++) pf[i] = float(std::exp(double(z[i] - peak)));
  const float total = PairwiseSum(pf.data(), pf.size());
  for (float& v : pf) v /= total;
  // The row is in slot order; everything below indexes by option.
  const std::vector<int>& order = q.OptionOrder();
  if (order.size() == p.size()) {
    std::vector<double> canonical(p.size());
    for (size_t slot = 0; slot < order.size(); slot++) canonical[size_t(order[slot])] = p[slot];
    p = std::move(canonical);
    std::vector<float> canonical_f(pf.size());
    for (size_t slot = 0; slot < order.size(); slot++) canonical_f[size_t(order[slot])] = pf[slot];
    pf = std::move(canonical_f);
  }
  schema::Answer a;
  a.type = typ;
  double conf = AnswerConfidence(p);
  if (t != nullptr && !override_used) conf = ApplyBinning(conf, Bucket(typ, k), t->binning());
  a.answer_confidence = Round4(conf);
  a.act_probability = Round4(double(row.act));
  if (typ == schema::Type::Noul) {
    if (k < 2) return R::Fail(Shape("a noul question needs two options"));
    const double yes = p[1];
    a.confidence = Round4(std::max(yes, 1 - yes));
    a.noul = Round4(yes);
    return R::Ok(std::move(a));
  }
  a.confidence = Round4(EntropyConfidence(p));
  const std::vector<schema::Label> labels = q.AnswerLabels();
  if (labels.size() != p.size()) return R::Fail(Shape("label count differs from option count"));
  a.probabilities = Probabilities(labels, p);
  if (typ == schema::Type::Choice) {
    size_t best = 0;
    for (size_t i = 0; i < pf.size(); i++) {
      if (pf[i] > pf[best]) best = i;  // strict: the first maximum wins, as numpy.argmax breaks a tie
    }
    a.choice = labels[best];
  } else {
    double expected = 0;
    for (size_t i = 0; i < p.size(); i++) expected += double(i) * p[i];
    a.score = Round4(expected);
    a.legend = q.Legend();
  }
  return R::Ok(std::move(a));
}

Result<std::vector<float>> ActionProbabilities(const std::vector<float>& x) {
  using R = Result<std::vector<float>>;
  if (x.empty()) return R::Fail(Shape("an empty action row"));
  double peak = -kInf;
  for (float v : x) {
    if (!Finite(double(v))) return R::Fail(Shape("non-finite action logit"));
    peak = std::max(peak, double(v));
  }
  std::vector<double> e(x.size());
  double sum = 0;
  for (size_t i = 0; i < x.size(); i++) {
    e[i] = std::exp(double(x[i]) - peak);
    sum += e[i];
  }
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); i++) out[i] = float(e[i] / sum);
  return R::Ok(std::move(out));
}

}  // namespace decode
}  // namespace laya
