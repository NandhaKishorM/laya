#pragma once
// Head logits to typed answers (laya-go/internal/decode, Agent._decode_answers): slice the row to
// the question's option count, divide by the temperature its bucket names (in float32, as numpy
// does), softmax with the maximum subtracted, undo the option order, derive the type's fields and
// round every number to four decimals as Python's round(x, 4) does.
//
// Reported numbers are double after the float32 quotient. The choice is made on float32 probabilities
// (exp rounded to float32, numpy's pairwise float32 sum), first maximum wins, as numpy.argmax does.
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "laya/pyvalue.hpp"
#include "laya/result.hpp"
#include "laya/schema.hpp"

namespace laya {
namespace decode {

// A fitted temperature below 0.5 would sharpen a coin flip into a certainty, so values are clamped.
constexpr double kTempMin = 0.5;
constexpr double kTempMax = 5.0;

double Round4(double v);  // round(v, 4): half to even on the exact binary value
// clamp_temperature: v confined to [kTempMin, kTempMax]; 1.0 for a bool, a non-number, NaN or infinity.
double Clamp(const pyvalue::Value& v);
// float(s) for a string: surrounding space, Unicode decimal digits, single underscores between digits.
std::optional<double> PyFloat(std::string_view s);
// temp_bucket: "<type>:2", ":3-5", ":6-10" or ":11+" by option count.
std::string Bucket(schema::Type t, int64_t options);

struct Bins {
  std::vector<double> values;  // a confidence c maps to values[min(n-1, max(0, int(c*n)))]
};
double ApplyBinning(double confidence, const std::string& bucket, const std::map<std::string, Bins>& m);

struct LanguageTemperatures {
  std::array<double, 3> by_type{1, 1, 1};
  std::map<std::string, double> by_options;
};

// A checkpoint's temperature tables, every value already clamped. Immutable once built apart from SetBinning.
class Temperatures {
 public:
  Temperatures() = default;  // all 1.0
  // The raw config values, each nullptr when the key is absent. by_type is null or a list of 3;
  // by_options an object of bucket -> number; by_language an object of tag -> null or
  // {"temperature": list of 3 or null, "temperature_by_options": object}. When two tags share a
  // primary subtag the last one listed wins, as in Python (laya-go refuses it).
  static Result<Temperatures> Of(const pyvalue::Value* by_type, const pyvalue::Value* by_options,
                                 const pyvalue::Value* by_language = nullptr);
  // The temperature for a question of type t with k options, and whether a language override chose it.
  double For(schema::Type t, int64_t k, std::string_view lang, bool* override_used = nullptr) const;
  const std::array<double, 3>& by_type() const { return by_type_; }
  const std::map<std::string, double>& by_options() const { return by_options_; }
  // Recalibration bins by bucket; empty means none.
  const std::map<std::string, Bins>& binning() const { return binning_; }
  void SetBinning(std::map<std::string, Bins> b) { binning_ = std::move(b); }

 private:
  std::map<std::string, Bins> binning_;
  std::array<double, 3> by_type_{1, 1, 1};
  std::map<std::string, double> by_options_;
  std::map<std::string, LanguageTemperatures> by_language_;
};

// One question's slice of the graph's outputs.
struct Row {
  std::vector<float> logits;  // at least k wide; columns past k are ignored
  int64_t k = 0;              // how many option markers the question's sequence kept
  float act = 0;              // the action probability
};

Result<schema::Answer> Decode(const schema::Question& q, const Row& row, std::string_view lang,
                              const Temperatures* t);
// softmax of one row of act_logits; the decoder takes probabilities, never the raw logit.
Result<std::vector<float>> ActionProbabilities(const std::vector<float>& act_logits);

}  // namespace decode
}  // namespace laya
