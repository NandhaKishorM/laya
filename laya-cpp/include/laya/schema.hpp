#pragma once
// Question, answer and prediction types (laya-go/schema). Wording of every refusal is laya's own
// Agent._check_question; JSON is spelled as Python's json.dumps spells it, keys in Python's order.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "laya/pyvalue.hpp"
#include "laya/result.hpp"

namespace laya {
namespace schema {

using pyvalue::Value;

enum class Type { Invalid, Choice, Score, Noul };
std::string_view Name(Type t);  // "choice", "score", "noul", "" for Invalid
int Code(Type t);               // the head's integer: 0, 1, 2; -1 for Invalid

// A choice label: a string, or the bool, int or float Python also allows.
class Label {
 public:
  enum class Kind { Text, Bool, Int, Float };
  Label() = default;
  static Label Text(std::string s);
  static Label OfBool(bool b);
  static Label OfInt(int64_t i);
  static Label OfFloat(double d);
  static Result<Label> Of(const Value& v);  // string, bool, int or float; null and containers refused

  Kind kind() const { return kind_; }
  std::string String() const;       // str(label): what the prompt shows
  const std::string& Key() const { return text_; }  // json.dumps spelling: the answer key
  std::string Repr() const;         // repr(label)
  std::string EqKey() const;        // equal for the labels Python's dict keys as one (1, 1.0, True)
  std::string Json() const;         // the label as a JSON value
  bool operator==(const Label& o) const { return kind_ == o.kind_ && text_ == o.text_; }

 private:
  Kind kind_ = Kind::Text;
  std::string text_;
};

struct Option {
  Label label;
  Value criterion;  // Null and "" mean no description
};

struct NoulSpec {
  Value true_text, false_text;  // Null or "" keep the default texts
  std::string true_label, false_label;  // both empty keeps "true" and "false"
};

// A description snapshotted when the question is built.
struct Criterion {
  bool set = false;
  std::string text;  // render_criterion of the value
  std::string raw;   // the value as JSON, for the wire form
};

struct ChoiceOption {
  Label label;
  Criterion crit;
};

// The reason alone is the error of a constructor; ParseQuestions prefixes "question 'id': ".
class Question {
 public:
  static Result<Question> Choice(std::string instructions, const std::vector<Option>& options);
  static Result<Question> ChoiceOf(std::string instructions, const std::vector<std::string>& labels);
  static Result<Question> Score(std::string instructions, const std::vector<Value>& levels);
  static Result<Question> Noul(std::string instructions, const NoulSpec& spec = {});
  // One definition in the Python wire form ({"type", "instructions", "criteria", ...}).
  static Result<Question> FromWire(const Value& definition);

  Result<Question> WithOptionOrder(const std::vector<int>& order) const;  // a permutation of 0..n-1, else refused
  Question WithoutOptionOrder() const;  // restores label order
  Result<Question> Subset(const std::vector<int>& indices) const;

  Type type() const { return typ_; }
  const std::string& instructions() const { return instructions_; }
  size_t NumOptions() const;
  const std::vector<int>& OptionOrder() const { return order_; }  // empty: label order
  std::vector<std::string> Labels() const;        // str() of each label, label-index order
  std::vector<Label> AnswerLabels() const;
  std::vector<std::string> RenderedOptions() const;  // render_options: the text the model reads
  std::vector<std::string> Legend() const;           // score only
  Result<std::string> ToJson() const;

 private:
  Type typ_ = Type::Invalid;
  std::string instructions_;
  std::vector<ChoiceOption> options_;
  std::vector<Criterion> levels_;
  Criterion noul_[2];  // false side, true side
  std::string false_label_, true_label_;
  bool custom_labels_ = false;
  std::vector<int> order_;
};

struct Entry {
  std::string id;
  Question question;
};

class Questions {
 public:
  static Result<Questions> Of(const std::vector<Entry>& entries);
  static Result<Questions> Parse(std::string_view json);
  size_t size() const { return entries_.size(); }
  const Entry& at(size_t i) const { return entries_[i]; }
  const Question* Get(std::string_view id) const;
  std::vector<std::string> Ids() const;
  Result<std::string> ToJson() const;

 private:
  std::vector<Entry> entries_;
  std::map<std::string, size_t, std::less<>> index_;
};

// json.dumps of a string, ensure_ascii=False.
std::string JsonString(std::string_view s);

struct Probability {
  Label label;
  double p = 0;
};

struct Window {
  int64_t index = 0, token_start = 0, token_end = 0, count = 0;
};

// One answer; the fields that apply depend on type.
struct Answer {
  Type type = Type::Invalid;
  Label choice;                              // choice
  double score = 0;                          // score
  double noul = 0;                           // noul
  std::vector<std::string> legend;           // score
  std::vector<Probability> probabilities;    // choice, score
  double confidence = 0, answer_confidence = 0, act_probability = 0;
  std::optional<Window> window;
  bool low_confidence = false;
  std::string abstention;  // "", "passed", "abstained" or "unevaluated"
  std::optional<double> abstention_threshold;

  Result<std::string> ToJson() const;
  static Result<Answer> FromJson(const Value& v);
};

struct Answered {
  std::string id;
  Answer answer;
};

struct CollapsedOptions {
  int64_t total = 0, distinct = 0;
  std::optional<int64_t> tokens_per_option;
};
struct CollapsedQuestion {
  std::string id;
  CollapsedOptions options;
};

struct Usage {
  int64_t input_tokens = 0, output_tokens = 0, state_tokens = 0, state_tokens_dropped = 0;
  bool truncated = false;
  std::vector<std::string> truncated_questions;
  std::vector<CollapsedQuestion> collapsed_options;
  bool bare = false;  // only input_tokens and output_tokens (a call with no questions)
  int64_t windows = 0;
  int64_t truncated_windows = 0;  // with windows > 1, the "truncated" key is this count
  Result<std::string> ToJson() const;
};

constexpr const char* kModel = "laya-rl-agent-onnx";

struct Prediction {
  std::string model = kModel;
  std::vector<Answered> answers;
  Usage usage;
  Result<std::string> ToJson() const;
};

}  // namespace schema
}  // namespace laya
