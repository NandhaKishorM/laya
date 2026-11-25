package agent

import (
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// The states of the confidence gate, laya.confidence.GATE_STATES. There is deliberately no state
// for "no gate was configured": an ungated call writes none of the gate's fields.
const (
	gatePassed      = "passed"
	gateAbstained   = "abstained"
	gateUnevaluated = "unevaluated"
)

// MinConfidenceError is a refused min_confidence; Error is laya's message.
type MinConfidenceError struct{ Message string }

func (e *MinConfidenceError) Error() string { return e.Message }

// WithMinConfidence turns on the confidence gate of Predict and PredictBatch at one threshold, a
// number in [0, 1], for every answer. Each answer then carries "abstention" ("passed",
// "abstained" or "unevaluated") and "abstention_threshold", and "low_confidence" when it fell
// below the threshold. An answer is gated on its AnswerConfidence, or on Confidence when it has
// no usable AnswerConfidence. A threshold outside [0, 1] or NaN makes the call fail with a
// *MinConfidenceError before anything runs. PredictLong takes no gate.
func WithMinConfidence(threshold float64) CallOption {
	return func(c *call) { c.minConfidence, c.gated = threshold, true }
}

// WithMinConfidenceMap is WithMinConfidence with a threshold per option-count bucket: keys like
// "choice:2", "score:6-10" or "noul:2", and "default" for every bucket the map does not name; a
// bucket it does not name and no default covers is gated at 0. An empty map or any other key
// makes the call fail with a *MinConfidenceError.
func WithMinConfidenceMap(thresholds map[string]float64) CallOption {
	keys := make([]string, 0, len(thresholds))
	for k := range thresholds {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	obj := make(pyjson.Object, len(keys))
	for i, k := range keys {
		obj[i] = pyjson.Member{Key: k, Value: thresholds[k]}
	}
	return func(c *call) { c.minConfidence, c.gated = obj, true }
}

// gate is a validated min_confidence: one threshold, or one per bucket.
type gate struct {
	scalar  float64
	buckets map[string]float64 // nil for a scalar gate
}

var bucketSizes = [...]string{"2", "3-5", "6-10", "11+"}

// checkMinConfidence is laya.confidence.check_min_confidence over the Python values a threshold can
// be: a float64 or int, a bool or string (refused), or a pyjson.Object for a map.
func checkMinConfidence(v any) (*gate, error) {
	if m, ok := v.(pyjson.Object); ok {
		return checkMap(m)
	}
	t, err := checkOne(v)
	if err != nil {
		return nil, err
	}
	return &gate{scalar: t}, nil
}

func checkOne(v any) (float64, error) {
	var f float64
	switch x := v.(type) {
	case float64:
		f = x
	case int:
		f = float64(x)
	default:
		return 0, &MinConfidenceError{"min_confidence must be a float in [0.0, 1.0], got " + reprValue(v)}
	}
	if math.IsNaN(f) || math.IsInf(f, 0) || f < 0 || f > 1 {
		return 0, &MinConfidenceError{"min_confidence must be a float in [0.0, 1.0], got " + reprValue(v)}
	}
	return f, nil
}

func checkMap(m pyjson.Object) (*gate, error) {
	if len(m) == 0 {
		return nil, &MinConfidenceError{"a min_confidence map must be a non-empty dict of bucket -> float, got " + reprValue(m)}
	}
	out := make(map[string]float64, len(m))
	for _, mem := range m {
		key, ok := mem.Key.(string)
		if !ok || !validBucketKey(key) {
			return nil, &MinConfidenceError{"min_confidence map keys must be a bucket like 'choice:3-5' or 'default', got " + reprValue(mem.Key)}
		}
		t, err := checkOne(mem.Value)
		if err != nil {
			return nil, err
		}
		out[key] = t
	}
	return &gate{buckets: out}, nil
}

func validBucketKey(k string) bool {
	if k == "default" {
		return true
	}
	qt, size, ok := strings.Cut(k, ":")
	return ok && slices.Contains([]string{"choice", "score", "noul"}, qt) && slices.Contains(bucketSizes[:], size)
}

// reprValue is repr() of the values a threshold can be.
func reprValue(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case string:
		return pyjson.Repr(x)
	case int:
		return strconv.Itoa(x)
	case float64:
		switch {
		case math.IsNaN(x):
			return "nan"
		case math.IsInf(x, 1):
			return "inf"
		case math.IsInf(x, -1):
			return "-inf"
		}
		return pyjson.ReprFloat(x)
	case pyjson.Object:
		parts := make([]string, len(x))
		for i, m := range x {
			parts[i] = reprValue(m.Key) + ": " + reprValue(m.Value)
		}
		return "{" + strings.Join(parts, ", ") + "}"
	}
	return "?"
}

// bucketOf is laya's _option_bucket: the answer's type and option-count band, or "" for an answer
// with no bucket (a choice or score with no probabilities). A noul without them counts as 2.
func bucketOf(a schema.Answer) string {
	var qt string
	var k int
	switch x := a.(type) {
	case *schema.ChoiceAnswer:
		qt, k = "choice", len(x.Probabilities)
	case *schema.ScoreAnswer:
		qt, k = "score", len(x.Probabilities)
	case *schema.NoulAnswer:
		qt, k = "noul", 2 // a noul carries no probabilities, which laya counts as two options
	default:
		return ""
	}
	if k == 0 {
		return ""
	}
	size := "11+"
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	}
	return qt + ":" + size
}

// threshold is laya's resolve_min_confidence: the bucket's entry, else "default", else 0.
func (g *gate) threshold(a schema.Answer) float64 {
	if g.buckets == nil {
		return g.scalar
	}
	if b := bucketOf(a); b != "" {
		if t, ok := g.buckets[b]; ok {
			return t
		}
	}
	return g.buckets["default"]
}

// usable is a number the gate can compare: finite.
func usable(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) }

// gateConfidence is laya's _gate_confidence: AnswerConfidence, else Confidence, else none.
func gateConfidence(c schema.Common) (float64, bool) {
	if usable(c.AnswerConfidence) {
		return c.AnswerConfidence, true
	}
	if usable(c.Confidence) {
		return c.Confidence, true
	}
	return 0, false
}

// apply is laya's apply_confidence_gate for a gate that was set: it flags, then reports the state
// and the threshold on every answer, clearing a flag an earlier run left.
func (g *gate) apply(preds []schema.Prediction) {
	for i := range preds {
		for _, ans := range preds[i].Answers {
			var c *schema.Common
			switch x := ans.Answer.(type) {
			case *schema.ChoiceAnswer:
				c = &x.Common
			case *schema.ScoreAnswer:
				c = &x.Common
			case *schema.NoulAnswer:
				c = &x.Common
			default:
				continue
			}
			thr := g.threshold(ans.Answer)
			conf, ok := gateConfidence(*c)
			c.LowConfidence = ok && conf < thr
			switch {
			case c.LowConfidence:
				c.Abstention = gateAbstained
			case !ok:
				c.Abstention = gateUnevaluated
			default:
				c.Abstention = gatePassed
			}
			c.AbstentionThreshold = &thr
		}
	}
}
