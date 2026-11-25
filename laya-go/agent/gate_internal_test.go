package agent

import (
	"math"
	"strconv"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func probsOf(n int) schema.Probabilities {
	ps := make(schema.Probabilities, n)
	for i := range ps {
		ps[i] = schema.Probability{Label: schema.Text(strconv.Itoa(i)), P: 1 / float64(n)}
	}
	return ps
}

func TestBucketEdges(t *testing.T) {
	for k, want := range map[int]string{1: "choice:2", 2: "choice:2", 3: "choice:3-5", 5: "choice:3-5", 6: "choice:6-10", 10: "choice:6-10", 11: "choice:11+", 40: "choice:11+"} {
		if got := bucketOf(&schema.ChoiceAnswer{Probabilities: probsOf(k)}); got != want {
			t.Errorf("choice k=%d: %q, want %q", k, got, want)
		}
	}
	if got := bucketOf(&schema.ScoreAnswer{Probabilities: probsOf(7)}); got != "score:6-10" {
		t.Errorf("score: %q", got)
	}
	if got := bucketOf(&schema.NoulAnswer{}); got != "noul:2" {
		t.Errorf("noul: %q", got)
	}
	if got := bucketOf(&schema.ChoiceAnswer{}); got != "" {
		t.Errorf("a choice without probabilities has a bucket %q", got)
	}
}

func TestValidThresholds(t *testing.T) {
	for _, v := range []any{0.0, 1.0, 0.5, 0, 1} {
		if _, err := checkMinConfidence(v); err != nil {
			t.Errorf("%v refused: %v", v, err)
		}
	}
	for _, v := range []any{math.Inf(1), math.Inf(-1), 2, -1, nil} {
		if _, err := checkMinConfidence(v); err == nil {
			t.Errorf("%v accepted", v)
		}
	}
	ok := pyjson.Object{{Key: "default", Value: 0.3}, {Key: "noul:11+", Value: 0.2}}
	if _, err := checkMinConfidence(ok); err != nil {
		t.Errorf("valid map refused: %v", err)
	}
	for _, k := range []string{"choice:1", "choice", "other:2", "choice:3-5 ", "Default"} {
		if _, err := checkMinConfidence(pyjson.Object{{Key: k, Value: 0.5}}); err == nil {
			t.Errorf("bucket key %q accepted", k)
		}
	}
}

func TestMapFallsBackToDefaultThenZero(t *testing.T) {
	a := &schema.ChoiceAnswer{Probabilities: probsOf(4)}
	g, _ := checkMinConfidence(pyjson.Object{{Key: "default", Value: 0.4}})
	if got := g.threshold(a); got != 0.4 {
		t.Errorf("default: %v", got)
	}
	g, _ = checkMinConfidence(pyjson.Object{{Key: "choice:2", Value: 0.9}})
	if got := g.threshold(a); got != 0 {
		t.Errorf("no default: %v", got)
	}
}

func TestStaleFlagIsCleared(t *testing.T) {
	c := &schema.ChoiceAnswer{Probabilities: probsOf(2), Common: schema.Common{AnswerConfidence: 0.9, Confidence: 0.5, LowConfidence: true}}
	preds := []schema.Prediction{{Answers: schema.Answers{{ID: "q", Answer: c}}}}
	g, _ := checkMinConfidence(0.5)
	g.apply(preds)
	if c.LowConfidence || c.Abstention != gatePassed {
		t.Fatalf("flag %v state %q", c.LowConfidence, c.Abstention)
	}
}

func TestGateConfidencePrefersAnswerConfidence(t *testing.T) {
	if v, ok := gateConfidence(schema.Common{AnswerConfidence: 0.2, Confidence: 0.9}); !ok || v != 0.2 {
		t.Errorf("got %v %v", v, ok)
	}
	if v, ok := gateConfidence(schema.Common{AnswerConfidence: math.NaN(), Confidence: 0.9}); !ok || v != 0.9 {
		t.Errorf("fallback got %v %v", v, ok)
	}
	if _, ok := gateConfidence(schema.Common{AnswerConfidence: math.NaN(), Confidence: math.Inf(1)}); ok {
		t.Error("non-finite confidences usable")
	}
}
