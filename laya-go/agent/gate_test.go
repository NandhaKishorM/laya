package agent_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

const gateKeys = `"abstention":`

func TestUngatedWritesNoGateField(t *testing.T) {
	a, _ := open(t, `{}`)
	p, err := a.Predict(context.Background(), "hello", questions(t))
	if err != nil {
		t.Fatal(err)
	}
	if s := jsonOf(t, p); strings.Contains(s, "abstention") || strings.Contains(s, "low_confidence") {
		t.Fatalf("an ungated answer carries a gate field: %s", s)
	}
}

func TestGateStatesOnPredict(t *testing.T) {
	a, _ := open(t, `{}`)
	ctx := context.Background()
	// 0 flags nothing and still reports the gate ran; 1 flags every answer short of certainty.
	p0, err := a.Predict(ctx, "hello", questions(t), agent.WithMinConfidence(0))
	if err != nil {
		t.Fatal(err)
	}
	if s := jsonOf(t, p0); strings.Count(s, `"abstention":"passed","abstention_threshold":0.0`) != 3 || strings.Contains(s, "low_confidence") {
		t.Fatalf("threshold 0: %s", s)
	}
	p1, err := a.Predict(ctx, "hello", questions(t), agent.WithMinConfidence(1))
	if err != nil {
		t.Fatal(err)
	}
	s := jsonOf(t, p1)
	if strings.Count(s, `"low_confidence":true,"abstention":"abstained","abstention_threshold":1.0`) != 3 {
		t.Fatalf("threshold 1: %s", s)
	}
}

func TestGateAppliesToEveryStateOfABatch(t *testing.T) {
	a, _ := open(t, `{}`)
	ps, err := a.PredictBatch(context.Background(), []any{"one", "two"}, questions(t), agent.WithMinConfidence(1))
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range ps {
		if n := strings.Count(jsonOf(t, p), gateKeys); n != 3 {
			t.Errorf("state %d: %d gated answers, want 3", i, n)
		}
	}
}

func TestGateMapPicksTheBucket(t *testing.T) {
	a, _ := open(t, `{}`)
	// choice has 3 options (3-5), score 3 levels (3-5), noul is 2. Only the choice bucket is named.
	p, err := a.Predict(context.Background(), "hello", questions(t),
		agent.WithMinConfidenceMap(map[string]float64{"choice:3-5": 1, "noul:2": 0.25}))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{"intent": 1, "urgency": 0, "angry": 0.25}
	for id, thr := range want {
		ans, _ := p.Answers.Get(id)
		var c schema.Common
		switch x := ans.(type) {
		case *schema.ChoiceAnswer:
			c = x.Common
		case *schema.ScoreAnswer:
			c = x.Common
		case *schema.NoulAnswer:
			c = x.Common
		}
		if c.AbstentionThreshold == nil || *c.AbstentionThreshold != thr {
			t.Errorf("%s threshold = %v, want %v", id, c.AbstentionThreshold, thr)
		}
	}
}

func TestInvalidThresholdFailsBeforeAnyWork(t *testing.T) {
	a, s := open(t, `{}`)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, thr := range []float64{1.5, -0.1} {
		// A done context is reported after the threshold, as laya validates first.
		_, err := a.PredictBatch(cancelled, []any{"x"}, questions(t), agent.WithMinConfidence(thr))
		var me *agent.MinConfidenceError
		if !errors.As(err, &me) {
			t.Fatalf("threshold %v: err = %v, want *MinConfidenceError", thr, err)
		}
		if _, err := a.Predict(context.Background(), "x", questions(t), agent.WithMinConfidence(thr)); !errors.As(err, &me) {
			t.Fatalf("Predict %v: err = %v", thr, err)
		}
	}
	if len(s.seen()) != 0 {
		t.Fatal("the session ran for an invalid threshold")
	}
}

func TestPredictLongWritesNoGateField(t *testing.T) {
	a, _ := open(t, `{}`)
	// PredictLong has no gate: laya calls it with None, which writes nothing.
	lp, err := a.PredictLong(context.Background(), "hello", questions(t))
	if err != nil {
		t.Fatal(err)
	}
	if s := jsonOf(t, lp.Prediction); strings.Contains(s, "abstention") || strings.Contains(s, "low_confidence") {
		t.Fatalf("PredictLong carries a gate field: %s", s)
	}
}

func TestPredictLongScanWritesNoGateField(t *testing.T) {
	a, _ := open(t, longConfig)
	lp, err := a.PredictLong(context.Background(), longState(900), questions(t))
	if err != nil {
		t.Fatal(err)
	}
	if lp.Usage.Windows < 3 {
		t.Fatalf("not a scanned state: %d windows", lp.Usage.Windows)
	}
	if s := jsonOf(t, lp.Prediction); strings.Contains(s, "abstention") || strings.Contains(s, "low_confidence") {
		t.Fatalf("a scanned PredictLong carries a gate field: %s", s)
	}
}

func TestPredictLongRefusesTheGate(t *testing.T) {
	// Python's predict_long raises TypeError for min_confidence; ignoring it would hand back
	// ungated answers to a caller who asked for a gate.
	for name, opt := range map[string]agent.CallOption{
		"threshold": agent.WithMinConfidence(0.5),
		"map":       agent.WithMinConfidenceMap(map[string]float64{"default": 0.5}),
	} {
		for _, state := range []any{"hello", longState(900)} {
			a, s := open(t, longConfig)
			_, err := a.PredictLong(context.Background(), state, questions(t), opt)
			if err == nil || !strings.Contains(err.Error(), "PredictLong takes no confidence gate") {
				t.Fatalf("%s: want the gate refused, got %v", name, err)
			}
			if len(s.seen()) != 0 {
				t.Fatalf("%s: the session ran", name)
			}
		}
	}
}
