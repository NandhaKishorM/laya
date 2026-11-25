package agent

import (
	"math"
	"strconv"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// goldenConf reads a recorded confidence: absent, a bool, "__nan__" or a number.
func goldenConf(tb testing.TB, o fixtures.Object, key string) float64 {
	tb.Helper()
	v, ok := o.Get(key)
	if !ok {
		return math.NaN() // laya's absent confidence reads as no usable value
	}
	switch x := v.(type) {
	case bool:
		return math.NaN() // laya refuses a bool as a confidence
	case string:
		if x != "__nan__" {
			tb.Fatalf("unexpected confidence %q", x)
		}
		return math.NaN()
	}
	return fixtures.Float(tb, v)
}

func goldenAnswer(tb testing.TB, o fixtures.Object) schema.Answer {
	tb.Helper()
	common := schema.Common{
		AnswerConfidence: goldenConf(tb, o, "answer_confidence"),
		Confidence:       goldenConf(tb, o, "confidence"),
	}
	// A recorded bool is not a confidence; a number under the other key must still be read.
	if v, ok := o.Get("low_confidence"); ok {
		common.LowConfidence = fixtures.Bool(tb, v)
	}
	var probs schema.Probabilities
	if v, ok := o.Get("probabilities"); ok {
		for i, m := range fixtures.Obj(tb, v) {
			probs = append(probs, schema.Probability{Label: schema.Text(strconv.Itoa(i)), P: fixtures.Float(tb, m.Value)})
		}
	}
	switch fixtures.Str(tb, o.Must("type")) {
	case "choice":
		return &schema.ChoiceAnswer{Probabilities: probs, Common: common}
	case "score":
		return &schema.ScoreAnswer{Probabilities: probs, Common: common}
	case "noul":
		return &schema.NoulAnswer{Common: common}
	}
	tb.Fatalf("unknown answer type in %v", o.Keys())
	return nil
}

func goldenThreshold(tb testing.TB, v any) (any, bool) {
	tb.Helper()
	switch x := v.(type) {
	case nil:
		return nil, false
	case fixtures.Object:
		m := make(pyjson.Object, len(x))
		for i, mem := range x {
			m[i] = pyjson.Member{Key: mem.Key, Value: fixtures.Float(tb, mem.Value)}
		}
		return m, true
	}
	return fixtures.Float(tb, v), true
}

func TestGateGoldenStates(t *testing.T) {
	got := []string{gatePassed, gateAbstained, gateUnevaluated}
	want := fixtures.Strings(t, fixtures.Load(t, "confidence_gate.json").Must("states"))
	if len(got) != len(want) {
		t.Fatalf("states %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("states %v, want %v", got, want)
		}
	}
}

func TestGateGoldenCases(t *testing.T) {
	g := fixtures.Load(t, "confidence_gate.json")
	cases := fixtures.List(t, g.Must("cases"))
	if len(cases) == 0 {
		t.Fatal("golden has no cases")
	}
	for _, c := range cases {
		c := fixtures.Obj(t, c)
		name := fixtures.Str(t, c.Must("case"))
		t.Run(name, func(t *testing.T) {
			before := fixtures.Obj(t, c.Must("before"))
			after := fixtures.Obj(t, c.Must("after"))
			var answers schema.Answers
			for _, m := range before {
				answers = append(answers, schema.Answered{ID: m.Key, Answer: goldenAnswer(t, fixtures.Obj(t, m.Value))})
			}
			preds := []schema.Prediction{{Answers: answers}}
			if th, gated := goldenThreshold(t, c.Must("min_confidence")); gated {
				gt, err := checkMinConfidence(th)
				if err != nil {
					t.Fatalf("a valid threshold refused: %v", err)
				}
				gt.apply(preds)
			}
			for _, m := range after {
				want := fixtures.Obj(t, m.Value)
				a, ok := preds[0].Answers.Get(m.Key)
				if !ok {
					t.Fatalf("no answer %q", m.Key)
				}
				var cm schema.Common
				switch x := a.(type) {
				case *schema.ChoiceAnswer:
					cm = x.Common
				case *schema.ScoreAnswer:
					cm = x.Common
				case *schema.NoulAnswer:
					cm = x.Common
				}
				_, wantLow := want.Get("low_confidence")
				if cm.LowConfidence != wantLow {
					t.Errorf("%s low_confidence = %v, want %v", m.Key, cm.LowConfidence, wantLow)
				}
				ws, wantState := want.Get("abstention")
				if !wantState {
					if cm.Abstention != "" || cm.AbstentionThreshold != nil {
						t.Errorf("%s wrote a gate field on an ungated answer", m.Key)
					}
					continue
				}
				if cm.Abstention != ws.(string) {
					t.Errorf("%s abstention = %q, want %q", m.Key, cm.Abstention, ws)
				}
				wt := fixtures.Float(t, want.Must("abstention_threshold"))
				if cm.AbstentionThreshold == nil || *cm.AbstentionThreshold != wt {
					t.Errorf("%s abstention_threshold = %v, want %v", m.Key, cm.AbstentionThreshold, wt)
				}
			}
		})
	}
}

// refusalInputs maps each recorded refusal's Python repr to the value Go checks. Python's bool,
// str and non-string key have no typed call option, so they are tested through checkMinConfidence.
var refusalInputs = map[string]any{
	"1.5":               1.5,
	"-0.1":              -0.1,
	"True":              true,
	"'0.5'":             "0.5",
	"nan":               math.NaN(),
	"{}":                pyjson.Object{},
	"{'choice:2': 2.0}": pyjson.Object{{Key: "choice:2", Value: 2.0}},
	"{1: 0.5}":          pyjson.Object{{Key: 1, Value: 0.5}},
}

func TestGateGoldenRefusals(t *testing.T) {
	refs := fixtures.List(t, fixtures.Load(t, "confidence_gate.json").Must("refusals"))
	seen := map[string]bool{}
	for _, r := range refs {
		r := fixtures.Obj(t, r)
		in := fixtures.Str(t, r.Must("input"))
		want := fixtures.Str(t, r.Must("error"))
		seen[in] = true
		v, ok := refusalInputs[in]
		if !ok {
			t.Errorf("recorded refusal %s has no Go input", in)
			continue
		}
		_, err := checkMinConfidence(v)
		if err == nil {
			t.Errorf("%s accepted", in)
			continue
		}
		if _, ok := err.(*MinConfidenceError); !ok {
			t.Errorf("%s: error type %T", in, err)
		}
		if err.Error() != want {
			t.Errorf("%s: error %q, want %q", in, err.Error(), want)
		}
	}
	for in := range refusalInputs {
		if !seen[in] {
			t.Errorf("Go input %s is not a recorded refusal", in)
		}
	}
}
