package agent_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func calibratedConfidence(t *testing.T, payload string, opts []agent.Option, call ...agent.CallOption) float64 {
	t.Helper()
	opts = append(opts, agent.WithCalibration([]byte(payload)))
	a, err := agent.Open(toyCheckpoint(`{"max_len": 128, "head_max_len": 64}`), scorer{}, opts...)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	qs, err := schema.NewQuestions(schema.Ask("intent",
		schema.MustQuestion(schema.ChoiceOf("What does the customer want?", "status", "refund"))))
	if err != nil {
		t.Fatal(err)
	}
	p, err := a.Predict(context.Background(), "I was charged twice", qs, call...)
	if err != nil {
		t.Fatal(err)
	}
	intent, _ := p.Answers.Get("intent")
	return intent.(*schema.ChoiceAnswer).AnswerConfidence
}

func sigmoid4(x float64) float64 { return math.Round(1/(1+math.Exp(-x))*1e4) / 1e4 }

// TestCalibrationChangesConfidence: the scorer puts logits 0 and 2 on a two-option question, so
// the answer confidence is sigmoid(2/T). A payload that were ignored would leave T at 1.
func TestCalibrationChangesConfidence(t *testing.T) {
	const base = `{"temperature": [1, 1, 1]`
	for _, c := range []struct {
		name, payload string
		opts          []agent.Option
		call          []agent.CallOption
		want          float64
	}{
		{"none", base + `}`, nil, nil, sigmoid4(2)},
		{"type temperature", `{"temperature": [2, 2, 2]}`, nil, nil, sigmoid4(1)},
		{"bucket beats type", `{"temperature": [2, 2, 2], "temperature_by_options": {"choice:2": 4}}`, nil, nil, sigmoid4(0.5)},
		{"clamped low", `{"temperature": [0.01, 0.01, 0.01]}`, nil, nil, sigmoid4(4)},
		{"clamped high", `{"temperature": [99, 99, 99]}`, nil, nil, sigmoid4(0.4)},
		{"rejected falls back to 1", `{"temperature": [true, 1, 1]}`, nil, nil, sigmoid4(2)},
		{"binning", base + `, "binning_map": {"choice:2": {"bins": 2, "values": [0.2, 0.3]}}}`, nil, nil, 0.3},
		{"binning of another bucket", base + `, "binning_map": {"choice:3-5": {"bins": 1, "values": [0.2]}}}`, nil, nil, sigmoid4(2)},
		{"empty binning", base + `, "binning_map": {}}`, nil, nil, sigmoid4(2)},
		{"language override ignores binning",
			base + `, "binning_map": {"choice:2": {"bins": 1, "values": [0.2]}}}`,
			[]agent.Option{agent.WithLangTemperatures(map[string]map[string]any{"fr": {"temperature": []any{2, 2, 2}}})},
			[]agent.CallOption{agent.WithLanguage("fr-CA")}, sigmoid4(1)},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := calibratedConfidence(t, c.payload, c.opts, c.call...); got != c.want {
				t.Errorf("confidence %v, want %v", got, c.want)
			}
		})
	}
}

func TestCalibrationWarnings(t *testing.T) {
	a, err := agent.Open(toyCheckpoint(`{"max_len": 128}`), scorer{},
		agent.WithCalibration([]byte(`{"version": 2, "temperature": [9, 1, 1], "model_id_or_path": "org/m"}`)))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	w := a.Config().Warnings
	if len(w) != 2 || !strings.HasPrefix(w[0], "calibration was fitted for model_id_or_path='org/m'") ||
		!strings.Contains(w[0], "Checkpoint config identity differs.") || !strings.Contains(w[1], "temperature[0]=9 -> 5") {
		t.Errorf("warnings %q", w)
	}
}

// TestCalibrationPayloadRoundTrip: a payload written for a checkpoint's own config reads back with no warning.
func TestCalibrationPayloadRoundTrip(t *testing.T) {
	p := agent.CalibrationPayload{Temperature: [3]float64{2, 2, 2}, Config: map[string]any{"max_len": 128, "head_max_len": 64,
		"temperature": []any{9, 9, 9}}}
	body, err := p.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	a, err := agent.Open(toyCheckpoint(`{"max_len": 128, "head_max_len": 64, "temperature": [3, 3, 3]}`), scorer{}, agent.WithCalibration(body))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if w := a.Config().Warnings; len(w) != 0 {
		t.Errorf("warnings %q", w)
	}
}
