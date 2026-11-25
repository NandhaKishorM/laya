package agent_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// scorer is a stand-in for a native session: it prefers the second option of every question.
type scorer struct{}

func (scorer) Run(_ context.Context, b infer.Batch) (infer.Output, error) {
	out := infer.Output{Logits: make([]float32, b.Rows*b.Markers), ActLogits: make([]float32, b.Rows*2)}
	for i := 0; i < b.Rows; i++ {
		out.Logits[i*b.Markers+1] = 2
	}
	return out, nil
}

func (scorer) Close() error { return nil }

// TestToyCheckpointEndToEnd runs the documented example on the toy tokenizer.
func TestToyCheckpointEndToEnd(t *testing.T) {
	a, err := agent.Open(toyCheckpoint(`{"max_len": 128, "head_max_len": 64}`), scorer{})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	qs, err := schema.NewQuestions(
		schema.Ask("intent", schema.MustQuestion(schema.ChoiceOf("What does the customer want?", "status", "refund"))),
		schema.Ask("angry", schema.MustQuestion(schema.Noul("Is the customer angry?"))),
	)
	if err != nil {
		t.Fatal(err)
	}
	preds, err := a.PredictBatch(context.Background(), []any{"I was charged twice", "Where is my order?"}, qs,
		agent.WithBatchSize(2), agent.WithSortByLength(true))
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, p := range preds {
		intent, _ := p.Answers.Get("intent")
		got = append(got, fmt.Sprint(intent.(*schema.ChoiceAnswer).Choice, " ", p.Usage.InputTokens, " ", p.Usage.Truncated))
	}
	got = append(got, fmt.Sprint(a.Config().MaxLen))
	want := []string{"refund 170 false", "refund 168 false", "128"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}
