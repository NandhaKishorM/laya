//go:build cgo

package onnx

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// longDocument is a support thread of about n sentences with one localized signal in the middle,
// which is what a scan has to find and an averaged reading would lose.
func longDocument(n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		if i == n/2 {
			b.WriteString("Customer: this is the third time I ask, I am furious and I demand a refund of the duplicate charge today. ")
			continue
		}
		fmt.Fprintf(&b, "Agent: note %d, the delivery window for order %d was confirmed and the parcel left the depot on schedule. ", i, 7000+i)
	}
	return b.String()
}

// TestPredictLongThroughTheGraph runs a scan on the real graph. A state that fits is answered as
// Predict answers it. A long one is split here, independently of PredictLong, and each window is
// scored by PredictBatch; the answer and window PredictLong names must be the ones that reading
// picks, and a window cap must not move them past the tolerance a batch shape allows.
func TestPredictLongThroughTheGraph(t *testing.T) {
	p := need(t)
	a := openAgent(t, p, sharedSession(t))
	ctx := context.Background()
	qs := singles(t)[0].qs

	short := "Customer: I was charged twice for one order and want my money back."
	plain, err := a.Predict(ctx, short, qs)
	if err != nil {
		t.Fatal(err)
	}
	fit, err := a.PredictLong(ctx, short, qs)
	if err != nil {
		t.Fatal(err)
	}
	if fit.Usage.Windows != 1 {
		t.Errorf("a state that fits reported %d windows", fit.Usage.Windows)
	}
	fit.Usage.Windows = 0
	if x, y := mustJSON(t, fit.Prediction), mustJSON(t, plain); x != y {
		t.Errorf("a state that fits was answered differently:\n%s\n%s", x, y)
	}

	tok, err := tokenizer.LoadFS(os.DirFS(p.checkpoints), "multilingual")
	if err != nil {
		t.Fatal(err)
	}
	cfg := a.Config()
	questions := make([]schema.Question, qs.Len())
	for i := range questions {
		questions[i] = qs.At(i).Question
	}
	plan, err := sequence.PlanWindows(tok, questions, sequence.Budget{MaxLen: cfg.MaxLen, HeadMaxLen: cfg.HeadMaxLen}, sequence.WindowRequest{})
	if err != nil {
		t.Fatal(err)
	}
	doc := longDocument(160)
	st, err := sequence.EncodeState(tok, doc)
	if err != nil {
		t.Fatal(err)
	}
	var texts []any
	var starts []int
	for i := 0; i < len(st.IDs); i += plan.Stride {
		end := min(i+plan.Window, len(st.IDs))
		texts, starts = append(texts, tok.Decode(st.IDs[i:end], false)), append(starts, i)
		if end >= len(st.IDs) {
			break
		}
	}
	if len(texts) < 3 {
		t.Fatalf("the document made %d windows; it must exceed %d tokens by more", len(texts), plan.Window)
	}
	scored, err := a.PredictBatch(ctx, texts, qs)
	if err != nil {
		t.Fatal(err)
	}

	long, err := a.PredictLong(ctx, doc, qs)
	if err != nil {
		t.Fatal(err)
	}
	if long.Usage.Windows != len(texts) || long.Usage.OutputTokens != 0 {
		t.Fatalf("usage = %+v, want %d windows", long.Usage, len(texts))
	}
	for r, an := range long.Answers {
		best, bestKey := 0, deciding(scored[0].Answers[r].Answer)
		for j := range scored {
			if k := deciding(scored[j].Answers[r].Answer); k > bestKey {
				best, bestKey = j, k
			}
		}
		w := an.Answer.Base().Window
		if topGap(scored, r) <= 2e-4 {
			// batch size changes the floats a little, so a near tie may go either way
			if d := deciding(an.Answer) - bestKey; w == nil || d > 2e-4 || d < -2e-4 {
				t.Errorf("%s: window = %+v, deciding value off by %.2e", an.ID, w, d)
			}
			continue
		}
		if w == nil || w.Index != best || w.Count != len(texts) || w.TokenStart != starts[best] || w.TokenEnd != min(starts[best]+plan.Window, len(st.IDs)) {
			t.Errorf("%s: window = %+v, want index %d of %d", an.ID, w, best, len(texts))
			continue
		}
		if g, want := mustJSON(t, strippedWindow(an.Answer)), mustJSON(t, scored[best].Answers[r].Answer); g != want {
			t.Errorf("%s: answer\n got %s\nwant %s", an.ID, g, want)
		}
		t.Logf("%s decided by window %d of %d, %s", an.ID, w.Index, w.Count, mustJSON(t, an.Answer))
	}

	capped, err := a.PredictLong(ctx, doc, qs, agent.WithBatchSize(1))
	if err != nil {
		t.Fatal(err)
	}
	for r, an := range long.Answers {
		other := capped.Answers[r].Answer
		if an.Answer.Base().Window.Index != other.Base().Window.Index && topGap(scored, r) > 2e-4 {
			t.Errorf("%s: a window cap moved the deciding window %d -> %d", an.ID, an.Answer.Base().Window.Index, other.Base().Window.Index)
		}
		if d := deciding(an.Answer) - deciding(other); d > 2e-4 || d < -2e-4 {
			t.Errorf("%s: a window cap moved the deciding value by %.2e", an.ID, d)
		}
	}
}

// topGap is the gap between the two highest deciding values of question r across the windows.
func topGap(scored []schema.Prediction, r int) float64 {
	hi, second := math.Inf(-1), math.Inf(-1)
	for j := range scored {
		switch k := deciding(scored[j].Answers[r].Answer); {
		case k > hi:
			hi, second = k, hi
		case k > second:
			second = k
		}
	}
	return hi - second
}

func deciding(a schema.Answer) float64 {
	if n, ok := a.(*schema.NoulAnswer); ok {
		return n.Noul
	}
	return a.Base().AnswerConfidence
}

func strippedWindow(a schema.Answer) schema.Answer {
	switch x := a.(type) {
	case *schema.ChoiceAnswer:
		c := *x
		c.Window = nil
		return &c
	case *schema.ScoreAnswer:
		c := *x
		c.Window = nil
		return &c
	case *schema.NoulAnswer:
		c := *x
		c.Window = nil
		return &c
	}
	return a
}

func mustJSON(t testing.TB, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
