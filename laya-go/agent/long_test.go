package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

const longConfig = `{"max_len": 128, "head_max_len": 64}`

// longState is a state of n bytes, which the toy tokenizer reads as about n tokens.
func longState(n int) string { return strings.Repeat("abcdefghij", n/10+1)[:n] }

func TestPredictLongScansAndNamesTheWindow(t *testing.T) {
	a, s := open(t, longConfig)
	got, err := a.PredictLong(context.Background(), longState(900), questions(t))
	if err != nil {
		t.Fatal(err)
	}
	u := got.Usage
	if u.Windows < 3 || len(s.seen()) != 1 || s.seen()[0].Rows != u.Windows*3 {
		t.Fatalf("windows = %d, runs = %+v: want one run of every window for each of 3 questions", u.Windows, s.seen())
	}
	if u.OutputTokens != 0 || u.TruncatedQuestions == nil {
		t.Errorf("usage = %+v", u)
	}
	for _, an := range got.Answers {
		w := an.Answer.Base().Window
		if w == nil || w.Count != u.Windows || w.Index < 0 || w.Index >= w.Count || w.TokenEnd <= w.TokenStart || w.TokenEnd > 900 {
			t.Errorf("%s: window = %+v", an.ID, w)
		}
	}
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var back schema.Prediction
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("%v in %s", err, b)
	}
	if !reflect.DeepEqual(back.Usage.Windows, u.Windows) || back.Usage.TruncatedWindows != u.TruncatedWindows {
		t.Errorf("round trip lost the window count: %+v vs %+v", back.Usage, u)
	}
}

func TestPredictLongFittingStateIsPlainPredict(t *testing.T) {
	a, _ := open(t, longConfig)
	ctx := context.Background()
	long, err := a.PredictLong(ctx, "short state", questions(t))
	if err != nil {
		t.Fatal(err)
	}
	plain, err := a.Predict(ctx, "short state", questions(t))
	if err != nil {
		t.Fatal(err)
	}
	if long.Usage.Windows != 1 {
		t.Errorf("windows = %d, want 1", long.Usage.Windows)
	}
	long.Usage.Windows = 0
	if !reflect.DeepEqual(long.Prediction, plain) {
		t.Errorf("a state that fits one window was answered differently:\n%+v\n%+v", long.Prediction, plain)
	}
	b, _ := json.Marshal(long)
	if strings.Contains(string(b), `"window":`) || !strings.Contains(string(b), `"truncated":false`) {
		t.Errorf("single-window result = %s", b)
	}
}

func TestPredictLongBatchSizeBoundsTheRun(t *testing.T) {
	a, s := open(t, longConfig)
	got, err := a.PredictLong(context.Background(), longState(900), questions(t), agent.WithBatchSize(2), agent.WithSortByLength(true))
	if err != nil {
		t.Fatal(err)
	}
	want := (got.Usage.Windows + 1) / 2
	if runs := s.seen(); len(runs) != want || runs[0].Rows != 6 {
		t.Errorf("%d runs of %d windows, want %d runs of 2", len(runs), got.Usage.Windows, want)
	}
}

func TestPredictLongExplicitWindowAndStride(t *testing.T) {
	a, _ := open(t, longConfig)
	got, err := a.PredictLong(context.Background(), longState(400), questions(t), agent.WithWindow(50), agent.WithStride(25))
	if err != nil {
		t.Fatal(err)
	}
	if got.Usage.Windows != 15 || len(got.Warnings) != 0 {
		t.Errorf("windows = %d, warnings = %q; want 15 and none", got.Usage.Windows, got.Warnings)
	}
	// A window wider than the room is cut, and the caller is told.
	got, err = a.PredictLong(context.Background(), longState(400), questions(t), agent.WithWindow(5000))
	if err != nil || len(got.Warnings) != 1 || !strings.Contains(got.Warnings[0], "window=5000 is wider than") {
		t.Errorf("warnings = %q, %v", got.Warnings, err)
	}
	// An explicit window decides whether a state fits, not the room.
	got, err = a.PredictLong(context.Background(), longState(60), questions(t), agent.WithWindow(30))
	if err != nil || got.Usage.Windows < 2 {
		t.Errorf("a state wider than the explicit window was not scanned: windows = %d, %v", got.Usage.Windows, err)
	}
}

func TestPredictLongRefusals(t *testing.T) {
	a, s := open(t, longConfig)
	ctx := context.Background()
	if _, err := a.PredictLong(ctx, nil, questions(t)); !errors.Is(err, sequence.ErrUnsupportedState) {
		t.Errorf("nil state: %v", err)
	}
	if _, err := a.PredictLong(ctx, longState(900), questions(t), agent.WithWindow(10), agent.WithStride(500)); !errors.Is(err, sequence.ErrStrideGap) {
		t.Errorf("stride gap: %v", err)
	}
	tight, ts := open(t, `{"max_len": 8, "head_max_len": 8}`)
	if _, err := tight.PredictLong(ctx, longState(900), questions(t)); !errors.Is(err, sequence.ErrNoRoom) || len(ts.seen()) != 0 {
		t.Errorf("no room: %v", err)
	}
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := a.PredictLong(cctx, longState(900), questions(t)); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled: %v", err)
	}
	if n := len(s.seen()); n != 0 {
		t.Errorf("%d runs after refusals", n)
	}
	a.Close()
	if _, err := a.PredictLong(ctx, "x", questions(t)); !errors.Is(err, agent.ErrClosed) {
		t.Errorf("closed: %v", err)
	}
}

func TestPredictLongWithoutQuestions(t *testing.T) {
	a, s := open(t, longConfig)
	got, err := a.PredictLong(context.Background(), longState(900), schema.Questions{})
	if err != nil {
		t.Fatal(err)
	}
	if js, _ := json.Marshal(got.Usage); string(js) != fmt.Sprintf(`{"input_tokens":0,"output_tokens":0,"windows":%d}`, got.Usage.Windows) {
		t.Errorf("usage = %s", js)
	}
	if len(got.Answers) != 0 || got.Usage.Windows < 3 || len(s.seen()) != 0 {
		t.Errorf("answers = %v, windows = %d, runs = %d", got.Answers, got.Usage.Windows, len(s.seen()))
	}
}

// TestPredictLongGoldenCases runs the recorded cases through the whole path on the shipped
// checkpoints: the windows are the ones the reference scanned, they are batched as it batched
// them, and the window each answer names is the one an independent reading of the per-window
// answers picks.
func TestPredictLongGoldenCases(t *testing.T) {
	root := fixtures.Checkpoints()
	if root == "" {
		t.Skip("set LAYA_CHECKPOINTS to a directory holding the english and multilingual checkpoints")
	}
	golden := fixtures.Load(t, "predict_long.json")
	by := fixtures.Obj(t, golden.Must("by"))
	scanned := 0
	for _, name := range []string{"english", "multilingual"} {
		entry := fixtures.Obj(t, fixtures.Field(t, by, name))
		states, splits := fixtures.Obj(t, entry.Must("states")), fixtures.Obj(t, entry.Must("splits"))
		for _, c := range fixtures.List(t, entry.Must("cases")) {
			rec := fixtures.Obj(t, c)
			t.Run(name+"/"+fixtures.Str(t, rec.Must("case")), func(t *testing.T) {
				sess := &fakeSession{}
				a, err := agent.Open(os.DirFS(filepath.Join(root, name)), sess)
				if err != nil {
					t.Fatal(err)
				}
				defer a.Close()
				qs, err := schema.ParseQuestions(fixtures.Marshal(t, rec.Must("questions")))
				if err != nil {
					t.Fatal(err)
				}
				var opts []agent.CallOption
				if v := rec.Must("window"); v != nil {
					opts = append(opts, agent.WithWindow(fixtures.Int(t, v)))
				}
				if v := rec.Must("stride"); v != nil {
					opts = append(opts, agent.WithStride(fixtures.Int(t, v)))
				}
				if v := rec.Must("batch_size"); v != nil {
					opts = append(opts, agent.WithBatchSize(fixtures.Int(t, v)))
				}
				state := fixtures.Str(t, fixtures.Field(t, states, fixtures.Str(t, rec.Must("state_key"))))
				got, err := a.PredictLong(context.Background(), state, qs, opts...)
				if err != nil {
					t.Fatal(err)
				}
				if rec.Must("split_key") == nil {
					if got.Usage.Windows != 1 || len(sess.seen()) != 1 || sess.seen()[0].Rows != qs.Len() {
						t.Errorf("fits one window: windows = %d, runs = %+v", got.Usage.Windows, sess.seen())
					}
					for _, an := range got.Answers {
						if an.Answer.Base().Window != nil {
							t.Errorf("%s names a window", an.ID)
						}
					}
					return
				}
				scanned++
				texts := fixtures.Strings(t, fixtures.Field(t, splits, fixtures.Str(t, rec.Must("split_key"))))
				starts := fixtures.Ints(t, rec.Must("starts"))
				if got.Usage.Windows != len(texts) {
					t.Fatalf("windows = %d, want %d", got.Usage.Windows, len(texts))
				}
				// Batching: one run unless a cap was recorded.
				runs, per := sess.seen(), len(texts)
				if v := rec.Must("batch_cap_used"); v != nil {
					per = fixtures.Int(t, v)
				}
				if want := (len(texts) + per - 1) / per; len(runs) != want {
					t.Errorf("%d runs, want %d of up to %d windows", len(runs), want, per)
				}
				// The reference reading of the same windows, scored by the same session.
				states := make([]any, len(texts))
				for i := range texts {
					states[i] = texts[i]
				}
				ref, err := a.PredictBatch(context.Background(), states, qs)
				if err != nil {
					t.Fatal(err)
				}
				for r, an := range got.Answers {
					best, bestKey := 0, key(ref[0].Answers[r].Answer)
					for j := range ref {
						if k := key(ref[j].Answers[r].Answer); k > bestKey {
							best, bestKey = j, k
						}
					}
					w := an.Answer.Base().Window
					if w == nil || w.Index != best || w.Count != len(texts) || w.TokenStart != int(starts[best]) ||
						w.TokenEnd != min(int(starts[best])+fixtures.Int(t, rec.Must("budget")), fixtures.Int(t, rec.Must("state_tokens"))) {
						t.Errorf("%s: window = %+v, want index %d of %d", an.ID, w, best, len(texts))
					}
					plain, _ := json.Marshal(ref[best].Answers[r].Answer)
					decided := an.Answer
					again, _ := json.Marshal(withoutWindow(decided))
					if string(plain) != string(again) {
						t.Errorf("%s: answer\n got %s\nwant %s", an.ID, again, plain)
					}
				}
				want := fixtures.Strings(t, rec.Must("warnings"))
				if !slices.Equal(want, append(slices.Clone(got.Warnings), got.Warnings...)) {
					t.Errorf("warnings = %q, recorded %q", got.Warnings, want)
				}
			})
		}
	}
	if scanned < 20 {
		t.Errorf("%d scanned cases", scanned)
	}
}

func key(a schema.Answer) float64 {
	if n, ok := a.(*schema.NoulAnswer); ok {
		return n.Noul
	}
	return a.Base().AnswerConfidence
}

// withoutWindow is a copy of a that carries no window.
func withoutWindow(a schema.Answer) schema.Answer {
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

func TestPredictLongImplicitBatchCapBoundsTheRun(t *testing.T) {
	a, s := open(t, longConfig)
	got, err := a.PredictLong(context.Background(), longState(900), questions(t), agent.WithWindow(10))
	if err != nil {
		t.Fatal(err)
	}
	n := got.Usage.Windows
	capW, _ := sequence.BatchCap(n, 10, 64, 0)
	runs := s.seen()
	if capW <= 0 || capW >= n || len(runs) < 2 || runs[0].Rows != capW*3 {
		t.Errorf("windows = %d, cap = %d, runs = %+v: want runs of %d windows", n, capW, runs, capW)
	}
}
