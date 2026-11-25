package agent

import (
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

var (
	tokMu  sync.Mutex
	tokens = map[string]*tokenizer.Tokenizer{}
)

// realTokenizer loads a shipped checkpoint's tokenizer once. CI fails the run on any skip, so a
// missing checkpoint cannot pass as green.
func realTokenizer(tb testing.TB, name string) *tokenizer.Tokenizer {
	tb.Helper()
	root := fixtures.Checkpoints()
	if root == "" {
		tb.Skip("set LAYA_CHECKPOINTS to a directory holding the english and multilingual checkpoints")
	}
	tokMu.Lock()
	defer tokMu.Unlock()
	if tok, ok := tokens[name]; ok {
		return tok
	}
	tok, err := tokenizer.LoadFS(os.DirFS(root), name)
	if err != nil {
		tb.Fatalf("load the %s tokenizer: %v", name, err)
	}
	tokens[name] = tok
	return tok
}

// longCases are the recorded cases of predict_long.json for one checkpoint.
func longCases(t *testing.T, name string) (entry fixtures.Object, cases []fixtures.Object) {
	t.Helper()
	golden := fixtures.Load(t, "predict_long.json")
	entry = fixtures.Obj(t, fixtures.Field(t, fixtures.Obj(t, golden.Must("by")), name))
	for _, c := range fixtures.List(t, entry.Must("cases")) {
		cases = append(cases, fixtures.Obj(t, c))
	}
	return entry, cases
}

func optionalInt(t *testing.T, v any) int {
	t.Helper()
	if v == nil {
		return 0
	}
	return fixtures.Int(t, v)
}

func caseQuestions(t *testing.T, rec fixtures.Object) schema.Questions {
	t.Helper()
	qs, err := schema.ParseQuestions(fixtures.Marshal(t, rec.Must("questions")))
	if err != nil {
		t.Fatal(err)
	}
	return qs
}

func planOf(t *testing.T, tok *tokenizer.Tokenizer, entry, rec fixtures.Object) sequence.WindowPlan {
	t.Helper()
	qs := caseQuestions(t, rec)
	questions := make([]schema.Question, qs.Len())
	for i := range questions {
		questions[i] = qs.At(i).Question
	}
	b := sequence.Budget{MaxLen: fixtures.Int(t, entry.Must("max_len")), HeadMaxLen: fixtures.Int(t, entry.Must("head_max_len"))}
	plan, err := sequence.PlanWindows(tok, questions, b, sequence.WindowRequest{
		Window: optionalInt(t, rec.Must("window")), Stride: optionalInt(t, rec.Must("stride"))})
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

// TestPredictLongSplitGolden holds the scan's windows to the ones predict_long handed to
// predict_batch: the state tokenization, the window offsets and the per-window decode together.
func TestPredictLongSplitGolden(t *testing.T) {
	scans := 0
	for _, name := range []string{"english", "multilingual"} {
		tok := realTokenizer(t, name)
		entry, cases := longCases(t, name)
		states, splits := fixtures.Obj(t, entry.Must("states")), fixtures.Obj(t, entry.Must("splits"))
		for _, rec := range cases {
			if rec.Must("split_key") == nil {
				continue
			}
			scans++
			id := name + "/" + fixtures.Str(t, rec.Must("case"))
			t.Run(id, func(t *testing.T) {
				plan := planOf(t, tok, entry, rec)
				if want := fixtures.Int(t, rec.Must("budget")); plan.Window != want {
					t.Errorf("window = %d, want %d", plan.Window, want)
				}
				st, err := sequence.EncodeState(tok, fixtures.Str(t, fixtures.Field(t, states, fixtures.Str(t, rec.Must("state_key")))))
				if err != nil {
					t.Fatal(err)
				}
				if want := fixtures.Int(t, rec.Must("state_tokens")); len(st.IDs) != want {
					t.Fatalf("state tokens = %d, want %d", len(st.IDs), want)
				}
				texts, starts := splitWindows(tok, st.IDs, plan)
				if want := fixtures.Strings(t, fixtures.Field(t, splits, fixtures.Str(t, rec.Must("split_key")))); !reflect.DeepEqual(texts, want) {
					t.Errorf("%d windows, want %d; they differ", len(texts), len(want))
				}
				var want []int
				for _, s := range fixtures.Ints(t, rec.Must("starts")) {
					want = append(want, int(s))
				}
				if !reflect.DeepEqual(starts, want) {
					t.Errorf("starts = %v, want %v", starts, want)
				}
				// The recorder's stub calls window_budget a second time inside predict_batch, so
				// it records each warning twice; predict_long itself raises it once.
				recorded := fixtures.Strings(t, rec.Must("warnings"))
				if want := append(slices.Clone(plan.Warnings), plan.Warnings...); len(want) != len(recorded) || len(want) > 0 && !slices.Equal(want, recorded) {
					t.Errorf("warnings = %q, recorded %q", plan.Warnings, recorded)
				}
			})
		}
	}
	if scans < 20 {
		t.Errorf("only %d scanned cases", scans)
	}
}

// TestPredictLongAggregateGolden drives the aggregation with the per-window results the Python
// stub produced, where a tie or a chosen deciding window can be constructed. The answers are
// compared as bytes, and so is usage, less keys a Prediction has no field for (the stub's
// "backend"). Python's own scan mutates the winning answers it recorded in per_window, so those
// carry a window the aggregation must replace, not trust.
func TestPredictLongAggregateGolden(t *testing.T) {
	ties, windows := 0, 0
	for _, name := range []string{"english", "multilingual"} {
		_, cases := longCases(t, name)
		for _, rec := range cases {
			if rec.Must("per_window") == nil {
				continue
			}
			id := fixtures.Str(t, rec.Must("case"))
			if strings.Contains(id, "tie") {
				ties++
			}
			t.Run(name+"/"+id, func(t *testing.T) {
				var results []schema.Prediction
				for _, w := range fixtures.List(t, rec.Must("per_window")) {
					var p schema.Prediction
					if err := json.Unmarshal(fixtures.Marshal(t, w), &p); err != nil {
						t.Fatal(err)
					}
					results = append(results, p)
				}
				var starts []int
				for _, s := range fixtures.Ints(t, rec.Must("starts")) {
					starts = append(starts, int(s))
				}
				got := aggregate(caseQuestions(t, rec), results, starts,
					fixtures.Int(t, rec.Must("budget")), fixtures.Int(t, rec.Must("state_tokens")))
				gotJSON, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				windows += len(results)

				want := fixtures.Obj(t, rec.Must("result"))
				usage := fixtures.Obj(t, want.Must("usage"))
				var kept fixtures.Object
				for _, m := range usage {
					if m.Key != "backend" {
						kept = append(kept, m)
					}
				}
				wantJSON := string(fixtures.Marshal(t, fixtures.Object{
					{Key: "model", Value: want.Must("model")}, {Key: "answers", Value: want.Must("answers")}, {Key: "usage", Value: kept}}))
				if g := compact(t, string(gotJSON)); g != compact(t, wantJSON) {
					t.Errorf("result\n got %s\nwant %s", g, compact(t, wantJSON))
				}
			})
		}
	}
	if ties < 4 || windows < 40 {
		t.Errorf("the cases reach %d ties and %d windows", ties, windows)
	}
}

// compact re-encodes JSON text without whitespace, keeping object order and number spelling.
func compact(t *testing.T, s string) string {
	t.Helper()
	v, err := fixtures.Parse([]byte(s))
	if err != nil {
		t.Fatal(err)
	}
	return string(fixtures.Marshal(t, v))
}
