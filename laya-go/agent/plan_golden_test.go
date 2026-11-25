package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

// Comparison rules, from laya-java's PredictionGolden: structure and usage exact, numbers within
// a tolerance because Python computes the softmax in float32 on whatever CPU it ran on.
const (
	tolerance      = 2e-4 // probabilities, confidences, act_probability, noul
	scoreTolerance = 8e-4 // score, the expectation over the probabilities
	decisiveGap    = 1e-3 // the chosen label is compared when the top two differ by more
	maxDiffering   = 0.05 // of all compared numbers may differ at all
)

type tally struct {
	t                  testing.TB
	numbers, differing int
	worst              float64
	where              string
	failures           int
	ties               int
	choicesChecked     int
}

func (c *tally) failf(format string, args ...any) {
	c.failures++
	if c.failures <= 10 {
		c.t.Errorf(c.where+": "+format, args...)
	}
}

func (c *tally) number(path string, got, want json.Number, tol float64) {
	g, errG := got.Float64()
	w, errW := want.Float64()
	if errG != nil || errW != nil {
		c.failf("%s: %q vs %q", path, got, want)
		return
	}
	c.numbers++
	if got != want {
		c.differing++
	}
	d := math.Abs(g - w)
	c.worst = math.Max(c.worst, d)
	if d > tol {
		c.failf("%s: got %v want %v (off by %.2e, tolerance %.0e)", path, g, w, d, tol)
	}
}

// value compares got to want. Strings, booleans, keys and their order are exact.
func (c *tally) value(path, key string, got, want any) {
	switch w := want.(type) {
	case fixtures.Object:
		g, ok := got.(fixtures.Object)
		if !ok || !reflect.DeepEqual(g.Keys(), w.Keys()) {
			c.failf("%s: keys %v, want %v", path, got, w.Keys())
			return
		}
		for _, m := range w {
			v, _ := g.Get(m.Key)
			c.value(path+"."+m.Key, m.Key, v, m.Value)
		}
	case []any:
		g, ok := got.([]any)
		if !ok || len(g) != len(w) {
			c.failf("%s: %v, want %d items", path, got, len(w))
			return
		}
		for i := range w {
			c.value(path+"[]", key, g[i], w[i])
		}
	case json.Number:
		g, ok := got.(json.Number)
		if !ok {
			c.failf("%s: %v, want a number", path, got)
			return
		}
		tol := tolerance
		if key == "score" {
			tol = scoreTolerance
		}
		c.number(path, g, w, tol)
	default:
		if !reflect.DeepEqual(got, want) {
			c.failf("%s: %v, want %v", path, got, want)
		}
	}
}

// answer compares one answer; the chosen label is held to Python's only when it was decisive.
func (c *tally) answer(path string, got, want fixtures.Object) {
	typ, _ := want.Get("type")
	if typ == "choice" {
		probs := fixtures.Obj(c.t, want.Must("probabilities"))
		var ps []float64
		for _, m := range probs {
			ps = append(ps, fixtures.Float(c.t, m.Value))
		}
		sort.Sort(sort.Reverse(sort.Float64Slice(ps)))
		if len(ps) > 1 && ps[0]-ps[1] <= decisiveGap {
			c.ties++
			var trimmed fixtures.Object
			for _, m := range want {
				if m.Key != "choice" {
					trimmed = append(trimmed, m)
				}
			}
			var gotTrimmed fixtures.Object
			for _, m := range got {
				if m.Key != "choice" {
					gotTrimmed = append(gotTrimmed, m)
				}
			}
			c.value(path, "", gotTrimmed, trimmed)
			return
		}
		c.choicesChecked++
	}
	c.value(path, "", got, want)
}

func intsOf(tb testing.TB, v any) []int64 {
	out := []int64{}
	for _, item := range fixtures.List(tb, v) {
		out = append(out, int64(fixtures.Int(tb, item)))
	}
	return out
}

func runFromGolden(tb testing.TB, v any) run {
	o := fixtures.Obj(tb, v)
	return run{
		Rows: fixtures.Int(tb, o.Must("rows")), Length: fixtures.Int(tb, o.Must("length")),
		Markers: fixtures.Int(tb, o.Must("markers")),
		IDSums:  intsOf(tb, o.Must("id_sums")), Tokens: intsOf(tb, o.Must("tokens")),
		MarkerSums: intsOf(tb, o.Must("marker_sums")), QTypes: intsOf(tb, o.Must("qtypes")),
	}
}

func roundTrip(t *testing.T, v any) any {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	out, err := fixtures.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestPredictPlanGolden drives the agent over a fake session and holds what it feeds the session
// (the shape of every run and a checksum of every row), the order of the results, the answers and
// the usage to ONNXAgent._infer_batch on the same states, on both shipped tokenizers.
func TestPredictPlanGolden(t *testing.T) {
	root := fixtures.Checkpoints()
	if root == "" {
		t.Skip("set LAYA_CHECKPOINTS to a directory holding the english and multilingual checkpoints")
	}
	golden := fixtures.Obj(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json"), "predict_plan"))
	defs := fixtures.Obj(t, golden.Must("questions"))
	for _, name := range []string{"english", "multilingual"} {
		t.Run(name, func(t *testing.T) {
			sess := &fakeSession{}
			a, err := agent.Open(os.DirFS(filepath.Join(root, name)), sess)
			if err != nil {
				t.Fatal(err)
			}
			defer a.Close()
			cases := fixtures.List(t, fixtures.Obj(t, golden.Must(name)).Must("cases"))
			if len(cases) != 24 {
				t.Fatalf("the golden holds %d cases", len(cases))
			}
			c := &tally{t: t}
			runs := 0
			for _, raw := range cases {
				rec := fixtures.Obj(t, raw)
				c.where = fixtures.Str(t, rec.Must("name"))
				qs, err := schema.ParseQuestions(fixtures.Marshal(t, defs.Must(fixtures.Str(t, rec.Must("questions")))))
				if err != nil {
					t.Fatal(err)
				}
				var states []any
				for _, s := range fixtures.List(t, rec.Must("states")) {
					st, err := sequence.ParseState(fixtures.Marshal(t, s))
					if err != nil {
						t.Fatal(err)
					}
					states = append(states, st)
				}
				var opts []agent.CallOption
				if n := rec.Must("batch_size"); n != nil {
					opts = append(opts, agent.WithBatchSize(fixtures.Int(t, n)))
				}
				if fixtures.Bool(t, rec.Must("sort")) {
					opts = append(opts, agent.WithSortByLength(true))
				}
				if tag := rec.Must("lang"); tag != nil {
					opts = append(opts, agent.WithLanguage(fixtures.Str(t, tag)))
				}
				sess.runs = nil
				preds, err := a.PredictBatch(context.Background(), states, qs, opts...)
				if err != nil {
					t.Fatalf("%s: %v", c.where, err)
				}

				wantRuns := fixtures.List(t, rec.Must("runs"))
				gotRuns := sess.seen()
				if len(gotRuns) != len(wantRuns) {
					t.Errorf("%s: %d runs, want %d (rows %v)", c.where, len(gotRuns), len(wantRuns), rowsOf(gotRuns))
					continue
				}
				runs += len(gotRuns)
				for i := range wantRuns {
					if want := runFromGolden(t, wantRuns[i]); !reflect.DeepEqual(gotRuns[i], want) {
						t.Errorf("%s: run %d\n got %+v\nwant %+v", c.where, i, gotRuns[i], want)
					}
				}

				wantResults := fixtures.List(t, rec.Must("results"))
				if len(preds) != len(wantResults) {
					t.Fatalf("%s: %d results, want %d", c.where, len(preds), len(wantResults))
				}
				for i, p := range preds {
					got := fixtures.Obj(t, roundTrip(t, p))
					want := fixtures.Obj(t, wantResults[i])
					path := fmt.Sprintf("%s/state%d", c.where, i)
					c.value(path+".model", "model", got.Must("model"), want.Must("model"))
					if gotJSON, wantJSON := fixtures.Marshal(t, got.Must("usage")), fixtures.Marshal(t, want.Must("usage")); string(gotJSON) != string(wantJSON) {
						c.failf("%s usage\n got %s\nwant %s", path, gotJSON, wantJSON)
					}
					gotAnswers, wantAnswers := fixtures.Obj(t, got.Must("answers")), fixtures.Obj(t, want.Must("answers"))
					if !reflect.DeepEqual(gotAnswers.Keys(), fixtures.Strings(t, want.Must("answer_ids"))) {
						c.failf("%s answer ids %v, want %v", path, gotAnswers.Keys(), want.Must("answer_ids"))
						continue
					}
					for _, m := range wantAnswers {
						g, _ := gotAnswers.Get(m.Key)
						gObj, ok := g.(fixtures.Object)
						if !ok {
							c.failf("%s.%s is not an object", path, m.Key)
							continue
						}
						c.answer(path+"."+m.Key, gObj, fixtures.Obj(t, m.Value))
					}
				}
			}
			t.Logf("%s: %d runs, %d numbers compared, %d differ at all, worst deviation %.2e, %d choices checked, %d near-ties skipped",
				name, runs, c.numbers, c.differing, c.worst, c.choicesChecked, c.ties)
			if c.numbers < 2000 || runs < 80 {
				t.Errorf("the golden compared only %d numbers over %d runs", c.numbers, runs)
			}
			if float64(c.differing) > maxDiffering*float64(c.numbers) {
				t.Errorf("%d of %d numbers differ (more than %.0f%%)", c.differing, c.numbers, maxDiffering*100)
			}
		})
	}
}

func rowsOf(runs []run) (rows []int) {
	for _, r := range runs {
		rows = append(rows, r.Rows)
	}
	return rows
}
