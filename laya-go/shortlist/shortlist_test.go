package shortlist

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/goldens"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// stubEmbed is the golden's embedder: FNV-1a of "text:component" mapped into [-1, 1).
func stubEmbed(_ context.Context, texts []string) ([][]float64, error) {
	rows := make([][]float64, len(texts))
	for i, text := range texts {
		for c := 0; c < 8; c++ {
			h := uint64(0xCBF29CE484222325)
			for _, b := range []byte(text + ":" + string(rune('0'+c))) {
				h = (h ^ uint64(b)) * 0x100000001B3
			}
			rows[i] = append(rows[i], float64(h%2000)/1000.0-1.0)
		}
	}
	return rows, nil
}

// tagged reads a number that the golden spells "nan", "inf" or "-inf" when JSON cannot.
func tagged(t *testing.T, v any) float64 {
	t.Helper()
	if s, ok := v.(string); ok {
		switch s {
		case "nan":
			return math.NaN()
		case "inf":
			return math.Inf(1)
		case "-inf":
			return math.Inf(-1)
		}
		t.Fatalf("unknown tag %q", s)
	}
	return fixtures.Float(t, v)
}

func floats(t *testing.T, v any) []float64 {
	t.Helper()
	var out []float64
	for _, x := range fixtures.List(t, v) {
		out = append(out, tagged(t, x))
	}
	return out
}

func sameScore(got, want float64) bool {
	if math.IsNaN(want) || math.IsInf(want, 0) {
		return math.IsNaN(got) == math.IsNaN(want) && (!math.IsInf(want, 0) || got == want)
	}
	return math.Abs(got-want) < 1e-9
}

func state(t *testing.T, v any) any {
	t.Helper()
	switch fixtures.Str(t, fixtures.Field(t, v, "kind")) {
	case "string":
		return fixtures.Str(t, fixtures.Field(t, v, "value"))
	case "json":
		return goldens.Value(fixtures.Field(t, v, "value"))
	}
	t.Fatalf("state kind %v", v)
	return nil
}

// choiceOf builds the choice question of a golden's criteria: a dict of label -> description or
// null, or a list of labels.
func choiceOf(t *testing.T, ins string, criteria any) schema.Question {
	t.Helper()
	var opts []schema.Option
	switch c := criteria.(type) {
	case fixtures.Object:
		for _, m := range c {
			var crit any
			if s, ok := m.Value.(string); ok {
				crit = s
			}
			opts = append(opts, schema.Option{Label: m.Key, Criterion: crit})
		}
	case []any:
		for _, l := range c {
			opts = append(opts, schema.Option{Label: fixtures.Str(t, l)})
		}
	default:
		t.Fatalf("criteria %T", criteria)
	}
	q, err := schema.Choice(ins, opts...)
	if err != nil {
		t.Fatal(err)
	}
	return q
}

func labelStrings(ls []schema.Label) []string {
	out := make([]string, len(ls))
	for i, l := range ls {
		out[i] = l.String()
	}
	return out
}

func TestStubEmbedMatchesPython(t *testing.T) {
	golden := fixtures.Load(t, "shortlist.json")
	probe := fixtures.Obj(t, golden.Must("embed_probe"))
	if len(probe) != 4 {
		t.Fatalf("%d probes", len(probe))
	}
	for _, m := range probe {
		got, _ := stubEmbed(context.Background(), []string{m.Key})
		if want := floats(t, m.Value); !reflect.DeepEqual(got[0], want) {
			t.Errorf("%q: %v, want %v", m.Key, got[0], want)
		}
	}
}

func TestRankingCasesMatchPython(t *testing.T) {
	golden := fixtures.Load(t, "shortlist.json")
	cases := fixtures.List(t, golden.Must("cases"))
	passthrough := 0
	for _, raw := range cases {
		c := fixtures.Obj(t, raw)
		name := fixtures.Str(t, c.Must("name"))
		t.Run(name, func(t *testing.T) {
			st := state(t, c.Must("state"))
			ins := fixtures.StrOrEmpty(t, c.Must("instructions"))
			q := choiceOf(t, "placeholder", c.Must("criteria"))
			k := fixtures.Int(t, c.Must("k"))
			calls := 0
			embed := func(ctx context.Context, texts []string) ([][]float64, error) {
				calls++
				return stubEmbed(ctx, texts)
			}
			got, err := RankWithInstructions(context.Background(), st, q, embed, k, ins)
			if err != nil {
				t.Fatal(err)
			}
			if query, _ := QueryText(st, ins); query != fixtures.Str(t, c.Must("query_text")) {
				t.Errorf("query text %q, want %q", query, fixtures.Str(t, c.Must("query_text")))
			}
			if !reflect.DeepEqual(q.RenderedOptions(), fixtures.Strings(t, c.Must("option_texts"))) {
				t.Errorf("option texts %q", q.RenderedOptions())
			}
			if want := fixtures.Strings(t, c.Must("labels")); !reflect.DeepEqual(labelStrings(got.Labels), want) {
				t.Errorf("labels %v, want %v", labelStrings(got.Labels), want)
			}
			if got.N != fixtures.Int(t, c.Must("n")) || got.K != k {
				t.Errorf("n=%d k=%d", got.N, got.K)
			}
			if got.Passthrough != fixtures.Bool(t, c.Must("passthrough")) {
				t.Fatalf("passthrough = %v", got.Passthrough)
			}
			if got.Passthrough {
				passthrough++
				for i, ix := range got.Indices {
					if ix != i {
						t.Errorf("passthrough indices %v, want option order", got.Indices)
						break
					}
				}
				if calls != 0 || got.Scores != nil {
					t.Errorf("passthrough called the embedder %d times, scores %v", calls, got.Scores)
				}
				return
			}
			if calls != 1 {
				t.Errorf("embedder called %d times, want 1", calls)
			}
			want := floats(t, c.Must("scores"))
			if len(got.Scores) != len(want) {
				t.Fatalf("scores %v, want %v", got.Scores, want)
			}
			for i := range want {
				if !sameScore(got.Scores[i], want[i]) {
					t.Errorf("score %d: %v, want %v", i, got.Scores[i], want[i])
				}
			}
			sub, err := q.Subset(got.Indices)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(sub.Labels(), labelStrings(got.Labels)) {
				t.Errorf("subset labels %v", sub.Labels())
			}
			if wantCrit := fixtures.Field(t, raw, "subset_criteria"); len(sub.Labels()) != lenOf(t, wantCrit) {
				t.Errorf("subset has %d options, Python %d", len(sub.Labels()), lenOf(t, wantCrit))
			}
		})
	}
	if len(cases) < 15 || passthrough < 3 {
		t.Errorf("%d cases, %d passthrough: the golden stopped covering both", len(cases), passthrough)
	}
}

func lenOf(t *testing.T, v any) int {
	t.Helper()
	switch x := v.(type) {
	case fixtures.Object:
		return len(x)
	case []any:
		return len(x)
	}
	t.Fatalf("%T", v)
	return 0
}

func TestSubsetKeepsOptionsAndDropsOrder(t *testing.T) {
	q, err := schema.Choice("Which?",
		schema.Option{Label: "a", Criterion: "alpha"}, schema.Option{Label: "b"},
		schema.Option{Label: "c", Criterion: map[string]any{"x": 1}})
	if err != nil {
		t.Fatal(err)
	}
	if q, err = q.WithOptionOrder([]int{2, 0, 1}); err != nil {
		t.Fatal(err)
	}
	sub, err := q.Subset([]int{2, 0})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := sub.RenderedOptions(), []string{`c: {"x": 1}`, "a: alpha"}; !reflect.DeepEqual(got, want) {
		t.Errorf("options %q, want %q", got, want)
	}
	if sub.OptionOrder() != nil || sub.Instructions() != "Which?" {
		t.Errorf("order %v, instructions %q", sub.OptionOrder(), sub.Instructions())
	}
	if q.OptionOrder() == nil {
		t.Error("Subset changed its receiver")
	}
	for name, idx := range map[string][]int{"empty": nil, "range": {3}, "negative": {-1}, "repeat": {1, 1}} {
		if _, err := q.Subset(idx); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	noul, _ := schema.Noul("Is it?")
	if _, err := noul.Subset([]int{0}); err == nil {
		t.Error("a noul question was subset")
	}
}

func TestNaNRankingMatchesPython(t *testing.T) {
	c := fixtures.Obj(t, fixtures.Load(t, "shortlist.json").Must("nan_ranking"))
	embed := func(_ context.Context, texts []string) ([][]float64, error) {
		rows := make([][]float64, len(texts))
		for i, text := range texts {
			head, _, hasColon := strings.Cut(text, ":")
			switch {
			case strings.HasPrefix(text, "a:") || text == "a" || strings.HasSuffix(text, " a"):
				rows[i] = []float64{1e200, 1e200}
			case hasColon && head == "b":
				rows[i] = []float64{1, 0}
			case hasColon && head == "c":
				rows[i] = []float64{0, 1}
			default:
				rows[i] = []float64{1e200, 1e200}
			}
		}
		return rows, nil
	}
	q := choiceOf(t, "x", c.Must("criteria"))
	got, err := RankWithInstructions(context.Background(), "state", q, embed, fixtures.Int(t, c.Must("k")), "")
	if err != nil {
		t.Fatal(err)
	}
	if want := fixtures.Strings(t, c.Must("labels")); !reflect.DeepEqual(labelStrings(got.Labels), want) {
		t.Errorf("labels %v, want %v: a NaN score must rank last", labelStrings(got.Labels), want)
	}
	if want := floats(t, c.Must("scores")); !reflect.DeepEqual(got.Scores, want) {
		t.Errorf("scores %v, want %v", got.Scores, want)
	}
	// The same rows through cosine: the first option is NaN, the others 0.
	rows, _ := embed(context.Background(), append([]string{"state"}, q.RenderedOptions()...))
	sims := cosine(rows[0], rows[1:])
	for i, want := range floats(t, c.Must("sims")) {
		if !sameScore(sims[i], want) {
			t.Errorf("sim %d: %v, want %v", i, sims[i], want)
		}
	}
}

func TestCosineMatchesPython(t *testing.T) {
	cases := fixtures.List(t, fixtures.Load(t, "shortlist.json").Must("cosine"))
	exact := map[string]float64{"underflowing-query": 0, "self-cosine-over-one": 1, "needs-clipping": 0}
	for _, raw := range cases {
		c := fixtures.Obj(t, raw)
		name := fixtures.Str(t, c.Must("name"))
		t.Run(name, func(t *testing.T) {
			var docs [][]float64
			for _, d := range fixtures.List(t, c.Must("docs")) {
				docs = append(docs, floats(t, d))
			}
			got := cosine(floats(t, c.Must("query")), docs)
			want := floats(t, c.Must("sims"))
			if len(got) != len(want) {
				t.Fatalf("%v, want %v", got, want)
			}
			for i := range want {
				if !sameScore(got[i], want[i]) {
					t.Errorf("%d: %v, want %v", i, got[i], want[i])
				}
				if e, ok := exact[name]; ok && got[i] != e {
					t.Errorf("%d: %v, want exactly %v", i, got[i], e)
				}
			}
		})
	}
	if len(cases) != 10 {
		t.Errorf("%d cosine cases", len(cases))
	}
}

// v is a vector whose self-cosine computes to 1.0000000000000002 before the clip, which no golden
// records because it moves with the summation order; the clip is the reference's contract.
func TestCosineIsClipped(t *testing.T) {
	v := []float64{34, 15.285714285714286, 70.85714285714286}
	neg := []float64{-34, -15.285714285714286, -70.85714285714286}
	if got := cosine(v, [][]float64{v, neg}); got[0] != 1 || got[1] != -1 {
		t.Errorf("self cosine %v, want exactly 1 and -1", got)
	}
}

// The products of v and q are inexact and cancel exactly when each is rounded to float64 first,
// as numpy does; a fused multiply-add (arm64) leaves a nonzero score. Python's _cosine returns exactly 0.0 here.
func TestCosineBlocksFusedMultiplyAdd(t *testing.T) {
	q := []float64{1.3333333333333333, 1, 0, 0, 2.333333333333333}
	v := []float64{-1.6666666666666664e+38, -1.6666666666666664e+38, 3.333333333333333e+38, -5e+38, 1.6666666666666664e+38}
	if got := cosine(q, [][]float64{v}); got[0] != 0 {
		t.Errorf("score %v, want exactly 0", got[0])
	}
}

func TestNonFiniteRowsBecomeZero(t *testing.T) {
	cases := fixtures.List(t, fixtures.Load(t, "shortlist.json").Must("nonfinite"))
	for _, raw := range cases {
		c := fixtures.Obj(t, raw)
		var rows [][]float64
		for _, r := range fixtures.List(t, c.Must("rows")) {
			rows = append(rows, floats(t, r))
		}
		got, err := checkRows(rows, len(rows))
		if err != nil {
			t.Fatal(err)
		}
		var want [][]float64
		for _, r := range fixtures.List(t, c.Must("cleaned")) {
			want = append(want, floats(t, r))
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: %v, want %v", fixtures.Str(t, c.Must("name")), got, want)
		}
	}
	if len(cases) != 4 {
		t.Errorf("%d cases", len(cases))
	}
}

// refusals maps each refusal Python records to what this port does with the same input. A name
// that is not here fails the test, so a refusal added to the reference is accounted for.
func TestRefusalsMatchPython(t *testing.T) {
	small := choiceOf(t, "x", fixtures.Object{{Key: "refund", Value: "money back"}, {Key: "technical", Value: "a bug"}, {Key: "billing", Value: "an invoice"}})
	ctx := context.Background()
	rank := func(embed Embedder, k int) error { _, err := Rank(ctx, "s", small, embed, k); return err }
	rows := func(r [][]float64) Embedder {
		return func(context.Context, []string) ([][]float64, error) { return r, nil }
	}
	build := func(opts ...schema.Option) error { _, err := schema.Choice("x", opts...); return err }
	invalid := func(err error) bool { return errors.Is(err, ErrInvalidArgument) }
	embedding := func(err error) bool { return errors.Is(err, ErrEmbedding) }
	refusedAtBuild := func(err error) bool { return errors.Is(err, schema.ErrInvalidQuestion) }
	check := map[string]struct {
		err error
		is  func(error) bool
	}{
		"k-zero":                   {rank(stubEmbed, 0), invalid},
		"k-negative":               {rank(stubEmbed, -1), invalid},
		"k-bool":                   {errors.New("n/a: k is an int"), nil},
		"criteria-empty-dict":      {build(), refusedAtBuild},
		"criteria-empty-list":      {build(), refusedAtBuild},
		"criteria-wrong-type":      {errors.New("n/a: a Question is typed"), nil},
		"criteria-duplicate-label": {build(schema.Option{Label: "a"}, schema.Option{Label: "a"}), refusedAtBuild},
		"embed-not-callable":       {rank(nil, 2), invalid},
		"embed-wrong-rows":         {rank(rows([][]float64{{1}}), 2), embedding},
		"embed-one-dimensional":    {rank(rows([][]float64{{}, {}, {}, {}}), 2), embedding},
		"questions-not-dict":       {errors.New("n/a: Questions is typed"), nil},
		"choice-without-criteria":  {build(), refusedAtBuild},
	}
	golden := fixtures.List(t, fixtures.Load(t, "shortlist.json").Must("refusals"))
	for _, raw := range golden {
		name := fixtures.Str(t, fixtures.Field(t, raw, "name"))
		c, ok := check[name]
		if !ok {
			t.Errorf("Python refuses %q and this test does not account for it", name)
			continue
		}
		if c.is == nil {
			continue
		}
		if c.err == nil || !c.is(c.err) {
			t.Errorf("%s: err = %v", name, c.err)
		}
	}
	if len(golden) != len(check) {
		t.Errorf("%d refusals recorded, %d accounted for", len(golden), len(check))
	}
	// An embedder that fails is reported, wrapped, and a cancelled context is not called.
	boom := errors.New("boom")
	if err := rank(func(context.Context, []string) ([][]float64, error) { return nil, boom }, 2); !errors.Is(err, boom) {
		t.Errorf("embedder error lost: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	called := false
	if _, err := Rank(cancelled, "s", small, func(context.Context, []string) ([][]float64, error) { called = true; return nil, nil }, 2); !errors.Is(err, context.Canceled) || called {
		t.Errorf("cancelled: err=%v called=%v", err, called)
	}
	// A passthrough needs no embedder at all.
	if err := rank(nil, 3); err != nil {
		t.Errorf("passthrough with a nil embedder: %v", err)
	}
	if _, err := Rank(ctx, "s", mustNoul(t), stubEmbed, 1); !invalid(err) {
		t.Errorf("a noul question was ranked: %v", err)
	}
	if _, err := Predict(ctx, nil, "s", schema.Questions{}, stubEmbed, 1); !invalid(err) {
		t.Errorf("nil predictor: %v", err)
	}
}

func mustNoul(t *testing.T) schema.Question {
	t.Helper()
	q, err := schema.Noul("Is it?")
	if err != nil {
		t.Fatal(err)
	}
	return q
}

type recorder struct {
	seen  schema.Questions
	opts  int
	calls int
	err   error
}

func (r *recorder) Predict(_ context.Context, _ any, qs schema.Questions, opts ...agent.CallOption) (schema.Prediction, error) {
	r.seen, r.opts, r.calls = qs, len(opts), r.calls+1
	return schema.Prediction{}, r.err
}

func TestPredictMatchesPython(t *testing.T) {
	golden := fixtures.Load(t, "shortlist.json")
	st := "My card was charged twice for the same order last Tuesday."
	predicts := fixtures.List(t, golden.Must("predicts"))
	for _, raw := range predicts {
		c := fixtures.Obj(t, raw)
		name := fixtures.Str(t, c.Must("name"))
		t.Run(name, func(t *testing.T) {
			qs, err := schema.ParseQuestions(fixtures.Marshal(t, c.Must("questions")))
			if err != nil {
				t.Fatal(err)
			}
			before, _ := qs.MarshalJSON()
			rec := &recorder{}
			k := fixtures.Int(t, c.Must("k"))
			out, err := Predict(context.Background(), rec, st, qs, stubEmbed, k, agent.WithLanguage("en"))
			if err != nil {
				t.Fatal(err)
			}
			if rec.calls != 1 || rec.opts != 1 {
				t.Errorf("predict called %d times with %d options", rec.calls, rec.opts)
			}
			if after, _ := qs.MarshalJSON(); string(before) != string(after) {
				t.Error("the caller's questions changed")
			}
			asked := fixtures.Obj(t, c.Must("asked"))
			if rec.seen.Len() != len(asked) {
				t.Fatalf("asked %v, Python %v", rec.seen.IDs(), asked.Keys())
			}
			for i, id := range asked.Keys() {
				got := rec.seen.At(i)
				if got.ID != id {
					t.Fatalf("question %d is %q, Python %q", i, got.ID, id)
				}
				want := fixtures.Obj(t, asked.Must(id))
				if fixtures.Str(t, want.Must("type")) != string(got.Question.Type()) {
					t.Errorf("%s: type %s", id, got.Question.Type())
				}
				if fixtures.Str(t, want.Must("instructions")) != got.Question.Instructions() {
					t.Errorf("%s: instructions %q", id, got.Question.Instructions())
				}
				if got.Question.Type() != schema.TypeChoice {
					orig, _ := qs.Get(id)
					a, _ := orig.MarshalJSON()
					b, _ := got.Question.MarshalJSON()
					if string(a) != string(b) {
						t.Errorf("%s: a non-choice question changed", id)
					}
					continue
				}
				orig, _ := qs.Get(id)
				wantQ := choiceOf(t, got.Question.Instructions(), want.Must("criteria"))
				if !reflect.DeepEqual(got.Question.Labels(), wantQ.Labels()) {
					t.Errorf("%s: labels %v, Python %v", id, got.Question.Labels(), wantQ.Labels())
				}
				if !reflect.DeepEqual(got.Question.RenderedOptions(), wantQ.RenderedOptions()) {
					t.Errorf("%s: options %q, Python %q", id, got.Question.RenderedOptions(), wantQ.RenderedOptions())
				}
				if len(got.Question.Labels()) > orig.NumOptions() {
					t.Errorf("%s: grew", id)
				}
			}
			meta := fixtures.Obj(t, c.Must("shortlist"))
			if len(out.Rankings) != len(meta) {
				t.Fatalf("%d rankings, Python %d", len(out.Rankings), len(meta))
			}
			for i, id := range meta.Keys() {
				m := fixtures.Obj(t, meta.Must(id))
				r := out.Rankings[i]
				if r.ID != id || r.K != fixtures.Int(t, m.Must("k")) || r.N != fixtures.Int(t, m.Must("n")) ||
					r.Passthrough != fixtures.Bool(t, m.Must("passthrough")) {
					t.Errorf("%s: ranking %+v", id, r)
				}
				if !reflect.DeepEqual(labelStrings(r.Labels), fixtures.Strings(t, m.Must("labels"))) {
					t.Errorf("%s: labels %v", id, labelStrings(r.Labels))
				}
				if isNil := m.Must("scores") == nil; isNil != (r.Scores == nil) {
					t.Errorf("%s: scores %v, Python %v", id, r.Scores, m.Must("scores"))
				} else if !isNil {
					for j, w := range floats(t, m.Must("scores")) {
						if !sameScore(r.Scores[j], w) {
							t.Errorf("%s: score %d %v, want %v", id, j, r.Scores[j], w)
						}
					}
				}
				if len(r.Labels) != min(r.K, r.N) {
					t.Errorf("%s: %d labels for k=%d n=%d", id, len(r.Labels), r.K, r.N)
				}
			}
			if !fixtures.Bool(t, c.Must("caller_questions_unmutated")) {
				t.Error("Python mutated the caller's questions")
			}
		})
	}
	if len(predicts) != 4 {
		t.Errorf("%d predict cases", len(predicts))
	}
}

func TestOptionOrderDroppedLikePython(t *testing.T) {
	cases := fixtures.List(t, fixtures.Load(t, "go_extra.json").Must("shortlist_order").(fixtures.Object).Must("cases"))
	embed := func(_ context.Context, texts []string) ([][]float64, error) {
		rows := make([][]float64, len(texts))
		for i := range texts {
			if i == 0 {
				rows[i] = []float64{1, 0}
			} else {
				rows[i] = []float64{1 / float64(i+1), float64(i % 3)}
			}
		}
		return rows, nil
	}
	warned := 0
	for _, raw := range cases {
		c := fixtures.Obj(t, raw)
		t.Run(fixtures.Str(t, c.Must("name")), func(t *testing.T) {
			qs, err := schema.ParseQuestions(fixtures.Marshal(t, c.Must("questions")))
			if err != nil {
				t.Fatal(err)
			}
			rec := &recorder{}
			out, err := Predict(context.Background(), rec, "state", qs, embed, fixtures.Int(t, c.Must("k")))
			if err != nil {
				t.Fatal(err)
			}
			if got, want := out.Warnings, fixtures.Strings(t, c.Must("warnings")); !(len(got) == 0 && len(want) == 0) && !reflect.DeepEqual(got, want) {
				t.Errorf("warnings\n got %q\nwant %q", got, want)
			}
			warned += len(out.Warnings)
			asked := fixtures.Obj(t, c.Must("asked"))
			for i, id := range asked.Keys() {
				var want []int32
				if o, ok := fixtures.Obj(t, asked.Must(id)).Get("option_order"); ok {
					want = fixtures.Ints(t, o)
				}
				var got []int32
				for _, v := range rec.seen.At(i).Question.OptionOrder() {
					got = append(got, int32(v))
				}
				if !reflect.DeepEqual(got, want) {
					t.Errorf("%s: option order %v, Python %v", id, got, want)
				}
			}
			for _, id := range fixtures.Obj(t, c.Must("labels")).Keys() {
				r, ok := out.Ranking(id)
				if !ok {
					t.Fatalf("no ranking for %s", id)
				}
				if want := fixtures.Strings(t, fixtures.Obj(t, c.Must("labels")).Must(id)); !reflect.DeepEqual(labelStrings(r.Labels), want) {
					t.Errorf("%s: labels %v, Python %v", id, labelStrings(r.Labels), want)
				}
			}
		})
	}
	if len(cases) != 5 || warned != 3 {
		t.Errorf("%d cases, %d warnings; the golden changed", len(cases), warned)
	}
}

func TestPredictStopsOnErrors(t *testing.T) {
	ctx := context.Background()
	big := choiceOf(t, "x", fixtures.Object{{Key: "a"}, {Key: "b"}, {Key: "c"}})
	qs, _ := schema.NewQuestions(schema.Ask("q", big))
	rec := &recorder{}
	if _, err := Predict(ctx, rec, "s", qs, stubEmbed, 0); !errors.Is(err, ErrInvalidArgument) || rec.calls != 0 {
		t.Errorf("k=0: %v, %d calls", err, rec.calls)
	}
	bad := func(context.Context, []string) ([][]float64, error) { return [][]float64{{1}}, nil }
	if _, err := Predict(ctx, rec, "s", qs, bad, 1); !errors.Is(err, ErrEmbedding) || rec.calls != 0 || !strings.Contains(err.Error(), "'q'") {
		t.Errorf("bad embedder: %v, %d calls", err, rec.calls)
	}
	rec.err = errors.New("model down")
	if _, err := Predict(ctx, rec, "s", qs, stubEmbed, 3); !errors.Is(err, rec.err) {
		t.Errorf("predict error lost: %v", err)
	}
}

func TestChoiceReturnsLabels(t *testing.T) {
	q := choiceOf(t, "x", fixtures.Object{{Key: "a"}, {Key: "b"}, {Key: "c"}})
	got, err := Choice(context.Background(), "s", q, stubEmbed, 2)
	if err != nil || len(got) != 2 {
		t.Fatalf("%v, %v", got, err)
	}
	r, _ := Rank(context.Background(), "s", q, stubEmbed, 2)
	if !reflect.DeepEqual(got, r.Labels) {
		t.Errorf("Choice %v, Rank %v", got, r.Labels)
	}
}
