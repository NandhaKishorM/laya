//go:build cgo

package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

// Comparison rules, from laya-java's PredictionGolden: structure and usage exact, numbers within
// a tolerance because Python computes them on whatever CPU it ran on.
const (
	tolerance      = 2e-4 // probabilities, confidences, act_probability, noul
	scoreTolerance = 8e-4 // score, the expectation over the probabilities
	decisiveGap    = 1e-3 // the chosen label is compared when the top two differ by more
	maxDiffering   = 0.05 // of all compared numbers may differ at all
)

type tally struct {
	t                  testing.TB
	where              string
	numbers, differing int
	worst              float64
	labels, ties       int
	labelDiffs         int
	failures           int
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

func without(o fixtures.Object, key string) fixtures.Object {
	var out fixtures.Object
	for _, m := range o {
		if m.Key != key {
			out = append(out, m)
		}
	}
	return out
}

// answer compares one answer; the chosen label is held to Python's only when it was decisive.
func (c *tally) answer(path string, got, want fixtures.Object) {
	if typ, _ := want.Get("type"); typ == "choice" {
		var ps []float64
		for _, m := range fixtures.Obj(c.t, want.Must("probabilities")) {
			ps = append(ps, fixtures.Float(c.t, m.Value))
		}
		sort.Sort(sort.Reverse(sort.Float64Slice(ps)))
		if len(ps) > 1 && ps[0]-ps[1] <= decisiveGap {
			c.ties++
			c.value(path, "", without(got, "choice"), without(want, "choice"))
			return
		}
		c.labels++
		if got.Must("choice") != want.Must("choice") {
			c.labelDiffs++
		}
	}
	c.value(path, "", got, want)
}

// prediction compares one result: model and usage exactly, answers by the rules above.
func (c *tally) prediction(path string, p schema.Prediction, want fixtures.Object) {
	data, err := json.Marshal(p)
	if err != nil {
		c.t.Fatal(err)
	}
	parsed, err := fixtures.Parse(data)
	if err != nil {
		c.t.Fatal(err)
	}
	got := fixtures.Obj(c.t, parsed)
	c.value(path+".model", "model", got.Must("model"), want.Must("model"))
	if g, w := fixtures.Marshal(c.t, got.Must("usage")), fixtures.Marshal(c.t, want.Must("usage")); string(g) != string(w) {
		c.failf("%s usage\n got %s\nwant %s", path, g, w)
	}
	gotAnswers, wantAnswers := fixtures.Obj(c.t, got.Must("answers")), fixtures.Obj(c.t, want.Must("answers"))
	if !reflect.DeepEqual(gotAnswers.Keys(), wantAnswers.Keys()) {
		c.failf("%s answer ids %v, want %v", path, gotAnswers.Keys(), wantAnswers.Keys())
		return
	}
	for _, m := range wantAnswers {
		g, _ := gotAnswers.Get(m.Key)
		c.answer(path+"."+m.Key, fixtures.Obj(c.t, g), fixtures.Obj(c.t, m.Value))
	}
}

func (c *tally) report(name string) {
	c.t.Logf("%s: %d numbers compared, %d differ at all, worst deviation %.2e, %d labels checked (%d differ), %d near-ties skipped",
		name, c.numbers, c.differing, c.worst, c.labels, c.labelDiffs, c.ties)
}

func (c *tally) finish(name string, minNumbers int) {
	c.report(name)
	if c.numbers < minNumbers {
		c.t.Errorf("%s: only %d numbers were compared", name, c.numbers)
	}
	if float64(c.differing) > maxDiffering*float64(c.numbers) {
		c.t.Errorf("%s: %d of %d numbers differ (more than %.0f%%)", name, c.differing, c.numbers, maxDiffering*100)
	}
}

func openAgent(t testing.TB, p paths, s *Session) *agent.Agent {
	t.Helper()
	a, err := agent.Open(os.DirFS(filepath.Join(p.checkpoints, "multilingual")), keepOpen{s})
	if err != nil {
		t.Fatalf("agent.Open: %v", err)
	}
	t.Cleanup(func() { a.Close() })
	return a
}

type singleCase struct {
	name  string
	state any
	lang  string
	qs    schema.Questions
	want  fixtures.Object
}

func loadState(t testing.TB, v any) any {
	t.Helper()
	st, err := sequence.ParseState(fixtures.Marshal(t, v))
	if err != nil {
		t.Fatal(err)
	}
	return st
}

func loadQuestions(t testing.TB, v any) schema.Questions {
	t.Helper()
	qs, err := schema.ParseQuestions(fixtures.Marshal(t, v))
	if err != nil {
		t.Fatal(err)
	}
	return qs
}

func singles(t testing.TB) []singleCase {
	t.Helper()
	golden := fixtures.Load(t, "predict.json")
	var out []singleCase
	for _, m := range fixtures.Obj(t, golden.Must("single")) {
		rec := fixtures.Obj(t, m.Value)
		out = append(out, singleCase{
			name: m.Key, state: loadState(t, rec.Must("state")), lang: fixtures.StrOrEmpty(t, rec.Must("lang")),
			qs: loadQuestions(t, rec.Must("questions")), want: rec,
		})
	}
	return out
}

func (sc singleCase) options() []agent.CallOption {
	if sc.lang == "" {
		return nil
	}
	return []agent.CallOption{agent.WithLanguage(sc.lang)}
}

// TestPredictGolden holds the whole path (agent, collator, this session, ONNX Runtime) to the
// answers Python's ONNXAgent recorded on the same graph.
func TestPredictGolden(t *testing.T) {
	p := need(t)
	a := openAgent(t, p, sharedSession(t))
	golden := fixtures.Load(t, "predict.json")
	if cp := fixtures.Str(t, golden.Must("checkpoint")); cp != "multilingual" {
		t.Fatalf("the golden was recorded on the %s checkpoint", cp)
	}
	ctx := context.Background()

	total := &tally{t: t}
	t.Run("single", func(t *testing.T) {
		cases := singles(t)
		if len(cases) != 6 {
			t.Fatalf("%d single cases", len(cases))
		}
		c := &tally{t: t}
		for _, sc := range cases {
			c.where = sc.name
			pred, err := a.Predict(ctx, sc.state, sc.qs, sc.options()...)
			if err != nil {
				t.Fatalf("%s: %v", sc.name, err)
			}
			c.prediction(sc.name, pred, sc.want)
		}
		c.finish("single", 60)
		merge(total, c)
	})

	t.Run("batch", func(t *testing.T) {
		batches := fixtures.List(t, golden.Must("batch"))
		if len(batches) != 3 {
			t.Fatalf("%d batch configurations", len(batches))
		}
		for _, raw := range batches {
			rec := fixtures.Obj(t, raw)
			var opts []agent.CallOption
			name := "all-in-one"
			if n := rec.Must("batch_size"); n != nil {
				size := fixtures.Int(t, n)
				opts = append(opts, agent.WithBatchSize(size))
				name = "batch-" + string(rune('0'+size))
			}
			if fixtures.Bool(t, rec.Must("sort_by_length")) {
				opts = append(opts, agent.WithSortByLength(true))
				name += "-sorted"
			}
			t.Run(name, func(t *testing.T) {
				var states []any
				for _, s := range fixtures.List(t, rec.Must("states")) {
					states = append(states, loadState(t, s))
				}
				preds, err := a.PredictBatch(ctx, states, loadQuestions(t, rec.Must("questions")), opts...)
				if err != nil {
					t.Fatal(err)
				}
				want := fixtures.List(t, rec.Must("results"))
				if len(preds) != len(want) || len(want) != 9 {
					t.Fatalf("%d results, golden has %d", len(preds), len(want))
				}
				c := &tally{t: t}
				for i, pred := range preds {
					c.where = name
					c.prediction(name+"/state"+string(rune('0'+i)), pred, fixtures.Obj(t, want[i]))
				}
				c.finish(name, 80)
				merge(total, c)
			})
		}
	})
	total.report("all")
	if total.labelDiffs != 0 {
		t.Errorf("%d decisive labels differ from Python", total.labelDiffs)
	}
}

func merge(dst, src *tally) {
	dst.numbers += src.numbers
	dst.differing += src.differing
	dst.worst = math.Max(dst.worst, src.worst)
	dst.labels += src.labels
	dst.labelDiffs += src.labelDiffs
	dst.ties += src.ties
}

// TestConcurrentPredict calls one agent from eight goroutines at once; each answer must equal
// the serial one, and -race watches the Go side.
func TestConcurrentPredict(t *testing.T) {
	p := need(t)
	a := openAgent(t, p, sharedSession(t))
	cases := singles(t)
	ctx := context.Background()
	serial := make([][]byte, len(cases))
	for i, sc := range cases {
		pred, err := a.Predict(ctx, sc.state, sc.qs, sc.options()...)
		if err != nil {
			t.Fatal(err)
		}
		serial[i], _ = json.Marshal(pred)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 8*len(cases))
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for k := range cases {
				i := (k + g) % len(cases)
				pred, err := a.Predict(ctx, cases[i].state, cases[i].qs, cases[i].options()...)
				if err != nil {
					errs <- err
					continue
				}
				if got, _ := json.Marshal(pred); string(got) != string(serial[i]) {
					errs <- errors.New(cases[i].name + ": concurrent answer differs from the serial one:\n" + string(got) + "\n" + string(serial[i]))
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestPredictWithCancelledContext(t *testing.T) {
	p := need(t)
	s := sharedSession(t)
	a := openAgent(t, p, s)
	sc := singles(t)[0]
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	before := s.runs.Load()
	if _, err := a.Predict(ctx, sc.state, sc.qs); !errors.Is(err, context.Canceled) {
		t.Fatalf("Predict = %v, want context.Canceled", err)
	}
	if s.runs.Load() != before {
		t.Fatal("the model ran for a cancelled context")
	}
}

func TestRuntimeVersion(t *testing.T) {
	s := sharedSession(t)
	t.Logf("ONNX Runtime %s", s.RuntimeVersion())
	if s.RuntimeVersion() == "" {
		t.Fatal("empty RuntimeVersion")
	}
}

// TestMarkerMaskReachesTheGraph pins that marker_mask is sent as given: padded slots must come
// back different from real ones, so a session that sent all true would be caught.
func TestMarkerMaskReachesTheGraph(t *testing.T) {
	s := sharedSession(t)
	masked := makeBatch(2, 16, 4)
	for r := 0; r < masked.Rows; r++ {
		masked.MarkerMask[r*4+2], masked.MarkerMask[r*4+3] = false, false
	}
	open := makeBatch(2, 16, 4)
	a, err := s.Run(context.Background(), masked)
	if err != nil {
		t.Fatal(err)
	}
	b, err := s.Run(context.Background(), open)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("masked %v\nopen   %v", a.Logits, b.Logits)
	same := true
	for i := range a.Logits {
		same = same && a.Logits[i] == b.Logits[i]
	}
	if same {
		t.Fatal("marker_mask has no effect on the logits")
	}
	if !masked.MarkerMask[0] || masked.MarkerMask[2] {
		t.Fatal("Run modified the batch")
	}
}

// TestTemperaturesReachTheAnswers pins that the checkpoint's temperatures are applied by type and
// by option bucket. The shipped multilingual checkpoint uses 1.0 everywhere, which hides a
// decode that ignores them, so this runs the same session under edited copies of its config.
func TestTemperaturesReachTheAnswers(t *testing.T) {
	p := need(t)
	s := sharedSession(t)
	src := filepath.Join(p.checkpoints, "multilingual")
	raw, err := os.ReadFile(filepath.Join(src, "rl_agent_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	sc := singles(t)[0]
	if sc.name != "billing-mixed" {
		t.Fatalf("first case is %s", sc.name)
	}
	answers := func(edit func(cfg map[string]any)) map[string]string {
		var cfg map[string]any
		if err := json.Unmarshal(raw, &cfg); err != nil {
			t.Fatal(err)
		}
		edit(cfg)
		dir := t.TempDir()
		entries, err := os.ReadDir(src)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range entries {
			if e.Name() != "rl_agent_config.json" {
				if err := os.Symlink(filepath.Join(src, e.Name()), filepath.Join(dir, e.Name())); err != nil {
					t.Fatal(err)
				}
			}
		}
		data, _ := json.Marshal(cfg)
		if err := os.WriteFile(filepath.Join(dir, "rl_agent_config.json"), data, 0o644); err != nil {
			t.Fatal(err)
		}
		a, err := agent.Open(os.DirFS(dir), keepOpen{s})
		if err != nil {
			t.Fatal(err)
		}
		defer a.Close()
		pred, err := a.Predict(context.Background(), sc.state, sc.qs)
		if err != nil {
			t.Fatal(err)
		}
		data, _ = json.Marshal(pred)
		parsed, err := fixtures.Parse(data)
		if err != nil {
			t.Fatal(err)
		}
		out := map[string]string{}
		for _, m := range fixtures.Obj(t, fixtures.Field(t, parsed, "answers")) {
			out[m.Key] = string(fixtures.Marshal(t, m.Value))
		}
		return out
	}
	base := answers(func(map[string]any) {})
	if len(base) != 3 {
		t.Fatalf("base answers: %v", base)
	}
	ids := map[string]string{"choice": "intent", "score": "urgency", "noul": "duplicate"}
	for i, typ := range []string{"choice", "score", "noul"} {
		temps := []any{1.0, 1.0, 1.0}
		temps[i] = 4.0
		got := answers(func(c map[string]any) { c["temperature"] = temps })
		for _, other := range []string{"choice", "score", "noul"} {
			changed := got[ids[other]] != base[ids[other]]
			if changed != (other == typ) {
				t.Errorf("temperature %v: %s answer changed = %v", temps, other, changed)
			}
		}
	}
	got := answers(func(c map[string]any) { c["temperature_by_options"] = map[string]any{"choice:3-5": 4.0} })
	if got["intent"] == base["intent"] || got["urgency"] != base["urgency"] || got["duplicate"] != base["duplicate"] {
		t.Errorf("a choice:3-5 bucket temperature did not change exactly the 3-option choice: %v", got)
	}
}
