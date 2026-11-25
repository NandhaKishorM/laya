package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// run is what a fake session saw of one batch: its shape and, per row, checksums that name the
// row. Row order inside a run and run order are what the chunking decides.
type run struct {
	Rows, Length, Markers int
	IDSums, Tokens        []int64
	MarkerSums, QTypes    []int64
}

// fakeSession stands in for onnxruntime. Its outputs are a function of the integers it is fed
// (exact in float32), which scripts/gen_go_fixtures.py computes the same way, so what the agent
// feeds it decides the answers.
type fakeSession struct {
	mu     sync.Mutex
	runs   []run
	pads   [][]int64 // per run, the distinct ids under a zero mask
	closes int
	// before, when set, runs at the start of each Run with the number of runs so far.
	before func(n int) error
	// shape, when set, edits the output before it is returned.
	shape func(*infer.Output)
	// onCtx, when set, sees the context each Run was given.
	onCtx func(context.Context)
}

func (s *fakeSession) Run(ctx context.Context, b infer.Batch) (infer.Output, error) {
	if s.onCtx != nil {
		s.onCtx(ctx)
	}
	if err := b.Validate(); err != nil {
		return infer.Output{}, err
	}
	r := run{Rows: b.Rows, Length: b.Length, Markers: b.Markers}
	var pads []int64
	out := infer.Output{Logits: make([]float32, b.Rows*b.Markers), ActLogits: make([]float32, b.Rows*2)}
	for i := 0; i < b.Rows; i++ {
		var sum, tokens, markers int64
		for j := 0; j < b.Length; j++ {
			if b.AttentionMask[i*b.Length+j] == 1 {
				sum += b.InputIDs[i*b.Length+j]
				tokens++
			} else if id := b.InputIDs[i*b.Length+j]; !slices.Contains(pads, id) {
				pads = append(pads, id)
			}
		}
		for j := 0; j < b.Markers; j++ {
			if b.MarkerMask[i*b.Markers+j] {
				markers += b.MarkerPos[i*b.Markers+j]
			}
			out.Logits[i*b.Markers+j] = float32((sum+31*int64(j))%97-48) / 8
		}
		for c := 0; c < 2; c++ {
			out.ActLogits[i*2+c] = float32((sum+17*int64(c))%97-48) / 8
		}
		r.IDSums = append(r.IDSums, sum)
		r.Tokens = append(r.Tokens, tokens)
		r.MarkerSums = append(r.MarkerSums, markers)
		r.QTypes = append(r.QTypes, b.QType[i])
	}
	s.mu.Lock()
	n := len(s.runs)
	s.runs = append(s.runs, r)
	s.pads = append(s.pads, pads)
	before, shape := s.before, s.shape
	s.mu.Unlock()
	if before != nil {
		if err := before(n); err != nil {
			return infer.Output{}, err
		}
	}
	if shape != nil {
		shape(&out)
	}
	return out, nil
}

func (s *fakeSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closes++
	return nil
}

func (s *fakeSession) seen() []run {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]run(nil), s.runs...)
}

// toyCheckpoint is a checkpoint directory holding the toy tokenizer and the given config.
func toyCheckpoint(config string) fstest.MapFS {
	spec, cfg := fixtures.ToyTokenizer()
	return fstest.MapFS{
		"rl_agent_config.json":            {Data: []byte(config)},
		"tokenizer/tokenizer.json":        {Data: spec},
		"tokenizer/tokenizer_config.json": {Data: cfg},
	}
}

func open(t *testing.T, config string) (*agent.Agent, *fakeSession) {
	t.Helper()
	s := &fakeSession{}
	a, err := agent.Open(toyCheckpoint(config), s)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { a.Close() })
	return a, s
}

func questions(t *testing.T) schema.Questions {
	t.Helper()
	qs, err := schema.NewQuestions(
		schema.Ask("intent", schema.MustQuestion(schema.ChoiceOf("What does the customer want to do?", "refund", "status", "other"))),
		schema.Ask("urgency", schema.MustQuestion(schema.Score("How urgent?", "low", "medium", "high"))),
		schema.Ask("angry", schema.MustQuestion(schema.Noul("Angry?"))),
	)
	if err != nil {
		t.Fatal(err)
	}
	return qs
}

func jsonOf(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestLoadConfig(t *testing.T) {
	cases := []struct {
		name     string
		config   string
		want     agent.Config
		warnings []string // substrings of the one warning, if any
	}{
		{"defaults", `{}`, agent.Config{MaxLen: 512, HeadMaxLen: 192, OptionLayout: "sequential"}, nil},
		{"multilingual", `{"max_len": 1024, "head_max_len": 256, "temperature": [1.0, 1.0, 1.0]}`,
			agent.Config{MaxLen: 1024, HeadMaxLen: 256, OptionLayout: "sequential"}, nil},
		{"explicit layout", `{"option_layout": "sequential", "max_len": 64}`,
			agent.Config{MaxLen: 64, HeadMaxLen: 192, OptionLayout: "sequential"}, nil},
		{"unknown keys are ignored", `{"head_layers": 2, "training": {"hours": 1.96}}`,
			agent.Config{MaxLen: 512, HeadMaxLen: 192, OptionLayout: "sequential"}, nil},
		{"in-range temperatures are quiet", `{"temperature": [1.6, 1.25, 2], "temperature_by_options": {"choice:2": 1.9}}`,
			agent.Config{MaxLen: 512, HeadMaxLen: 192, OptionLayout: "sequential"}, nil},
		{"clamped temperatures warn once", `{"temperature": [1.0, 9, 0.1], "temperature_by_options": {"choice:11+": 0.1006}}`,
			agent.Config{MaxLen: 512, HeadMaxLen: 192, OptionLayout: "sequential"},
			[]string{"[0.5, 5]", "choice:11+=0.1006 -> 0.5", "temperature[1]=9 -> 5", "temperature[2]=0.1 -> 0.5"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := agent.LoadConfig(toyCheckpoint(c.config))
			if err != nil {
				t.Fatal(err)
			}
			warnings := got.Warnings
			got.Warnings = nil
			if got.MaxLen != c.want.MaxLen || got.HeadMaxLen != c.want.HeadMaxLen || got.OptionLayout != c.want.OptionLayout {
				t.Fatalf("Config = %+v, want %+v", got, c.want)
			}
			if len(c.warnings) == 0 && len(warnings) != 0 {
				t.Fatalf("warnings = %q, want none", warnings)
			}
			if len(c.warnings) > 0 {
				if len(warnings) != 1 {
					t.Fatalf("%d warnings, want one: %q", len(warnings), warnings)
				}
				for _, part := range c.warnings {
					if !strings.Contains(warnings[0], part) {
						t.Errorf("warning %q lacks %q", warnings[0], part)
					}
				}
			}
		})
	}
}

func TestLoadConfigErrors(t *testing.T) {
	cases := []struct {
		name   string
		config string // "-" for no file
		field  string
		is     error
	}{
		{"missing file", "-", "", fs.ErrNotExist},
		{"not JSON", `{`, "", nil},
		{"not an object", `[1]`, "", nil},
		{"other layout", `{"option_layout": "dual"}`, "option_layout", agent.ErrUnsupportedLayout},
		{"null layout", `{"option_layout": null}`, "option_layout", agent.ErrUnsupportedLayout},
		{"max_len is text", `{"max_len": "512"}`, "max_len", nil},
		{"max_len is a float", `{"max_len": 512.5}`, "max_len", nil},
		{"max_len is zero", `{"max_len": 0}`, "max_len", nil},
		{"head_max_len is negative", `{"head_max_len": -4}`, "head_max_len", nil},
		{"temperature is not a list", `{"temperature": 1.5}`, "temperature", nil},
		{"temperature has two values", `{"temperature": [1, 1]}`, "temperature", nil},
		{"buckets are a list", `{"temperature_by_options": [1]}`, "temperature_by_options", nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fsys := toyCheckpoint(c.config)
			if c.config == "-" {
				delete(fsys, "rl_agent_config.json")
			}
			_, err := agent.LoadConfig(fsys)
			var ce *agent.ConfigError
			if !errors.As(err, &ce) || ce.Field != c.field {
				t.Fatalf("LoadConfig = %v, want a ConfigError for %q", err, c.field)
			}
			if c.is != nil && !errors.Is(err, c.is) {
				t.Fatalf("LoadConfig = %v, want it to wrap %v", err, c.is)
			}
			if !strings.Contains(err.Error(), "rl_agent_config.json") {
				t.Errorf("error %q does not name the file", err)
			}
		})
	}
}

// TestConfigInUse pins that the checkpoint's own max_len is the budget, not the default.
func TestConfigInUse(t *testing.T) {
	long := strings.Repeat("x", 800)
	one := func(config string) (agent.Config, schema.Usage) {
		a, _ := open(t, config)
		p, err := a.Predict(context.Background(), long, questions(t))
		if err != nil {
			t.Fatal(err)
		}
		return a.Config(), p.Usage
	}
	cfg, u := one(`{"max_len": 1024, "head_max_len": 256}`)
	if cfg.MaxLen != 1024 || cfg.HeadMaxLen != 256 {
		t.Fatalf("Config = %+v", cfg)
	}
	if u.Truncated || u.StateTokens != 800 {
		t.Errorf("max_len 1024: usage = %s", jsonOf(t, u))
	}
	if _, u = one(`{}`); !u.Truncated || u.StateTokensDropped < 280 {
		t.Errorf("default max_len 512: usage = %s", jsonOf(t, u))
	}
}

func TestConfigReturnsACopy(t *testing.T) {
	a, _ := open(t, `{"temperature": [9, 1, 1]}`)
	a.Config().Warnings[0] = "changed"
	if w := a.Config().Warnings[0]; w == "changed" {
		t.Fatal("Config shares its Warnings with the agent")
	}
}

func TestOpenClosesTheSessionOnError(t *testing.T) {
	noTokenizer := toyCheckpoint(`{}`)
	delete(noTokenizer, "tokenizer/tokenizer.json")
	for name, fsys := range map[string]fs.FS{
		"bad config":    toyCheckpoint(`{"option_layout": "x"}`),
		"no tokenizer":  noTokenizer,
		"no config":     fstest.MapFS{},
		"empty tokeniz": fstest.MapFS{"rl_agent_config.json": {Data: []byte(`{}`)}, "tokenizer.json": {Data: []byte(`{}`)}},
	} {
		t.Run(name, func(t *testing.T) {
			s := &fakeSession{}
			if a, err := agent.Open(fsys, s); err == nil || a != nil {
				t.Fatalf("Open = %v, %v", a, err)
			}
			if s.closes != 1 {
				t.Fatalf("the session was closed %d times, want once", s.closes)
			}
		})
	}
	if _, err := agent.Open(toyCheckpoint(`{}`), nil); err == nil {
		t.Fatal("Open accepted a nil session")
	}
}

func TestOpenWithBadLayoutWraps(t *testing.T) {
	_, err := agent.Open(toyCheckpoint(`{"option_layout": "dual"}`), &fakeSession{})
	if !errors.Is(err, agent.ErrUnsupportedLayout) || !strings.Contains(err.Error(), "'dual'") {
		t.Fatalf("Open = %v", err)
	}
}

func TestClose(t *testing.T) {
	s := &fakeSession{}
	a, err := agent.Open(toyCheckpoint(`{}`), s)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("second Close = %v", err)
	}
	if s.closes != 1 {
		t.Fatalf("session closed %d times, want once", s.closes)
	}
	if _, err := a.Predict(context.Background(), "x", questions(t)); !errors.Is(err, agent.ErrClosed) {
		t.Fatalf("Predict after Close = %v", err)
	}
	if _, err := a.PredictBatch(context.Background(), []any{"x"}, questions(t)); !errors.Is(err, agent.ErrClosed) {
		t.Fatalf("PredictBatch after Close = %v", err)
	}
	if len(s.seen()) != 0 {
		t.Fatal("the closed agent still ran the session")
	}
}

func TestEmptyQuestionsAndStates(t *testing.T) {
	a, s := open(t, `{}`)
	ctx := context.Background()
	preds, err := a.PredictBatch(ctx, []any{"one", "two"}, schema.Questions{})
	if err != nil || len(preds) != 2 {
		t.Fatalf("PredictBatch = %v, %v", preds, err)
	}
	want := `{"model":"laya-rl-agent-onnx","answers":{},"usage":{"input_tokens":0,"output_tokens":0,"state_tokens":0,"state_tokens_dropped":0,"truncated":false,"truncated_questions":[]}}`
	for _, p := range preds {
		if got := jsonOf(t, p); got != want {
			t.Errorf("got %s\nwant %s", got, want)
		}
	}
	for _, states := range [][]any{nil, {}} {
		preds, err := a.PredictBatch(ctx, states, questions(t))
		if err != nil || preds == nil || len(preds) != 0 {
			t.Errorf("PredictBatch(%v) = %#v, %v, want an empty non-nil slice", states, preds, err)
		}
	}
	if len(s.seen()) != 0 {
		t.Fatal("the session ran for nothing")
	}
}

func TestStateErrors(t *testing.T) {
	a, s := open(t, `{}`)
	ctx := context.Background()
	if _, err := a.PredictBatch(ctx, []any{"fine", nil}, questions(t)); err == nil || !strings.Contains(err.Error(), "state 1: is nil") {
		t.Errorf("nil state: %v", err)
	}
	if _, err := a.Predict(ctx, struct{ X int }{1}, questions(t)); !errors.Is(err, sequence.ErrUnsupportedState) {
		t.Errorf("struct state: %v", err)
	}
	if len(s.seen()) != 0 {
		t.Error("a refused request still ran the session")
	}
}

func TestMarkersErrorPassesThrough(t *testing.T) {
	a, _ := open(t, `{"max_len": 40, "head_max_len": 30}`)
	labels := make([]string, 12)
	for i := range labels {
		labels[i] = fmt.Sprintf("option number %d", i)
	}
	qs, err := schema.NewQuestions(
		schema.Ask("fine", schema.MustQuestion(schema.Noul("ok"))),
		schema.Ask("wide", schema.MustQuestion(schema.ChoiceOf("pick", labels...))))
	if err != nil {
		t.Fatal(err)
	}
	_, err = a.Predict(context.Background(), "state", qs)
	var me *sequence.MarkersError
	if !errors.As(err, &me) || me.ID != "wide" || me.Total != 12 || me.MaxLen != 40 {
		t.Fatalf("Predict = %v (%+v)", err, me)
	}
}

// TestOneOptionQuestion pins that a question with one option still sends two marker slots.
func TestOneOptionQuestion(t *testing.T) {
	a, s := open(t, `{}`)
	qs, err := schema.NewQuestions(schema.Ask("only", schema.MustQuestion(schema.ChoiceOf("pick", "yes"))))
	if err != nil {
		t.Fatal(err)
	}
	p, err := a.Predict(context.Background(), "state", qs)
	if err != nil {
		t.Fatal(err)
	}
	if runs := s.seen(); len(runs) != 1 || runs[0].Markers != 2 {
		t.Fatalf("runs = %+v, want one run with 2 marker slots", runs)
	}
	if got := jsonOf(t, p.Answers); !strings.Contains(got, `"choice":"yes"`) || !strings.Contains(got, `"yes":1`) {
		t.Fatalf("answers = %s", got)
	}
}

func TestOutputShapeMismatch(t *testing.T) {
	a, s := open(t, `{}`)
	s.shape = func(o *infer.Output) { o.Logits = o.Logits[1:] }
	var se *infer.ShapeError
	if _, err := a.Predict(context.Background(), "state", questions(t)); !errors.As(err, &se) || se.Name != "logits" {
		t.Fatalf("Predict = %v", err)
	}
	s.shape = func(o *infer.Output) { o.ActLogits = nil }
	if _, err := a.Predict(context.Background(), "state", questions(t)); !errors.As(err, &se) || se.Name != "act_logits" {
		t.Fatalf("Predict = %v", err)
	}
}

func TestSessionErrorIsWrapped(t *testing.T) {
	a, s := open(t, `{}`)
	boom := errors.New("boom")
	s.before = func(int) error { return boom }
	if _, err := a.Predict(context.Background(), "state", questions(t)); !errors.Is(err, boom) {
		t.Fatalf("Predict = %v", err)
	}
}

func TestNonFiniteLogitsAreRefused(t *testing.T) {
	a, s := open(t, `{}`)
	s.shape = func(o *infer.Output) { o.Logits[0] = float32(nan()) }
	if _, err := a.Predict(context.Background(), "state", questions(t)); err == nil || !strings.Contains(err.Error(), `question "intent"`) && !strings.Contains(err.Error(), "question intent") {
		t.Fatalf("Predict = %v", err)
	}
}

func nan() float64 { z := 0.0; return z / z }

func varied() []any {
	var states []any
	for _, n := range []int{30, 5, 20, 10, 25, 1, 17, 8, 22, 3, 14} {
		states = append(states, strings.Repeat("a", n)+fmt.Sprint(n))
	}
	states = append(states, []any{"hello", "there"}, map[string]any{"k": "v"}, "")
	return states
}

// TestResultsAreTheSameWhateverTheBatching is the contract that makes chunking and sorting safe:
// each state's prediction depends on its own rows only, and comes back at its own index.
func TestResultsAreTheSameWhateverTheBatching(t *testing.T) {
	ctx := context.Background()
	states := varied()
	qs := questions(t)
	a, _ := open(t, `{}`)
	var want []string
	for _, st := range states {
		p, err := a.Predict(ctx, st, qs)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, jsonOf(t, p.Answers))
	}
	if want[0] == want[1] {
		t.Fatal("the states are not told apart by the fake session")
	}
	for _, size := range []int{0, 1, 2, 3, 5, len(states) - 1, len(states), len(states) + 4} {
		for _, sorted := range []bool{false, true} {
			opts := []agent.CallOption{agent.WithBatchSize(size)}
			if sorted {
				opts = append(opts, agent.WithSortByLength(true))
			}
			got, err := a.PredictBatch(ctx, states, qs, opts...)
			if err != nil || len(got) != len(states) {
				t.Fatalf("batch %d sorted %v: %v (%d results)", size, sorted, err, len(got))
			}
			for i, p := range got {
				if jsonOf(t, p.Answers) != want[i] {
					t.Errorf("batch %d sorted %v: state %d got another state's answers", size, sorted, i)
				}
			}
		}
	}
}

func lengths(runs []run) []int {
	var out []int
	for _, r := range runs {
		out = append(out, r.Length)
	}
	return out
}

func TestChunkingAndSorting(t *testing.T) {
	ctx := context.Background()
	states := varied()[:11] // distinct lengths, none a list
	qs := questions(t)
	record := func(opts ...agent.CallOption) []run {
		a, s := open(t, `{}`)
		if _, err := a.PredictBatch(ctx, states, qs, opts...); err != nil {
			t.Fatal(err)
		}
		return s.seen()
	}
	rowsOf := func(runs []run) (rows []int) {
		for _, r := range runs {
			rows = append(rows, r.Rows)
		}
		return rows
	}
	if got := rowsOf(record()); !reflect.DeepEqual(got, []int{33}) {
		t.Errorf("no batch size: runs of %v rows", got)
	}
	if got := rowsOf(record(agent.WithBatchSize(4))); !reflect.DeepEqual(got, []int{12, 12, 9}) {
		t.Errorf("batch 4: runs of %v rows, want 4 states x 3 questions each", got)
	}
	// Sorting reorders only when 1 < chunk < states: otherwise the runs are the unsorted ones.
	for _, size := range []int{1, 11, 40} {
		plain := record(agent.WithBatchSize(size))
		sorted := record(agent.WithBatchSize(size), agent.WithSortByLength(true))
		if !reflect.DeepEqual(plain, sorted) {
			t.Errorf("batch %d: sorting changed the runs", size)
		}
	}
	unsorted := lengths(record(agent.WithBatchSize(3)))
	sorted := lengths(record(agent.WithBatchSize(3), agent.WithSortByLength(true)))
	for i := 1; i < len(sorted); i++ {
		if sorted[i] < sorted[i-1] {
			t.Errorf("sorted run lengths %v are not ascending", sorted)
		}
	}
	if reflect.DeepEqual(unsorted, sorted) {
		t.Errorf("sorting changed nothing: %v", sorted)
	}
}

// TestUsageIsPerState pins that a state's input_tokens are the tokens of its own rows only.
func TestUsageIsPerState(t *testing.T) {
	a, s := open(t, `{}`)
	states := []any{strings.Repeat("a", 9), strings.Repeat("b", 31), "c"}
	got, err := a.PredictBatch(context.Background(), states, questions(t))
	if err != nil {
		t.Fatal(err)
	}
	runs := s.seen()
	if len(runs) != 1 {
		t.Fatalf("%d runs", len(runs))
	}
	var total int64
	for i, p := range got {
		var own int64
		for _, n := range runs[0].Tokens[i*3 : i*3+3] {
			own += n
		}
		total += own
		if int64(p.Usage.InputTokens) != own {
			t.Errorf("state %d: input_tokens %d, want %d", i, p.Usage.InputTokens, own)
		}
		if p.Usage.OutputTokens != 0 {
			t.Errorf("state %d: output_tokens %d", i, p.Usage.OutputTokens)
		}
	}
	if got[0].Usage.StateTokens != 9 || got[1].Usage.StateTokens != 31 || got[2].Usage.StateTokens != 1 {
		t.Errorf("state_tokens = %d, %d, %d", got[0].Usage.StateTokens, got[1].Usage.StateTokens, got[2].Usage.StateTokens)
	}
	if got[0].Usage.InputTokens == got[1].Usage.InputTokens || int64(got[0].Usage.InputTokens) == total {
		t.Error("input_tokens do not differ per state")
	}
}

// TestTruncationReport pins state_tokens_dropped as the worst question, and truncated_questions as
// the ones that were cut, when the questions leave different room.
func TestTruncationReport(t *testing.T) {
	a, _ := open(t, `{"max_len": 200, "head_max_len": 120}`)
	wide := make([]string, 10)
	for i := range wide {
		wide[i] = fmt.Sprintf("a fairly long option text number %d", i)
	}
	qs, err := schema.NewQuestions(
		schema.Ask("small", schema.MustQuestion(schema.Noul("ok"))),
		schema.Ask("wide", schema.MustQuestion(schema.ChoiceOf("pick one", wide...))),
		schema.Ask("small2", schema.MustQuestion(schema.Noul("ok"))))
	if err != nil {
		t.Fatal(err)
	}
	p, err := a.Predict(context.Background(), strings.Repeat("x", 100), qs)
	if err != nil {
		t.Fatal(err)
	}
	u := p.Usage
	spec, cfg := fixtures.ToyTokenizer()
	tok, err := tokenizer.Parse(spec, cfg)
	if err != nil {
		t.Fatal(err)
	}
	worst, cut := 0, []string{}
	for i := 0; i < qs.Len(); i++ {
		room, err := sequence.StateRoom(tok, qs.At(i).Question, sequence.Budget{MaxLen: 200, HeadMaxLen: 120})
		if err != nil {
			t.Fatal(err)
		}
		if dropped := 100 - room; dropped > 0 {
			worst = max(worst, dropped)
			cut = append(cut, qs.At(i).ID)
		}
	}
	if worst == 0 || len(cut) != 1 || cut[0] != "wide" {
		t.Fatalf("the test no longer cuts only the wide question: %v, %d", cut, worst)
	}
	if !u.Truncated || u.StateTokensDropped != worst || !reflect.DeepEqual(u.TruncatedQuestions, cut) || u.StateTokens != 100 {
		t.Fatalf("usage = %s, want dropped %d for %v", jsonOf(t, u), worst, cut)
	}
}

func TestLanguageWithoutOverridesChangesNothing(t *testing.T) {
	a, _ := open(t, `{}`)
	plain, err := a.Predict(context.Background(), "state", questions(t))
	if err != nil {
		t.Fatal(err)
	}
	tagged, err := a.Predict(context.Background(), "state", questions(t), agent.WithLanguage("yo-NG"))
	if err != nil || jsonOf(t, plain) != jsonOf(t, tagged) {
		t.Fatalf("a language tag changed the answer: %v", err)
	}
}

func TestContextStopsTheBatchBetweenChunks(t *testing.T) {
	states := varied()
	qs := questions(t)
	t.Run("cancelled during the first chunk", func(t *testing.T) {
		a, s := open(t, `{}`)
		ctx, cancel := context.WithCancel(context.Background())
		s.before = func(n int) error {
			if n == 0 {
				cancel()
			}
			return nil
		}
		got, err := a.PredictBatch(ctx, states, qs, agent.WithBatchSize(3))
		if err != ctx.Err() || got != nil {
			t.Fatalf("PredictBatch = %v, %v, want ctx.Err()", got, err)
		}
		if n := len(s.seen()); n != 1 {
			t.Fatalf("%d runs after the cancel, want 1", n)
		}
	})
	t.Run("cancelled before the call", func(t *testing.T) {
		a, s := open(t, `{}`)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := a.PredictBatch(ctx, states, qs, agent.WithBatchSize(3)); !errors.Is(err, context.Canceled) {
			t.Fatalf("PredictBatch = %v", err)
		}
		if len(s.seen()) != 0 {
			t.Fatal("ran with a cancelled context")
		}
	})
	t.Run("deadline between sorted chunks", func(t *testing.T) {
		a, s := open(t, `{}`)
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		s.before = func(int) error { <-ctx.Done(); return nil }
		if _, err := a.PredictBatch(ctx, states, qs, agent.WithBatchSize(2), agent.WithSortByLength(true)); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("PredictBatch = %v", err)
		}
		if n := len(s.seen()); n != 1 {
			t.Fatalf("%d runs, want 1", n)
		}
	})
}

func TestConcurrentPredict(t *testing.T) {
	a, _ := open(t, `{}`)
	qs := questions(t)
	states := varied()
	want := make([]string, len(states))
	for i, st := range states {
		p, err := a.Predict(context.Background(), st, qs)
		if err != nil {
			t.Fatal(err)
		}
		want[i] = jsonOf(t, p)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for round := 0; round < 5; round++ {
				if g%2 == 0 {
					i := (g + round) % len(states)
					p, err := a.Predict(context.Background(), states[i], qs)
					if err != nil || jsonOf(t, p) != want[i] {
						errs <- fmt.Errorf("goroutine %d: Predict(%d) = %v", g, i, err)
						return
					}
					continue
				}
				got, err := a.PredictBatch(context.Background(), states, qs, agent.WithBatchSize(3+g), agent.WithSortByLength(true))
				if err != nil {
					errs <- err
					return
				}
				for i, p := range got {
					if jsonOf(t, p) != want[i] {
						errs <- fmt.Errorf("goroutine %d: state %d differs", g, i)
						return
					}
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

func TestCloseWaitsForCallsInFlight(t *testing.T) {
	s := &fakeSession{}
	a, err := agent.Open(toyCheckpoint(`{}`), s)
	if err != nil {
		t.Fatal(err)
	}
	started, release := make(chan struct{}), make(chan struct{})
	s.before = func(int) error { close(started); <-release; return nil }
	done := make(chan error, 1)
	go func() { _, err := a.Predict(context.Background(), "state", questions(t)); done <- err }()
	<-started
	closed := make(chan struct{})
	go func() { a.Close(); close(closed) }()
	select {
	case <-closed:
		t.Fatal("Close returned while a call was still running")
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	<-closed
}
