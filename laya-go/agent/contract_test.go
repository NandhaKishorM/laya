package agent_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
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
)

// within fails the test if f does not return in a second, and returns its error.
func within(t *testing.T, what string, f func() error) error {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- f() }()
	select {
	case err := <-done:
		return err
	case <-time.After(time.Second):
		t.Fatalf("%s did not return within a second", what)
		return nil
	}
}

// blockedClose opens an agent whose first Run blocks until release is closed, starts a Predict
// into that Run, then starts Close behind it and waits until the agent reports itself closed.
func blockedClose(t *testing.T, s *fakeSession, states []any, opts ...agent.CallOption) (a *agent.Agent, release chan struct{}, predict chan error, closed chan struct{}) {
	t.Helper()
	a, err := agent.Open(toyCheckpoint(`{}`), s)
	if err != nil {
		t.Fatal(err)
	}
	started := make(chan struct{})
	release = make(chan struct{})
	s.before = func(n int) error {
		if n == 0 {
			close(started)
			<-release
		}
		return nil
	}
	predict = make(chan error, 1)
	go func() { _, err := a.PredictBatch(context.Background(), states, questions(t), opts...); predict <- err }()
	<-started
	closed = make(chan struct{})
	go func() { a.Close(); close(closed) }()
	for deadline := time.Now().Add(time.Second); ; time.Sleep(time.Millisecond) {
		err := within(t, "a probe Predict", func() error {
			_, err := a.Predict(context.Background(), "probe", questions(t))
			return err
		})
		if errors.Is(err, agent.ErrClosed) {
			return a, release, predict, closed
		}
		if time.Now().After(deadline) {
			t.Fatal("Close never marked the agent closed")
		}
	}
}

// TestCloseBehindABlockedRunDoesNotStallNewCalls covers this scenario: Close waits on a
// call stuck in the session, and a new call, even one with a dead context, must not wait too.
func TestCloseBehindABlockedRunDoesNotStallNewCalls(t *testing.T) {
	s := &fakeSession{}
	a, release, predict, closed := blockedClose(t, s, []any{"state"})
	err := within(t, "Predict with a live context", func() error {
		_, err := a.Predict(context.Background(), "x", questions(t))
		return err
	})
	if !errors.Is(err, agent.ErrClosed) {
		t.Errorf("new Predict = %v, want ErrClosed", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = within(t, "Predict with a cancelled context", func() error {
		_, err := a.Predict(ctx, "x", questions(t))
		return err
	})
	if err != ctx.Err() {
		t.Errorf("Predict with a cancelled context = %v, want ctx.Err()", err)
	}
	select {
	case <-closed:
		t.Fatal("Close returned while a call was in the session")
	default:
	}
	close(release)
	if err := <-predict; err != nil {
		t.Errorf("the call in flight = %v", err)
	}
	<-closed
}

func TestCloseStopsAMultiChunkCallBetweenRuns(t *testing.T) {
	s := &fakeSession{}
	_, release, predict, closed := blockedClose(t, s, []any{"a", "b", "c"}, agent.WithBatchSize(1))
	close(release)
	if err := <-predict; !errors.Is(err, agent.ErrClosed) {
		t.Errorf("the call in flight = %v, want ErrClosed after its first run", err)
	}
	<-closed
	// The call's first run, and at most the probes that ran before Close began.
	for i, r := range s.seen() {
		if r.Rows == 9 {
			t.Errorf("run %d held 3 states: the batch size was lost", i)
		}
	}
	if got := len(s.seen()); got > 4 {
		t.Errorf("%d runs: the call went on after Close", got)
	}
}

func TestZeroAgentIsClosed(t *testing.T) {
	var a agent.Agent
	if err := a.Close(); err != nil {
		t.Errorf("Close = %v", err)
	}
	if err := a.Close(); err != nil {
		t.Errorf("second Close = %v", err)
	}
	if _, err := a.Predict(context.Background(), "x", questions(t)); !errors.Is(err, agent.ErrClosed) {
		t.Errorf("Predict = %v", err)
	}
	if cfg := a.Config(); cfg.MaxLen != 0 {
		t.Errorf("Config = %+v", cfg)
	}
}

func TestConfigErrorWithoutACause(t *testing.T) {
	for _, e := range []*agent.ConfigError{{}, {Field: "max_len"}} {
		if msg := e.Error(); !strings.HasPrefix(msg, "agent: rl_agent_config.json") {
			t.Errorf("Error = %q", msg)
		}
	}
}

type failingClose struct {
	fakeSession
	err error
}

func (f *failingClose) Close() error { f.fakeSession.Close(); return f.err }

func TestOpenReportsTheSessionsCloseError(t *testing.T) {
	boom := errors.New("close failed")
	s := &failingClose{err: boom}
	_, err := agent.Open(toyCheckpoint(`{"option_layout": "dual"}`), s)
	if !errors.Is(err, agent.ErrUnsupportedLayout) || !errors.Is(err, boom) {
		t.Fatalf("Open = %v, want both the load error and the close error", err)
	}
	if s.closes != 1 {
		t.Errorf("closed %d times", s.closes)
	}
}

func TestOpenRefusesWhatCannotRun(t *testing.T) {
	t.Run("nil checkpoint", func(t *testing.T) {
		s := &fakeSession{}
		if a, err := agent.Open(nil, s); err == nil || a != nil || s.closes != 1 {
			t.Fatalf("Open = %v, %v with %d closes", a, err, s.closes)
		}
	})
	roles := map[string]string{"pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"}
	for missing := range roles {
		t.Run("no "+missing, func(t *testing.T) {
			var parts []string
			for role, content := range roles {
				if role != missing {
					parts = append(parts, fmt.Sprintf("%q:%q", role, content))
				}
			}
			slices.Sort(parts)
			fsys := toyCheckpoint(`{}`)
			fsys["tokenizer/tokenizer_config.json"].Data = []byte("{" + strings.Join(parts, ",") + "}")
			s := &fakeSession{}
			_, err := agent.Open(fsys, s)
			if !errors.Is(err, sequence.ErrMissingSpecialToken) || s.closes != 1 {
				t.Fatalf("Open = %v with %d closes", err, s.closes)
			}
		})
	}
	t.Run("tokenizer load error", func(t *testing.T) {
		fsys := toyCheckpoint(`{}`)
		fsys["tokenizer/tokenizer.json"].Data = []byte(`{`)
		var ce *agent.ConfigError
		if _, err := agent.Open(fsys, &fakeSession{}); !errors.As(err, &ce) || ce.Field != "tokenizer" {
			t.Fatalf("Open = %v, want a ConfigError for the tokenizer", err)
		}
	})
}

func TestNilOptionsAreSkipped(t *testing.T) {
	s := &fakeSession{}
	a, err := agent.Open(toyCheckpoint(`{}`), s, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if _, err := a.PredictBatch(context.Background(), []any{"x", "y"}, questions(t), nil, agent.WithBatchSize(1), nil); err != nil {
		t.Fatal(err)
	}
	if n := len(s.seen()); n != 2 {
		t.Errorf("%d runs, want 2: the option after the nil one was lost", n)
	}
}

func TestSortByLengthCanBeTurnedOff(t *testing.T) {
	states := varied()[:11]
	runs := func(opts ...agent.CallOption) []int {
		a, s := open(t, `{}`)
		if _, err := a.PredictBatch(context.Background(), states, questions(t), opts...); err != nil {
			t.Fatal(err)
		}
		return lengths(s.seen())
	}
	plain := runs(agent.WithBatchSize(3))
	if got := runs(agent.WithBatchSize(3), agent.WithSortByLength(true)); reflect.DeepEqual(got, plain) {
		t.Fatal("sorting changed nothing; the test proves nothing")
	}
	if got := runs(agent.WithBatchSize(3), agent.WithSortByLength(true), agent.WithSortByLength(false)); !reflect.DeepEqual(got, plain) {
		t.Errorf("WithSortByLength(false) left runs %v, want %v", got, plain)
	}
}

func TestRunErrors(t *testing.T) {
	boom := errors.New("boom")
	cases := []struct {
		name  string
		setup func(*fakeSession)
		is    error
		as    any // a pointer to the typed error that must also be found
	}{
		{"the session fails", func(s *fakeSession) { s.before = func(int) error { return boom } }, boom, nil},
		{"the session reports a cancel", func(s *fakeSession) { s.before = func(int) error { return context.Canceled } }, context.Canceled, nil},
		{"the session is closed under the agent", func(s *fakeSession) { s.before = func(int) error { return infer.ErrClosed } }, infer.ErrClosed, nil},
		{"a wrong-shaped output", func(s *fakeSession) { s.shape = func(o *infer.Output) { o.Logits = o.Logits[1:] } }, nil, new(*infer.ShapeError)},
		{"a NaN logit", func(s *fakeSession) { s.shape = func(o *infer.Output) { o.Logits[0] = float32(nan()) } }, nil, nil},
		{"an infinite action logit", func(s *fakeSession) {
			s.shape = func(o *infer.Output) { o.ActLogits[1] = float32(math.Inf(1)) }
		}, nil, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a, s := open(t, `{}`)
			c.setup(s)
			_, err := a.Predict(context.Background(), "state", questions(t))
			var re *agent.RunError
			if !errors.As(err, &re) {
				t.Fatalf("Predict = %v, want a *RunError", err)
			}
			if r := s.seen()[0]; re.Rows != r.Rows || re.Length != r.Length || re.Markers != r.Markers || re.Rows != 3 {
				t.Errorf("RunError = %+v, run was %+v", re, r)
			}
			if c.is != nil && !errors.Is(err, c.is) {
				t.Errorf("Predict = %v, want it to wrap %v", err, c.is)
			}
			if c.as != nil && !errors.As(err, c.as) {
				t.Errorf("Predict = %v, want %T", err, c.as)
			}
			if !strings.HasPrefix(err.Error(), "agent: ") {
				t.Errorf("message %q lacks the package prefix", err)
			}
		})
	}
}

// TestActionLogitFailureNamesTheQuestion pins the message of C8: the question id and the action
// logits, not "logits" alone.
func TestActionLogitFailureNamesTheQuestion(t *testing.T) {
	a, s := open(t, `{}`)
	s.shape = func(o *infer.Output) { o.ActLogits[2] = float32(nan()) } // row 1: urgency
	_, err := a.Predict(context.Background(), "state", questions(t))
	var re *agent.RunError
	if !errors.As(err, &re) || !strings.Contains(err.Error(), "action logits of question urgency") {
		t.Fatalf("Predict = %v", err)
	}
}

func TestStateErrors_Batch(t *testing.T) {
	a, _ := open(t, `{}`)
	ctx := context.Background()
	type S struct{ X int }
	var nilPtr *string
	cases := []struct {
		name   string
		states []any
		index  int
		opts   []agent.CallOption
	}{
		{"unsupported", []any{"a", "b", S{1}, "d"}, 2, nil},
		{"unsupported in a later window", []any{"a", "b", "c", S{1}, "e"}, 3, []agent.CallOption{agent.WithBatchSize(1)}},
		{"untyped nil", []any{"a", nil}, 1, nil},
		{"nil pointer", []any{"a", "b", nilPtr}, 2, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := a.PredictBatch(ctx, c.states, questions(t), c.opts...)
			var se *agent.StateError
			if !errors.As(err, &se) || se.Index != c.index || !errors.Is(err, sequence.ErrUnsupportedState) {
				t.Fatalf("PredictBatch = %v, want a StateError at %d wrapping ErrUnsupportedState", err, c.index)
			}
			if !strings.Contains(err.Error(), fmt.Sprintf("state %d", c.index)) {
				t.Errorf("message %q does not name the state", err)
			}
		})
	}
	// A nil slice and a nil map are an empty list and dict, as in the SDK's other packages.
	if _, err := a.PredictBatch(ctx, []any{[]any(nil), map[string]any(nil)}, questions(t)); err != nil {
		t.Errorf("nil slice and map: %v", err)
	}
	// A failing question names its state too; Predict returns the inner error alone.
	wide := make([]string, 12)
	for i := range wide {
		wide[i] = fmt.Sprintf("option number %d", i)
	}
	qs, err := schema.NewQuestions(schema.Ask("wide", schema.MustQuestion(schema.ChoiceOf("pick", wide...))))
	if err != nil {
		t.Fatal(err)
	}
	narrow, _ := open(t, `{"max_len": 40, "head_max_len": 30}`)
	var se *agent.StateError
	var me *sequence.MarkersError
	if _, err := narrow.PredictBatch(ctx, []any{"x", "y"}, qs); !errors.As(err, &se) || se.Index != 0 || !errors.As(err, &me) || me.ID != "wide" {
		t.Errorf("PredictBatch = %v", err)
	}
	_, err = narrow.Predict(ctx, "x", qs)
	if errors.As(err, &se) || !errors.As(err, &me) {
		t.Errorf("Predict = %v, want the MarkersError without a StateError", err)
	}
	_, err = a.Predict(ctx, S{1}, questions(t))
	if errors.As(err, &se) || !errors.Is(err, sequence.ErrUnsupportedState) {
		t.Errorf("Predict(struct) = %v, want the bare error", err)
	}
}

// TestNilStateIsRefusedWithoutQuestions pins that Go refuses the nil state Python raises a
// TypeError for, even when there is nothing to ask and nothing would run.
func TestNilStateIsRefusedWithoutQuestions(t *testing.T) {
	a, _ := open(t, `{}`)
	if _, err := a.PredictBatch(context.Background(), []any{"fine", nil}, schema.Questions{}); !errors.Is(err, sequence.ErrUnsupportedState) {
		t.Fatalf("PredictBatch = %v", err)
	}
}

func TestContextIsCheckedOnEntry(t *testing.T) {
	a, s := open(t, `{}`)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := a.PredictBatch(ctx, nil, questions(t)); err != ctx.Err() {
		t.Errorf("no states: %v, want ctx.Err()", err)
	}
	if _, err := a.PredictBatch(ctx, []any{"x"}, schema.Questions{}); err != ctx.Err() {
		t.Errorf("no questions: %v, want ctx.Err()", err)
	}
	if len(s.seen()) != 0 {
		t.Error("ran with a cancelled context")
	}
}

type ctxKey struct{}

// TestSessionSeesTheCallersContext pins that the context is forwarded to Run: a value on it,
// and its cancellation, both reach the session.
func TestSessionSeesTheCallersContext(t *testing.T) {
	a, s := open(t, `{}`)
	var mu sync.Mutex
	var got []any
	s.onCtx = func(ctx context.Context) {
		mu.Lock()
		got = append(got, ctx.Value(ctxKey{}))
		mu.Unlock()
	}
	ctx := context.WithValue(context.Background(), ctxKey{}, "mine")
	if _, err := a.Predict(ctx, "state", questions(t)); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []any{"mine"}) {
		t.Fatalf("the session saw context values %v", got)
	}
	cctx, cancel := context.WithCancel(context.Background())
	s.onCtx = func(c context.Context) { cancel(); <-c.Done() }
	s.before = func(int) error { return cctx.Err() }
	if _, err := within2(t, a, cctx); !errors.Is(err, context.Canceled) {
		t.Errorf("Predict = %v, want the cancel to have reached the session", err)
	}
}

// within2 runs one Predict that would hang forever if the session's context were not cancelled.
func within2(t *testing.T, a *agent.Agent, ctx context.Context) (schema.Prediction, error) {
	t.Helper()
	var p schema.Prediction
	err := within(t, "Predict", func() (err error) { p, err = a.Predict(ctx, "state", questions(t)); return })
	return p, err
}

// TestMaskedPositionsHoldThePadID pins the pad id the collator is given, on the toy tokenizer
// and on the shipped English one (pad id 50283), neither of which is 0.
func TestMaskedPositionsHoldThePadID(t *testing.T) {
	check := func(t *testing.T, a *agent.Agent, s *fakeSession, pad int64) {
		t.Helper()
		if _, err := a.PredictBatch(context.Background(), varied(), questions(t)); err != nil {
			t.Fatal(err)
		}
		var padded bool
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, pads := range s.pads {
			if len(pads) > 0 {
				padded = true
				if !reflect.DeepEqual(pads, []int64{pad}) {
					t.Errorf("masked positions hold %v, want only %d", pads, pad)
				}
			}
		}
		if !padded {
			t.Fatal("no row was padded; the test proves nothing")
		}
	}
	t.Run("toy", func(t *testing.T) {
		a, s := open(t, `{}`)
		check(t, a, s, fixtures.ToyPad)
	})
	t.Run("english", func(t *testing.T) {
		root := fixtures.Checkpoints()
		if root == "" {
			t.Skip("set LAYA_CHECKPOINTS to a directory holding the english and multilingual checkpoints")
		}
		s := &fakeSession{}
		a, err := agent.Open(os.DirFS(filepath.Join(root, "english")), s)
		if err != nil {
			t.Fatal(err)
		}
		defer a.Close()
		check(t, a, s, 50283)
	})
}

func TestEmptyQuestionsUsageIsAnEmptyList(t *testing.T) {
	a, _ := open(t, `{}`)
	preds, err := a.PredictBatch(context.Background(), []any{"x"}, schema.Questions{})
	if err != nil {
		t.Fatal(err)
	}
	if tq := preds[0].Usage.TruncatedQuestions; tq == nil || len(tq) != 0 {
		t.Fatalf("TruncatedQuestions = %#v, want an empty non-nil slice", tq)
	}
}

// pyWarning is the text ONNXAgent.__init__ warns with, less its "laya ONNX: " prefix. The
// expected strings below were produced by running laya.onnx_agent.ONNXAgent on each config with
// onnxruntime, transformers and the tokenizer fix patched out, and reading the RuntimeWarning.
const pyWarning = "this checkpoint ships temperatures outside [0.5, 5] which would distort confidence; clamping %s. " +
	"Treat confidence from the affected buckets as uncalibrated."

func TestClampWarningsMatchPython(t *testing.T) {
	cases := []struct {
		name, config, clamped string // clamped "" means no warning
	}{
		// A bucket and a type entry both clamped: Python lists buckets first.
		{"buckets before types", `{"temperature": [1.0, 9, 1.0], "temperature_by_options": {"choice:11+": 0.1006}}`,
			"choice:11+=0.1006 -> 0.5, temperature[1]=9 -> 5"},
		// A bool is no number to clamp_temperature, so both spellings are reported.
		{"a bool temperature", `{"temperature": [false, true, 1.0]}`, "temperature[0]=False -> 1, temperature[1]=True -> 1"},
		// float("2.0") is 2.0, applied unchanged; float("9") is 9.0, clamped to 5.
		{"numeric strings", `{"temperature": ["2.0", "9", 1.0]}`, "temperature[1]='9' -> 5"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg, err := agent.LoadConfig(toyCheckpoint(c.config))
			if err != nil {
				t.Fatal(err)
			}
			if want := []string{fmt.Sprintf(pyWarning, c.clamped)}; !reflect.DeepEqual(cfg.Warnings, want) {
				t.Fatalf("warnings = %q\nwant       %q", cfg.Warnings, want)
			}
		})
	}
	for _, quiet := range []string{`{"temperature": ["2.0", " 1.5 ", 1]}`} {
		if cfg, err := agent.LoadConfig(toyCheckpoint(quiet)); err != nil || len(cfg.Warnings) != 0 {
			t.Errorf("%s: %v, %q, want no warning", quiet, err, cfg.Warnings)
		}
	}
}

func TestMaxLenTooLarge(t *testing.T) {
	_, err := agent.LoadConfig(toyCheckpoint(`{"max_len": 99999999999999999999}`))
	var ce *agent.ConfigError
	if !errors.As(err, &ce) || ce.Field != "max_len" || !strings.Contains(err.Error(), "99999999999999999999 is too large") {
		t.Fatalf("LoadConfig = %v", err)
	}
	_, err = agent.LoadConfig(toyCheckpoint(`{"head_max_len": -99999999999999999999}`))
	if !errors.As(err, &ce) || strings.Contains(err.Error(), "too large") || !strings.Contains(err.Error(), "positive integer") {
		t.Fatalf("LoadConfig = %v, want the positive-integer message", err)
	}
}

func TestMessagesCarryThePackageName(t *testing.T) {
	for want, err := range map[string]error{
		"agent: ": agent.ErrClosed, "infer: ": infer.ErrClosed,
	} {
		if !strings.HasPrefix(err.Error(), want) {
			t.Errorf("%q lacks the prefix %q", err, want)
		}
	}
	if !strings.HasPrefix(agent.ErrUnsupportedLayout.Error(), "agent: ") {
		t.Errorf("%q", agent.ErrUnsupportedLayout)
	}
	if _, err := agent.LoadConfig(fstestEmpty()); !strings.HasPrefix(err.Error(), "agent: ") {
		t.Errorf("%q", err)
	}
	if got := (&agent.StateError{Index: 3, Err: errors.New("x")}).Error(); got != "agent: state 3: x" {
		t.Errorf("StateError = %q", got)
	}
}

func fstestEmpty() fstest.MapFS { return fstest.MapFS{} }

// TestTokenizerErrorDoesNotBlameTheConfigFile: a tokenizer that cannot be loaded is not a field of
// rl_agent_config.json, so its message must not point there.
func TestTokenizerErrorDoesNotBlameTheConfigFile(t *testing.T) {
	err := &agent.ConfigError{Field: "tokenizer", Err: errors.New("no tokenizer.json")}
	if got, want := err.Error(), "agent: tokenizer: no tokenizer.json"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
