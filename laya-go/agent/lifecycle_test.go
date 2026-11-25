package agent_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

// TestEnteringNeverWaitsForACallInFlight is this scenario: a Run is blocked,
// predictors with a short context arrive as Close begins, and none may wait for that Run.
func TestEnteringNeverWaitsForACallInFlight(t *testing.T) {
	s := &fakeSession{}
	a, err := agent.Open(toyCheckpoint(`{}`), s)
	if err != nil {
		t.Fatal(err)
	}
	qs := questions(t)
	started, release := make(chan struct{}), make(chan struct{})
	s.before = func(n int) error {
		if n == 0 {
			close(started)
			<-release
		}
		return nil
	}
	first := make(chan error, 1)
	go func() { _, err := a.Predict(context.Background(), "state", qs); first <- err }()
	<-started

	const predictors = 16
	var wg sync.WaitGroup
	var slow atomic.Int32
	gate := make(chan struct{})
	for i := 0; i < predictors; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-gate
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
			defer cancel()
			begin := time.Now()
			a.Predict(ctx, "state", qs)
			if time.Since(begin) > 500*time.Millisecond {
				slow.Add(1)
			}
		}()
	}
	closed := make(chan error, 1)
	close(gate)
	go func() { closed <- a.Close() }()
	wg.Wait()
	if n := slow.Load(); n != 0 {
		t.Errorf("%d of %d predictors waited for the blocked Run", n, predictors)
	}
	select {
	case <-closed:
		t.Fatal("Close returned while a Run was blocked")
	default:
	}
	close(release)
	if err := <-first; err != nil {
		t.Errorf("the call in flight = %v", err)
	}
	if err := <-closed; err != nil {
		t.Errorf("Close = %v", err)
	}
}

// waitRefused polls until a is refusing calls, which is when Close has begun. The probe is an
// empty batch: it enters and leaves without reaching the session, so it adds no run of its own.
func waitRefused(t *testing.T, a *agent.Agent, qs schema.Questions) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := a.PredictBatch(context.Background(), nil, qs); errors.Is(err, agent.ErrClosed) {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("the agent never started refusing calls")
}

func TestCallsAfterCloseBeganNeverReachTheSession(t *testing.T) {
	qs := questions(t)
	t.Run("a new call", func(t *testing.T) {
		s := &fakeSession{}
		a, _ := agent.Open(toyCheckpoint(`{}`), s)
		started, release := make(chan struct{}), make(chan struct{})
		s.before = func(n int) error {
			if n == 0 {
				close(started)
				<-release
			}
			return nil
		}
		go a.Predict(context.Background(), "state", qs)
		<-started
		go a.Close()
		waitRefused(t, a, qs)
		if _, err := a.PredictBatch(context.Background(), []any{"a", "b"}, qs); !errors.Is(err, agent.ErrClosed) {
			t.Errorf("PredictBatch = %v, want ErrClosed", err)
		}
		close(release)
		a.Close()
		if n := len(s.seen()); n != 1 {
			t.Errorf("the session ran %d times, want only the call in flight", n)
		}
	})
	t.Run("the next chunk of a call in flight", func(t *testing.T) {
		s := &fakeSession{}
		a, _ := agent.Open(toyCheckpoint(`{}`), s)
		s.before = func(n int) error {
			if n == 0 {
				go a.Close()
				waitRefused(t, a, qs)
			}
			return nil
		}
		_, err := a.PredictBatch(context.Background(), varied(), qs, agent.WithBatchSize(2))
		if !errors.Is(err, agent.ErrClosed) {
			t.Errorf("PredictBatch = %v, want ErrClosed", err)
		}
		if n := len(s.seen()); n != 1 {
			t.Errorf("%d runs, want 1", n)
		}
	})
}

type closeFails struct {
	*fakeSession
	err error
}

func (c closeFails) Close() error { c.fakeSession.Close(); return c.err }

func TestConcurrentClosesShareTheError(t *testing.T) {
	boom := errors.New("boom")
	fs := &fakeSession{}
	a, err := agent.Open(toyCheckpoint(`{}`), closeFails{fs, boom})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make([]error, 8)
	for i := range errs {
		wg.Add(1)
		go func() { defer wg.Done(); errs[i] = a.Close() }()
	}
	wg.Wait()
	for i, err := range errs {
		if !errors.Is(err, boom) {
			t.Errorf("Close #%d = %v, want the session's error", i, err)
		}
	}
	if fs.closes != 1 {
		t.Errorf("session closed %d times, want once", fs.closes)
	}
}

func TestNilStateText(t *testing.T) {
	a, _ := open(t, `{}`)
	_, err := a.Predict(context.Background(), nil, questions(t))
	if want := "agent: state is nil; pass a string, a map or a slice: " + sequence.ErrUnsupportedState.Error(); err == nil || err.Error() != want {
		t.Errorf("Predict(nil) = %v\nwant %q", err, want)
	}
	if !errors.Is(err, sequence.ErrUnsupportedState) {
		t.Errorf("Predict(nil) = %v, want it to wrap ErrUnsupportedState", err)
	}
	_, err = a.PredictBatch(context.Background(), []any{"ok", nil}, questions(t))
	var se *agent.StateError
	if !errors.As(err, &se) || se.Index != 1 || !strings.HasPrefix(err.Error(), "agent: state 1: is nil;") {
		t.Errorf("PredictBatch = %v", err)
	}
}

// TestPredictUnwrapsOnlyItsOwnStateError: a session error that wraps a *StateError must still
// come back as the *RunError around it.
func TestPredictUnwrapsOnlyItsOwnStateError(t *testing.T) {
	a, s := open(t, `{}`)
	inner := &agent.StateError{Index: 3, Err: errors.New("from the session")}
	s.before = func(int) error { return inner }
	_, err := a.Predict(context.Background(), "state", questions(t))
	var re *agent.RunError
	if _, top := err.(*agent.RunError); !top || !errors.As(err, &re) || !errors.Is(err, inner) {
		t.Fatalf("Predict = %T %v, want the *RunError around the session's error", err, err)
	}
}

type nilMapSession map[string]int

func (nilMapSession) Run(context.Context, infer.Batch) (infer.Output, error) {
	return infer.Output{}, nil
}
func (nilMapSession) Close() error { return nil }

func TestOpenRefusesATypedNilSession(t *testing.T) {
	var ptr *fakeSession
	var m nilMapSession
	_, untyped := agent.Open(toyCheckpoint(`{}`), nil)
	for name, s := range map[string]infer.Session{"nil pointer": ptr, "nil map": m} {
		a, err := agent.Open(toyCheckpoint(`{}`), s)
		if a != nil || err == nil || err.Error() != untyped.Error() {
			t.Errorf("%s: Open = %v, %v, want %v", name, a, err, untyped)
		}
	}
	if a, err := agent.Open(toyCheckpoint(`{}`), nilMapSession{}); err != nil || a == nil {
		t.Errorf("an empty, non-nil map session was refused: %v", err)
	}
}

func TestConfigErrorTexts(t *testing.T) {
	for config, want := range map[string]string{
		`{"option_layout": "dual"}`: "agent: rl_agent_config.json: option_layout: option_layout is not supported by the ONNX runtime: 'dual'",
		`{"temperature": [1, 1]}`:   "agent: rl_agent_config.json: temperature: temperature must list 3 values, one per question type, got 2",
	} {
		_, err := agent.LoadConfig(toyCheckpoint(config))
		if err == nil || err.Error() != want {
			t.Errorf("%s: got %v\nwant %q", config, err, want)
		}
	}
	if _, err := agent.LoadConfig(toyCheckpoint(`{"option_layout": "dual"}`)); !errors.Is(err, agent.ErrUnsupportedLayout) {
		t.Errorf("the sentinel is lost: %v", err)
	}
}

// ctxCancelsAfter is a context whose Err turns to Canceled after n calls that found it live.
type ctxCancelsAfter struct {
	context.Context
	left int
}

func (c *ctxCancelsAfter) Err() error {
	if c.left <= 0 {
		return context.Canceled
	}
	c.left--
	return nil
}

// TestContextIsCheckedBeforeEachState cancels while the states are being encoded: the entry check
// and the first state's pass, the second state's does not, and nothing may run.
func TestContextIsCheckedBeforeEachState(t *testing.T) {
	a, s := open(t, `{}`)
	ctx := &ctxCancelsAfter{Context: context.Background(), left: 3}
	got, err := a.PredictBatch(ctx, varied(), questions(t))
	if !errors.Is(err, context.Canceled) || got != nil {
		t.Fatalf("PredictBatch = %v, %v, want Canceled", got, err)
	}
	if n := len(s.seen()); n != 0 {
		t.Fatalf("the session ran %d times after the cancel", n)
	}
}

// TestHexTemperaturesAreNotNumbers pins sameNumber's guard against Go's hex floats. Python 3's
// float() on the same strings, run with the reference's clamp_temperature:
//
//	float('0x10')  raises ValueError    float('0x1p0') raises ValueError    float('1_0') -> 10.0
//
// so "0x1p0" is applied as 1.0 and reported as rejected, though Go parses it as 1. The "1_0"
// spelling is never a Go float, so the underscore half of the guard cannot change a result.
func TestHexTemperaturesAreNotNumbers(t *testing.T) {
	for _, raw := range []string{"0x1p0", "0x10", "0X1P0"} {
		cfg, err := agent.LoadConfig(toyCheckpoint(`{"temperature": ["` + raw + `", 1, 1]}`))
		if err != nil {
			t.Fatal(err)
		}
		if len(cfg.Warnings) != 1 || !strings.Contains(cfg.Warnings[0], "temperature[0]='"+raw+"' -> 1") {
			t.Errorf("%s: warnings = %q, want the entry reported as rejected", raw, cfg.Warnings)
		}
	}
}

func TestStateErrorWithoutACause(t *testing.T) {
	if got := (&agent.StateError{Index: 2}).Error(); got != "agent: state 2" {
		t.Errorf("got %q", got)
	}
}

func TestZeroAgentRefusesCallsBeforeClose(t *testing.T) {
	var a agent.Agent
	if _, err := a.Predict(context.Background(), "x", questions(t)); !errors.Is(err, agent.ErrClosed) {
		t.Errorf("Predict on the zero Agent = %v, want ErrClosed", err)
	}
}

type panickyClose struct{ *fakeSession }

func (panickyClose) Close() error { panic("boom") }

func TestPanickingSessionCloseIsReportedAgain(t *testing.T) {
	a, err := agent.Open(toyCheckpoint(`{}`), panickyClose{&fakeSession{}})
	if err != nil {
		t.Fatal(err)
	}
	func() {
		defer func() { recover() }()
		a.Close()
	}()
	if err := a.Close(); err == nil || !strings.Contains(err.Error(), "panicked: boom") {
		t.Errorf("second Close = %v, want the panic reported", err)
	}
}

func TestStateErrorTexts(t *testing.T) {
	for _, c := range []struct {
		err  *agent.StateError
		want string
	}{
		{&agent.StateError{Index: 0, Err: errors.New("state machine jammed")}, "agent: state 0: state machine jammed"},
		{&agent.StateError{Index: 1, Err: &agent.StateError{Index: 3, Err: errors.New("x")}}, "agent: state 1: agent: state 3: x"},
		{&agent.StateError{Index: 2, Err: errors.New("agent: state is nil")}, "agent: state 2: is nil"},
	} {
		if got := c.err.Error(); got != c.want {
			t.Errorf("got %q, want %q", got, c.want)
		}
	}
}
