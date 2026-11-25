package agent

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/decode"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// ErrClosed is returned by an Agent that has been closed. A session closed underneath an open
// agent yields a wrapped infer.ErrClosed instead.
var ErrClosed = errors.New("agent: agent is closed")

// StateError identifies the state at fault in a PredictBatch call.
type StateError struct {
	// Index is the state's position in the states slice.
	Index int
	Err   error
}

// Error names the state and the cause.
func (e *StateError) Error() string {
	if e.Err == nil {
		return fmt.Sprintf("agent: state %d", e.Index)
	}
	// The cause is a complete message ("agent: state is nil; ..."); keep one prefix and one subject.
	msg := e.Err.Error()
	if rest, ok := strings.CutPrefix(msg, "agent: state "); ok {
		if _, nested := e.Err.(*StateError); !nested {
			msg = rest
		}
	} else {
		msg = strings.TrimPrefix(msg, "agent: ")
	}
	return fmt.Sprintf("agent: state %d: %s", e.Index, msg)
}

// Unwrap returns the cause.
func (e *StateError) Unwrap() error { return e.Err }

// Agent answers typed questions about a state with a laya checkpoint, through an infer.Session.
// It is the port of laya's ONNXAgent, and it is safe for concurrent use. Use Open: the zero
// Agent is closed.
type Agent struct {
	tok   *tokenizer.Tokenizer
	padID int32
	cfg   Config
	temps *decode.Temperatures
	sess  infer.Session

	mu       sync.Mutex     // guards shut, so a call enters or is refused without waiting
	shut     bool           // set by Close under mu
	inflight sync.WaitGroup // calls that entered
	closed   atomic.Bool    // mirrors shut, for the check between chunks
	once     sync.Once      // runs Close's body
	closeErr error          // the session's Close error, read after once.Do

	hk *hookState // hooks; see hooks.go
}

// Option adjusts an Agent at Open: WithInstalledHooks, WithHookTimeout, WithHookErrors,
// WithConcurrentHooks, WithHookWarnings, WithLangTemperatures, WithExpectedSHA256 and WithCalibration.
type Option func(*options)

// Open loads rl_agent_config.json and the tokenizer from checkpoint, a directory holding a
// laya checkpoint, and takes ownership of s: on error it closes s, and otherwise Close does.
// The tokenizer must name the pad, CLS, SEP and mask tokens. The session must run the graph
// exported from that same checkpoint; nothing here can tell.
func Open(checkpoint fs.FS, s infer.Session, opts ...Option) (*Agent, error) {
	if hasNilValue(s) {
		return nil, errors.New("agent: Open needs a session")
	}
	o := options{raise: true}
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	if o.err != nil {
		return nil, errors.Join(o.err, s.Close())
	}
	a, err := open(checkpoint, s, o)
	if err != nil {
		return nil, errors.Join(err, s.Close())
	}
	return a, nil
}

func open(checkpoint fs.FS, s infer.Session, o options) (*Agent, error) {
	if checkpoint == nil {
		return nil, errors.New("agent: Open needs a checkpoint")
	}
	want, err := expectedDigests(o.sha256)
	if err != nil {
		return nil, err
	}
	if err := verifyDigests(checkpoint, want); err != nil {
		return nil, err
	}
	cfg, temps, err := loadConfig(checkpoint, o.langTemps)
	if err != nil {
		return nil, err
	}
	if len(o.calibration) > 0 {
		if temps, err = applyCalibration(&cfg, temps, o.calibration); err != nil {
			return nil, &ConfigError{Field: calibrationField, Err: err}
		}
	}
	tok, err := tokenizer.LoadFS(checkpoint, ".")
	if err != nil {
		return nil, &ConfigError{Field: tokenizerField, Err: err}
	}
	var pad int32
	for _, t := range []struct {
		name string
		id   func() (int32, bool)
	}{{"pad", tok.Pad}, {"CLS", tok.CLS}, {"SEP", tok.SEP}, {"mask", tok.Mask}} {
		id, ok := t.id()
		if !ok {
			return nil, fmt.Errorf("agent: %w: no %s token", sequence.ErrMissingSpecialToken, t.name)
		}
		if t.name == "pad" {
			pad = id
		}
	}
	return &Agent{tok: tok, padID: pad, cfg: cfg, temps: temps, sess: s, hk: newHookState(o)}, nil
}

// Config returns a copy of the checkpoint configuration the agent runs with.
func (a *Agent) Config() Config {
	cfg := a.cfg
	cfg.Warnings = append([]string(nil), cfg.Warnings...)
	return cfg
}

// Close marks the agent closed, waits for calls in flight and closes the session. Close blocks
// until calls in flight return. Cancelling a call's context ends it between chunks and releases a
// call still queued for the serial hook lock, but a hook that has started is always waited for:
// only WithHookTimeout bounds that wait. A call that has not yet entered returns ErrClosed without
// waiting. Close is idempotent: a second call, concurrent or not, waits for the first and returns
// the same error. Close must not be called from inside a call, from the session's Run or from a
// hook: it would wait for that call, which waits for Close (with a hook timeout set, a hook's
// Close returns only after the timeout releases its call).
func (a *Agent) Close() error {
	a.once.Do(func() {
		defer func() {
			if r := recover(); r != nil { // once is spent either way: later calls report this
				a.closeErr = fmt.Errorf("agent: the session's Close panicked: %v", r)
				panic(r)
			}
		}()
		a.mu.Lock()
		a.shut = true
		a.closed.Store(true)
		a.mu.Unlock()
		a.hk.quit() // releases calls queued for the serial lock before OnPredictStart
		a.inflight.Wait()
		if a.sess != nil {
			a.closeErr = a.sess.Close()
		}
	})
	return a.closeErr
}

// enter registers a call, or reports ErrClosed. It never waits for a call in flight. Every
// successful enter is paired with leave.
func (a *Agent) enter() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.shut || a.sess == nil {
		return ErrClosed
	}
	a.inflight.Add(1)
	return nil
}

func (a *Agent) leave() { a.inflight.Done() }

// CallOption adjusts one Predict or PredictBatch call.
type CallOption func(*call)

type call struct {
	lang   string
	batch  int
	sort   bool
	window int
	stride int

	// gated is set by WithMinConfidence or WithMinConfidenceMap; minConfidence is then a float64
	// or a pyjson.Object, checked by checkMinConfidence.
	gated         bool
	minConfidence any

	hooks       []Hook // WithHooks, then the callables, in Python's order
	starts      []Hook // WithOnPredictStart callables
	ends        []Hook // WithOnPredictEnd callables
	skipDefault bool   // WithoutDefaultHooks
	hooked      bool   // a hook option was given; PredictLong refuses it

	// maxLen and headMaxLen override the checkpoint's budgets when above 0; a start hook sets them.
	maxLen, headMaxLen int
}

// WithLanguage names the request's language as a BCP 47 tag. Its primary subtag selects a
// language's temperatures when the agent has any; with none configured it changes nothing.
func WithLanguage(tag string) CallOption { return func(c *call) { c.lang = tag } }

// WithBatchSize caps the states sent per session run; 0 or less means all in one run. A run
// holds states x questions rows.
func WithBatchSize(states int) CallOption { return func(c *call) { c.batch = states } }

// WithSortByLength(true) pads less: states are ordered by length inside a lookahead of eight
// batches before they are chunked. It needs a batch size above 1 and below the number of states,
// and results stay in input order. A different batch shape can move a probability by a rounding
// step. WithSortByLength(false) is the default.
func WithSortByLength(on bool) CallOption { return func(c *call) { c.sort = on } }

// Predict answers every question about one state. Its errors are those of PredictBatch, except
// that a failing state's error is returned without the *StateError around it.
func (a *Agent) Predict(ctx context.Context, state any, qs schema.Questions, opts ...CallOption) (schema.Prediction, error) {
	preds, err := a.PredictBatch(ctx, []any{state}, qs, opts...)
	if err != nil {
		if se, ok := err.(*StateError); ok { // the top-level error only, never one inside a RunError
			err = se.Err
		}
		return schema.Prediction{}, err
	}
	return preds[0], nil
}

// PredictBatch answers the questions for each state; results are in input order. A state is a
// string, or any value sequence.SerializeState accepts: a slice is a conversation whose end
// survives truncation. A nil state or nil pointer is refused; a nil slice or map is an empty
// list or dict. With no questions each state gets an empty answer and zero usage, with no model
// run.
//
// WithMinConfidence or WithMinConfidenceMap gates every answer: it sets low_confidence,
// abstention and abstention_threshold as laya does. An invalid threshold is refused first, as a
// *MinConfidenceError, before any state is encoded or run.
//
// Errors may be wrapped; match them with errors.Is and errors.As:
//   - *MinConfidenceError, for an invalid threshold, before anything else is checked;
//   - ErrClosed, once Close has been called (a context already done is reported first);
//   - ctx.Err(), checked on entry, before each state is encoded and between runs;
//   - *StateError, naming the state, around sequence.ErrUnsupportedState (also for a nil
//     state), sequence.ErrMissingSpecialToken or a *sequence.MarkersError with its ID set;
//   - *RunError, around whatever the session returned (so errors.Is finds context.Canceled and
//     infer.ErrClosed through it), a *infer.ShapeError, or a logit that cannot be decoded.
func (a *Agent) PredictBatch(ctx context.Context, states []any, qs schema.Questions, opts ...CallOption) ([]schema.Prediction, error) {
	var c call
	for _, opt := range opts {
		if opt != nil {
			opt(&c)
		}
	}
	if err := checkHooks(c.hooks); err != nil {
		return nil, err
	}
	// Validated up front, as laya does, so a bad threshold fails before any work.
	var g *gate
	if c.gated {
		var err error
		if g, err = checkMinConfidence(c.minConfidence); err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := a.enter(); err != nil {
		return nil, err
	}
	defer a.leave()
	if hooks := a.composeHooks(c); len(hooks) > 0 {
		return a.predictHooked(ctx, states, qs, c, g, hooks)
	}
	preds, err := a.inferStates(ctx, states, qs, c)
	if err != nil {
		return nil, err
	}
	if g != nil {
		g.apply(preds)
	}
	return preds, nil
}

// predictStates is PredictBatch past the entry checks, for a caller that has already entered.
func (a *Agent) predictStates(ctx context.Context, states []any, qs schema.Questions, c call) ([]schema.Prediction, error) {
	for i, state := range states {
		if isNil(state) {
			return nil, &StateError{Index: i, Err: fmt.Errorf("agent: state is nil; pass a string, a map or a slice: %w", sequence.ErrUnsupportedState)}
		}
	}
	if len(states) == 0 {
		return []schema.Prediction{}, nil
	}
	if qs.Len() == 0 {
		preds := make([]schema.Prediction, len(states))
		for i := range preds {
			preds[i] = schema.Prediction{Model: schema.Model, Answers: schema.Answers{},
				Usage: schema.Usage{TruncatedQuestions: []string{}}}
		}
		return preds, nil
	}
	return a.predictChunks(ctx, states, qs, c)
}

// isNil reports an untyped nil or a nil pointer; a nil slice or map is an empty list or dict.
func isNil(state any) bool {
	if state == nil {
		return true
	}
	rv := reflect.ValueOf(state)
	return rv.Kind() == reflect.Pointer && rv.IsNil()
}

// hasNilValue reports an untyped nil or an interface holding a nil pointer, map, slice, func or
// chan, which a method call would trip over.
func hasNilValue(v any) bool {
	if v == nil {
		return true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan:
		return rv.IsNil()
	}
	return false
}
