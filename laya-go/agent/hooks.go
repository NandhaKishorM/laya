package agent

// Hooks are laya/hooks.py for the three events a single agent raises: OnPredictStart, OnPredictEnd
// and OnError. on_route lives in package router, which runs it through the shared internal/hookrun; on_load and
// on_evict are not ported yet. Python's
// AsyncHook has no counterpart: a hook here is a plain method, and one that needs to wait runs
// its own goroutines (the agent itself waits for the method to return, bounded by
// WithHookTimeout).
//
// Composition order, as in Python: process-wide defaults, then the agent's installed hooks, then
// the call's WithHooks, then its WithOnPredictStart and WithOnPredictEnd callables.
// WithoutDefaultHooks drops only the first group.
//
// Unlike Python's ONNXAgent, whose hooks_concurrent defaults to true, hooks run serially unless
// WithConcurrentHooks is given: one hook runs at a time across all calls of an agent, so a hook
// needs no locking of its own. A hook must not call back into the agent that is running it while
// the hooks are serial; the call would wait for itself.
//
// A hook that panics is recovered and counts as a failed hook, as a raised exception does in
// Python. With WithHookErrors(true), the default, the first failure ends the chain and fails the
// call; with false it is reported to the WithHookWarnings sink and the chain goes on.
//
// WithHookTimeout bounds the wait for each hook. Go cannot stop a goroutine, so a hook that
// overruns keeps running after the call has returned an error: the timeout protects the request,
// not the process. Such a hook must not touch the PredictContext it was given (the call has moved
// on and owns it), and it no longer holds the serial lock. Each overrun leaks one goroutine until
// the hook returns, so "a hook needs no locking" holds only while no timeout has fired. A started
// hook is waited for until it returns or times out, whatever Close or the caller's context does,
// so a hook that may block needs a timeout; OnError and OnPredictEnd still run after a cancelled
// context. A hook that calls back into its own agent hangs on the serial lock, where Python's
// lock is re-entrant.
// A hook's MaxLen or HeadMaxLen of 0 or less means "no override", as Go has no None.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/internal/hookrun"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// Hook observes one Predict or PredictBatch call. A hook that wants only some events embeds
// BaseHook, or uses HookFuncs.
type Hook interface {
	// OnPredictStart runs before inference. It may rewrite ctx.States, ctx.Questions,
	// ctx.MaxLen and ctx.HeadMaxLen, or answer the call with ctx.Skip.
	OnPredictStart(ctx *PredictContext)
	// OnPredictEnd runs once the call is over, on success and on failure. ctx.Err is set on
	// failure. It may rewrite ctx.Results.
	OnPredictEnd(ctx *PredictContext)
	// OnError runs when the call failed, before OnPredictEnd, with ctx.Err set.
	OnError(ctx *PredictContext)
}

// BaseHook implements Hook with no-ops, for embedding.
type BaseHook struct{}

func (BaseHook) OnPredictStart(*PredictContext) {}
func (BaseHook) OnPredictEnd(*PredictContext)   {}
func (BaseHook) OnError(*PredictContext)        {}

// HookFuncs is a Hook made of functions; a nil one is a no-op. Use it as *HookFuncs: a value
// holds funcs, which cannot be compared, so RemoveHook could never find it. The error function
// is OnErr because a field cannot share its name with the OnError method.
type HookFuncs struct {
	OnStart, OnEnd, OnErr func(*PredictContext)
}

func (f *HookFuncs) OnPredictStart(c *PredictContext) { callIf(f.OnStart, c) }
func (f *HookFuncs) OnPredictEnd(c *PredictContext)   { callIf(f.OnEnd, c) }
func (f *HookFuncs) OnError(c *PredictContext)        { callIf(f.OnErr, c) }

func callIf(fn func(*PredictContext), c *PredictContext) {
	if fn != nil {
		fn(c)
	}
}

// HookUsage is the token count of a whole call: the sum over its predictions.
type HookUsage struct {
	InputTokens, OutputTokens int
}

// PredictContext is what a hook sees of one call. Hooks of one call share it and run one after
// another, so a rewrite made by one is seen by the next.
type PredictContext struct {
	States    []any
	Questions schema.Questions
	// RunID is 32 hex characters, shared by every hook of the call.
	RunID string
	// Results is nil until the call has answered (or a hook has skipped it); an empty non-nil
	// slice is a real answer to a call with no states.
	Results []schema.Prediction
	Model   string
	Agent   *Agent
	// MaxLen and HeadMaxLen override the checkpoint's budgets for this call when above 0.
	MaxLen, HeadMaxLen int
	// Usage is summed over Results; set before OnPredictEnd.
	Usage     HookUsage
	StartedAt time.Time
	// ElapsedMS is set before OnPredictEnd, not before OnError.
	ElapsedMS float64
	// Err is the failure, set before OnError.
	Err error
}

// Skip answers the call without inference: one result for the whole call, or one per state in
// ctx.States. It does not stop the chain, and the last Skip wins. Any other count is an error,
// returned so the hook can panic with it or give up.
func (c *PredictContext) Skip(results []schema.Prediction) error {
	if len(results) != 1 && len(results) != len(c.States) {
		return fmt.Errorf("ctx.skip() takes one result for the whole call or one per state in ctx.states (%d); got %d", len(c.States), len(results))
	}
	c.Results = results
	return nil
}

// aggregateUsage sums the per-state usage blocks.
func aggregateUsage(results []schema.Prediction) HookUsage {
	var u HookUsage
	for _, r := range results {
		u.InputTokens += r.Usage.InputTokens
		u.OutputTokens += r.Usage.OutputTokens
	}
	return u
}

// --- options ---

type options struct {
	installed   []Hook
	timeout     time.Duration
	raise       bool
	concurrent  bool
	warn        func(string)
	err         error
	langTemps   map[string]any    // WithLangTemperatures
	sha256      map[string]string // WithExpectedSHA256
	calibration []byte            // WithCalibration
}

// WithInstalledHooks installs hooks on the agent, as AddHook does after Open.
func WithInstalledHooks(hooks ...Hook) Option {
	return func(o *options) {
		if err := checkHooks(hooks); err != nil && o.err == nil {
			o.err = err
		}
		o.installed = append(o.installed, hooks...)
	}
}

// WithHookTimeout bounds the wait for each hook. d must be positive; leave the option out for no
// limit.
func WithHookTimeout(d time.Duration) Option {
	return func(o *options) {
		if err := validateTimeout(d.Seconds()); err != nil && o.err == nil {
			o.err = err
		}
		o.timeout = d
	}
}

// WithHookErrors chooses what a failing hook does: raise true (the default) fails the call,
// false warns and goes on.
func WithHookErrors(raise bool) Option { return func(o *options) { o.raise = raise } }

// WithConcurrentHooks lets hooks of concurrent calls run at the same time; the default runs one
// at a time.
func WithConcurrentHooks() Option { return func(o *options) { o.concurrent = true } }

// WithHookWarnings sets where a hook failure that does not fail the call is reported; the
// default is the log package.
func WithHookWarnings(sink func(msg string)) Option { return func(o *options) { o.warn = sink } }

// validateTimeout is hooks.validate_timeout: positive and finite.
func validateTimeout(seconds float64) error { return hookrun.ValidateTimeout(seconds) }

// WithHooks adds hooks to one call, after the defaults and the installed ones.
func WithHooks(hooks ...Hook) CallOption {
	return func(c *call) {
		c.hooked = true
		c.hooks = append(c.hooks, hooks...)
	}
}

// WithOnPredictStart adds a function run as an OnPredictStart hook, after the WithHooks ones.
func WithOnPredictStart(fns ...func(*PredictContext)) CallOption {
	return func(c *call) {
		c.hooked = true
		for _, fn := range fns {
			c.starts = append(c.starts, &HookFuncs{OnStart: fn})
		}
	}
}

// WithOnPredictEnd adds a function run as an OnPredictEnd hook, after every WithOnPredictStart.
func WithOnPredictEnd(fns ...func(*PredictContext)) CallOption {
	return func(c *call) {
		c.hooked = true
		for _, fn := range fns {
			c.ends = append(c.ends, &HookFuncs{OnEnd: fn})
		}
	}
}

// WithoutDefaultHooks leaves the process-wide defaults out of this call.
func WithoutDefaultHooks() CallOption {
	return func(c *call) {
		c.hooked = true
		c.skipDefault = true
	}
}

// checkHooks refuses a nil hook, Python's "entries must be instances" for the one mistake the
// type system leaves open.
func checkHooks(hooks []Hook) error {
	for i, h := range hooks {
		if hasNilValue(h) {
			return fmt.Errorf("agent: hook %d is nil", i)
		}
	}
	return nil
}

// --- registries ---

var defaults struct {
	mu    sync.RWMutex
	hooks []Hook
}

// SetDefaultHooks replaces the process-wide hooks every agent runs first.
func SetDefaultHooks(hooks ...Hook) error {
	if err := checkHooks(hooks); err != nil {
		return err
	}
	defaults.mu.Lock()
	defaults.hooks = append([]Hook(nil), hooks...)
	defaults.mu.Unlock()
	return nil
}

// AddDefaultHook appends to the process-wide hooks.
func AddDefaultHook(hooks ...Hook) error {
	if err := checkHooks(hooks); err != nil {
		return err
	}
	defaults.mu.Lock()
	defaults.hooks = append(defaults.hooks, hooks...)
	defaults.mu.Unlock()
	return nil
}

// ClearDefaultHooks removes every process-wide hook.
func ClearDefaultHooks() {
	defaults.mu.Lock()
	defaults.hooks = nil
	defaults.mu.Unlock()
}

// DefaultHooks returns a copy of the process-wide hooks.
func DefaultHooks() []Hook {
	defaults.mu.RLock()
	defer defaults.mu.RUnlock()
	return append([]Hook(nil), defaults.hooks...)
}

// hookState is the agent's hook registry and dispatch settings.
type hookState struct {
	mu        sync.RWMutex
	installed []Hook
	timeout   time.Duration
	raise     bool
	lock      *hookrun.Lock // nil when hooks run concurrently
	warn      func(string)
	done      chan struct{} // closed by Close
	closeOnce sync.Once

	// stub replaces inference in tests; nil in production.
	stub func(ctx context.Context, states []any, qs schema.Questions, c call) ([]schema.Prediction, error)
}

func newHookState(o options) *hookState {
	h := &hookState{installed: append([]Hook(nil), o.installed...), timeout: o.timeout, raise: o.raise,
		warn: o.warn, done: make(chan struct{})}
	if !o.concurrent {
		h.lock = new(hookrun.Lock)
	}
	if h.warn == nil {
		h.warn = func(msg string) { log.Print(msg) }
	}
	return h
}

func (h *hookState) quit() {
	if h == nil {
		return
	}
	h.closeOnce.Do(func() { close(h.done) })
}

// AddHook installs hooks on the agent, after those already there. A hook added twice runs twice.
func (a *Agent) AddHook(hooks ...Hook) error {
	if err := checkHooks(hooks); err != nil {
		return err
	}
	if a.hk == nil {
		return errors.New("agent: hooks need an Agent from Open")
	}
	a.hk.mu.Lock()
	a.hk.installed = append(a.hk.installed, hooks...)
	a.hk.mu.Unlock()
	return nil
}

// RemoveHook removes every installed copy of hook, compared by identity, and reports whether
// there was one. A hook whose dynamic type cannot be compared (a struct holding a func, used as
// a value) is never found; pass a pointer.
func (a *Agent) RemoveHook(hook Hook) bool {
	if a.hk == nil {
		return false
	}
	a.hk.mu.Lock()
	defer a.hk.mu.Unlock()
	kept := a.hk.installed[:0:0]
	for _, h := range a.hk.installed {
		if !sameHook(h, hook) {
			kept = append(kept, h)
		}
	}
	removed := len(kept) != len(a.hk.installed)
	a.hk.installed = kept
	return removed
}

// InstallHooks installs hooks until the returned function is called, which removes one copy of
// each (the most recent) and leaves everything else, including what was added meanwhile.
// Overlapping installs may be undone in either order.
func (a *Agent) InstallHooks(hooks ...Hook) (undo func(), err error) {
	if err := a.AddHook(hooks...); err != nil {
		return nil, err
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			a.hk.mu.Lock()
			defer a.hk.mu.Unlock()
			for _, h := range hooks {
				for i := len(a.hk.installed) - 1; i >= 0; i-- {
					if sameHook(a.hk.installed[i], h) {
						a.hk.installed = append(a.hk.installed[:i:i], a.hk.installed[i+1:]...)
						break
					}
				}
			}
		})
	}, nil
}

// Hooks returns a copy of the installed hooks.
func (a *Agent) Hooks() []Hook {
	if a.hk == nil {
		return nil
	}
	a.hk.mu.RLock()
	defer a.hk.mu.RUnlock()
	return append([]Hook(nil), a.hk.installed...)
}

func sameHook(a, b Hook) bool { return hookrun.Same(a, b) }

// composeHooks is the hooks of one call, in order.
func (a *Agent) composeHooks(c call) []Hook {
	var hooks []Hook
	if !c.skipDefault {
		hooks = DefaultHooks()
	}
	a.hk.mu.RLock()
	hooks = append(hooks, a.hk.installed...)
	a.hk.mu.RUnlock()
	hooks = append(hooks, c.hooks...)
	hooks = append(hooks, c.starts...)
	return append(hooks, c.ends...)
}

// --- dispatch ---

type hookEvent struct {
	name string // Python's event name
	call func(Hook, *PredictContext)
}

var (
	evStart = hookEvent{"on_predict_start", Hook.OnPredictStart}
	evEnd   = hookEvent{"on_predict_end", Hook.OnPredictEnd}
	evError = hookEvent{"on_error", Hook.OnError}
)

// dispatch runs one event over the hooks in order.
func (a *Agent) dispatch(ctx context.Context, ev hookEvent, hooks []Hook, pc *PredictContext) error {
	hk := a.hk
	run := hookrun.Run{Lock: hk.lock, ErrQuit: ErrClosed, Detached: ev.name != evStart.name,
		Policy: hookrun.Policy{Timeout: hk.timeout, Raise: hk.raise, Warn: hk.warn}}
	if ev.name == evStart.name {
		run.Quit = hk.done // OnError and OnPredictEnd report on work already done: they wait for the lock however the call ends
	}
	return run.Dispatch(ctx, ev.name, len(hooks), func(i int) string { return hookrun.Name(hooks[i]) },
		func(i int) { ev.call(hooks[i], pc) })
}

// attached is an error that keeps the message and identity of the first failure and carries a
// second one along, as Python's __context__ does: a failing OnError or OnPredictEnd never masks
// the error that caused it.
type attached struct{ first, second error }

func (e *attached) Error() string   { return e.first.Error() }
func (e *attached) Unwrap() []error { return []error{e.first, e.second} }

func newRunID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// predictHooked is PredictBatch with hooks, as ONNXAgent.predict_batch: start, inference unless a
// hook answered, then on any failure OnError, and always OnPredictEnd.
// inferStates is predictStates, or the test stub standing in for it.
func (a *Agent) inferStates(ctx context.Context, states []any, qs schema.Questions, c call) ([]schema.Prediction, error) {
	if a.hk != nil && a.hk.stub != nil {
		return a.hk.stub(ctx, states, qs, c)
	}
	return a.predictStates(ctx, states, qs, c)
}

func (a *Agent) predictHooked(ctx context.Context, states []any, qs schema.Questions, c call, g *gate, hooks []Hook) ([]schema.Prediction, error) {
	pc := &PredictContext{States: states, Questions: qs, RunID: newRunID(), Model: schema.Model, Agent: a, StartedAt: time.Now()}
	err := a.dispatch(ctx, evStart, hooks, pc)
	if err == nil && pc.Results == nil {
		c.maxLen, c.headMaxLen = pc.MaxLen, pc.HeadMaxLen
		pc.Results, err = a.inferStates(ctx, pc.States, pc.Questions, c)
	}
	if err != nil {
		pc.Err = err
		if err2 := a.dispatch(ctx, evError, hooks, pc); err2 != nil {
			err = &attached{err, err2}
		}
	}
	pc.ElapsedMS = float64(time.Since(pc.StartedAt)) / float64(time.Millisecond)
	if pc.Results != nil {
		pc.Usage = aggregateUsage(pc.Results)
		if g != nil {
			g.apply(pc.Results)
		}
	}
	if err2 := a.dispatch(ctx, evEnd, hooks, pc); err2 != nil {
		if err == nil {
			err = err2
		} else {
			err = &attached{err, err2}
		}
	}
	if err != nil {
		return nil, err
	}
	return pc.Results, nil
}
