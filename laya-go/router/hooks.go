package router

// Hooks are laya/hooks.py for the three events a Router raises: on_route, on_load and on_evict. A
// hook is any value with an OnRoute, OnLoad or OnEvict method; the same value may also be an
// agent.Hook, as in Python, where one object implements several events. OnLoad and OnEvict run
// after the Router's lock is released (see lifecycle.go), so a hook may call the Router. As in Python, an Unload that races a build of the same checkpoint
// can fire OnEvict before that build's OnLoad. Each call of Route
// dispatches over: the process-wide agent.DefaultHooks (which take only an agent.Hook, so a
// hook that is both embeds agent.BaseHook), the Router's installed hooks, then the
// call's ForHooks, and a hook without OnRoute is skipped.
//
// The dispatch is internal/hookrun, the one the agent uses, so ordering, WithHookErrors,
// WithHookTimeout and panic recovery behave as there. As in Python's Router, hooks run
// concurrently unless WithSerialHooks is given; with it, one hook runs at a time across all
// goroutines, and a hook must not call Route, or Load, Lease, Preload, Unload, UnloadAll or Close
// when they would fire a hook, on the Router that is running it (Python's lock is re-entrant, this
// one is not, so the call would wait for itself until a timeout). A hook is
// compared by identity, so pass a pointer: a value whose type cannot be compared is refused.
//
// A hook may replace ctx.Decision; the replacement is what Route returns. Route does not check
// it: a Model that names no checkpoint is the hook's business, and the caller that loads it
// sees the error. A hook that overruns WithHookTimeout is abandoned, not stopped (see
// agent/hooks.go). After a timeout the call gives its lease back, so the agent in an abandoned
// OnLoad hook's context may be closed while that hook still runs.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/hookrun"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// RouteHook is a hook that sees the routing decision.
type RouteHook interface {
	// OnRoute runs after the Router has decided. It may replace ctx.Decision.
	OnRoute(ctx *RouteContext)
}

// RouteContext is what an OnRoute hook sees of one Route call. The hooks of a call share it and
// run one after another, so a replacement made by one is seen by the next.
type RouteContext struct {
	// States holds the one state being routed; rewriting it changes nothing.
	States []any
	// Questions are the questions asked, as passed to Route.
	Questions schema.Questions
	// RunID is 32 hex characters, shared by every hook of the call. It is not the prediction's.
	RunID string
	// Decision is the routing decision; assign to replace it.
	Decision Decision
	// Router is the Router that is routing.
	Router *Router
}

// LoadHook is a hook that sees a checkpoint being built.
type LoadHook interface {
	// OnLoad runs after the Router built and published a checkpoint, once per build, after the
	// OnEvict calls of the checkpoints that build pushed out.
	OnLoad(ctx *LoadContext)
}

// EvictHook is a hook that sees a checkpoint leave the Router.
type EvictHook interface {
	// OnEvict runs after a checkpoint left the Router, by LRU eviction, Unload, UnloadAll or
	// Close. The agent may still be open: it closes when its last lease is released.
	OnEvict(ctx *EvictContext)
}

// LoadContext is what an OnLoad hook sees. The agent is the one the Router now serves, which for
// a build that lost a race with Attach is the attached one.
type LoadContext struct {
	// Model is the checkpoint that was built.
	Model Checkpoint
	// Agent is the agent serving Model. It can be evicted and closed at any moment; take a Lease
	// to use it.
	Agent *agent.Agent
	// RunID is 32 hex characters, new for each event.
	RunID string
	// Router is the Router that loaded.
	Router *Router
}

// EvictContext is what an OnEvict hook sees.
type EvictContext struct {
	// Model is the checkpoint that left.
	Model Checkpoint
	// RunID is 32 hex characters, new for each event.
	RunID string
	// Router is the Router it left.
	Router *Router
}

// BaseHook implements RouteHook, LoadHook and EvictHook with no-ops, for embedding.
type BaseHook struct{}

// OnRoute does nothing.
func (BaseHook) OnRoute(*RouteContext) {}

// OnLoad does nothing.
func (BaseHook) OnLoad(*LoadContext) {}

// OnEvict does nothing.
func (BaseHook) OnEvict(*EvictContext) {}

// HookFuncs is a RouteHook, LoadHook and EvictHook made of functions; nil is a no-op. Use it as
// *HookFuncs: a value holds funcs, which cannot be compared, so RemoveHook could never find it.
type HookFuncs struct {
	OnRouteFunc func(*RouteContext)
	OnLoadFunc  func(*LoadContext)
	OnEvictFunc func(*EvictContext)
}

// OnLoad calls OnLoadFunc.
func (f *HookFuncs) OnLoad(c *LoadContext) {
	if f.OnLoadFunc != nil {
		f.OnLoadFunc(c)
	}
}

// OnEvict calls OnEvictFunc.
func (f *HookFuncs) OnEvict(c *EvictContext) {
	if f.OnEvictFunc != nil {
		f.OnEvictFunc(c)
	}
}

// OnRoute calls OnRouteFunc.
func (f *HookFuncs) OnRoute(c *RouteContext) {
	if f.OnRouteFunc != nil {
		f.OnRouteFunc(c)
	}
}

// hookState is the Router's hook registry and dispatch settings.
type hookState struct {
	mu        sync.RWMutex
	installed []installedHook
	nextID    uint64
	timeout   time.Duration
	raise     bool
	lock      *hookrun.Lock
	warn      func(string)
}

// installedHook is one registry entry; id tells an InstallHooks undo which entries are its own.
type installedHook struct {
	hook any
	id   uint64
}

type hookConfig struct {
	installed []any
	timeout   time.Duration
	raise     bool
	serial    bool
	warn      func(string)
}

func newHookState(c hookConfig) *hookState {
	h := &hookState{timeout: c.timeout, raise: c.raise, warn: c.warn}
	for _, hook := range c.installed {
		h.nextID++
		h.installed = append(h.installed, installedHook{hook, h.nextID})
	}
	if c.serial {
		h.lock = new(hookrun.Lock)
	}
	return h
}

// checkHooks refuses a nil hook and one that implements no event, Python's "hooks entries must
// implement at least one of ...".
func checkHooks(hooks []any) error {
	for i, h := range hooks {
		if isNil(h) {
			return fmt.Errorf("router: hook %d is nil", i)
		}
		if !hookrun.Comparable(h) {
			return fmt.Errorf("router: hook %d (%T) cannot be compared, so it could never be removed; pass a pointer", i, h)
		}
		_, route := h.(RouteHook)
		_, load := h.(LoadHook)
		_, evict := h.(EvictHook)
		_, predict := h.(agent.Hook)
		if !route && !load && !evict && !predict {
			return fmt.Errorf("router: hook %d (%T) implements none of OnRoute, OnLoad, OnEvict or agent.Hook", i, h)
		}
	}
	return nil
}

func isNil(v any) bool {
	if v == nil {
		return true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan:
		return rv.IsNil()
	}
	return false
}

// WithInstalledHooks installs hooks on the Router, as AddHook does after New.
func WithInstalledHooks(hooks ...any) Option {
	return func(c *config) error {
		if err := checkHooks(hooks); err != nil {
			return err
		}
		c.hooks.installed = append(c.hooks.installed, hooks...)
		return nil
	}
}

// WithHookTimeout bounds the wait for each hook. d must be positive; leave the option out for no
// limit.
func WithHookTimeout(d time.Duration) Option {
	return func(c *config) error {
		if err := hookrun.ValidateTimeout(d.Seconds()); err != nil {
			return err
		}
		c.hooks.timeout = d
		return nil
	}
}

// WithHookErrors chooses what a failing hook does: raise true (the default) makes Route return
// its error, false warns and goes on.
func WithHookErrors(raise bool) Option {
	return func(c *config) error { c.hooks.raise = raise; return nil }
}

// WithSerialHooks runs one hook at a time across all calls, for hooks that are not safe to run
// concurrently; the default, as in Python, lets hooks of concurrent calls overlap. A serial hook
// must not call Route, or a Load that builds or evicts, on its own Router: the lock is not re-entrant and the call would wait for
// itself.
func WithSerialHooks() Option {
	return func(c *config) error { c.hooks.serial = true; return nil }
}

// WithHookWarnings sets where a hook failure that does not fail the call is reported; the
// default is the log package.
func WithHookWarnings(sink func(msg string)) Option {
	return func(c *config) error { c.hooks.warn = sink; return nil }
}

// AddHook installs hooks after those already there. A hook added twice runs twice.
func (r *Router) AddHook(hooks ...any) error {
	_, err := r.add(hooks)
	return err
}

// add installs hooks and returns the ids of the new entries.
func (r *Router) add(hooks []any) ([]uint64, error) {
	if err := checkHooks(hooks); err != nil {
		return nil, err
	}
	r.hk.mu.Lock()
	defer r.hk.mu.Unlock()
	ids := make([]uint64, len(hooks))
	for i, h := range hooks {
		r.hk.nextID++
		ids[i] = r.hk.nextID
		r.hk.installed = append(r.hk.installed, installedHook{h, r.hk.nextID})
	}
	return ids, nil
}

// RemoveHook removes every installed copy of hook, compared by identity, and reports whether
// there was one.
func (r *Router) RemoveHook(hook any) bool {
	r.hk.mu.Lock()
	defer r.hk.mu.Unlock()
	kept := r.hk.installed[:0:0]
	for _, e := range r.hk.installed {
		if !hookrun.Same(e.hook, hook) {
			kept = append(kept, e)
		}
	}
	removed := len(kept) != len(r.hk.installed)
	r.hk.installed = kept
	return removed
}

// InstallHooks installs hooks until the returned function is called, which removes exactly the
// entries this call added and leaves everything else, including equal hooks added meanwhile or
// before. Overlapping installs may be undone in either order.
func (r *Router) InstallHooks(hooks ...any) (undo func(), err error) {
	ids, err := r.add(hooks)
	if err != nil {
		return nil, err
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			own := make(map[uint64]bool, len(ids))
			for _, id := range ids {
				own[id] = true
			}
			r.hk.mu.Lock()
			defer r.hk.mu.Unlock()
			kept := r.hk.installed[:0:0]
			for _, e := range r.hk.installed {
				if !own[e.id] {
					kept = append(kept, e)
				}
			}
			r.hk.installed = kept
		})
	}, nil
}

// Hooks returns a copy of the installed hooks.
func (r *Router) Hooks() []any {
	r.hk.mu.RLock()
	defer r.hk.mu.RUnlock()
	out := make([]any, len(r.hk.installed))
	for i, e := range r.hk.installed {
		out[i] = e.hook
	}
	return out
}

// composeHooks is the hooks of one call, in order: defaults, installed, the call's.
func (r *Router) composeHooks(c call) []any {
	var hooks []any
	for _, h := range agent.DefaultHooks() {
		hooks = append(hooks, h)
	}
	r.hk.mu.RLock()
	for _, e := range r.hk.installed {
		hooks = append(hooks, e.hook)
	}
	r.hk.mu.RUnlock()
	return append(hooks, c.hooks...)
}

// ForHooks adds hooks to one request, after the defaults and the installed ones.
func ForHooks(hooks ...any) CallOption {
	return func(c *call) {
		if err := checkHooks(hooks); err != nil && c.err == nil {
			c.err = err
		}
		c.hooks = append(c.hooks, hooks...)
	}
}

// ForHookErrors overrides WithHookErrors for this request.
func ForHookErrors(raise bool) CallOption {
	return func(c *call) { c.raise = &raise }
}

// ForHookTimeout overrides WithHookTimeout for this request; d must be positive.
func ForHookTimeout(d time.Duration) CallOption {
	return func(c *call) { c.timeout = &d }
}

func newRunID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// dispatchRoute runs OnRoute over the hooks of c and returns the decision they leave.
func (r *Router) dispatchRoute(state any, qs schema.Questions, d Decision, c call) (Decision, error) {
	policy := hookrun.Policy{Timeout: r.hk.timeout, Raise: r.hk.raise, Warn: r.hk.warn}
	if c.raise != nil {
		policy.Raise = *c.raise
	}
	if c.timeout != nil {
		if err := hookrun.ValidateTimeout(c.timeout.Seconds()); err != nil {
			return d, err
		}
		policy.Timeout = *c.timeout
	}
	var hooks []RouteHook
	var all []any
	for _, h := range r.composeHooks(c) {
		if rh, ok := h.(RouteHook); ok {
			hooks, all = append(hooks, rh), append(all, h)
		}
	}
	if len(hooks) == 0 {
		return d, nil
	}
	rc := &RouteContext{States: []any{state}, Questions: qs, RunID: newRunID(), Decision: d, Router: r}
	run := hookrun.Run{Lock: r.hk.lock, Policy: policy}
	err := run.Dispatch(context.Background(), "on_route", len(hooks),
		func(i int) string { return hookrun.Name(all[i]) }, func(i int) { hooks[i].OnRoute(rc) })
	if err != nil {
		return d, err
	}
	return rc.Decision, nil
}

// dispatchLifecycle runs one lifecycle event over the hooks that implement it, with the Router's
// own policy; a call has no overrides. pick returns the hook's call, or nil when it lacks the
// event. It must be called with no lock of the lifecycle held.
func (r *Router) dispatchLifecycle(event string, pick func(h any) func()) error {
	var calls []func()
	var all []any
	for _, h := range r.composeHooks(call{}) {
		if f := pick(h); f != nil {
			calls, all = append(calls, f), append(all, h)
		}
	}
	if len(calls) == 0 {
		return nil
	}
	run := hookrun.Run{Lock: r.hk.lock, Policy: hookrun.Policy{Timeout: r.hk.timeout, Raise: r.hk.raise, Warn: r.hk.warn}}
	return run.Dispatch(context.Background(), event, len(calls),
		func(i int) string { return hookrun.Name(all[i]) }, func(i int) { calls[i]() })
}

func (r *Router) dispatchLoad(key Checkpoint, a *agent.Agent) error {
	ctx := &LoadContext{Model: key, Agent: a, RunID: newRunID(), Router: r}
	return r.dispatchLifecycle("on_load", func(h any) func() {
		if lh, ok := h.(LoadHook); ok {
			return func() { lh.OnLoad(ctx) }
		}
		return nil
	})
}

func (r *Router) dispatchEvict(key Checkpoint) error {
	ctx := &EvictContext{Model: key, RunID: newRunID(), Router: r}
	return r.dispatchLifecycle("on_evict", func(h any) func() {
		if eh, ok := h.(EvictHook); ok {
			return func() { eh.OnEvict(ctx) }
		}
		return nil
	})
}
