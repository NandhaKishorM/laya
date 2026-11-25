package router

// The model lifecycle is laya/router.py's load, attach, preload, unload and eviction, with one
// addition Python does not need: a Lease. Python frees an agent by dropping its reference and lets
// the garbage collector decide when; an Agent here owns a native session that Close frees, so
// eviction must not close an agent a caller is still predicting with. A slot counts its leases,
// and an agent the Router built closes when it has left the Router and its last lease is
// released.
//
// Locking, as in Python: mu guards the slots, the LRU order, the in-flight builds and the flags.
// It is never held while a factory runs, while an agent closes, or while a hook runs. build
// serialises factory calls across checkpoints, so two cold loads never hold two checkpoints in
// flight at once. A load of a resident checkpoint never touches build.

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/NandhaKishorM/laya/laya-go/agent"
)

// DefaultMaxLoaded is how many checkpoints a Router keeps resident unless WithMaxLoaded says
// otherwise.
const DefaultMaxLoaded = 2

// ErrClosed is returned by a Router that has been closed.
var ErrClosed = errors.New("router: closed")

// AgentFactory builds the agent for a checkpoint. It owns where the weights come from, the
// device and the agent options. It is called at most once at a time per Router and never with a
// lock held, and ctx is the context of the call that triggered the build. The Router closes the
// agent it returns, so a factory must return a fresh agent on every call, never one it shares. A
// factory must not call Load, Lease, Preload, Unload, UnloadAll or Close on the Router that is
// calling it: it holds the build token, so a call that builds, or waits for a build, waits for
// itself.
type AgentFactory func(ctx context.Context, c Checkpoint) (*agent.Agent, error)

// WithAgentFactory lets the Router load checkpoints. Without it Route works and every load fails.
func WithAgentFactory(f AgentFactory) Option {
	return func(c *config) error { c.factory = f; return nil }
}

// WithMaxLoaded sets how many checkpoints stay resident before the least recently used is
// evicted; n must be at least 1. Attach and Preload raise it to fit what they make resident,
// and never lower it.
func WithMaxLoaded(n int) Option {
	return func(c *config) error {
		if n < 1 {
			return fmt.Errorf("router: max loaded must be at least 1, got %d", n)
		}
		c.maxLoaded = n
		return nil
	}
}

type slot struct {
	agent   *agent.Agent
	leases  int
	retired bool // left the Router
	own     *own // non-nil when the Router built the agent; Attach clears it
}

// own is the Router's claim to close an agent it built. It is shared by every slot of that agent,
// so attaching the agent hands it to the caller even while an earlier, retired slot is leased.
type own struct{ router bool }

func (s *slot) owns() bool { return s.own != nil && s.own.router }

type inflight struct {
	done chan struct{}
	err  error // set before done closes
}

type lifecycle struct {
	factory AgentFactory
	build   chan struct{} // one token: serialises factory calls

	mu      sync.Mutex
	slots   map[Checkpoint]*slot
	order   []Checkpoint // least recently used first
	loading map[Checkpoint]*inflight
	max     int
	closed  bool
	owners  map[*agent.Agent]*own // agents the Router built and has not closed
}

func (l *lifecycle) init(f AgentFactory, max int) {
	l.factory, l.max = f, max
	l.build = make(chan struct{}, 1)
	l.slots = map[Checkpoint]*slot{}
	l.loading = map[Checkpoint]*inflight{}
	l.owners = map[*agent.Agent]*own{}
}

// outcome is what one acquire did, so its hooks fire after the locks are released.
type outcome struct {
	built    bool
	agent    *agent.Agent
	evicted  []Checkpoint
	closeErr error // closing what the build pushed out failed; the build still happened
}

// touch moves key to the most recently used end. mu is held.
func (l *lifecycle) touch(key Checkpoint) {
	for i, k := range l.order {
		if k == key {
			l.order = append(l.order[:i], l.order[i+1:]...)
			break
		}
	}
	l.order = append(l.order, key)
}

// retire marks s as gone from the Router and collects its agent for closing when nothing is using
// it and the Router built it. mu is held.
func (l *lifecycle) retire(s *slot, toClose *[]*agent.Agent) {
	s.retired = true
	if s.leases <= 0 && s.owns() {
		delete(l.owners, s.agent)
		*toClose = append(*toClose, s.agent)
	}
}

// evictLocked drops least recently used slots until max holds and returns who left. mu is held.
func (l *lifecycle) evictLocked(toClose *[]*agent.Agent) []Checkpoint {
	var evicted []Checkpoint
	for len(l.order) > l.max {
		victim := l.order[0]
		l.order = l.order[1:]
		if s := l.slots[victim]; s != nil {
			delete(l.slots, victim)
			evicted = append(evicted, victim)
			l.retire(s, toClose)
		}
	}
	return evicted
}

// safeClose closes a, turning a panic in its session's Close into an error so one bad agent
// cannot skip the others or the events that follow.
func safeClose(a *agent.Agent) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("router: closing an agent panicked: %v", r)
		}
	}()
	return a.Close()
}

func closeAll(agents []*agent.Agent) error {
	var errs []error
	for _, a := range agents {
		if err := safeClose(a); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// release gives back one lease and closes the agent when it has left and this was the last.
func (l *lifecycle) release(s *slot) error {
	var toClose *agent.Agent
	l.mu.Lock()
	s.leases--
	if s.leases <= 0 && s.retired && s.owns() {
		toClose = s.agent
		delete(l.owners, s.agent)
	}
	l.mu.Unlock()
	if toClose != nil {
		return safeClose(toClose)
	}
	return nil
}

func isContextErr(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// acquire returns the slot for key with one lease taken, building it if needed. Concurrent
// callers for one key share a single build; a caller whose build failed returns the error and
// caches nothing, and the others get the same error. A waiter does not inherit the builder's
// cancellation: it builds for itself.
func (l *lifecycle) acquire(ctx context.Context, key Checkpoint, out *outcome) (*slot, error) {
	for {
		l.mu.Lock()
		if l.closed {
			l.mu.Unlock()
			return nil, ErrClosed
		}
		if s := l.slots[key]; s != nil { // not behind build: a resident load never waits for a cold one
			l.touch(key)
			s.leases++
			l.mu.Unlock()
			return s, nil
		}
		wait := l.loading[key]
		var mine *inflight
		if wait == nil {
			mine = &inflight{done: make(chan struct{})}
			l.loading[key] = mine
		}
		l.mu.Unlock()
		if mine != nil {
			return l.doBuild(ctx, key, mine, out)
		}
		select {
		case <-wait.done:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		if wait.err != nil {
			if isContextErr(wait.err) && ctx.Err() == nil {
				continue
			}
			return nil, fmt.Errorf("router: loading %s failed on the call that was building it: %w", key, wait.err)
		}
	}
}

func (l *lifecycle) create(ctx context.Context, key Checkpoint) (*agent.Agent, error) {
	if l.factory == nil {
		return nil, errors.New("router: this Router can route but not load: no agent factory was given; use WithAgentFactory")
	}
	a, err := l.factory(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("router: cannot load the %s checkpoint: %w", key, err)
	}
	if a == nil {
		return nil, fmt.Errorf("router: the agent factory returned no agent for %s", key)
	}
	return a, nil
}

// doBuild is the build this caller owns: mine is the handle the others wait on.
func (l *lifecycle) doBuild(ctx context.Context, key Checkpoint, mine *inflight, out *outcome) (built *slot, err error) {
	var toClose []*agent.Agent
	finish := func() { // mu is held; runs once: nothing after it can panic (closes are safeClose)
		if l.loading[key] == mine {
			delete(l.loading, key)
		}
		close(mine.done)
	}
	defer func() {
		rec := recover()
		if rec != nil {
			err = fmt.Errorf("router: building %s panicked: %v", key, rec)
		}
		if err != nil {
			built = nil
			l.mu.Lock()
			mine.err = err
			finish()
			l.mu.Unlock()
		}
		if rec != nil {
			panic(rec)
		}
	}()
	select {
	case l.build <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	released := false
	releaseBuild := func() {
		if !released {
			released = true
			<-l.build
		}
	}
	defer releaseBuild()

	l.mu.Lock()
	if l.closed { // Close came while this caller waited for the token: do not start a build
		l.mu.Unlock()
		return nil, ErrClosed
	}
	if s := l.slots[key]; s != nil { // built or attached while this caller waited for the build token
		l.touch(key)
		s.leases++
		finish()
		l.mu.Unlock()
		return s, nil
	}
	l.mu.Unlock()

	a, err := l.create(ctx, key)
	if err != nil {
		return nil, err
	}

	l.mu.Lock()
	if s := l.slots[key]; s != nil { // attached while it was building: keep that one
		l.touch(key)
		s.leases++
		built = s
		toClose = append(toClose, a)
		out.built, out.agent = true, s.agent
	} else {
		o := &own{router: true}
		l.owners[a] = o
		built = &slot{agent: a, own: o, leases: 1}
		l.slots[key] = built
		l.order = append(l.order, key)
		out.evicted = l.evictLocked(&toClose)
		out.built, out.agent = true, a
	}
	finish()
	l.mu.Unlock()
	releaseBuild()
	// Outside every lock: closing a native session can be slow.
	out.closeErr = closeAll(toClose)
	return built, nil
}

// take acquires a lease for key and settles what the build owes: a failed close of an evicted
// agent fails the call, but the events are still delivered.
func (r *Router) take(ctx context.Context, key Checkpoint) (*slot, *outcome, error) {
	out := &outcome{}
	s, err := r.lc.acquire(ctx, key, out)
	if err != nil {
		return nil, nil, err
	}
	if out.closeErr != nil {
		err := out.closeErr
		if rerr := r.lc.release(s); rerr != nil {
			err = errors.Join(err, rerr)
		}
		if herr := r.dispatchOutcome(key, out); herr != nil {
			err = errors.Join(err, herr)
		}
		return nil, nil, err
	}
	return s, out, nil
}

// dispatchOutcome fires the evictions a build caused, then its load, stopping at the first hook
// error as Python does. It must be called with no lock held.
func (r *Router) dispatchOutcome(key Checkpoint, out *outcome) error {
	for _, ev := range out.evicted {
		if err := r.dispatchEvict(ev); err != nil {
			return err
		}
	}
	if out.built {
		return r.dispatchLoad(key, out.agent)
	}
	return nil
}

// Lease is a borrowed agent. The agent stays open until Close is called, however many times the
// Router evicts or unloads it meanwhile. Close it when done; the zero Lease holds nothing.
type Lease struct {
	r        *Router
	s        *slot
	key      Checkpoint
	released atomic.Bool
}

// Checkpoint is the checkpoint the lease was taken for.
func (l *Lease) Checkpoint() Checkpoint { return l.key }

// Agent is the leased agent, or nil once the lease is closed.
func (l *Lease) Agent() *agent.Agent {
	if l == nil || l.s == nil || l.released.Load() {
		return nil
	}
	return l.s.agent
}

// Close gives the lease back; when the agent has left the Router and this was its last lease, the
// agent closes and Close returns its error. Closing twice is safe: only the first counts.
func (l *Lease) Close() error {
	if l == nil || l.s == nil || !l.released.CompareAndSwap(false, true) {
		return nil
	}
	return l.r.lc.release(l.s)
}

// Lease borrows the agent for a checkpoint (a name or alias), building it on first use. OnEvict
// and OnLoad hooks have run when it returns, with the lease already held, so the agent is never
// closed under them. If a hook fails the lease is given back and the error returned, and the
// checkpoint stays resident.
func (r *Router) Lease(ctx context.Context, name string) (*Lease, error) {
	key, err := NormaliseName(name)
	if err != nil {
		return nil, err
	}
	s, out, err := r.take(ctx, key)
	if err != nil {
		return nil, err
	}
	if err := r.dispatchOutcome(key, out); err != nil {
		return nil, errors.Join(err, r.lc.release(s))
	}
	return &Lease{r: r, s: s, key: key}, nil
}

// Load returns the agent for a checkpoint, building it on first use. Concurrent callers share one
// build. The agent is valid only while it stays resident: eviction and Unload close an agent the
// Router built, so a reference held across either is a closed session. Use Lease, which holds the
// agent open, unless the caller manages the lifetime itself. If an OnLoad hook unloads, or closes
// the Router, for the same checkpoint, Load returns an agent that is already closed and no error. If a hook fails Load returns the
// error and no agent, and the checkpoint stays resident.
func (r *Router) Load(ctx context.Context, name string) (*agent.Agent, error) {
	key, err := NormaliseName(name)
	if err != nil {
		return nil, err
	}
	s, out, err := r.take(ctx, key)
	if err != nil {
		return nil, err
	}
	// The lease is held across the hooks, so a concurrent eviction cannot close ctx.Agent under
	// OnLoad; it is given back before Load returns.
	herr := r.dispatchOutcome(key, out)
	if rerr := r.lc.release(s); herr != nil || rerr != nil {
		return nil, errors.Join(herr, rerr)
	}
	return s.agent, nil
}

// Attach makes an agent the caller built resident under a checkpoint name, instead of loading a
// second copy. The Router never closes an attached agent: the caller keeps ownership. It does not
// evict; MaxLoaded is raised to fit everything resident. An agent already resident under the name
// is replaced, and closes when its last lease is released if the Router built it. Attaching the
// agent already there hands it to the caller: the Router stops owning it and never closes it. Python registers an unknown name on the spot; the
// names here are fixed, so an unknown one is an *UnknownModelError. No hook runs.
func (r *Router) Attach(name string, a *agent.Agent) error {
	key, err := NormaliseName(name)
	if err != nil {
		return err
	}
	if a == nil {
		return errors.New("router: Attach needs an agent")
	}
	l := &r.lc
	var toClose []*agent.Agent
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return ErrClosed
	}
	prev := l.slots[key]
	if o := l.owners[a]; o != nil { // built here, perhaps evicted but still leased: now the caller's
		o.router = false
		delete(l.owners, a)
	}
	l.slots[key] = &slot{agent: a}
	if prev != nil && prev.agent != a { // the same agent changes hands, it is not closed
		l.retire(prev, &toClose)
	}
	l.touch(key)
	if len(l.slots) > l.max {
		l.max = len(l.slots)
	}
	l.mu.Unlock()
	return closeAll(toClose)
}

// Preload builds checkpoints up front so no request pays a cold load; with no names, all of them.
// MaxLoaded is raised to fit both the checkpoints asked for and everything resident, so staged
// preloading evicts neither. A resident checkpoint is left where it is in the LRU order. It stops
// at the first failure.
func (r *Router) Preload(ctx context.Context, names ...string) error {
	keys := Checkpoints()
	if len(names) > 0 {
		keys = keys[:0:0]
		for _, n := range names {
			key, err := NormaliseName(n)
			if err != nil {
				return err
			}
			keys = append(keys, key)
		}
	}
	l := &r.lc
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return ErrClosed
	}
	union := map[Checkpoint]bool{}
	for _, k := range keys {
		union[k] = true
	}
	for k := range l.slots {
		union[k] = true
	}
	if len(union) > l.max {
		l.max = len(union)
	}
	l.mu.Unlock()
	for _, key := range keys {
		l.mu.Lock()
		already := l.slots[key] != nil
		l.mu.Unlock()
		if already {
			continue
		}
		s, out, err := r.take(ctx, key)
		if err != nil {
			return err
		}
		herr := r.dispatchOutcome(key, out) // the lease is held across the hooks, as in Load
		if rerr := l.release(s); herr != nil || rerr != nil {
			return errors.Join(herr, rerr)
		}
	}
	return nil
}

// evictAll fires OnEvict for each freed checkpoint, stopping at the first hook error, after the
// agents that could close have. The events are owed whether or not a close failed.
func (r *Router) evictAll(freed []Checkpoint, toClose []*agent.Agent) error {
	cerr := closeAll(toClose)
	for _, k := range freed {
		if err := r.dispatchEvict(k); err != nil {
			return errors.Join(cerr, err)
		}
	}
	return cerr
}

// Unload frees one checkpoint (a name or alias) and returns it if it was resident. It waits for
// an in-flight build of that checkpoint only. The slot goes at once; its agent closes when the
// last lease is released. OnEvict fires for a checkpoint freed.
func (r *Router) Unload(ctx context.Context, name string) ([]Checkpoint, error) {
	key, err := NormaliseName(name)
	if err != nil {
		return nil, err
	}
	l := &r.lc
	for {
		var toClose []*agent.Agent
		var freed []Checkpoint
		l.mu.Lock()
		wait := l.loading[key]
		if wait == nil {
			if s := l.slots[key]; s != nil {
				delete(l.slots, key)
				freed = append(freed, key)
				l.retire(s, &toClose)
			}
			l.removeOrder(key)
		}
		l.mu.Unlock()
		if wait == nil {
			return freed, r.evictAll(freed, toClose)
		}
		select {
		case <-wait.done:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// removeOrder drops key from the LRU order. mu is held.
func (l *lifecycle) removeOrder(key Checkpoint) {
	for i, k := range l.order {
		if k == key {
			l.order = append(l.order[:i], l.order[i+1:]...)
			return
		}
	}
}

// UnloadAll frees every checkpoint, least recently used first, after any build in flight has
// finished, and returns who was freed.
func (r *Router) UnloadAll(ctx context.Context) ([]Checkpoint, error) {
	l := &r.lc
	for {
		var toClose []*agent.Agent
		var freed []Checkpoint
		var waits []*inflight
		l.mu.Lock()
		for _, w := range l.loading {
			waits = append(waits, w)
		}
		if len(waits) == 0 {
			freed = append(freed, l.order...)
			for _, k := range l.order {
				if s := l.slots[k]; s != nil {
					l.retire(s, &toClose)
				}
			}
			l.slots = map[Checkpoint]*slot{}
			l.order = nil
		}
		l.mu.Unlock()
		if len(waits) == 0 {
			return freed, r.evictAll(freed, toClose)
		}
		for _, w := range waits {
			select {
			case <-w.done:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}
}

// Loaded lists the resident checkpoints, least recently used first.
func (r *Router) Loaded() []Checkpoint {
	r.lc.mu.Lock()
	defer r.lc.mu.Unlock()
	return append([]Checkpoint(nil), r.lc.order...)
}

// MaxLoaded is how many checkpoints the Router keeps before it evicts.
func (r *Router) MaxLoaded() int {
	r.lc.mu.Lock()
	defer r.lc.mu.Unlock()
	return r.lc.max
}

// Close seals the Router and frees every checkpoint. A build in flight finishes first, and its
// caller gets the lease it asked for; that agent closes when the lease is released. An agent
// under lease closes at its last release, an attached agent is left to its owner, and Close is
// idempotent. Loads after Close return ErrClosed; Route still works.
func (r *Router) Close() error {
	r.lc.mu.Lock()
	r.lc.closed = true
	r.lc.mu.Unlock()
	_, err := r.UnloadAll(context.Background())
	return err
}
