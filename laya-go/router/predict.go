package router

// Routed inference: Router.Predict, PredictRouted, PredictBatch and PredictLong are
// Router.predict, predict_batch and predict_long of laya/router.py.
//
// A call is validated before anything is routed or loaded, so a malformed call never builds a
// checkpoint. The agent is borrowed with a Lease and the lease is held for the whole call, end
// hooks included, so an eviction by another goroutine cannot close it under the call.
//
// Router-level predict hooks are the agent.Hook values among the Router's hooks (defaults,
// installed, the call's ForHooks). They wrap route, load and inference as one call and see the
// decision in PredictContext.Decision (a router.Decision) and the Router in PredictContext.Router.
// The agent that runs the request is called without the process-wide default hooks, so a default
// hook fires once per request, at the Router. Hooks installed on the agent itself still fire.
//
// Not ported: Decide and DecideBatch (Go has no typed-decisions package yet), and
// predict_batch's sort diagnostics.

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/hookrun"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// Routed is a prediction and the decision that produced it: laya's result dict with its "routing"
// key.
type Routed struct {
	Prediction schema.Prediction
	// Routing is the decision the request was routed with, before any start hook rewrote
	// PredictContext.Decision.
	Routing Decision
	// Warnings are the texts PredictLong's scan raised for a window or stride it reduced.
	Warnings []string
}

// ForMaxLen overrides the routed checkpoint's sequence budget for this request; 0 or less is no
// override. A start hook may replace it through PredictContext.MaxLen. PredictLong ignores it.
func ForMaxLen(tokens int) CallOption { return func(c *call) { c.maxLen = tokens } }

// ForHeadMaxLen overrides the question half of the budget for this request; 0 or less is no
// override. PredictLong ignores it.
func ForHeadMaxLen(tokens int) CallOption { return func(c *call) { c.headMaxLen = tokens } }

// ForMinConfidence gates every answer at one threshold, as agent.WithMinConfidence does. The
// Router applies the gate to what it returns, after the start hooks and before the end hooks. An
// invalid threshold fails the call before anything is routed. PredictLong takes no gate.
func ForMinConfidence(threshold float64) CallOption {
	return func(c *call) { c.gate = []agent.CallOption{agent.WithMinConfidence(threshold)} }
}

// ForMinConfidenceMap is ForMinConfidence with a threshold per bucket, as
// agent.WithMinConfidenceMap.
func ForMinConfidenceMap(thresholds map[string]float64) CallOption {
	return func(c *call) { c.gate = []agent.CallOption{agent.WithMinConfidenceMap(thresholds)} }
}

// ForWindow sets the tokens per window of a PredictLong scan; 0 or less is the checkpoint's budget.
func ForWindow(tokens int) CallOption { return func(c *call) { c.window = tokens } }

// ForStride sets the tokens a PredictLong scan steps between windows; 0 or less is half the window.
func ForStride(tokens int) CallOption { return func(c *call) { c.stride = tokens } }

// ForBatchSize caps the windows per run of a PredictLong scan; 0 or less leaves it to the scan.
// PredictBatch takes its batch size as an argument instead.
func ForBatchSize(n int) CallOption { return func(c *call) { c.batch = n } }

// ForSortByLength asks PredictBatch to pad less by ordering states by length, as
// agent.WithSortByLength. It applies to the whole batch.
func ForSortByLength(on bool) CallOption { return func(c *call) { c.sort = on } }

// AgentCall is what the Router asks of an agent for one forward pass.
type AgentCall struct {
	Lang                               string
	Batch                              int
	Sort                               bool
	MaxLen, HeadMaxLen, Window, Stride int
}

// runner is the Router's only contact with an agent's inference, a seam for tests.
type runner interface {
	predict(ctx context.Context, a *agent.Agent, state any, qs schema.Questions, ac AgentCall) (schema.Prediction, error)
	predictBatch(ctx context.Context, a *agent.Agent, states []any, qs schema.Questions, ac AgentCall) ([]schema.Prediction, error)
	predictLong(ctx context.Context, a *agent.Agent, state any, qs schema.Questions, ac AgentCall) (agent.LongPrediction, error)
}

type agentRunner struct{}

// opts are the call options of a Predict or PredictBatch: no default hooks (the Router has run
// them), and each override only when set, as laya passes them.
func (ac AgentCall) opts() []agent.CallOption {
	o := []agent.CallOption{agent.WithoutDefaultHooks()}
	if ac.Lang != "" {
		o = append(o, agent.WithLanguage(ac.Lang))
	}
	if ac.Batch > 0 {
		o = append(o, agent.WithBatchSize(ac.Batch))
	}
	if ac.Sort {
		o = append(o, agent.WithSortByLength(true))
	}
	if ac.MaxLen > 0 {
		o = append(o, agent.WithMaxLen(ac.MaxLen))
	}
	if ac.HeadMaxLen > 0 {
		o = append(o, agent.WithHeadMaxLen(ac.HeadMaxLen))
	}
	return o
}

func (agentRunner) predict(ctx context.Context, a *agent.Agent, state any, qs schema.Questions, ac AgentCall) (schema.Prediction, error) {
	return a.Predict(ctx, state, qs, ac.opts()...)
}

func (agentRunner) predictBatch(ctx context.Context, a *agent.Agent, states []any, qs schema.Questions, ac AgentCall) ([]schema.Prediction, error) {
	return a.PredictBatch(ctx, states, qs, ac.opts()...)
}

// predictLong passes no hook option: the agent's scan runs no hooks and refuses them.
func (agentRunner) predictLong(ctx context.Context, a *agent.Agent, state any, qs schema.Questions, ac AgentCall) (agent.LongPrediction, error) {
	var o []agent.CallOption
	if ac.Lang != "" {
		o = append(o, agent.WithLanguage(ac.Lang))
	}
	if ac.Batch > 0 {
		o = append(o, agent.WithBatchSize(ac.Batch))
	}
	if ac.Window > 0 {
		o = append(o, agent.WithWindow(ac.Window))
	}
	if ac.Stride > 0 {
		o = append(o, agent.WithStride(ac.Stride))
	}
	return a.PredictLong(ctx, state, qs, o...)
}

func (r *Router) runner() runner {
	if r.rn != nil {
		return r.rn
	}
	return agentRunner{}
}

// scanLong is the scan PredictLong asks the shared predict path to run instead of Predict.
type scanLong struct {
	window, stride, batch int
	lang                  *string
}

// nilState is an untyped nil or a nil pointer: Python's None. A nil slice or map is an empty list or dict.
func nilState(state any) bool {
	if state == nil {
		return true
	}
	rv := reflect.ValueOf(state)
	return rv.Kind() == reflect.Pointer && rv.IsNil()
}

// errNilState is laya's text for a missing state.
var errNilState = errors.New("state must not be None; pass a string, dict, or list")

// attached keeps the first failure's message and identity and carries a second along, as Python's
// __context__ does: a failing hook never masks the error that caused it.
type attached struct{ first, second error }

func (e *attached) Error() string   { return e.first.Error() }
func (e *attached) Unwrap() []error { return []error{e.first, e.second} }

func join(first, second error) error {
	if second == nil {
		return first
	}
	if first == nil {
		return second
	}
	return &attached{first, second}
}

// policy is the hook policy of a call: the Router's, with the call's overrides.
func (r *Router) policy(c call) (hookrun.Policy, error) {
	p := hookrun.Policy{Timeout: r.hk.timeout, Raise: r.hk.raise, Warn: r.hk.warn}
	if c.raise != nil {
		p.Raise = *c.raise
	}
	if c.timeout != nil {
		if err := hookrun.ValidateTimeout(c.timeout.Seconds()); err != nil {
			return p, err
		}
		p.Timeout = *c.timeout
	}
	return p, nil
}

// predictHooks are the agent.Hook values among the hooks of c.
func (r *Router) predictHooks(c call) []agent.Hook {
	var out []agent.Hook
	for _, h := range r.composeHooks(c) {
		if ah, ok := h.(agent.Hook); ok {
			out = append(out, ah)
		}
	}
	return out
}

const (
	evStart = "on_predict_start"
	evEnd   = "on_predict_end"
	evError = "on_error"
)

func (r *Router) fire(ctx context.Context, ev string, hooks []agent.Hook, pc *agent.PredictContext, pol hookrun.Policy) error {
	if len(hooks) == 0 {
		return nil
	}
	run := hookrun.Run{Lock: r.hk.lock, Detached: ev != evStart, Policy: pol}
	return run.Dispatch(ctx, ev, len(hooks), func(i int) string { return hookrun.Name(hooks[i]) }, func(i int) {
		switch ev {
		case evStart:
			hooks[i].OnPredictStart(pc)
		case evEnd:
			hooks[i].OnPredictEnd(pc)
		default:
			hooks[i].OnError(pc)
		}
	})
}

func sumUsage(rs []schema.Prediction) agent.HookUsage {
	var u agent.HookUsage
	for _, p := range rs {
		u.InputTokens += p.Usage.InputTokens
		u.OutputTokens += p.Usage.OutputTokens
	}
	return u
}

func applyCalls(opts []CallOption) call {
	var c call
	for _, o := range opts {
		if o != nil {
			o(&c)
		}
	}
	return c
}

// detectedLang is the language the detection named, or "".
func detectedLang(d Decision) string {
	if d.Detection != nil {
		return d.Detection.Language
	}
	return ""
}

// langFor is the language handed to the agent: the request's own when it gave one, else the
// detected one. An explicit empty code stays empty, as an explicit "" does in laya.
func langFor(requested *string, d Decision) string {
	if requested != nil {
		return *requested
	}
	return detectedLang(d)
}

// Predict routes state, then answers every question on the chosen checkpoint in one forward
// pass. The result carries the decision. opts take the routing options of Route (ForModel,
// ForTask, ForLang, ForLangGuess), the hook options, ForMaxLen, ForHeadMaxLen and
// ForMinConfidence.
//
// Errors are the router's (an *UnknownModelError, a hook's failure, ErrClosed) and the agent's,
// unchanged. A nil state is refused first, with laya's text.
func (r *Router) Predict(ctx context.Context, state any, qs schema.Questions, opts ...CallOption) (Routed, error) {
	return r.predictOne(ctx, nil, state, qs, applyCalls(opts), nil)
}

// PredictRouted is Predict for a decision already made: it does not route, so no OnRoute hook
// runs and the routing options are not read. d.Model names the checkpoint.
func (r *Router) PredictRouted(ctx context.Context, d Decision, state any, qs schema.Questions, opts ...CallOption) (Routed, error) {
	return r.predictOne(ctx, &d, state, qs, applyCalls(opts), nil)
}

// PredictLong routes state as Predict does, then scores every window of it on the chosen
// checkpoint with Agent.PredictLong. A start hook that answers (Skip) or rewrites the state wins:
// the scan runs last. ForWindow, ForStride, ForBatchSize and the language (ForLang, else the
// detected one) reach the scan; ForMaxLen, ForHeadMaxLen and ForMinConfidence do not. The scan's
// errors are always returned, and Routed.Warnings holds its warnings.
func (r *Router) PredictLong(ctx context.Context, state any, qs schema.Questions, opts ...CallOption) (Routed, error) {
	c := applyCalls(opts)
	c.maxLen, c.headMaxLen, c.gate = 0, 0, nil
	return r.predictOne(ctx, nil, state, qs, c, &scanLong{window: c.window, stride: c.stride, batch: c.batch, lang: c.lang})
}

func (r *Router) predictOne(ctx context.Context, fixed *Decision, state any, qs schema.Questions, c call, scan *scanLong) (res Routed, err error) {
	if nilState(state) {
		return Routed{}, errNilState
	}
	if c.err != nil {
		return Routed{}, c.err
	}
	gate, err := agent.NewGate(c.gate...)
	if err != nil {
		return Routed{}, err
	}
	pol, err := r.policy(c)
	if err != nil {
		return Routed{}, err
	}
	hooks := r.predictHooks(c)
	pc := &agent.PredictContext{States: []any{state}, Questions: qs, RunID: newRunID(), Router: r,
		MaxLen: c.maxLen, HeadMaxLen: c.headMaxLen, StartedAt: time.Now()}
	var decision Decision
	var lease *Lease
	var warnings []string
	defer func() {
		// The prediction is already made, and an evicted agent's close failure belongs to whoever
		// evicted it, so it is not folded into this call's result.
		_ = lease.Close()
	}()

	infer := func() error {
		if fixed != nil {
			decision = *fixed
		} else {
			d, err := r.routeHooked(state, qs, c)
			if err != nil {
				return err
			}
			decision = d
		}
		pc.Decision, pc.Model = decision, string(decision.Model)
		l, err := r.Lease(ctx, string(decision.Model))
		if err != nil {
			return err
		}
		lease = l
		a := l.Agent()
		pc.Agent = a
		pc.StartedAt = time.Now()
		if err := r.fire(ctx, evStart, hooks, pc, pol); err != nil {
			return err
		}
		if pc.Results != nil {
			return nil
		}
		if len(pc.States) == 0 {
			return errors.New("router: a start hook left the call no state")
		}
		if scan != nil {
			// ctx.agent and ctx.decision, read when the scan runs: a start hook can have rewritten them.
			a = pc.Agent
			if a == nil {
				return errors.New("router: a start hook cleared the agent")
			}
			d := decision
			if pd, ok := pc.Decision.(Decision); ok {
				d = pd
			}
			lp, err := r.runner().predictLong(ctx, a, pc.States[0], pc.Questions,
				AgentCall{Lang: langFor(scan.lang, d), Batch: scan.batch, Window: scan.window, Stride: scan.stride})
			if err != nil {
				return err
			}
			warnings = lp.Warnings
			pc.Results = []schema.Prediction{lp.Prediction}
			return nil
		}
		p, err := r.runner().predict(ctx, a, pc.States[0], pc.Questions,
			AgentCall{Lang: langFor(c.lang, decision), MaxLen: pc.MaxLen, HeadMaxLen: pc.HeadMaxLen})
		if err != nil {
			return err
		}
		pc.Results = []schema.Prediction{p}
		return nil
	}

	err = infer()
	if err != nil {
		pc.Err = err
		err = join(err, r.fire(ctx, evError, hooks, pc, pol))
	}
	pc.ElapsedMS = float64(time.Since(pc.StartedAt)) / float64(time.Millisecond)
	if pc.Results != nil {
		pc.Usage = sumUsage(pc.Results)
		gate.Apply(pc.Results)
	}
	err = join(err, r.fire(ctx, evEnd, hooks, pc, pol))
	if err != nil {
		return Routed{}, err
	}
	if len(pc.Results) == 0 {
		return Routed{}, errors.New("router: the call ended with no result")
	}
	return Routed{Prediction: pc.Results[0], Routing: decision, Warnings: warnings}, nil
}

// batchKey is what splits a checkpoint's requests into forward passes after the start hooks ran.
type batchKey struct {
	schema     string
	maxLen     int
	headMaxLen int
	langSet    bool
	lang       string
}

type batchGroup struct {
	key    batchKey
	qs     schema.Questions
	lang   string
	maxLen int
	head   int
	items  []int // positions in the checkpoint group
}

// PredictBatch routes and answers a heterogeneous batch with the fewest forward passes. Requests
// are routed first and grouped by checkpoint; within a checkpoint, requests whose questions are
// the same (same ids, types, options and orders) and whose token budgets and language agree share
// one Agent.PredictBatch call. Results are returned in the order of reqs.
//
// Each request is routed and run with its own Options, after opts; opts also carry the hooks and
// ForMinConfidence of the whole batch, and ForSortByLength. A request's own hook options reach
// only its routing. batchSize caps the states per forward pass; 0 or less is no cap.
//
// Router predict hooks run per request, each with its own PredictContext: a start hook may
// rewrite that request's state, questions or budgets or Skip it, and groups are formed after the
// start hooks ran. A checkpoint group's requests end in reverse of the order they started. If
// anything in a checkpoint group fails, every request of it whose start hook ran fails with the
// error (each gets OnError, then OnPredictEnd) and PredictBatch returns it with no results; a
// Skip-ped request does not shift the others.
//
// QuestionSchema is the grouping key, so questions that differ in key order or in an option
// order never share a pass. Unlike Python, a request whose language is the empty string shares a
// group with one that has none, and a MaxLen or HeadMaxLen of 0 or less is no override.
func (r *Router) PredictBatch(ctx context.Context, reqs []Request, batchSize int, opts ...CallOption) ([]Routed, error) {
	c0 := applyCalls(opts)
	gate, err := agent.NewGate(c0.gate...)
	if err != nil {
		return nil, err
	}
	decisions, err := r.RouteBatch(reqs, opts...)
	if err != nil {
		return nil, err
	}
	if len(decisions) == 0 {
		return []Routed{}, nil
	}
	if c0.err != nil {
		return nil, c0.err
	}
	pol, err := r.policy(c0)
	if err != nil {
		return nil, err
	}
	calls := make([]call, len(reqs))
	for i, q := range reqs {
		calls[i] = applyCalls(append(append([]CallOption(nil), opts...), q.Options...))
	}
	// Checkpoint groups in order of first appearance, keyed by the canonical name so an alias
	// shares its checkpoint's pass.
	var names []Checkpoint
	groups := map[Checkpoint][]int{}
	for i, d := range decisions {
		key, err := NormaliseName(string(d.Model))
		if err != nil {
			return nil, err
		}
		if _, ok := groups[key]; !ok {
			names = append(names, key)
		}
		groups[key] = append(groups[key], i)
	}
	hooks := r.predictHooks(c0)
	out := make([]Routed, len(reqs))
	for _, name := range names {
		if err := r.runGroup(ctx, name, groups[name], reqs, decisions, calls, batchSize, c0.sort, hooks, pol, gate, out); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// runGroup runs one checkpoint's requests under one lease, and releases it before the next group
// can load.
func (r *Router) runGroup(ctx context.Context, name Checkpoint, idx []int, reqs []Request, decisions []Decision,
	calls []call, batchSize int, sortLen bool, hooks []agent.Hook, pol hookrun.Policy, gate *agent.Gate, out []Routed) error {
	lease, err := r.Lease(ctx, string(name))
	if err != nil {
		return err
	}
	defer func() { _ = lease.Close() }() // as in predictOne
	a := lease.Agent()
	rn := r.runner()
	started := make([]*agent.PredictContext, 0, len(idx))

	run := func() error {
		for _, i := range idx {
			c := calls[i]
			pc := &agent.PredictContext{States: []any{reqs[i].State}, Questions: reqs[i].Questions, RunID: newRunID(),
				Decision: decisions[i], Model: string(name), Agent: a, Router: r,
				MaxLen: c.maxLen, HeadMaxLen: c.headMaxLen, StartedAt: time.Now()}
			started = append(started, pc)
			if err := r.fire(ctx, evStart, hooks, pc, pol); err != nil {
				return err
			}
		}
		// Group again on what the start hooks left. Language is part of the key only for an agent
		// that has per-language temperatures: otherwise it is unused and would split a shared pass.
		temps := a.HasLangTemperatures()
		var order []*batchGroup
		for j, pc := range started {
			if pc.Results != nil {
				continue
			}
			if len(pc.States) == 0 {
				return errors.New("router: a start hook left the call no state")
			}
			i := idx[j]
			k := batchKey{maxLen: max(pc.MaxLen, 0), headMaxLen: max(pc.HeadMaxLen, 0)}
			var lang string
			if temps {
				lang = langFor(calls[i].lang, decisions[i])
				k.langSet, k.lang = calls[i].lang != nil || detectedLang(decisions[i]) != "", lang
			}
			sch, err := QuestionSchema(pc.Questions)
			if err != nil {
				return err
			}
			k.schema = sch
			var g *batchGroup
			for _, o := range order {
				if o.key == k {
					g = o
					break
				}
			}
			if g == nil {
				g = &batchGroup{key: k, qs: pc.Questions, lang: lang, maxLen: k.maxLen, head: k.headMaxLen}
				order = append(order, g)
			}
			g.items = append(g.items, j)
		}
		for _, g := range order {
			states := make([]any, len(g.items))
			for n, j := range g.items {
				states[n] = started[j].States[0]
			}
			preds, err := rn.predictBatch(ctx, a, states, g.qs,
				AgentCall{Lang: g.lang, Batch: batchSize, Sort: sortLen, MaxLen: g.maxLen, HeadMaxLen: g.head})
			if err != nil {
				return err
			}
			if len(preds) != len(g.items) {
				return fmt.Errorf("internal error: Agent.predict_batch returned %d results for %d states", len(preds), len(g.items))
			}
			for n, j := range g.items {
				started[j].Results = []schema.Prediction{preds[n]}
			}
		}
		// Usage is summed before any end hook, so a malformed block fails the group like any
		// other error.
		for _, pc := range started {
			pc.Usage = sumUsage(pc.Results)
		}
		return nil
	}

	if err := run(); err != nil {
		for _, pc := range started {
			if len(pc.Results) > 0 {
				gate.Apply(pc.Results)
			}
		}
		return r.endContexts(ctx, hooks, started, pol, err)
	}
	for _, pc := range started {
		if len(pc.Results) > 0 {
			gate.Apply(pc.Results)
		}
	}
	if err := r.endContexts(ctx, hooks, started, pol, nil); err != nil {
		return err
	}
	for j, i := range idx {
		pc := started[j]
		if len(pc.Results) == 0 {
			return errors.New("router: a request ended with no result")
		}
		out[i] = Routed{Prediction: pc.Results[0], Routing: decisions[i]}
	}
	return nil
}

// endContexts finishes each request of a batch as Predict finishes one. With failure set every
// context fails with it and gets OnError before OnPredictEnd. They end in reverse of the order
// they started, and every context gets its OnPredictEnd even if another's hooks fail; the first
// such failure is returned afterwards. On a failed context a failing hook is attached to its
// error instead. ElapsedMS is set on all of them first, so one request's never includes another's
// hooks. It returns failure (with attachments) when there is one.
func (r *Router) endContexts(ctx context.Context, hooks []agent.Hook, started []*agent.PredictContext, pol hookrun.Policy, failure error) error {
	now := time.Now()
	for _, pc := range started {
		pc.ElapsedMS = float64(now.Sub(pc.StartedAt)) / float64(time.Millisecond)
		if failure != nil {
			pc.Err = failure
		}
	}
	var first error
	for n := len(started) - 1; n >= 0; n-- {
		pc := started[n]
		events := []string{evEnd}
		if pc.Err != nil {
			events = []string{evError, evEnd}
		}
		for _, ev := range events {
			if err := r.fire(ctx, ev, hooks, pc, pol); err != nil {
				if pc.Err != nil {
					failure = join(failure, err)
				} else if first == nil {
					first = err
				}
			}
		}
	}
	if failure != nil {
		return failure
	}
	return first
}
