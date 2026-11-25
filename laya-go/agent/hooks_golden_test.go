package agent

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// nopSession lets a bare Agent enter calls; hooks tests replace inference with hookState.stub.
type nopSession struct{}

func (nopSession) Run(context.Context, infer.Batch) (infer.Output, error) {
	return infer.Output{}, errors.New("nopSession runs nothing")
}
func (nopSession) Close() error { return nil }

// recorder collects what hooks saw, in order.
type recorder struct {
	mu    sync.Mutex
	calls []hookCall
	warns []string
}

type hookCall struct {
	name, event string
	states      []string
	questions   []string
	model       string
	maxLen      int
	headMaxLen  int
	results     []int
	usage       HookUsage
	err         string
	elapsedSet  bool
	runID       string
}

func (r *recorder) warn(msg string) {
	r.mu.Lock()
	r.warns = append(r.warns, msg)
	r.mu.Unlock()
}

// StubHook implements the events in events, records each call and then acts out behaviour[event].
// Its type name is what the recorded warnings carry.
type StubHook struct {
	name      string
	events    map[string]bool
	behaviour map[string]string
	rec       *recorder
}

func (h *StubHook) OnPredictStart(c *PredictContext) { h.fire("on_predict_start", c) }
func (h *StubHook) OnPredictEnd(c *PredictContext)   { h.fire("on_predict_end", c) }
func (h *StubHook) OnError(c *PredictContext)        { h.fire("on_error", c) }

func predOf(v int) schema.Prediction {
	return schema.Prediction{Usage: schema.Usage{InputTokens: v, OutputTokens: v - 10}}
}

func resultInts(ps []schema.Prediction) []int {
	if ps == nil {
		return nil
	}
	out := make([]int, len(ps))
	for i, p := range ps {
		out[i] = p.Usage.InputTokens
	}
	return out
}

func questionsOf(ids ...string) schema.Questions {
	var es []schema.Entry
	for _, id := range ids {
		es = append(es, schema.Ask(id, schema.MustQuestion(schema.Noul("q "+id))))
	}
	qs, err := schema.NewQuestions(es...)
	if err != nil {
		panic(err)
	}
	return qs
}

func (h *StubHook) fire(ev string, c *PredictContext) {
	if len(h.events) > 0 && !h.events[ev] {
		return
	}
	hc := hookCall{name: h.name, event: ev, model: c.Model, maxLen: c.MaxLen, headMaxLen: c.HeadMaxLen,
		results: resultInts(c.Results), usage: c.Usage, elapsedSet: c.ElapsedMS > 0, runID: c.RunID}
	for _, s := range c.States {
		hc.states = append(hc.states, fmt.Sprint(s))
	}
	hc.questions = c.Questions.IDs()
	if c.Err != nil {
		hc.err = c.Err.Error()
	}
	h.rec.mu.Lock()
	h.rec.calls = append(h.rec.calls, hc)
	h.rec.mu.Unlock()
	b := h.behaviour[ev]
	if b == "" {
		return
	}
	kind, arg, _ := strings.Cut(b, ":")
	switch kind {
	case "raise":
		panic(errors.New("boom from " + h.name))
	case "skip":
		n, _ := strconv.Atoi(arg)
		rs := make([]schema.Prediction, n)
		for i := range rs {
			rs[i] = predOf(10 + i)
		}
		if err := c.Skip(rs); err != nil {
			panic(err)
		}
	case "states":
		c.States = nil
		for _, s := range strings.Split(arg, ",") {
			c.States = append(c.States, s)
		}
	case "questions":
		c.Questions = questionsOf(strings.Split(arg, ",")...)
	case "max_len":
		c.MaxLen, _ = strconv.Atoi(arg)
	case "head_max_len":
		c.HeadMaxLen, _ = strconv.Atoi(arg)
	case "results":
		c.Results = []schema.Prediction{predOf(500 + 10)}
	default:
		panic("unknown behaviour " + b)
	}
}

// testAgent is a bare Agent whose inference is the golden's stub: result i is 110+i.
func testAgent(t testing.TB, o options) *Agent {
	t.Helper()
	a := &Agent{sess: nopSession{}, hk: newHookState(o)}
	a.hk.stub = func(_ context.Context, states []any, qs schema.Questions, c call) ([]schema.Prediction, error) {
		rs := make([]schema.Prediction, len(states))
		for i := range rs {
			rs[i] = predOf(110 + i)
			rs[i].Usage.InputTokens = 110 + i
		}
		return rs, nil
	}
	t.Cleanup(func() { a.Close() })
	return a
}

func behaviourOf(t testing.TB, v any) map[string]string {
	t.Helper()
	m := map[string]string{}
	if v == nil {
		return m
	}
	o := fixtures.Obj(t, v)
	for _, k := range o.Keys() {
		m[k] = fixtures.Str(t, o.Must(k))
	}
	return m
}

func TestHooksGoldenDispatch(t *testing.T) {
	g := fixtures.Load(t, "hooks.json")
	routed := 0
	for _, cv := range fixtures.List(t, g.Must("dispatch")) {
		c := fixtures.Obj(t, cv)
		name := fixtures.Str(t, c.Must("case"))
		if e := fixtures.Str(t, c.Must("event")); e == "on_route" {
			routed++ // the router PR's event; Go has no router yet
			continue
		}
		t.Run(name, func(t *testing.T) {
			event := fixtures.Str(t, c.Must("event"))
			var ev hookEvent
			switch event {
			case "on_predict_start":
				ev = evStart
			case "on_predict_end":
				ev = evEnd
			case "on_error":
				ev = evError
			default:
				t.Fatalf("event %s is not ported", event)
			}
			rec := &recorder{}
			a := testAgent(t, options{raise: fixtures.Bool(t, c.Must("raise_errors")), warn: rec.warn})
			var chain []Hook
			for _, hv := range fixtures.List(t, c.Must("chain")) {
				h := fixtures.Obj(t, hv)
				beh := map[string]string{}
				if b, ok := h.Must("behaviour").(string); ok {
					beh[event] = b
				}
				events := map[string]bool{}
				for _, e := range fixtures.Strings(t, h.Must("events")) {
					events[e] = true
				}
				chain = append(chain, &StubHook{name: fixtures.Str(t, h.Must("name")), events: events, behaviour: beh, rec: rec})
			}
			pc := &PredictContext{States: []any{"s0", "s1", "s2"}, Questions: questionsOf("q"), Agent: a}
			err := a.dispatch(context.Background(), ev, chain, pc)
			var calls []string
			for _, hc := range rec.calls {
				calls = append(calls, hc.name+"."+hc.event)
			}
			if got, want := strings.Join(calls, ","), strings.Join(fixtures.Strings(t, c.Must("calls")), ","); got != want {
				t.Errorf("calls = %q, want %q", got, want)
			}
			if raised, ok := c.Must("raised").(fixtures.Object); ok {
				want := fixtures.Str(t, raised.Must("message"))
				if err == nil || err.Error() != want {
					t.Errorf("err = %v, want %q", err, want)
				}
			} else if err != nil {
				t.Errorf("err = %v, want none", err)
			}
			if got, want := strings.Join(rec.warns, "|"), strings.Join(fixtures.Strings(t, c.Must("warnings")), "|"); got != want {
				t.Errorf("warnings = %q, want %q", got, want)
			}
			if n, ok := c.Must("results").(interface{ String() string }); ok {
				want, _ := strconv.Atoi(n.String())
				if len(pc.Results) != want {
					t.Errorf("results = %d, want %d", len(pc.Results), want)
				}
			} else if pc.Results != nil {
				t.Errorf("results = %v, want none", pc.Results)
			}
		})
	}
	if routed != 2 { // a new on_route case, or one renamed, must be looked at, not silently dropped
		t.Fatalf("%d on_route cases excluded, want 2", routed)
	}
}

func TestHooksGoldenSkip(t *testing.T) {
	for _, cv := range fixtures.List(t, fixtures.Load(t, "hooks.json").Must("skip")) {
		c := fixtures.Obj(t, cv)
		t.Run(fixtures.Str(t, c.Must("case")), func(t *testing.T) {
			pc := &PredictContext{States: make([]any, fixtures.Int(t, c.Must("states")))}
			n := fixtures.Int(t, c.Must("results"))
			rs := make([]schema.Prediction, n)
			for i := range rs {
				rs[i] = predOf(10 + i)
			}
			err := pc.Skip(rs)
			if fixtures.Bool(t, c.Must("accepted")) {
				if err != nil {
					t.Fatalf("err = %v", err)
				}
				if got, want := fmt.Sprint(resultInts(pc.Results)), fmt.Sprint(fixtures.Ints(t, c.Must("assigned"))); got != want {
					t.Errorf("assigned = %s, want %s", got, want)
				}
				return
			}
			if err == nil {
				t.Fatal("Skip accepted a wrong count")
			}
			if want := fixtures.Str(t, c.Must("error")); err.Error() != want {
				t.Errorf("err = %q, want %q", err, want)
			}
			if pc.Results != nil {
				t.Errorf("a refused Skip assigned %v", pc.Results)
			}
		})
	}
}

func TestHooksGoldenUsage(t *testing.T) {
	pyNone := 0
	for _, cv := range fixtures.List(t, fixtures.Load(t, "hooks.json").Must("usage")) {
		c := fixtures.Obj(t, cv)
		if usageIsNone(t, c) {
			pyNone++ // a Go Prediction's Usage is a value, never None
			continue
		}
		t.Run(fixtures.Str(t, c.Must("case")), func(t *testing.T) {
			var rs []schema.Prediction
			for _, rv := range fixtures.List(t, c.Must("results")) {
				r := fixtures.Obj(t, rv)
				tok := func(k string) int { // missing, null and float inputs are Python-only shapes
					switch v, _ := r.Get(k); x := v.(type) {
					case nil:
						return 0
					default:
						_ = x
						return int(fixtures.Float(t, v)) // JSON numbers; a float truncates as in Python
					}
				}
				rs = append(rs, schema.Prediction{Usage: schema.Usage{InputTokens: tok("input_tokens"), OutputTokens: tok("output_tokens")}})
			}
			tot := fixtures.Obj(t, c.Must("total"))
			want := HookUsage{InputTokens: fixtures.Int(t, tot.Must("input_tokens")), OutputTokens: fixtures.Int(t, tot.Must("output_tokens"))}
			if got := aggregateUsage(rs); got != want {
				t.Errorf("usage = %+v, want %+v", got, want)
			}
		})
	}
	if pyNone != 1 { // Python-only shape: a result whose usage is None
		t.Fatalf("%d None-usage cases excluded, want 1", pyNone)
	}
}

func TestHooksGoldenTimeouts(t *testing.T) {
	for _, cv := range fixtures.List(t, fixtures.Load(t, "hooks.json").Must("timeouts")) {
		c := fixtures.Obj(t, cv)
		in := fixtures.Str(t, c.Must("input"))
		t.Run(in, func(t *testing.T) {
			v, ok := c.Must("value").(interface{ String() string })
			if !ok { // None: no limit, which is leaving the option out
				return
			}
			secs := fixtures.Float(t, c.Must("value"))
			_ = v
			err := validateTimeout(secs)
			if !fixtures.Bool(t, c.Must("accepted")) {
				t.Fatalf("accepted rows carry a value")
			}
			if err != nil {
				t.Fatalf("validateTimeout(%v) = %v", secs, err)
			}
			if got, want := strconv.FormatFloat(secs, 'g', -1, 64), fixtures.Str(t, c.Must("seconds")); got != want && in != "1e-07" {
				t.Errorf("rendered %q, want %q", got, want)
			}
		})
	}
	// Refusals are recorded without a value; their input is the number.
	for _, cv := range fixtures.List(t, fixtures.Load(t, "hooks.json").Must("timeouts")) {
		c := fixtures.Obj(t, cv)
		if fixtures.Bool(t, c.Must("accepted")) {
			continue
		}
		in := fixtures.Str(t, c.Must("input"))
		secs, err := strconv.ParseFloat(in, 64)
		if err != nil {
			t.Fatalf("input %q: %v", in, err)
		}
		verr := validateTimeout(secs)
		if verr == nil || !strings.HasPrefix(verr.Error(), strings.SplitN(fixtures.Str(t, c.Must("error")), ";", 2)[0]) {
			t.Errorf("validateTimeout(%q) = %v, want the golden refusal %q", in, verr, fixtures.Str(t, c.Must("error")))
		}
	}
}

func TestHooksGoldenRegistry(t *testing.T) {
	for _, cv := range fixtures.List(t, fixtures.Load(t, "hooks.json").Must("registry")) {
		c := fixtures.Obj(t, cv)
		t.Run(fixtures.Str(t, c.Must("case")), func(t *testing.T) {
			a := testAgent(t, options{raise: true})
			named := map[string]*StubHook{}
			get := func(n string) *StubHook {
				if named[n] == nil {
					named[n] = &StubHook{name: n, rec: &recorder{}}
				}
				return named[n]
			}
			names := func(list string) []Hook {
				var hs []Hook
				for _, n := range strings.FieldsFunc(list, func(r rune) bool { return r == '[' || r == ']' || r == ',' }) {
					hs = append(hs, get(n))
				}
				return hs
			}
			undo := map[string]func(){}
			steps := fixtures.List(t, c.Must("steps"))
			for i, op := range fixtures.Strings(t, c.Must("script")) {
				st := fixtures.Obj(t, steps[i])
				if got := fixtures.Str(t, st.Must("op")); got != op {
					t.Fatalf("step %d is %q, script says %q", i, got, op)
				}
				verb, rest, _ := strings.Cut(op, ":")
				switch verb {
				case "add":
					if err := a.AddHook(names(rest)...); err != nil {
						t.Fatal(err)
					}
				case "remove":
					got := a.RemoveHook(get(rest))
					want, isBool := st.Must("returned").(bool)
					if !isBool || got != want {
						t.Errorf("%s returned %v, want %v", op, got, st.Must("returned"))
					}
				case "enter":
					id, list, _ := strings.Cut(rest, ":")
					u, err := a.InstallHooks(names(list)...)
					if err != nil {
						t.Fatal(err)
					}
					undo[id] = u
				case "exit":
					undo[rest]()
				default:
					t.Fatalf("unknown op %q", op)
				}
				if got, want := hookNames(a.Hooks()), strings.Join(fixtures.Strings(t, st.Must("hooks")), ","); got != want {
					t.Errorf("after %s hooks = %q, want %q", op, got, want)
				}
			}
			if got, want := hookNames(a.Hooks()), strings.Join(fixtures.Strings(t, c.Must("final")), ","); got != want {
				t.Errorf("final = %q, want %q", got, want)
			}
		})
	}
}

func hookNames(hs []Hook) string {
	var ns []string
	for _, h := range hs {
		ns = append(ns, h.(*StubHook).name)
	}
	return strings.Join(ns, ",")
}

func TestHooksGoldenDefaultHooksAPI(t *testing.T) {
	t.Cleanup(ClearDefaultHooks)
	g := fixtures.Obj(t, fixtures.Load(t, "hooks.json").Must("default_hooks_api"))
	rec := &recorder{}
	probe, added := &StubHook{name: "probe", rec: rec}, &StubHook{name: "added", rec: rec}
	check := func(label string, key string) {
		t.Helper()
		if got, want := hookNames(DefaultHooks()), strings.Join(fixtures.Strings(t, g.Must(key)), ","); got != want {
			t.Errorf("%s: defaults = %q, want %q", label, got, want)
		}
	}
	if err := SetDefaultHooks(probe); err != nil {
		t.Fatal(err)
	}
	check("after set", "after_set")
	DefaultHooks()[0] = added // a copy: the registry must not change
	check("after mutating the returned list", "mutating_the_returned_list_changes_nothing")
	if err := AddDefaultHook(added); err != nil {
		t.Fatal(err)
	}
	check("after add", "after_add")
	ClearDefaultHooks()
	check("after clear", "after_clear")
}

func TestHooksGoldenNormaliseRefusals(t *testing.T) {
	// Python refuses a class, an object with no event and so on; the typed Go API cannot be given
	// those, and refuses what it can be given: nil.
	if n := len(fixtures.List(t, fixtures.Load(t, "hooks.json").Must("normalise_refusals"))); n != 6 {
		t.Fatalf("golden has %d refusals, the port was written against 6", n)
	}
	var nilPtr *StubHook
	for _, h := range []Hook{nil, nilPtr} {
		if err := checkHooks([]Hook{h}); err == nil {
			t.Errorf("checkHooks accepted %#v", h)
		}
	}
}

func TestHooksGoldenAsSequence(t *testing.T) {
	// The Go API is variadic, so a sequence needs no coercion: the length is the argument count.
	rec := &recorder{}
	h := &StubHook{name: "h", rec: rec}
	for _, cv := range fixtures.List(t, fixtures.Load(t, "hooks.json").Must("as_sequence")) {
		c := fixtures.Obj(t, cv)
		n := fixtures.Int(t, c.Must("length"))
		if n > 3 {
			t.Fatalf("length %d", n)
		}
		hs := make([]Hook, n)
		for i := range hs {
			hs[i] = h
		}
		a := testAgent(t, options{raise: true})
		if err := a.AddHook(hs...); err != nil || len(a.Hooks()) != n {
			t.Errorf("%v: AddHook of %d hooks gave %d, %v", c.Must("input"), n, len(a.Hooks()), err)
		}
	}
}

func TestHooksGoldenComposition(t *testing.T) {
	t.Cleanup(ClearDefaultHooks)
	for _, cv := range fixtures.List(t, fixtures.Load(t, "hooks.json").Must("composition")) {
		c := fixtures.Obj(t, cv)
		t.Run(fixtures.Str(t, c.Must("case")), func(t *testing.T) {
			rec := &recorder{}
			mk := func(n string) *StubHook {
				return &StubHook{name: n, events: map[string]bool{"on_predict_start": true}, rec: rec}
			}
			var defaults, installed, perCall []Hook
			for _, n := range fixtures.Strings(t, c.Must("defaults")) {
				defaults = append(defaults, mk(n))
			}
			for _, n := range fixtures.Strings(t, c.Must("installed")) {
				installed = append(installed, mk(n))
			}
			for _, n := range fixtures.Strings(t, c.Must("hooks")) {
				perCall = append(perCall, mk(n))
			}
			if err := SetDefaultHooks(defaults...); err != nil {
				t.Fatal(err)
			}
			a := testAgent(t, options{raise: true, installed: installed})
			var order []string
			note := func(s string) { rec.mu.Lock(); order = append(order, s); rec.mu.Unlock() }
			opts := []CallOption{WithHooks(perCall...)}
			for _, n := range fixtures.Strings(t, c.Must("on_predict_start")) {
				opts = append(opts, WithOnPredictStart(func(*PredictContext) { note("start:" + n) }))
			}
			for _, n := range fixtures.Strings(t, c.Must("on_predict_end")) {
				opts = append(opts, WithOnPredictEnd(func(*PredictContext) { note("end:" + n) }))
			}
			if fixtures.Bool(t, c.Must("skip_defaults")) {
				opts = append(opts, WithoutDefaultHooks())
			}
			if _, err := a.PredictBatch(context.Background(), []any{"s"}, questionsOf("q"), opts...); err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, hc := range rec.calls {
				got = append(got, hc.name)
			}
			got = append(got, order...)
			if g, w := strings.Join(got, ","), strings.Join(fixtures.Strings(t, c.Must("order")), ","); g != w {
				t.Errorf("order = %q, want %q", g, w)
			}
		})
	}
}

func TestHooksGoldenPredictBatch(t *testing.T) {
	t.Cleanup(ClearDefaultHooks)
	for _, cv := range fixtures.List(t, fixtures.Load(t, "hooks.json").Must("predict_batch")) {
		c := fixtures.Obj(t, cv)
		t.Run(fixtures.Str(t, c.Must("case")), func(t *testing.T) {
			rec := &recorder{}
			mk := func(v any) Hook {
				h := fixtures.Obj(t, v)
				return &StubHook{name: fixtures.Str(t, h.Must("name")), behaviour: behaviourOf(t, h.Must("behaviour")), rec: rec}
			}
			var defaults, installed, perCall []Hook
			for _, v := range fixtures.List(t, c.Must("defaults")) {
				defaults = append(defaults, mk(v))
			}
			for _, v := range fixtures.List(t, c.Must("installed")) {
				installed = append(installed, mk(v))
			}
			for _, v := range fixtures.List(t, c.Must("hooks")) {
				perCall = append(perCall, mk(v))
			}
			if err := SetDefaultHooks(defaults...); err != nil {
				t.Fatal(err)
			}
			a := testAgent(t, options{raise: fixtures.Bool(t, c.Must("hooks_raise")), installed: installed, warn: rec.warn})
			infRaises := fixtures.Bool(t, c.Must("inference_raises"))
			var inf struct {
				ran                bool
				states, questions  []string
				maxLen, headMaxLen int
			}
			base := a.hk.stub
			a.hk.stub = func(ctx context.Context, states []any, qs schema.Questions, cl call) ([]schema.Prediction, error) {
				if len(states) == 0 {
					return []schema.Prediction{}, nil // predictStates answers an empty call itself
				}
				inf.ran = true
				for _, s := range states {
					inf.states = append(inf.states, fmt.Sprint(s))
				}
				inf.questions = qs.IDs()
				inf.maxLen, inf.headMaxLen = cl.maxLen, cl.headMaxLen
				if infRaises {
					return nil, errors.New("the graph refused this batch")
				}
				return base(ctx, states, qs, cl)
			}
			var states []any
			for _, s := range fixtures.Strings(t, c.Must("states")) {
				states = append(states, s)
			}
			res, err := a.PredictBatch(context.Background(), states, questionsOf(fixtures.Strings(t, c.Must("questions"))...), WithHooks(perCall...))

			if want, ok := c.Must("inference").(fixtures.Object); ok {
				if !inf.ran {
					t.Fatal("inference did not run")
				}
				if g, w := strings.Join(inf.states, ","), strings.Join(fixtures.Strings(t, want.Must("states")), ","); g != w {
					t.Errorf("inference states = %q, want %q", g, w)
				}
				if g, w := strings.Join(inf.questions, ","), strings.Join(fixtures.Strings(t, want.Must("questions")), ","); g != w {
					t.Errorf("inference questions = %q, want %q", g, w)
				}
				ov := fixtures.Obj(t, want.Must("overrides"))
				for _, o := range []struct {
					key string
					got int
				}{{"max_len", inf.maxLen}, {"head_max_len", inf.headMaxLen}} {
					w := 0
					if v, ok := ov.Get(o.key); ok {
						w = fixtures.Int(t, v)
					}
					if o.got != w {
						t.Errorf("inference %s = %d, want %d", o.key, o.got, w)
					}
				}
			} else if inf.ran {
				t.Error("inference ran, the golden says a hook answered the call")
			}

			wantCalls := fixtures.List(t, c.Must("calls"))
			if len(rec.calls) != len(wantCalls) {
				t.Fatalf("%d hook calls, want %d: %+v", len(rec.calls), len(wantCalls), rec.calls)
			}
			runIDs := map[string]bool{}
			for i, wv := range wantCalls {
				w, g := fixtures.Obj(t, wv), rec.calls[i]
				runIDs[g.runID] = true
				if want := fixtures.Str(t, w.Must("call")); g.name+"."+g.event != want {
					t.Errorf("call %d = %s.%s, want %s", i, g.name, g.event, want)
				}
				eq := func(field string, got []string, want any) {
					if g, w := strings.Join(got, ","), strings.Join(fixtures.Strings(t, want), ","); g != w {
						t.Errorf("call %d %s = %q, want %q", i, field, g, w)
					}
				}
				eq("states", g.states, w.Must("states"))
				eq("questions", g.questions, w.Must("questions"))
				if g.model != schema.Model {
					t.Errorf("call %d model = %q, want %q", i, g.model, schema.Model)
				}
				for _, n := range []struct {
					key string
					got int
				}{{"max_len", g.maxLen}, {"head_max_len", g.headMaxLen}} {
					want := 0
					if v := w.Must(n.key); v != nil {
						want = fixtures.Int(t, v)
					}
					if n.got != want {
						t.Errorf("call %d %s = %d, want %d", i, n.key, n.got, want)
					}
				}
				if rv := w.Must("results"); rv == nil {
					if g.results != nil {
						t.Errorf("call %d results = %v, want none", i, g.results)
					}
				} else if gs, ws := fmt.Sprint(g.results), fmt.Sprint(fixtures.Ints(t, rv)); gs != ws {
					t.Errorf("call %d results = %s, want %s", i, gs, ws)
				}
				wantUsage := HookUsage{}
				if uv, ok := w.Must("usage").(fixtures.Object); ok {
					wantUsage = HookUsage{InputTokens: fixtures.Int(t, uv.Must("input_tokens")), OutputTokens: fixtures.Int(t, uv.Must("output_tokens"))}
				}
				if g.usage != wantUsage {
					t.Errorf("call %d usage = %+v, want %+v", i, g.usage, wantUsage)
				}
				wantErr := ""
				if ev, ok := w.Must("error").(string); ok {
					wantErr = ev
				}
				if g.err != wantErr {
					t.Errorf("call %d error = %q, want %q", i, g.err, wantErr)
				}
				if want := fixtures.Bool(t, w.Must("elapsed_ms_set")); g.elapsedSet != want {
					t.Errorf("call %d elapsed set = %v, want %v", i, g.elapsedSet, want)
				}
			}
			if want := fixtures.Bool(t, c.Must("one_run_id_per_call")); want && len(runIDs) > 1 {
				t.Errorf("hooks saw %d run ids, want one", len(runIDs))
			}
			for id := range runIDs {
				switch n := c.Must("run_id_hex_length").(type) {
				case nil:
				default:
					if len(id) != fixtures.Int(t, n) {
						t.Errorf("run id %q is not %d hex characters", id, fixtures.Int(t, n))
					}
				}
			}

			if raised, ok := c.Must("raised").(fixtures.Object); ok {
				if err == nil || err.Error() != fixtures.Str(t, raised.Must("message")) {
					t.Fatalf("err = %v, want %q", err, fixtures.Str(t, raised.Must("message")))
				}
				if ctxv, ok := raised.Must("context").(fixtures.Object); ok {
					var at *attached
					if !errors.As(err, &at) || at.second.Error() != fixtures.Str(t, ctxv.Must("message")) {
						t.Errorf("err %v does not carry the second error %q", err, fixtures.Str(t, ctxv.Must("message")))
					}
				}
				if res != nil {
					t.Errorf("a failed call returned %v", resultInts(res))
				}
			} else {
				if err != nil {
					t.Fatalf("err = %v", err)
				}
				if g, w := fmt.Sprint(resultInts(res)), fmt.Sprint(fixtures.Ints(t, c.Must("result"))); g != w {
					t.Errorf("result = %s, want %s", g, w)
				}
			}
			if g, w := strings.Join(rec.warns, "|"), strings.Join(fixtures.Strings(t, c.Must("warnings")), "|"); g != w {
				t.Errorf("warnings = %q, want %q", g, w)
			}
		})
	}
}

func usageIsNone(t *testing.T, c fixtures.Object) bool {
	for _, rv := range fixtures.List(t, c.Must("results")) {
		if _, none := fixtures.Obj(t, rv).Get("__usage_is_none__"); none {
			return true
		}
	}
	return false
}
