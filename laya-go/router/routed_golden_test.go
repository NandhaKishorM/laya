package router_test

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/router"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// routedRig is one golden case: attached agents, a Runner that logs what the Router asked, and
// the hooks, all in the shape the Python stub log has.
type routedRig struct {
	mu     sync.Mutex
	calls  []string
	events []string
	pools  map[string]schema.Questions
	idx    map[*agent.Agent]int
}

func (g *routedRig) poolName(qs schema.Questions) string {
	sig, err := router.QuestionSchema(qs)
	if err != nil {
		return "?"
	}
	for name, p := range g.pools {
		if want, _ := router.QuestionSchema(p); want == sig {
			return name
		}
	}
	return "?"
}

func (g *routedRig) log(a *agent.Agent, m string, states []string, qs schema.Questions, ac router.AgentCall) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.calls = append(g.calls, fmt.Sprintf("%d|%s|%s|%s|%s|%d|%t|%d|%d|%d|%d", g.idx[a], m, strings.Join(states, "\x1f"),
		g.poolName(qs), ac.Lang, ac.Batch, ac.Sort, ac.MaxLen, ac.HeadMaxLen, ac.Window, ac.Stride))
}

func stubPrediction(a *agent.Agent, g *routedRig, state any, extra int) schema.Prediction {
	n := len([]rune(state.(string)))
	return schema.Prediction{Model: "stub", Usage: schema.Usage{InputTokens: 10*n + g.idx[a] + extra, OutputTokens: 1}}
}

func (g *routedRig) runner() router.Runner {
	return router.Runner{
		Predict: func(_ context.Context, a *agent.Agent, state any, qs schema.Questions, ac router.AgentCall) (schema.Prediction, error) {
			g.log(a, "predict", []string{state.(string)}, qs, ac)
			return stubPrediction(a, g, state, 0), nil
		},
		Batch: func(_ context.Context, a *agent.Agent, states []any, qs schema.Questions, ac router.AgentCall) ([]schema.Prediction, error) {
			var ss []string
			var out []schema.Prediction
			for _, s := range states {
				ss = append(ss, s.(string))
				out = append(out, stubPrediction(a, g, s, 0))
			}
			g.log(a, "batch", ss, qs, ac)
			return out, nil
		},
		Long: func(_ context.Context, a *agent.Agent, state any, qs schema.Questions, ac router.AgentCall) (agent.LongPrediction, error) {
			g.log(a, "long", []string{state.(string)}, qs, ac)
			return agent.LongPrediction{Prediction: stubPrediction(a, g, state, 5)}, nil
		},
	}
}

// pyCall renders one Python stub call in the rig's format.
func pyCall(t *testing.T, c fixtures.Object) string {
	t.Helper()
	num := func(k string) int {
		if v, ok := c.Get(k); ok && v != nil {
			return fixtures.Int(t, v)
		}
		return 0
	}
	lang := ""
	if v, ok := c.Get("lang"); ok && v != nil {
		lang = fixtures.Str(t, v)
	}
	var states []string
	for _, s := range fixtures.List(t, c.Must("states")) {
		states = append(states, fixtures.Str(t, s))
	}
	sorted := false
	if v, ok := c.Get("sort"); ok {
		sorted = fixtures.Bool(t, v)
	}
	return fmt.Sprintf("%d|%s|%s|%s|%s|%d|%t|%d|%d|%d|%d", num("agent"), str(t, c, "m"),
		strings.Join(states, "\x1f"), str(t, c, "q"), lang, num("batch_size"), sorted, num("max_len"), num("head_max_len"),
		num("window"), num("stride"))
}

// routedHook is the golden's hook: it logs the three predict events and acts on its behaviour.
type routedHook struct {
	name  string
	parts [][2]string
	rig   *routedRig
	pools map[string]schema.Questions
}

func (h *routedHook) hit(pc *agent.PredictContext, arg string) bool {
	return strings.Contains(fmt.Sprint(pc.States[0]), arg)
}

func (h *routedHook) add(s string) {
	h.rig.mu.Lock()
	h.rig.events = append(h.rig.events, s)
	h.rig.mu.Unlock()
}

func (h *routedHook) OnPredictStart(pc *agent.PredictContext) {
	d := pc.Decision.(router.Decision)
	h.add(fmt.Sprintf("%s.start:%s:%s:%v", h.name, pc.Model, d.Model, pc.States[0]))
	for _, p := range h.parts {
		switch {
		case p[0] == "skip" && h.hit(pc, p[1]):
			_ = pc.Skip([]schema.Prediction{{Model: "stub", Usage: schema.Usage{InputTokens: 7}}})
		case p[0] == "raise" && h.hit(pc, p[1]):
			panic(fmt.Errorf("boom from %s", h.name))
		case p[0] == "rewrite" && h.hit(pc, p[1]):
			pc.Questions = h.pools["B"]
		case p[0] == "maxlen":
			fmt.Sscan(p[1], &pc.MaxLen)
		}
	}
}

func (h *routedHook) OnPredictEnd(pc *agent.PredictContext) {
	used := "None"
	if pc.Usage != (agent.HookUsage{}) { // every golden result has tokens, so zero means "never summed"
		used = fmt.Sprint(pc.Usage.InputTokens)
	}
	status := "ok"
	if pc.Err != nil {
		status = "err"
	}
	h.add(fmt.Sprintf("%s.end:%v:%s:%s", h.name, pc.States[0], used, status))
	for _, p := range h.parts {
		if p[0] == "raise_end" && h.hit(pc, p[1]) {
			panic(fmt.Errorf("boom from %s", h.name))
		}
	}
}

func (h *routedHook) OnError(pc *agent.PredictContext) {
	h.add(fmt.Sprintf("%s.error:%v:%v", h.name, pc.States[0], pc.Err))
}

func routedCallOptions(t *testing.T, kw fixtures.Object) []router.CallOption {
	var opts []router.CallOption
	for _, m := range kw {
		switch m.Key {
		case "s", "q":
		case "model":
			opts = append(opts, router.ForModel(fixtures.Str(t, m.Value)))
		case "task":
			opts = append(opts, router.ForTask(fixtures.Str(t, m.Value)))
		case "lang":
			opts = append(opts, router.ForLang(fixtures.Str(t, m.Value)))
		case "max_len":
			opts = append(opts, router.ForMaxLen(fixtures.Int(t, m.Value)))
		case "head_max_len":
			opts = append(opts, router.ForHeadMaxLen(fixtures.Int(t, m.Value)))
		case "window":
			opts = append(opts, router.ForWindow(fixtures.Int(t, m.Value)))
		case "stride":
			opts = append(opts, router.ForStride(fixtures.Int(t, m.Value)))
		case "batch_size":
			opts = append(opts, router.ForBatchSize(fixtures.Int(t, m.Value)))
		default:
			t.Fatalf("unhandled routed kwarg %q", m.Key)
		}
	}
	return opts
}

func TestRoutedGolden(t *testing.T) {
	root := fixtures.Load(t, "go_extra.json")
	cases := fixtures.List(t, root.Must("routed"))
	if len(cases) != 65 {
		t.Fatalf("%d cases", len(cases))
	}
	var withError, withEvents, batches int
	for _, c := range cases {
		rec := fixtures.Obj(t, c)
		t.Run(str(t, rec, "case"), func(t *testing.T) {
			res := replayRouted(t, rec)
			if res.err {
				withError++
			}
			if res.events {
				withEvents++
			}
			if res.batch {
				batches++
			}
		})
	}
	if withError < 8 || withEvents < 15 || batches < 18 {
		t.Errorf("golden thin: %d errors, %d with events, %d batches", withError, withEvents, batches)
	}
}

type routedOutcome struct{ err, events, batch bool }

// replayRouted runs one recorded case against the Router and checks it; the fuzz test replays the
// same records.
func replayRouted(t *testing.T, rec fixtures.Object) (res routedOutcome) {
	rig := &routedRig{pools: map[string]schema.Questions{}, idx: map[*agent.Agent]int{}}
	for _, p := range fixtures.Obj(t, rec.Must("pools")) {
		rig.pools[p.Key] = questionsOf(t, p.Value)
	}
	states := fixtures.Obj(t, rec.Must("states"))
	var hooks []any
	for _, h := range fixtures.List(t, rec.Must("hooks")) {
		pair := fixtures.List(t, h)
		hk := &routedHook{name: fixtures.Str(t, pair[0]), rig: rig, pools: rig.pools}
		for _, part := range strings.Split(fixtures.Str(t, pair[1]), ";") {
			if part != "" {
				k, v, _ := strings.Cut(part, ":")
				hk.parts = append(hk.parts, [2]string{k, v})
			}
		}
		hooks = append(hooks, hk)
	}
	r, err := router.New(router.WithInstalledHooks(hooks...))
	if err != nil {
		t.Fatal(err)
	}
	var agentOpts []agent.Option
	if fixtures.Bool(t, rec.Must("temps")) {
		agentOpts = append(agentOpts, agent.WithLangTemperatures(map[string]map[string]any{
			"de": {"temperature": []any{1.0, 1.0, 1.0}}}))
	}
	for i, name := range []string{"english", "multilingual", "typed-decisions"} {
		a, err := agent.Open(toyCheckpoint(), &probeSession{}, agentOpts...)
		if err != nil {
			t.Fatal(err)
		}
		rig.idx[a] = i + 1
		defer a.Close()
		if err := r.Attach(name, a); err != nil {
			t.Fatal(err)
		}
	}
	router.SetRunner(r, rig.runner())

	request := func(item any) router.Request {
		kw := fixtures.Obj(t, item)
		var state any
		if v := kw.Must("s"); v != nil {
			key := fixtures.Str(t, v)
			state = key
			if s, ok := states.Get(key); ok {
				state = fixtures.Str(t, s)
			}
		}
		return router.Request{State: state, Questions: rig.pools[str(t, kw, "q")], Options: routedCallOptions(t, kw)}
	}
	reqs := fixtures.List(t, rec.Must("requests"))
	var opts []router.CallOption
	if v := rec.Must("min_confidence"); v != nil {
		opts = append(opts, router.ForMinConfidence(fixtures.Float(t, v)))
	}
	var got []router.Routed
	var callErr error
	switch method := str(t, rec, "method"); method {
	case "batch":
		res.batch = true
		bs := 0
		if v := rec.Must("batch_size"); v != nil {
			bs = fixtures.Int(t, v)
		}
		if fixtures.Bool(t, rec.Must("sort")) {
			opts = append(opts, router.ForSortByLength(true))
		}
		var rs []router.Request
		for _, it := range reqs {
			rs = append(rs, request(it))
		}
		got, callErr = r.PredictBatch(context.Background(), rs, bs, opts...)
	default:
		rq := request(reqs[0])
		opts = append(opts, rq.Options...)
		opts = append(opts, routedCallOptions(t, fixtures.Obj(t, rec.Must("call")))...)
		var one router.Routed
		if method == "predict" {
			one, callErr = r.Predict(context.Background(), rq.State, rq.Questions, opts...)
		} else {
			one, callErr = r.PredictLong(context.Background(), rq.State, rq.Questions, opts...)
		}
		if callErr == nil {
			got = []router.Routed{one}
		}
	}

	if want, bad := rec.Get("error"); bad {
		res.err = true
		if callErr == nil || callErr.Error() != fixtures.Str(t, want) {
			t.Fatalf("error = %v, want %q", callErr, want)
		}
	} else if callErr != nil {
		t.Fatalf("unexpected error %v", callErr)
	} else {
		wantRes := fixtures.List(t, rec.Must("results"))
		if len(got) != len(wantRes) {
			t.Fatalf("%d results, want %d", len(got), len(wantRes))
		}
		for i, w := range wantRes {
			wo := fixtures.Obj(t, w)
			if got[i].Prediction.Usage.InputTokens != fixtures.Int(t, wo.Must("in")) ||
				got[i].Prediction.Usage.OutputTokens != fixtures.Int(t, wo.Must("out")) ||
				string(got[i].Routing.Model) != str(t, wo, "model") || got[i].Routing.Reason != str(t, wo, "reason") {
				t.Errorf("result %d = %d/%d %s %q, want %v", i, got[i].Prediction.Usage.InputTokens,
					got[i].Prediction.Usage.OutputTokens, got[i].Routing.Model, got[i].Routing.Reason, w)
			}
		}
	}
	var wantCalls []string
	for _, c := range fixtures.List(t, rec.Must("calls")) {
		wantCalls = append(wantCalls, pyCall(t, fixtures.Obj(t, c)))
	}
	if strings.Join(rig.calls, "\n") != strings.Join(wantCalls, "\n") {
		t.Errorf("calls:\n%s\nwant:\n%s", strings.Join(rig.calls, "\n"), strings.Join(wantCalls, "\n"))
	}
	var wantEvents []string
	for _, e := range fixtures.List(t, rec.Must("events")) {
		wantEvents = append(wantEvents, fixtures.Str(t, e))
	}
	if len(wantEvents) > 0 {
		res.events = true
	}
	if strings.Join(rig.events, "\n") != strings.Join(wantEvents, "\n") {
		t.Errorf("events:\n%s\nwant:\n%s", strings.Join(rig.events, "\n"), strings.Join(wantEvents, "\n"))
	}
	var loaded []string
	for _, k := range r.Loaded() {
		loaded = append(loaded, string(k))
	}
	var wantLoaded []string
	for _, k := range fixtures.List(t, rec.Must("loaded")) {
		wantLoaded = append(wantLoaded, fixtures.Str(t, k))
	}
	if strings.Join(loaded, ",") != strings.Join(wantLoaded, ",") {
		t.Errorf("loaded = %v, want %v", loaded, wantLoaded)
	}
	return res
}
