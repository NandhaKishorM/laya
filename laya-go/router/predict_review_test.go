package router_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/router"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func lowPrediction() schema.Prediction {
	return schema.Prediction{Model: "stub", Usage: schema.Usage{InputTokens: 5},
		Answers: schema.Answers{{ID: "intent", Answer: &schema.ChoiceAnswer{Common: schema.Common{Confidence: 0.4, AnswerConfidence: 0.4}}}}}
}

func abstention(p schema.Prediction) string {
	return p.Answers[0].Answer.(*schema.ChoiceAnswer).Abstention
}

func lowRunner() router.Runner {
	return router.Runner{
		Predict: func(context.Context, *agent.Agent, any, schema.Questions, router.AgentCall) (schema.Prediction, error) {
			return lowPrediction(), nil
		},
		Batch: func(_ context.Context, _ *agent.Agent, states []any, _ schema.Questions, _ router.AgentCall) ([]schema.Prediction, error) {
			out := make([]schema.Prediction, len(states))
			for i := range out {
				out[i] = lowPrediction()
			}
			return out, nil
		},
	}
}

// gateProbe skips requests that say SKIP (with an answer the gate acts on), fails those that say
// BOOM, and records what the answer looked like when each hook saw it.
type gateProbe struct {
	agent.BaseHook
	mu         sync.Mutex
	atStart    []string
	atEnd      []string
	atErrorEnd []string
}

func (h *gateProbe) OnPredictStart(pc *agent.PredictContext) {
	s := fmt.Sprint(pc.States[0])
	if strings.Contains(s, "SKIP") {
		_ = pc.Skip([]schema.Prediction{lowPrediction()})
	}
	if strings.Contains(s, "BOOM") {
		panic("boom")
	}
	h.mu.Lock()
	h.atStart = append(h.atStart, fmt.Sprintf("%d", len(pc.Results)))
	h.mu.Unlock()
}

func (h *gateProbe) OnPredictEnd(pc *agent.PredictContext) {
	if len(pc.Results) == 0 {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if pc.Err != nil {
		h.atErrorEnd = append(h.atErrorEnd, abstention(pc.Results[0]))
		return
	}
	h.atEnd = append(h.atEnd, abstention(pc.Results[0]))
}

func TestGateActsOnPredictAndEndHooksSeeIt(t *testing.T) {
	f := &scoringFactory{}
	h := &gateProbe{}
	r := scoringRouter(t, f, router.WithInstalledHooks(h))
	router.SetRunner(r, lowRunner())
	for _, state := range []string{enText, "SKIP " + enText} { // inferred, and answered by a hook
		got, err := r.Predict(context.Background(), state, qsA(t), router.ForMinConfidence(0.9))
		if err != nil {
			t.Fatal(err)
		}
		if abstention(got.Prediction) != "abstained" {
			t.Errorf("%q: Predict result not gated: %q", state, abstention(got.Prediction))
		}
	}
	if strings.Join(h.atEnd, ",") != "abstained,abstained" {
		t.Errorf("end hooks saw %v: the gate must run before them", h.atEnd)
	}
	got, err := r.Predict(context.Background(), enText, qsA(t))
	if err != nil || abstention(got.Prediction) != "" {
		t.Errorf("without ForMinConfidence the answer was gated: %q (%v)", abstention(got.Prediction), err)
	}
}

func TestGateActsOnBatchSuccessAndFailure(t *testing.T) {
	f := &scoringFactory{}
	h := &gateProbe{}
	r := scoringRouter(t, f, router.WithInstalledHooks(h))
	router.SetRunner(r, lowRunner())
	qs := qsA(t)
	reqs := []router.Request{{State: enText, Questions: qs}, {State: "SKIP " + enText, Questions: qs}}
	out, err := r.PredictBatch(context.Background(), reqs, 0, router.ForMinConfidence(0.9))
	if err != nil {
		t.Fatal(err)
	}
	for i, o := range out {
		if abstention(o.Prediction) != "abstained" {
			t.Errorf("batch result %d not gated: %q", i, abstention(o.Prediction))
		}
	}
	if strings.Join(h.atEnd, ",") != "abstained,abstained" {
		t.Errorf("batch end hooks saw %v", h.atEnd)
	}
	// A failing group: the request that was already answered still ends gated.
	h.atEnd, h.atErrorEnd = nil, nil
	reqs = []router.Request{{State: "SKIP " + enText, Questions: qs}, {State: "BOOM " + enText, Questions: qs}}
	if _, err := r.PredictBatch(context.Background(), reqs, 0, router.ForMinConfidence(0.9)); err == nil {
		t.Fatal("a raising start hook did not fail the batch")
	}
	if strings.Join(h.atErrorEnd, ",") != "abstained" {
		t.Errorf("the answered request ended with %v on the failure path, want it gated", h.atErrorEnd)
	}
}

// The real agentRunner turns AgentCall into agent options; the stub Runner never exercises that.
func TestRealAgentGetsTheBudgetsBatchSizeAndSort(t *testing.T) {
	long := strings.Repeat("please refund my order ", 20)
	run := func(f *scoringFactory, do func(r *router.Router)) []([2]int) {
		r := scoringRouter(t, f)
		do(r)
		var all [][2]int
		for _, s := range f.sessions {
			all = append(all, s.shapes...)
		}
		return all
	}
	ctx, qs := context.Background(), qsA(t)
	// max_len
	plain := run(&scoringFactory{}, func(r *router.Router) { _, _ = r.Predict(ctx, long, qs, router.ForModel("english")) })
	capped := run(&scoringFactory{}, func(r *router.Router) {
		_, _ = r.Predict(ctx, long, qs, router.ForModel("english"), router.ForMaxLen(64))
	})
	if len(plain) != 1 || len(capped) != 1 || capped[0][1] > 64 || plain[0][1] <= 64 {
		t.Errorf("ForMaxLen: lengths plain %v capped %v", plain, capped)
	}
	// head_max_len
	wide := run(&scoringFactory{}, func(r *router.Router) {
		_, _ = r.Predict(ctx, "refund it", qs, router.ForModel("english"), router.ForMaxLen(64), router.ForHeadMaxLen(56))
	})
	narrow := run(&scoringFactory{}, func(r *router.Router) {
		_, _ = r.Predict(ctx, "refund it", qs, router.ForModel("english"), router.ForMaxLen(64), router.ForHeadMaxLen(8))
	})
	if len(wide) != 1 || len(narrow) != 1 || narrow[0][1] >= wide[0][1] {
		t.Errorf("ForHeadMaxLen: lengths wide %v narrow %v", wide, narrow)
	}
	// batch size and sort: alternating long and short states
	var reqs []router.Request
	for i := 0; i < 6; i++ {
		s := "refund"
		if i%2 == 0 {
			s = long[:150]
		}
		reqs = append(reqs, router.Request{State: s, Questions: qs, Options: []router.CallOption{router.ForModel("english")}})
	}
	cells := func(shapes [][2]int) (n int) {
		for _, s := range shapes {
			n += s[0] * s[1]
		}
		return
	}
	one := run(&scoringFactory{}, func(r *router.Router) { _, _ = r.PredictBatch(ctx, reqs, 0) })
	by2 := run(&scoringFactory{}, func(r *router.Router) { _, _ = r.PredictBatch(ctx, reqs, 2) })
	sorted := run(&scoringFactory{}, func(r *router.Router) { _, _ = r.PredictBatch(ctx, reqs, 2, router.ForSortByLength(true)) })
	if len(one) != 1 || len(by2) != 3 {
		t.Errorf("batch size: %d runs for 0, %d for 2 (want 1 and 3)", len(one), len(by2))
	}
	if len(sorted) != 3 || cells(sorted) >= cells(by2) {
		t.Errorf("sort: padded cells %d sorted, %d unsorted", cells(sorted), cells(by2))
	}
}

func TestBatchTypedNilStateRefusedBeforeRouting(t *testing.T) {
	f := &scoringFactory{}
	r := scoringRouter(t, f)
	var p *string
	_, err := r.PredictBatch(context.Background(), []router.Request{{State: enText, Questions: qsA(t)}, {State: p, Questions: qsA(t)}}, 0)
	if err == nil || !strings.Contains(err.Error(), "request 1 'state' must not be None") || f.count() != 0 {
		t.Errorf("err %v, builds %d", err, f.count())
	}
}

func TestCancelledContextBuildsNothing(t *testing.T) {
	f := &scoringFactory{}
	routed := 0
	r := scoringRouter(t, f, router.WithInstalledHooks(&router.HookFuncs{OnRouteFunc: func(*router.RouteContext) { routed++ }}))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, e1 := r.Predict(ctx, enText, qsA(t))
	_, e2 := r.PredictBatch(ctx, []router.Request{{State: enText, Questions: qsA(t)}}, 0)
	_, e3 := r.PredictLong(ctx, enText, qsA(t))
	if !errors.Is(e1, context.Canceled) || !errors.Is(e2, context.Canceled) || !errors.Is(e3, context.Canceled) || f.count() != 0 || routed != 0 {
		t.Errorf("errors %v %v %v, builds %d, route hooks %d", e1, e2, e3, f.count(), routed)
	}
}

func TestBatchOptionsAreValidatedEvenWithNoRequests(t *testing.T) {
	r := scoringRouter(t, &scoringFactory{})
	var mc *agent.MinConfidenceError
	if _, err := r.PredictBatch(context.Background(), nil, 0, router.ForMinConfidence(7)); !errors.As(err, &mc) {
		t.Errorf("bad gate with no requests: %v", err)
	}
	if _, err := r.PredictBatch(context.Background(), nil, 0, router.ForHookTimeout(-1)); err == nil {
		t.Error("bad timeout with no requests accepted")
	}
	req := router.Request{State: enText, Questions: qsA(t), Options: []router.CallOption{router.ForMinConfidence(0.5)}}
	if _, err := r.PredictBatch(context.Background(), []router.Request{req}, 0); err == nil {
		t.Error("a per-request gate was silently ignored")
	}
	if _, err := r.PredictLong(context.Background(), enText, qsA(t), router.ForMaxLen(-5)); err != nil {
		t.Errorf("ForMaxLen(0) is no override: %v", err)
	}
}

func TestEmptyLanguageIsItsOwnGroup(t *testing.T) {
	temps := agent.WithLangTemperatures(map[string]map[string]any{"de": {"temperature": []any{3.0, 3.0, 3.0}}})
	f := &scoringFactory{opts: []agent.Option{temps}}
	r := scoringRouter(t, f)
	passes := 0
	router.SetRunner(r, router.Runner{Batch: func(_ context.Context, _ *agent.Agent, states []any, _ schema.Questions, _ router.AgentCall) ([]schema.Prediction, error) {
		passes++
		return make([]schema.Prediction, len(states)), nil
	}})
	qs := qsA(t)
	reqs := []router.Request{
		{State: "12345", Questions: qs, Options: []router.CallOption{router.ForModel("multilingual"), router.ForLang("")}},
		{State: "12345", Questions: qs, Options: []router.CallOption{router.ForModel("multilingual")}},
	}
	if _, err := r.PredictBatch(context.Background(), reqs, 0); err != nil {
		t.Fatal(err)
	}
	if passes != 2 {
		t.Errorf("%d passes: an explicit empty language and none are different groups in laya", passes)
	}
}

type clearDecision struct{ agent.BaseHook }

func (clearDecision) OnPredictStart(pc *agent.PredictContext) { pc.Decision = nil }

func TestScanLanguageAfterAHookClearsTheDecision(t *testing.T) {
	f := &scoringFactory{}
	r := scoringRouter(t, f, router.WithInstalledHooks(&clearDecision{}))
	var lang string
	router.SetRunner(r, router.Runner{Long: func(_ context.Context, _ *agent.Agent, _ any, _ schema.Questions, ac router.AgentCall) (agent.LongPrediction, error) {
		lang = ac.Lang
		return agent.LongPrediction{}, nil
	}})
	if _, err := r.PredictLong(context.Background(), deText, qsA(t)); err != nil {
		t.Fatal(err)
	}
	if lang != "" {
		t.Errorf("scan got language %q after a hook cleared the decision, want none (Python: ctx.decision or {})", lang)
	}
}

type panicStart struct{ agent.BaseHook }

func (panicStart) OnPredictStart(pc *agent.PredictContext) {
	if strings.Contains(fmt.Sprint(pc.States[0]), "BOOM") {
		panic("boom")
	}
}

func TestBatchFailureReleasesTheLease(t *testing.T) {
	for _, raise := range []bool{true, false} {
		f := &scoringFactory{}
		r := scoringRouter(t, f, router.WithInstalledHooks(&panicStart{}), router.WithHookErrors(raise))
		qs := qsA(t)
		_, _ = r.PredictBatch(context.Background(), []router.Request{{State: enText, Questions: qs}, {State: "BOOM " + enText, Questions: qs}}, 0)
		if _, err := r.UnloadAll(context.Background()); err != nil {
			t.Fatal(err)
		}
		for _, s := range f.sessions {
			if s.closes.Load() != 1 {
				t.Errorf("raise=%v: session closed %d times after UnloadAll: lease leaked", raise, s.closes.Load())
			}
		}
	}
}

type setMaxLen struct{ agent.BaseHook }

func (setMaxLen) OnPredictStart(pc *agent.PredictContext) {
	if strings.Contains(fmt.Sprint(pc.States[0]), "WIDE") {
		pc.MaxLen = 99
	}
}

func TestBatchHookBudgetSplitsAndKeepsOrder(t *testing.T) {
	f := &scoringFactory{}
	r := scoringRouter(t, f, router.WithInstalledHooks(&setMaxLen{}))
	var calls []router.AgentCall
	router.SetRunner(r, router.Runner{Batch: func(_ context.Context, _ *agent.Agent, states []any, _ schema.Questions, ac router.AgentCall) ([]schema.Prediction, error) {
		calls = append(calls, ac)
		out := make([]schema.Prediction, len(states))
		for i, s := range states {
			out[i] = schema.Prediction{Model: fmt.Sprint(s)}
		}
		return out, nil
	}})
	qs := qsA(t)
	states := []string{enText, "WIDE " + enText, enText + " two", "WIDE again " + enText}
	var reqs []router.Request
	for _, s := range states {
		reqs = append(reqs, router.Request{State: s, Questions: qs, Options: []router.CallOption{router.ForModel("english"), router.ForMaxLen(50)}})
	}
	got, err := r.PredictBatch(context.Background(), reqs, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i, g := range got {
		if g.Prediction.Model != states[i] {
			t.Errorf("result %d is %q, want %q", i, g.Prediction.Model, states[i])
		}
	}
	if len(calls) != 2 || calls[0].MaxLen != 50 || calls[1].MaxLen != 99 {
		t.Errorf("calls %+v", calls)
	}
}
