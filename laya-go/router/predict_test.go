package router_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/router"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// scoreSession is a session that scores: it prefers the second option of every question. It counts
// its Run and Close calls and can hold a Run until released.
type scoreSession struct {
	runs, closes atomic.Int32
	entered      chan struct{} // sent to (non-blocking) when a Run starts, if set
	hold         chan struct{} // a Run waits for it to close, if set
}

func (s *scoreSession) Run(ctx context.Context, b infer.Batch) (infer.Output, error) {
	s.runs.Add(1)
	if s.entered != nil {
		select {
		case s.entered <- struct{}{}:
		default:
		}
	}
	if s.hold != nil {
		select {
		case <-s.hold:
		case <-ctx.Done():
			return infer.Output{}, ctx.Err()
		}
	}
	out := infer.Output{Logits: make([]float32, b.Rows*b.Markers), ActLogits: make([]float32, b.Rows*2)}
	for i := 0; i < b.Rows; i++ {
		out.Logits[i*b.Markers+1] = 2
	}
	return out, nil
}

func (s *scoreSession) Close() error { s.closes.Add(1); return nil }

// scoringFactory builds a scoring agent per checkpoint and remembers each build.
type scoringFactory struct {
	mu       sync.Mutex
	builds   int
	sessions []*scoreSession
	byAgent  map[*agent.Agent]*scoreSession
	opts     []agent.Option
	hold     chan struct{}
	entered  chan struct{}
}

func (f *scoringFactory) build(_ context.Context, _ router.Checkpoint) (*agent.Agent, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.builds++
	s := &scoreSession{hold: f.hold, entered: f.entered}
	a, err := agent.Open(toyCheckpoint(), s, f.opts...)
	if err != nil {
		return nil, err
	}
	f.sessions = append(f.sessions, s)
	if f.byAgent == nil {
		f.byAgent = map[*agent.Agent]*scoreSession{}
	}
	f.byAgent[a] = s
	return a, nil
}

func (f *scoringFactory) count() int { f.mu.Lock(); defer f.mu.Unlock(); return f.builds }

func scoringRouter(t testing.TB, f *scoringFactory, opts ...router.Option) *router.Router {
	t.Helper()
	r, err := router.New(append([]router.Option{router.WithAgentFactory(f.build)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}

func qsA(t testing.TB) schema.Questions {
	t.Helper()
	qs, err := schema.NewQuestions(schema.Ask("intent", schema.MustQuestion(schema.ChoiceOf("What?", "status", "refund"))))
	if err != nil {
		t.Fatal(err)
	}
	return qs
}

func qsB(t testing.TB) schema.Questions {
	t.Helper()
	qs, err := schema.NewQuestions(schema.Ask("intent", schema.MustQuestion(schema.ChoiceOf("What?", "refund", "status"))))
	if err != nil {
		t.Fatal(err)
	}
	return qs
}

const (
	enText = "please refund my order"
	deText = "bitte erstatten Sie mir das Geld für die Bestellung"
	zhText = "请退款给我"
)

func TestPredictValidatesBeforeRouting(t *testing.T) {
	f := &scoringFactory{}
	r := scoringRouter(t, f)
	ctx := context.Background()
	qs := qsA(t)
	if _, err := r.Predict(ctx, nil, qs); err == nil || err.Error() != "state must not be None; pass a string, dict, or list" {
		t.Errorf("nil state: %v", err)
	}
	var mc *agent.MinConfidenceError
	if _, err := r.Predict(ctx, enText, qs, router.ForMinConfidence(2)); !errors.As(err, &mc) {
		t.Errorf("min_confidence: %v", err)
	}
	if _, err := r.Predict(ctx, enText, qs, router.ForHookTimeout(-time.Second)); err == nil {
		t.Error("negative timeout accepted")
	}
	if _, err := r.PredictLong(ctx, nil, qs); err == nil {
		t.Error("PredictLong accepted a nil state")
	}
	reqs := []router.Request{{State: enText, Questions: qs}, {State: nil, Questions: qs}}
	if _, err := r.PredictBatch(ctx, reqs, 0); err == nil || !strings.Contains(err.Error(), "request 1 'state' must not be None") {
		t.Errorf("batch nil state: %v", err)
	}
	if _, err := r.PredictBatch(ctx, reqs[:1], 0, router.ForMinConfidence(-0.5)); !errors.As(err, &mc) {
		t.Errorf("batch min_confidence: %v", err)
	}
	if _, err := r.PredictBatch(ctx, []router.Request{{State: enText, Questions: qs, Options: []router.CallOption{router.ForModel("nope")}}}, 0); err == nil {
		t.Error("unknown model accepted")
	}
	if n := f.count(); n != 0 {
		t.Fatalf("%d checkpoints built by calls that were refused before routing", n)
	}
}

func TestPredictRoutesAndAnswers(t *testing.T) {
	f := &scoringFactory{}
	r := scoringRouter(t, f)
	got, err := r.Predict(context.Background(), deText, qsA(t))
	if err != nil {
		t.Fatal(err)
	}
	if got.Routing.Model != router.Multilingual || got.Prediction.Usage.InputTokens == 0 {
		t.Fatalf("routing %v usage %d", got.Routing.Model, got.Prediction.Usage.InputTokens)
	}
	if a, _ := got.Prediction.Answers.Get("intent"); a.(*schema.ChoiceAnswer).Choice.String() != "refund" {
		t.Errorf("answer %v", a)
	}
	// PredictRouted does not route: the decision is the caller's, even against the text.
	d := router.Decision{Model: router.English, Reason: "mine"}
	got, err = r.PredictRouted(context.Background(), d, deText, qsA(t))
	if err != nil || got.Routing.Reason != "mine" || got.Routing.Model != router.English {
		t.Fatalf("PredictRouted = %+v, %v", got.Routing, err)
	}
	if f.count() != 2 {
		t.Errorf("builds %d", f.count())
	}
}

// langProb is the probability of the preferred option, which a language temperature moves.
func langProb(t *testing.T, p schema.Prediction) float64 {
	t.Helper()
	a, _ := p.Answers.Get("intent")
	return a.(*schema.ChoiceAnswer).AnswerConfidence
}

func TestLanguageReachesTheAgent(t *testing.T) {
	temps := agent.WithLangTemperatures(map[string]map[string]any{"de": {"temperature": []any{3.0, 3.0, 3.0}}})
	f := &scoringFactory{opts: []agent.Option{temps}}
	r := scoringRouter(t, f)
	ctx, qs := context.Background(), qsA(t)
	plain, err := r.Predict(ctx, enText, qs, router.ForModel("multilingual"))
	if err != nil {
		t.Fatal(err)
	}
	for name, run := range map[string]func() (float64, error){
		"explicit": func() (float64, error) {
			p, err := r.Predict(ctx, enText, qs, router.ForModel("multilingual"), router.ForLang("de"))
			return langProb(t, p.Prediction), err
		},
		"detected": func() (float64, error) {
			p, err := r.Predict(ctx, deText, qs)
			return langProb(t, p.Prediction), err
		},
		"batch": func() (float64, error) {
			ps, err := r.PredictBatch(ctx, []router.Request{{State: deText, Questions: qs}}, 0)
			if err != nil {
				return 0, err
			}
			return langProb(t, ps[0].Prediction), nil
		},
	} {
		got, err := run()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got >= langProb(t, plain.Prediction) {
			t.Errorf("%s: confidence %v is not below the untempered %v: the language did not reach the agent", name, got, langProb(t, plain.Prediction))
		}
	}
}

func TestBatchGroupsAndRestoresOrder(t *testing.T) {
	f := &scoringFactory{}
	r := scoringRouter(t, f, router.WithMaxLoaded(3))
	states := []string{enText, deText, enText + " now", zhText, deText + " bitte", "where is my parcel"}
	var reqs []router.Request
	for i, s := range states {
		qs := qsA(t)
		if i == 2 {
			qs = qsB(t)
		}
		reqs = append(reqs, router.Request{State: s, Questions: qs})
	}
	got, err := r.PredictBatch(context.Background(), reqs, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []router.Checkpoint{router.English, router.Multilingual, router.English, router.Multilingual, router.Multilingual, router.English}
	for i, g := range got {
		single, err := r.Predict(context.Background(), states[i], reqs[i].Questions)
		if err != nil {
			t.Fatal(err)
		}
		if g.Routing.Model != want[i] || g.Prediction.Usage.InputTokens != single.Prediction.Usage.InputTokens {
			t.Errorf("result %d: %s %d, want %s %d", i, g.Routing.Model, g.Prediction.Usage.InputTokens, want[i], single.Prediction.Usage.InputTokens)
		}
	}
	// english: A-schema pair (0, 5) share a pass, B (2) is its own; multilingual: one pass for 1, 3, 4.
	runs := 0
	for _, s := range f.sessions {
		runs += int(s.runs.Load())
	}
	if runs != 6+3 { // 6 single Predicts, plus 3 batch passes (the 3-row group is one run)
		t.Errorf("%d session runs, want 9", runs)
	}
}

type skipMiddle struct {
	agent.BaseHook
	needle string
}

func (h *skipMiddle) OnPredictStart(pc *agent.PredictContext) {
	if strings.Contains(fmt.Sprint(pc.States[0]), h.needle) {
		_ = pc.Skip([]schema.Prediction{{Model: "cached", Usage: schema.Usage{InputTokens: 7}}})
	}
}

func TestSkippedRequestDoesNotShiftOthers(t *testing.T) {
	f := &scoringFactory{}
	hook := &skipMiddle{needle: "SKIPME"}
	r := scoringRouter(t, f, router.WithInstalledHooks(hook))
	qs := qsA(t)
	states := []string{enText, "SKIPME one", enText + " again", "SKIPME two", "where is my parcel"}
	var reqs []router.Request
	for _, s := range states {
		reqs = append(reqs, router.Request{State: s, Questions: qs})
	}
	got, err := r.PredictBatch(context.Background(), reqs, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range states {
		skipped := strings.HasPrefix(s, "SKIPME")
		if skipped != (got[i].Prediction.Model == "cached") {
			t.Errorf("result %d (%q): model %q", i, s, got[i].Prediction.Model)
		}
		if !skipped {
			one, err := r.Predict(context.Background(), s, qs)
			if err != nil || one.Prediction.Usage.InputTokens != got[i].Prediction.Usage.InputTokens {
				t.Errorf("result %d moved: %d vs %d (%v)", i, got[i].Prediction.Usage.InputTokens, one.Prediction.Usage.InputTokens, err)
			}
		}
	}
}

// A call keeps its agent open while another goroutine evicts the checkpoint.
func TestPredictHoldsItsLeaseThroughEviction(t *testing.T) {
	f := &scoringFactory{hold: make(chan struct{}), entered: make(chan struct{}, 1)}
	r := scoringRouter(t, f, router.WithMaxLoaded(1))
	ctx, qs := context.Background(), qsA(t)
	done := make(chan error, 1)
	go func() {
		_, err := r.Predict(ctx, enText, qs) // english
		done <- err
	}()
	<-f.entered
	f.mu.Lock()
	first := f.sessions[0]
	f.mu.Unlock()
	l, err := r.Lease(ctx, "multilingual") // evicts english while it is mid-call
	if err != nil {
		t.Fatal(err)
	}
	l.Close()
	if got := r.Loaded(); len(got) != 1 || got[0] != router.Multilingual {
		t.Fatalf("loaded = %v, english should have been evicted", got)
	}
	if first.closes.Load() != 0 {
		t.Fatal("the agent was closed under a running Predict")
	}
	close(f.hold)
	if err := <-done; err != nil {
		t.Fatalf("Predict on an evicted agent: %v", err)
	}
	if first.closes.Load() != 1 {
		t.Fatalf("agent closed %d times after the last lease, want 1", first.closes.Load())
	}
}

func TestConcurrentPredictBatchWithEviction(t *testing.T) {
	f := &scoringFactory{}
	r := scoringRouter(t, f, router.WithMaxLoaded(1))
	qs := qsA(t)
	mixed := []router.Request{{State: enText, Questions: qs}, {State: deText, Questions: qs}, {State: zhText, Questions: qs},
		{State: "where is my parcel", Questions: qs}, {State: deText + " noch", Questions: qs}}
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 6; i++ {
				reqs := append([]router.Request(nil), mixed...)
				if (g+i)%2 == 1 {
					reqs[0], reqs[1] = reqs[1], reqs[0]
				}
				got, err := r.PredictBatch(context.Background(), reqs, 2)
				if err != nil {
					errs <- err
					return
				}
				for k, req := range reqs {
					d, _ := r.Route(req.State, req.Questions)
					if got[k].Routing.Model != d.Model {
						errs <- fmt.Errorf("goroutine %d: result %d routed %s, want %s", g, k, got[k].Routing.Model, d.Model)
					}
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	for i, s := range f.sessions {
		if s.closes.Load() != 1 {
			t.Errorf("session %d closed %d times, want 1", i, s.closes.Load())
		}
	}
}

func TestPredictLongCarriesScanOptions(t *testing.T) {
	f := &scoringFactory{}
	r := scoringRouter(t, f)
	long := strings.Repeat("please refund my order ", 80)
	got, err := r.PredictLong(context.Background(), long, qsA(t), router.ForWindow(40), router.ForStride(20))
	if err != nil {
		t.Fatal(err)
	}
	if got.Prediction.Usage.Windows < 2 || got.Routing.Model != router.English {
		t.Fatalf("windows %d routed %s", got.Prediction.Usage.Windows, got.Routing.Model)
	}
	if _, err := r.PredictLong(context.Background(), long, qsA(t), router.ForWindow(40), router.ForStride(400)); err == nil {
		t.Error("a stride past the window was accepted")
	}
}

type callOrder struct {
	agent.BaseHook
	mu  sync.Mutex
	log []string
}

func (h *callOrder) add(s string) { h.mu.Lock(); h.log = append(h.log, s); h.mu.Unlock() }

func (h *callOrder) OnPredictStart(pc *agent.PredictContext) {
	h.add("start:" + fmt.Sprint(pc.States[0]))
}
func (h *callOrder) OnPredictEnd(pc *agent.PredictContext) { h.add("end:" + fmt.Sprint(pc.States[0])) }

func TestBatchEndsInReverseAndSeesDecision(t *testing.T) {
	f := &scoringFactory{}
	h := &callOrder{}
	r := scoringRouter(t, f, router.WithInstalledHooks(h))
	qs := qsA(t)
	_, err := r.PredictBatch(context.Background(), []router.Request{{State: "one a", Questions: qs}, {State: "two b", Questions: qs}, {State: "three c", Questions: qs}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := "start:one a,start:two b,start:three c,end:three c,end:two b,end:one a"
	if got := strings.Join(h.log, ","); got != want {
		t.Errorf("events %s, want %s", got, want)
	}
}
