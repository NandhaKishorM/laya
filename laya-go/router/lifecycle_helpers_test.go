package router_test

import (
	"context"
	"errors"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"testing/fstest"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/router"
)

// probeSession runs nothing and counts its Close calls.
type probeSession struct {
	closes   atomic.Int32
	closeErr error
	// hold, when set, blocks Close until it is closed.
	hold chan struct{}
	// inClose is closed when Close is entered, if set.
	inClose chan struct{}
	once    sync.Once
}

func (s *probeSession) Run(context.Context, infer.Batch) (infer.Output, error) {
	return infer.Output{}, errors.New("probe session runs nothing")
}

func (s *probeSession) Close() error {
	s.closes.Add(1)
	if s.inClose != nil {
		s.once.Do(func() { close(s.inClose) })
	}
	if s.hold != nil {
		<-s.hold
	}
	return s.closeErr
}

func toyCheckpoint() fstest.MapFS {
	spec, cfg := fixtures.ToyTokenizer()
	return fstest.MapFS{
		"rl_agent_config.json":            {Data: []byte(`{}`)},
		"tokenizer/tokenizer.json":        {Data: spec},
		"tokenizer/tokenizer_config.json": {Data: cfg},
	}
}

func newProbe(tb testing.TB, s *probeSession) *agent.Agent {
	tb.Helper()
	if s == nil {
		s = &probeSession{}
	}
	a, err := agent.Open(toyCheckpoint(), s)
	if err != nil {
		tb.Fatal(err)
	}
	return a
}

// built is one agent a fakeFactory made.
type built struct {
	label string
	agent *agent.Agent
	sess  *probeSession
}

// fakeFactory is a deterministic AgentFactory: it counts calls, can fail or block per checkpoint,
// and remembers every agent it made.
type fakeFactory struct {
	t  testing.TB
	mu sync.Mutex
	n  int
	// fail maps a checkpoint to the error its builds return.
	fail map[router.Checkpoint]error
	// gate blocks a build of the checkpoint until the channel is closed; entered, if present, is
	// sent to when a build starts.
	gate    map[router.Checkpoint]chan struct{}
	entered chan router.Checkpoint
	// panicOn makes a build of the checkpoint panic.
	panicOn map[router.Checkpoint]bool
	made    []*built
	byAgent map[*agent.Agent]*built
}

func newFactory(t testing.TB) *fakeFactory {
	return &fakeFactory{t: t, fail: map[router.Checkpoint]error{}, gate: map[router.Checkpoint]chan struct{}{},
		panicOn: map[router.Checkpoint]bool{}, byAgent: map[*agent.Agent]*built{}}
}

func (f *fakeFactory) build(ctx context.Context, c router.Checkpoint) (*agent.Agent, error) {
	f.mu.Lock()
	f.n++
	label := "built-" + strconv.Itoa(f.n)
	gate, entered, err, boom := f.gate[c], f.entered, f.fail[c], f.panicOn[c]
	f.mu.Unlock()
	if entered != nil {
		entered <- c
	}
	if gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if boom {
		panic("factory exploded for " + string(c))
	}
	if err != nil {
		return nil, err
	}
	s := &probeSession{}
	a := newProbe(f.t, s)
	b := &built{label: label, agent: a, sess: s}
	f.mu.Lock()
	f.made = append(f.made, b)
	f.byAgent[a] = b
	f.mu.Unlock()
	return a, nil
}

func (f *fakeFactory) calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.n
}

func (f *fakeFactory) of(a *agent.Agent) *built {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.byAgent[a]
}

func (f *fakeFactory) setFail(c router.Checkpoint, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err == nil {
		delete(f.fail, c)
	} else {
		f.fail[c] = err
	}
}

func (f *fakeFactory) open(c router.Checkpoint) chan struct{} {
	f.mu.Lock()
	defer f.mu.Unlock()
	g := make(chan struct{})
	f.gate[c] = g
	return g
}

func newRouter(t testing.TB, f *fakeFactory, opts ...router.Option) *router.Router {
	t.Helper()
	opts = append([]router.Option{router.WithAgentFactory(f.build)}, opts...)
	r, err := router.New(opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}

// within fails the test if fn does not return in time, instead of hanging the run.
func within(t testing.TB, d time.Duration, what string, fn func()) {
	t.Helper()
	done := make(chan struct{})
	go func() { defer close(done); fn() }()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatalf("%s did not finish within %v", what, d)
	}
}

func names(cs []router.Checkpoint) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = string(c)
	}
	return out
}
