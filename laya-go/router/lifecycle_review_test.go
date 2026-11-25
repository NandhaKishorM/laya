package router_test

import (
	"context"
	"errors"
	"math/rand"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/router"
)

type panicSession struct{ probeSession }

func (s *panicSession) Close() error { panic("session close exploded") }

func TestPanicInEvictionCloseDoesNotWedgeTheRouter(t *testing.T) {
	var evicted []string
	var mu sync.Mutex
	first := true
	f := func(ctx context.Context, c router.Checkpoint) (*agent.Agent, error) {
		if first {
			first = false
			return agent.Open(toyCheckpoint(), &panicSession{})
		}
		return newProbe(t, nil), nil
	}
	h := &router.HookFuncs{OnEvictFunc: func(c *router.EvictContext) { mu.Lock(); evicted = append(evicted, string(c.Model)); mu.Unlock() }}
	r, err := router.New(router.WithAgentFactory(f), router.WithMaxLoaded(1), router.WithInstalledHooks(h))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err := r.Load(ctx, "english"); err != nil {
		t.Fatal(err)
	}
	_, err = r.Lease(ctx, "multilingual") // evicts english, whose Close panics
	if err == nil || !strings.Contains(err.Error(), "session close exploded") {
		t.Fatalf("err = %v", err)
	}
	within(t, 2*time.Second, "Loaded after a panicking close", func() { r.Loaded() })
	mu.Lock()
	if len(evicted) != 1 || evicted[0] != "english" {
		t.Fatalf("on_evict events %v: the panic skipped them", evicted)
	}
	mu.Unlock()
	within(t, 2*time.Second, "next Load", func() {
		if _, err := r.Load(ctx, "multilingual"); err != nil {
			t.Error(err)
		}
	})
}

func TestPanicInOneCloseDoesNotSkipTheOthers(t *testing.T) {
	good := &probeSession{}
	n := 0
	f := func(ctx context.Context, c router.Checkpoint) (*agent.Agent, error) {
		n++
		if n == 1 {
			return agent.Open(toyCheckpoint(), &panicSession{})
		}
		return newProbe(t, good), nil
	}
	r, _ := router.New(router.WithAgentFactory(f), router.WithMaxLoaded(3))
	ctx := context.Background()
	r.Load(ctx, "english")
	r.Load(ctx, "multilingual")
	if _, err := r.UnloadAll(ctx); err == nil {
		t.Fatal("the panic was lost")
	}
	if good.closes.Load() != 1 {
		t.Fatal("a panicking close skipped the next agent")
	}
}

func TestAttachOfAnEvictedLeasedAgentIsNeverClosedByTheRouter(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(1))
	ls, err := r.Lease(context.Background(), "english")
	if err != nil {
		t.Fatal(err)
	}
	a := ls.Agent()
	mustLoad(t, r, "multilingual") // evicts english while leased
	if err := r.Attach("english", a); err != nil {
		t.Fatal(err)
	}
	ls.Close()
	if n := f.of(a).sess.closes.Load(); n != 0 {
		t.Fatalf("an attached agent was closed by the Router (closes=%d)", n)
	}
}

func TestLoadKeepsItsLeaseAcrossTheLoadHook(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(1))
	var sawClosed atomic.Bool
	var once sync.Once
	r.AddHook(&router.HookFuncs{OnLoadFunc: func(c *router.LoadContext) {
		if c.Model != router.English {
			return
		}
		once.Do(func() {
			go r.Load(context.Background(), "multilingual") // evicts english under the hook
			b := f.of(c.Agent)
			for i := 0; i < 100 && b.sess.closes.Load() == 0; i++ {
				time.Sleep(5 * time.Millisecond)
			}
			sawClosed.Store(b.sess.closes.Load() > 0)
		})
	}})
	mustLoad(t, r, "english")
	if sawClosed.Load() {
		t.Fatal("OnLoad held an agent that a concurrent eviction closed")
	}
}

func TestFactoryNotCalledForABuildQueuedBehindClose(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 4)
	r := newRouter(t, f)
	gate := f.open(en)
	go r.Load(context.Background(), "english")
	<-f.entered
	qerr := make(chan error, 1)
	go func() { _, err := r.Load(context.Background(), "multilingual"); qerr <- err }() // queued on the token
	time.Sleep(30 * time.Millisecond)
	cdone := make(chan struct{})
	go func() { r.Close(); close(cdone) }()
	time.Sleep(30 * time.Millisecond)
	close(gate)
	<-cdone
	if err := <-qerr; !errors.Is(err, router.ErrClosed) {
		t.Fatalf("queued build: %v", err)
	}
	if f.calls() != 1 {
		t.Fatalf("factory called %d times; a build started after Close", f.calls())
	}
}

func TestBuildQueuedBehindTheTokenSeesAnAttach(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 4)
	gate := f.open(en)
	r := newRouter(t, f)
	go r.Load(context.Background(), "english")
	<-f.entered
	res := make(chan *agent.Agent, 1)
	go func() { a, _ := r.Load(context.Background(), "multilingual"); res <- a }()
	time.Sleep(30 * time.Millisecond)
	attached := newProbe(t, nil)
	if err := r.Attach("multilingual", attached); err != nil {
		t.Fatal(err)
	}
	close(gate)
	if got := <-res; got != attached {
		t.Fatal("the queued build did not use the attached agent")
	}
	if f.calls() != 1 {
		t.Fatalf("factory called %d times for a checkpoint that was attached", f.calls())
	}
}

func TestStressLeaseAccountingWithCancellation(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(1))
	names := []string{"english", "multilingual", "typed-decisions"}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rnd := rand.New(rand.NewSource(seed))
			for i := 0; i < 400; i++ {
				n := names[rnd.Intn(3)]
				ctx, cancel := context.WithTimeout(context.Background(), time.Duration(1+rnd.Intn(20))*time.Millisecond)
				switch rnd.Intn(5) {
				case 0, 1:
					if l, err := r.Lease(ctx, n); err == nil {
						if l.Agent() == nil || f.of(l.Agent()).sess.closes.Load() != 0 {
							t.Error("lease handed out a closed agent")
						}
						l.Close()
						l.Close()
					}
				case 2:
					r.Load(ctx, n)
				case 3:
					r.Unload(ctx, n)
				default:
					r.UnloadAll(ctx)
				}
				cancel()
			}
		}(int64(g))
	}
	wg.Wait()
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, b := range f.made {
		if c := b.sess.closes.Load(); c != 1 {
			t.Errorf("%s closed %d times", b.label, c)
		}
	}
}

func TestPreloadKeepsItsLeaseAcrossTheLoadHook(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	var sawClosed atomic.Bool
	var once sync.Once
	r.AddHook(&router.HookFuncs{OnLoadFunc: func(c *router.LoadContext) {
		once.Do(func() {
			go r.Unload(context.Background(), "english") // frees the checkpoint under the hook
			b := f.of(c.Agent)
			for i := 0; i < 100 && b.sess.closes.Load() == 0; i++ {
				time.Sleep(5 * time.Millisecond)
			}
			sawClosed.Store(b.sess.closes.Load() > 0)
		})
	}})
	if err := r.Preload(context.Background(), "english"); err != nil {
		t.Fatal(err)
	}
	if sawClosed.Load() {
		t.Fatal("OnLoad held an agent that a concurrent Unload closed")
	}
}
