package router_test

import (
	"context"
	"errors"
	"math/rand"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/router"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

const (
	en = router.English
	ml = router.Multilingual
	td = router.TypedDecisions
)

func loadedIs(t *testing.T, r *router.Router, want ...router.Checkpoint) {
	t.Helper()
	got := r.Loaded()
	if len(got) == 0 && len(want) == 0 {
		return
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("loaded = %v, want %v", got, want)
	}
}

func mustLoad(t *testing.T, r *router.Router, name string) *agent.Agent {
	t.Helper()
	a, err := r.Load(context.Background(), name)
	if err != nil {
		t.Fatalf("Load(%q): %v", name, err)
	}
	return a
}

func TestLoadIsCached(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	a, b := mustLoad(t, r, "english"), mustLoad(t, r, "en")
	if a != b || f.calls() != 1 {
		t.Fatalf("same agent %v, factory calls %d; want one build", a == b, f.calls())
	}
}

func TestLoadedOrderIsLRU(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(3))
	for _, n := range []string{"english", "multilingual", "typed-decisions", "english"} {
		mustLoad(t, r, n)
	}
	loadedIs(t, r, ml, td, en)
}

func TestEvictionClosesTheVictim(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(1))
	a := mustLoad(t, r, "english")
	b := mustLoad(t, r, "multilingual")
	loadedIs(t, r, ml)
	if got := f.of(a).sess.closes.Load(); got != 1 {
		t.Fatalf("evicted agent closed %d times, want 1", got)
	}
	if got := f.of(b).sess.closes.Load(); got != 0 {
		t.Fatalf("resident agent closed %d times", got)
	}
}

func TestEvictionOrderIsLeastRecentlyUsed(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	a := mustLoad(t, r, "english")
	mustLoad(t, r, "multilingual")
	mustLoad(t, r, "english") // touch: multilingual is now the oldest
	mustLoad(t, r, "typed-decisions")
	loadedIs(t, r, en, td)
	if f.of(a).sess.closes.Load() != 0 {
		t.Fatal("the recently used agent was closed")
	}
}

func TestAttachedAgentsAreNotClosed(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(1))
	s := &probeSession{}
	if err := r.Attach("english", newProbe(t, s)); err != nil {
		t.Fatal(err)
	}
	mustLoad(t, r, "multilingual") // evicts nothing: Attach raised the limit to 1, and this makes 2
	if _, err := r.Unload(context.Background(), "english"); err != nil {
		t.Fatal(err)
	}
	if _, err := r.UnloadAll(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s.closes.Load() != 0 {
		t.Fatal("the Router closed an agent it did not build")
	}
}

func TestAttachRaisesMaxLoaded(t *testing.T) {
	r := newRouter(t, newFactory(t))
	for i, n := range []string{"english", "multilingual", "typed-decisions"} {
		if err := r.Attach(n, newProbe(t, nil)); err != nil {
			t.Fatal(err)
		}
		if got := r.MaxLoaded(); got != max(2, i+1) {
			t.Fatalf("after %d attaches MaxLoaded = %d", i+1, got)
		}
	}
	loadedIs(t, r, en, ml, td)
}

func TestAttachErrors(t *testing.T) {
	r := newRouter(t, newFactory(t))
	var unknown *router.UnknownModelError
	if err := r.Attach("gpt", newProbe(t, nil)); !errors.As(err, &unknown) {
		t.Fatalf("unknown name: %v", err)
	}
	if err := r.Attach("english", nil); err == nil {
		t.Fatal("nil agent accepted")
	}
}

func TestPreloadFitsEverythingAndStages(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(1))
	if err := r.Preload(context.Background()); err != nil {
		t.Fatal(err)
	}
	loadedIs(t, r, en, ml, td)
	if r.MaxLoaded() != 3 || f.calls() != 3 {
		t.Fatalf("max %d calls %d", r.MaxLoaded(), f.calls())
	}
	r2 := newRouter(t, newFactory(t))
	for _, n := range []string{"english", "multilingual", "typed-decisions"} {
		if err := r2.Preload(context.Background(), n); err != nil {
			t.Fatal(err)
		}
	}
	loadedIs(t, r2, en, ml, td)
}

func TestPreloadResidentIsNotTouched(t *testing.T) {
	r := newRouter(t, newFactory(t))
	mustLoad(t, r, "english")
	mustLoad(t, r, "multilingual")
	if err := r.Preload(context.Background(), "english"); err != nil {
		t.Fatal(err)
	}
	loadedIs(t, r, en, ml)
}

func TestPreloadUnknownNameChangesNothing(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	if err := r.Preload(context.Background(), "english", "gpt"); err == nil || f.calls() != 0 || r.MaxLoaded() != 2 {
		t.Fatalf("err %v calls %d max %d", err, f.calls(), r.MaxLoaded())
	}
}

func TestUnloadFrees(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	a := mustLoad(t, r, "english")
	mustLoad(t, r, "multilingual")
	freed, err := r.Unload(context.Background(), "en")
	if err != nil || !reflect.DeepEqual(freed, []router.Checkpoint{en}) {
		t.Fatalf("freed %v err %v", freed, err)
	}
	if f.of(a).sess.closes.Load() != 1 {
		t.Fatal("unloaded agent not closed")
	}
	if freed, _ := r.Unload(context.Background(), "en"); len(freed) != 0 {
		t.Fatalf("second unload freed %v", freed)
	}
	all, err := r.UnloadAll(context.Background())
	if err != nil || !reflect.DeepEqual(all, []router.Checkpoint{ml}) {
		t.Fatalf("UnloadAll = %v, %v", all, err)
	}
	loadedIs(t, r)
}

func TestLeaseOutlivesEviction(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(1))
	l1, err := r.Lease(context.Background(), "english")
	if err != nil {
		t.Fatal(err)
	}
	l2, err := r.Lease(context.Background(), "english")
	if err != nil {
		t.Fatal(err)
	}
	mustLoad(t, r, "multilingual") // evicts english while it is leased twice
	loadedIs(t, r, ml)
	sess := f.of(l1.Agent()).sess
	if sess.closes.Load() != 0 {
		t.Fatal("evicted agent closed under a lease")
	}
	if err := l1.Close(); err != nil || sess.closes.Load() != 0 {
		t.Fatalf("closed with a lease outstanding: %v closes=%d", err, sess.closes.Load())
	}
	if err := l2.Close(); err != nil {
		t.Fatal(err)
	}
	if sess.closes.Load() != 1 {
		t.Fatalf("closes after the last release = %d", sess.closes.Load())
	}
	if l1.Agent() != nil || l1.Close() != nil {
		t.Fatal("a closed lease must give no agent and close quietly")
	}
}

func TestLeaseOnResidentIsReleasedWithoutClosing(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	l, err := r.Lease(context.Background(), "ml")
	if err != nil {
		t.Fatal(err)
	}
	if l.Checkpoint() != ml {
		t.Fatalf("checkpoint %v", l.Checkpoint())
	}
	sess := f.of(l.Agent()).sess
	l.Close()
	if sess.closes.Load() != 0 {
		t.Fatal("a resident agent closed on release")
	}
}

func TestUnloadWithLeaseClosesAtRelease(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	l, _ := r.Lease(context.Background(), "english")
	sess := f.of(l.Agent()).sess
	if _, err := r.Unload(context.Background(), "english"); err != nil {
		t.Fatal(err)
	}
	if sess.closes.Load() != 0 {
		t.Fatal("closed under the lease")
	}
	l.Close()
	if sess.closes.Load() != 1 {
		t.Fatal("not closed at release")
	}
}

func TestClosingALeaseTwiceUnderARaceCountsOnce(t *testing.T) {
	for i := 0; i < 50; i++ {
		f := newFactory(t)
		r := newRouter(t, f)
		l1, _ := r.Lease(context.Background(), "english")
		l2, _ := r.Lease(context.Background(), "english")
		sess := f.of(l1.Agent()).sess
		if _, err := r.Unload(context.Background(), "english"); err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		for j := 0; j < 4; j++ {
			wg.Add(1)
			go func() { defer wg.Done(); l1.Close() }()
		}
		wg.Wait()
		if sess.closes.Load() != 0 {
			t.Fatal("one lease released several times closed the agent under the other")
		}
		l2.Close()
		if sess.closes.Load() != 1 {
			t.Fatal("not closed after the last release")
		}
	}
}

func TestConcurrentLoadsBuildOnce(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 64)
	gate := f.open(en)
	r := newRouter(t, f)
	const n = 24
	got := make([]*agent.Agent, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			a, err := r.Load(context.Background(), "english")
			if err != nil {
				t.Error(err)
			}
			got[i] = a
		}(i)
	}
	<-f.entered // the one build has started; the others are waiting on it
	time.Sleep(50 * time.Millisecond)
	close(gate)
	wg.Wait()
	for _, a := range got {
		if a != got[0] || a == nil {
			t.Fatal("callers did not share one agent")
		}
	}
	if f.calls() != 1 {
		t.Fatalf("factory called %d times", f.calls())
	}
}

func TestResidentLoadDoesNotWaitForAColdBuild(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 4)
	r := newRouter(t, f)
	mustLoad(t, r, "english")
	<-f.entered
	gate := f.open(ml)
	done := make(chan error, 1)
	go func() { _, err := r.Load(context.Background(), "multilingual"); done <- err }()
	<-f.entered // the cold build is inside the factory
	within(t, 2*time.Second, "a resident Load behind a cold build", func() { mustLoad(t, r, "english") })
	within(t, 2*time.Second, "Loaded behind a cold build", func() { r.Loaded() })
	close(gate)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestColdBuildsAreSerialised(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 4)
	gate := f.open(en)
	r := newRouter(t, f)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); r.Load(context.Background(), "english") }()
	<-f.entered
	go func() { defer wg.Done(); r.Load(context.Background(), "multilingual") }()
	select {
	case c := <-f.entered:
		close(gate)
		t.Fatalf("%s entered the factory while another build was in it", c)
	case <-time.After(100 * time.Millisecond):
	}
	close(gate)
	<-f.entered
	wg.Wait()
}

var errBoom = errors.New("boom building")

func TestFailedBuildIsNotCachedAndWakesWaiters(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 64)
	gate := f.open(en)
	f.setFail(en, errBoom)
	r := newRouter(t, f)
	const n = 8
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) { defer wg.Done(); _, errs[i] = r.Load(context.Background(), "english") }(i)
	}
	<-f.entered
	time.Sleep(50 * time.Millisecond)
	close(gate)
	within(t, 5*time.Second, "waiters on a failed build", wg.Wait)
	for i, err := range errs {
		if !errors.Is(err, errBoom) {
			t.Fatalf("caller %d: %v", i, err)
		}
	}
	loadedIs(t, r)
	if f.calls() != 1 {
		t.Fatalf("waiters rebuilt: %d factory calls", f.calls())
	}
	f.setFail(en, nil)
	f.mu.Lock()
	delete(f.gate, en)
	f.mu.Unlock()
	mustLoad(t, r, "english") // not wedged, not cached
	if f.calls() != 2 {
		t.Fatalf("factory calls %d", f.calls())
	}
	loadedIs(t, r, en)
}

func TestFailedBuildEvictsNothing(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(1))
	mustLoad(t, r, "english")
	f.setFail(ml, errBoom)
	if _, err := r.Load(context.Background(), "multilingual"); !errors.Is(err, errBoom) {
		t.Fatal(err)
	}
	loadedIs(t, r, en)
}

func TestPanickingFactoryWakesWaiters(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 8)
	gate := f.open(en)
	f.panicOn[en] = true
	r := newRouter(t, f)
	waiter := make(chan error, 1)
	builder := make(chan any, 1)
	go func() {
		defer func() { builder <- recover() }()
		r.Load(context.Background(), "english")
	}()
	<-f.entered
	go func() { _, err := r.Load(context.Background(), "english"); waiter <- err }()
	time.Sleep(50 * time.Millisecond)
	close(gate)
	select {
	case err := <-waiter:
		if err == nil {
			t.Fatal("waiter got no error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("waiter hung on a panicking build")
	}
	if <-builder == nil {
		t.Fatal("the panic was swallowed")
	}
	f.panicOn[en] = false
	mustLoad(t, r, "english")
}

func TestWaiterDoesNotInheritTheBuildersCancellation(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 8)
	f.open(en) // never opened: the build ends only when its context does
	r := newRouter(t, f)
	bctx, cancel := context.WithCancel(context.Background())
	berr := make(chan error, 1)
	go func() { _, err := r.Load(bctx, "english"); berr <- err }()
	<-f.entered
	werr := make(chan error, 1)
	go func() { _, err := r.Load(context.Background(), "english"); werr <- err }()
	time.Sleep(50 * time.Millisecond)
	f.mu.Lock()
	delete(f.gate, en)
	f.mu.Unlock()
	cancel()
	if err := <-berr; !errors.Is(err, context.Canceled) {
		t.Fatalf("builder: %v", err)
	}
	select {
	case err := <-werr:
		if err != nil {
			t.Fatalf("waiter inherited the cancellation: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("waiter hung")
	}
}

func TestWaiterHonoursItsOwnContext(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 8)
	gate := f.open(en)
	r := newRouter(t, f)
	go r.Load(context.Background(), "english")
	<-f.entered
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := r.Load(ctx, "english"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	close(gate)
}

func TestNoFactory(t *testing.T) {
	r, err := router.New()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Load(context.Background(), "english"); err == nil {
		t.Fatal("a Router with no factory loaded")
	}
	if _, err := r.Route("hello", schema.Questions{}); err != nil {
		t.Fatalf("routing must not need a factory: %v", err)
	}
	loadedIs(t, r)
}

func TestFactoryReturningNothing(t *testing.T) {
	r, _ := router.New(router.WithAgentFactory(func(context.Context, router.Checkpoint) (*agent.Agent, error) { return nil, nil }))
	if _, err := r.Load(context.Background(), "english"); err == nil {
		t.Fatal("a nil agent was accepted")
	}
	loadedIs(t, r)
}

func TestMaxLoadedMustBePositive(t *testing.T) {
	for _, n := range []int{0, -1} {
		if _, err := router.New(router.WithMaxLoaded(n)); err == nil {
			t.Fatalf("WithMaxLoaded(%d) accepted", n)
		}
	}
	r, _ := router.New()
	if r.MaxLoaded() != 2 {
		t.Fatalf("default %d", r.MaxLoaded())
	}
}

func TestAttachReplacesAndRetiresTheBuiltAgent(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	l, _ := r.Lease(context.Background(), "english")
	sess := f.of(l.Agent()).sess
	attached := newProbe(t, nil)
	if err := r.Attach("english", attached); err != nil {
		t.Fatal(err)
	}
	if got := mustLoad(t, r, "english"); got != attached {
		t.Fatal("Load did not return the attached agent")
	}
	if sess.closes.Load() != 0 {
		t.Fatal("replaced agent closed under a lease")
	}
	l.Close()
	if sess.closes.Load() != 1 {
		t.Fatal("replaced agent never closed")
	}
	// and one nobody leased closes at once
	r2 := newRouter(t, f)
	a := mustLoad(t, r2, "english")
	if err := r2.Attach("english", newProbe(t, nil)); err != nil {
		t.Fatal(err)
	}
	if f.of(a).sess.closes.Load() != 1 {
		t.Fatal("unleased replaced agent not closed")
	}
}

func TestAttachingTheSameInstanceIsSafe(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	a := mustLoad(t, r, "english")
	if err := r.Attach("english", a); err != nil {
		t.Fatal(err)
	}
	if f.of(a).sess.closes.Load() != 0 {
		t.Fatal("re-attaching the loaded agent closed it")
	}
	if got := mustLoad(t, r, "english"); got != a {
		t.Fatal("agent changed")
	}
	// ownership moved to the caller: the Router no longer closes it
	r.Unload(context.Background(), "english")
	if f.of(a).sess.closes.Load() != 0 {
		t.Fatal("the Router closed an agent that was handed back to the caller")
	}
}

func TestAttachWhileBuildingKeepsTheAttachedAgent(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 4)
	gate := f.open(en)
	var loadedAgent *agent.Agent
	var hmu sync.Mutex
	hook := &router.HookFuncs{OnLoadFunc: func(c *router.LoadContext) { hmu.Lock(); loadedAgent = c.Agent; hmu.Unlock() }}
	r := newRouter(t, f, router.WithInstalledHooks(hook))
	attached := newProbe(t, nil)
	res := make(chan *agent.Agent, 1)
	go func() { a, _ := r.Load(context.Background(), "english"); res <- a }()
	<-f.entered
	if err := r.Attach("english", attached); err != nil {
		t.Fatal(err)
	}
	close(gate)
	if got := <-res; got != attached {
		t.Fatal("the build overwrote the attached agent")
	}
	hmu.Lock()
	if loadedAgent != attached {
		t.Fatal("on_load saw the discarded build, not the agent the Router serves")
	}
	hmu.Unlock()
	f.mu.Lock()
	dup := f.made[0]
	f.mu.Unlock()
	if dup.sess.closes.Load() != 1 {
		t.Fatalf("the duplicate build closed %d times", dup.sess.closes.Load())
	}
}

func TestCloseFreesAndSeals(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f)
	a := mustLoad(t, r, "english")
	s := &probeSession{}
	r.Attach("multilingual", newProbe(t, s))
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if f.of(a).sess.closes.Load() != 1 || s.closes.Load() != 0 {
		t.Fatal("Close must close what it built and leave what was attached")
	}
	loadedIs(t, r)
	ctx := context.Background()
	if _, err := r.Load(ctx, "english"); !errors.Is(err, router.ErrClosed) {
		t.Fatalf("Load after Close: %v", err)
	}
	if _, err := r.Lease(ctx, "english"); !errors.Is(err, router.ErrClosed) {
		t.Fatalf("Lease after Close: %v", err)
	}
	if err := r.Preload(ctx); !errors.Is(err, router.ErrClosed) {
		t.Fatalf("Preload after Close: %v", err)
	}
	if err := r.Attach("english", newProbe(t, nil)); !errors.Is(err, router.ErrClosed) {
		t.Fatalf("Attach after Close: %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestCloseWaitsForAnInFlightBuild(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 4)
	gate := f.open(en)
	r := newRouter(t, f)
	type res struct {
		l   *router.Lease
		err error
	}
	got := make(chan res, 1)
	go func() { l, err := r.Lease(context.Background(), "english"); got <- res{l, err} }()
	<-f.entered
	closed := make(chan error, 1)
	go func() { closed <- r.Close() }()
	select {
	case <-closed:
		t.Fatal("Close returned while a build was in flight")
	case <-time.After(100 * time.Millisecond):
	}
	close(gate)
	if err := <-closed; err != nil {
		t.Fatal(err)
	}
	g := <-got
	if g.err != nil {
		t.Fatalf("the building caller must get its lease: %v", g.err)
	}
	loadedIs(t, r)
	sess := f.of(g.l.Agent()).sess
	if sess.closes.Load() != 0 {
		t.Fatal("agent closed under the lease Close raced")
	}
	g.l.Close()
	if sess.closes.Load() != 1 {
		t.Fatal("the agent built during Close leaked")
	}
}

func TestCloseWithAWaiterOnTheBuild(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 4)
	gate := f.open(en)
	r := newRouter(t, f)
	go func() {
		if l, err := r.Lease(context.Background(), "english"); err == nil {
			l.Close()
		}
	}()
	<-f.entered
	werr := make(chan error, 1)
	go func() { _, err := r.Load(context.Background(), "english"); werr <- err }()
	time.Sleep(50 * time.Millisecond)
	go r.Close()
	time.Sleep(50 * time.Millisecond)
	close(gate)
	select {
	case err := <-werr:
		if err != nil && !errors.Is(err, router.ErrClosed) {
			t.Fatalf("waiter: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("waiter hung across Close")
	}
}

func TestUnloadWaitsForItsOwnBuildOnly(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 4)
	gate := f.open(en)
	r := newRouter(t, f)
	go r.Load(context.Background(), "english")
	<-f.entered
	within(t, 2*time.Second, "Unload of another checkpoint behind a build", func() {
		r.Unload(context.Background(), "multilingual")
	})
	done := make(chan []router.Checkpoint, 1)
	go func() { freed, _ := r.Unload(context.Background(), "english"); done <- freed }()
	select {
	case <-done:
		t.Fatal("Unload returned while its checkpoint was building")
	case <-time.After(100 * time.Millisecond):
	}
	close(gate)
	if freed := <-done; len(freed) != 1 {
		t.Fatalf("freed %v", freed)
	}
	loadedIs(t, r)
}

func TestUnloadAllWaitsForBuildsAndHonoursContext(t *testing.T) {
	f := newFactory(t)
	f.entered = make(chan router.Checkpoint, 4)
	gate := f.open(en)
	r := newRouter(t, f)
	go r.Load(context.Background(), "english")
	<-f.entered
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := r.UnloadAll(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	if _, err := r.Unload(ctx, "english"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	close(gate)
}

func TestASlowCloseHoldsNoLock(t *testing.T) {
	f := newFactory(t)
	slow := &probeSession{hold: make(chan struct{}), inClose: make(chan struct{})}
	r := newRouter(t, f, router.WithMaxLoaded(1), router.WithAgentFactory(func(ctx context.Context, c router.Checkpoint) (*agent.Agent, error) {
		if c == en {
			return newProbe(t, slow), nil
		}
		return f.build(ctx, c)
	}))
	mustLoad(t, r, "english")
	go r.Load(context.Background(), "multilingual") // evicts english; its Close blocks on slow.hold
	<-slow.inClose
	within(t, 2*time.Second, "Loaded during a slow close", func() { r.Loaded() })
	within(t, 2*time.Second, "MaxLoaded during a slow close", func() { r.MaxLoaded() })
	within(t, 2*time.Second, "Unload during a slow close", func() { r.Unload(context.Background(), "typed-decisions") })
	close(slow.hold)
}

func TestCloseErrorFailsTheLoadButNotTheEvents(t *testing.T) {
	f := newFactory(t)
	bad := errors.New("close failed")
	var events []string
	var mu sync.Mutex
	h := &router.HookFuncs{
		OnLoadFunc: func(c *router.LoadContext) { mu.Lock(); events = append(events, "load:"+string(c.Model)); mu.Unlock() },
		OnEvictFunc: func(c *router.EvictContext) {
			mu.Lock()
			events = append(events, "evict:"+string(c.Model))
			mu.Unlock()
		},
	}
	r := newRouter(t, f, router.WithMaxLoaded(1), router.WithInstalledHooks(h),
		router.WithAgentFactory(func(ctx context.Context, c router.Checkpoint) (*agent.Agent, error) {
			if c == en {
				return newProbe(t, &probeSession{closeErr: bad}), nil
			}
			return f.build(ctx, c)
		}))
	mustLoad(t, r, "english")
	l, err := r.Lease(context.Background(), "multilingual")
	if !errors.Is(err, bad) || l != nil {
		t.Fatalf("lease %v err %v", l, err)
	}
	loadedIs(t, r, ml)
	mu.Lock()
	got := append([]string(nil), events...)
	mu.Unlock()
	if want := []string{"load:english", "evict:english", "load:multilingual"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("events %v, want %v", got, want)
	}
	// the lease the failed call took was given back: eviction closes the new agent at once
	b := f.made[0]
	r.Unload(context.Background(), "multilingual")
	if b.sess.closes.Load() != 1 {
		t.Fatal("the failed call stranded its lease")
	}
}

// ---- hooks

func TestHooksRunOutsideTheLock(t *testing.T) {
	f := newFactory(t)
	var seen [][]router.Checkpoint
	var mu sync.Mutex
	var r *router.Router
	h := &router.HookFuncs{
		OnLoadFunc: func(c *router.LoadContext) {
			mu.Lock()
			seen = append(seen, c.Router.Loaded())
			mu.Unlock()
			c.Router.MaxLoaded()
			if c.Model == en {
				c.Router.Load(context.Background(), "multilingual") // re-enters the Router from a hook
			}
		},
		OnEvictFunc: func(c *router.EvictContext) { c.Router.Loaded() },
	}
	r = newRouter(t, f, router.WithInstalledHooks(h), router.WithMaxLoaded(1))
	within(t, 5*time.Second, "loads with hooks that call back in", func() {
		mustLoad(t, r, "english")
		mustLoad(t, r, "typed-decisions")
		r.UnloadAll(context.Background())
	})
	mu.Lock()
	defer mu.Unlock()
	if len(seen) == 0 || len(seen[0]) == 0 {
		t.Fatalf("on_load saw %v: the checkpoint was not resident yet", seen)
	}
}

func TestLifecycleHookContext(t *testing.T) {
	f := newFactory(t)
	var load *router.LoadContext
	var evict *router.EvictContext
	h := &router.HookFuncs{
		OnLoadFunc:  func(c *router.LoadContext) { load = c },
		OnEvictFunc: func(c *router.EvictContext) { evict = c },
	}
	r := newRouter(t, f, router.WithInstalledHooks(h), router.WithMaxLoaded(1))
	a := mustLoad(t, r, "english")
	if load == nil || load.Model != en || load.Agent != a || load.Router != r || len(load.RunID) != 32 {
		t.Fatalf("load context %+v", load)
	}
	mustLoad(t, r, "multilingual")
	if evict == nil || evict.Model != en || evict.Router != r || len(evict.RunID) != 32 || evict.RunID == load.RunID {
		t.Fatalf("evict context %+v", evict)
	}
}

func TestResidentLoadFiresNoHooks(t *testing.T) {
	f := newFactory(t)
	n := 0
	h := &router.HookFuncs{OnLoadFunc: func(*router.LoadContext) { n++ }}
	r := newRouter(t, f, router.WithInstalledHooks(h))
	mustLoad(t, r, "english")
	mustLoad(t, r, "english")
	r.Lease(context.Background(), "english")
	if n != 1 {
		t.Fatalf("on_load fired %d times", n)
	}
}

func TestLeaseHeldAcrossTheLoadHook(t *testing.T) {
	f := newFactory(t)
	h := &router.HookFuncs{OnLoadFunc: func(c *router.LoadContext) {
		c.Router.Unload(context.Background(), string(c.Model)) // evicts the agent under the hook
	}}
	r := newRouter(t, f, router.WithInstalledHooks(h))
	l, err := r.Lease(context.Background(), "english")
	if err != nil {
		t.Fatal(err)
	}
	sess := f.of(l.Agent()).sess
	if sess.closes.Load() != 0 {
		t.Fatal("the agent was closed under the lease while on_load ran")
	}
	l.Close()
	if sess.closes.Load() != 1 {
		t.Fatal("not closed at release")
	}
}

func TestFailingHookKeepsTheCheckpointAndGivesBackTheLease(t *testing.T) {
	f := newFactory(t)
	fail := &router.HookFuncs{OnLoadFunc: func(*router.LoadContext) { panic(errors.New("hook says no")) }}
	r := newRouter(t, f, router.WithMaxLoaded(1), router.WithInstalledHooks(fail))
	l, err := r.Lease(context.Background(), "english")
	if err == nil || l != nil {
		t.Fatalf("lease %v err %v", l, err)
	}
	loadedIs(t, r, en)
	r.RemoveHook(fail)
	mustLoad(t, r, "multilingual") // evicts english
	if got := f.made[0].sess.closes.Load(); got != 1 {
		t.Fatalf("the failed Lease stranded its lease: closes=%d", got)
	}
	if a, err := r.Load(context.Background(), "multilingual"); err != nil || a == nil {
		t.Fatal(err)
	}
}

func TestSerialHooksDoNotOverlapAcrossLoads(t *testing.T) {
	f := newFactory(t)
	var in, peak int32
	var mu sync.Mutex
	h := &router.HookFuncs{OnLoadFunc: func(*router.LoadContext) {
		mu.Lock()
		in++
		if in > peak {
			peak = in
		}
		mu.Unlock()
		time.Sleep(20 * time.Millisecond)
		mu.Lock()
		in--
		mu.Unlock()
	}}
	r := newRouter(t, f, router.WithSerialHooks(), router.WithInstalledHooks(h), router.WithMaxLoaded(3))
	var wg sync.WaitGroup
	for _, n := range []string{"english", "multilingual", "typed-decisions"} {
		wg.Add(1)
		go func() { defer wg.Done(); r.Load(context.Background(), n) }()
	}
	wg.Wait()
	if peak != 1 {
		t.Fatalf("serial hooks overlapped: peak %d", peak)
	}
}

func TestHookValueWithoutAnyEventIsRefused(t *testing.T) {
	type nothing struct{}
	if _, err := router.New(router.WithInstalledHooks(&nothing{})); err == nil {
		t.Fatal("a hook with no event accepted")
	}
	if _, err := router.New(router.WithInstalledHooks(&router.HookFuncs{})); err != nil {
		t.Fatal(err)
	}
}

// ---- stress

func TestConcurrentStress(t *testing.T) {
	f := newFactory(t)
	r := newRouter(t, f, router.WithMaxLoaded(2))
	attached := map[*agent.Agent]*probeSession{}
	var amu sync.Mutex
	var wg sync.WaitGroup
	check := func(a *agent.Agent) {
		if b := f.of(a); b != nil && b.sess.closes.Load() != 0 {
			t.Error("an agent under lease was closed")
		}
	}
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))
			names := []string{"english", "multilingual", "typed-decisions", "en", "ml"}
			ctx := context.Background()
			for i := 0; i < 150; i++ {
				n := names[rng.Intn(len(names))]
				switch rng.Intn(8) {
				case 0, 1, 2, 3:
					l, err := r.Lease(ctx, n)
					if err != nil {
						if !errors.Is(err, router.ErrClosed) {
							t.Error(err)
						}
						continue
					}
					check(l.Agent())
					time.Sleep(time.Duration(rng.Intn(200)) * time.Microsecond)
					check(l.Agent())
					l.Close()
				case 4:
					r.Load(ctx, n)
				case 5:
					r.Unload(ctx, n)
				case 6:
					s := &probeSession{}
					a := newProbe(t, s)
					amu.Lock()
					attached[a] = s
					amu.Unlock()
					r.Attach(n, a)
				default:
					r.Preload(ctx, n)
					r.Loaded()
				}
			}
		}(int64(g))
	}
	wg.Wait()
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	for _, b := range f.made {
		if got := b.sess.closes.Load(); got != 1 {
			t.Errorf("%s closed %d times, want exactly 1", b.label, got)
		}
	}
	amu.Lock()
	for a, s := range attached {
		if s.closes.Load() != 0 {
			t.Errorf("attached agent %p was closed by the Router", a)
		}
	}
	amu.Unlock()
}
