package router_test

import (
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/router"
)

type fn = func(*router.RouteContext)

func hook(f fn) *router.HookFuncs { return &router.HookFuncs{OnRouteFunc: f} }

func TestHookSeesTheContext(t *testing.T) {
	var seen *router.RouteContext
	var copyOf router.RouteContext
	r := mustNew(t)
	_ = r.AddHook(hook(func(c *router.RouteContext) { seen, copyOf = c, *c }))
	qs := noul(t, "urgency")
	d, err := r.Route(englishText, qs)
	if err != nil {
		t.Fatal(err)
	}
	if seen == nil || copyOf.Router != r || len(copyOf.States) != 1 || copyOf.States[0] != englishText ||
		strings.Join(copyOf.Questions.IDs(), ",") != "urgency" || len(copyOf.RunID) != 32 {
		t.Fatalf("context %+v", copyOf)
	}
	if copyOf.Decision.Model != router.English || d.Model != router.English || copyOf.Decision.Reason != d.Reason {
		t.Fatalf("hook saw %+v, Route returned %+v", copyOf.Decision, d)
	}
}

func TestHookReplacementIsReturnedAsIs(t *testing.T) {
	r := mustNew(t)
	_ = r.AddHook(hook(func(c *router.RouteContext) { c.Decision = router.Decision{Model: "nonsense", Reason: "r"} }))
	d, err := r.Route(englishText, none)
	if err != nil || d.Model != "nonsense" || d.Reason != "r" {
		t.Fatalf("%+v, %v", d, err)
	}
}

func TestRunIDDiffersPerCallAndIsSharedWithinOne(t *testing.T) {
	var ids []string
	r := mustNew(t)
	h := hook(func(c *router.RouteContext) { ids = append(ids, c.RunID) })
	_ = r.AddHook(h, h)
	for i := 0; i < 2; i++ {
		if _, err := r.Route("x", none); err != nil {
			t.Fatal(err)
		}
	}
	if len(ids) != 4 || ids[0] != ids[1] || ids[2] != ids[3] || ids[0] == ids[2] {
		t.Fatalf("run ids %v", ids)
	}
}

func TestRegistry(t *testing.T) {
	var order []string
	mk := func(n string) *router.HookFuncs { return hook(func(*router.RouteContext) { order = append(order, n) }) }
	a, b, c := mk("a"), mk("b"), mk("c")
	r := mustNew(t, router.WithInstalledHooks(a))
	run := func() string {
		order = nil
		if _, err := r.Route("x", none); err != nil {
			t.Fatal(err)
		}
		return strings.Join(order, "")
	}
	_ = r.AddHook(b, a)
	if got := run(); got != "aba" {
		t.Fatalf("add order %q", got)
	}
	undo, err := r.InstallHooks(c, a)
	if err != nil {
		t.Fatal(err)
	}
	if got := run(); got != "abaca" {
		t.Fatalf("installed %q", got)
	}
	_ = r.AddHook(c) // added while installed: must survive the undo
	undo()
	undo() // idempotent
	if got := run(); got != "abac" {
		t.Fatalf("after undo %q (the most recent copy of a goes, the earlier ones stay)", got)
	}
	if !r.RemoveHook(a) || r.RemoveHook(a) {
		t.Fatal("RemoveHook should find a once")
	}
	if got := run(); got != "bc" {
		t.Fatalf("after remove %q", got)
	}
	if n := len(r.Hooks()); n != 2 {
		t.Fatalf("%d hooks", n)
	}
	r.Hooks()[0] = nil // a copy
	if r.Hooks()[0] == nil {
		t.Fatal("Hooks returned the live slice")
	}
}

func TestHookValidation(t *testing.T) {
	r := mustNew(t)
	var nilHook *router.HookFuncs
	for name, bad := range map[string]any{"nil": nil, "typed nil": nilHook, "no event": struct{}{}, "a func": func() {}} {
		if err := r.AddHook(bad); err == nil {
			t.Errorf("%s accepted", name)
		}
		if _, err := router.New(router.WithInstalledHooks(bad)); err == nil {
			t.Errorf("%s accepted by New", name)
		}
		if _, err := r.Route("x", none, router.ForHooks(bad)); err == nil {
			t.Errorf("%s accepted by ForHooks", name)
		}
	}
	if _, err := r.InstallHooks(nil); err == nil {
		t.Error("InstallHooks accepted nil")
	}
	if len(r.Hooks()) != 0 {
		t.Fatal("a refused hook was installed")
	}
	// An agent.Hook alone is a valid entry: it has no OnRoute and is skipped.
	if err := r.AddHook(agent.BaseHook{}); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Route("x", none); err != nil {
		t.Fatal(err)
	}
}

func TestRouteErrorBeatsHookErrorsAndRunsNoHook(t *testing.T) {
	var ran bool
	r := mustNew(t)
	_ = r.AddHook(hook(func(*router.RouteContext) { ran = true }))
	_, err := r.Route("x", none, router.ForModel("nope"), router.ForHookTimeout(0), router.ForHooks(nil))
	var ue *router.UnknownModelError
	if !errors.As(err, &ue) || ran {
		t.Fatalf("%v, ran=%v", err, ran)
	}
}

func TestPerCallTimeoutIsValidatedEvenWithNoHooks(t *testing.T) {
	r := mustNew(t)
	for _, d := range []time.Duration{0, -time.Second} {
		_, err := r.Route("x", none, router.ForHookTimeout(d))
		if err == nil || !strings.HasPrefix(err.Error(), "hooks_timeout must be a positive finite number or None; got ") {
			t.Errorf("%v: %v", d, err)
		}
	}
	if _, err := router.New(router.WithHookTimeout(0)); err == nil {
		t.Error("New accepted a zero timeout")
	}
}

func TestPanicIsAFailedHookAndGoexitToo(t *testing.T) {
	r := mustNew(t)
	_ = r.AddHook(hook(func(*router.RouteContext) { panic("plain string") }))
	if _, err := r.Route("x", none); err == nil || err.Error() != "plain string" {
		t.Fatalf("%v", err)
	}
	var warns []string
	r = mustNew(t, router.WithHookErrors(false), router.WithHookWarnings(func(m string) { warns = append(warns, m) }))
	var after bool
	_ = r.AddHook(hook(func(*router.RouteContext) { panic(errors.New("x")) }), hook(func(*router.RouteContext) { after = true }))
	if _, err := r.Route("x", none); err != nil || !after || len(warns) != 1 ||
		warns[0] != "laya: hook HookFuncs.on_route failed: x" {
		t.Fatalf("err=%v after=%v warns=%q", err, after, warns)
	}
}

func TestTimeoutAbandonsTheHookButNotTheRouter(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	r := mustNew(t, router.WithHookTimeout(30*time.Millisecond))
	_ = r.AddHook(hook(func(*router.RouteContext) { <-release }))
	start := time.Now()
	_, err := r.Route("x", none)
	if err == nil || err.Error() != "laya: hook HookFuncs exceeded 0.03s" || time.Since(start) > 2*time.Second {
		t.Fatalf("%v after %v", err, time.Since(start))
	}
	// The slow hook no longer holds the serial lock: a call without it goes through.
	r.RemoveHook(r.Hooks()[0])
	if _, err := r.Route("x", none); err != nil {
		t.Fatal(err)
	}
}

// Serial hooks never overlap across goroutines; by default they can.
func TestSerialHooksDoNotOverlapAndConcurrentOnesCan(t *testing.T) {
	overlap := func(opts ...router.Option) int32 {
		var inside, worst int32
		r := mustNew(t, opts...)
		_ = r.AddHook(hook(func(*router.RouteContext) {
			n := atomic.AddInt32(&inside, 1)
			for {
				w := atomic.LoadInt32(&worst)
				if n <= w || atomic.CompareAndSwapInt32(&worst, w, n) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			atomic.AddInt32(&inside, -1)
		}))
		var wg sync.WaitGroup
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if _, err := r.Route(englishText, none); err != nil {
					t.Error(err)
				}
			}()
		}
		wg.Wait()
		return worst
	}
	if w := overlap(router.WithSerialHooks()); w != 1 {
		t.Errorf("serial hooks overlapped %d deep", w)
	}
	if w := overlap(); w < 2 {
		t.Errorf("concurrent hooks never overlapped (%d)", w)
	}
}

// Hooks added, removed and run from many goroutines: the race detector is the assertion, plus
// every call returning one of the two decisions a hook can leave.
func TestConcurrentRegistryAndRoutes(t *testing.T) {
	r := mustNew(t)
	pin := hook(func(c *router.RouteContext) {
		c.Decision = router.Decision{Model: router.Multilingual, Reason: "pin"}
	})
	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				undo, err := r.InstallHooks(pin)
				if err != nil {
					t.Error(err)
					return
				}
				r.Hooks()
				undo()
				_ = r.AddHook(pin)
				r.RemoveHook(pin)
			}
		}()
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				d, err := r.Route(englishText, none)
				if err != nil || (d.Model != router.English && d.Reason != "pin") {
					t.Errorf("%+v, %v", d, err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestDefaultHooksRunFirstAndAreSkippedWithoutOnRoute(t *testing.T) {
	agent.ClearDefaultHooks()
	t.Cleanup(agent.ClearDefaultHooks)
	var order []string
	d := &routeAndAgentHook{fn: func(*router.RouteContext) { order = append(order, "default") }}
	if err := agent.AddDefaultHook(d, agent.BaseHook{}); err != nil {
		t.Fatal(err)
	}
	r := mustNew(t, router.WithInstalledHooks(hook(func(*router.RouteContext) { order = append(order, "installed") })))
	if _, err := r.Route("x", none, router.ForHooks(hook(func(*router.RouteContext) { order = append(order, "call") }))); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(order, ","); got != "default,installed,call" {
		t.Fatalf("%q", got)
	}
}

type routeAndAgentHook struct {
	agent.BaseHook
	fn fn
}

func (h *routeAndAgentHook) OnRoute(c *router.RouteContext) { h.fn(c) }

func TestRouteBatchAppliesBatchHookOptionsToEveryRequestAndStopsAtFirstFailure(t *testing.T) {
	var seen []string
	r := mustNew(t)
	_ = r.AddHook(hook(func(c *router.RouteContext) {
		seen = append(seen, c.States[0].(string))
		if c.States[0] == "bad" {
			panic("no")
		}
	}))
	reqs := []router.Request{{State: "a"}, {State: "bad"}, {State: "c"}}
	if _, err := r.RouteBatch(reqs); err == nil || err.Error() != "no" || strings.Join(seen, ",") != "a,bad" {
		t.Fatalf("%v %v", err, seen)
	}
	seen = nil
	ds, err := r.RouteBatch(reqs, router.ForHookErrors(false))
	if err != nil || len(ds) != 3 || strings.Join(seen, ",") != "a,bad,c" {
		t.Fatalf("%v %v %v", ds, err, seen)
	}
	if _, err := r.RouteBatch(reqs[:1], router.ForHookTimeout(0)); err == nil {
		t.Fatal("batch timeout not validated")
	}
}

func TestBaseHookIsANoOp(t *testing.T) {
	type embed struct{ router.BaseHook }
	r := mustNew(t, router.WithInstalledHooks(&embed{}))
	if d, err := r.Route(englishText, none); err != nil || d.Model != router.English {
		t.Fatal(d, err)
	}
}

// A hook that calls Route on its own Router: fine by default (as in Python, where hooks are
// concurrent), a wait-for-itself under WithSerialHooks that only a timeout ends.
func TestHookMayCallRouteOnItsOwnRouterUnlessSerial(t *testing.T) {
	reenter := func(opts ...router.Option) (string, error) {
		r := mustNew(t, opts...)
		var inner atomic.Value
		var entered atomic.Bool
		_ = r.AddHook(hook(func(c *router.RouteContext) {
			if entered.CompareAndSwap(false, true) {
				d, err := r.Route("x", none, router.ForModel("english"))
				if err == nil {
					inner.Store(string(d.Model))
				}
			}
		}))
		_, err := r.Route("x", none)
		got, _ := inner.Load().(string)
		return got, err
	}
	if inner, err := reenter(); err != nil || inner != "english" {
		t.Fatalf("default: inner=%q err=%v", inner, err)
	}
	_, err := reenter(router.WithSerialHooks(), router.WithHookTimeout(100*time.Millisecond))
	if err == nil || !strings.Contains(err.Error(), "exceeded 0.1s") {
		t.Fatalf("serial: %v, want the timeout that ends the self-wait", err)
	}
}

type valueHook struct{ _ func() }

func (valueHook) OnRoute(*router.RouteContext) {}

type anyHook struct{ v any }

func (anyHook) OnRoute(*router.RouteContext) {}

func TestNonComparableHooksAreRefused(t *testing.T) {
	r := mustNew(t)
	for name, bad := range map[string]any{
		"struct holding a func":           valueHook{},
		"comparable struct, slice inside": anyHook{v: []int{1}},
	} {
		err := r.AddHook(bad)
		if err == nil || !strings.Contains(err.Error(), "pass a pointer") {
			t.Errorf("%s: %v", name, err)
		}
		if _, err := router.New(router.WithInstalledHooks(bad)); err == nil {
			t.Errorf("%s accepted by New", name)
		}
		if undo, err := r.InstallHooks(bad); err == nil {
			undo()
			t.Errorf("%s accepted by InstallHooks", name)
		}
	}
	// A comparable value is fine, and can be removed.
	ok := anyHook{v: 1}
	if err := r.AddHook(ok); err != nil || !r.RemoveHook(ok) {
		t.Fatalf("comparable value hook: %v", err)
	}
	// A pointer to the same non-comparable struct is fine.
	if err := r.AddHook(&valueHook{}); err != nil {
		t.Fatal(err)
	}
}

func TestInstallUndoRemovesExactlyItsOwnEntries(t *testing.T) {
	var order []string
	mk := func(n string) *router.HookFuncs { return hook(func(*router.RouteContext) { order = append(order, n) }) }
	a, b := mk("a"), mk("b")
	r := mustNew(t)
	undo, err := r.InstallHooks(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = r.AddHook(b, a) // an equal hook added after the install
	undo()
	if _, err := r.Route("x", none); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(order, ""); got != "ba" {
		t.Fatalf("after undo %q: the install's own a goes, the later a stays", got)
	}
}
