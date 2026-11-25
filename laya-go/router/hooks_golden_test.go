package router_test

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/hookrun"
	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/router"
)

// hookRecorder collects what the hooks saw, in order, and the warnings the Router reported.
type hookRecorder struct {
	mu     sync.Mutex
	calls  []hookSeen
	runIDs []string
	warns  []string
}

type hookSeen struct {
	hook, model, reason string
	states, questions   []string
	runIDLen            int
	runIDShared         bool
	routerIsSelf        bool
}

func (r *hookRecorder) warn(msg string) {
	r.mu.Lock()
	r.warns = append(r.warns, msg)
	r.mu.Unlock()
}

// routeStub is the golden's hook: it records what it sees, then acts out its behaviour. Hooks that
// the golden gives no on_route are otherStub, which has no OnRoute method at all.
type routeStub struct {
	agent.BaseHook  // a process-wide default must be an agent.Hook
	name, behaviour string
	rec             *hookRecorder
	r               **router.Router
}

func (h *routeStub) OnRoute(c *router.RouteContext) {
	seen := hookSeen{hook: h.name, model: string(c.Decision.Model), reason: c.Decision.Reason,
		runIDLen: len(c.RunID), routerIsSelf: c.Router == *h.r}
	for _, s := range c.States {
		switch v := s.(type) {
		case string:
			seen.states = append(seen.states, v)
		case lang.Object:
			seen.states = append(seen.states, "<dict>")
		case []any:
			seen.states = append(seen.states, "<list>")
		default:
			seen.states = append(seen.states, fmt.Sprintf("<%T>", s))
		}
	}
	seen.questions = c.Questions.IDs()
	h.rec.mu.Lock()
	h.rec.runIDs = append(h.rec.runIDs, c.RunID)
	seen.runIDShared = c.RunID == h.rec.runIDs[0]
	h.rec.calls = append(h.rec.calls, seen)
	h.rec.mu.Unlock()
	kind, arg, _ := strings.Cut(h.behaviour, ":")
	switch kind {
	case "replace", "replace_raise":
		spec, _ := c.Router.Spec(router.Checkpoint(arg))
		c.Decision = router.Decision{Model: router.Checkpoint(arg), Repo: spec.String(), Reason: "replaced by " + h.name}
	case "suffix":
		c.Decision.Reason += "|" + h.name
	}
	switch kind {
	case "raise", "replace_raise":
		panic(fmt.Errorf("boom from %s", h.name))
	case "sleep":
		secs, _ := strconv.ParseFloat(arg, 64)
		time.Sleep(time.Duration(secs * float64(time.Second)))
	}
}

type otherStub struct {
	agent.BaseHook
	name string
	rec  *hookRecorder
}

func (h *otherStub) OnPredictEnd(*agent.PredictContext) {
	h.rec.mu.Lock()
	h.rec.calls = append(h.rec.calls, hookSeen{hook: h.name + ":other"})
	h.rec.mu.Unlock()
}

// timeoutName rewrites Python's "laya: hook X.on_route exceeded" to Go's, which names the hook's
// type and not the bound method.
var (
	timeoutName = regexp.MustCompile(`laya: hook \w+\.on_route exceeded`)
	failedName  = regexp.MustCompile(`laya: hook \w+\.on_route failed`)
)

func goText(s string) string {
	s = timeoutName.ReplaceAllString(s, "laya: hook routeStub exceeded")
	return failedName.ReplaceAllString(s, "laya: hook routeStub.on_route failed")
}

type hookCase struct {
	r    *router.Router
	rec  *hookRecorder
	opts []router.CallOption
}

func (hc *hookCase) snapshot() ([]hookSeen, []string) {
	hc.rec.mu.Lock()
	defer hc.rec.mu.Unlock()
	return append([]hookSeen(nil), hc.rec.calls...), append([]string(nil), hc.rec.warns...)
}

// buildHooked makes the Router and the call options for one golden record; the defaults are left
// installed for the caller to clear. ok is false when New refused the router.
func buildHooked(t *testing.T, rec fixtures.Object) (hc *hookCase, newErr error) {
	t.Helper()
	hc = &hookCase{rec: &hookRecorder{}}
	var ptr *router.Router
	var installed, call, defaults []any
	for _, hv := range fixtures.List(t, rec.Must("chain")) {
		h := fixtures.Obj(t, hv)
		name := str(t, h, "name")
		var hook any = &otherStub{name: name, rec: hc.rec}
		for _, e := range fixtures.Strings(t, h.Must("events")) {
			if e == "on_route" {
				hook = &routeStub{name: name, behaviour: str(t, h, "behaviour"), rec: hc.rec, r: &ptr}
			}
		}
		switch str(t, h, "where") {
		case "default":
			defaults = append(defaults, hook)
		case "installed":
			installed = append(installed, hook)
		default:
			call = append(call, hook)
		}
	}
	opts := []router.Option{router.WithInstalledHooks(installed...), router.WithHookWarnings(hc.rec.warn)}
	serial := false // Python's default is concurrent
	for _, m := range fixtures.Obj(t, rec.Must("router")) {
		switch m.Key {
		case "hooks_raise":
			opts = append(opts, router.WithHookErrors(fixtures.Bool(t, m.Value)))
		case "hooks_timeout":
			opts = append(opts, router.WithHookTimeout(seconds(t, m.Value)))
		case "hooks_concurrent":
			serial = !fixtures.Bool(t, m.Value)
		case "auto_task_detection":
			opts = append(opts, router.WithAutoTaskDetection(fixtures.Bool(t, m.Value)))
		default:
			t.Fatalf("unhandled router kwarg %q", m.Key)
		}
	}
	if serial {
		opts = append(opts, router.WithSerialHooks())
	}
	var err error
	hc.r, err = router.New(opts...)
	ptr = hc.r
	if err != nil {
		return nil, err
	}
	if len(call) > 0 {
		hc.opts = append(hc.opts, router.ForHooks(call...))
	}
	for _, m := range fixtures.Obj(t, rec.Must("call")) {
		switch m.Key {
		case "hooks_raise":
			hc.opts = append(hc.opts, router.ForHookErrors(fixtures.Bool(t, m.Value)))
		case "hooks_timeout":
			hc.opts = append(hc.opts, router.ForHookTimeout(seconds(t, m.Value)))
		default:
			t.Fatalf("unhandled call kwarg %q", m.Key)
		}
	}
	agent.ClearDefaultHooks()
	t.Cleanup(agent.ClearDefaultHooks)
	for _, d := range defaults {
		if err := agent.AddDefaultHook(d.(agent.Hook)); err != nil {
			t.Fatal(err)
		}
	}
	return hc, nil
}

func seconds(t *testing.T, v any) time.Duration {
	t.Helper()
	return time.Duration(fixtures.Float(t, v) * float64(time.Second))
}

func checkCalls(t *testing.T, got []hookSeen, want []any) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%d hook calls, want %d: %+v", len(got), len(want), got)
	}
	for i, wv := range want {
		w := fixtures.Obj(t, wv)
		g := got[i]
		if g.hook != str(t, w, "hook") || g.model != str(t, w, "model") || g.reason != str(t, w, "reason") ||
			strings.Join(g.states, "\x00") != strings.Join(fixtures.Strings(t, w.Must("states")), "\x00") ||
			strings.Join(g.questions, ",") != strings.Join(fixtures.Strings(t, w.Must("questions")), ",") ||
			g.runIDLen != fixtures.Int(t, w.Must("run_id_len")) || g.runIDShared != fixtures.Bool(t, w.Must("run_id_shared")) ||
			g.routerIsSelf != fixtures.Bool(t, w.Must("router_is_self")) {
			t.Errorf("call %d\n got %+v\nwant %v", i, g, w)
		}
	}
}

func checkWarnings(t *testing.T, got []string, want []any) {
	t.Helper()
	var w []string
	for _, v := range want {
		w = append(w, goText(fixtures.Str(t, v)))
	}
	if strings.Join(got, "|") != strings.Join(w, "|") {
		t.Errorf("warnings\n got %q\nwant %q", got, w)
	}
}

func TestRouteHooksGolden(t *testing.T) {
	cases := family(t, "hooks")
	if len(cases) != 339 {
		t.Fatalf("%d cases", len(cases))
	}
	var decided, failed, refusedRouters, warned, replaced, timedOut int
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		t.Run(fmt.Sprintf("%d-%s", i, str(t, rec, "case")), func(t *testing.T) {
			hc, err := buildHooked(t, rec)
			if want, bad := rec.Get("router_error"); bad {
				refusedRouters++
				if err == nil || err.Error() != fixtures.Str(t, want) {
					t.Fatalf("New error = %v, want %q", err, want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			state, qs, opts := callOptions(t, fixtures.Obj(t, rec.Must("route")))
			got, err := hc.r.Route(state, qs, append(opts, hc.opts...)...)
			calls, warns := hc.snapshot()
			checkCalls(t, calls, fixtures.List(t, rec.Must("calls")))
			if len(warns) > 0 {
				warned++
			}
			checkWarnings(t, warns, fixtures.List(t, rec.Must("warnings")))
			if want, bad := rec.Get("error"); bad && want != nil {
				failed++
				msg := fixtures.Str(t, fixtures.Obj(t, want).Must("message"))
				if strings.Contains(msg, "exceeded") {
					timedOut++
				}
				if err == nil || err.Error() != goText(msg) {
					t.Fatalf("Route error = %v, want %q", err, goText(msg))
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			decided++
			want := fixtures.Obj(t, rec.Must("decision"))
			if strings.HasPrefix(got.Reason, "replaced by") {
				replaced++
			}
			checkDecision(t, got, want)
		})
	}
	t.Logf("%d decided, %d failed (%d timeouts), %d routers refused, %d warned, %d replaced", decided, failed, timedOut, refusedRouters, warned, replaced)
	if decided < 150 || failed < 100 || timedOut < 15 || refusedRouters < 2 || warned < 25 || replaced < 25 {
		t.Fatalf("the golden stopped covering every outcome")
	}
}

func TestRouteBatchHooksGolden(t *testing.T) {
	cases := family(t, "hook_batches")
	if len(cases) != 42 {
		t.Fatalf("%d cases", len(cases))
	}
	var failed, hooked int
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		t.Run(fmt.Sprintf("%d-%s", i, str(t, rec, "case")), func(t *testing.T) {
			hc, err := buildHooked(t, rec)
			if err != nil {
				t.Fatal(err)
			}
			var reqs []router.Request
			for _, item := range fixtures.List(t, rec.Must("requests")) {
				state, qs, opts := callOptions(t, fixtures.Obj(t, item))
				reqs = append(reqs, router.Request{State: state, Questions: qs, Options: opts})
			}
			got, err := hc.r.RouteBatch(reqs, hc.opts...)
			calls, warns := hc.snapshot()
			checkCalls(t, calls, fixtures.List(t, rec.Must("calls")))
			checkWarnings(t, warns, fixtures.List(t, rec.Must("warnings")))
			if want, bad := rec.Get("error"); bad && want != nil {
				failed++
				msg := goText(fixtures.Str(t, fixtures.Obj(t, want).Must("message")))
				if err == nil || err.Error() != msg {
					t.Fatalf("RouteBatch error = %v, want %q", err, msg)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := fixtures.List(t, rec.Must("decisions"))
			if len(got) != len(want) {
				t.Fatalf("%d decisions, want %d", len(got), len(want))
			}
			if len(calls) > 0 {
				hooked++
			}
			for j := range got {
				checkDecision(t, got[j], fixtures.Obj(t, want[j]))
			}
		})
	}
	if failed < 3 || hooked < 15 {
		t.Fatalf("the golden stopped covering both outcomes: %d failed, %d hooked", failed, hooked)
	}
}

func TestTimeoutTextGolden(t *testing.T) {
	cases := family(t, "timeout_text")
	if len(cases) < 20 {
		t.Fatalf("%d cases", len(cases))
	}
	for _, cv := range cases {
		c := fixtures.Obj(t, cv)
		x := fixtures.Float(t, c.Must("seconds"))
		if got, want := hookrun.PyG(x), str(t, c, "text"); got != want {
			t.Errorf("PyG(%v) = %q, want %q", x, got, want)
		}
	}
}
