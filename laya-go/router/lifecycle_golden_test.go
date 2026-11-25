package router_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/router"
)

// lifeLog is the golden's event log, shared by every hook of a case.
type lifeLog struct {
	mu     sync.Mutex
	events []string
	labels map[*agent.Agent]string
}

func (l *lifeLog) add(s string) {
	l.mu.Lock()
	l.events = append(l.events, s)
	l.mu.Unlock()
}

func (l *lifeLog) label(a *agent.Agent) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.labels[a]
}

// lifeAct is what a golden hook does after it logs: nothing, raise, or sleep.
type lifeAct struct {
	name, behaviour string
	log             *lifeLog
}

func (h *lifeAct) act() {
	kind, arg, _ := strings.Cut(h.behaviour, ":")
	switch kind {
	case "raise":
		panic(fmt.Errorf("boom from %s", h.name))
	case "sleep":
		var secs float64
		fmt.Sscanf(arg, "%g", &secs)
		time.Sleep(time.Duration(secs * float64(time.Second)))
	}
}

func (h *lifeAct) onLoad(c *router.LoadContext) {
	h.log.add(fmt.Sprintf("%s.on_load:%s:%s", h.name, c.Model, h.log.label(c.Agent)))
	h.act()
}

func (h *lifeAct) onEvict(c *router.EvictContext) {
	h.log.add(fmt.Sprintf("%s.on_evict:%s", h.name, c.Model))
	h.act()
}

// The golden's hooks implement exactly the events it names, so each shape is its own type; the
// failure text names the Go type, which lifeText maps back from Python's.
type lifeLoad struct{ lifeAct }

func (h *lifeLoad) OnLoad(c *router.LoadContext) { h.onLoad(c) }

type lifeEvict struct{ lifeAct }

func (h *lifeEvict) OnEvict(c *router.EvictContext) { h.onEvict(c) }

type lifeBoth struct{ lifeAct }

func (h *lifeBoth) OnLoad(c *router.LoadContext)   { h.onLoad(c) }
func (h *lifeBoth) OnEvict(c *router.EvictContext) { h.onEvict(c) }

type lifeOnlyRoute struct{}

func (lifeOnlyRoute) OnRoute(*router.RouteContext) {}

type lifePred struct{ agent.BaseHook }

var lifeText = regexp.MustCompile(`laya: hook (\w+)\.(on_load|on_evict) (failed|exceeded)`)

// goLifeText rewrites Python's "laya: hook Boom.on_load failed" to the Go type's name.
func goLifeText(s string, types map[string]string) string {
	return lifeText.ReplaceAllStringFunc(s, func(m string) string {
		p := lifeText.FindStringSubmatch(m)
		if p[3] == "exceeded" {
			return "laya: hook " + types[p[1]] + " exceeded"
		}
		return "laya: hook " + types[p[1]] + "." + p[2] + " failed"
	})
}

func TestLifecycleGolden(t *testing.T) {
	root := fixtures.Load(t, "go_extra.json")
	cases := fixtures.List(t, root.Must("lifecycle"))
	if len(cases) < 250 {
		t.Fatalf("only %d lifecycle cases", len(cases))
	}
	ops := 0
	for _, cv := range cases {
		c := fixtures.Obj(t, cv)
		label := str(t, c, "case")
		t.Run(label, func(t *testing.T) { ops += replayLifecycle(t, c) })
	}
	if ops < 1500 {
		t.Fatalf("only %d ops replayed", ops)
	}
}

func replayLifecycle(t *testing.T, c fixtures.Object) int {
	t.Helper()
	lg := &lifeLog{labels: map[*agent.Agent]string{}}
	var mu sync.Mutex
	calls := 0
	fail := map[string]bool{}
	factory := func(ctx context.Context, cp router.Checkpoint) (*agent.Agent, error) {
		mu.Lock()
		calls++
		n := calls
		bad := fail[string(cp)]
		mu.Unlock()
		if bad {
			return nil, errors.New("boom building " + string(cp))
		}
		a := newProbe(t, nil)
		lg.mu.Lock()
		lg.labels[a] = fmt.Sprintf("built-%d", n)
		lg.mu.Unlock()
		return a, nil
	}
	var warns []string
	var wmu sync.Mutex
	opts := []router.Option{router.WithAgentFactory(factory), router.WithHookWarnings(func(m string) {
		wmu.Lock()
		warns = append(warns, m)
		wmu.Unlock()
	})}
	for _, m := range fixtures.Obj(t, c.Must("router")) {
		switch m.Key {
		case "max_loaded":
			opts = append(opts, router.WithMaxLoaded(fixtures.Int(t, m.Value)))
		case "hooks_raise":
			opts = append(opts, router.WithHookErrors(fixtures.Bool(t, m.Value)))
		case "hooks_timeout":
			opts = append(opts, router.WithHookTimeout(seconds(t, m.Value)))
		case "hooks_concurrent":
			if !fixtures.Bool(t, m.Value) {
				opts = append(opts, router.WithSerialHooks())
			}
		case "standalone_repos":
			opts = append(opts, router.WithStandaloneRepos(fixtures.Bool(t, m.Value)))
		default:
			t.Fatalf("unhandled router kwarg %q", m.Key)
		}
	}
	types := map[string]string{}
	var hooks []any
	for _, hv := range fixtures.List(t, c.Must("hooks")) {
		h := fixtures.Obj(t, hv)
		name, beh := str(t, h, "name"), str(t, h, "behaviour")
		act := lifeAct{name: name, behaviour: beh, log: lg}
		ev := fixtures.Strings(t, h.Must("events"))
		has := func(e string) bool {
			for _, x := range ev {
				if x == e {
					return true
				}
			}
			return false
		}
		switch {
		case has("on_load") && has("on_evict"):
			hooks, types[name] = append(hooks, &lifeBoth{act}), "lifeBoth"
		case has("on_load"):
			hooks, types[name] = append(hooks, &lifeLoad{act}), "lifeLoad"
		case has("on_evict"):
			hooks, types[name] = append(hooks, &lifeEvict{act}), "lifeEvict"
		case has("on_route"):
			hooks, types[name] = append(hooks, &lifeOnlyRoute{}), "lifeOnlyRoute"
		default:
			hooks, types[name] = append(hooks, &lifePred{}), "lifePred"
		}
	}
	opts = append(opts, router.WithInstalledHooks(hooks...))
	r, err := router.New(opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	if got, want := r.MaxLoaded(), fixtures.Int(t, c.Must("max_loaded")); got != want {
		t.Fatalf("initial max_loaded = %d, want %d", got, want)
	}
	attached := map[string]*agent.Agent{}
	ctx := context.Background()
	steps := fixtures.List(t, c.Must("ops"))
	for i, sv := range steps {
		op := fixtures.Obj(t, sv)
		lg.mu.Lock()
		before := len(lg.events)
		lg.mu.Unlock()
		wmu.Lock()
		warns = warns[:0]
		wmu.Unlock()
		var gotErr error
		gotAgent := ""
		name, _ := op.Get("name")
		switch kind := str(t, op, "op"); kind {
		case "load":
			var a *agent.Agent
			a, gotErr = r.Load(ctx, fixtures.Str(t, name))
			gotAgent = lg.label(a)
		case "preload":
			var names []string
			if v, _ := op.Get("names"); v != nil {
				names = fixtures.Strings(t, v)
			}
			gotErr = r.Preload(ctx, names...)
		case "attach":
			label := str(t, op, "agent")
			a := attached[label]
			if a == nil {
				a = newProbe(t, nil)
				attached[label] = a
				lg.mu.Lock()
				lg.labels[a] = label
				lg.mu.Unlock()
			}
			gotErr = r.Attach(fixtures.Str(t, name), a)
		case "unload":
			if name == nil {
				_, gotErr = r.UnloadAll(ctx)
			} else {
				_, gotErr = r.Unload(ctx, fixtures.Str(t, name))
			}
		case "fail":
			mu.Lock()
			fail[fixtures.Str(t, name)] = true
			mu.Unlock()
		case "heal":
			mu.Lock()
			delete(fail, fixtures.Str(t, name))
			mu.Unlock()
		default:
			t.Fatalf("unknown op %q", kind)
		}
		where := fmt.Sprintf("op %d (%s %v)", i, str(t, op, "op"), name)
		if want, ok := op.Get("error"); ok {
			w := goLifeText(fixtures.Str(t, want), types)
			if gotErr == nil || !(gotErr.Error() == w || strings.HasSuffix(gotErr.Error(), ": "+w)) {
				t.Fatalf("%s: error %v, want %q", where, gotErr, w)
			}
		} else if gotErr != nil {
			t.Fatalf("%s: unexpected error %v", where, gotErr)
		}
		if want, ok := op.Get("agent"); ok && str(t, op, "op") == "load" && gotAgent != fixtures.Str(t, want) {
			t.Fatalf("%s: agent %q, want %q", where, gotAgent, want)
		}
		lg.mu.Lock()
		events := append([]string{}, lg.events[before:]...)
		lg.mu.Unlock()
		if want := fixtures.Strings(t, op.Must("events")); !reflect.DeepEqual(events, append([]string{}, want...)) {
			t.Fatalf("%s: events %q, want %q", where, events, want)
		}
		wmu.Lock()
		gotWarns := append([]string{}, warns...)
		wmu.Unlock()
		var wantWarns []string
		for _, w := range fixtures.Strings(t, op.Must("warnings")) {
			wantWarns = append(wantWarns, goLifeText(w, types))
		}
		if !reflect.DeepEqual(append([]string{}, gotWarns...), append([]string{}, wantWarns...)) {
			t.Fatalf("%s: warnings %q, want %q", where, gotWarns, wantWarns)
		}
		if got, want := names(r.Loaded()), fixtures.Strings(t, op.Must("loaded")); !reflect.DeepEqual(append([]string{}, got...), append([]string{}, want...)) {
			t.Fatalf("%s: loaded %v, want %v", where, got, want)
		}
		if got, want := r.MaxLoaded(), fixtures.Int(t, op.Must("max_loaded")); got != want {
			t.Fatalf("%s: max_loaded %d, want %d", where, got, want)
		}
	}
	return len(steps)
}
