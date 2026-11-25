package hookrun

import (
	"bytes"
	"context"
	"errors"
	"log"
	"math"
	"strings"
	"testing"
	"time"
)

type withAny struct{ v any }

func TestSameDoesNotPanicOnHiddenNonComparable(t *testing.T) {
	a, b := withAny{[]int{1}}, withAny{[]int{1}}
	if Same(a, b) || Same(a, a) {
		t.Fatal("a value holding a slice is never the same as anything")
	}
	if Comparable(a) || Comparable(func() {}) || Comparable(nil) || !Comparable(&a) || !Comparable(withAny{1}) {
		t.Fatal("Comparable misjudged")
	}
	p := &a
	if !Same(p, p) || Same(p, &b) || Same(1, "1") || Same(nil, nil) {
		t.Fatal("Same misjudged pointers or nil")
	}
}

func TestZeroValuesAreUsable(t *testing.T) {
	var buf bytes.Buffer
	prev := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(prev) })
	var run Run // zero Policy: Raise false, Warn nil logs; zero Lock field is nil: concurrent
	err := run.Dispatch(context.Background(), "on_route", 2, func(i int) string { return "H" }, func(i int) {
		if i == 0 {
			panic(errors.New("boom"))
		}
	})
	if err != nil || !strings.Contains(buf.String(), "laya: hook H.on_route failed: boom") {
		t.Fatalf("err=%v log=%q", err, buf.String())
	}
	run = Run{Lock: new(Lock), Policy: Policy{Raise: true, Timeout: 20 * time.Millisecond}} // zero Lock: free
	if err := run.Dispatch(context.Background(), "e", 1, func(int) string { return "S" }, func(int) {}); err != nil {
		t.Fatal(err)
	}
	var l Lock
	run = Run{Lock: &l, Policy: Policy{Raise: true}}
	if err := run.Dispatch(context.Background(), "e", 3, func(int) string { return "S" }, func(int) {}); err != nil {
		t.Fatal(err)
	}
}

func TestTimeoutTextIsPythonG(t *testing.T) {
	for d, want := range map[time.Duration]string{
		123456700 * time.Nanosecond: "0.123457", 100 * time.Nanosecond: "1e-07",
		50 * time.Millisecond: "0.05",
	} {
		run := Run{Policy: Policy{Raise: true, Timeout: d}}
		err := run.Dispatch(context.Background(), "e", 1, func(int) string { return "S" }, func(int) { time.Sleep(time.Second) })
		if want := "laya: hook S exceeded " + want + "s"; err == nil || err.Error() != want {
			t.Errorf("%v: %v, want %q", d, err, want)
		}
	}
}

func TestNameIsOnlyComputedOnTheErrorPath(t *testing.T) {
	calls := 0
	run := Run{Policy: Policy{Raise: true}}
	_ = run.Dispatch(context.Background(), "e", 3, func(int) string { calls++; return "x" }, func(int) {})
	if calls != 0 {
		t.Fatalf("name called %d times on success", calls)
	}
}

type nanHook struct{ v float64 }

func TestComparableRefusesAValueThatIsNotEqualToItself(t *testing.T) {
	if Comparable(nanHook{math.NaN()}) {
		t.Fatal("a NaN-holding value can never be removed by equality, so it must be refused")
	}
	if !Comparable(&nanHook{math.NaN()}) {
		t.Fatal("a pointer is always removable by identity")
	}
}
