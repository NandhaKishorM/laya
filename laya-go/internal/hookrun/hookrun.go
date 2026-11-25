// Package hookrun is the hook dispatcher the agent and the router share (hooks.dispatch): an
// optional serial lock, a per-hook timeout, panic recovery and raise-or-warn. It is internal so
// neither package grows public API for it.
package hookrun

import (
	"context"
	"fmt"
	"log"
	"math"
	"reflect"
	"strconv"
	"sync"
	"time"
)

// Name is the type name a hook failure message carries.
func Name(h any) string {
	t := reflect.TypeOf(h)
	if t == nil {
		return "<nil>"
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t.Name()
}

// Comparable reports whether h can be compared with == without panicking: its dynamic type is
// comparable and so is every value it holds.
func Comparable(h any) (ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	return h != nil && h == h
}

// Same compares hooks by identity: equal dynamic types that are comparable and equal. A value
// that cannot be compared is never the same as anything.
func Same(a, b any) (same bool) {
	defer func() {
		if recover() != nil {
			same = false
		}
	}()
	ta, tb := reflect.TypeOf(a), reflect.TypeOf(b)
	return ta == tb && ta != nil && ta.Comparable() && a == b
}

// Lock serialises hook calls: one hook runs at a time across every dispatch that shares it. It is
// a channel, so a waiter can give up. The zero value is a free lock; a nil *Lock means hooks run
// concurrently.
type Lock struct {
	once sync.Once
	sem  chan struct{}
}

func (l *Lock) ch() chan struct{} {
	l.once.Do(func() { l.sem = make(chan struct{}, 1) })
	return l.sem
}

// Policy is how a failing or slow hook is treated: Raise ends the chain with the error, else Warn
// gets the message (nil logs it) and the chain goes on; Timeout above 0 bounds the wait for each
// hook.
type Policy struct {
	Timeout time.Duration
	Raise   bool
	Warn    func(msg string)
}

// Run is one dispatcher configuration.
type Run struct {
	Lock *Lock
	// Quit ends the wait for Lock with ErrQuit; nil never does.
	Quit    <-chan struct{}
	ErrQuit error
	// Detached makes the wait for Lock ignore ctx, for events that report on finished work.
	Detached bool
	Policy   Policy
}

// ValidateTimeout is hooks.validate_timeout for seconds: positive and finite.
func ValidateTimeout(seconds float64) error {
	if math.IsNaN(seconds) || math.IsInf(seconds, 0) || seconds <= 0 {
		return fmt.Errorf("hooks_timeout must be a positive finite number or None; got %s", strconv.FormatFloat(seconds, 'g', -1, 64))
	}
	return nil
}

// PyG formats x as Python's "%g": six significant digits, trailing zeros dropped, an exponent of
// at least two digits when it is below -4 or at least 6.
func PyG(x float64) string { return strconv.FormatFloat(x, 'g', 6, 64) }

// Dispatch calls call(i) for i in 0..n-1 in order, each under Lock, Policy.Timeout and the panic
// recovery, as hooks.dispatch. name(i) names hook i; it is only called to build a message.
// event is Python's event name.
func (r Run) Dispatch(ctx context.Context, event string, n int, name func(int) string, call func(int)) error {
	for i := 0; i < n; i++ {
		err := r.callHook(ctx, func() string { return name(i) }, func() { call(i) })
		if err == nil {
			continue
		}
		if r.Policy.Raise {
			return err
		}
		msg := fmt.Sprintf("laya: hook %s.%s failed: %s", name(i), event, err)
		if r.Policy.Warn != nil {
			r.Policy.Warn(msg)
		} else {
			log.Print(msg)
		}
	}
	return nil
}

// callHook runs one hook in its own goroutine, so a timeout can stop the wait for a started
// hook; Quit and the caller's context only end a wait for the serial lock. Under the lock the
// lock covers the wait, not the goroutine, as in Python.
func (r Run) callHook(ctx context.Context, name func() string, fn func()) error {
	if r.Lock != nil {
		sem := r.Lock.ch()
		if r.Detached {
			ctx = context.WithoutCancel(ctx)
		}
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
		case <-r.Quit:
			return r.ErrQuit
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	res := make(chan error, 1) // buffered: an abandoned hook still finishes without blocking
	go func() {
		returned := false
		defer func() {
			rec := recover()
			switch {
			case returned:
			case rec == nil: // runtime.Goexit, e.g. t.FailNow: no return and no panic
				res <- fmt.Errorf("laya: hook %s exited without returning", name())
			default:
				if err, ok := rec.(error); ok {
					res <- err
				} else {
					res <- fmt.Errorf("%v", rec)
				}
			}
		}()
		fn()
		returned = true
		res <- nil
	}()
	var expired <-chan time.Time
	if r.Policy.Timeout > 0 {
		t := time.NewTimer(r.Policy.Timeout)
		defer t.Stop()
		expired = t.C
	}
	// Once started, a hook is waited for until it returns or times out: abandoning it on Quit
	// or on the caller's context would leave it racing the next hook and the call's results.
	select {
	case err := <-res:
		return err
	case <-expired:
		return fmt.Errorf("laya: hook %s exceeded %ss", name(), PyG(r.Policy.Timeout.Seconds()))
	}
}
