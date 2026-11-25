//go:build cgo

package onnx

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/infer"
)

func TestAlreadyCancelledContextDoesNotRun(t *testing.T) {
	s := sharedSession(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	before := s.runs.Load()
	_, err := s.Run(ctx, makeBatch(2, 12, 3))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Run = %v, want context.Canceled", err)
	}
	dctx, dcancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer dcancel()
	if _, err := s.Run(dctx, makeBatch(2, 12, 3)); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Run with an expired deadline = %v", err)
	}
	if n := s.runs.Load() - before; n != 0 {
		t.Fatalf("the model ran %d times for cancelled contexts", n)
	}
}

func TestInvalidBatchIsRefusedBeforeRunning(t *testing.T) {
	s := sharedSession(t)
	before := s.runs.Load()
	b := makeBatch(2, 12, 3)
	b.QType = b.QType[:1]
	var se *infer.ShapeError
	if _, err := s.Run(context.Background(), b); !errors.As(err, &se) || se.Name != infer.NameQType {
		t.Fatalf("Run = %v, want a ShapeError for qtype", err)
	}
	if s.runs.Load() != before {
		t.Fatal("the model ran on an invalid batch")
	}
	out, err := s.Run(context.Background(), infer.Batch{})
	if err != nil || len(out.Logits) != 0 || s.runs.Load() != before {
		t.Fatalf("empty batch: %v, %v", out, err)
	}
}

func TestCancelMidRun(t *testing.T) {
	s := sharedSession(t)
	b := makeBatch(24, 256, 4)
	start := time.Now()
	if _, err := s.Run(context.Background(), b); err != nil {
		t.Fatal(err)
	}
	full := time.Since(start)
	t.Logf("uncancelled run of the large batch: %v", full)

	ctx, cancel := context.WithCancel(context.Background())
	s.runs.Store(0)
	go func() {
		for s.runs.Load() == 0 {
			runtime.Gosched()
		}
		time.Sleep(full / 10)
		cancel()
	}()
	start = time.Now()
	_, err := s.Run(ctx, b)
	took := time.Since(start)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Run = %v after %v, want context.Canceled", err, took)
	}
	if took > full/2 {
		t.Fatalf("a run cancelled at 10%% took %v of %v: the terminate flag did not stop it", took, full)
	}
	t.Logf("cancelled run returned after %v", took)
	// The session is still usable, and a fresh context is not poisoned by the old flag.
	mustRun(t, s)
}

func TestNoGoroutineLeak(t *testing.T) {
	s := sharedSession(t)
	small := makeBatch(2, 16, 3)
	mustRun(t, s)
	settle := func() int {
		for i := 0; i < 100; i++ {
			runtime.GC()
			time.Sleep(10 * time.Millisecond)
		}
		return runtime.NumGoroutine()
	}
	baseline := settle()
	for i := 0; i < 50; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := s.Run(ctx, small); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled run %d: %v", i, err)
		}
	}
	for i := 0; i < 50; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(i%5)*time.Millisecond)
		_, err := s.Run(ctx, small)
		cancel()
		if err != nil && !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("racing run %d: %v", i, err)
		}
	}
	for i := 0; i < 50; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		if _, err := s.Run(ctx, small); err != nil {
			t.Fatalf("normal run %d: %v", i, err)
		}
		cancel()
		if _, err := s.Run(context.Background(), small); err != nil {
			t.Fatalf("background run %d: %v", i, err)
		}
	}
	after := settle()
	t.Logf("goroutines: baseline %d, after 150 cancellable and 50 plain runs %d", baseline, after)
	if after > baseline {
		t.Fatalf("goroutine count rose from %d to %d", baseline, after)
	}
}
