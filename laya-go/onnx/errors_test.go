//go:build cgo

package onnx

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	ort "github.com/yalue/onnxruntime_go"
)

func TestMissingSidecarIsAGraphError(t *testing.T) {
	p := need(t)
	data, err := os.ReadFile(p.graph)
	if err != nil {
		t.Fatal(err)
	}
	alone := filepath.Join(t.TempDir(), "laya.onnx")
	if err := os.WriteFile(alone, data, 0o644); err != nil {
		t.Fatal(err)
	}
	before := references()
	_, err = Open(alone)
	var ge *GraphError
	if !errors.As(err, &ge) {
		t.Fatalf("Open without the sidecar = %v, want a *GraphError", err)
	}
	if ge.Path != alone || !strings.Contains(ge.Problem, ".data") || !strings.Contains(ge.Problem, "beside") {
		t.Errorf("GraphError = %+v", ge)
	}
	if references() != before {
		t.Errorf("a failed Open leaked a runtime reference: %d, was %d", references(), before)
	}
}

type recorder struct{ calls []string }

func (r *recorder) SetIntraOpNumThreads(n int) error {
	r.calls = append(r.calls, fmt.Sprint("intra ", n))
	return nil
}
func (r *recorder) SetInterOpNumThreads(n int) error {
	r.calls = append(r.calls, fmt.Sprint("inter ", n))
	return nil
}
func (r *recorder) SetExecutionMode(m ort.ExecutionMode) error {
	r.calls = append(r.calls, "mode "+m.String())
	return nil
}

func TestInterOpThreadsSelectParallelMode(t *testing.T) {
	var r recorder
	if err := applyThreads(&r, options{inter: 3, intra: 2}); err != nil {
		t.Fatal(err)
	}
	got := strings.Join(r.calls, ",")
	if !strings.Contains(got, "mode "+ort.ExecutionMode(ort.ExecutionModeParallel).String()) || !strings.Contains(got, "inter 3") || !strings.Contains(got, "intra 2") {
		t.Fatalf("calls = %s", got)
	}
	r = recorder{}
	if err := applyThreads(&r, options{intra: 2}); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(r.calls, ","); got != "intra 2" {
		t.Fatalf("intra only: calls = %s", got)
	}
	if err := applyThreads(&r, options{}); err != nil || len(r.calls) != 1 {
		t.Fatalf("zero options touched the session options: %v %v", err, r.calls)
	}
}

func TestOpenWithInterOpThreads(t *testing.T) {
	p := need(t)
	closeShared(t)
	s, err := Open(p.graph, WithInterOpThreads(2), WithIntraOpThreads(2))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.RuntimeVersion() == "" {
		t.Error("RuntimeVersion is empty")
	}
	mustRun(t, s)
}

func TestRunErrorKeepsORTErrorWhenContextEnded(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	oe := errors.New("ORT said: bad shape")
	b := makeBatch(2, 3, 2)

	err := runError(ctx, false, oe, b)
	if !errors.Is(err, context.Canceled) || !errors.Is(err, oe) || !strings.Contains(err.Error(), "bad shape") {
		t.Errorf("failed run with ended context = %v, want both errors", err)
	}
	err = runError(ctx, true, oe, b)
	if !errors.Is(err, context.Canceled) || !strings.Contains(err.Error(), "run cancelled") || errors.Is(err, oe) {
		t.Errorf("terminated run = %v, want the cancellation message", err)
	}
	err = runError(context.Background(), false, oe, b)
	if !errors.Is(err, oe) || errors.Is(err, context.Canceled) {
		t.Errorf("plain failure = %v", err)
	}
}

func TestZeroValueSession(t *testing.T) {
	var s Session
	before := references()
	if err := s.Close(); err != nil {
		t.Fatalf("Close = %v", err)
	}
	if references() != before {
		t.Errorf("Close of a zero Session released a reference: %d, was %d", references(), before)
	}
	if _, err := s.Run(context.Background(), infer.Batch{}); !errors.Is(err, infer.ErrClosed) {
		t.Fatalf("Run = %v, want infer.ErrClosed", err)
	}
}
