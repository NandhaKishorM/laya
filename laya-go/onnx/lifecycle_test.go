//go:build cgo

package onnx

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	ort "github.com/yalue/onnxruntime_go"
)

func mustRun(t *testing.T, s *Session) infer.Output {
	t.Helper()
	b := makeBatch(2, 12, 3)
	out, err := s.Run(context.Background(), b)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if err := out.Check(b); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestOpenCloseReopen(t *testing.T) {
	p := need(t)
	closeShared(t)
	link := filepath.Join(t.TempDir(), filepath.Base(p.library))
	if err := os.Symlink(p.library, link); err != nil {
		t.Fatal(err)
	}
	for i, opts := range [][]Option{nil, nil, {WithLibrary(p.library)}, {WithLibrary(link)}} {
		s, err := Open(p.graph, opts...)
		if err != nil {
			t.Fatalf("open %d: %v", i, err)
		}
		if references() != 1 || !ort.IsInitialized() {
			t.Fatalf("open %d: refs %d, initialized %v", i, references(), ort.IsInitialized())
		}
		if v := s.RuntimeVersion(); v == "" || v != ort.GetVersion() {
			t.Fatalf("RuntimeVersion = %q, runtime says %q", v, ort.GetVersion())
		}
		mustRun(t, s)
		if err := s.Close(); err != nil {
			t.Fatalf("close %d: %v", i, err)
		}
		if references() != 0 || ort.IsInitialized() {
			t.Fatalf("close %d: refs %d, initialized %v; the last Close must unload the runtime", i, references(), ort.IsInitialized())
		}
	}
}

func TestTwoSessionsAtOnce(t *testing.T) {
	p := need(t)
	closeShared(t)
	a, err := Open(p.graph)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Open(p.graph, WithLibrary(p.library), WithIntraOpThreads(2), WithInterOpThreads(1))
	if err != nil {
		t.Fatalf("second session: %v", err)
	}
	if references() != 2 {
		t.Fatalf("refs = %d, want 2", references())
	}
	want := mustRun(t, a)
	got := mustRun(t, b)
	for i := range want.Logits {
		if d := want.Logits[i] - got.Logits[i]; d > 1e-3 || d < -1e-3 {
			t.Fatalf("logit %d differs between sessions: %v vs %v", i, want.Logits[i], got.Logits[i])
		}
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if references() != 1 || !ort.IsInitialized() {
		t.Fatalf("closing one of two: refs %d, initialized %v; the runtime must stay loaded", references(), ort.IsInitialized())
	}
	mustRun(t, b)
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if references() != 0 || ort.IsInitialized() {
		t.Fatalf("after both: refs %d, initialized %v", references(), ort.IsInitialized())
	}
}

func TestConflictingExplicitPath(t *testing.T) {
	p := need(t)
	closeShared(t)
	first, err := Open(p.graph) // loaded through LibraryEnv
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	_, err = Open(p.graph, WithLibrary("/nonexistent/libonnxruntime.dylib"))
	if !errors.Is(err, ErrRuntimeConflict) {
		t.Fatalf("Open with another library = %v, want ErrRuntimeConflict", err)
	}
	if references() != 1 {
		t.Fatalf("a refused Open changed the reference count to %d", references())
	}
	mustRun(t, first)
	same, err := Open(p.graph, WithLibrary(p.library))
	if err != nil {
		t.Fatalf("the same library named explicitly: %v", err)
	}
	if references() != 2 {
		t.Fatalf("refs = %d, want 2", references())
	}
	same.Close()
}

func TestBadLibraryPath(t *testing.T) {
	p := need(t)
	closeShared(t)
	// The environment names a good library; the explicit path must win over it.
	bad := "/nonexistent/libonnxruntime.dylib"
	s, err := Open(p.graph, WithLibrary(bad))
	if !errors.Is(err, ErrRuntime) || s != nil || !strings.Contains(err.Error(), bad) {
		t.Fatalf("Open = %v, %v; want ErrRuntime naming %s", s, err, bad)
	}
	if references() != 0 || ort.IsInitialized() {
		t.Fatalf("a failed Open left refs %d, initialized %v", references(), ort.IsInitialized())
	}
	t.Setenv(LibraryEnv, bad)
	if _, err := Open(p.graph); !errors.Is(err, ErrRuntime) {
		t.Fatalf("Open with a bad %s = %v", LibraryEnv, err)
	}
	// A good explicit path overrides a bad environment, and the process recovers from the failures.
	good, err := Open(p.graph, WithLibrary(p.library))
	if err != nil {
		t.Fatalf("recovery: %v", err)
	}
	mustRun(t, good)
	good.Close()
}

func TestAdoptsAnInitializedRuntime(t *testing.T) {
	p := need(t)
	closeShared(t)
	ort.SetSharedLibraryPath(p.library)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if ort.IsInitialized() {
			ort.DestroyEnvironment()
		}
	}()
	// The path is ignored: the application's runtime is used.
	s, err := Open(p.graph, WithLibrary("/nonexistent/libonnxruntime.dylib"))
	if err != nil {
		t.Fatalf("Open beside an application-initialized runtime: %v", err)
	}
	mustRun(t, s)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if !ort.IsInitialized() {
		t.Fatal("Close destroyed a runtime the application owns")
	}
	if references() != 0 {
		t.Fatalf("refs = %d", references())
	}
	if err := ort.DestroyEnvironment(); err != nil {
		t.Fatal(err)
	}
	// With the application's runtime gone, the next Open loads its own again.
	s, err = Open(p.graph)
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	if ort.IsInitialized() {
		t.Fatal("a runtime we loaded was left behind")
	}
}

func TestRunAfterCloseAndDoubleClose(t *testing.T) {
	p := need(t)
	closeShared(t)
	s, err := Open(p.graph)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close = %v", err)
	}
	if references() != 0 {
		t.Fatalf("double Close released twice: refs %d", references())
	}
	if _, err := s.Run(context.Background(), makeBatch(1, 8, 2)); !errors.Is(err, infer.ErrClosed) {
		t.Fatalf("Run after Close = %v, want infer.ErrClosed", err)
	}
	if s.RuntimeVersion() == "" {
		t.Error("RuntimeVersion is empty after Close")
	}
	// A second session opened and closed around the first one's double Close must not be affected.
	other, err := Open(p.graph)
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	if references() != 1 {
		t.Fatalf("a stale Close dropped another session's hold: refs %d", references())
	}
	mustRun(t, other)
	other.Close()
}

func TestCloseWaitsForRunsInFlight(t *testing.T) {
	p := need(t)
	closeShared(t)
	s, err := Open(p.graph)
	if err != nil {
		t.Fatal(err)
	}
	b := makeBatch(24, 256, 4)
	var wg sync.WaitGroup
	var runErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, runErr = s.Run(context.Background(), b)
	}()
	for s.runs.Load() == 0 {
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	if runErr != nil {
		t.Fatalf("the in-flight run failed: %v", runErr)
	}
}

// TestLayoutErrors needs no graph beyond the shared one: the split layout and a missing file are
// built in a temporary directory.
func TestLayoutErrors(t *testing.T) {
	need(t)
	closeShared(t)
	splitDir := t.TempDir()
	for _, name := range []string{"encoder.onnx", "head.onnx"} {
		if err := os.WriteFile(filepath.Join(splitDir, name), []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	var ge *GraphError
	if s, err := Open(splitDir); !errors.As(err, &ge) || s != nil || !strings.Contains(ge.Problem, "split encoder.onnx + head.onnx layout is not supported") {
		t.Fatalf("Open(split) = %v, %v", s, err)
	}
	if _, err := Open(filepath.Join(splitDir, "laya.onnx")); !errors.As(err, &ge) || !strings.Contains(ge.Problem, "split") {
		t.Fatalf("Open(split/laya.onnx) = %v", err)
	}
	if _, err := Open(filepath.Join(splitDir, "nope.onnx")); !errors.As(err, &ge) {
		t.Fatalf("Open(missing) = %v", err)
	}
	if references() != 0 {
		t.Fatalf("refs = %d", references())
	}
}

func TestOpenRejectsBadOptions(t *testing.T) {
	p := need(t)
	closeShared(t)
	if _, err := Open(p.graph, WithIntraOpThreads(-1)); err == nil || references() != 0 {
		t.Fatalf("Open = %v, refs %d", err, references())
	}
}
