//go:build cgo

package onnx

import (
	"os"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/infer"
)

// paths holds the three locations the model-backed tests need.
type paths struct{ graph, library, checkpoints string }

// need returns the model-backed tests' inputs, skipping when one is unset. CI turns a skip into
// a failure.
func need(t testing.TB) paths {
	t.Helper()
	p := paths{os.Getenv("LAYA_ONNX_GRAPH"), os.Getenv(LibraryEnv), os.Getenv("LAYA_CHECKPOINTS")}
	if p.graph == "" || p.library == "" || p.checkpoints == "" {
		t.Skip("set LAYA_ONNX_GRAPH, LAYA_ONNXRUNTIME_LIBRARY and LAYA_CHECKPOINTS")
	}
	return p
}

// The graph is 1.29 GB, so tests that are not about lifecycle share one session. Lifecycle tests
// call closeShared first, so they start with no runtime loaded.
var shared struct {
	mu sync.Mutex
	s  *Session
}

func sharedSession(t testing.TB) *Session {
	t.Helper()
	p := need(t)
	shared.mu.Lock()
	defer shared.mu.Unlock()
	if shared.s == nil {
		s, err := Open(p.graph)
		if err != nil {
			t.Fatalf("Open(%s): %v", p.graph, err)
		}
		shared.s = s
	}
	return shared.s
}

func closeShared(t testing.TB) {
	t.Helper()
	shared.mu.Lock()
	defer shared.mu.Unlock()
	if shared.s != nil {
		if err := shared.s.Close(); err != nil {
			t.Fatalf("closing the shared session: %v", err)
		}
		shared.s = nil
	}
	if n := references(); n != 0 {
		t.Fatalf("%d references to the runtime remain", n)
	}
}

func TestMain(m *testing.M) {
	code := m.Run()
	shared.mu.Lock()
	if shared.s != nil {
		shared.s.Close()
	}
	shared.mu.Unlock()
	os.Exit(code)
}

// keepOpen lets a test hand the shared session to agent.Open without the agent closing it.
type keepOpen struct{ infer.Session }

func (keepOpen) Close() error { return nil }

// makeBatch builds a valid batch of rows x length tokens with the given marker slots.
func makeBatch(rows, length, markers int) infer.Batch {
	b := infer.Batch{
		InputIDs: make([]int64, rows*length), AttentionMask: make([]int64, rows*length),
		MarkerPos: make([]int64, rows*markers), MarkerMask: make([]bool, rows*markers),
		QType: make([]int64, rows), Rows: rows, Length: length, Markers: markers,
	}
	for i := range b.InputIDs {
		b.InputIDs[i] = int64(5 + i%97)
		b.AttentionMask[i] = 1
	}
	for i := range b.MarkerPos {
		b.MarkerPos[i] = int64(1 + i%markers*2%(length-1))
		b.MarkerMask[i] = true
	}
	for i := range b.QType {
		b.QType[i] = int64(i % 3)
	}
	return b
}
