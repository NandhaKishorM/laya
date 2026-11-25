//go:build cgo

package onnx

import (
	"errors"
	"strings"
	"testing"
)

// A wrong digest is refused before the runtime is touched: no library is loaded and no reference
// is taken, so the library path below, which does not exist, is never reached.
func TestWrongDigestIsRefusedBeforeTheRuntimeLoads(t *testing.T) {
	path, sum := stubGraph(t)
	before := references()
	_, err := Open(path, WithLibrary("/nonexistent/libonnxruntime"), WithExpectedSHA256(strings.Repeat("0", 64)))
	if !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("Open = %v, want ErrDigestMismatch", err)
	}
	if references() != before {
		t.Errorf("a refused Open took a runtime reference")
	}
	// the right digest passes the check and reaches the runtime, which then fails on the library
	_, err = Open(path, WithLibrary("/nonexistent/libonnxruntime"), WithExpectedSHA256(sum))
	if err == nil || errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("Open with the right digest = %v, want a runtime error", err)
	}
}
