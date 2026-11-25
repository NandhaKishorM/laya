//go:build cgo && localgraphs

package onnx

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

// TestOtherGraphs needs the older exports beside the shared graph (full-multilingual: a fixed
// batch dimension; legacy-multilingual: weights embedded), which CI does not have. Run with
// -tags localgraphs. The fixed-batch rejection itself is covered without a graph by TestCheckGraph.
func TestOtherGraphs(t *testing.T) {
	p := need(t)
	closeShared(t)

	var ge *GraphError
	full := p.sibling("full-multilingual")
	if s, err := Open(full); !errors.As(err, &ge) || s != nil {
		t.Fatalf("Open(full) = %v, %v; want a *GraphError", s, err)
	}
	if ge.Path != full || !strings.Contains(ge.Problem, `"act_logits"`) || !strings.Contains(ge.Problem, "fixed batch") {
		t.Fatalf("GraphError = %v", ge)
	}
	if references() != 0 || ort.IsInitialized() {
		t.Fatalf("a rejected graph left refs %d, initialized %v", references(), ort.IsInitialized())
	}

	legacy, err := Open(p.sibling("legacy-multilingual"))
	if err != nil {
		t.Fatalf("legacy graph: %v", err)
	}
	b := makeBatch(3, 24, 4)
	out, err := legacy.Run(context.Background(), b)
	if err != nil || out.Check(b) != nil {
		t.Fatalf("legacy Run = %v, %v", out, err)
	}
	legacy.Close()
}

// sibling is another export beside the main one: <exports>/<name>/laya.onnx.
func (p paths) sibling(name string) string {
	return filepath.Join(filepath.Dir(filepath.Dir(p.graph)), name, "laya.onnx")
}
