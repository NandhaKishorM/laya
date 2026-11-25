package onnx

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/NandhaKishorM/laya/laya-go/infer"
)

// GraphError reports a graph file laya cannot run: it is missing, is not the fused layout, or
// declares tensors other than the ones the agent feeds.
type GraphError struct {
	// Path is the graph file as passed to Open.
	Path string
	// Problem says what is wrong and, where it helps, how to fix it.
	Problem string
}

// Error names the file and the problem.
func (e *GraphError) Error() string { return fmt.Sprintf("onnx: graph %s: %s", e.Path, e.Problem) }

const reexport = "re-export with scripts/export_onnx.py"

// tensorInfo is what checkGraph needs to know about one declared input or output. Dims holds -1
// for a dynamic dimension.
type tensorInfo struct {
	Name  string
	DType string // "int64", "bool", "float32", or whatever else the graph declares
	Dims  []int64
}

type tensorSpec struct {
	name, dtype string
	rank        int
	// fixedOK is the one fixed size a non-batch dimension may have, or 0 for none.
	fixedOK int64
}

var (
	wantInputs = []tensorSpec{
		{infer.NameInputIDs, "int64", 2, 0}, {infer.NameAttentionMask, "int64", 2, 0},
		{infer.NameMarkerPos, "int64", 2, 0}, {infer.NameMarkerMask, "bool", 2, 0},
		{infer.NameQType, "int64", 1, 0},
	}
	wantOutputs = []tensorSpec{
		{infer.NameLogits, "float32", 2, 0}, {infer.NameActLogits, "float32", 2, 2},
	}
)

// checkGraph verifies the declared tensors are exactly the five inputs and two outputs, with
// the dtypes and ranks the agent feeds, and that every dimension that varies per call is
// dynamic. The returned *GraphError has no Path; Open fills it in.
func checkGraph(inputs, outputs []tensorInfo) error {
	if err := checkSide("input", inputs, wantInputs); err != nil {
		return err
	}
	return checkSide("output", outputs, wantOutputs)
}

func checkSide(kind string, got []tensorInfo, want []tensorSpec) error {
	byName := make(map[string]tensorInfo, len(got))
	for _, g := range got {
		byName[g.Name] = g
	}
	for _, w := range want {
		g, ok := byName[w.name]
		if !ok {
			return &GraphError{Problem: fmt.Sprintf("missing %s %q; %s", kind, w.name, reexport)}
		}
		if g.DType != w.dtype {
			return &GraphError{Problem: fmt.Sprintf("%s %q is %s, want %s; %s", kind, w.name, g.DType, w.dtype, reexport)}
		}
		if len(g.Dims) != w.rank {
			return &GraphError{Problem: fmt.Sprintf("%s %q has rank %d, want %d; %s", kind, w.name, len(g.Dims), w.rank, reexport)}
		}
		for i, d := range g.Dims {
			if d == -1 || (i > 0 && d == w.fixedOK && d > 0) {
				continue
			}
			what := "a fixed size"
			if i == 0 {
				what = "a fixed batch dimension"
			}
			return &GraphError{Problem: fmt.Sprintf("%s %q declares %s %d in shape %v; %s", kind, w.name, what, d, g.Dims, reexport)}
		}
	}
	if len(got) != len(want) {
		for _, g := range got {
			known := false
			for _, w := range want {
				known = known || w.name == g.Name
			}
			if !known {
				return &GraphError{Problem: fmt.Sprintf("unexpected %s %q; the agent feeds only %d inputs and %d outputs; %s", kind, g.Name, len(wantInputs), len(wantOutputs), reexport)}
			}
		}
	}
	return nil
}

// checkPath refuses what is plainly not a fused graph file before any runtime is started.
func checkPath(path string) error {
	info, err := os.Stat(path)
	switch {
	case err == nil && info.IsDir():
		if split(path) {
			return &GraphError{Path: path, Problem: splitProblem}
		}
		return &GraphError{Path: path, Problem: "is a directory; pass the path of laya.onnx"}
	case err == nil:
		if base := filepath.Base(path); (base == "encoder.onnx" || base == "head.onnx") && split(filepath.Dir(path)) {
			return &GraphError{Path: path, Problem: splitProblem}
		}
		return nil
	case errors.Is(err, fs.ErrNotExist):
		if split(filepath.Dir(path)) {
			return &GraphError{Path: path, Problem: "no such file; " + splitProblem}
		}
		return &GraphError{Path: path, Problem: "no such file"}
	}
	return &GraphError{Path: path, Problem: err.Error()}
}

const splitProblem = "the split encoder.onnx + head.onnx layout is not supported; " + reexport + " to write one fused laya.onnx"

// split reports whether dir holds both halves of the split layout.
func split(dir string) bool {
	for _, name := range []string{"encoder.onnx", "head.onnx"} {
		if info, err := os.Stat(filepath.Join(dir, name)); err != nil || info.IsDir() {
			return false
		}
	}
	return true
}

// outTensor is one output as the runtime returns it.
type outTensor struct {
	shape []int64
	data  []float32
}

// collect checks both outputs against the batch and copies them into slices the caller owns.
func collect(b infer.Batch, logits, act outTensor) (infer.Output, error) {
	for _, c := range []struct {
		name string
		t    outTensor
		cols int
	}{{infer.NameLogits, logits, b.Markers}, {infer.NameActLogits, act, 2}} {
		if len(c.t.shape) != 2 || c.t.shape[0] != int64(b.Rows) || c.t.shape[1] != int64(c.cols) {
			return infer.Output{}, fmt.Errorf("onnx: output %s has shape %v, want [%d %d]", c.name, c.t.shape, b.Rows, c.cols)
		}
	}
	o := infer.Output{
		Logits:    append([]float32(nil), logits.data...),
		ActLogits: append([]float32(nil), act.data...),
	}
	if err := o.Check(b); err != nil {
		return infer.Output{}, err
	}
	return o, nil
}
