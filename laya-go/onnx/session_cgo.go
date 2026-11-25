//go:build cgo

package onnx

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	ort "github.com/yalue/onnxruntime_go"
)

// Session runs a fused laya graph on ONNX Runtime's CPU provider. It is safe for concurrent use.
type Session struct {
	mu      sync.RWMutex // read-held by Run, write-held by Close
	closed  bool
	s       *ort.DynamicAdvancedSession
	version string
	runs    atomic.Int64 // model runs started, for tests
}

var _ infer.Session = (*Session)(nil)

var (
	inputNames  = []string{infer.NameInputIDs, infer.NameAttentionMask, infer.NameMarkerPos, infer.NameMarkerMask, infer.NameQType}
	outputNames = []string{infer.NameLogits, infer.NameActLogits}
)

// Open loads a fused laya.onnx, whose .data sidecar must sit beside it, on the CPU provider.
// A graph that is missing, split, lacks its sidecar, or declares other tensors is a *GraphError.
// Open reads the graph twice: once to check it, once to run it, because the binding has no
// session-level query for a graph's inputs and outputs.
func Open(graphPath string, opts ...Option) (*Session, error) {
	o, err := resolve(opts)
	if err != nil {
		return nil, err
	}
	if err := checkPath(graphPath); err != nil {
		return nil, err
	}
	if err := verifyGraph(graphPath, o); err != nil {
		return nil, err
	}
	version, err := acquire(o.library)
	if err != nil {
		return nil, err
	}
	s, err := load(graphPath, o)
	if err != nil {
		return nil, errors.Join(err, release())
	}
	s.version = version
	return s, nil
}

func load(path string, o options) (*Session, error) {
	in, out, err := ort.GetInputOutputInfo(path)
	if err != nil {
		if strings.Contains(err.Error(), "External data") {
			return nil, &GraphError{Path: path, Problem: fmt.Sprintf(
				"its external-data (.data) sidecar is missing; it must sit beside the graph file (%v)", err)}
		}
		return nil, fmt.Errorf("onnx: cannot read %s: %w", path, err)
	}
	if err := checkGraph(infos(in), infos(out)); err != nil {
		var ge *GraphError
		if errors.As(err, &ge) {
			ge.Path = path
		}
		return nil, err
	}
	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("onnx: session options: %w", err)
	}
	defer so.Destroy()
	if err := so.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, fmt.Errorf("onnx: optimization level: %w", err)
	}
	if err := applyThreads(so, o); err != nil {
		return nil, err
	}
	ds, err := ort.NewDynamicAdvancedSession(path, inputNames, outputNames, so)
	if err != nil {
		return nil, fmt.Errorf("onnx: cannot load %s (its .data sidecar must sit beside it): %w", path, err)
	}
	return &Session{s: ds}, nil
}

// threadConfig is the part of ort.SessionOptions that applyThreads sets, so a test can record it.
type threadConfig interface {
	SetIntraOpNumThreads(n int) error
	SetInterOpNumThreads(n int) error
	SetExecutionMode(m ort.ExecutionMode) error
}

// applyThreads sets the thread options. ONNX Runtime ignores the inter-op count in sequential
// execution mode, so a requested inter-op count also selects parallel mode.
func applyThreads(c threadConfig, o options) error {
	if o.intra > 0 {
		if err := c.SetIntraOpNumThreads(o.intra); err != nil {
			return fmt.Errorf("onnx: intra-op threads: %w", err)
		}
	}
	if o.inter > 0 {
		if err := c.SetExecutionMode(ort.ExecutionModeParallel); err != nil {
			return fmt.Errorf("onnx: execution mode: %w", err)
		}
		if err := c.SetInterOpNumThreads(o.inter); err != nil {
			return fmt.Errorf("onnx: inter-op threads: %w", err)
		}
	}
	return nil
}

func infos(in []ort.InputOutputInfo) []tensorInfo {
	out := make([]tensorInfo, len(in))
	for i, t := range in {
		out[i] = tensorInfo{Name: t.Name, Dims: []int64(t.Dimensions), DType: "non-tensor"}
		if t.OrtValueType != ort.ONNXTypeTensor {
			continue
		}
		switch t.DataType {
		case ort.TensorElementDataTypeInt64:
			out[i].DType = "int64"
		case ort.TensorElementDataTypeBool:
			out[i].DType = "bool"
		case ort.TensorElementDataTypeFloat:
			out[i].DType = "float32"
		default:
			out[i].DType = t.DataType.String()
		}
	}
	return out
}

// RuntimeVersion is the version of the ONNX Runtime library in use, such as "1.30.0".
func (s *Session) RuntimeVersion() string { return s.version }

// Close waits for runs in flight, releases the graph and drops its hold on ONNX Runtime, which
// is unloaded when the last session closes, unless the application initialized that runtime
// itself. It is idempotent, and a Session not made by Open closes with no effect.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.s == nil {
		return nil
	}
	s.closed = true
	err := s.s.Destroy()
	s.s = nil
	return errors.Join(err, release())
}

// Run runs one batch. A context that ends mid-run terminates the run and Run returns an error
// wrapping ctx.Err(). Inputs are not modified or retained, and the returned slices are the
// caller's.
func (s *Session) Run(ctx context.Context, b infer.Batch) (infer.Output, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed || s.s == nil {
		return infer.Output{}, infer.ErrClosed
	}
	if err := b.Validate(); err != nil {
		return infer.Output{}, err
	}
	if err := ctx.Err(); err != nil {
		return infer.Output{}, fmt.Errorf("onnx: run not started: %w", err)
	}
	if b.Rows == 0 {
		return infer.Output{}, nil
	}

	var vals []ort.Value
	defer func() {
		for _, v := range vals {
			if v != nil {
				v.Destroy()
			}
		}
	}()
	rows, length, markers := int64(b.Rows), int64(b.Length), int64(b.Markers)
	ids, err := ort.NewTensor(ort.NewShape(rows, length), b.InputIDs)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameInputIDs, err)
	}
	vals = append(vals, ids)
	mask, err := ort.NewTensor(ort.NewShape(rows, length), b.AttentionMask)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameAttentionMask, err)
	}
	vals = append(vals, mask)
	pos, err := ort.NewTensor(ort.NewShape(rows, markers), b.MarkerPos)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameMarkerPos, err)
	}
	vals = append(vals, pos)
	mm, err := ort.NewTensor(ort.NewShape(rows, markers), b.MarkerMask)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameMarkerMask, err)
	}
	vals = append(vals, mm)
	qt, err := ort.NewTensor(ort.NewShape(rows), b.QType)
	if err != nil {
		return infer.Output{}, fmt.Errorf("onnx: %s tensor: %w", infer.NameQType, err)
	}
	vals = append(vals, qt)
	inputs := append([]ort.Value(nil), vals...)

	outputs := []ort.Value{nil, nil}
	defer func() {
		for _, v := range outputs {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	s.runs.Add(1)
	if terminated, err := s.execute(ctx, inputs, outputs); err != nil {
		return infer.Output{}, runError(ctx, terminated, err, b)
	}
	logits, lok := outputs[0].(*ort.Tensor[float32])
	act, aok := outputs[1].(*ort.Tensor[float32])
	if !lok || !aok {
		return infer.Output{}, fmt.Errorf("onnx: outputs are %T and %T, want float32 tensors", outputs[0], outputs[1])
	}
	return collect(b, outTensor{logits.GetShape(), logits.GetData()}, outTensor{act.GetShape(), act.GetData()})
}

// runError explains a failed run. A run the context terminated reports the cancellation; any
// other failure keeps ONNX Runtime's error, and when the context has also ended it wraps both.
func runError(ctx context.Context, terminated bool, err error, b infer.Batch) error {
	cerr := ctx.Err()
	switch {
	case cerr != nil && terminated:
		return fmt.Errorf("onnx: run cancelled: %w", cerr)
	case cerr != nil:
		return fmt.Errorf("onnx: run failed on %d rows of %d tokens: %w (context: %w)", b.Rows, b.Length, err, cerr)
	}
	return fmt.Errorf("onnx: run failed on %d rows of %d tokens: %w", b.Rows, b.Length, err)
}

// execute runs the graph, terminating it if ctx ends first, and reports whether it did. The
// watcher is always joined before the run options are destroyed.
func (s *Session) execute(ctx context.Context, inputs, outputs []ort.Value) (terminated bool, err error) {
	done := ctx.Done()
	if done == nil {
		return false, s.s.Run(inputs, outputs)
	}
	ro, err := ort.NewRunOptions()
	if err != nil {
		return false, fmt.Errorf("run options: %w", err)
	}
	defer ro.Destroy()
	stop := make(chan struct{})
	joined := make(chan struct{})
	var fired atomic.Bool
	go func() {
		defer close(joined)
		select {
		case <-done:
			fired.Store(true)
			ro.Terminate()
		case <-stop:
		}
	}()
	defer func() {
		close(stop)
		// The join must precede ro.Destroy (deferred above, so it runs after this): the watcher
		// may still be inside ro.Terminate, and destroying the options under it is a
		// use-after-free in native code.
		<-joined
		terminated = fired.Load()
	}()
	err = s.s.RunWithOptions(inputs, outputs, ro)
	return false, err
}
