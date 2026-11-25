package infer

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// ErrClosed is returned by a Session that has been closed.
var ErrClosed = errors.New("infer: session is closed")

// The graph's tensor names, as ShapeError.Name carries them and as a native session binds them.
const (
	NameInputIDs      = "input_ids"
	NameAttentionMask = "attention_mask"
	NameMarkerPos     = "marker_pos"
	NameMarkerMask    = "marker_mask"
	NameQType         = "qtype"
	NameLogits        = "logits"
	NameActLogits     = "act_logits"
)

// Batch is the graph's five inputs, each flattened row-major, as onnxruntime tensors take them.
type Batch struct {
	InputIDs              []int64 // [Rows*Length]; padding is the pad id
	AttentionMask         []int64 // [Rows*Length]; 1 for real tokens
	MarkerPos             []int64 // [Rows*Markers]
	MarkerMask            []bool  // [Rows*Markers]; true for real markers
	QType                 []int64 // [Rows]; the question type code
	Rows, Length, Markers int
}

// Output is the graph's two outputs, row-major: Logits is [Rows*Markers], ActLogits [Rows*2].
type Output struct {
	Logits, ActLogits []float32
}

// Session runs one collated batch. Implementations must be safe for concurrent Run. The agent
// never calls Close concurrently with Run.
type Session interface {
	// Run returns the logits for b. It should return ctx.Err() if ctx ends before the run does.
	// Run must not modify b's slices or retain them after it returns. The returned Output's
	// slices belong to the caller, and the session must not reuse them.
	Run(ctx context.Context, b Batch) (Output, error)
	// Close releases the runtime. It is idempotent; Run after Close returns ErrClosed.
	Close() error
}

// ShapeError is returned when a slice's length does not match the batch it belongs to.
type ShapeError struct {
	// Name is the tensor: NameInputIDs, NameLogits and so on.
	Name string
	// Got is the slice's length, Want the length the batch's dimensions call for.
	Got, Want int
}

// Error names the tensor and both lengths.
func (e *ShapeError) Error() string {
	return fmt.Sprintf("infer: %s has %d values, want %d", e.Name, e.Got, e.Want)
}

// Validate checks, in order, that:
//   - no dimension is negative;
//   - a batch with rows has Length of at least 1 and Markers of at least 2, since the exported
//     head takes topk(2);
//   - Rows*Length and Rows*Markers do not overflow an int (Markers of at least 2 keeps Rows*2 safe too);
//   - every slice's length matches Rows, Length and Markers, else a *ShapeError;
//   - every MarkerPos lies in [0, Length);
//   - no token id is negative, and every QType is 0, 1 or 2 (the three question types).
//
// All but the ShapeError are plain errors. Token ids at or above the vocabulary size and mask
// values are not checked: a backend passes them to the graph (a nonzero mask counts as 1). The
// negative ids and types are refused here because ONNX Runtime wraps them to the last rows of the
// embedding tables and answers from the wrong rows without an error.
func (b Batch) Validate() error {
	if b.Rows < 0 || b.Length < 0 || b.Markers < 0 {
		return fmt.Errorf("infer: batch has negative dimensions %d x %d x %d", b.Rows, b.Length, b.Markers)
	}
	if b.Rows > 0 {
		if b.Length == 0 {
			return fmt.Errorf("infer: batch has %d rows of 0 tokens", b.Rows)
		}
		if b.Markers < 2 {
			return fmt.Errorf("infer: batch has %d marker slots, the graph needs at least 2", b.Markers)
		}
		if b.Length > math.MaxInt/b.Rows || b.Markers > math.MaxInt/b.Rows {
			return fmt.Errorf("infer: batch dimensions %d x %d x %d overflow", b.Rows, b.Length, b.Markers)
		}
	}
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{NameInputIDs, len(b.InputIDs), b.Rows * b.Length},
		{NameAttentionMask, len(b.AttentionMask), b.Rows * b.Length},
		{NameMarkerPos, len(b.MarkerPos), b.Rows * b.Markers},
		{NameMarkerMask, len(b.MarkerMask), b.Rows * b.Markers},
		{NameQType, len(b.QType), b.Rows},
	} {
		if c.got != c.want {
			return &ShapeError{Name: c.name, Got: c.got, Want: c.want}
		}
	}
	for i, p := range b.MarkerPos {
		if p < 0 || p >= int64(b.Length) {
			return fmt.Errorf("infer: marker_pos[%d] is %d, outside [0, %d)", i, p, b.Length)
		}
	}
	for i, id := range b.InputIDs {
		if id < 0 {
			return fmt.Errorf("infer: input_ids row %d column %d is %d, token ids must not be negative", i/b.Length, i%b.Length, id)
		}
	}
	for i, q := range b.QType {
		if q < 0 || q > 2 {
			return fmt.Errorf("infer: qtype row %d is %d, want 0, 1 or 2", i, q)
		}
	}
	return nil
}

// Check reports a *ShapeError when o does not hold Rows*Markers logits and Rows*2 action logits
// for b.
func (o Output) Check(b Batch) error {
	if len(o.Logits) != b.Rows*b.Markers {
		return &ShapeError{Name: NameLogits, Got: len(o.Logits), Want: b.Rows * b.Markers}
	}
	if len(o.ActLogits) != b.Rows*2 {
		return &ShapeError{Name: NameActLogits, Got: len(o.ActLogits), Want: b.Rows * 2}
	}
	return nil
}
