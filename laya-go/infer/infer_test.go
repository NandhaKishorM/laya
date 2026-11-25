package infer_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/collate"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func batch(rows, length, markers int) infer.Batch {
	return infer.Batch{
		InputIDs: make([]int64, rows*length), AttentionMask: make([]int64, rows*length),
		MarkerPos: make([]int64, rows*markers), MarkerMask: make([]bool, rows*markers),
		QType: make([]int64, rows), Rows: rows, Length: length, Markers: markers,
	}
}

func TestBatchValidate(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*infer.Batch)
		field  string // the ShapeError's tensor, "" for none, "other" for a non-ShapeError
	}{
		{"consistent", func(*infer.Batch) {}, ""},
		{"empty", func(b *infer.Batch) { *b = infer.Batch{} }, ""},
		{"short input_ids", func(b *infer.Batch) { b.InputIDs = b.InputIDs[1:] }, "input_ids"},
		{"long attention_mask", func(b *infer.Batch) { b.AttentionMask = append(b.AttentionMask, 0) }, "attention_mask"},
		{"short marker_pos", func(b *infer.Batch) { b.MarkerPos = nil }, "marker_pos"},
		{"short marker_mask", func(b *infer.Batch) { b.MarkerMask = b.MarkerMask[:1] }, "marker_mask"},
		{"short qtype", func(b *infer.Batch) { b.QType = nil }, "qtype"},
		{"one marker slot", func(b *infer.Batch) { *b = batch(3, 4, 1) }, "other"},
		{"no marker slots", func(b *infer.Batch) { *b = batch(3, 4, 0) }, "other"},
		{"negative rows", func(b *infer.Batch) { b.Rows = -1 }, "other"},
		{"negative length", func(b *infer.Batch) { b.Length = -1 }, "other"},
		{"negative markers", func(b *infer.Batch) { b.Markers = -2 }, "other"},
		{"rows of no tokens", func(b *infer.Batch) { *b = batch(3, 0, 2) }, "other"},
		{"marker position below zero", func(b *infer.Batch) { b.MarkerPos[2] = -1 }, "other"},
		{"marker position at the length", func(b *infer.Batch) { b.MarkerPos[5] = 4 }, "other"},
		{"negative token id", func(b *infer.Batch) { b.InputIDs[5] = -1 }, "other"},
		{"qtype below zero", func(b *infer.Batch) { b.QType[1] = -1 }, "other"},
		{"qtype above two", func(b *infer.Batch) { b.QType[2] = 3 }, "other"},
		{"qtype two", func(b *infer.Batch) { b.QType[2] = 2 }, ""},
		{"marker position in range", func(b *infer.Batch) { b.MarkerPos[5] = 3 }, ""},
		// 2 * 2^62 wraps to the most negative int, which no slice length equals.
		{"rows x length overflows", func(b *infer.Batch) { *b = infer.Batch{Rows: 2, Length: math.MaxInt/2 + 1, Markers: 2} }, "overflow"},
		{"rows x markers overflows", func(b *infer.Batch) { *b = infer.Batch{Rows: 4, Length: 1, Markers: math.MaxInt/2 + 1} }, "overflow"},
		{"rows x length wraps to zero", func(b *infer.Batch) { *b = infer.Batch{Rows: math.MaxInt/2 + 1, Length: 2, Markers: 2} }, "overflow"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b := batch(3, 4, 2)
			c.mutate(&b)
			err := b.Validate()
			var se *infer.ShapeError
			switch {
			case c.field == "" && err != nil:
				t.Fatalf("Validate = %v, want nil", err)
			case c.field == "overflow" && (err == nil || errors.As(err, &se) || !strings.Contains(err.Error(), "overflow")):
				t.Fatalf("Validate = %v, want an overflow error", err)
			case c.field == "other" && (err == nil || errors.As(err, &se)):
				t.Fatalf("Validate = %v, want a plain error", err)
			case c.field != "" && c.field != "other" && c.field != "overflow" && (!errors.As(err, &se) || se.Name != c.field):
				t.Fatalf("Validate = %v, want a ShapeError for %s", err, c.field)
			}
		})
	}
}

func TestOutputCheck(t *testing.T) {
	b := batch(3, 4, 5)
	good := infer.Output{Logits: make([]float32, 15), ActLogits: make([]float32, 6)}
	if err := good.Check(b); err != nil {
		t.Fatalf("Check = %v", err)
	}
	for name, o := range map[string]infer.Output{
		"logits":     {Logits: make([]float32, 14), ActLogits: make([]float32, 6)},
		"act_logits": {Logits: make([]float32, 15), ActLogits: make([]float32, 5)},
	} {
		var se *infer.ShapeError
		if err := o.Check(b); !errors.As(err, &se) || se.Name != name {
			t.Errorf("%s: Check = %v", name, err)
		} else if se.Got == se.Want || se.Error() == "" {
			t.Errorf("%s: %+v", name, se)
		}
	}
}

func TestMessagesAndNamesCarryThePackage(t *testing.T) {
	for _, err := range []error{
		infer.ErrClosed, &infer.ShapeError{Name: infer.NameLogits, Got: 1, Want: 2},
		infer.Batch{Rows: -1}.Validate(), batch(1, 1, 1).Validate(),
	} {
		if !strings.HasPrefix(err.Error(), "infer: ") {
			t.Errorf("%q lacks the package prefix", err)
		}
	}
	for got, want := range map[string]string{
		infer.NameInputIDs: "input_ids", infer.NameAttentionMask: "attention_mask", infer.NameMarkerPos: "marker_pos",
		infer.NameMarkerMask: "marker_mask", infer.NameQType: "qtype", infer.NameLogits: "logits", infer.NameActLogits: "act_logits",
	} {
		if got != want {
			t.Errorf("tensor name %q, want %q", got, want)
		}
	}
}

// TestCollateBatchIsInferBatch pins the alias: a Batch the collator builds is a valid infer.Batch.
func TestCollateBatchIsInferBatch(t *testing.T) {
	b, err := collate.Collate([]collate.Row{
		{IDs: []int32{1, 2, 3}, Markers: []int{1}, Type: schema.TypeChoice},
		{IDs: []int32{4, 5}, Markers: []int{0, 1}, Type: schema.TypeNoul},
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	var ib infer.Batch = b
	if err := ib.Validate(); err != nil || ib.Rows != 2 || ib.Length != 3 || ib.Markers != 2 {
		t.Fatalf("Validate = %v on %+v", err, ib)
	}
}

// TestValidateNamesTheEmptyRow pins that rows of no tokens are refused for that reason, not as a
// side effect of the marker range check.
func TestValidateNamesTheEmptyRow(t *testing.T) {
	err := batch(3, 0, 2).Validate()
	if err == nil || !strings.Contains(err.Error(), "0 tokens") {
		t.Fatalf("Validate = %v", err)
	}
}

// TestValidateNamesTheBadRowAndColumn pins that the error says where the value is.
func TestValidateNamesTheBadRowAndColumn(t *testing.T) {
	b := batch(3, 4, 2)
	b.InputIDs[9] = -1 // row 2, column 1
	err := b.Validate()
	if err == nil || !strings.Contains(err.Error(), "row 2 column 1") {
		t.Fatalf("Validate = %v", err)
	}
	b = batch(3, 4, 2)
	b.QType[1] = -1
	err = b.Validate()
	if err == nil || !strings.Contains(err.Error(), "qtype row 1") {
		t.Fatalf("Validate = %v", err)
	}
}

// TestNegativeDimensionsAreRefusedWithoutRows pins that the sign check does not depend on Rows.
func TestNegativeDimensionsAreRefusedWithoutRows(t *testing.T) {
	for name, b := range map[string]infer.Batch{
		"markers": {Markers: -1},
		"length":  {Length: -1},
		"both":    {Length: 4, Markers: -3},
		"rows":    {Rows: -1, Length: 4, Markers: 2},
	} {
		err := b.Validate()
		var se *infer.ShapeError
		if err == nil || errors.As(err, &se) || !strings.Contains(err.Error(), "negative") {
			t.Errorf("%s: Validate = %v, want the negative-dimensions error", name, err)
		}
	}
}

// TestOverflowGuardBoundary pins the exact edge: the largest dimension whose product with Rows
// does not overflow passes the guard, and is then refused by the slice-length check.
func TestOverflowGuardBoundary(t *testing.T) {
	cases := []struct {
		name string
		b    infer.Batch
		want string // the ShapeError's tensor
	}{
		{"length at the edge, 1 row", infer.Batch{Rows: 1, Length: math.MaxInt, Markers: 2}, infer.NameInputIDs},
		{"length at the edge, 2 rows", infer.Batch{Rows: 2, Length: math.MaxInt / 2, Markers: 2}, infer.NameInputIDs},
		{"markers at the edge, 1 row", infer.Batch{Rows: 1, Length: 1, Markers: math.MaxInt,
			InputIDs: make([]int64, 1), AttentionMask: make([]int64, 1)}, infer.NameMarkerPos},
		{"markers at the edge, 2 rows", infer.Batch{Rows: 2, Length: 1, Markers: math.MaxInt / 2,
			InputIDs: make([]int64, 2), AttentionMask: make([]int64, 2)}, infer.NameMarkerPos},
	}
	for _, c := range cases {
		var se *infer.ShapeError
		if err := c.b.Validate(); !errors.As(err, &se) || se.Name != c.want {
			t.Errorf("%s: Validate = %v, want a ShapeError for %s", c.name, err, c.want)
		}
	}
}
