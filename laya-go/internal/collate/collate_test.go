package collate

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func TestCollate(t *testing.T) {
	rows := []Row{
		{IDs: []int32{1, 5, 6, 2}, Markers: []int{1, 2, 3}, Type: schema.TypeChoice},
		{IDs: []int32{1, 7}, Markers: []int{1}, Type: schema.TypeNoul},
	}
	b, err := Collate(rows, 9)
	if err != nil {
		t.Fatal(err)
	}
	want := Batch{
		InputIDs:      []int64{1, 5, 6, 2, 1, 7, 9, 9},
		AttentionMask: []int64{1, 1, 1, 1, 1, 1, 0, 0},
		MarkerPos:     []int64{1, 2, 3, 1, 0, 0},
		MarkerMask:    []bool{true, true, true, true, false, false},
		QType:         []int64{0, 2},
		Rows:          2, Length: 4, Markers: 3,
	}
	if !reflect.DeepEqual(b, want) {
		t.Errorf("batch =\n%+v\nwant\n%+v", b, want)
	}
}

func TestCollateMarkerFloor(t *testing.T) {
	b, err := Collate([]Row{{IDs: []int32{1, 4, 2}, Markers: []int{1}, Type: schema.TypeScore}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if b.Markers != MinMarkerSlots || !reflect.DeepEqual(b.MarkerMask, []bool{true, false}) ||
		!reflect.DeepEqual(b.MarkerPos, []int64{1, 0}) {
		t.Errorf("a one-option batch is not padded to %d slots: %+v", MinMarkerSlots, b)
	}
}

func TestCollateRefuses(t *testing.T) {
	ok := Row{IDs: []int32{1, 2}, Markers: []int{0}, Type: schema.TypeChoice}
	cases := map[string]struct {
		rows []Row
		want string
	}{
		"no rows":               {nil, ErrEmpty.Error()},
		"no ids":                {[]Row{{Type: schema.TypeChoice}}, "no token ids"},
		"unknown type":          {[]Row{{IDs: []int32{1}, Type: "mood"}}, "question type"},
		"marker past the ids":   {[]Row{{IDs: []int32{1, 2}, Markers: []int{2}, Type: schema.TypeChoice}}, "outside"},
		"negative marker":       {[]Row{{IDs: []int32{1, 2}, Markers: []int{-1}, Type: schema.TypeChoice}}, "outside"},
		"second row is the bad": {[]Row{ok, {Type: schema.TypeNoul}}, "row 1"},
	}
	for name, c := range cases {
		_, err := Collate(c.rows, 0)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", name, err, c.want)
		}
	}
	if _, err := Collate(nil, 0); !errors.Is(err, ErrEmpty) {
		t.Errorf("an empty batch must wrap ErrEmpty, got %v", err)
	}
}
