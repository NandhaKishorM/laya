// Package goldens turns the question and state values recorded in the golden files into the
// values the library takes, for tests. Like internal/fixtures it is not part of the library.
package goldens

import (
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// Value converts a parsed golden value into one the library accepts as a state or a criterion:
// an Object becomes a lang.Object, which keeps its key order, and numbers stay json.Number.
func Value(v any) any {
	switch x := v.(type) {
	case fixtures.Object:
		out := make(lang.Object, len(x))
		for i, m := range x {
			out[i] = lang.Member{Key: m.Key, Value: Value(m.Value)}
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = Value(item)
		}
		return out
	}
	return v
}

// Question builds a question from the internal dict form the goldens record ({"t", "ins",
// "crit", "labels"}) with the optional option order applied. The typed constructors, not the wire
// reader, build it, so the goldens exercise both paths.
func Question(tb testing.TB, internal fixtures.Object, optionOrder []int32) schema.Question {
	tb.Helper()
	kind := fixtures.Str(tb, fixtures.Field(tb, internal, "t"))
	ins := fixtures.Str(tb, fixtures.Field(tb, internal, "ins"))
	crit, _ := internal.Get("crit")
	var (
		q   schema.Question
		err error
	)
	switch kind {
	case "choice":
		var opts []schema.Option
		for _, m := range fixtures.Obj(tb, crit) {
			opts = append(opts, schema.Option{Label: m.Key, Criterion: Value(m.Value)})
		}
		q, err = schema.Choice(ins, opts...)
	case "score":
		var levels []any
		for _, level := range fixtures.List(tb, crit) {
			levels = append(levels, Value(level))
		}
		q, err = schema.Score(ins, levels...)
	case "noul":
		var spec schema.NoulSpec
		if obj, ok := crit.(fixtures.Object); ok {
			spec.True, _ = obj.Get("true")
			spec.False, _ = obj.Get("false")
			spec.True, spec.False = Value(spec.True), Value(spec.False)
		}
		if labels, ok := internal.Get("labels"); ok && labels != nil {
			l := fixtures.Obj(tb, labels)
			spec.FalseLabel = fixtures.Str(tb, l.Must("false"))
			spec.TrueLabel = fixtures.Str(tb, l.Must("true"))
		}
		q, err = schema.NoulWith(ins, spec)
	default:
		tb.Fatalf("golden question type %q", kind)
	}
	if err != nil {
		tb.Fatalf("build the %s question: %v", kind, err)
	}
	if optionOrder != nil {
		order := make([]int, len(optionOrder))
		for i, v := range optionOrder {
			order[i] = int(v)
		}
		if q, err = q.WithOptionOrder(order); err != nil {
			tb.Fatalf("option order %v: %v", order, err)
		}
	}
	return q
}
