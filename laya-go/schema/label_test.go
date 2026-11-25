package schema_test

import (
	"encoding/json"
	"errors"
	"math"
	"math/big"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func TestLabelSpellings(t *testing.T) {
	huge, _ := new(big.Int).SetString("123456789012345678901234567890", 10)
	cases := []struct {
		name            string
		in              any
		str, key, value string
	}{
		{"text", "a b", "a b", "a b", `"a b"`},
		{"true", true, "True", "true", "true"},
		{"false", false, "False", "false", "false"},
		{"int", 1, "1", "1", "1"},
		{"negative int", int8(-7), "-7", "-7", "-7"},
		{"unsigned", uint64(18446744073709551615), "18446744073709551615", "18446744073709551615", "18446744073709551615"},
		{"big int", huge, huge.String(), huge.String(), huge.String()},
		{"float", 1.5, "1.5", "1.5", "1.5"},
		{"whole float", 2.0, "2.0", "2.0", "2.0"},
		{"large float", 1e16, "1e+16", "1e+16", "1e+16"},
		{"small float", 1e-5, "1e-05", "1e-05", "1e-05"},
		{"overflow", math.Inf(1), "inf", "Infinity", "Infinity"},
		{"negative overflow", math.Inf(-1), "-inf", "-Infinity", "-Infinity"},
		{"nan", math.NaN(), "nan", "NaN", "NaN"},
		{"number literal int", json.Number("12"), "12", "12", "12"},
		{"number literal float", json.Number("100E-2"), "1.0", "1.0", "1.0"},
		{"number literal overflow", json.Number("1e400"), "inf", "Infinity", "Infinity"},
		{"number literal big int", json.Number("123456789012345678901234567890"), huge.String(), huge.String(), huge.String()},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			l, err := schema.LabelOf(c.in)
			if err != nil {
				t.Fatal(err)
			}
			if l.String() != c.str || l.Key() != c.key {
				t.Errorf("String, Key = %q, %q; want %q, %q", l.String(), l.Key(), c.str, c.key)
			}
			if b, err := l.MarshalJSON(); err != nil || string(b) != c.value {
				t.Errorf("MarshalJSON = %s, %v; want %s", b, err, c.value)
			}
		})
	}
}

func TestLabelEquality(t *testing.T) {
	one, _ := schema.LabelOf(1)
	if one == schema.Text("1") {
		t.Error("the int 1 and the string \"1\" are one label")
	}
	if one.Key() != schema.Text("1").Key() {
		t.Error("they share a json.dumps key, as in Python")
	}
	again, _ := schema.LabelOf(json.Number("1"))
	if one != again {
		t.Error("the same int is two labels")
	}
	var zero schema.Label
	if zero != schema.Text("") {
		t.Error("the zero Label is not the empty string")
	}
}

func TestLabelOfRefuses(t *testing.T) {
	type thing struct{}
	for name, v := range map[string]any{"nil": nil, "struct": thing{}, "slice": []any{1}, "map": map[string]any{}, "bad number": json.Number("x"),
		"nil big": (*big.Int)(nil)} {
		if _, err := schema.LabelOf(v); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestLabelJSON(t *testing.T) {
	var l schema.Label
	for in, want := range map[string]string{`"a"`: "a", `true`: "True", `3`: "3", `2.50`: "2.5", `Infinity`: "inf"} {
		if err := json.Unmarshal([]byte(in), &l); err != nil && in != `Infinity` {
			t.Fatalf("%s: %v", in, err)
		}
		if in == `Infinity` { // encoding/json rejects the bare token before UnmarshalJSON sees it
			if err := l.UnmarshalJSON([]byte(in)); err != nil {
				t.Fatal(err)
			}
		}
		if l.String() != want {
			t.Errorf("%s -> %q, want %q", in, l.String(), want)
		}
	}
	for _, in := range []string{`null`, `[1]`, `{}`, `x`} {
		if err := l.UnmarshalJSON([]byte(in)); err == nil {
			t.Errorf("%s: accepted", in)
		}
	}
}

func TestChoiceWithTypedLabels(t *testing.T) {
	q, err := schema.Choice("Pick.",
		schema.Option{LabelValue: true, Criterion: "yes"},
		schema.Option{LabelValue: 1.5},
		schema.Option{Label: "1"},
		schema.Option{LabelValue: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got := q.RenderedOptions(); strings.Join(got, "|") != "True: yes|1.5|1|2" {
		t.Errorf("rendered = %q", got)
	}
	out, _ := q.MarshalJSON()
	if want := `{"type":"choice","instructions":"Pick.","criteria":{"true":"yes","1.5":null,"1":null,"2":null}}`; string(out) != want {
		t.Errorf("marshal\n got %s\nwant %s", out, want)
	}
	for name, opts := range map[string][]schema.Option{
		"1 and 1.0":   {{LabelValue: 1}, {LabelValue: 1.0}},
		"True and 1":  {{LabelValue: true}, {LabelValue: 1}},
		"both fields": {{Label: "a", LabelValue: 1}},
		"bad value":   {{LabelValue: []int{1}}},
		"same string": {{Label: "a"}, {Label: "a"}},
	} {
		if _, err := schema.Choice("p", opts...); !errors.Is(err, schema.ErrInvalidQuestion) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
	// "1" and 1 are two labels in Python, so they are two here
	if _, err := schema.Choice("p", schema.Option{Label: "1"}, schema.Option{LabelValue: 1}); err != nil {
		t.Errorf("\"1\" and 1: %v", err)
	}
}

func TestMarshalPythonKeepsBytes(t *testing.T) {
	p := schema.Prediction{Model: schema.Model, Answers: schema.Answers{{ID: "a<b&c", Answer: &schema.ScoreAnswer{
		Legend: []string{"x>y   z"}, Probabilities: schema.Probabilities{{Label: schema.Text("0"), P: 1}}}}}}
	std, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	py, err := schema.MarshalPython(p)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(std), "\\u003c") || strings.Contains(string(py), "\\u003c") || strings.Contains(string(py), "\\u2028") {
		t.Errorf("std %s\npy  %s", std, py)
	}
	if !strings.Contains(string(py), `"a<b&c":{"type":"score"`) || !strings.Contains(string(py), "x>y   z") || strings.HasSuffix(string(py), "\n") {
		t.Errorf("py = %q", py)
	}
	var a, b any
	if json.Unmarshal(std, &a) != nil || json.Unmarshal(py, &b) != nil {
		t.Fatal("not JSON")
	}
	if x, _ := json.Marshal(a); string(x) != func() string { y, _ := json.Marshal(b); return string(y) }() {
		t.Error("the two are different JSON values")
	}
	if _, err := schema.MarshalPython(make(chan int)); err == nil {
		t.Error("a channel marshaled")
	}
}

func TestAnswerValueMarshals(t *testing.T) {
	// A value, not a pointer, keeps its "type": the receiver is not the pointer's alone.
	a := schema.NoulAnswer{Noul: 0.5, Common: schema.Common{Confidence: 0.5, AnswerConfidence: 0.5}}
	out, err := json.Marshal(a)
	if err != nil || !strings.HasPrefix(string(out), `{"type":"noul","noul":0.5`) {
		t.Errorf("value = %s, %v", out, err)
	}
	var nilAnswer *schema.NoulAnswer
	if out, err := json.Marshal(nilAnswer); err != nil || string(out) != "null" {
		t.Errorf("nil pointer = %s, %v", out, err)
	}
	if _, err := json.Marshal(schema.Answers{{ID: "x", Answer: nilAnswer}}); err == nil {
		t.Error("a typed nil answer in Answers marshaled")
	}
	c := schema.ChoiceAnswer{Choice: schema.Text("a")}
	if out, err := json.Marshal(c); err != nil || !strings.HasPrefix(string(out), `{"type":"choice"`) {
		t.Errorf("choice value = %s, %v", out, err)
	}
	s := schema.ScoreAnswer{}
	if out, err := json.Marshal(s); err != nil || !strings.HasPrefix(string(out), `{"type":"score"`) {
		t.Errorf("score value = %s, %v", out, err)
	}
}

// TestLabelNaN: a NaN label is one label. It equals itself, so the answer's probabilities can be
// looked up by it, and it survives a JSON round trip.
func TestLabelNaN(t *testing.T) {
	nan, err := schema.LabelOf(math.NaN())
	if err != nil {
		t.Fatal(err)
	}
	again := mustLabel(t, math.NaN())
	if nan != again || nan.String() != "nan" || nan.Key() != "NaN" {
		t.Errorf("NaN label: equal to another %v, str %q, key %q", nan == again, nan.String(), nan.Key())
	}
	ps := schema.Probabilities{{Label: nan, P: 0.5}, {Label: schema.Text("x"), P: 0.5}}
	if p, ok := ps.Get(nan); !ok || p != 0.5 {
		t.Errorf("Get(NaN) = %v, %v", p, ok)
	}
	data, _ := nan.MarshalJSON()
	var back schema.Label
	if err := back.UnmarshalJSON(data); err != nil || back != nan {
		t.Errorf("round trip of %s = %v, %v", data, back, err)
	}
	if _, err := schema.Choice("pick", schema.Option{LabelValue: math.NaN()}, schema.Option{LabelValue: math.NaN()}); err == nil {
		t.Error("two NaN labels were accepted")
	}
}

// TestLabelOfGoTypes: the Go spellings of a Python scalar that are not the built-in types.
func TestLabelOfGoTypes(t *testing.T) {
	type name string
	type flag bool
	for _, c := range []struct {
		in   any
		want schema.Label
	}{
		{name("billing"), schema.Text("billing")},
		{flag(true), mustLabel(t, true)},
		{flag(false), mustLabel(t, false)},
		{float32(0.1), mustLabel(t, 0.1)},
		{float32(2), mustLabel(t, 2.0)},
	} {
		if got, err := schema.LabelOf(c.in); err != nil || got != c.want {
			t.Errorf("LabelOf(%T %v) = %v (key %q), %v; want key %q", c.in, c.in, got, got.Key(), err, c.want.Key())
		}
	}
	limit, _ := new(big.Int).SetString("-"+strings.Repeat("9", 4300), 10)
	if _, err := schema.LabelOf(limit); err != nil {
		t.Errorf("an int label of 4300 digits: %v", err)
	}
	over, _ := new(big.Int).SetString("-"+strings.Repeat("9", 4301), 10)
	if _, err := schema.LabelOf(over); err == nil {
		t.Error("an int label of 4301 digits was accepted; Python cannot render it")
	}
}

func mustLabel(t *testing.T, v any) schema.Label {
	t.Helper()
	l, err := schema.LabelOf(v)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

// TestMarshalPythonNilPointer: a nil pointer to a type that marshals itself is null, not a call
// through the nil receiver.
func TestMarshalPythonNilPointer(t *testing.T) {
	var q *schema.Questions
	out, err := schema.MarshalPython(q)
	if err != nil || string(out) != "null" {
		t.Errorf("MarshalPython(nil pointer) = %q, %v", out, err)
	}
}
