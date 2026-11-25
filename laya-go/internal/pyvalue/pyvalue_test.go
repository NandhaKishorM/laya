package pyvalue

import (
	"encoding/json"
	"errors"
	"math"
	"math/big"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/lang"
)

func TestParse(t *testing.T) {
	cases := []struct {
		name, in, repr string
	}{
		{"scalars", `[1, 1.0, -0, 1e2, "é", true, null]`, `[1, 1.0, 0, 100.0, 'é', True, None]`},
		{"repeated key keeps its first place and last value", `{"a":1,"b":2,"a":3}`, `{'a': 3, 'b': 2}`},
		{"nested repeats", `{"o":{"x":1,"x":2}}`, `{'o': {'x': 2}}`},
		{"huge int", `123456789012345678901234567890`, `123456789012345678901234567890`},
		{"float spelling", `[1e16, 1e-5, 0.1, 2.5e+300]`, `[1e+16, 1e-05, 0.1, 2.5e+300]`},
		{"overflow is inf", `1e999`, `inf`},
		{"white space", " \n[ ]\t", `[]`},
		{"python constants", `[NaN, Infinity, -Infinity]`, `[nan, inf, -inf]`},
		{"surrogate pair", `"\ud83d\ude00 \u00e9"`, `'😀 é'`},
		{"escapes", `"a\"b\\c\/d\n"`, `'a"b\\c/d\n'`},
		{"4300 digits is fine", strings.Repeat("9", 4300), strings.Repeat("9", 4300)},
		{"the sign is not a digit", "-" + strings.Repeat("9", 4300), "-" + strings.Repeat("9", 4300)},
		{"a float has no digit limit", strings.Repeat("1", 4301) + ".0", "inf"},
		{"U+007F is not a control character", "\"\x7f\"", `'\x7f'`},
		{"nested to the limit", strings.Repeat("[", MaxDepth+1) + strings.Repeat("]", MaxDepth+1),
			strings.Repeat("[", MaxDepth+1) + strings.Repeat("]", MaxDepth+1)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v, err := Parse([]byte(c.in))
			if err != nil {
				t.Fatal(err)
			}
			if got := Repr(v); got != c.repr {
				t.Errorf("repr = %s, want %s", got, c.repr)
			}
		})
	}
	for name, in := range map[string]string{
		"empty": ``, "trailing": `1 2`, "bad UTF-8": "\"\xff\"", "leading zero": `01`, "trailing comma": `[1,]`,
		"lone high surrogate": `"\ud800"`, "lone low surrogate": `"\udc00x"`, "high then text": `"\ud800a"`,
		"high then non-low": `"\ud800\u0041"`, "raw control": "\"a\tb\"", "bad escape": `"\q"`, "short unicode": `"\u12"`,
		"bare minus": `-`, "dangling dot": `1.`, "single quotes": `'a'`, "int over 4300 digits": strings.Repeat("9", 4301),
		"negative int over 4300 digits": "-" + strings.Repeat("9", 4301), "raw U+001F": "\"\x1f\"",
		"deep": strings.Repeat("[", MaxDepth+2) + strings.Repeat("]", MaxDepth+2), "unterminated": `{"a":[`,
	} {
		if _, err := Parse([]byte(in)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestNumber(t *testing.T) {
	cases := map[string]any{"7": int64(7), "-0": int64(0), "2.0": 2.0, "1E2": 100.0, "9223372036854775808": mustBig("9223372036854775808")}
	for in, want := range cases {
		got, err := Number(json.Number(in))
		if err != nil {
			t.Fatal(err)
		}
		if b, ok := want.(*big.Int); ok {
			if gb, ok := got.(*big.Int); !ok || gb.Cmp(b) != 0 {
				t.Errorf("Number(%s) = %v", in, got)
			}
		} else if got != want {
			t.Errorf("Number(%s) = %#v, want %#v", in, got, want)
		}
	}
	if f, _ := Number("1e999"); !math.IsInf(f.(float64), 1) {
		t.Errorf("out of range = %v", f)
	}
	if _, err := Number("abc"); err == nil {
		t.Error("abc is not a number")
	}
}

func mustBig(s string) *big.Int { b, _ := new(big.Int).SetString(s, 10); return b }

func TestNormalizeAndDumps(t *testing.T) {
	type named string
	type thing struct{}
	n := 5
	var nilPtr *int
	cases := []struct {
		name string
		in   any
		want string
	}{
		{"pointer", &n, "5"}, {"nil pointer", nilPtr, "null"}, {"named string", named("a"), `"a"`},
		{"typed map sorted", map[string]int{"b": 1, "a": 2}, `{"a": 2, "b": 1}`},
		{"array", [2]string{"a", "b"}, `["a", "b"]`}, {"nil map", map[string]any(nil), "{}"},
		{"unsigned", uint8(200), "200"}, {"lang.Object", lang.Object{{Key: "k", Value: lang.Object{{Key: "j", Value: 1}}}}, `{"k": {"j": 1}}`},
		{"pyjson.Object", pyjson.Object{{Key: 1, Value: true}}, `{"1": true}`},
		{"json.Number int", json.Number("12"), "12"}, {"json.Number float", json.Number("12.0"), "12.0"},
		{"NaN", math.NaN(), "NaN"}, {"big", mustBig("12345678901234567890"), "12345678901234567890"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got, err := Dumps(c.in); err != nil || got != c.want {
				t.Errorf("Dumps = %q, %v; want %q", got, err, c.want)
			}
		})
	}
	for name, v := range map[string]any{"struct": thing{}, "chan": make(chan int), "bytes": []byte("x"), "int keys": map[int]int{1: 1}, "func": func() {}} {
		if _, err := Dumps(v); !errors.Is(err, ErrUnsupported) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
	loop := []any{nil}
	loop[0] = loop
	if _, err := Normalize(loop); !errors.Is(err, ErrUnsupported) {
		t.Errorf("a cycle: %v", err)
	}
	if _, err := Dumps("\xff"); !errors.Is(err, ErrUnsupported) {
		t.Errorf("invalid UTF-8: %v", err)
	}
}

func TestReprStrTypeNameKeyOf(t *testing.T) {
	v, _ := Parse([]byte(`["it's", 1, 1.0, true, null, {"k": [2.5]}, "a\"b'c"]`))
	if got := Repr(v); got != `["it's", 1, 1.0, True, None, {'k': [2.5]}, 'a"b\'c']` {
		t.Errorf("Repr = %s", got)
	}
	items := v.([]any)
	for i, want := range []string{"it's", "1", "1.0", "True", "None"} {
		if got := Str(items[i]); got != want {
			t.Errorf("Str(%d) = %q, want %q", i, got, want)
		}
	}
	for i, want := range []string{"str", "int", "float", "bool", "NoneType", "dict"} {
		if got := TypeName(items[i]); got != want {
			t.Errorf("TypeName(%d) = %s, want %s", i, got, want)
		}
	}
	if TypeName([]any{}) != "list" {
		t.Error("TypeName(list)")
	}
	same := [][2]any{{json.Number("1"), json.Number("1.0")}, {json.Number("1"), true}, {json.Number("0"), json.Number("-0.0")},
		{json.Number("0"), false}, {json.Number("3"), json.Number("3e0")}}
	for _, p := range same {
		if KeyOf(p[0]) != KeyOf(p[1]) {
			t.Errorf("%v and %v are one Python key", p[0], p[1])
		}
	}
	distinct := [][2]any{{json.Number("1"), "1"}, {json.Number("1"), json.Number("1.5")}, {"a", "b"},
		{json.Number("9007199254740993"), json.Number("9007199254740992.0")},
		{json.Number("1000000000000000000000000000000"), json.Number("1e30")}}
	for _, p := range distinct {
		if KeyOf(p[0]) == KeyOf(p[1]) {
			t.Errorf("%v and %v are two Python keys", p[0], p[1])
		}
	}
	for in, want := range map[json.Number]string{"1e999": "inf", "-1e999": "-inf", "1e16": "1e+16", "100": "100"} {
		if got := Str(in); got != want {
			t.Errorf("Str(%s) = %s, want %s", in, got, want)
		}
	}
}

// TestNumberHoldsTheGrammar: a json.Number a caller builds is held to the spellings json.loads
// reads, and to the same digit limit, which counts digits and not the sign.
func TestNumberHoldsTheGrammar(t *testing.T) {
	for _, s := range []string{"", "-", "+1", ".5", "5.", "1_0.0", "0x1.8p1", "01", "1e", "1 ", "NaN", "Infinity"} {
		if v, err := Number(json.Number(s)); err == nil {
			t.Errorf("Number(%q) = %v, want an error", s, v)
		}
	}
	if _, err := Number(json.Number("-" + strings.Repeat("9", 4300))); err != nil {
		t.Errorf("a negative int of 4300 digits: %v", err)
	}
	if _, err := Number(json.Number("-" + strings.Repeat("9", 4301))); err == nil {
		t.Error("a negative int of 4301 digits was accepted")
	}
	if v, err := Number(json.Number(strings.Repeat("9", 4301) + "e0")); err != nil || !math.IsInf(v.(float64), 1) {
		t.Errorf("a float of 4301 digits = %v, %v", v, err)
	}
}
