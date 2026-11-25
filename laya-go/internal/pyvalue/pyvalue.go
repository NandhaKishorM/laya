// Package pyvalue reads, writes and describes the values laya's Python code handles as dicts,
// lists, strs, ints and floats: the state of a request, a question's criteria, a definition in
// the wire form.
//
// It sits above pyjson (which writes the text) and adds what a Go caller needs to reach it: a
// decoder that keeps object order and number spelling as json.loads does, a normalizer that maps
// the Go shapes a caller can hold (maps, slices, pointers, json.Number, lang.Object) onto what
// pyjson accepts, and the repr and str of a decoded value, which laya puts in its error messages.
package pyvalue

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/lang"
)

// MaxDepth bounds the nesting of a decoded or normalized value, so a hostile document or a
// self-referential pointer ends in an error. Python has a bound too, its recursion limit, but a
// deeper one: json.loads reads about 1,500 levels at the default limit.
const MaxDepth = 512

// ErrUnsupported is wrapped when a Go value has no Python equivalent json.dumps would write.
var ErrUnsupported = errors.New("value has no JSON equivalent in Python")

// MaxIntDigits is CPython's limit on the digits of an int literal (sys.int_info
// .default_max_str_digits): json.loads raises ValueError beyond it.
const MaxIntDigits = 4300

// Parse decodes one JSON document as json.loads does: an object is a pyjson.Object with string
// keys in document order (a repeated key keeps its first position and takes its last value),
// numbers are json.Number so that 2 stays an int and 2.0 a float, and anything after the value
// is an error. The literals NaN, Infinity and -Infinity are accepted, as Python accepts them, and
// become float64. Invalid UTF-8, a lone surrogate escape, a raw control character in a string and
// an int literal of more than 4300 digits are refused.
func Parse(data []byte) (any, error) {
	if !utf8.Valid(data) {
		return nil, errors.New("JSON text is not valid UTF-8")
	}
	p := &parser{s: string(data)}
	p.space()
	v, err := p.value(0)
	if err != nil {
		return nil, err
	}
	p.space()
	if p.i < len(p.s) {
		return nil, p.fail("data after the JSON value")
	}
	return v, nil
}

type parser struct {
	s string
	i int
}

func (p *parser) fail(msg string) error { return fmt.Errorf("JSON: %s at offset %d", msg, p.i) }

func (p *parser) space() {
	for p.i < len(p.s) && (p.s[p.i] == ' ' || p.s[p.i] == '\t' || p.s[p.i] == '\n' || p.s[p.i] == '\r') {
		p.i++
	}
}

func (p *parser) literal(word string) bool {
	if strings.HasPrefix(p.s[p.i:], word) {
		p.i += len(word)
		return true
	}
	return false
}

func (p *parser) value(depth int) (any, error) {
	if depth > MaxDepth {
		return nil, fmt.Errorf("JSON nested deeper than %d levels", MaxDepth)
	}
	if p.i >= len(p.s) {
		return nil, p.fail("unexpected end of text")
	}
	switch c := p.s[p.i]; {
	case c == '{':
		p.i++
		return p.object(depth + 1)
	case c == '[':
		p.i++
		return p.array(depth + 1)
	case c == '"':
		p.i++
		return p.str()
	case p.literal("true"):
		return true, nil
	case p.literal("false"):
		return false, nil
	case p.literal("null"):
		return nil, nil
	case p.literal("NaN"):
		return math.NaN(), nil
	case p.literal("Infinity"):
		return math.Inf(1), nil
	case p.literal("-Infinity"):
		return math.Inf(-1), nil
	case c == '-' || (c >= '0' && c <= '9'):
		return p.number()
	}
	return nil, p.fail("unexpected character")
}

func (p *parser) array(depth int) (any, error) {
	list := []any{}
	p.space()
	if p.i < len(p.s) && p.s[p.i] == ']' {
		p.i++
		return list, nil
	}
	for {
		p.space()
		v, err := p.value(depth)
		if err != nil {
			return nil, err
		}
		list = append(list, v)
		p.space()
		if p.i >= len(p.s) {
			return nil, p.fail("unterminated array")
		}
		switch p.s[p.i] {
		case ',':
			p.i++
		case ']':
			p.i++
			return list, nil
		default:
			return nil, p.fail("expected , or ]")
		}
	}
}

func (p *parser) object(depth int) (any, error) {
	members := pyjson.Object{}
	at := map[string]int{}
	p.space()
	if p.i < len(p.s) && p.s[p.i] == '}' {
		p.i++
		return members, nil
	}
	for {
		p.space()
		if p.i >= len(p.s) || p.s[p.i] != '"' {
			return nil, p.fail("expected a string key")
		}
		p.i++
		key, err := p.str()
		if err != nil {
			return nil, err
		}
		p.space()
		if p.i >= len(p.s) || p.s[p.i] != ':' {
			return nil, p.fail("expected :")
		}
		p.i++
		p.space()
		v, err := p.value(depth)
		if err != nil {
			return nil, err
		}
		if j, seen := at[key.(string)]; seen {
			members[j].Value = v // Python's dict: first position, last value
		} else {
			at[key.(string)] = len(members)
			members = append(members, pyjson.Member{Key: key, Value: v})
		}
		p.space()
		if p.i >= len(p.s) {
			return nil, p.fail("unterminated object")
		}
		switch p.s[p.i] {
		case ',':
			p.i++
		case '}':
			p.i++
			return members, nil
		default:
			return nil, p.fail("expected , or }")
		}
	}
}

// str reads a string whose opening quote was consumed.
func (p *parser) str() (any, error) {
	var b strings.Builder
	for p.i < len(p.s) {
		c := p.s[p.i]
		switch {
		case c == '"':
			p.i++
			return b.String(), nil
		case c < 0x20:
			return nil, p.fail("control character in a string")
		case c != '\\':
			b.WriteByte(c)
			p.i++
			continue
		}
		p.i++
		if p.i >= len(p.s) {
			break
		}
		esc := p.s[p.i]
		p.i++
		switch esc {
		case '"', '\\', '/':
			b.WriteByte(esc)
		case 'b':
			b.WriteByte('\b')
		case 'f':
			b.WriteByte('\f')
		case 'n':
			b.WriteByte('\n')
		case 'r':
			b.WriteByte('\r')
		case 't':
			b.WriteByte('\t')
		case 'u':
			r, err := p.hex4()
			if err != nil {
				return nil, err
			}
			if r >= 0xD800 && r < 0xDC00 { // a high surrogate needs its low half
				if !strings.HasPrefix(p.s[p.i:], "\\u") {
					return nil, p.fail("lone surrogate escape")
				}
				p.i += 2
				lo, err := p.hex4()
				if err != nil {
					return nil, err
				}
				if lo < 0xDC00 || lo >= 0xE000 {
					return nil, p.fail("lone surrogate escape")
				}
				r = 0x10000 + (r-0xD800)<<10 + (lo - 0xDC00)
			} else if r >= 0xDC00 && r < 0xE000 {
				return nil, p.fail("lone surrogate escape")
			}
			b.WriteRune(r)
		default:
			return nil, p.fail("invalid escape")
		}
	}
	return nil, p.fail("unterminated string")
}

func (p *parser) hex4() (rune, error) {
	if p.i+4 > len(p.s) {
		return 0, p.fail("truncated \\u escape")
	}
	n, err := strconv.ParseUint(p.s[p.i:p.i+4], 16, 16) // four hex digits: at most 0xFFFF
	if err != nil || strings.ContainsAny(p.s[p.i:p.i+4], "+-_") {
		return 0, p.fail("invalid \\u escape")
	}
	p.i += 4
	return rune(n), nil
}

// scanNumber consumes one JSON number and reports whether it was an int literal and how many
// digits that int has; ok is false when the text at p.i is not a number.
func (p *parser) scanNumber() (isInt bool, intDigits int, ok bool) {
	if p.i < len(p.s) && p.s[p.i] == '-' {
		p.i++
	}
	digits := func() int {
		n := 0
		for p.i < len(p.s) && p.s[p.i] >= '0' && p.s[p.i] <= '9' {
			p.i++
			n++
		}
		return n
	}
	intStart := p.i
	if p.i < len(p.s) && p.s[p.i] == '0' {
		p.i++
	} else if digits() == 0 {
		return false, 0, false
	}
	intDigits = p.i - intStart
	isInt = true
	if p.i < len(p.s) && p.s[p.i] == '.' {
		p.i++
		if digits() == 0 {
			return false, 0, false
		}
		isInt = false
	}
	if p.i < len(p.s) && (p.s[p.i] == 'e' || p.s[p.i] == 'E') {
		p.i++
		if p.i < len(p.s) && (p.s[p.i] == '+' || p.s[p.i] == '-') {
			p.i++
		}
		if digits() == 0 {
			return false, 0, false
		}
		isInt = false
	}
	return isInt, intDigits, true
}

func (p *parser) number() (any, error) {
	start := p.i
	isInt, intDigits, ok := p.scanNumber()
	if !ok {
		return nil, p.fail("invalid number")
	}
	if isInt && intDigits > MaxIntDigits {
		return nil, p.fail("int literal exceeds the limit of 4300 digits")
	}
	return json.Number(p.s[start:p.i]), nil
}

// validNumber reports whether text is one JSON number, whatever its length.
func validNumber(text string) bool {
	p := &parser{s: text}
	_, _, ok := p.scanNumber()
	return ok && p.i == len(text)
}

// Number converts a json.Number to what json.loads would have produced: an int (int64, or
// *big.Int when it does not fit) when the text has no fraction or exponent, else a float64.
func Number(n json.Number) (any, error) {
	text := string(n)
	if !validNumber(text) {
		return nil, fmt.Errorf("%q is not a JSON number", text)
	}
	if !strings.ContainsAny(text, ".eE") {
		if len(strings.TrimPrefix(text, "-")) > MaxIntDigits {
			return nil, fmt.Errorf("int literal exceeds the limit of %d digits", MaxIntDigits)
		}
		if i, err := strconv.ParseInt(text, 10, 64); err == nil {
			return i, nil
		}
		b, ok := new(big.Int).SetString(text, 10)
		if !ok {
			return nil, fmt.Errorf("%q is not a JSON number", text)
		}
		return b, nil
	}
	f, err := strconv.ParseFloat(text, 64)
	if err != nil && !errors.Is(err, strconv.ErrRange) {
		return nil, fmt.Errorf("%q is not a JSON number", text)
	}
	return f, nil // out of range is +-Inf, as float("1e400") is in Python
}

// Normalize maps v onto the shapes pyjson writes: nil, bool, string, integers, floats, *big.Int,
// []any, pyjson.Object and map[string]any. Pointers are followed, a json.Number becomes an int or
// a float by its spelling, lang.Object keeps its order, a map with string keys is written in
// sorted key order, and a slice or array (but not bytes) is a list. A nil slice is an empty list
// and a nil map an empty object. Anything else, a struct or a channel or a map with other keys,
// is refused rather than stringified.
func Normalize(v any) (any, error) { return normalize(v, 0) }

func normalize(v any, depth int) (any, error) {
	if depth > MaxDepth {
		return nil, fmt.Errorf("%w: nested deeper than %d levels", ErrUnsupported, MaxDepth)
	}
	switch x := v.(type) {
	case nil, bool, string, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64,
		*big.Int, float32, float64:
		return v, nil
	case json.Number:
		return Number(x)
	case []string:
		return x, nil
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			n, err := normalize(item, depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	case pyjson.Object:
		out := make(pyjson.Object, len(x))
		for i, m := range x {
			n, err := normalize(m.Value, depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = pyjson.Member{Key: m.Key, Value: n}
		}
		return out, nil
	case lang.Object:
		out := make(pyjson.Object, len(x))
		for i, m := range x {
			n, err := normalize(m.Value, depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = pyjson.Member{Key: m.Key, Value: n}
		}
		return out, nil
	}
	return normalizeReflect(reflect.ValueOf(v), depth)
}

func normalizeReflect(rv reflect.Value, depth int) (any, error) {
	if depth > MaxDepth {
		return nil, fmt.Errorf("%w: nested deeper than %d levels", ErrUnsupported, MaxDepth)
	}
	switch rv.Kind() {
	case reflect.Pointer, reflect.Interface:
		if rv.IsNil() {
			return nil, nil
		}
		return normalize(rv.Elem().Interface(), depth+1)
	case reflect.String:
		return rv.String(), nil
	case reflect.Bool:
		return rv.Bool(), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return rv.Uint(), nil
	case reflect.Float32, reflect.Float64:
		return rv.Float(), nil
	case reflect.Slice, reflect.Array:
		if rv.Type().Elem().Kind() == reflect.Uint8 {
			return nil, fmt.Errorf("%w: %s is bytes, which json.dumps refuses", ErrUnsupported, rv.Type())
		}
		out := make([]any, rv.Len())
		for i := range out {
			n, err := normalize(rv.Index(i).Interface(), depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			return nil, fmt.Errorf("%w: %s has non-string keys, whose order Go does not define",
				ErrUnsupported, rv.Type())
		}
		keys := rv.MapKeys()
		sort.Slice(keys, func(i, j int) bool { return keys[i].String() < keys[j].String() })
		out := make(pyjson.Object, len(keys))
		for i, k := range keys {
			n, err := normalize(rv.MapIndex(k).Interface(), depth+1)
			if err != nil {
				return nil, err
			}
			out[i] = pyjson.Member{Key: k.String(), Value: n}
		}
		return out, nil
	}
	return nil, fmt.Errorf("%w: %s", ErrUnsupported, rv.Type())
}

// Dumps is json.dumps(v, ensure_ascii=False) for a Go value, with the default ", " and ": "
// separators. It is Normalize followed by pyjson.Dumps.
func Dumps(v any) (string, error) {
	n, err := Normalize(v)
	if err != nil {
		return "", err
	}
	s, err := pyjson.Dumps(n)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrUnsupported, err)
	}
	return s, nil
}

// TypeName is type(v).__name__ for a decoded JSON value.
func TypeName(v any) string {
	switch x := v.(type) {
	case nil:
		return "NoneType"
	case bool:
		return "bool"
	case string:
		return "str"
	case float64:
		return "float"
	case json.Number:
		if strings.ContainsAny(string(x), ".eE") {
			return "float"
		}
		return "int"
	case []any:
		return "list"
	case pyjson.Object:
		return "dict"
	}
	return fmt.Sprintf("%T", v)
}

// Repr is repr(v) for a decoded JSON value: None, True, 'text', 1, 2.5, [..] and {'k': v}.
func Repr(v any) string {
	var b strings.Builder
	writeRepr(&b, v)
	return b.String()
}

func writeRepr(b *strings.Builder, v any) {
	switch x := v.(type) {
	case nil:
		b.WriteString("None")
	case bool:
		if x {
			b.WriteString("True")
		} else {
			b.WriteString("False")
		}
	case string:
		b.WriteString(pyjson.Repr(x))
	case json.Number, float64:
		b.WriteString(Str(x))
	case []any:
		b.WriteByte('[')
		for i, item := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			writeRepr(b, item)
		}
		b.WriteByte(']')
	case pyjson.Object:
		b.WriteByte('{')
		for i, m := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			writeRepr(b, m.Key)
			b.WriteString(": ")
			writeRepr(b, m.Value)
		}
		b.WriteByte('}')
	default:
		fmt.Fprint(b, v)
	}
}

// Str is str(v) for a decoded JSON scalar: the text itself for a string, True and False, the
// digits of an int, and the repr of a float (1.0, 1e+16, inf).
func Str(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case json.Number:
		n, err := Number(x)
		if err != nil {
			return string(x)
		}
		switch n := n.(type) {
		case int64:
			return strconv.FormatInt(n, 10)
		case *big.Int:
			return n.String()
		case float64:
			return floatStr(n)
		}
	case float64:
		return floatStr(x)
	}
	return Repr(v)
}

func floatStr(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	return pyjson.ReprFloat(f)
}

// KeyOf is a string that is equal for two decoded scalars exactly when Python's == and hash treat
// them as one dict key: 1, 1.0 and True are one key, and a str never equals a number.
func KeyOf(v any) string {
	switch x := v.(type) {
	case string:
		return "s:" + x
	case bool:
		if x {
			return "n:1"
		}
		return "n:0"
	case json.Number:
		n, err := Number(x)
		if err != nil {
			return "x:" + string(x)
		}
		switch n := n.(type) {
		case int64:
			return "n:" + strconv.FormatInt(n, 10)
		case *big.Int:
			return "n:" + n.String()
		case float64:
			if math.IsInf(n, 0) {
				return "n:" + floatStr(n)
			}
			r := new(big.Rat).SetFloat64(n) // the exact binary value, not its shortest spelling
			if r.IsInt() {
				return "n:" + r.Num().String()
			}
			return "n:" + r.String()
		}
	}
	return "x:" + Repr(v)
}
