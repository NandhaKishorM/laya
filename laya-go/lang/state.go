package lang

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// MaxChars is how many characters of a state are inspected (lang.py's max_chars default).
const MaxChars = 4000

// MaxDepth is how deep a state is read: containers nested deeper contribute nothing, as in
// lang.py.
const MaxDepth = 6

// maxHops bounds the pointers and interfaces followed without descending into a container, so a
// pointer that points at itself ends instead of looping. A hop is not a level of nesting, because
// Python has no pointers; a container cycle is ended by MaxDepth, as in Python.
const maxHops = 32

// Member is one key and value of an Object.
type Member struct {
	Key   string
	Value any
}

// Object is a mapping that keeps its insertion order. Python reads a dict's values in insertion
// order, and that order decides which text is inspected first, so a state that came from JSON
// must keep it: decoding JSON into an Object does (see UnmarshalJSON). A map is read in sorted
// key order instead.
type Object []Member

// UnmarshalJSON decodes a JSON object keeping its key order, as Python's json.loads does: a
// repeated key keeps its first position and takes its last value. Nested objects become Objects
// and arrays become []any; numbers become json.Number, which keeps an int an int and a float
// a float (a state's text ignores both, but a serializer must not). JSON null leaves
// the Object unchanged, the encoding/json convention for an Unmarshaler.
func (o *Object) UnmarshalJSON(data []byte) error {
	if string(bytes.Trim(data, " \t\r\n")) == "null" { // JSON white space only
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if tok != json.Delim('{') {
		return fmt.Errorf("lang.Object: want a JSON object, got %v", tok)
	}
	v, err := decodeObject(dec)
	if err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("lang.Object: data after the JSON object")
	}
	*o = v
	return nil
}

// decodeObject reads members up to the closing brace of an object whose opening brace was read.
func decodeObject(dec *json.Decoder) (Object, error) {
	obj := Object{}
	at := map[string]int{} // each key's position, so a repeated key overwrites in place
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		value, err := decodeValue(dec)
		if err != nil {
			return nil, err
		}
		key := keyTok.(string)
		if i, seen := at[key]; seen {
			obj[i].Value = value
			continue
		}
		at[key] = len(obj)
		obj = append(obj, Member{key, value})
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	return obj, nil
}

func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return tok, nil
	}
	if delim == '{' {
		return decodeObject(dec)
	}
	list := []any{}
	for dec.More() {
		v, err := decodeValue(dec)
		if err != nil {
			return nil, err
		}
		list = append(list, v)
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	return list, nil
}

// collector gathers string leaves in reading order. With a budget it stops as soon as the
// characters StateText can use are spent, which cannot change the result: later leaves would be
// discarded.
type collector struct {
	out     []string
	limited bool
	budget  int
	done    bool
}

func newCollector(limit int, limited bool) *collector {
	return &collector{limited: limited, budget: limit, done: limited && limit <= 0}
}

func (c *collector) leaf(s string) {
	if c.done {
		return
	}
	if !c.limited {
		c.out = append(c.out, s)
		return
	}
	n := utf8.RuneCountInString(s)
	if n > c.budget {
		c.out = append(c.out, head(s, c.budget))
		c.done = true
		return
	}
	c.out = append(c.out, s)
	if c.budget -= n + 1; c.budget <= 0 {
		c.done = true
	}
}

// iterText collects every string leaf of state, in reading order. It is O(number of leaves).
func iterText(state any) []string {
	c := newCollector(0, false)
	collectText(state, 0, c)
	return c.out
}

func collectText(state any, depth int, c *collector) {
	if depth > MaxDepth || state == nil || c.done {
		return
	}
	switch v := state.(type) {
	case json.Number:
		return // a number is not text, though its Go kind is string
	case string:
		c.leaf(unicodedata.ToValid(v))
		return
	case []byte:
		if utf8.Valid(v) {
			c.leaf(string(v))
		}
		return
	case []string:
		for _, s := range v {
			if c.done {
				return
			}
			collectText(s, depth+1, c)
		}
		return
	case []any:
		for _, item := range v {
			if c.done {
				return
			}
			collectText(item, depth+1, c)
		}
		return
	case Object:
		for _, m := range v {
			if c.done {
				return
			}
			collectText(m.Value, depth+1, c)
		}
		return
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		for _, k := range keys {
			if c.done {
				return
			}
			collectText(v[k], depth+1, c)
		}
		return
	}
	collectReflect(reflect.ValueOf(state), depth, 0, c)
}

var (
	objectType = reflect.TypeOf(Object(nil))
	numberType = reflect.TypeOf(json.Number(""))
)

// collectReflect walks every other shape the way lang.py walks a Mapping, list or tuple: the
// values of any map (in sorted key order, since a Go map has no insertion order), slice and array
// elements, strings and byte slices as leaves, pointers and interfaces followed. Anything else
// (numbers, structs, channels) contributes nothing, as in Python, where such objects are not
// strings, bytes, mappings or lists.
func collectReflect(v reflect.Value, depth, hops int, c *collector) {
	if depth > MaxDepth || hops > maxHops || !v.IsValid() || c.done {
		return
	}
	if v.Type() == numberType {
		return
	}
	if v.Type() == objectType && v.CanInterface() {
		collectText(v.Interface(), depth, c) // the ordered mapping, found below another container
		return
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			collectReflect(v.Elem(), depth, hops+1, c)
		}
	case reflect.String:
		c.leaf(unicodedata.ToValid(v.String()))
	case reflect.Slice, reflect.Array:
		if v.Type().Elem().Kind() == reflect.Uint8 {
			var raw []byte
			if v.Kind() == reflect.Slice {
				raw = v.Bytes()
			} else {
				raw = make([]byte, v.Len())
				reflect.Copy(reflect.ValueOf(raw), v)
			}
			if utf8.Valid(raw) {
				c.leaf(string(raw))
			}
			return
		}
		for i := 0; i < v.Len(); i++ {
			if c.done {
				return
			}
			collectReflect(v.Index(i), depth+1, 0, c)
		}
	case reflect.Map:
		// lang.py reads the values of any Mapping whatever its keys.
		keys := v.MapKeys()
		slices.SortStableFunc(keys, compareKeys)
		for _, k := range keys {
			if c.done {
				return
			}
			collectReflect(v.MapIndex(k), depth+1, 0, c)
		}
	}
}

// isScalar reports a state that is one string, bytes or nothing, which Analyze reads whole.
func isScalar(state any) bool {
	if state == nil {
		return true
	}
	switch state.(type) {
	case string, []byte:
		return true
	}
	v := reflect.ValueOf(state)
	for hops := 0; v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface; hops++ {
		if v.IsNil() || hops > maxHops {
			return true // nothing to read, or a pointer loop that reads as nothing
		}
		v = v.Elem()
	}
	return v.Kind() == reflect.String ||
		((v.Kind() == reflect.Slice || v.Kind() == reflect.Array) && v.Type().Elem().Kind() == reflect.Uint8)
}

// StateText flattens a state into one string of at most MaxChars characters, joining leaves with
// a space. A state is a string, bytes, or any map, slice or array of them, nested up to MaxDepth
// levels; numbers, structs and other values contribute nothing, as in lang.py. It stops reading
// once the budget is spent.
func StateText(state any) string {
	return StateTextN(state, MaxChars)
}

// StateTextN is StateText with an explicit character budget.
func StateTextN(state any, maxChars int) string {
	c := newCollector(maxChars, true)
	collectText(state, 0, c)
	return head(strings.Join(c.out, " "), maxChars)
}

// joinLeaves is StateTextN over leaves already collected.
func joinLeaves(leaves []string, maxChars int) string {
	c := newCollector(maxChars, true)
	for _, l := range leaves {
		c.leaf(l)
	}
	return head(strings.Join(c.out, " "), maxChars)
}

// head is the first count characters of s.
func head(s string, count int) string {
	if count <= 0 {
		return ""
	}
	if len(s) <= count {
		return s
	}
	for i := range s {
		if count == 0 {
			return s[:i]
		}
		count--
	}
	return s
}

// compareKeys orders map keys like fmt does for a map with a single key type: numbers
// numerically (NaN first), strings lexically, false before true, arrays and structs field by
// field, pointers and channels by address. Keys of different dynamic types (a map[any]T) are
// ordered by kind, then type name, then package path, then printed value, and are not compared
// field by field, since two distinct types may share a name.
func compareKeys(a, b reflect.Value) int {
	if c := cmp.Compare(a.Kind(), b.Kind()); c != 0 {
		return c
	}
	switch a.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return cmp.Compare(a.Int(), b.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return cmp.Compare(a.Uint(), b.Uint())
	case reflect.String:
		return strings.Compare(a.String(), b.String())
	case reflect.Float32, reflect.Float64:
		return cmp.Compare(a.Float(), b.Float()) // cmp orders NaN before every number
	case reflect.Complex64, reflect.Complex128:
		x, y := a.Complex(), b.Complex()
		if c := cmp.Compare(real(x), real(y)); c != 0 {
			return c
		}
		return cmp.Compare(imag(x), imag(y))
	case reflect.Bool:
		return cmp.Compare(b2i(a.Bool()), b2i(b.Bool()))
	case reflect.Pointer, reflect.Chan, reflect.UnsafePointer:
		return cmp.Compare(uint64(a.Pointer()), uint64(b.Pointer()))
	case reflect.Array:
		for i := 0; i < a.Len(); i++ {
			if c := compareKeys(a.Index(i), b.Index(i)); c != 0 {
				return c
			}
		}
	case reflect.Struct:
		for i := 0; i < a.NumField(); i++ {
			if c := compareKeys(a.Field(i), b.Field(i)); c != 0 {
				return c
			}
		}
	case reflect.Interface:
		switch {
		case a.IsNil() && b.IsNil():
			return 0
		case a.IsNil():
			return -1
		case b.IsNil():
			return 1
		}
		ea, eb := a.Elem(), b.Elem()
		if ea.Type() != eb.Type() {
			return compareTypes(ea, eb)
		}
		return compareKeys(ea, eb)
	}
	return 0
}

// compareTypes orders two values of different dynamic types without looking inside them.
func compareTypes(a, b reflect.Value) int {
	if c := cmp.Compare(a.Kind(), b.Kind()); c != 0 {
		return c
	}
	if c := strings.Compare(a.Type().String(), b.Type().String()); c != 0 {
		return c
	}
	if c := strings.Compare(a.Type().PkgPath(), b.Type().PkgPath()); c != 0 {
		return c
	}
	if a.CanInterface() && b.CanInterface() {
		return strings.Compare(fmt.Sprintf("%v", a.Interface()), fmt.Sprintf("%v", b.Interface()))
	}
	return 0
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
