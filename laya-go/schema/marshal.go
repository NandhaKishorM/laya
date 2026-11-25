package schema

import (
	"bytes"
	"encoding/json"
	"reflect"
)

// MarshalPython is json.Marshal that keeps the bytes this package writes. encoding/json rewrites
// <, > and & (and always U+2028 and U+2029) as \u escapes in everything it writes, including the
// output of a MarshalJSON method, so json.Marshal of a Prediction is the same JSON value as
// Python's json.dumps(ensure_ascii=False) but not the same bytes when a label, a legend or a
// question id holds one of those characters. For a value that marshals itself (Prediction,
// Answers, an answer, Usage, Question, Questions, Label) MarshalPython returns its bytes as they
// are, which are Python's. Any other value goes through an Encoder with HTML escaping off, which
// escapes U+2028 and U+2029 in its own strings and, depending on the Go release, in the output
// of a MarshalJSON nested inside the value: only a value passed directly is sure to keep
// Python's bytes.
func MarshalPython(v any) ([]byte, error) {
	if m, ok := v.(json.Marshaler); ok {
		if rv := reflect.ValueOf(v); rv.Kind() == reflect.Pointer && rv.IsNil() {
			return []byte("null"), nil
		}
		return m.MarshalJSON()
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}
