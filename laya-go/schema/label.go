package schema

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"reflect"
	"strconv"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
)

type labelKind uint8

const (
	labelText labelKind = iota
	labelBool
	labelInt
	labelFloat
)

// Label is a choice's answer key. In Python a label is a str, an int, a float or a bool, and the
// answer reports it as that type: "choice": 1 for the int 1, "choice": true for True. Label holds
// the original value, so the three spellings laya uses can each be reproduced:
//
//   - String is str(label), the spelling in the prompt the model reads ("True", "1", "inf");
//   - MarshalJSON is the label as a JSON value (true, 1, "a");
//   - Key is the label as json.dumps writes it as an object key ("true", "1", "Infinity").
//
// Label is comparable, and two Labels are equal when they hold the same type and spelling: the
// string "1" and the int 1 are different labels, as they are in Python. The zero Label is the empty
// string. A NaN label equals another NaN label, where Python tells two NaN objects apart: json.loads
// hands out one object for every NaN in a document, so the wire form agrees, and two NaN options
// built in Go are refused as a repeat. A plain string is the easy path: Text("billing"), or ChoiceOf with string labels.
type Label struct {
	kind labelKind
	text string // the json.dumps spelling of the value: the string itself, true, 1, 1.5, Infinity
}

// Text is the label that is the string s.
func Text(s string) Label { return Label{kind: labelText, text: s} }

// LabelOf converts a Python scalar to a Label: a string, bool, any integer type, *big.Int, float,
// json.Number (an int literal is an int, anything with a fraction or exponent a float, as
// json.loads reads it), or a Label. Anything else is an error.
func LabelOf(v any) (Label, error) {
	switch x := v.(type) {
	case Label:
		return x, nil
	case string:
		return Text(x), nil
	case bool:
		return Label{kind: labelBool, text: strconv.FormatBool(x)}, nil
	case float64:
		return floatLabel(x), nil
	case float32:
		// the float the caller wrote: 0.1, not float32(0.1) widened to 0.10000000149011612
		f, _ := strconv.ParseFloat(strconv.FormatFloat(float64(x), 'g', -1, 32), 64)
		return floatLabel(f), nil
	case *big.Int:
		if x == nil {
			return Label{}, errors.New("schema: a nil *big.Int is not a label")
		}
		text := x.String()
		if len(strings.TrimPrefix(text, "-")) > pyvalue.MaxIntDigits {
			// str() of it raises in Python, so the question could never be rendered
			return Label{}, fmt.Errorf("schema: an int label exceeds the limit of %d digits", pyvalue.MaxIntDigits)
		}
		return Label{kind: labelInt, text: text}, nil
	case json.Number:
		n, err := pyvalue.Number(x)
		if err != nil {
			return Label{}, fmt.Errorf("schema: label %q: %w", string(x), err)
		}
		return LabelOf(n)
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.String:
		return Text(rv.String()), nil
	case reflect.Bool:
		return Label{kind: labelBool, text: strconv.FormatBool(rv.Bool())}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return Label{kind: labelInt, text: strconv.FormatInt(rv.Int(), 10)}, nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return Label{kind: labelInt, text: strconv.FormatUint(rv.Uint(), 10)}, nil
	}
	return Label{}, fmt.Errorf("schema: %T is not a label; use a string, bool, number or json.Number", v)
}

func floatLabel(f float64) Label {
	return Label{kind: labelFloat, text: pyjson.ReprFloat(f)}
}

// String is str(label): "True" for a bool, "1.0" for a float, "inf" for an overflow.
func (l Label) String() string {
	switch l.kind {
	case labelBool:
		if l.text == "true" {
			return "True"
		}
		return "False"
	case labelFloat:
		switch l.text {
		case "NaN":
			return "nan"
		case "Infinity":
			return "inf"
		case "-Infinity":
			return "-inf"
		}
	}
	return l.text
}

// Key is the label as json.dumps writes it as an object key: the string itself, "true" for
// True, "1" for 1, "1.5" for 1.5 and "Infinity" for an overflow. The keys of an answer's
// probabilities are Keys, and two labels can share one (1 and "1").
func (l Label) Key() string { return l.text }

// repr is repr(label), for laya's error messages.
func (l Label) repr() string {
	if l.kind == labelText {
		return pyjson.Repr(l.text)
	}
	return l.String()
}

// eqKey is equal for two labels exactly when Python treats them as one dict key: 1, 1.0 and True
// are one key, and a str is never a number.
func (l Label) eqKey() string {
	switch l.kind {
	case labelText:
		return "s:" + l.text
	case labelBool:
		if l.text == "true" {
			return "n:1"
		}
		return "n:0"
	case labelInt:
		return pyvalue.KeyOf(json.Number(l.text))
	}
	if l.text == "NaN" || l.text == "Infinity" || l.text == "-Infinity" {
		return "n:" + l.text
	}
	return pyvalue.KeyOf(json.Number(l.text))
}

// MarshalJSON writes the label as the JSON value it was: "a", true, 1, 1.5. An overflowing float
// is written as Python writes it, the bare token Infinity, which is not valid JSON; encoding/json
// refuses to carry it, and a direct call returns the bytes.
func (l Label) MarshalJSON() ([]byte, error) {
	if l.kind == labelText {
		text, err := pyjson.Dumps(l.text)
		return []byte(text), err
	}
	return []byte(l.text), nil
}

// UnmarshalJSON reads a string, bool or number, keeping its type; Infinity and NaN are accepted
// as Python accepts them.
func (l *Label) UnmarshalJSON(data []byte) error {
	v, err := pyvalue.Parse(data)
	if err != nil {
		return fmt.Errorf("schema: label: %w", err)
	}
	if v == nil {
		return errors.New("schema: a label is a string, bool or number, not null")
	}
	got, err := LabelOf(v)
	if err != nil {
		return fmt.Errorf("schema: a label is a string, bool or number, got %s", pyvalue.TypeName(v))
	}
	*l = got
	return nil
}
