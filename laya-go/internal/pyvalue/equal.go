package pyvalue

import (
	"encoding/json"
	"math"
	"math/big"
	"strconv"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
)

// Equal is Python's == on decoded JSON values: True == 1 == 1.0 (numbers compare by exact value,
// an int against a float included), NaN equals nothing, a dict ignores order, and a list is
// compared item by item. A value of any type this package does not decode equals nothing.
func Equal(a, b any) bool {
	if na, ok := numeric(a); ok {
		nb, ok := numeric(b)
		return ok && na.equal(nb)
	}
	switch x := a.(type) {
	case nil:
		return b == nil
	case string:
		y, ok := b.(string)
		return ok && x == y
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !Equal(x[i], y[i]) {
				return false
			}
		}
		return true
	case pyjson.Object:
		y, ok := b.(pyjson.Object)
		if !ok || len(x) != len(y) {
			return false
		}
		for _, m := range x {
			other, found := member(y, m.Key)
			if !found || !Equal(m.Value, other) {
				return false
			}
		}
		return true
	}
	return false
}

func member(o pyjson.Object, key any) (any, bool) {
	for _, m := range o {
		if k, ok := m.Key.(string); ok {
			if kk, ok := key.(string); ok && k == kk {
				return m.Value, true
			}
		}
	}
	return nil, false
}

// num is a number as Python compares it: NaN, an infinity, or an exact rational.
type num struct {
	nan bool
	inf int // -1, 0 or +1
	rat *big.Rat
}

func (n num) equal(o num) bool {
	switch {
	case n.nan || o.nan:
		return false
	case n.inf != 0 || o.inf != 0:
		return n.inf == o.inf
	}
	return n.rat.Cmp(o.rat) == 0
}

func numeric(v any) (num, bool) {
	switch x := v.(type) {
	case bool:
		if x {
			return num{rat: big.NewRat(1, 1)}, true
		}
		return num{rat: new(big.Rat)}, true
	case float64:
		return floatNum(x), true
	case json.Number:
		text := string(x)
		if !strings.ContainsAny(text, ".eE") {
			b, ok := new(big.Int).SetString(text, 10)
			if !ok {
				return num{}, false
			}
			return num{rat: new(big.Rat).SetInt(b)}, true
		}
		f, err := strconv.ParseFloat(text, 64)
		if err != nil && !math.IsInf(f, 0) {
			return num{}, false
		}
		return floatNum(f), true
	}
	return num{}, false
}

func floatNum(f float64) num {
	switch {
	case math.IsNaN(f):
		return num{nan: true}
	case math.IsInf(f, 1):
		return num{inf: 1}
	case math.IsInf(f, -1):
		return num{inf: -1}
	}
	return num{rat: new(big.Rat).SetFloat64(f)}
}
