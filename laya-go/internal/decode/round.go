package decode

import (
	"math"
	"strconv"
)

// Round4 is Python's round(v, 4): half to even on the exact binary value of v, keeping the sign
// of a negative zero and passing NaN and the infinities through. Math.Round(v*1e4)/1e4 is not
// it: it rounds half up and the multiply is itself inexact.
//
// strconv formats the exact binary value, rounding an exact tie to even, and parses the result
// back to the nearest double, which is the correctly rounded string method CPython uses.
func Round4(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return v
	}
	r, err := strconv.ParseFloat(strconv.FormatFloat(v, 'f', 4, 64), 64)
	if err != nil { // unreachable for a finite v: the text is a plain decimal
		return v
	}
	return r
}
