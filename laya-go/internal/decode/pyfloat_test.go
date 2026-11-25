package decode

import (
	"math"
	"testing"
)

func TestPyFloat(t *testing.T) {
	for s, want := range map[string]float64{"1_5": 15, "1_0.5": 10.5, "2e1_0": 2e10, "１.５": 1.5, "٣": 3, "\U0001d7d8": 0, "\U0001d7ff": 9, "1\U0001d7ff": 19,
		"  2 ": 2, "-Infinity": math.Inf(-1), "INF": math.Inf(1), "1e400": math.Inf(1), ".5": 0.5, "5.": 5} {
		if f, ok := PyFloat(s); !ok || f != want {
			t.Errorf("PyFloat(%q) = %v %v, want %v", s, f, ok, want)
		}
	}
	for _, s := range []string{"+nan", "-NaN", "nan"} {
		if f, ok := PyFloat(s); !ok || !math.IsNaN(f) {
			t.Errorf("PyFloat(%q) = %v %v", s, f, ok)
		}
	}
	for _, s := range []string{"1_.5", "1._5", "_1", "1_", "1__0", "0x1p1", "infinit", "²", "2\u200b", "", "1e_1", "\x1c1", "1\x1f"} {
		if _, ok := PyFloat(s); ok {
			t.Errorf("PyFloat(%q) accepted", s)
		}
	}
}
