package pyvalue

import (
	"math"
	"testing"
)

func TestEqual(t *testing.T) {
	parse := func(s string) any {
		v, err := Parse([]byte(s))
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	for _, c := range []struct {
		a, b string
		want bool
	}{
		{`1`, `1.0`, true}, {`true`, `1`, true}, {`false`, `0.0`, true}, {`1`, `true`, true}, {`2`, `true`, false},
		{`1e22`, `10000000000000000000000`, true}, {`9007199254740993`, `9007199254740992.0`, false},
		{`{"a":1,"b":[1,2]}`, `{"b":[1.0,2],"a":true}`, true}, {`{"a":1}`, `{"a":1,"b":2}`, false},
		{`{"a":1}`, `{"b":1}`, false}, {`[1,2]`, `[2,1]`, false}, {`[]`, `{}`, false}, {`null`, `null`, true},
		{`null`, `0`, false}, {`"a"`, `"a"`, true}, {`"1"`, `1`, false}, {`Infinity`, `1e999`, true},
		{`Infinity`, `-Infinity`, false}, {`NaN`, `NaN`, false}, {`[NaN]`, `[NaN]`, false},
	} {
		if got := Equal(parse(c.a), parse(c.b)); got != c.want {
			t.Errorf("%s == %s: %v, want %v", c.a, c.b, got, c.want)
		}
	}
	if Equal(math.NaN(), math.NaN()) || Equal(struct{}{}, struct{}{}) {
		t.Error("NaN or an unknown type equals itself")
	}
}
