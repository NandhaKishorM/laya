package agent

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"
	"testing/fstest"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/decode"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
)

type idleSession struct{}

func (idleSession) Run(context.Context, infer.Batch) (infer.Output, error) {
	return infer.Output{}, errors.New("not run")
}
func (idleSession) Close() error { return nil }

func calibrationGolden(t *testing.T) fixtures.Object {
	t.Helper()
	return fixtures.Obj(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json"), "calibration"))
}

func checkpointWith(config string) fstest.MapFS {
	spec, cfg := fixtures.ToyTokenizer()
	return fstest.MapFS{
		"rl_agent_config.json":            {Data: []byte(config)},
		"tokenizer/tokenizer.json":        {Data: spec},
		"tokenizer/tokenizer_config.json": {Data: cfg},
	}
}

func binsOf(t *testing.T, v any) map[string]decode.Bins {
	t.Helper()
	if v == nil {
		return nil
	}
	out := map[string]decode.Bins{}
	obj := fixtures.Obj(t, v)
	for _, k := range obj.Keys() {
		e := obj.Must(k)
		if eo, ok := e.(fixtures.Object); ok { // a raw payload entry: {"bins", "values"}
			e = eo.Must("values")
		}
		var vals []float64
		for _, x := range fixtures.List(t, e) {
			vals = append(vals, fixtures.Float(t, x))
		}
		out[k] = decode.Bins{Values: vals}
	}
	return out
}

// TestCalibrationGolden holds every accepted and rejected payload shape to what laya's
// apply_calibration_payload does on an ONNXAgent-shaped object.
func TestCalibrationGolden(t *testing.T) {
	cases := fixtures.List(t, calibrationGolden(t).Must("cases"))
	if len(cases) != 128 {
		t.Fatalf("the golden holds %d cases", len(cases))
	}
	rejected := 0
	for _, c := range cases {
		c := fixtures.Obj(t, c)
		name := fixtures.Str(t, c.Must("name"))
		t.Run(name, func(t *testing.T) {
			a, err := Open(checkpointWith(fixtures.Str(t, c.Must("config"))), idleSession{},
				WithCalibration([]byte(fixtures.Str(t, c.Must("payload")))))
			if want, bad := c.Get("error"); bad {
				rejected++
				var ce *ConfigError
				if !errors.As(err, &ce) || ce.Field != "calibration" || ce.Err.Error() != want.(string) {
					t.Fatalf("got %v, want a calibration ConfigError %q", err, want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer a.Close()
			for i, w := range fixtures.List(t, c.Must("temperature")) {
				if got := a.temps.ByType[i]; got != fixtures.Float(t, w) {
					t.Errorf("temperature[%d] = %v, want %v", i, got, w)
				}
			}
			byOptions := fixtures.Obj(t, c.Must("by_options"))
			if len(a.temps.ByOptions) != len(byOptions.Keys()) {
				t.Errorf("by_options %v, want %d entries", a.temps.ByOptions, len(byOptions.Keys()))
			}
			for _, k := range byOptions.Keys() {
				if got := a.temps.ByOptions[k]; got != fixtures.Float(t, byOptions.Must(k)) {
					t.Errorf("by_options[%s] = %v, want %v", k, got, byOptions.Must(k))
				}
			}
			want := binsOf(t, c.Must("binning"))
			if len(a.temps.Binning) != len(want) {
				t.Errorf("binning %v, want %v", a.temps.Binning, want)
			}
			for k, w := range want {
				g, found := a.temps.Binning[k]
				if !found || len(g.Values) != len(w.Values) {
					t.Errorf("binning[%s] = %v, want %v", k, g.Values, w.Values)
					continue
				}
				for i := range w.Values {
					if g.Values[i] != w.Values[i] {
						t.Errorf("binning[%s][%d] = %v, want %v", k, i, g.Values[i], w.Values[i])
					}
				}
			}
			var wantWarnings []string
			for _, w := range fixtures.List(t, c.Must("warnings")) {
				wantWarnings = append(wantWarnings, fixtures.Str(t, fixtures.List(t, w)[1]))
			}
			got := a.Config().Warnings
			if len(got) != len(wantWarnings) {
				t.Fatalf("warnings %q, want %q", got, wantWarnings)
			}
			for i := range got {
				if got[i] != wantWarnings[i] {
					t.Errorf("warning %d:\n got %q\nwant %q", i, got[i], wantWarnings[i])
				}
			}
		})
	}
	if rejected != 63 {
		t.Errorf("%d cases were refused, the golden refuses 63", rejected)
	}
}

// TestApplyBinningGolden checks apply_binning_map over maps, buckets and confidences, edges included.
func TestApplyBinningGolden(t *testing.T) {
	cases := fixtures.List(t, calibrationGolden(t).Must("binning"))
	if len(cases) != 750 {
		t.Fatalf("the golden holds %d cases", len(cases))
	}
	for i, c := range cases {
		c := fixtures.Obj(t, c)
		m := binsOf(t, c.Must("map"))
		got := decode.ApplyBinning(fixtures.Float(t, c.Must("confidence")), fixtures.Str(t, c.Must("bucket")), m)
		if want := fixtures.Float(t, c.Must("result")); math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("case %d: %v, want %v", i, got, want)
		}
	}
}

// TestCalibrationPayloadMarshalGolden checks CalibrationPayload writes what calibration_payload
// does, and that WithCalibration reads it back.
func TestCalibrationPayloadMarshalGolden(t *testing.T) {
	cases := fixtures.List(t, calibrationGolden(t).Must("writer"))
	if len(cases) != 4 {
		t.Fatalf("the golden holds %d cases", len(cases))
	}
	for i, c := range cases {
		c := fixtures.Obj(t, c)
		p := CalibrationPayload{Temperature: [3]float64{1.5, 2, 0.75},
			TemperatureByOptions: map[string]float64{"choice:2": 2, "score:3-5": 0.5}}
		if m, ok := c.Must("model").(string); ok {
			p.ModelIDOrPath = &m
		}
		if s, ok := c.Must("subfolder").(string); ok {
			p.Subfolder = &s
		}
		if cfg, ok := c.Must("config").(fixtures.Object); ok {
			p.Config = map[string]any{}
			for _, k := range cfg.Keys() {
				p.Config[k] = plain(cfg.Must(k))
			}
		}
		if bm, ok := c.Must("binning_map").(fixtures.Object); ok {
			p.BinningMap = map[string]BinningEntry{}
			for _, k := range bm.Keys() {
				v := binsOf(t, bm)[k].Values
				p.BinningMap[k] = BinningEntry{Bins: len(v), Values: v}
			}
		}
		got, err := p.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		gv, err1 := pyvalue.Parse(got)
		wv, err2 := pyvalue.Parse([]byte(fixtures.Str(t, c.Must("payload"))))
		if err1 != nil || err2 != nil || !sameJSON(gv, wv) {
			t.Errorf("case %d: wrote %s, want %s", i, got, c.Must("payload"))
		}
		a, err := Open(checkpointWith(`{}`), idleSession{}, WithCalibration(got))
		if err != nil {
			t.Fatalf("case %d: read back: %v", i, err)
		}
		if a.temps.ByType != [3]float64{1.5, 2, 0.75} || a.temps.ByOptions["score:3-5"] != 0.5 || len(a.temps.Binning) != len(p.BinningMap) {
			t.Errorf("case %d: read back %+v", i, a.temps)
		}
		a.Close()
	}
}

func TestCalibrationEdges(t *testing.T) {
	open := func(opts ...Option) (*Agent, error) {
		return Open(checkpointWith(`{}`), idleSession{}, opts...)
	}
	for name, payload := range map[string]string{"not JSON": `{"temperature": [1,`, "trailing": `{"temperature": [1,1,1]} x`,
		"huge int": `{"version": 1` + `, "temperature": [1,1,1], "x": 1` + repeat("0", 4400) + `}`} {
		if _, err := open(WithCalibration([]byte(payload))); err == nil {
			t.Errorf("%s: opened", name)
		}
	}
	for _, empty := range [][]byte{nil, {}} {
		a, err := open(WithCalibration(empty))
		if err != nil {
			t.Fatalf("an empty payload: %v", err)
		}
		a.Close()
	}
	a, err := open(WithLangTemperatures(map[string]map[string]any{"fr": {"temperature": []any{2, 2, 2}}}),
		WithCalibration([]byte(`{"temperature": [3, 3, 3]}`)))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if a.temps.ByType != [3]float64{3, 3, 3} || a.temps.ByLanguage["fr"].ByType != [3]float64{2, 2, 2} {
		t.Errorf("calibration must replace the table and keep the language override: %+v", a.temps)
	}
}

func repeat(s string, n int) string {
	b := make([]byte, 0, n)
	for i := 0; i < n; i++ {
		b = append(b, s...)
	}
	return string(b)
}

// plain turns a fixture value into what encoding/json writes the way the golden spells it.
func plain(v any) any {
	switch x := v.(type) {
	case fixtures.Object:
		m := map[string]any{}
		for _, k := range x.Keys() {
			m[k] = plain(x.Must(k))
		}
		return m
	case []any:
		out := make([]any, len(x))
		for i := range x {
			out[i] = plain(x[i])
		}
		return out
	}
	return v
}

// sameJSON is pyvalue.Equal with the type kept: a bool, a number and a string never match each
// other, where Python's == has True == 1 == 1.0.
func sameJSON(a, b any) bool {
	switch x := a.(type) {
	case bool:
		y, ok := b.(bool)
		return ok && x == y
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !sameJSON(x[i], y[i]) {
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
			found := false
			for _, n := range y {
				if n.Key == m.Key {
					found = sameJSON(m.Value, n.Value)
					break
				}
			}
			if !found {
				return false
			}
		}
		return true
	}
	if _, ok := b.(bool); ok {
		return false
	}
	return pyvalue.Equal(a, b)
}

func TestSameNumberRefusesBool(t *testing.T) {
	for _, applied := range []float64{0, 1} {
		for _, b := range []bool{true, false} {
			if sameNumber(b, applied) {
				t.Errorf("sameNumber(%v, %v)", b, applied)
			}
		}
	}
	if !sameNumber(json.Number("1.0"), 1) || !sameNumber("１_0", 10) {
		t.Error("a number or a Unicode-digit string is the applied value")
	}
}

func TestVersionAndIntLiterals(t *testing.T) {
	for _, v := range []json.Number{"1.5", "2.5", "0.5"} {
		if _, err := versionChecksIdentity(v); err == nil {
			t.Errorf("version %s accepted", v)
		}
	}
	if ok, err := versionChecksIdentity(json.Number("2.0")); err != nil || !ok {
		t.Errorf("2.0: %v %v", ok, err)
	}
	for s, want := range map[string]int64{"２": 2, "٢": 2, "0_2": 2, "  +2\n": 2, "-０_２": -2, "\U0001d7ff": 9, "1\U0001d7d8": 10} {
		if n, ok := pyInt(s); !ok || n.Int64() != want {
			t.Errorf("pyInt(%q) = %v %v", s, n, ok)
		}
	}
	for _, s := range []string{"\x1c2", "2\x1f", "²", "2__0", "_2", "2_", "", "2.0"} {
		if _, ok := pyInt(s); ok {
			t.Errorf("pyInt(%q) accepted", s)
		}
	}
}

func TestMarshalChecksBins(t *testing.T) {
	for _, e := range []BinningEntry{{Bins: 2, Values: []float64{0.5}}, {Bins: 1, Values: []float64{0.5, 0.5}}, {Bins: 0}} {
		p := CalibrationPayload{BinningMap: map[string]BinningEntry{"c": e}}
		if _, err := p.MarshalJSON(); err == nil {
			t.Errorf("%+v written", e)
		}
	}
}

func TestSameJSONKeepsType(t *testing.T) {
	parse := func(s string) any {
		v, err := pyvalue.Parse([]byte(s))
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	for _, c := range [][2]string{{`true`, `1`}, {`1`, `true`}, {`[1, 2.5, true]`, `[1, 2.5, 1]`}, {`{"a": [true]}`, `{"a": [1.0]}`}, {`false`, `0`}} {
		if sameJSON(parse(c[0]), parse(c[1])) {
			t.Errorf("%s matched %s", c[0], c[1])
		}
	}
	if !sameJSON(parse(`{"a": [1, 2.0, true], "b": null}`), parse(`{"b": null, "a": [1.0, 2, true]}`)) {
		t.Error("equal payloads differ")
	}
}
