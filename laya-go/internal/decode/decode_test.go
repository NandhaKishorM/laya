package decode

import (
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/goldens"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// plain converts a parsed golden value into the maps and slices a JSON config decodes to.
func plain(v any) any {
	switch x := v.(type) {
	case fixtures.Object:
		out := make(map[string]any, len(x))
		for _, m := range x {
			out[m.Key] = plain(m.Value)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = plain(item)
		}
		return out
	}
	return v
}

func floats32(tb testing.TB, v any) []float32 {
	tb.Helper()
	list := fixtures.List(tb, v)
	out := make([]float32, len(list))
	for i, item := range list {
		out[i] = float32(fixtures.Float(tb, item))
	}
	return out
}

// TestDecodeGolden holds the decoder to Agent._decode_answers: the real method, on logit rows
// aimed at every temperature bucket, a tie, a uniform row, overflowing magnitudes, the option
// permutation and the language override. The answer is compared as the JSON it marshals to, so
// key order, rounding and float spelling are all part of the contract.
func TestDecodeGolden(t *testing.T) {
	golden := fixtures.Load(t, "decode.json")
	for _, name := range []string{"english", "multilingual"} {
		entry := fixtures.Obj(t, fixtures.Field(t, golden, name))
		config := fixtures.Obj(t, entry.Must("config"))
		temps, err := NewTemperatures(
			fixtures.List(t, config.Must("temperature")),
			plain(config.Must("temperature_by_options")).(map[string]any),
			plain(config.Must("lang_temperatures_raw")).(map[string]any))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		cases := fixtures.Obj(t, entry.Must("cases"))
		if len(cases) < 20 {
			t.Fatalf("%s: the golden holds %d cases", name, len(cases))
		}
		for _, c := range cases {
			rec := fixtures.Obj(t, c.Value)
			t.Run(name+"/"+c.Key, func(t *testing.T) {
				var order []int32
				if o := rec.Must("option_order"); o != nil {
					order = fixtures.Ints(t, o)
				}
				q := goldens.Question(t, fixtures.Obj(t, rec.Must("question")), order)
				lang := fixtures.StrOrEmpty(t, rec.Must("lang"))
				logits := floats32(t, rec.Must("logits"))
				k := len(logits)
				width := fixtures.Int(t, rec.Must("logits_block_width"))
				for len(logits) < width { // the columns past k hold 99, which a decoder must not read
					logits = append(logits, 99)
				}
				row := Row{Logits: logits, K: k, Act: floats32(t, rec.Must("act"))}
				answer, err := Decode(q, row, lang, temps)
				if err != nil {
					t.Fatal(err)
				}
				got, err := answer.(interface{ MarshalJSON() ([]byte, error) }).MarshalJSON()
				if err != nil {
					t.Fatal(err)
				}
				if want := fixtures.Marshal(t, rec.Must("answer")); string(got) != string(want) {
					t.Errorf("answer differs from Python's\n got %s\nwant %s", got, want)
				}
			})
		}
	}
}

// TestTemperaturesGolden resolves the checkpoints' own configs, as the agent does: the raw
// tables from sequences.json, clamped, must be the tables decode.json was recorded with.
func TestTemperaturesGolden(t *testing.T) {
	seqs := fixtures.Load(t, "sequences.json")
	golden := fixtures.Load(t, "decode.json")
	for _, name := range []string{"english", "multilingual"} {
		raw := fixtures.Obj(t, fixtures.Obj(t, fixtures.Field(t, seqs, name)).Must("config"))
		recorded := fixtures.Obj(t, fixtures.Obj(t, fixtures.Field(t, golden, name)).Must("config"))
		byOptions, _ := plain(raw.Must("temperature_by_options")).(map[string]any)
		temps, err := NewTemperatures(fixtures.List(t, raw.Must("temperature")), byOptions, nil)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range fixtures.List(t, recorded.Must("temperature")) {
			if want := fixtures.Float(t, v); temps.ByType[i] != want {
				t.Errorf("%s: ByType[%d] = %v, Python clamps to %v", name, i, temps.ByType[i], want)
			}
		}
		wantBy := fixtures.Obj(t, recorded.Must("temperature_by_options"))
		if len(temps.ByOptions) != len(wantBy) {
			t.Errorf("%s: %d bucket temperatures, Python has %d", name, len(temps.ByOptions), len(wantBy))
		}
		for _, m := range wantBy {
			if got, want := temps.ByOptions[m.Key], fixtures.Float(t, m.Value); got != want {
				t.Errorf("%s: %s = %v, Python clamps to %v", name, m.Key, got, want)
			}
		}
	}
}

func TestRound4Golden(t *testing.T) {
	golden := fixtures.Obj(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json"), "round4"))
	values, rounded := fixtures.List(t, golden.Must("values")), fixtures.List(t, golden.Must("rounded"))
	if len(values) < 5000 || len(values) != len(rounded) {
		t.Fatalf("golden holds %d values and %d results", len(values), len(rounded))
	}
	bad := 0
	for i := range values {
		v, want := fixtures.Float(t, values[i]), fixtures.Float(t, rounded[i])
		got := Round4(v)
		if math.Float64bits(got) != math.Float64bits(want) {
			if bad++; bad <= 5 {
				t.Errorf("Round4(%v) = %v, Python round gives %v", v, got, want)
			}
		}
	}
	if bad > 5 {
		t.Errorf("%d of %d values differ", bad, len(values))
	}
}

func TestRound4Edges(t *testing.T) {
	cases := map[string]struct{ in, want float64 }{
		"tie rounds down to even": {0.03125, 0.0312},
		"tie rounds up to even":   {0.09375, 0.0938},
		"negative zero kept":      {math.Copysign(0, -1), math.Copysign(0, -1)},
		"tiny negative is -0":     {-1e-5, math.Copysign(0, -1)},
		"already four places":     {0.1234, 0.1234},
	}
	for name, c := range cases {
		if got := Round4(c.in); math.Float64bits(got) != math.Float64bits(c.want) {
			t.Errorf("%s: Round4(%v) = %v, want %v", name, c.in, got, c.want)
		}
	}
	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if got := Round4(v); !(got == v || (math.IsNaN(got) && math.IsNaN(v))) {
			t.Errorf("Round4(%v) = %v, want it passed through", v, got)
		}
	}
}

func TestClamp(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want float64
	}{
		{"inside", 2.0, 2.0}, {"below is raised", 0.1006, 0.5}, {"above is lowered", 9.0, 5.0},
		{"bool is not a number", true, 1}, {"false is not a number", false, 1},
		{"nil", nil, 1}, {"NaN", math.NaN(), 1}, {"+Inf", math.Inf(1), 1}, {"-Inf", math.Inf(-1), 1},
		{"integer", 3, 3}, {"small integer raised", 0, 0.5}, {"numeric string", "2.5", 2.5},
		{"padded string", " 2.5 ", 2.5}, {"text", "warm", 1}, {"hex float is not Python", "0x1p1", 1},
		{"list", []any{1.0}, 1}, {"float32", float32(1.5), 1.5},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Clamp(c.in); got != c.want {
				t.Errorf("Clamp(%v) = %v, want %v", c.in, got, c.want)
			}
		})
	}
}

func TestBucket(t *testing.T) {
	cases := []struct {
		typ  schema.Type
		k    int
		want string
	}{
		{schema.TypeChoice, 0, "choice:2"}, {schema.TypeChoice, 2, "choice:2"}, {schema.TypeChoice, 3, "choice:3-5"},
		{schema.TypeScore, 5, "score:3-5"}, {schema.TypeScore, 6, "score:6-10"}, {schema.TypeNoul, 10, "noul:6-10"},
		{schema.TypeChoice, 11, "choice:11+"}, {schema.TypeChoice, 400, "choice:11+"},
	}
	for _, c := range cases {
		if got := Bucket(c.typ, c.k); got != c.want {
			t.Errorf("Bucket(%s, %d) = %s, want %s", c.typ, c.k, got, c.want)
		}
	}
}

func TestTemperaturesFor(t *testing.T) {
	temps, err := NewTemperatures([]any{2.0, 3.0, 4.0}, map[string]any{"choice:3-5": 1.5, "noul:2": 0.1},
		map[string]any{
			"ZH-hans": map[string]any{"temperature": []any{1.1, 1.2, 1.3}, "temperature_by_options": map[string]any{"choice:2": 9}},
			"de":      nil,
			"fr":      map[string]any{"temperature": nil},
		})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name     string
		typ      schema.Type
		k        int
		lang     string
		want     float64
		override bool
	}{
		{"type default", schema.TypeScore, 4, "", 3, false},
		{"bucket beats type", schema.TypeChoice, 4, "", 1.5, false},
		{"bucket value is clamped", schema.TypeNoul, 2, "", 0.5, false},
		{"language replaces both", schema.TypeChoice, 4, "zh", 1.1, true},
		{"language subtag and case", schema.TypeScore, 4, "ZH-Hant-TW", 1.2, true},
		{"language bucket clamped", schema.TypeChoice, 2, "zh", 5, true},
		{"unknown language falls through", schema.TypeChoice, 4, "xx", 1.5, false},
		{"null override inherits the raw table", schema.TypeScore, 4, "de", 3, true},
		{"null temperature inherits", schema.TypeNoul, 4, "fr", 4, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, over := temps.For(c.typ, c.k, c.lang)
			if got != c.want || over != c.override {
				t.Errorf("For(%s, %d, %q) = %v, %v; want %v, %v", c.typ, c.k, c.lang, got, over, c.want, c.override)
			}
		})
	}
	if got, over := (*Temperatures)(nil).For(schema.TypeChoice, 3, "zh"); got != 1 || over {
		t.Errorf("a nil table gives %v, %v", got, over)
	}
}

func TestNewTemperaturesRefuses(t *testing.T) {
	cases := map[string]struct {
		byType []any
		langs  map[string]any
		want   string
	}{
		"two temperatures":         {[]any{1.0, 1.0}, nil, "3 values"},
		"language not a mapping":   {nil, map[string]any{"de": 5}, "must be a mapping"},
		"language list too short":  {nil, map[string]any{"de": map[string]any{"temperature": []any{1.0}}}, "list of 3"},
		"language list not a list": {nil, map[string]any{"de": map[string]any{"temperature": 2}}, "list of 3"},
		"by options not a mapping": {nil, map[string]any{"de": map[string]any{"temperature_by_options": []any{1.0}}}, "mapping of bucket"},
	}
	for name, c := range cases {
		_, err := NewTemperatures(c.byType, nil, c.langs)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want it to mention %q", name, err, c.want)
		}
	}
}

func mustQ(q schema.Question, err error) schema.Question { return schema.MustQuestion(q, err) }

func TestDecodeBranches(t *testing.T) {
	two := mustQ(schema.ChoiceOf("pick", "a", "b"))
	temps := &Temperatures{ByType: [3]float64{1, 1, 1}}
	act := []float32{0.25}

	t.Run("binning applies without a language override only", func(t *testing.T) {
		binned := &Temperatures{ByType: [3]float64{1, 1, 1},
			Binning:    map[string]Bins{"choice:2": {Values: []float64{0.1, 0.2, 0.3, 0.9}}},
			ByLanguage: map[string]LanguageTemperatures{"zh": {ByType: [3]float64{1, 1, 1}}}}
		row := Row{Logits: []float32{3, 0}, K: 2, Act: act} // max(p) = 0.9526 -> bin 3
		a, err := Decode(two, row, "", binned)
		if err != nil {
			t.Fatal(err)
		}
		if got := a.Base().AnswerConfidence; got != 0.9 {
			t.Errorf("binned AnswerConfidence = %v, want 0.9", got)
		}
		a, _ = Decode(two, row, "zh", binned)
		if got := a.Base().AnswerConfidence; got != 0.9526 {
			t.Errorf("with a language override AnswerConfidence = %v, want the raw 0.9526", got)
		}
		other := &Temperatures{ByType: [3]float64{1, 1, 1}, Binning: map[string]Bins{"score:2": {Values: []float64{0.1}}}}
		a, _ = Decode(two, row, "", other)
		if got := a.Base().AnswerConfidence; got != 0.9526 {
			t.Errorf("a bucket the map was not fitted for = %v, want the raw 0.9526", got)
		}
	})

	t.Run("a single option has confidence one", func(t *testing.T) {
		one := mustQ(schema.ChoiceOf("pick", "only"))
		a, err := Decode(one, Row{Logits: []float32{7}, K: 1, Act: act}, "", temps)
		if err != nil {
			t.Fatal(err)
		}
		c := a.(*schema.ChoiceAnswer)
		if c.Choice != schema.Text("only") || c.Confidence != 1 || c.AnswerConfidence != 1 {
			t.Errorf("single option answer = %+v", c)
		}
	})

	t.Run("score reports the expected level", func(t *testing.T) {
		s := mustQ(schema.Score("rate", "bad", "ok", "good"))
		a, err := Decode(s, Row{Logits: []float32{0, 0, 0}, K: 3, Act: act}, "", temps)
		if err != nil {
			t.Fatal(err)
		}
		if got := a.(*schema.ScoreAnswer).Score; got != 1 {
			t.Errorf("uniform three levels score %v, want 1", got)
		}
	})

	t.Run("first maximum wins a tie", func(t *testing.T) {
		a, _ := Decode(two, Row{Logits: []float32{1, 1}, K: 2, Act: act}, "", temps)
		if got := a.(*schema.ChoiceAnswer).Choice; got != schema.Text("a") {
			t.Errorf("tie chose %q", got)
		}
	})

	t.Run("a permuted row is put back in label order", func(t *testing.T) {
		q, err := mustQ(schema.ChoiceOf("pick", "a", "b", "c")).WithOptionOrder([]int{2, 0, 1})
		if err != nil {
			t.Fatal(err)
		}
		// slot 0 shows option 2 and carries the largest logit, so option "c" wins
		a, _ := Decode(q, Row{Logits: []float32{5, 0, 0}, K: 3, Act: act}, "", temps)
		if got := a.(*schema.ChoiceAnswer).Choice; got != schema.Text("c") {
			t.Errorf("permuted decode chose %q, want c", got)
		}
	})
}

func TestDecodeRefuses(t *testing.T) {
	q := mustQ(schema.ChoiceOf("pick", "a", "b"))
	noul := mustQ(schema.Noul("holds"))
	cases := map[string]struct {
		q   schema.Question
		row Row
	}{
		"zero question":       {schema.Question{}, Row{Logits: []float32{1, 2}, K: 2, Act: []float32{0}}},
		"wrong marker count":  {q, Row{Logits: []float32{1, 2, 3}, K: 3, Act: []float32{0}}},
		"row too narrow":      {q, Row{Logits: []float32{1}, K: 2, Act: []float32{0}}},
		"no action":           {q, Row{Logits: []float32{1, 2}, K: 2}},
		"NaN logit":           {q, Row{Logits: []float32{float32(math.NaN()), 2}, K: 2, Act: []float32{0}}},
		"infinite logit":      {q, Row{Logits: []float32{float32(math.Inf(1)), 2}, K: 2, Act: []float32{0}}},
		"NaN action":          {q, Row{Logits: []float32{1, 2}, K: 2, Act: []float32{float32(math.NaN())}}},
		"noul row too narrow": {noul, Row{Logits: []float32{1}, K: 1, Act: []float32{0}}},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Decode(c.q, c.row, "", nil); err == nil {
				t.Fatal("no error")
			} else if !strings.Contains(err.Error(), ErrShape.Error()) {
				t.Errorf("err = %v, want it to wrap ErrShape", err)
			}
		})
	}
}

func TestActionProbabilities(t *testing.T) {
	got, err := ActionProbabilities([]float32{0, 0})
	if err != nil || len(got) != 2 || got[0] != 0.5 || got[1] != 0.5 {
		t.Errorf("uniform = %v, %v", got, err)
	}
	got, err = ActionProbabilities([]float32{1000, 0})
	if err != nil || got[0] != 1 || got[1] != 0 {
		t.Errorf("a large logit overflowed: %v, %v", got, err)
	}
	for name, row := range map[string][]float32{"empty": nil, "NaN": {float32(math.NaN())}} {
		if _, err := ActionProbabilities(row); err == nil {
			t.Errorf("%s row: no error", name)
		}
	}
}

func fixtureQuestion(t *testing.T, kind string, crit any, order []int) schema.Question {
	t.Helper()
	var (
		q   schema.Question
		err error
	)
	switch kind {
	case "choice":
		labels := []string{}
		for _, l := range crit.([]any) {
			labels = append(labels, l.(string))
		}
		q, err = schema.ChoiceOf("i", labels...)
	case "score":
		levels := crit.([]any)
		q, err = schema.Score("i", levels...)
	default:
		q, err = schema.Noul("i")
	}
	if err != nil {
		t.Fatal(err)
	}
	if order != nil {
		if q, err = q.WithOptionOrder(order); err != nil {
			t.Fatal(err)
		}
	}
	return q
}

func orderOf(t *testing.T, v any) []int {
	t.Helper()
	if v == nil {
		return nil
	}
	var out []int
	for _, i := range fixtures.Ints(t, v) {
		out = append(out, int(i))
	}
	return out
}

// TestLabelAnswersGolden holds the answer JSON to the bytes Python writes for bool, int, float and
// str labels: "choice": true, "choice": 1, the key "Infinity", the key "true". A label's type
// survives the answer, and 1 and "1" stay two labels.
func TestLabelAnswersGolden(t *testing.T) {
	cases := fixtures.List(t, fixtures.Obj(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json"), "label_answers")).Must("cases"))
	if len(cases) < 400 {
		t.Fatalf("the golden holds %d cases", len(cases))
	}
	typed := 0
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		criteria := fixtures.Str(t, rec.Must("criteria_text"))
		qs, err := schema.ParseQuestions([]byte(`{"q":{"type":"choice","instructions":"i","criteria":` + criteria + `}}`))
		if err != nil {
			t.Fatalf("case %d %s: %v", i, criteria, err)
		}
		q, _ := qs.Get("q")
		if order := orderOf(t, rec.Must("option_order")); order != nil {
			if q, err = q.WithOptionOrder(order); err != nil {
				t.Fatal(err)
			}
		}
		if got, want := q.RenderedOptions(), fixtures.Strings(t, rec.Must("options")); strings.Join(got, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("case %d %s: options %q, want %q", i, criteria, got, want)
		}
		logits := floats32(t, rec.Must("logits"))
		answer, err := Decode(q, Row{Logits: logits, K: len(logits), Act: []float32{0.25}}, "", &Temperatures{ByType: [3]float64{1, 1, 1}})
		if err != nil {
			t.Fatal(err)
		}
		got, err := answer.(json.Marshaler).MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		if want := fixtures.Str(t, rec.Must("answer_text")); string(got) != want {
			t.Errorf("case %d %s\n got %s\nwant %s", i, criteria, got, want)
		}
		if !strings.HasPrefix(string(got), `{"type":"choice","choice":"`) {
			typed++
		}
	}
	if typed < 200 {
		t.Errorf("only %d answers have a non-string choice", typed)
	}
}

// TestDecodeFuzzGolden decodes 5,000 random, tied and near-tied rows and compares the marshaled
// answer with Python's bytes. Two kinds of difference are tolerated and counted, because Decode
// computes in float64 where numpy computes in float32, and numpy's float32 exp is a SIMD routine
// whose last bits differ between CPUs (the same row is answered differently by Python on arm64
// and on x86-64):
//
//   - a number that differs by one unit in the fourth decimal (a value within 1e-4 sitting on a
//     rounding boundary);
//   - the choice, when the option Python chose and the one Go chose both have a quotient within
//     nearTieBand of the largest. Go's is the first maximum of the float64 probabilities;
//     everything else in the answer must still be Python's.
//
// Anything else fails.
func TestDecodeFuzzGolden(t *testing.T) {
	cases := fixtures.List(t, fixtures.Obj(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json"), "answers_fuzz")).Must("cases"))
	if len(cases) != 5000 {
		t.Fatalf("the golden holds %d cases", len(cases))
	}
	identical, nearTie, rounded, bad := 0, 0, 0, 0
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		kind := fixtures.Str(t, rec.Must("type"))
		order := orderOf(t, rec.Must("option_order"))
		q := fixtureQuestion(t, kind, rec.Must("criteria"), order)
		var logits []float32
		for _, s := range fixtures.Strings(t, rec.Must("logits")) {
			f, err := strconv.ParseFloat(s, 32)
			if err != nil {
				t.Fatal(err)
			}
			logits = append(logits, float32(f))
		}
		temp := rec.Must("temperature")
		temps, err := NewTemperatures([]any{temp, temp, temp}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		answer, err := Decode(q, Row{Logits: logits, K: len(logits), Act: []float32{0.25}}, "", temps)
		if err != nil {
			t.Fatal(err)
		}
		out, _ := answer.(json.Marshaler).MarshalJSON()
		want := fixtures.Str(t, rec.Must("answer_text"))
		if string(out) == want {
			identical++
			continue
		}
		gotTree, _ := pyvalue.Parse(out)
		wantTree, _ := pyvalue.Parse([]byte(want))
		var diffs []jsonDiff
		diffJSON("", gotTree, wantTree, &diffs)
		ok, sawRound, sawChoice := true, false, false
		for _, d := range diffs {
			switch {
			case d.path == "/choice" && kind == "choice":
				goIdx, _ := strconv.Atoi(d.got[2:])
				pyIdx, _ := strconv.Atoi(d.want[2:])
				sawChoice = true
				ok = ok && nearTied(logits, order, temps, kind, goIdx, pyIdx)
			case d.delta > 0 && d.delta < 1.5e-4:
				sawRound = true
			default:
				ok = false
			}
		}
		switch {
		case !ok:
			if bad++; bad <= 5 {
				t.Errorf("case %d (%s, k=%d, T=%v, logits %v)\n go: %s\n py: %s", i, kind, len(logits), temp, logits, out, want)
			}
		case sawChoice:
			nearTie++
		case sawRound:
			rounded++
		}
	}
	t.Logf("%d of %d decodes are Python's bytes; %d differ only in the choice among float32 near-ties; %d only by 1e-4 on a rounding boundary; %d differ otherwise",
		identical, len(cases), nearTie, rounded, bad)
	if bad > 0 {
		t.Errorf("%d of %d decodes differ from Python beyond the tolerated float32 effects", bad, len(cases))
	}
	if rounded > 25 {
		t.Errorf("%d rounding rows: the tolerance is hiding more than float32 arithmetic", rounded)
	}
}

type jsonDiff struct {
	path, got, want string
	delta           float64
}

// diffJSON lists where two decoded answers differ; numbers carry their distance.
func diffJSON(path string, got, want any, out *[]jsonDiff) {
	switch w := want.(type) {
	case pyjson.Object:
		g, ok := got.(pyjson.Object)
		if !ok || len(g) != len(w) {
			*out = append(*out, jsonDiff{path: path, got: pyvalue.Repr(got), want: pyvalue.Repr(want), delta: -1})
			return
		}
		for i := range w {
			if g[i].Key != w[i].Key {
				*out = append(*out, jsonDiff{path: path, got: pyvalue.Repr(got), want: pyvalue.Repr(want), delta: -1})
				return
			}
			diffJSON(path+"/"+w[i].Key.(string), g[i].Value, w[i].Value, out)
		}
	case json.Number:
		g, ok := got.(json.Number)
		if !ok {
			*out = append(*out, jsonDiff{path: path, got: pyvalue.Repr(got), want: pyvalue.Repr(want), delta: -1})
			return
		}
		if g != w {
			a, _ := g.Float64()
			b, _ := w.Float64()
			*out = append(*out, jsonDiff{path: path, got: string(g), want: string(w), delta: math.Abs(a - b)})
		}
	default:
		if pyvalue.Repr(got) != pyvalue.Repr(want) {
			*out = append(*out, jsonDiff{path: path, got: pyvalue.Repr(got), want: pyvalue.Repr(want), delta: -1})
		}
	}
}

// nearTieBand is how far below the largest quotient an option may sit and still be one Python can
// pick: about sixteen float32 steps at 1, a probability ratio of 1 - 1e-6.
const nearTieBand = 1e-6

// nearTied reports whether the options Go and Python chose both sit within nearTieBand of the
// largest quotient.
func nearTied(logits []float32, order []int, temps *Temperatures, kind string, goIdx, pyIdx int) bool {
	t, _ := temps.For(schema.Type(kind), len(logits), "")
	z := make([]float32, len(logits))
	top := float32(math.Inf(-1))
	for i, l := range logits {
		z[i] = l / float32(t)
		top = max(top, z[i])
	}
	if order != nil {
		canonical := make([]float32, len(z))
		for slot, option := range order {
			canonical[option] = z[slot]
		}
		z = canonical
	}
	for _, i := range []int{goIdx, pyIdx} {
		if i < 0 || i >= len(z) || float64(top)-float64(z[i]) > nearTieBand {
			return false
		}
	}
	return true
}

// TestQuotientTieFallsToTheFirst: Python divides the float32 logits by the temperature in float32,
// so two different logits can become one quotient, an exact tie that numpy.argmax gives to the
// lower index. Each pair is two adjacent float32 values that tie at its temperature, and numpy
// answers option 0 for both. The first pair fails if the quotient is kept in float64; the second,
// whose temperature is not a float32, also fails if the division is done in float64 and rounded.
func TestQuotientTieFallsToTheFirst(t *testing.T) {
	q, err := schema.ChoiceOf("pick", "first", "second")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		temperature  float64
		lower, upper uint32
	}{
		{3.0, 1069547521, 1069547522},
		{1.3, 1072020843, 1072020844},
	} {
		lower, upper := math.Float32frombits(c.lower), math.Float32frombits(c.upper)
		if !(lower < upper) || lower/float32(c.temperature) != upper/float32(c.temperature) {
			t.Fatalf("%v and %v no longer tie when divided by %v in float32", lower, upper, c.temperature)
		}
		temps, err := NewTemperatures([]any{c.temperature, c.temperature, c.temperature}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		answer, err := Decode(q, Row{Logits: []float32{lower, upper}, K: 2, Act: []float32{0.5}}, "", temps)
		if err != nil {
			t.Fatal(err)
		}
		if got := answer.(*schema.ChoiceAnswer).Choice; got != schema.Text("first") {
			t.Errorf("T=%v: choice = %v, want first: the quotients are one float32", c.temperature, got)
		}
	}
}

// TestLargeLogitsDoNotOverflow: the maximum is subtracted before exp, so a logit of 800, whose
// exp is not a float64, still answers 1.0 and 0.0 as Python does.
func TestLargeLogitsDoNotOverflow(t *testing.T) {
	q, err := schema.ChoiceOf("pick", "first", "second")
	if err != nil {
		t.Fatal(err)
	}
	temps, err := NewTemperatures([]any{1.0, 1.0, 1.0}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	answer, err := Decode(q, Row{Logits: []float32{800, 0}, K: 2, Act: []float32{0.5}}, "", temps)
	if err != nil {
		t.Fatal(err)
	}
	out, err := answer.(json.Marshaler).MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"type":"choice","choice":"first","probabilities":{"first":1.0,"second":0.0},"confidence":1.0,"answer_confidence":1.0,"action":{"act_probability":0.5}}`
	if string(out) != want {
		t.Errorf("got  %s\nwant %s", out, want)
	}
}

// TestQuotientOverflowIsRefused: a finite logit can overflow float32 when divided by a
// temperature below 1. Python answers NaN for every field; Decode refuses the row. A quotient
// that overflows downwards is no trouble on either side: its probability is 0.
func TestQuotientOverflowIsRefused(t *testing.T) {
	q, err := schema.ChoiceOf("pick", "first", "second")
	if err != nil {
		t.Fatal(err)
	}
	temps, err := NewTemperatures([]any{0.5, 0.5, 0.5}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if answer, err := Decode(q, Row{Logits: []float32{3e38, 1}, K: 2, Act: []float32{0.5}}, "", temps); !errors.Is(err, ErrShape) {
		t.Errorf("answer %v, error %v; want ErrShape", answer, err)
	}
	answer, err := Decode(q, Row{Logits: []float32{1, -3e38}, K: 2, Act: []float32{0.5}}, "", temps)
	if err != nil {
		t.Fatal(err)
	}
	if ps := answer.(*schema.ChoiceAnswer).Probabilities; ps[0].P != 1 || ps[1].P != 0 {
		t.Errorf("probabilities = %v, want 1 and 0", ps)
	}
}

func TestNewTemperaturesFalsyByOptionsIsEmpty(t *testing.T) {
	for _, v := range []any{false, 0, 0.0, "", []any{}, map[string]any{}, nil, json.Number("0"), json.Number("0.0")} {
		if _, err := NewTemperatures(nil, nil, map[string]any{"de": map[string]any{"temperature_by_options": v}}); err != nil {
			t.Errorf("%#v: %v", v, err)
		}
	}
}

func TestNewTemperaturesRefusesSharedPrimarySubtag(t *testing.T) {
	_, err := NewTemperatures(nil, nil, map[string]any{"PT": map[string]any{}, "pt-BR": map[string]any{}})
	if err == nil || !strings.Contains(err.Error(), `"PT"`) || !strings.Contains(err.Error(), `"pt-BR"`) {
		t.Fatalf("err = %v", err)
	}
}
