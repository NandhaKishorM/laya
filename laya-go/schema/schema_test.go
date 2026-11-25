package schema_test

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func TestTypeCodes(t *testing.T) {
	for typ, want := range map[schema.Type]int{schema.TypeChoice: 0, schema.TypeScore: 1, schema.TypeNoul: 2, "mood": -1, "": -1} {
		if got := typ.Code(); got != want || typ.Valid() != (want >= 0) {
			t.Errorf("%q: Code = %d, Valid = %v", typ, got, typ.Valid())
		}
	}
}

func TestRendering(t *testing.T) {
	cases := []struct {
		name string
		q    schema.Question
		want []string
	}{
		{"label alone", schema.MustQuestion(schema.ChoiceOf("p", "a", "b")), []string{"a", "b"}},
		{"described", schema.MustQuestion(schema.Choice("p", schema.Option{Label: "a", Criterion: "send money"})), []string{"a: send money"}},
		{"empty criterion is no description", schema.MustQuestion(schema.Choice("p", schema.Option{Label: "a", Criterion: ""})), []string{"a"}},
		{"zero and false are descriptions", schema.MustQuestion(schema.Choice("p",
			schema.Option{Label: "z", Criterion: 0}, schema.Option{Label: "f", Criterion: false})), []string{"z: 0", "f: false"}},
		{"structured criterion", schema.MustQuestion(schema.Choice("p",
			schema.Option{Label: "a", Criterion: map[string]any{"desc": "é", "n": 1.0}})), []string{`a: {"desc": "é", "n": 1.0}`}},
		{"ordered criterion keeps its order", schema.MustQuestion(schema.Choice("p",
			schema.Option{Label: "a", Criterion: lang.Object{{Key: "z", Value: 1}, {Key: "a", Value: []any{2, nil}}}})), []string{`a: {"z": 1, "a": [2, null]}`}},
		{"score", schema.MustQuestion(schema.Score("p", "low", 3, []any{1})), []string{"level 0: low", "level 1: 3", "level 2: [1]"}},
		{"score level empty string", schema.MustQuestion(schema.Score("p", "")), []string{"level 0: "}},
		{"noul defaults", schema.MustQuestion(schema.Noul("p")), []string{"false: no, the statement does not hold", "true: yes, the statement holds"}},
		{"noul criteria and labels", schema.MustQuestion(schema.NoulWith("p", schema.NoulSpec{
			True: "it does", False: map[string]any{"k": 0}, TrueLabel: " yep ", FalseLabel: "nope"})), []string{`nope: {"k": 0}`, "yep: it does"}},
		{"noul zero description", schema.MustQuestion(schema.NoulWith("p", schema.NoulSpec{True: 0})), []string{"false: no, the statement does not hold", "true: 0"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.q.RenderedOptions(); !slices.Equal(got, c.want) {
				t.Errorf("RenderedOptions = %q, want %q", got, c.want)
			}
		})
	}
}

func TestQuestionIsASnapshot(t *testing.T) {
	crit := map[string]any{"k": "before"}
	levels := []any{"low"}
	labels := []string{"a", "b"}
	q := schema.MustQuestion(schema.Choice("p", schema.Option{Label: "x", Criterion: crit}))
	s := schema.MustQuestion(schema.Score("p", levels...))
	c := schema.MustQuestion(schema.ChoiceOf("p", labels...))
	crit["k"], levels[0], labels[0] = "after", "high", "zzz"
	if got := q.RenderedOptions()[0]; got != `x: {"k": "before"}` {
		t.Errorf("choice saw the caller's later change: %s", got)
	}
	if got := s.Legend()[0]; got != "low" {
		t.Errorf("score saw the caller's later change: %s", got)
	}
	if got := c.Labels()[0]; got != "a" {
		t.Errorf("labels saw the caller's later change: %s", got)
	}
	got := q.Labels()
	got[0] = "mutated"
	if q.Labels()[0] != "x" {
		t.Error("Labels returns the question's own slice")
	}
	opts := q.RenderedOptions()
	opts[0] = "mutated"
	if q.RenderedOptions()[0] == "mutated" {
		t.Error("RenderedOptions returns the question's own slice")
	}
}

func TestWithOptionOrder(t *testing.T) {
	q := schema.MustQuestion(schema.ChoiceOf("p", "a", "b", "c"))
	ordered, err := q.WithOptionOrder([]int{2, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if q.OptionOrder() != nil {
		t.Error("WithOptionOrder changed the receiver")
	}
	if got := ordered.OptionOrder(); !slices.Equal(got, []int{2, 0, 1}) {
		t.Errorf("OptionOrder = %v", got)
	}
	if !slices.Equal(ordered.Labels(), q.Labels()) || !slices.Equal(ordered.RenderedOptions(), q.RenderedOptions()) {
		t.Error("an option order changed the labels or the rendered text; it may only change the presentation")
	}
	order := ordered.OptionOrder()
	order[0] = 9
	if ordered.OptionOrder()[0] != 2 {
		t.Error("OptionOrder returns the question's own slice")
	}
	in := []int{1, 2, 0}
	again, _ := q.WithOptionOrder(in)
	in[0] = 0
	if again.OptionOrder()[0] != 1 {
		t.Error("WithOptionOrder kept the caller's slice")
	}
	back, err := ordered.WithOptionOrder(nil)
	if err != nil || back.OptionOrder() != nil {
		t.Errorf("nil order = %v, %v", back.OptionOrder(), err)
	}
	if _, err := (schema.Question{}).WithOptionOrder([]int{0}); err == nil {
		t.Error("the zero Question accepted an order")
	}
}

func TestZeroQuestion(t *testing.T) {
	var q schema.Question
	if q.Type().Valid() || q.NumOptions() != 0 || q.Labels() != nil || q.RenderedOptions() != nil || q.Legend() != nil {
		t.Errorf("the zero Question is not inert: %+v", q)
	}
	if _, err := q.MarshalJSON(); err == nil {
		t.Error("the zero Question marshaled")
	}
	if _, err := schema.NewQuestions(schema.Ask("x", q)); !errors.Is(err, schema.ErrInvalidQuestion) {
		t.Errorf("NewQuestions accepted the zero Question: %v", err)
	}
}

func TestMustQuestionPanics(t *testing.T) {
	defer func() {
		r := recover()
		if err, ok := r.(error); !ok || !errors.Is(err, schema.ErrInvalidQuestion) {
			t.Errorf("recovered %v, want the QuestionError", r)
		}
	}()
	schema.MustQuestion(schema.Choice(""))
}

func TestQuestionErrorText(t *testing.T) {
	cases := []struct {
		e    schema.QuestionError
		want string
	}{
		{schema.QuestionError{ID: "team", Reason: "r"}, "question 'team': r"},
		{schema.QuestionError{ID: "it's", Reason: "r"}, `question "it's": r`},
		{schema.QuestionError{ID: "a\tb", Reason: "r"}, `question 'a\tb': r`},
		{schema.QuestionError{Reason: "r"}, "r"},
	}
	for _, c := range cases {
		if got := c.e.Error(); got != c.want {
			t.Errorf("Error() = %q, want %q", got, c.want)
		}
	}
}

func TestInvalidUTF8IsRefused(t *testing.T) {
	bad := "x\xff"
	for name, err := range map[string]error{
		"instructions": second(schema.Choice(bad, schema.Option{Label: "a"})),
		"label":        second(schema.ChoiceOf("p", bad)),
		"criterion":    second(schema.Choice("p", schema.Option{Label: "a", Criterion: bad})),
		"level":        second(schema.Score("p", bad)),
		"noul label":   second(schema.NoulWith("p", schema.NoulSpec{TrueLabel: bad, FalseLabel: "f"})),
		"noul text":    second(schema.NoulWith("p", schema.NoulSpec{True: bad})),
		"nested":       second(schema.Choice("p", schema.Option{Label: "a", Criterion: []any{bad}})),
	} {
		if !errors.Is(err, schema.ErrInvalidQuestion) {
			t.Errorf("%s: err = %v, want a refusal", name, err)
		}
	}
}

func second(_ schema.Question, err error) error { return err }

func TestUnsupportedCriterionIsRefused(t *testing.T) {
	type thing struct{ N int }
	for name, v := range map[string]any{"struct": thing{1}, "channel": make(chan int), "bytes": []byte("x"), "int map": map[int]int{1: 2}} {
		if err := second(schema.Choice("p", schema.Option{Label: "a", Criterion: v})); !errors.Is(err, schema.ErrInvalidQuestion) {
			t.Errorf("%s: err = %v", name, err)
		}
		if _, err := schema.RenderCriterion(v); err == nil {
			t.Errorf("%s: RenderCriterion accepted it", name)
		}
	}
}

func TestRenderCriterion(t *testing.T) {
	cases := []struct {
		in   any
		want string
	}{
		{"as is {}", "as is {}"}, {nil, "null"}, {true, "true"}, {0, "0"}, {2.5, "2.5"}, {1e16, "1e+16"},
		{[]any{1, "a"}, `[1, "a"]`}, {map[string]any{"b": 1, "a": nil}, `{"a": null, "b": 1}`},
		{json.Number("12345678901234567890123"), "12345678901234567890123"}, {json.Number("1.50"), "1.5"},
	}
	for _, c := range cases {
		if got, err := schema.RenderCriterion(c.in); err != nil || got != c.want {
			t.Errorf("RenderCriterion(%#v) = %q, %v; want %q", c.in, got, err, c.want)
		}
	}
}

func TestQuestionsSet(t *testing.T) {
	team := schema.MustQuestion(schema.ChoiceOf("Which team?", "billing", "tech"))
	refund := schema.MustQuestion(schema.Noul("Refund?"))
	qs, err := schema.NewQuestions(schema.Ask("team", team), schema.Ask("refund", refund))
	if err != nil {
		t.Fatal(err)
	}
	if qs.Len() != 2 || !slices.Equal(qs.IDs(), []string{"team", "refund"}) || qs.At(1).ID != "refund" {
		t.Errorf("set = %v", qs.IDs())
	}
	if q, ok := qs.Get("refund"); !ok || q.Type() != schema.TypeNoul {
		t.Errorf("Get(refund) = %v, %v", q.Type(), ok)
	}
	if _, ok := qs.Get("nope"); ok {
		t.Error("Get found a question that is not there")
	}
	if got := qs.List(); len(got) != 2 || got[0].Type() != schema.TypeChoice {
		t.Errorf("List = %v", got)
	}
	var zero schema.Questions
	if _, ok := zero.Get("x"); ok || zero.Len() != 0 || len(zero.IDs()) != 0 {
		t.Error("the zero Questions is not empty")
	}
	if out, err := json.Marshal(zero); err != nil || string(out) != "{}" {
		t.Errorf("zero marshals to %s, %v", out, err)
	}

	out, err := json.Marshal(qs)
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"team":{"type":"choice","instructions":"Which team?","criteria":["billing","tech"]},` +
		`"refund":{"type":"noul","instructions":"Refund?"}}`
	if string(out) != want {
		t.Errorf("marshal\n got %s\nwant %s", out, want)
	}
	var back schema.Questions
	if err := json.Unmarshal(out, &back); err != nil || !slices.Equal(back.IDs(), []string{"team", "refund"}) {
		t.Errorf("round trip = %v, %v", back.IDs(), err)
	}
}

func TestNewQuestionsRefuses(t *testing.T) {
	q := schema.MustQuestion(schema.Noul("x"))
	cases := map[string]struct {
		entries []schema.Entry
		want    string
	}{
		"empty id":    {[]schema.Entry{schema.Ask("", q)}, "question id must be a non-empty string, got ''"},
		"blank id":    {[]schema.Entry{schema.Ask("  ", q)}, "got ' \\xa0'"},
		"repeated id": {[]schema.Entry{schema.Ask("a", q), schema.Ask("a", q)}, "question 'a': id is given twice"},
		"zero":        {[]schema.Entry{schema.Ask("a", schema.Question{})}, "question 'a': zero Question"},
	}
	for name, c := range cases {
		_, err := schema.NewQuestions(c.entries...)
		if err == nil || !strings.Contains(err.Error(), c.want) || !errors.Is(err, schema.ErrInvalidQuestion) {
			t.Errorf("%s: err = %v, want %q", name, err, c.want)
		}
	}
}

func TestParseQuestionsRefuses(t *testing.T) {
	cases := map[string]struct{ in, want string }{
		"repeated id, last is null": {`{"a":{"type":"noul","instructions":"x"},"a":null}`, "question 'a': definition must be a dict, got NoneType"},
		"not an object":             {`[]`, "must be a JSON object, got list"},
		"null":                      {`null`, "must be a JSON object, got NoneType"},
		"trailing data":             {`{} {}`, "data after the JSON value"},
		"truncated":                 {`{"a":`, "unexpected end"},
		"lone surrogate":            {`{"a\ud800":{}}`, "lone surrogate"},
		"blank id":                  {`{" ": {"type":"noul","instructions":"x"}}`, "got ' '"},
		"first bad question":        {`{"a":{"type":"noul","instructions":"x"},"b":{"type":"zzz"}}`, "question 'b': unknown type 'zzz'"},
		"invalid UTF-8":             {"{\"a\":\"\xff\"}", "UTF-8"},
		"too deep":                  {`{"a":` + strings.Repeat("[", 600) + strings.Repeat("]", 600) + `}`, "nested deeper"},
	}
	for name, c := range cases {
		_, err := schema.ParseQuestions([]byte(c.in))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", name, err, c.want)
		}
	}
	if qs, err := schema.ParseQuestions([]byte(` {} `)); err != nil || qs.Len() != 0 {
		t.Errorf("an empty object = %v, %v", qs.Len(), err)
	}
}

func TestParseQuestionsRepeatedIDTakesTheLast(t *testing.T) {
	// json.loads keeps the first position and the last value; so do we, and Python validates what is left.
	qs, err := schema.ParseQuestions([]byte(`{"a":{"type":"noul","instructions":"first"},"b":{"type":"noul","instructions":"b"},"a":{"type":"noul","instructions":"last"}}`))
	if err != nil || !slices.Equal(qs.IDs(), []string{"a", "b"}) {
		t.Fatalf("ids = %v, %v", qs.IDs(), err)
	}
	if q, _ := qs.Get("a"); q.Instructions() != "last" {
		t.Errorf("a = %q, want the last value", q.Instructions())
	}
	if _, err := schema.ParseQuestions([]byte(`{"a":{"type":"noul"},"a":{"type":"noul","instructions":"ok"}}`)); err != nil {
		t.Errorf("the first, invalid value was validated: %v", err)
	}
}

func TestParseQuestionsAcceptsPythonConstants(t *testing.T) {
	qs, err := schema.ParseQuestions([]byte(`{"q":{"type":"choice","instructions":"x","criteria":[Infinity,-Infinity,NaN,1e400]}}`))
	if err == nil {
		t.Fatalf("1e400 and Infinity are one key in Python; got %v", qs.IDs())
	}
	if !strings.Contains(err.Error(), "repeats label 0") {
		t.Errorf("err = %v", err)
	}
	qs, err = schema.ParseQuestions([]byte(`{"q":{"type":"choice","instructions":"x","criteria":[Infinity,-Infinity,NaN]}}`))
	if err != nil {
		t.Fatal(err)
	}
	q, _ := qs.Get("q")
	if got := q.Labels(); !slices.Equal(got, []string{"inf", "-inf", "nan"}) {
		t.Errorf("Labels = %q", got)
	}
}

func TestParseQuestionsKeepsOrder(t *testing.T) {
	qs, err := schema.ParseQuestions([]byte(`{"z":{"type":"noul","instructions":"1"},"a":{"type":"noul","instructions":"2"},"m":{"type":"noul","instructions":"3"}}`))
	if err != nil || !slices.Equal(qs.IDs(), []string{"z", "a", "m"}) {
		t.Errorf("ids = %v, %v", qs.IDs(), err)
	}
	// a repeated key inside a definition is Python's dict: first position, last value
	q, err := schema.ParseQuestions([]byte(`{"a":{"type":"choice","instructions":"x","criteria":{"p":"1","q":"2","p":"3"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	one, _ := q.Get("a")
	if got := one.RenderedOptions(); !slices.Equal(got, []string{"p: 3", "q: 2"}) {
		t.Errorf("options = %q", got)
	}
}

func TestQuestionJSON(t *testing.T) {
	var q schema.Question
	if err := json.Unmarshal([]byte(`{"type":"score","instructions":["a",1],"criteria":["x",{"k":2}],"option_order":[1,0]}`), &q); err != nil {
		t.Fatal(err)
	}
	if q.Instructions() != `["a", 1]` || !slices.Equal(q.Legend(), []string{"x", `{"k": 2}`}) || !slices.Equal(q.OptionOrder(), []int{1, 0}) {
		t.Errorf("q = %q %q %v", q.Instructions(), q.Legend(), q.OptionOrder())
	}
	err := json.Unmarshal([]byte(`{"type":"choice","instructions":"x"}`), &q)
	if err == nil || !strings.Contains(err.Error(), "takes 'criteria' as a dict") || strings.Contains(err.Error(), "question '") {
		t.Errorf("err = %v; a bare question has no id to name", err)
	}
	if err := json.Unmarshal([]byte(`{"type":`), &q); err == nil {
		t.Error("truncated JSON accepted")
	}
	if q.Type() != schema.TypeScore {
		t.Error("a failed Unmarshal changed the question")
	}
}

func TestQuestionMarshalForms(t *testing.T) {
	cases := []struct {
		name string
		q    schema.Question
		want string
	}{
		{"undescribed choice is the list form", schema.MustQuestion(schema.ChoiceOf("p", "a", "b")),
			`{"type":"choice","instructions":"p","criteria":["a","b"]}`},
		{"described choice is the dict form", schema.MustQuestion(schema.Choice("p", schema.Option{Label: "a", Criterion: 1.0}, schema.Option{Label: "b"})),
			`{"type":"choice","instructions":"p","criteria":{"a":1.0,"b":null}}`},
		{"score", schema.MustQuestion(schema.Score("p", "x", map[string]any{"k": 1})),
			`{"type":"score","instructions":"p","criteria":["x",{"k": 1}]}`},
		{"noul with sides and labels and order", mustOrder(t, schema.MustQuestion(schema.NoulWith("p", schema.NoulSpec{
			True: "t", FalseLabel: "no", TrueLabel: "yes"})), []int{1, 0}),
			`{"type":"noul","instructions":"p","criteria":{"true":"t"},"labels":{"false":"no","true":"yes"},"option_order":[1,0]}`},
		{"escaping", schema.MustQuestion(schema.ChoiceOf("a\"b\n<é>", "x")),
			`{"type":"choice","instructions":"a\"b\n<é>","criteria":["x"]}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := c.q.MarshalJSON()
			if err != nil || string(got) != c.want {
				t.Errorf("MarshalJSON = %s, %v\n want %s", got, err, c.want)
			}
			if !json.Valid(got) {
				t.Error("not valid JSON")
			}
		})
	}
}

func mustOrder(t *testing.T, q schema.Question, order []int) schema.Question {
	t.Helper()
	q, err := q.WithOptionOrder(order)
	if err != nil {
		t.Fatal(err)
	}
	return q
}

func TestAnswersJSON(t *testing.T) {
	common := schema.Common{Confidence: 0.5, AnswerConfidence: 1, Action: schema.Action{ActProbability: 0.0001}}
	choice := &schema.ChoiceAnswer{Choice: schema.Text("b"), Probabilities: schema.Probabilities{{Label: schema.Text("a"), P: 0.25}, {Label: schema.Text("b"), P: 0.75}}, Common: common}
	score := &schema.ScoreAnswer{Score: 1.5, Legend: []string{"lo", "hi"}, Probabilities: schema.Probabilities{{Label: schema.Text("0"), P: 0.5}, {Label: schema.Text("1"), P: 0.5}}, Common: common}
	noul := &schema.NoulAnswer{Noul: 0.125, Common: schema.Common{Confidence: 0.875, AnswerConfidence: 0.875, Action: schema.Action{ActProbability: 1e-05}}}

	want := map[string]string{
		"choice": `{"type":"choice","choice":"b","probabilities":{"a":0.25,"b":0.75},"confidence":0.5,"answer_confidence":1.0,"action":{"act_probability":0.0001}}`,
		"score":  `{"type":"score","score":1.5,"legend":{"0":"lo","1":"hi"},"probabilities":{"0":0.5,"1":0.5},"confidence":0.5,"answer_confidence":1.0,"action":{"act_probability":0.0001}}`,
		"noul":   `{"type":"noul","noul":0.125,"confidence":0.875,"answer_confidence":0.875,"action":{"act_probability":1e-05}}`,
	}
	for name, a := range map[string]schema.Answer{"choice": choice, "score": score, "noul": noul} {
		t.Run(name, func(t *testing.T) {
			got, err := a.(json.Marshaler).MarshalJSON()
			if err != nil || string(got) != want[name] {
				t.Errorf("marshal\n got %s\nwant %s (%v)", got, want[name], err)
			}
			back, err := schema.UnmarshalAnswer(got)
			if err != nil {
				t.Fatal(err)
			}
			if back.Type() != a.Type() || back.Base() != a.Base() {
				t.Errorf("round trip: %v %v", back.Type(), back.Base())
			}
			again, _ := back.(json.Marshaler).MarshalJSON()
			if string(again) != string(got) {
				t.Errorf("round trip changed the JSON:\n%s\n%s", got, again)
			}
		})
	}
	if v, ok := choice.Probabilities.Get(schema.Text("b")); !ok || v != 0.75 {
		t.Errorf("Get(b) = %v, %v", v, ok)
	}
	if _, ok := choice.Probabilities.Get(schema.Text("z")); ok {
		t.Error("Get found a label that is not there")
	}

	answers := schema.Answers{{ID: "z", Answer: noul}, {ID: "a<b", Answer: choice}}
	out, err := json.Marshal(answers)
	if err != nil {
		t.Fatal(err)
	}
	var back schema.Answers
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatal(err)
	}
	if len(back) != 2 || back[0].ID != "z" || back[1].ID != "a<b" {
		t.Errorf("answers lost their order: %+v", back)
	}
	if a, ok := back.Get("a<b"); !ok || a.(*schema.ChoiceAnswer).Choice != schema.Text("b") {
		t.Errorf("Get = %v, %v", a, ok)
	}
	if _, ok := back.Get("nope"); ok {
		t.Error("Get found an id that is not there")
	}
}

func TestAnswerMarshalRefuses(t *testing.T) {
	bad := map[string]json.Marshaler{
		"NaN":       &schema.NoulAnswer{Noul: math.NaN()},
		"Inf":       &schema.ScoreAnswer{Score: math.Inf(1)},
		"bad UTF-8": &schema.ChoiceAnswer{Choice: schema.Text("\xff")},
		"bad act":   &schema.NoulAnswer{Common: schema.Common{Action: schema.Action{ActProbability: math.NaN()}}},
		"bad prob":  &schema.ChoiceAnswer{Choice: schema.Text("a"), Probabilities: schema.Probabilities{{Label: schema.Text("a"), P: math.Inf(-1)}}},
	}
	for name, m := range bad {
		if _, err := m.MarshalJSON(); err == nil {
			t.Errorf("%s: marshaled", name)
		}
	}
	if _, err := json.Marshal(schema.Answers{{ID: "x"}}); err == nil {
		t.Error("a nil answer marshaled")
	}
}

func TestUnmarshalAnswerRefuses(t *testing.T) {
	const tail = `,"confidence":0.5,"answer_confidence":0.5,"action":{"act_probability":0.5}}`
	cases := map[string]string{
		"not an object":      `[]`,
		"no type":            `{"noul":0.5` + tail,
		"unknown type":       `{"type":"mood","noul":0.5` + tail,
		"type not a string":  `{"type":5}`,
		"no confidence":      `{"type":"noul","noul":0.5,"answer_confidence":0.5,"action":{"act_probability":0.5}}`,
		"no action":          `{"type":"noul","noul":0.5,"confidence":0.5,"answer_confidence":0.5}`,
		"action not object":  `{"type":"noul","noul":0.5,"confidence":0.5,"answer_confidence":0.5,"action":1}`,
		"no noul":            `{"type":"noul"` + tail,
		"string noul":        `{"type":"noul","noul":"0.5"` + tail,
		"no choice":          `{"type":"choice","probabilities":{}` + tail,
		"no probabilities":   `{"type":"choice","choice":"a"` + tail,
		"probabilities list": `{"type":"choice","choice":"a","probabilities":[]` + tail,
		"probability string": `{"type":"choice","choice":"a","probabilities":{"a":"x"}` + tail,
		"no legend":          `{"type":"score","score":1,"probabilities":{}` + tail,
		"legend not object":  `{"type":"score","score":1,"legend":[],"probabilities":{}` + tail,
		"legend gap":         `{"type":"score","score":1,"legend":{"1":"x"},"probabilities":{}` + tail,
		"legend not text":    `{"type":"score","score":1,"legend":{"0":5},"probabilities":{}` + tail,
		"not JSON":           `{`,
	}
	for name, in := range cases {
		if _, err := schema.UnmarshalAnswer([]byte(in)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	var as schema.Answers
	for name, in := range map[string]string{"list": `[]`, "bad member": `{"a":[]}`, "not JSON": `{`} {
		if err := json.Unmarshal([]byte(in), &as); err == nil {
			t.Errorf("Answers %s: accepted", name)
		}
	}
	var ps schema.Probabilities
	if err := json.Unmarshal([]byte(`{"b":0.25,"a":0.5}`), &ps); err != nil || ps[0].Label != schema.Text("b") {
		t.Errorf("Probabilities = %v, %v", ps, err)
	}
	if err := json.Unmarshal([]byte(`[1]`), &ps); err == nil {
		t.Error("Probabilities accepted a list")
	}
}

func TestPredictionJSON(t *testing.T) {
	n := 4
	p := schema.Prediction{
		Model: schema.Model,
		Answers: schema.Answers{{ID: "r", Answer: &schema.NoulAnswer{Noul: 0.5,
			Common: schema.Common{Confidence: 0.5, AnswerConfidence: 0.5}}}},
		Usage: schema.Usage{InputTokens: 10, StateTokens: 7, Truncated: false,
			CollapsedOptions: schema.CollapsedQuestions{
				{ID: "z", CollapsedOptions: schema.CollapsedOptions{Total: 5, Distinct: 1, TokensPerOption: &n}},
				{ID: "a", CollapsedOptions: schema.CollapsedOptions{Total: 3, Distinct: 2}},
			}},
	}
	out, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"model":"laya-rl-agent-onnx","answers":{"r":{"type":"noul","noul":0.5,"confidence":0.5,"answer_confidence":0.5,"action":{"act_probability":0.0}}},` +
		`"usage":{"input_tokens":10,"output_tokens":0,"state_tokens":7,"state_tokens_dropped":0,"truncated":false,"truncated_questions":[],` +
		`"options":{"z":{"total":5,"distinct":1,"tokens_per_option":4},"a":{"total":3,"distinct":2,"tokens_per_option":null}}}}`
	if string(out) != want {
		t.Errorf("marshal\n got %s\nwant %s", out, want)
	}
	var back schema.Prediction
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatal(err)
	}
	if got := back.Usage.CollapsedOptions; len(got) != 2 || got[0].ID != "z" || got[1].ID != "a" ||
		got[0].TokensPerOption == nil || *got[0].TokensPerOption != 4 || got[1].TokensPerOption != nil {
		t.Errorf("collapsed options lost their order or values: %+v", got)
	}
	if c, ok := back.Usage.CollapsedOptions.Get("a"); !ok || c.Total != 3 {
		t.Errorf("Get(a) = %+v, %v", c, ok)
	}
	if _, ok := back.Usage.CollapsedOptions.Get("q"); ok {
		t.Error("Get found an id that is not there")
	}

	// No collapse: the key is absent, as Python omits it, and the cost report is still a list.
	out, _ = json.Marshal(schema.Usage{TruncatedQuestions: []string{"r"}, Truncated: true})
	if strings.Contains(string(out), "options") || !strings.Contains(string(out), `"truncated_questions":["r"]`) {
		t.Errorf("usage = %s", out)
	}
	var cs schema.CollapsedQuestions
	for _, in := range []string{`[]`, `{"a":[]}`, `{`} {
		if err := json.Unmarshal([]byte(in), &cs); err == nil {
			t.Errorf("CollapsedQuestions accepted %s", in)
		}
	}
}

func TestConcurrentReads(t *testing.T) {
	qs, err := schema.ParseQuestions([]byte(`{"a":{"type":"choice","instructions":"x","criteria":{"p":{"k":[1,2]},"q":"2"}},"b":{"type":"score","instructions":"y","criteria":["l","h"]}}`))
	if err != nil {
		t.Fatal(err)
	}
	want, _ := json.Marshal(qs)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				q, _ := qs.Get("a")
				o, _ := q.WithOptionOrder([]int{1, 0})
				got, _ := json.Marshal(qs)
				if string(got) != string(want) || o.RenderedOptions()[0] != `p: {"k": [1, 2]}` {
					t.Error("a concurrent read disagreed")
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestScanUsageAndWindowJSON(t *testing.T) {
	scan := schema.Usage{InputTokens: 9, StateTokens: 4, Truncated: true, TruncatedWindows: 3,
		TruncatedQuestions: []string{}, Windows: 5}
	out, err := json.Marshal(scan)
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"input_tokens":9,"output_tokens":0,"state_tokens":4,"state_tokens_dropped":0,"truncated":3,"truncated_questions":[],"windows":5}`
	if string(out) != want {
		t.Errorf("scan usage\n got %s\nwant %s", out, want)
	}
	// One window keeps the flag; Python's single call writes a bool and then adds windows.
	one, _ := json.Marshal(schema.Usage{Truncated: true, TruncatedQuestions: []string{"q"}, Windows: 1})
	if !strings.Contains(string(one), `"truncated":true`) || !strings.HasSuffix(string(one), `"windows":1}`) {
		t.Errorf("single-window usage = %s", one)
	}
	var back schema.Usage
	if err := json.Unmarshal(out, &back); err != nil || back.TruncatedWindows != 3 || !back.Truncated || back.Windows != 5 {
		t.Errorf("round trip = %+v, %v", back, err)
	}
	if err := json.Unmarshal(one, &back); err != nil || back.TruncatedWindows != 0 || !back.Truncated || back.Windows != 1 {
		t.Errorf("flag round trip = %+v, %v", back, err)
	}
	for _, bad := range []string{`{"truncated":"yes"}`, `{"truncated":-1}`, `{"truncated":1.5}`} {
		if err := json.Unmarshal([]byte(bad), &back); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}

	a := &schema.NoulAnswer{Noul: 0.5, Common: schema.Common{Confidence: 0.5, AnswerConfidence: 0.5,
		Window: &schema.Window{Index: 1, TokenStart: 380, TokenEnd: 1140, Count: 3}}}
	b, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	const wantAnswer = `{"type":"noul","noul":0.5,"confidence":0.5,"answer_confidence":0.5,"action":{"act_probability":0.0},` +
		`"window":{"index":1,"token_start":380,"token_end":1140,"count":3}}`
	if string(b) != wantAnswer {
		t.Errorf("answer\n got %s\nwant %s", b, wantAnswer)
	}
	got, err := schema.UnmarshalAnswer(b)
	if err != nil || !reflect.DeepEqual(got.Base().Window, a.Window) {
		t.Errorf("window round trip = %+v, %v", got, err)
	}
	for _, bad := range []string{`"window":3`, `"window":{"index":1.5,"token_start":0,"token_end":1,"count":1}`, `"window":{"index":1}`} {
		if _, err := schema.UnmarshalAnswer([]byte(strings.Replace(wantAnswer, `"window":{"index":1,"token_start":380,"token_end":1140,"count":3}`, bad, 1))); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
}

func TestUsageTruncatedMatchesWindows(t *testing.T) {
	for _, bad := range []string{`{"truncated":true,"windows":3}`, `{"truncated":2}`, `{"truncated":2,"windows":1}`} {
		var u schema.Usage
		if err := json.Unmarshal([]byte(bad), &u); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
	for _, ok := range []string{`{"truncated":2,"windows":3}`, `{"truncated":true,"windows":1}`, `{"input_tokens":0,"output_tokens":0,"windows":3}`} {
		var u schema.Usage
		if err := json.Unmarshal([]byte(ok), &u); err != nil {
			t.Errorf("%s: %v", ok, err)
		}
	}
}

func TestUsageOfScanWithoutQuestionsIsBare(t *testing.T) {
	const want = `{"input_tokens":0,"output_tokens":0,"windows":3}`
	b, err := json.Marshal(schema.Usage{TruncatedQuestions: []string{}, Windows: 3})
	if err != nil || string(b) != want {
		t.Fatalf("got %s, %v", b, err)
	}
	var u schema.Usage
	if err := json.Unmarshal(b, &u); err != nil {
		t.Fatal(err)
	}
	if again, _ := json.Marshal(u); string(again) != want {
		t.Errorf("round trip %s", again)
	}
	full, _ := json.Marshal(schema.Usage{StateTokens: 1, TruncatedQuestions: []string{}, Windows: 3})
	if !strings.Contains(string(full), `"truncated":0`) {
		t.Errorf("a scan that read state lost its keys: %s", full)
	}
}

func TestWindowRange(t *testing.T) {
	const base = `{"type":"noul","noul":0.5,"confidence":0.5,"answer_confidence":0.5,"action":{"act_probability":0.0},"window":`
	for _, bad := range []string{
		`{"index":3,"token_start":0,"token_end":1,"count":3}`,
		`{"index":-1,"token_start":0,"token_end":1,"count":3}`,
		`{"index":0,"token_start":-1,"token_end":1,"count":3}`,
		`{"index":0,"token_start":5,"token_end":4,"count":3}`,
		`{"index":0,"token_start":0,"token_end":1,"count":0}`,
	} {
		if _, err := schema.UnmarshalAnswer([]byte(base + bad + `}`)); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
	if _, err := schema.UnmarshalAnswer([]byte(base + `{"index":2,"token_start":4,"token_end":4,"count":3}}`)); err != nil {
		t.Error(err)
	}
}
