package schema_test

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/goldens"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func questionCases(t *testing.T) []fixtures.Object {
	t.Helper()
	root := fixtures.Load(t, "go_extra.json")
	list := fixtures.List(t, fixtures.Obj(t, fixtures.Field(t, root, "questions")).Must("cases"))
	if len(list) != 140 {
		t.Fatalf("the questions golden holds %d cases", len(list))
	}
	out := make([]fixtures.Object, len(list))
	for i, c := range list {
		out[i] = fixtures.Obj(t, c)
	}
	return out
}

// TestQuestionsGolden feeds each definition to ParseQuestions and holds it to Agent._check_question:
// a refusal must carry Python's exact text, and an accepted question must render what
// render_options renders, with the same labels, legend, instructions and option order.
func TestQuestionsGolden(t *testing.T) {
	refused, accepted := 0, 0
	for _, rec := range questionCases(t) {
		t.Run(fixtures.Str(t, rec.Must("case")), func(t *testing.T) {
			id := fixtures.Str(t, rec.Must("id"))
			doc := fixtures.Marshal(t, fixtures.Object{{Key: id, Value: rec.Must("definition")}})
			qs, err := schema.ParseQuestions(doc)

			if want, isRefusal := rec.Get("error"); isRefusal {
				refused++
				text := fixtures.Str(t, want)
				if err == nil {
					t.Fatalf("accepted; Python refuses with %q", text)
				}
				if err.Error() != text {
					t.Errorf("error text\n got %q\nwant %q", err.Error(), text)
				}
				var qe *schema.QuestionError
				if !errors.As(err, &qe) || !errors.Is(err, schema.ErrInvalidQuestion) {
					t.Fatalf("err = %T %v, want a *QuestionError wrapping ErrInvalidQuestion", err, err)
				}
				if reason := strings.TrimPrefix(text, "question "); qe.ID != "" && !strings.HasSuffix(text, qe.Reason) {
					t.Errorf("Reason %q is not the tail of %q", qe.Reason, reason)
				}
				return
			}
			accepted++
			if err != nil {
				t.Fatalf("refused: %v", err)
			}
			q, ok := qs.Get(id)
			if !ok || qs.Len() != 1 {
				t.Fatalf("Get(%q) = %v, %v with %d questions", id, q, ok, qs.Len())
			}
			checkQuestion(t, q, rec)

			// The same question through the typed constructors, from the decoded definition.
			if built, ok := viaConstructors(t, rec); ok {
				checkQuestion(t, built, rec)
			}

			// And back through the wire form: marshaling is stable and loses nothing.
			out, err := json.Marshal(qs)
			if err != nil {
				t.Fatal(err)
			}
			again, err := schema.ParseQuestions(out)
			if err != nil {
				t.Fatalf("own output %s: %v", out, err)
			}
			q2, _ := again.Get(id)
			checkQuestion(t, q2, rec)
			if out2, _ := json.Marshal(again); string(out) != string(out2) {
				t.Errorf("marshal is not stable:\n%s\n%s", out, out2)
			}
		})
	}
	if refused != 93 || accepted != 47 {
		t.Errorf("%d refusals and %d acceptances; the golden was 93 and 47", refused, accepted)
	}
}

func checkQuestion(t *testing.T, q schema.Question, rec fixtures.Object) {
	t.Helper()
	if got, want := string(q.Type()), fixtures.Str(t, rec.Must("type")); got != want {
		t.Errorf("Type = %s, want %s", got, want)
	}
	if got, want := q.Instructions(), fixtures.Str(t, rec.Must("instructions")); got != want {
		t.Errorf("Instructions = %q, want %q", got, want)
	}
	if got, want := q.RenderedOptions(), fixtures.Strings(t, rec.Must("options")); !slices.Equal(got, want) {
		t.Errorf("RenderedOptions\n got %q\nwant %q", got, want)
	}
	if got, want := q.Labels(), fixtures.Strings(t, rec.Must("labels")); !slices.Equal(got, want) {
		t.Errorf("Labels = %q, want %q", got, want)
	}
	if q.NumOptions() != len(q.Labels()) || q.NumOptions() != len(q.RenderedOptions()) {
		t.Errorf("NumOptions = %d for %d labels", q.NumOptions(), len(q.Labels()))
	}
	if want, isScore := rec.Get("legend"); isScore {
		if got := q.Legend(); !slices.Equal(got, fixtures.Strings(t, want)) {
			t.Errorf("Legend = %q, want %q", got, fixtures.Strings(t, want))
		}
	} else if q.Legend() != nil {
		t.Errorf("Legend = %q for a %s", q.Legend(), q.Type())
	}
	var wantOrder []int
	if o := rec.Must("option_order"); o != nil {
		for _, v := range fixtures.Ints(t, o) {
			wantOrder = append(wantOrder, int(v))
		}
	}
	if got := q.OptionOrder(); !slices.Equal(got, wantOrder) || (wantOrder == nil) != (got == nil) {
		t.Errorf("OptionOrder = %v, want %v", got, wantOrder)
	}
}

// viaConstructors rebuilds an accepted definition with Choice, Score or NoulWith, so the goldens
// check the typed path too. Definitions the typed API cannot say (a structured instruction, a
// repeated noul key) are skipped.
func viaConstructors(t *testing.T, rec fixtures.Object) (schema.Question, bool) {
	t.Helper()
	def := fixtures.Obj(t, rec.Must("definition"))
	ins, isStr := def.Must("instructions").(string)
	if !isStr {
		return schema.Question{}, false
	}
	crit, _ := def.Get("criteria")
	var (
		q   schema.Question
		err error
	)
	switch fixtures.Str(t, rec.Must("type")) {
	case "choice":
		var opts []schema.Option
		switch c := crit.(type) {
		case fixtures.Object:
			for _, m := range c {
				opts = append(opts, schema.Option{Label: m.Key, Criterion: goldens.Value(m.Value)})
			}
		case []any:
			for _, l := range c {
				if text, ok := l.(string); ok {
					opts = append(opts, schema.Option{Label: text})
				} else {
					opts = append(opts, schema.Option{LabelValue: l}) // a bool or a json.Number
				}
			}
		}
		q, err = schema.Choice(ins, opts...)
	case "score":
		var levels []any
		for _, l := range fixtures.List(t, crit) {
			levels = append(levels, goldens.Value(l))
		}
		q, err = schema.Score(ins, levels...)
	default:
		var spec schema.NoulSpec
		if c, ok := crit.(fixtures.Object); ok {
			for _, m := range c {
				switch strings.ToLower(m.Key) {
				case "true":
					spec.True = goldens.Value(m.Value)
				case "false":
					spec.False = goldens.Value(m.Value)
				}
			}
		}
		if _, custom := def.Get("labels"); custom {
			if l, ok := def.Must("labels").(fixtures.Object); ok {
				spec.FalseLabel, spec.TrueLabel = fixtures.Str(t, l.Must("false")), fixtures.Str(t, l.Must("true"))
			}
		}
		q, err = schema.NoulWith(ins, spec)
	}
	if err != nil {
		t.Fatalf("typed constructor refused an accepted definition: %v", err)
	}
	if o := rec.Must("option_order"); o != nil {
		var order []int
		for _, v := range fixtures.Ints(t, o) {
			order = append(order, int(v))
		}
		if q, err = q.WithOptionOrder(order); err != nil {
			t.Fatal(err)
		}
	}
	return q, true
}

// TestBuilderReasonsMatchPython checks that the typed constructors refuse with Python's words:
// each row names the golden case whose text must end the constructor's Reason.
func TestBuilderReasonsMatchPython(t *testing.T) {
	golden := map[string]string{}
	for _, rec := range questionCases(t) {
		if e, ok := rec.Get("error"); ok {
			golden[fixtures.Str(t, rec.Must("case"))] = strings.TrimPrefix(fixtures.Str(t, e), "question 'q': ")
		}
	}
	mustOrder := func(q schema.Question, order []int) error {
		_, err := q.WithOptionOrder(order)
		return err
	}
	two := schema.MustQuestion(schema.ChoiceOf("Pick.", "a", "b"))
	three := schema.MustQuestion(schema.Score("Rate.", "bad", "ok", "good"))
	noul := func(spec schema.NoulSpec) error { _, err := schema.NoulWith("Holds?", spec); return err }
	build := func(_ schema.Question, err error) error { return err }

	cases := []struct {
		name string
		gold string
		err  error
	}{
		{"blank choice instructions", "ins-blank", build(schema.ChoiceOf(" \t\n", "a", "b"))},
		{"blank score instructions", "ins-nbsp", build(schema.Score(" \u0085", "a"))},
		{"blank noul instructions", "ins-ideographic-space", build(schema.Noul("　"))},
		{"empty instructions", "ins-empty", build(schema.Choice("", schema.Option{Label: "a"}))},
		{"no options", "choice-empty-list", build(schema.Choice("Pick."))},
		{"repeated label", "choice-list-repeat", build(schema.ChoiceOf("Pick.", "a", "b", "a"))},
		{"no levels", "score-empty", build(schema.Score("Rate."))},
		{"null level", "score-null-second", build(schema.Score("Rate.", "a", nil, "c", nil))},
		{"labels the same", "labels-same", noul(schema.NoulSpec{FalseLabel: "a", TrueLabel: "a"})},
		{"labels the same after strip", "labels-same-after-strip", noul(schema.NoulSpec{FalseLabel: "a", TrueLabel: " a "})},
		{"one label only", "labels-missing-true", noul(schema.NoulSpec{FalseLabel: "a"})},
		{"blank label", "labels-blank-value", noul(schema.NoulSpec{FalseLabel: " ", TrueLabel: "x"})},
		{"order too short", "order-short", mustOrder(two, []int{0})},
		{"order too long", "order-long", mustOrder(two, []int{0, 1, 2})},
		{"order repeats", "order-repeat", mustOrder(two, []int{0, 0})},
		{"order out of range", "order-range", mustOrder(two, []int{0, 2})},
		{"order negative", "order-negative", mustOrder(two, []int{-1, 0})},
		{"order empty", "order-empty", mustOrder(two, []int{})},
		{"order for a score", "order-noul-wrong-size", mustOrder(schema.MustQuestion(schema.Noul("Holds?")), []int{0, 1, 2})},
	}
	_ = three
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var qe *schema.QuestionError
			if !errors.As(c.err, &qe) {
				t.Fatalf("err = %v, want a *QuestionError", c.err)
			}
			want, ok := golden[c.gold]
			if !ok {
				t.Fatalf("no refused golden case %q", c.gold)
			}
			if qe.Reason != want {
				t.Errorf("Reason\n got %q\nwant %q", qe.Reason, want)
			}
			if qe.ID != "" || qe.Error() != qe.Reason {
				t.Errorf("a constructor error has no id: %q", qe.Error())
			}
			if !errors.Is(c.err, schema.ErrInvalidQuestion) {
				t.Error("does not wrap ErrInvalidQuestion")
			}
		})
	}
}
