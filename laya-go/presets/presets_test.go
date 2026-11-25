package presets_test

import (
	"sort"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/presets"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// canonical is the wire form of what Python recorded, read through ParseQuestions and written
// back by MarshalJSON, so that both sides of a comparison are spelled by the same writer.
func canonical(t *testing.T, recorded any) string {
	t.Helper()
	qs, err := schema.ParseQuestions(fixtures.Marshal(t, recorded))
	if err != nil {
		t.Fatalf("recorded preset refused: %v", err)
	}
	return marshal(t, qs)
}

func marshal(t *testing.T, qs schema.Questions) string {
	t.Helper()
	out, err := qs.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func TestPresetsMatchPython(t *testing.T) {
	golden := fixtures.Load(t, "presets.json")
	var recorded []string
	for _, key := range golden.Keys() {
		if strings.HasSuffix(key, "_questions") {
			recorded = append(recorded, key)
		}
	}
	var ours []string
	for _, p := range presets.All() {
		ours = append(ours, p.Name)
		t.Run(p.Name, func(t *testing.T) {
			if got, want := marshal(t, p.Questions), canonical(t, golden.Must(p.Name)); got != want {
				t.Errorf("preset differs from Python\n got %s\nwant %s", got, want)
			}
		})
	}
	sort.Strings(recorded)
	sort.Strings(ours)
	if strings.Join(recorded, ",") != strings.Join(ours, ",") {
		t.Errorf("presets %v, Python has %v", ours, recorded)
	}
	if len(ours) != 5 {
		t.Errorf("%d presets, want 5", len(ours))
	}
}

func TestEmailCategories(t *testing.T) {
	golden := fixtures.Load(t, "presets.json")
	custom := []schema.Option{
		{Label: "ops", Criterion: "incidents and deploys"},
		{Label: "legal", Criterion: "contracts and compliance"},
	}
	if got, want := marshal(t, presets.EmailWith(custom)), canonical(t, golden.Must("email_questions_custom")); got != want {
		t.Errorf("custom categories\n got %s\nwant %s", got, want)
	}
	defaults := canonical(t, golden.Must("email_questions"))
	// An empty mapping is falsy in Python, so it asks for the defaults.
	for name, qs := range map[string]schema.Questions{
		"nil":   presets.EmailWith(nil),
		"empty": presets.EmailWith([]schema.Option{}),
		"none":  presets.Email(),
	} {
		if got := marshal(t, qs); got != defaults {
			t.Errorf("%s categories are not the defaults:\n%s", name, got)
		}
	}
	if want := canonical(t, golden.Must("email_questions_empty_categories")); want != defaults {
		t.Fatal("Python no longer reads empty categories as the defaults")
	}
}

func TestStateFieldMatchesPython(t *testing.T) {
	golden := fixtures.Load(t, "presets.json")
	field := func(v any) (string, bool) {
		s, isStr := v.(string)
		return s, isStr
	}
	for _, p := range presets.All() {
		want, wantOK := field(fixtures.Field(t, golden.Must("state_field"), p.Name))
		if got, ok := presets.StateField(p.Questions); got != want || ok != wantOK {
			t.Errorf("%s: StateField = %q, %v; Python %q, %v", p.Name, got, ok, want, wantOK)
		}
	}
	hostile := fixtures.Obj(t, golden.Must("state_field_hostile"))
	specs := fixtures.Obj(t, golden.Must("hostile_specs"))
	checked, unbuildable := 0, 0
	for _, name := range hostile.Keys() {
		want, wantOK := field(hostile.Must(name))
		qs, err := schema.ParseQuestions(fixtures.Marshal(t, specs.Must(name)))
		if err != nil {
			// A definition without instructions cannot be built here, so there is nothing to ask.
			if name != "missing-instructions" && name != "null-instructions" {
				t.Errorf("%s: spec refused: %v", name, err)
			}
			unbuildable++
			continue
		}
		checked++
		if got, ok := presets.StateField(qs); got != want || ok != wantOK {
			t.Errorf("%s: StateField = %q, %v; Python %q, %v", name, got, ok, want, wantOK)
		}
	}
	if checked < 7 || unbuildable != 2 {
		t.Errorf("hostile specs: %d checked, %d unbuildable; want at least 7 and exactly 2", checked, unbuildable)
	}
}

func TestStateFieldScansLikeFindall(t *testing.T) {
	for _, c := range []struct {
		ins, want string
		ok        bool
	}{
		{"read `a`b`c`", "", false},      // a and c: two fields
		{"read `a` then `b`", "", false}, // two fields
		{"read `a`b", "a", true},         // b has no closing backtick
		{"read `a`b`", "a", true},        // findall resumes after a, so b is not found
		{"read ``a``", "a", true},        // empty pair, then a
		{"read `café`", "café", true},    // Python \w admits letters beyond ASCII
		{"read `a b`", "", false},
		{"read `x` and `x`", "x", true},
		{"no field here", "", false},
	} {
		q, err := schema.Noul(c.ins)
		if err != nil {
			t.Fatal(err)
		}
		qs, err := schema.NewQuestions(schema.Ask("q", q))
		if err != nil {
			t.Fatal(err)
		}
		if got, ok := presets.StateField(qs); got != c.want || ok != c.ok {
			t.Errorf("%q: StateField = %q, %v; want %q, %v", c.ins, got, ok, c.want, c.ok)
		}
	}
}

func TestStateFieldReadsInstructionsOnly(t *testing.T) {
	q, err := schema.ChoiceOf("Pick one.", "look at `message`")
	if err != nil {
		t.Fatal(err)
	}
	withCriterion, err := schema.Choice("Pick one.", schema.Option{Label: "x", Criterion: "look at `message`"})
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []schema.Question{q, withCriterion} {
		qs, _ := schema.NewQuestions(schema.Ask("q", q))
		if got, ok := presets.StateField(qs); ok {
			t.Errorf("StateField read a criterion or label: %q", got)
		}
	}
	if _, ok := presets.StateField(schema.Questions{}); ok {
		t.Error("an empty set names a field")
	}
}

func TestEveryPresetNamesOneField(t *testing.T) {
	want := map[string]string{
		"triage_questions": "message", "email_questions": "body", "guard_questions": "prompt",
		"moderation_questions": "post", "router_questions": "request",
	}
	for _, p := range presets.All() {
		if got, ok := presets.StateField(p.Questions); !ok || got != want[p.Name] {
			t.Errorf("%s: %q, %v; want %q", p.Name, got, ok, want[p.Name])
		}
	}
}
