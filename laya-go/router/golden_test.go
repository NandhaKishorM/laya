package router_test

import (
	"errors"
	"fmt"
	"strconv"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/goldens"
	"github.com/NandhaKishorM/laya/laya-go/router"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func family(t *testing.T, key string) []any {
	t.Helper()
	root := fixtures.Load(t, "go_extra.json")
	return fixtures.List(t, fixtures.Obj(t, fixtures.Field(t, root, "router")).Must(key))
}

func str(t *testing.T, o fixtures.Object, key string) string {
	t.Helper()
	return fixtures.Str(t, o.Must(key))
}

// questionsOf builds the question set a golden holds in wire form; null is the zero set.
func questionsOf(t *testing.T, wire any) schema.Questions {
	t.Helper()
	if wire == nil {
		return schema.Questions{}
	}
	qs, err := schema.ParseQuestions(fixtures.Marshal(t, wire))
	if err != nil {
		t.Fatalf("questions: %v", err)
	}
	return qs
}

func TestNamesGolden(t *testing.T) {
	cases := family(t, "names")
	if len(cases) != 263 {
		t.Fatalf("%d cases", len(cases))
	}
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		name := str(t, rec, "name")
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			if got, want := router.CanonicalName(name), str(t, rec, "canonical"); got != want {
				t.Errorf("CanonicalName(%q) = %q, want %q", name, got, want)
			}
			got, err := router.NormaliseName(name)
			if want, bad := rec.Get("error"); bad {
				var ue *router.UnknownModelError
				if !errors.As(err, &ue) || err.Error() != fixtures.Str(t, want) {
					t.Errorf("NormaliseName(%q) error = %v, want %q", name, err, want)
				}
			} else if err != nil || string(got) != str(t, rec, "normalised") {
				t.Errorf("NormaliseName(%q) = %q, %v, want %q", name, got, err, str(t, rec, "normalised"))
			}
			spec, ok := router.ResolveModelSpec(name)
			want := rec.Must("spec")
			if want == nil {
				if ok {
					t.Errorf("ResolveModelSpec(%q) = %v, want none", name, spec)
				}
				return
			}
			pair := fixtures.List(t, want)
			sub := fixtures.StrOrEmpty(t, pair[1])
			if !ok || spec.Repo != fixtures.Str(t, pair[0]) || spec.Subfolder != sub {
				t.Errorf("ResolveModelSpec(%q) = %+v, %v, want %v", name, spec, ok, pair)
			}
		})
	}
}

func TestEnglishFromCodeGolden(t *testing.T) {
	cases := family(t, "codes")
	if len(cases) != 279 {
		t.Fatalf("%d cases", len(cases))
	}
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		code := str(t, rec, "code")
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			english, ok := router.EnglishFromCode(code)
			switch want := rec.Must("english"); want {
			case nil:
				if ok {
					t.Errorf("EnglishFromCode(%q) = %v, want abstain", code, english)
				}
			default:
				if !ok || english != fixtures.Bool(t, want) {
					t.Errorf("EnglishFromCode(%q) = %v, %v, want %v", code, english, ok, want)
				}
			}
		})
	}
}

func TestWorkflowGolden(t *testing.T) {
	matched := 0
	for i, c := range family(t, "workflows") {
		rec := fixtures.Obj(t, c)
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			var entries []schema.Entry
			for _, id := range fixtures.Strings(t, rec.Must("ids")) {
				entries = append(entries, schema.Ask(id, schema.MustQuestion(schema.Noul("x"))))
			}
			qs, err := schema.NewQuestions(entries...)
			if err != nil {
				t.Fatal(err)
			}
			got := router.MatchTypedDecisionsWorkflow(qs)
			want := fixtures.StrOrEmpty(t, rec.Must("workflow"))
			if got != want {
				t.Errorf("ids %v: %q, want %q", rec.Must("ids"), got, want)
			}
			if want != "" {
				matched++
			}
		})
	}
	if matched < 4 {
		t.Fatalf("only %d workflows matched", matched)
	}
}

// TestQuestionSchemaGolden holds QuestionSchema to _question_schema. Python signs the dict the
// caller wrote; Go signs the validated Questions, which are written in canonical form (instructions
// as text, a plain noul without criteria, keys in the order type, instructions, criteria). On a
// definition already in that form the two must be identical; on any other the Go signature must
// still be stable, because it is the signature of the canonical form it re-reads.
func TestQuestionSchemaGolden(t *testing.T) {
	cases := family(t, "signatures")
	if len(cases) != 107 {
		t.Fatalf("%d cases", len(cases))
	}
	canonical := 0
	sigs := make([]string, len(cases))
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		t.Run(fmt.Sprintf("%d-%s", i, str(t, rec, "case")), func(t *testing.T) {
			qs := questionsOf(t, rec.Must("questions"))
			got, err := router.QuestionSchema(qs)
			if err != nil {
				t.Fatal(err)
			}
			sigs[i] = got
			if want := str(t, rec, "signature"); got == want {
				canonical++
				return
			}
			again, err := schema.ParseQuestions([]byte(got))
			if err != nil {
				t.Fatalf("the signature %s does not re-read: %v", got, err)
			}
			if second, _ := router.QuestionSchema(again); second != got {
				t.Errorf("not stable:\n  %s\n  %s", got, second)
			}
		})
	}
	if canonical != 50 {
		t.Fatalf("only %d signatures equal Python's, want 50", canonical)
	}
	checkGrouping(t, cases, sigs)
	pairs := 0
	for i, c := range cases {
		if str(t, fixtures.Obj(t, c), "case") == "multi-reversed" {
			pairs++
			if sigs[i] == sigs[i-1] {
				t.Errorf("case %d: reversing the questions kept the signature %s", i, sigs[i])
			}
		}
	}
	if pairs != 30 {
		t.Fatalf("%d reversed pairs", pairs)
	}
}

// build makes the Router a golden's kwargs describe.
func build(t *testing.T, kw fixtures.Object) (*router.Router, error) {
	t.Helper()
	var opts []router.Option
	for _, m := range kw {
		switch m.Key {
		case "default":
			opts = append(opts, router.WithDefault(fixtures.Str(t, m.Value)))
		case "auto_task_detection":
			opts = append(opts, router.WithAutoTaskDetection(fixtures.Bool(t, m.Value)))
		case "standalone_repos":
			opts = append(opts, router.WithStandaloneRepos(fixtures.Bool(t, m.Value)))
		case "lang_guess":
			opts = append(opts, router.WithLangGuess(router.LangCode(fixtures.Str(t, m.Value))))
		case "models":
			for _, src := range fixtures.Obj(t, m.Value) {
				switch v := src.Value.(type) {
				case string:
					opts = append(opts, router.WithSource(src.Key, v))
				default:
					pair := fixtures.List(t, v)
					opts = append(opts, router.WithSourceIn(src.Key, fixtures.Str(t, pair[0]), fixtures.StrOrEmpty(t, pair[1])))
				}
			}
		default:
			t.Fatalf("unhandled router kwarg %q", m.Key)
		}
	}
	return router.New(opts...)
}

// callOptions are a golden's route kwargs as call options, with the state and questions.
func callOptions(t *testing.T, kw fixtures.Object) (any, schema.Questions, []router.CallOption) {
	t.Helper()
	var (
		state any
		qs    schema.Questions
		opts  []router.CallOption
	)
	for _, m := range kw {
		switch m.Key {
		case "state":
			state = goldens.Value(m.Value)
		case "questions":
			qs = questionsOf(t, m.Value)
		case "model":
			opts = append(opts, router.ForModel(fixtures.Str(t, m.Value)))
		case "task":
			opts = append(opts, router.ForTask(fixtures.Str(t, m.Value)))
		case "lang":
			opts = append(opts, router.ForLang(fixtures.Str(t, m.Value)))
		case "lang_guess":
			opts = append(opts, router.ForLangGuess(router.LangCode(fixtures.Str(t, m.Value))))
		default:
			t.Fatalf("unhandled route kwarg %q", m.Key)
		}
	}
	return state, qs, opts
}

func checkDecision(t *testing.T, got router.Decision, want fixtures.Object) {
	t.Helper()
	if string(got.Model) != str(t, want, "model") {
		t.Errorf("model %q, want %q", got.Model, str(t, want, "model"))
	}
	if got.Repo != str(t, want, "repo") {
		t.Errorf("repo %q, want %q", got.Repo, str(t, want, "repo"))
	}
	if got.Reason != str(t, want, "reason") {
		t.Errorf("reason\n got %q\nwant %q", got.Reason, str(t, want, "reason"))
	}
	if got.Workflow != fixtures.StrOrEmpty(t, want.Must("workflow")) {
		t.Errorf("workflow %q, want %v", got.Workflow, want.Must("workflow"))
	}
	det := want.Must("detection")
	if det == nil {
		if got.Detection != nil {
			t.Errorf("detection %+v, want none", got.Detection)
		}
		return
	}
	d := fixtures.Obj(t, det)
	g := got.Detection
	if g == nil {
		t.Fatal("detection missing")
	}
	if string(g.Script) != str(t, d, "script") || g.Language != fixtures.StrOrEmpty(t, d.Must("language")) ||
		g.English != fixtures.Bool(t, d.Must("is_english")) ||
		g.LanguageUndecided() != fixtures.Bool(t, d.Must("language_undecided")) ||
		g.NonLatinFraction != fixtures.Float(t, d.Must("non_latin_fraction")) ||
		g.DiacriticRate != fixtures.Float(t, d.Must("diacritic_rate")) ||
		g.MixedSegment != fixtures.StrOrEmpty(t, d.Must("mixed_segment")) {
		t.Errorf("detection\n got %+v\nwant %v", *g, d)
	}
}

func TestRouteGolden(t *testing.T) {
	cases := family(t, "routes")
	if len(cases) != 1058 {
		t.Fatalf("%d cases", len(cases))
	}
	var decided, refused, refusedRouters int
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		t.Run(fmt.Sprintf("%d-%s", i, str(t, rec, "case")), func(t *testing.T) {
			r, err := build(t, fixtures.Obj(t, rec.Must("router")))
			if want, bad := rec.Get("router_error"); bad {
				refusedRouters++
				if err == nil || err.Error() != fixtures.Str(t, want) {
					t.Fatalf("New error = %v, want %q", err, want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			state, qs, opts := callOptions(t, fixtures.Obj(t, rec.Must("route")))
			got, err := r.Route(state, qs, opts...)
			if want, bad := rec.Get("error"); bad {
				refused++
				var ue *router.UnknownModelError
				if !errors.As(err, &ue) || err.Error() != fixtures.Str(t, want) {
					t.Fatalf("Route error = %v, want %q", err, want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			decided++
			checkDecision(t, got, fixtures.Obj(t, rec.Must("decision")))
		})
	}
	t.Logf("%d decided, %d routes refused, %d routers refused", decided, refused, refusedRouters)
	if decided < 900 || refused < 100 || refusedRouters < 4 {
		t.Fatalf("the golden stopped covering both outcomes: %d, %d, %d", decided, refused, refusedRouters)
	}
}

func TestRouteBatchGolden(t *testing.T) {
	cases := family(t, "batches")
	if len(cases) != 63 {
		t.Fatalf("%d cases", len(cases))
	}
	var failed int
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			r, err := build(t, fixtures.Obj(t, rec.Must("router")))
			if err != nil {
				t.Fatal(err)
			}
			var reqs []router.Request
			for _, item := range fixtures.List(t, rec.Must("requests")) {
				kw := fixtures.Obj(t, item)
				state, qs, opts := callOptions(t, kw)
				reqs = append(reqs, router.Request{State: state, Questions: qs, Options: opts})
			}
			got, err := r.RouteBatch(reqs)
			if want, bad := rec.Get("error"); bad {
				failed++
				if err == nil || err.Error() != fixtures.Str(t, want) {
					t.Fatalf("error = %v, want %q", err, want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := fixtures.List(t, rec.Must("decisions"))
			if len(got) != len(want) {
				t.Fatalf("%d decisions, want %d", len(got), len(want))
			}
			for j := range got {
				checkDecision(t, got[j], fixtures.Obj(t, want[j]))
			}
		})
	}
	if failed < 3 {
		t.Fatalf("only %d failing batches", failed)
	}
}

// coarser are the pairs of definitions Python signs apart and Go signs alike. Each renders
// identically (a noul's criteria {} against none or blank, labels in another key order), so sharing a forward
// pass between them changes no prompt.
var coarser = [][2]string{
	{"noul-plain", "noul-null-criteria"},
	{"noul-plain", "noul-empty-criteria"},
	{"noul-plain", "noul-empty-is-default"},
	{"noul-plain", "labels-null"},
	{"noul-null-criteria", "noul-empty-criteria"},
	{"noul-null-criteria", "noul-empty-is-default"},
	{"noul-null-criteria", "labels-null"},
	{"noul-empty-criteria", "noul-empty-is-default"},
	{"noul-empty-criteria", "labels-null"},
	{"noul-empty-is-default", "labels-null"},
	{"labels-ok", "labels-reversed-keys"},
}

// checkGrouping compares every pair of single definitions: whatever Python groups Go groups, and
// whatever Go groups renders the same; Go groups more only for the pairs listed above.
func checkGrouping(t *testing.T, cases []any, sigs []string) {
	t.Helper()
	type def struct{ name, py, render string }
	var defs []def
	var goSig []string
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		render, ok := rec.Get("render")
		if !ok {
			continue
		}
		defs = append(defs, def{str(t, rec, "case"), str(t, rec, "signature"), render.(string)})
		goSig = append(goSig, sigs[i])
	}
	allowed := map[[2]string]bool{}
	for _, p := range coarser {
		allowed[p] = true
	}
	var seen int
	pairs := 0
	for i := range defs {
		for j := i + 1; j < len(defs); j++ {
			pairs++
			pyEq, goEq := defs[i].py == defs[j].py, goSig[i] == goSig[j]
			key := [2]string{defs[i].name, defs[j].name}
			switch {
			case pyEq && !goEq:
				t.Errorf("Python groups %v, Go splits them", key)
			case goEq && defs[i].render != defs[j].render:
				t.Errorf("Go groups %v, which render differently", key)
			case goEq && !pyEq:
				seen++
				if !allowed[key] {
					t.Errorf("Go groups %v, which Python signs apart, and it is not listed", key)
				}
			}
		}
	}
	if seen != len(coarser) || pairs != 1081 {
		t.Errorf("%d coarser pairs seen, %d listed, over %d pairs", seen, len(coarser), pairs)
	}
}
