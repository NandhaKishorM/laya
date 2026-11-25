package router_test

import (
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/router"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

const (
	englishText = "I was charged twice, please refund the duplicate payment on my invoice."
	hindiText   = "मुझसे दो बार शुल्क लिया गया"
	romanian    = "Care este ora in Tokyo?"
)

var none schema.Questions

func noul(t *testing.T, ids ...string) schema.Questions {
	t.Helper()
	var entries []schema.Entry
	for _, id := range ids {
		entries = append(entries, schema.Ask(id, schema.MustQuestion(schema.Noul("x"))))
	}
	qs, err := schema.NewQuestions(entries...)
	if err != nil {
		t.Fatal(err)
	}
	return qs
}

func mustNew(t *testing.T, opts ...router.Option) *router.Router {
	t.Helper()
	r, err := router.New(opts...)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func route(t *testing.T, r *router.Router, state any, qs schema.Questions, opts ...router.CallOption) router.Decision {
	t.Helper()
	d, err := r.Route(state, qs, opts...)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

var customerService = []string{"action", "category", "churn_risk", "needs_human", "urgency"}

// TestPrecedence pins the order of the evidence: each row has two signals that disagree, and the
// earlier one in the package comment must win.
func TestPrecedence(t *testing.T) {
	auto := mustNew(t, router.WithAutoTaskDetection(true))
	plain := mustNew(t)
	installed := mustNew(t, router.WithLangGuess(router.LangCode("ro")))
	cs := noul(t, customerService...)
	rows := []struct {
		name  string
		r     *router.Router
		state any
		qs    schema.Questions
		opts  []router.CallOption
		want  router.Checkpoint
	}{
		{"model beats task", plain, "x", cs, []router.CallOption{router.ForModel("english"), router.ForTask("typed_decisions")}, router.English},
		{"model beats workflow", auto, "x", cs, []router.CallOption{router.ForModel("multilingual")}, router.Multilingual},
		{"task beats workflow", auto, "x", cs, []router.CallOption{router.ForTask("english")}, router.English},
		{"workflow beats lang", auto, "x", cs, []router.CallOption{router.ForLang("en")}, router.TypedDecisions},
		{"workflow beats guess", auto, "x", cs, []router.CallOption{router.ForLangGuess(router.LangCode("en"))}, router.TypedDecisions},
		{"lang beats guess", plain, "x", cs, []router.CallOption{router.ForLang("en"), router.ForLangGuess(router.LangCode("ro"))}, router.English},
		{"lang beats installed guess", installed, "hello", none, []router.CallOption{router.ForLang("en")}, router.English},
		{"call guess beats installed", installed, romanian, none, []router.CallOption{router.ForLangGuess(router.LangCode("en"))}, router.English},
		{"installed guess beats detection", installed, englishText, none, nil, router.Multilingual},
		{"guess beats detection", plain, hindiText, none, []router.CallOption{router.ForLangGuess(router.LangCode("en"))}, router.English},
		{"abstaining lang falls to guess", plain, "hello", none, []router.CallOption{router.ForLang(""), router.ForLangGuess(router.LangCode("ro"))}, router.Multilingual},
		{"abstaining call guess falls to installed", installed, englishText, none, []router.CallOption{router.ForLangGuess(router.LangCode("C"))}, router.Multilingual},
		{"detection beats default", mustNew(t, router.WithDefault("typed")), hindiText, none, nil, router.Multilingual},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			if got := route(t, row.r, row.state, row.qs, row.opts...); got.Model != row.want {
				t.Errorf("model %q (%s), want %q", got.Model, got.Reason, row.want)
			}
		})
	}
}

func TestDefaultIsMultilingual(t *testing.T) {
	r := mustNew(t)
	if r.Default() != router.Multilingual {
		t.Fatalf("default %q", r.Default())
	}
	for _, state := range []any{"", "12345", nil, "Hello"} {
		d := route(t, r, state, none)
		if d.Model != router.Multilingual || !strings.HasSuffix(d.Reason, "using default (multilingual)") {
			t.Errorf("%#v: %q %q", state, d.Model, d.Reason)
		}
	}
	e := mustNew(t, router.WithDefault("EN"))
	if d := route(t, e, "12345", none); d.Model != router.English {
		t.Errorf("WithDefault(EN): %q", d.Model)
	}
}

func TestWorkflowReportedWithoutAutoDetection(t *testing.T) {
	cs := noul(t, customerService...)
	off := route(t, mustNew(t), englishText, cs)
	if off.Workflow != "customer_service" || off.Model != router.English {
		t.Errorf("auto off: workflow %q model %q", off.Workflow, off.Model)
	}
	on := route(t, mustNew(t, router.WithAutoTaskDetection(true)), englishText, cs)
	if on.Model != router.TypedDecisions || on.Workflow != "customer_service" || on.Detection != nil ||
		on.Reason != "question ids match the 'customer_service' typed-decisions workflow" {
		t.Errorf("auto on: %+v", on)
	}
	// An explicit model or task decides before the workflow is looked at, so none is reported.
	if d := route(t, mustNew(t), "x", cs, router.ForModel("en")); d.Workflow != "" {
		t.Errorf("explicit model reported workflow %q", d.Workflow)
	}
	if d := route(t, mustNew(t), "x", cs, router.ForTask("en")); d.Workflow != "" {
		t.Errorf("explicit task reported workflow %q", d.Workflow)
	}
}

func TestWorkflowNeedsExactIDs(t *testing.T) {
	for _, ids := range [][]string{
		customerService[:4],
		append(append([]string{}, customerService...), "extra"),
		{"urgency"},
		{},
		append(append([]string{}, customerService...), "zzz"),
	} {
		if got := router.MatchTypedDecisionsWorkflow(noul(t, ids...)); got != "" {
			t.Errorf("%v matched %q", ids, got)
		}
	}
	reordered := []string{"urgency", "needs_human", "churn_risk", "category", "action"}
	if got := router.MatchTypedDecisionsWorkflow(noul(t, reordered...)); got != "customer_service" {
		t.Errorf("reordered ids: %q", got)
	}
}

func TestTaskSpelling(t *testing.T) {
	r := mustNew(t)
	for _, task := range []string{"typed_decisions", "typed-decisions", "TYPED-DECISIONS", "Typed_Decisions"} {
		d := route(t, r, "x", none, router.ForTask(task))
		if d.Model != router.TypedDecisions || d.Reason != "explicit task='"+task+"'" {
			t.Errorf("%q: %+v", task, d)
		}
	}
}

func TestReasonQuoting(t *testing.T) {
	r := mustNew(t)
	for _, c := range []struct{ lang, reason string }{
		{"EN", `explicit lang='EN'`},
		{"it's", `explicit lang="it's"`},
		{"x\u200by", "explicit lang='x\\u200by'"},
		{"e\x07\U0001f600", `explicit lang='e\x07` + "\U0001f600" + `'`},
		{"\U000e0001", `explicit lang='\U000e0001'`},
	} {
		if d := route(t, r, "x", none, router.ForLang(c.lang)); d.Reason != c.reason {
			t.Errorf("%q: %s, want %s", c.lang, d.Reason, c.reason)
		}
	}
}

// TestMixedSegmentCutsAtCodePoints holds the foreign line to 60 code points, not bytes.
func TestMixedSegmentCutsAtCodePoints(t *testing.T) {
	eng := strings.Repeat("Please review the error and send me the details of this and what you would like to do.\n", 6)
	line := "Eu nao consigo \U0001f600 entrar na minha conta e depois eu nao sei mais o que fazer com isso tudo"
	d := route(t, mustNew(t), eng+line, none)
	if d.Detection == nil || d.Detection.MixedSegment != line {
		t.Fatalf("no mixed segment: %+v", d)
	}
	cut := string([]rune(line)[:60])
	want := "Latin script, mostly English, but a line or field reads as 'pt' ('" + cut + "'); the English checkpoint cannot read it"
	if d.Reason != want {
		t.Errorf("reason\n got %q\nwant %q", d.Reason, want)
	}
}

// TestPercentRoundsHalfToEven: 5 of 8 letters outside Latin is 62.5%, which prints as 62.
func TestPercentRoundsHalfToEven(t *testing.T) {
	state := "aaa" + strings.Repeat("ㄅ", 5)
	if a := lang.Analyze(state); a.Script == lang.ScriptLatin || a.NonLatinFraction != 0.625 {
		t.Fatalf("the fixture drifted: %+v", a)
	}
	d := route(t, mustNew(t), state, none)
	if want := "non-Latin script (other, 62% of letters); the English checkpoint cannot read it"; d.Reason != want {
		t.Errorf("reason %q, want %q", d.Reason, want)
	}
	// 3 Latin of 16 letters is 18.75% Latin, 81.25% not: no tie, rounds to 81.
	state = "aaa" + strings.Repeat("ㄅ", 13)
	if d := route(t, mustNew(t), state, none); !strings.Contains(d.Reason, "81% of letters") {
		t.Errorf("reason %q", d.Reason)
	}
}

func TestUnknownModelText(t *testing.T) {
	r := mustNew(t)
	_, err := r.Route("x", none, router.ForModel("nope"))
	var ue *router.UnknownModelError
	if !errors.As(err, &ue) || ue.Name != "nope" {
		t.Fatalf("err = %v", err)
	}
	want := "unknown model 'nope'; choose one of ['english', 'multilingual', 'typed-decisions'] " +
		"(or an alias: ['decisions', 'default', 'en', 'laya', 'laya-multilingual', 'laya-typed-decisions', " +
		"'ml', 'multi', 'typed', 'typed_decisions'])"
	if err.Error() != want {
		t.Errorf("text\n got %s\nwant %s", err, want)
	}
	if _, err := r.Route("x", none, router.ForTask("nope")); err == nil {
		t.Error("an unknown task must fail")
	}
	if _, err := router.New(router.WithDefault("nope")); !errors.As(err, &ue) {
		t.Errorf("WithDefault(nope): %v", err)
	}
	if _, err := router.New(router.WithSource("nope", "/x")); !errors.As(err, &ue) {
		t.Errorf("WithSource(nope): %v", err)
	}
}

func TestRouteBatchRefusesNilState(t *testing.T) {
	r := mustNew(t)
	reqs := []router.Request{{State: "hello there my friend"}, {State: nil}}
	if _, err := r.RouteBatch(reqs); err == nil ||
		err.Error() != "request 1 'state' must not be None; pass a string, dict, or list" {
		t.Fatalf("err = %v", err)
	}
	out, err := r.RouteBatch(nil)
	if err != nil || out == nil || len(out) != 0 {
		t.Fatalf("empty batch: %v %v", out, err)
	}
}

func TestRouteBatchKeepsOrder(t *testing.T) {
	r := mustNew(t)
	reqs := []router.Request{
		{State: hindiText},
		{State: englishText, Options: []router.CallOption{router.ForModel("typed")}},
		{State: englishText},
		{State: "x", Options: []router.CallOption{router.ForLang("de")}},
	}
	got, err := r.RouteBatch(reqs)
	if err != nil {
		t.Fatal(err)
	}
	want := []router.Checkpoint{router.Multilingual, router.TypedDecisions, router.English, router.Multilingual}
	for i, d := range got {
		if d.Model != want[i] {
			t.Errorf("request %d: %q, want %q", i, d.Model, want[i])
		}
	}
}

func TestSources(t *testing.T) {
	r := mustNew(t, router.WithStandaloneRepos(true), router.WithSource("en", "/models/en"),
		router.WithSourceIn("ML", "org/repo", "sub"))
	for _, c := range []struct {
		model router.Checkpoint
		repo  string
	}{
		{router.English, "/models/en"},
		{router.Multilingual, "org/repo/sub"},
		{router.TypedDecisions, "convaiinnovations/laya-typed-decisions"},
	} {
		if d := route(t, r, "x", none, router.ForModel(string(c.model))); d.Repo != c.repo {
			t.Errorf("%s: repo %q, want %q", c.model, d.Repo, c.repo)
		}
	}
	for model, repo := range map[string]string{
		"english": "convaiinnovations/laya", "multilingual": "convaiinnovations/laya/multilingual",
		"typed-decisions": "convaiinnovations/laya/typed-decisions",
	} {
		if d := route(t, mustNew(t), "x", none, router.ForModel(model)); d.Repo != repo {
			t.Errorf("%s: bundle repo %q, want %q", model, d.Repo, repo)
		}
	}
}

func TestExpandUser(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	repo := func(source string, opts ...router.Option) string {
		r := mustNew(t, append(opts, router.WithSource("english", source))...)
		return route(t, r, "x", none, router.ForModel("english")).Repo
	}
	for _, c := range []struct{ in, want string }{
		{"~", home},
		{"~/", home + "/"},
		{"~/models/en", filepath.Join(home, "models/en")},
		{"/abs/~/x", "/abs/~/x"},
		{"x~", "x~"},
		{"~nosuchuser_zz/x", "~nosuchuser_zz/x"},
	} {
		if got := repo(c.in); got != c.want {
			t.Errorf("%q -> %q, want %q", c.in, got, c.want)
		}
	}
	t.Setenv("HOME", "")
	if got := repo("~"); got != "/" {
		t.Errorf("empty HOME: %q, want /", got)
	}
	t.Setenv("HOME", home+"//")
	if got := repo("~/m"); got != home+"/m" {
		t.Errorf("trailing slashes: %q", got)
	}
	// A pair is a repo and a subfolder, never a path to expand.
	r := mustNew(t, router.WithSourceIn("english", "~/x", "s"))
	if got := route(t, r, "x", none, router.ForModel("english")).Repo; got != "~/x/s" {
		t.Errorf("pair: %q", got)
	}
}

func TestLangHintReceivesState(t *testing.T) {
	calls := 0
	hint := func(state any) string {
		calls++
		if s, ok := state.(string); ok && strings.HasPrefix(s, "de:") {
			return "de"
		}
		return ""
	}
	r := mustNew(t, router.WithLangGuess(hint))
	if d := route(t, r, "de: hallo", none); d.Model != router.Multilingual ||
		d.Reason != "Router(lang_guess=...): the caller identified this as non-English text" {
		t.Errorf("%+v", d)
	}
	if d := route(t, r, englishText, none); d.Model != router.English || d.Detection == nil {
		t.Errorf("an abstaining hint must fall through to detection: %+v", d)
	}
	if calls != 2 {
		t.Errorf("hint called %d times", calls)
	}
	d := route(t, r, "de: x", none, router.ForLangGuess(router.LangCode("en")))
	if d.Reason != "lang_guess: the caller identified this as English text" {
		t.Errorf("%q", d.Reason)
	}
	if calls != 2 {
		t.Errorf("the installed hint ran after a call hint answered")
	}
}

func TestDecisionString(t *testing.T) {
	d := route(t, mustNew(t), englishText, none)
	if got, want := d.String(), "RouteDecision(model='english', reason='English Latin text')"; got != want {
		t.Errorf("%s != %s", got, want)
	}
}

func TestConcurrentRoute(t *testing.T) {
	r := mustNew(t, router.WithAutoTaskDetection(true), router.WithLangGuess(router.LangCode("C")))
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				d, err := r.Route(hindiText, none)
				if err != nil || d.Model != router.Multilingual {
					t.Errorf("%d: %v %v", i, d, err)
					return
				}
			}
		}(i)
	}
	wg.Wait()
}
