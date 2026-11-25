package router

import (
	"fmt"
	"os"
	"os/user"
	"strings"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// Decision is the routing outcome: which checkpoint, from where, why, and what was detected.
type Decision struct {
	// Model is the chosen checkpoint.
	Model Checkpoint
	// Repo is where the Router would load it from: "repo" or "repo/subfolder".
	Repo string
	// Reason says why, in laya's words.
	Reason string
	// Detection is the language analysis the decision rests on, or nil when an explicit model,
	// task, lang, hint or workflow decided before detection ran.
	Detection *lang.Analysis
	// Workflow is the typed-decisions workflow the question ids match, or "". It is reported
	// whether or not WithAutoTaskDetection is on, except for an explicit model or task.
	Workflow string
}

// String is laya's repr of a RouteDecision.
func (d Decision) String() string {
	return "RouteDecision(model=" + pyjson.Repr(string(d.Model)) + ", reason=" + pyjson.Repr(d.Reason) + ")"
}

// LangHint answers, for a state, which language it is in: a code such as "en", "de-DE" or
// "en_US.UTF-8". Returning "" or a code that names no language ("C", "und") abstains, which sends
// the request on to detection. It only decides whether the English checkpoint can read the text,
// so every non-English code selects Multilingual. A Router calls its hint from every goroutine
// that routes, so a hint must be safe for concurrent use.
type LangHint func(state any) string

// LangCode is a hint that always answers code.
func LangCode(code string) LangHint { return func(any) string { return code } }

// Router routes requests. Build one with New.
type Router struct {
	models  map[Checkpoint]ModelSpec
	def     Checkpoint
	auto    bool
	hint    LangHint
	hk      *hookState
	lc      lifecycle
	digests digestTable
}

// Option configures New.
type Option func(*config) error

type config struct {
	hooks      hookConfig
	standalone bool
	def        string
	auto       bool
	hint       LangHint
	sources    []source
	factory    AgentFactory
	maxLoaded  int
	digests    []digestPin
}

type source struct {
	name   string
	spec   ModelSpec
	expand bool
}

// WithDefault sets the checkpoint for text whose language cannot be identified and for text with
// no letters. It takes a name or alias. The default is Multilingual.
func WithDefault(name string) Option {
	return func(c *config) error { c.def = name; return nil }
}

// WithAutoTaskDetection routes a request whose question ids are exactly those of a typed-decisions
// workflow to TypedDecisions. It is off by default: that checkpoint is fine-tuned on four
// synthetic workflows and is never a silent default.
func WithAutoTaskDetection(on bool) Option {
	return func(c *config) error { c.auto = on; return nil }
}

// WithStandaloneRepos names the per-checkpoint hub repos instead of the bundle repo's subfolders.
func WithStandaloneRepos(on bool) Option {
	return func(c *config) error { c.standalone = on; return nil }
}

// WithLangGuess installs a hint for every request. It is checked after a call's ForLang and
// ForLangGuess, and before detection; it never overrides an explicit model, task or lang.
func WithLangGuess(h LangHint) Option {
	return func(c *config) error { c.hint = h; return nil }
}

// WithSource points a built-in checkpoint at a repo id or a local path, such as a fine-tune that
// stands in for English. A leading "~" is expanded to the home directory, as laya does.
func WithSource(name, repo string) Option {
	return func(c *config) error {
		c.sources = append(c.sources, source{name, ModelSpec{Repo: repo}, true})
		return nil
	}
}

// WithSourceIn is WithSource for a repo and a subfolder of it. The repo is not expanded.
func WithSourceIn(name, repo, subfolder string) Option {
	return func(c *config) error {
		c.sources = append(c.sources, source{name, ModelSpec{Repo: repo, Subfolder: subfolder}, false})
		return nil
	}
}

// New builds a Router. Hooks run concurrently unless WithSerialHooks is given, and a serial hook
// must not call Route on its own Router. A default or source name that is neither a checkpoint nor an alias is an
// *UnknownModelError. Sources are applied in the order given, the last one for a name winning.
func New(opts ...Option) (*Router, error) {
	c := config{def: string(Multilingual), hooks: hookConfig{raise: true}, maxLoaded: DefaultMaxLoaded}
	for _, o := range opts {
		if err := o(&c); err != nil {
			return nil, err
		}
	}
	base := bundled
	if c.standalone {
		base = standalone
	}
	r := &Router{models: make(map[Checkpoint]ModelSpec, len(base)), auto: c.auto, hint: c.hint, hk: newHookState(c.hooks)}
	for k, v := range base {
		r.models[k] = v
	}
	var err error
	for _, s := range c.sources {
		var key Checkpoint
		key, err = NormaliseName(s.name)
		if err != nil {
			return nil, err
		}
		if s.expand {
			s.spec.Repo = expandUser(s.spec.Repo)
		}
		r.models[key] = s.spec
	}
	if r.digests, err = digestsFromEnv(r.models); err != nil {
		return nil, err
	}
	pins := make(digestTable, len(c.digests))
	spelled := map[*byte]map[Checkpoint]string{}
	for _, p := range c.digests {
		key, err := NormaliseName(p.name)
		if err != nil {
			return nil, err
		}
		if prior, dup := spelled[p.group][key]; dup {
			return nil, fmt.Errorf("sha256_digests names %q twice: %q and %q are the same checkpoint",
				string(key), prior, p.name)
		}
		if spelled[p.group] == nil {
			spelled[p.group] = map[Checkpoint]string{}
		}
		spelled[p.group][key] = p.name
		pins[key] = p.files
	}
	for k, v := range pins {
		r.digests[k] = v
	}
	def, err := NormaliseName(c.def)
	if err != nil {
		return nil, err
	}
	r.def = def
	r.lc.init(c.factory, c.maxLoaded)
	return r, nil
}

// expandUser is os.path.expanduser: "~" and "~/x" use $HOME, else the account's home; "~name"
// uses that account's home. A path it cannot expand is returned unchanged.
func expandUser(path string) string {
	if !strings.HasPrefix(path, "~") {
		return path
	}
	end := strings.IndexByte(path, '/')
	if end < 0 {
		end = len(path)
	}
	var home string
	if end == 1 {
		h, set := os.LookupEnv("HOME")
		if !set {
			u, err := user.Current()
			if err != nil {
				return path
			}
			h = u.HomeDir
		}
		home = h
	} else {
		u, err := user.Lookup(path[1:end])
		if err != nil {
			return path
		}
		home = u.HomeDir
	}
	if out := strings.TrimRight(home, "/") + path[end:]; out != "" {
		return out
	}
	return "/"
}

// Default is the checkpoint used when detection abstains.
func (r *Router) Default() Checkpoint { return r.def }

// Spec is where the Router would load a checkpoint from.
func (r *Router) Spec(c Checkpoint) (ModelSpec, bool) {
	s, ok := r.models[c]
	return s, ok
}

// CallOption sets one routing override for one request.
type CallOption func(*call)

type call struct {
	model, task, lang *string
	hint              LangHint
	hooks             []any
	raise             *bool
	timeout           *time.Duration
	err               error
}

// ForModel routes to the named checkpoint or alias, ahead of every other signal.
func ForModel(name string) CallOption { return func(c *call) { c.model = &name } }

// ForTask routes by task. "typed_decisions" and "typed-decisions" in any case select TypedDecisions;
// any other value must name a checkpoint.
func ForTask(task string) CallOption { return func(c *call) { c.task = &task } }

// ForLang routes by language code. It decides only when the code names a language; a blank code or
// one such as "C" falls through to ForLangGuess and detection.
func ForLang(code string) CallOption { return func(c *call) { c.lang = &code } }

// ForLangGuess adds a hint for this request, checked after ForLang and before the Router's own.
func ForLangGuess(h LangHint) CallOption { return func(c *call) { c.hint = h } }

// Route decides which checkpoint reads state, without loading or running anything. qs is only
// looked at for its ids, to recognise a typed-decisions workflow; the zero value is fine.
//
// Hooks with an OnRoute method then see the decision and may replace it; the replacement is
// returned (see hooks.go). Under WithSerialHooks a hook must not call Route on the Router that is
// running it: the call would wait for itself.
//
// The error is an *UnknownModelError for a model or task that names no checkpoint, or a hook's
// failure under WithHookErrors(true).
func (r *Router) Route(state any, qs schema.Questions, opts ...CallOption) (Decision, error) {
	var c call
	for _, o := range opts {
		o(&c)
	}
	return r.routeHooked(state, qs, c)
}

// routeHooked is route followed by the OnRoute hooks, Python's Router.route.
func (r *Router) routeHooked(state any, qs schema.Questions, c call) (Decision, error) {
	d, err := r.route(state, qs, c)
	if err != nil {
		return Decision{}, err
	}
	if c.err != nil {
		return Decision{}, c.err
	}
	return r.dispatchRoute(state, qs, d, c)
}

func (r *Router) decision(key Checkpoint, reason string, det *lang.Analysis, wf string) Decision {
	return Decision{Model: key, Repo: r.models[key].String(), Reason: reason, Detection: det, Workflow: wf}
}

func (r *Router) route(state any, qs schema.Questions, c call) (Decision, error) {
	if c.model != nil {
		key, err := NormaliseName(*c.model)
		if err != nil {
			return Decision{}, err
		}
		return r.decision(key, "explicit model="+pyjson.Repr(*c.model), nil, ""), nil
	}
	if c.task != nil {
		name := *c.task
		if strings.ReplaceAll(unicodedata.Lower(name), "-", "_") == "typed_decisions" {
			name = string(TypedDecisions)
		}
		key, err := NormaliseName(name)
		if err != nil {
			return Decision{}, err
		}
		return r.decision(key, "explicit task="+pyjson.Repr(*c.task), nil, ""), nil
	}

	wf := MatchTypedDecisionsWorkflow(qs)
	if wf != "" && r.auto {
		return r.decision(TypedDecisions, "question ids match the "+pyjson.Repr(wf)+" typed-decisions workflow", nil, wf), nil
	}

	if c.lang != nil {
		if english, ok := EnglishFromCode(*c.lang); ok {
			return r.decision(pick(english), "explicit lang="+pyjson.Repr(*c.lang), nil, wf), nil
		}
	}

	for _, h := range [...]struct {
		source string
		hint   LangHint
	}{{"lang_guess", c.hint}, {"Router(lang_guess=...)", r.hint}} {
		if h.hint == nil {
			continue
		}
		if english, ok := EnglishFromCode(h.hint(state)); ok {
			kind := "non-English"
			if english {
				kind = "English"
			}
			return r.decision(pick(english), h.source+": the caller identified this as "+kind+" text", nil, wf), nil
		}
	}

	det := lang.Analyze(state)
	key, reason := r.def, ""
	switch {
	case det.Script == lang.ScriptUnknown:
		reason = "no letters detected in state; using default (" + string(key) + ")"
	case det.Script != lang.ScriptLatin:
		key = Multilingual
		reason = fmt.Sprintf("non-Latin script (%s, %s%% of letters); the English checkpoint cannot read it",
			det.Script, pyjson.Percent0(det.NonLatinFraction))
	case !det.English:
		key = Multilingual
		switch {
		case det.MixedSegment != "":
			reason = "Latin script, mostly English, but a line or field reads as " + pyjson.Repr(det.Language) +
				" (" + pyjson.Repr(head(det.MixedSegment, 60)) + "); the English checkpoint cannot read it"
		case det.Language != "":
			reason = "Latin script but language looks like " + pyjson.Repr(det.Language) + ", not English"
		default:
			reason = "Latin script, language not identified but " + pyjson.Percent0(det.DiacriticRate) +
				"% non-English letters; not safe for the English checkpoint"
		}
	case det.LanguageUndecided():
		reason = "Latin script, language not identified and no non-English letters; using default (" + string(key) + ")"
	default:
		key, reason = English, "English Latin text"
	}
	return r.decision(key, reason, &det, wf), nil
}

func pick(english bool) Checkpoint {
	if english {
		return English
	}
	return Multilingual
}

// head is s[:n] for a Python str: the first n code points.
func head(s string, n int) string {
	for i := range s {
		if n == 0 {
			return s[:i]
		}
		n--
	}
	return s
}

// Request is one entry of RouteBatch.
type Request struct {
	// State is what the checkpoint will read. It must not be nil.
	State any
	// Questions are the questions asked; only their ids are read to route.
	Questions schema.Questions
	// Options are this request's overrides.
	Options []CallOption
}

// RouteBatch routes a heterogeneous batch without loading anything. The decisions keep the
// order of reqs. A nil State is refused with laya's text; the first failing request ends the
// batch.
//
// opts apply to every request, before the request's own Options; they are meant for the hook
// options (ForHooks, ForHookErrors, ForHookTimeout), as route_batch's hooks, hooks_raise and
// hooks_timeout are.
func (r *Router) RouteBatch(reqs []Request, opts ...CallOption) ([]Decision, error) {
	out := make([]Decision, 0, len(reqs))
	for i, req := range reqs {
		if req.State == nil {
			return nil, fmt.Errorf("request %d 'state' must not be None; pass a string, dict, or list", i)
		}
		d, err := r.Route(req.State, req.Questions, append(append([]CallOption(nil), opts...), req.Options...)...)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, nil
}
