package router

import (
	"sort"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// Checkpoint names one of the three shipped checkpoints.
type Checkpoint string

// The shipped checkpoints.
const (
	English        Checkpoint = "english"
	Multilingual   Checkpoint = "multilingual"
	TypedDecisions Checkpoint = "typed-decisions"
)

// String is the checkpoint's name as Python spells it.
func (c Checkpoint) String() string { return string(c) }

// BundleRepo is the hub repo that bundles all three checkpoints.
const BundleRepo = "convaiinnovations/laya"

// ModelSpec is where a checkpoint comes from: a repo id or a local path, and an optional
// subfolder of it.
type ModelSpec struct {
	Repo      string
	Subfolder string
}

// String is laya's _repo_str: "repo", or "repo/subfolder" when there is a subfolder.
func (m ModelSpec) String() string {
	if m.Subfolder != "" {
		return m.Repo + "/" + m.Subfolder
	}
	return m.Repo
}

// bundled and standalone are where each checkpoint comes from by default.
var bundled = map[Checkpoint]ModelSpec{
	English:        {BundleRepo, ""},
	Multilingual:   {BundleRepo, "multilingual"},
	TypedDecisions: {BundleRepo, "typed-decisions"},
}

var standalone = map[Checkpoint]ModelSpec{
	English:        {"convaiinnovations/laya", ""},
	Multilingual:   {"convaiinnovations/laya-multilingual", ""},
	TypedDecisions: {"convaiinnovations/laya-typed-decisions", ""},
}

var aliases = map[string]Checkpoint{
	"en": English, "laya": English, "default": English,
	"multi": Multilingual, "ml": Multilingual, "laya-multilingual": Multilingual,
	"typed": TypedDecisions, "typed_decisions": TypedDecisions,
	"laya-typed-decisions": TypedDecisions, "decisions": TypedDecisions,
}

// Checkpoints lists the three checkpoints in sorted order.
func Checkpoints() []Checkpoint { return []Checkpoint{English, Multilingual, TypedDecisions} }

// workflow is one typed-decisions workflow and the question ids that identify it.
type workflow struct {
	name string
	ids  []string // sorted
}

var workflows = []workflow{
	{"agent_trace_observability", []string{"action", "needs_review", "outcome", "risk", "urgency"}},
	{"customer_service", []string{"action", "category", "churn_risk", "needs_human", "urgency"}},
	{"invoice_processing", []string{"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"}},
	{"security_incidents", []string{"credential_compromise", "disposition", "severity", "true_positive", "urgency"}},
}

// UnknownModelError is a model or task name no checkpoint answers to. Its text is Python's.
type UnknownModelError struct {
	Name string
	msg  string
}

func (e *UnknownModelError) Error() string { return e.msg }

// CanonicalName trims, lowercases and resolves an alias, without asking whether the result names
// a checkpoint: laya's canonical_name.
func CanonicalName(name string) string {
	key := unicodedata.Lower(unicodedata.Strip(name))
	if c, ok := aliases[key]; ok {
		return string(c)
	}
	return key
}

// NormaliseName is the checkpoint a name or alias stands for, or an *UnknownModelError carrying
// laya's text, which lists the sorted names and aliases.
func NormaliseName(name string) (Checkpoint, error) {
	key := Checkpoint(CanonicalName(name))
	if _, ok := bundled[key]; ok {
		return key, nil
	}
	return "", unknownModel(name)
}

func unknownModel(name string) *UnknownModelError {
	names := make([]string, 0, len(bundled))
	for c := range bundled {
		names = append(names, string(c))
	}
	alias := make([]string, 0, len(aliases))
	for a := range aliases {
		alias = append(alias, a)
	}
	return &UnknownModelError{Name: name, msg: "unknown model " + pyjson.Repr(name) + "; choose one of " +
		pyList(names) + " (or an alias: " + pyList(alias) + ")"}
}

// pyList is repr(sorted(names)).
func pyList(items []string) string {
	sort.Strings(items)
	parts := make([]string, len(items))
	for i, s := range items {
		parts[i] = pyjson.Repr(s)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// ResolveModelSpec is the shipped source of a checkpoint name or alias, and false when the name
// is neither: laya's resolve_model_spec, which lets a caller pass anything else through.
func ResolveModelSpec(name string) (ModelSpec, bool) {
	c, err := NormaliseName(name)
	if err != nil {
		return ModelSpec{}, false
	}
	return bundled[c], true
}

// MatchTypedDecisionsWorkflow is the typed-decisions workflow whose question ids these are, or
// "". The ids must match exactly, so an unrelated schema that happens to hold "urgency" is never
// captured.
func MatchTypedDecisionsWorkflow(qs schema.Questions) string {
	ids := qs.IDs()
	sort.Strings(ids)
	for _, wf := range workflows {
		if equal(ids, wf.ids) {
			return wf.name
		}
	}
	return ""
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Codes that are valid $LANG values but name no language: they abstain.
var agnostic = map[string]bool{"c": true, "posix": true, "und": true, "zxx": true, "mul": true}

// EnglishFromCode answers whether a language code names English (english true) or another
// language (english false), and ok false when the code identifies nothing: blank, "C", "POSIX",
// "C.UTF-8", "und", "zxx", "mul". It reads "en", "EN", "en-US", "en_US" and "en_US.UTF-8".
func EnglishFromCode(code string) (english, ok bool) {
	code = unicodedata.Lower(unicodedata.Strip(code))
	if code == "" {
		return false, false
	}
	if i := strings.IndexByte(code, '.'); i >= 0 {
		code = code[:i]
	}
	code = strings.ReplaceAll(code, "_", "-")
	if i := strings.IndexByte(code, '-'); i >= 0 {
		code = code[:i]
	}
	if code == "" || agnostic[code] {
		return false, false
	}
	return code == "en" || code == "eng" || code == "english", true
}

// QuestionSchema is the order-sensitive signature of a question set, for sharing a forward pass
// between requests that ask the same thing: laya's _question_schema, json.dumps of the set with
// its insertion order kept at every level.
func QuestionSchema(qs schema.Questions) (string, error) {
	raw, err := qs.MarshalJSON()
	if err != nil {
		return "", err
	}
	v, err := pyvalue.Parse(raw)
	if err != nil {
		return "", err
	}
	return pyvalue.Dumps(v)
}
