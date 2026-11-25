// Package gateway is the policy half of the router sample: it turns the answers of
// presets.Router into a model tier and prints them as examples/29_presets_model_router.py does.
package gateway

import (
	"fmt"
	"io"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// Request is a labelled request to route.
type Request struct{ Label, Text string }

// Requests are the three requests of Python example 29, in its order.
var Requests = []Request{
	{"trivial lookup", "What is the capital of Portugal?"},
	{"hard refactor", "Refactor this 900-line Django view into service classes with dependency " +
		"injection, keeping all existing tests green and preserving transaction " +
		"boundaries."},
	{"sensitive legal", "My employer is withholding my final paycheck after I reported a safety " +
		"violation. Should I sue, and can I claim damages under California law?"},
}

// Router is the four router answers, read by their types.
type Router struct {
	Difficulty  float64
	Domain      string
	DomainP     float64
	NeedsTools  float64
	IsSensitive float64
}

// Read pulls the router answers out of a prediction.
func Read(answers schema.Answers) (Router, error) {
	var r Router
	d, ok := answers.Get("difficulty")
	if !ok {
		return r, fmt.Errorf("gateway: no difficulty answer")
	}
	ds, ok := d.(*schema.ScoreAnswer)
	if !ok {
		return r, fmt.Errorf("gateway: difficulty is %T, not a score", d)
	}
	r.Difficulty = ds.Score
	c, ok := answers.Get("domain")
	if !ok {
		return r, fmt.Errorf("gateway: no domain answer")
	}
	cs, ok := c.(*schema.ChoiceAnswer)
	if !ok {
		return r, fmt.Errorf("gateway: domain is %T, not a choice", c)
	}
	// The first label with the highest probability, as Python's max over the dict.
	for i, p := range cs.Probabilities {
		if i == 0 || p.P > r.DomainP {
			r.Domain, r.DomainP = p.Label.String(), p.P
		}
	}
	for _, f := range []struct {
		id  string
		dst *float64
	}{{"needs_tools", &r.NeedsTools}, {"is_sensitive", &r.IsSensitive}} {
		a, ok := answers.Get(f.id)
		if !ok {
			return r, fmt.Errorf("gateway: no %s answer", f.id)
		}
		n, ok := a.(*schema.NoulAnswer)
		if !ok {
			return r, fmt.Errorf("gateway: %s is %T, not a noul", f.id, a)
		}
		*f.dst = n.Noul
	}
	return r, nil
}

// Tier is the gateway policy: the cheap model unless the request is sensitive, needs tools or is
// hard. Sensitivity and tools override difficulty, never the other way round.
func Tier(r Router) (tier, why string) {
	switch {
	case r.IsSensitive >= 0.5:
		return "frontier", fmt.Sprintf("sensitive: money/legal/medical/safety (p=%.2f)", r.IsSensitive)
	case r.NeedsTools >= 0.5:
		return "frontier+tools", fmt.Sprintf("needs external data (p=%.2f)", r.NeedsTools)
	case r.Difficulty >= 1.5:
		return "frontier", fmt.Sprintf("difficulty %.2f >= 1.5", r.Difficulty)
	}
	return "small", fmt.Sprintf("difficulty %.2f < 1.5, no tools, not sensitive", r.Difficulty)
}

// Row is one line of the side-by-side table.
func Row(label string, r Router) string {
	return fmt.Sprintf("   %-17s %.2f / 3   %-16s %-11.3f %.3f", label, r.Difficulty,
		fmt.Sprintf("%s (%.2f)", r.Domain, r.DomainP), r.NeedsTools, r.IsSensitive)
}

// Report prints the side-by-side table, the policy output and the closing notes. results is in
// Requests order.
func Report(w io.Writer, results []Router) {
	p := func(format string, a ...any) { fmt.Fprintf(w, format+"\n", a...) }
	p("-- side by side --")
	p("   %-17s %-10s %-16s %-11s %s", "request", "difficulty", "domain", "needs_tools", "is_sensitive")
	for i, req := range Requests {
		p("%s", Row(req.Label, results[i]))
	}
	p("-- policy output --")
	for i, req := range Requests {
		tier, why := Tier(results[i])
		p("   %-17s -> %-13s %s", req.Label, tier, why)
	}
	triv, sens := results[0], results[2]
	p("   The threshold 1.5 sits between the lookup cluster and the reasoning end of the scale.")
	p("   Two honest notes. First, `difficulty` is an expected level over four buckets, so even")
	p("   a one-line lookup scores %.2f -- the zero point is \"trivial\", not \"easy\".", triv.Difficulty)
	p("   Second, the sensitive request scores only %.2f on difficulty, so difficulty alone", sens.Difficulty)
	p("   would send it to the cheap model; `is_sensitive` at %.3f is what promotes it. A", sens.IsSensitive)
	p("   gateway should always let sensitivity and tool needs override the difficulty score,")
	p("   never the other way round.")
}
