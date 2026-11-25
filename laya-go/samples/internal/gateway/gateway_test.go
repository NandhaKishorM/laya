package gateway

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

type golden struct {
	Name        string
	Difficulty  float64
	Domains     json.RawMessage
	NeedsTools  float64 `json:"needs_tools"`
	IsSensitive float64 `json:"is_sensitive"`
	Tier        string
	Why         string
	Row         string
}

// The golden is the tier policy and the table row of Python example 29 on the same numbers,
// recorded by scripts/gen_router_policy_golden.py.
func TestPolicyAndRowMatchPython(t *testing.T) {
	data, err := os.ReadFile("testdata/policy.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []golden
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) != 10 {
		t.Fatalf("%d cases", len(cases))
	}
	for _, c := range cases {
		var probs schema.Probabilities
		if err := json.Unmarshal(c.Domains, &probs); err != nil {
			t.Fatal(err)
		}
		answers := schema.Answers{
			{ID: "difficulty", Answer: &schema.ScoreAnswer{Score: c.Difficulty}},
			{ID: "domain", Answer: &schema.ChoiceAnswer{Choice: probs[0].Label, Probabilities: probs}},
			{ID: "needs_tools", Answer: &schema.NoulAnswer{Noul: c.NeedsTools}},
			{ID: "is_sensitive", Answer: &schema.NoulAnswer{Noul: c.IsSensitive}},
		}
		r, err := Read(answers)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		tier, why := Tier(r)
		if tier != c.Tier || why != c.Why {
			t.Errorf("%s: tier = %q %q, want %q %q", c.Name, tier, why, c.Tier, c.Why)
		}
		if got := Row("label", r); got != c.Row {
			t.Errorf("%s: row\n got %q\nwant %q", c.Name, got, c.Row)
		}
	}
}

func TestReadRejectsWrongAnswerTypes(t *testing.T) {
	good := schema.Answers{
		{ID: "difficulty", Answer: &schema.ScoreAnswer{}},
		{ID: "domain", Answer: &schema.ChoiceAnswer{}},
		{ID: "needs_tools", Answer: &schema.NoulAnswer{}},
		{ID: "is_sensitive", Answer: &schema.NoulAnswer{}},
	}
	if _, err := Read(good); err != nil {
		t.Fatal(err)
	}
	for i, wrong := range []schema.Answer{&schema.NoulAnswer{}, &schema.NoulAnswer{}, &schema.ScoreAnswer{}, &schema.ScoreAnswer{}} {
		bad := append(schema.Answers(nil), good...)
		bad[i].Answer = wrong
		if _, err := Read(bad); err == nil {
			t.Errorf("%s as %T: no error", bad[i].ID, wrong)
		}
		if _, err := Read(append(schema.Answers(nil), good[:i]...)); err == nil {
			t.Errorf("missing %s: no error", good[i].ID)
		}
	}
}

func TestReportShape(t *testing.T) {
	var b bytes.Buffer
	r := Router{Difficulty: 0.5, Domain: "code", DomainP: 0.9}
	Report(&b, []Router{r, r, {Difficulty: 0.9, IsSensitive: 0.97}})
	out := b.String()
	for _, want := range []string{"trivial lookup", "hard refactor", "sensitive legal",
		"sensitive legal   -> frontier      sensitive: money/legal/medical/safety (p=0.97)",
		"even\n   a one-line lookup scores 0.50", "only 0.90 on difficulty", "`is_sensitive` at 0.970"} {
		if !strings.Contains(out, want) {
			t.Errorf("report lacks %q:\n%s", want, out)
		}
	}
}
