package schema_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func TestGateFieldsRoundTrip(t *testing.T) {
	thr := 0.25
	a := &schema.NoulAnswer{Noul: 0.6, Common: schema.Common{Confidence: 0.6, AnswerConfidence: 0.6, LowConfidence: true, Abstention: "abstained", AbstentionThreshold: &thr}}
	b, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.HasSuffix(s, `"low_confidence":true,"abstention":"abstained","abstention_threshold":0.25}`) {
		t.Fatalf("gate fields not last, in order: %s", s)
	}
	back, err := schema.UnmarshalAnswer(b)
	if err != nil {
		t.Fatal(err)
	}
	c := back.(*schema.NoulAnswer).Common
	if !c.LowConfidence || c.Abstention != "abstained" || c.AbstentionThreshold == nil || *c.AbstentionThreshold != 0.25 {
		t.Fatalf("round trip lost the gate: %+v", c)
	}
}

func TestGateFieldsAbsentWhenUngated(t *testing.T) {
	b, _ := json.Marshal(&schema.NoulAnswer{Noul: 0.6, Common: schema.Common{Confidence: 0.6, AnswerConfidence: 0.6}})
	if strings.Contains(string(b), "abstention") || strings.Contains(string(b), "low_confidence") {
		t.Fatalf("%s", b)
	}
}

func TestGateFieldsRefuseBadInput(t *testing.T) {
	base := `{"type":"noul","probabilities":{"0":0.4,"1":0.6},"action":{"act_probability":0.5},"noul":0.6,"answer_confidence":0.6,"confidence":0.6`
	for _, extra := range []string{
		`,"abstention":"passed"`,                           // state without threshold
		`,"abstention_threshold":0.5`,                      // threshold without state
		`,"abstention":"maybe","abstention_threshold":0.5`, // unknown state
		`,"low_confidence":false`,                          // laya removes the key, never writes false
	} {
		if _, err := schema.UnmarshalAnswer([]byte(base + extra + "}")); err == nil {
			t.Errorf("accepted %s", extra)
		}
	}
}
