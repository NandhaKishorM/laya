package demo

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

// Expected texts are json.dumps(state, ensure_ascii=False) from Python.
func TestStateSerializesInPythonDictOrder(t *testing.T) {
	for _, c := range []struct{ subject, body, want string }{
		{"Charged twice for order 8812", "I was billed two times this morning.",
			`{"subject": "Charged twice for order 8812", "body": "I was billed two times this morning."}`},
		{"ऑर्डर 8812 का दो बार बिल", "कृपया रिफंड करें।",
			`{"subject": "ऑर्डर 8812 का दो बार बिल", "body": "कृपया रिफंड करें।"}`},
	} {
		got, err := sequence.SerializeState(State(c.subject, c.body))
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Errorf("state = %s, want %s", got, c.want)
		}
	}
}

// The golden is laya.presets.email_questions() minus is_phishing, recorded by
// scripts/gen_email_golden.py. Every field of every question is compared, in order.
func TestQuestionsAreTheEmailPreset(t *testing.T) {
	golden, err := os.ReadFile("testdata/email_questions.json")
	if err != nil {
		t.Fatal(err)
	}
	qs, err := Questions()
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(qs)
	if err != nil {
		t.Fatal(err)
	}
	var g, w any
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(golden, &w); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(g, w) {
		t.Fatalf("questions differ from the Python preset:\n got %s\nwant %s", got, golden)
	}
	if ids := qs.IDs(); !reflect.DeepEqual(ids, []string{"category", "is_spam", "urgency", "needs_reply"}) {
		t.Fatalf("ids = %v", ids)
	}
}
