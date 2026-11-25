package lang

import (
	"encoding/json"
	"strings"
	"testing"
)

// JSON decoded into an Object keeps the key order Python's dict would have, and that order
// decides which text a long state is read from first.
func TestObjectUnmarshalJSONKeepsOrder(t *testing.T) {
	var o Object
	doc := `{"z":"first","a":{"y":"second","b":["third",{"q":"fourth"}]},"m":"fifth","n":3,"o":null}`
	if err := json.Unmarshal([]byte(doc), &o); err != nil {
		t.Fatal(err)
	}
	if got := StateText(o); got != "first second third fourth fifth" {
		t.Errorf("StateText = %q", got)
	}
	keys := make([]string, len(o))
	for i, m := range o {
		keys[i] = m.Key
	}
	if strings.Join(keys, ",") != "z,a,m,n,o" {
		t.Errorf("keys = %v", keys)
	}
	if _, ok := o[1].Value.(Object); !ok {
		t.Errorf("a nested object must be an Object, got %T", o[1].Value)
	}
	// The same document through a map is read in sorted key order, which is the difference.
	var m map[string]any
	if err := json.Unmarshal([]byte(doc), &m); err != nil {
		t.Fatal(err)
	}
	if StateText(m) == StateText(o) {
		t.Error("a map and an Object of the same JSON should read in different orders here")
	}
}

// A number decodes to a json.Number, so a serializer can keep an int an int and a float a float, and
// the language verdict ignores it: a number is not text, though json.Number's Go kind is string.
func TestObjectUnmarshalJSONKeepsNumbersAndTheyAreNotText(t *testing.T) {
	var o Object
	doc := `{"a":1,"b":2.0,"c":1e16,"d":[3,{"e":-0}],"text":"Ich kann mich nicht einloggen"}`
	if err := json.Unmarshal([]byte(doc), &o); err != nil {
		t.Fatal(err)
	}
	want := map[string]json.Number{"a": "1", "b": "2.0", "c": "1e16"}
	for _, m := range o[:3] {
		if n, ok := m.Value.(json.Number); !ok || n != want[m.Key] {
			t.Errorf("%s = %#v, want json.Number(%q)", m.Key, m.Value, want[m.Key])
		}
	}
	if got := StateText(o); got != "Ich kann mich nicht einloggen" {
		t.Errorf("StateText = %q; a number must not be read as text", got)
	}
	if got := StateText([]json.Number{"12", "34"}); got != "" {
		t.Errorf("StateText of a json.Number slice = %q", got)
	}
	if got := StateText(map[string]any{"n": json.Number("7"), "s": "text"}); got != "text" {
		t.Errorf("StateText of a map with a json.Number = %q", got)
	}
}

func TestObjectUnmarshalJSONRefusesWhatIsNotAnObject(t *testing.T) {
	var o Object
	for _, doc := range []string{`[1]`, `"x"`, `{"a":`, `{"a":1`, `{`} {
		if err := json.Unmarshal([]byte(doc), &o); err == nil {
			t.Errorf("%s must be refused", doc)
		}
	}
	var holder struct{ State Object }
	if err := json.Unmarshal([]byte(`{"State":{"b":"1","a":"2"}}`), &holder); err != nil || StateText(holder.State) != "1 2" {
		t.Errorf("Object inside a struct: %v %q", err, StateText(holder.State))
	}
}

// A repeated key keeps its first position and takes its last value, as Python's json.loads
// does; keeping both members would read text Python never sees and can flip the verdict.
func TestObjectUnmarshalJSONRepeatedKeyIsLastWins(t *testing.T) {
	var o Object
	doc := `{"a":"Ich kann mich nicht anmelden bitte hilfe","b":"please help me with my card refund today","a":"ok"}`
	if err := json.Unmarshal([]byte(doc), &o); err != nil {
		t.Fatal(err)
	}
	if len(o) != 2 || o[0].Key != "a" || o[0].Value != "ok" || o[1].Key != "b" {
		t.Fatalf("members = %#v", o)
	}
	if got := StateText(o); got != "ok please help me with my card refund today" {
		t.Errorf("StateText = %q", got)
	}
	if !IsEnglish(o) {
		t.Error("Python reads this state as English")
	}
}

// null is a no-op, as encoding/json treats it for its own types; trailing data is refused.
func TestObjectUnmarshalJSONNullAndTrailingData(t *testing.T) {
	o := Object{{Key: "k", Value: "kept"}}
	if err := json.Unmarshal([]byte(`null`), &o); err != nil || len(o) != 1 {
		t.Errorf("null: err %v, members %v", err, o)
	}
	var holder struct{ State Object }
	if err := json.Unmarshal([]byte(`{"State":null}`), &holder); err != nil || holder.State != nil {
		t.Errorf("null field: err %v, state %v", err, holder.State)
	}
	var direct Object
	for _, doc := range []string{`{} x`, `{"a":[1}`, `{"a":"b"} {}`, "\u00a0null"} {
		if err := direct.UnmarshalJSON([]byte(doc)); err == nil {
			t.Errorf("%s must be refused", doc)
		}
	}
}

func TestScriptConstantsAreTheNamesPythonUses(t *testing.T) {
	for script, text := range map[Script]string{
		ScriptLatin: "hello", ScriptHan: "你好吗", ScriptDevanagari: "नमस्ते", ScriptCyrillic: "привет",
		ScriptGreek: "γεια", ScriptHangul: "안녕하세요", ScriptKana: "こんにちは", ScriptUnknown: "123",
	} {
		if got := DetectScript(text); got != script {
			t.Errorf("DetectScript(%q) = %q, want %q", text, got, script)
		}
	}
	if string(ScriptOther) != "other" || len(scriptNames) != 25 {
		t.Errorf("ScriptOther = %q, %d named scripts", ScriptOther, len(scriptNames))
	}
}
