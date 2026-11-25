package router_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/router"
)

func hexOf(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }

const cfgFile = "rl_agent_config.json"

func newDigestRouter(t *testing.T, env string, opts ...router.Option) *router.Router {
	t.Helper()
	t.Setenv("LAYA_SHA256_DIGESTS", env)
	r, err := router.New(opts...)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// digestFactory builds the toy checkpoint with what ExpectedSHA256 says, as a real factory does.
func digestFactory(rp **router.Router, shared map[string]string) router.AgentFactory {
	return func(_ context.Context, c router.Checkpoint) (*agent.Agent, error) {
		var opts []agent.Option
		exp, err := (*rp).ExpectedSHA256(string(c), shared)
		if err != nil {
			return nil, err
		}
		if exp != nil {
			opts = append(opts, agent.WithExpectedSHA256(exp))
		}
		return agent.Open(toyCheckpoint(), &probeSession{}, opts...)
	}
}

func loadVia(t *testing.T, env string, shared map[string]string, name string, opts ...router.Option) error {
	t.Helper()
	var r *router.Router
	t.Setenv("LAYA_SHA256_DIGESTS", env)
	r, err := router.New(append(opts, router.WithAgentFactory(digestFactory(&r, shared)))...)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	a, err := r.Load(context.Background(), name)
	if err == nil && a == nil {
		t.Fatal("nil agent")
	}
	return err
}

func TestDigestsEnvLayering(t *testing.T) {
	good, bad := hexOf(`{}`), hexOf("x")
	flat := `{"` + cfgFile + `":"` + bad + `"}`
	nested := `{"english":{"` + cfgFile + `":"` + bad + `"},"multilingual":{"` + cfgFile + `":"` + good + `"}}`

	t.Run("flat is left to the agent", func(t *testing.T) {
		r := newDigestRouter(t, flat)
		if n := len(r.SHA256Digests()); n != 0 {
			t.Fatalf("table = %v", r.SHA256Digests())
		}
		if err := loadVia(t, flat, nil, "english"); !errors.Is(err, agent.ErrDigestMismatch) {
			t.Fatalf("flat env must still reach the agent: %v", err)
		}
	})
	t.Run("nested is split per checkpoint", func(t *testing.T) {
		if err := loadVia(t, nested, nil, "english"); !errors.Is(err, agent.ErrDigestMismatch) {
			t.Fatalf("english = %v", err)
		}
		if err := loadVia(t, nested, nil, "multilingual"); err != nil {
			t.Fatalf("multilingual = %v", err)
		}
	})
	t.Run("a checkpoint the nested map omits is masked, not read as an artifact map", func(t *testing.T) {
		if err := loadVia(t, nested, nil, "typed-decisions"); err != nil {
			t.Fatalf("typed-decisions = %v", err)
		}
		r := newDigestRouter(t, nested)
		got := r.SHA256Digests()
		if len(got) != 3 || got["typed-decisions"] == nil || len(got["typed-decisions"]) != 0 {
			t.Fatalf("placeholder = %#v", got)
		}
	})
	t.Run("keys are normalised", func(t *testing.T) {
		r := newDigestRouter(t, `{"EN ":{"a":"b"}}`)
		if got := r.SHA256Digests()["english"]; !reflect.DeepEqual(got, map[string]string{"a": "b"}) {
			t.Fatalf("english = %v", got)
		}
	})
	t.Run("a later spelling of one checkpoint wins", func(t *testing.T) {
		r := newDigestRouter(t, `{"english":{"a":"1"},"en":{"a":"2"}}`)
		if got := r.SHA256Digests()["english"]["a"]; got != "2" {
			t.Fatalf("a = %q", got)
		}
	})
	t.Run("unparseable or empty is not this layer's error", func(t *testing.T) {
		for _, v := range []string{"", "  ", "{not json", "[]", `"d"`, "{}", "null", "7"} {
			if n := len(newDigestRouter(t, v).SHA256Digests()); n != 0 {
				t.Errorf("%q: table has %d entries", v, n)
			}
		}
	})
}

func TestDigestsEnvErrors(t *testing.T) {
	for _, c := range []struct{ env, want string }{
		{`{"engligh":{"a":"b"}}`, "unknown model 'engligh'"},
		{`{"model.safetensors":"x","english":{"a":"b"}}`, "LAYA_SHA256_DIGESTS must be either {artifact: digest} for every checkpoint or {model: {artifact: digest}} per checkpoint; ['english', 'model.safetensors'] mixes the two or holds a value that is neither"},
		{`{"english":{"a":"b"},"x":1}`, "mixes the two"},
		{`{"english":null}`, "mixes the two"},
		{`{"english":{"a":1}}`, `the digest of "a" in "english" must be a string`},
	} {
		t.Setenv("LAYA_SHA256_DIGESTS", c.env)
		_, err := router.New()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v", c.env, err)
		}
	}
	t.Setenv("LAYA_SHA256_DIGESTS", `{"engligh":{}}`)
	var ue *router.UnknownModelError
	if _, err := router.New(); !errors.As(err, &ue) {
		t.Errorf("an unknown name is an *UnknownModelError: %v", err)
	}
}

func TestDigestsArgument(t *testing.T) {
	a, b := map[string]string{"model.safetensors": "a"}, map[string]string{"model.safetensors": "b"}
	t.Run("keys are normalised", func(t *testing.T) {
		r := newDigestRouter(t, "", router.WithSHA256Digests(map[string]map[string]string{"ml": a, "typed": nil}))
		got := r.SHA256Digests()
		if len(got) != 2 || !reflect.DeepEqual(got["multilingual"], a) {
			t.Fatalf("table = %v", got)
		}
		if v, ok := got["typed-decisions"]; !ok || v != nil {
			t.Fatalf("explicit nil must stay an entry: %v", got)
		}
		if n := len(newDigestRouter(t, "").SHA256Digests()); n != 0 {
			t.Fatalf("no pins, no entries: %d", n)
		}
	})
	t.Run("a misspelled name fails New", func(t *testing.T) {
		t.Setenv("LAYA_SHA256_DIGESTS", "")
		var ue *router.UnknownModelError
		_, err := router.New(router.WithSHA256Digests(map[string]map[string]string{"engligh": a}))
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("the argument replaces the environment's entry whole", func(t *testing.T) {
		env := `{"english":{"x":"1","y":"2"},"multilingual":{"model.safetensors":"b"}}`
		r := newDigestRouter(t, env, router.WithSHA256Digests(map[string]map[string]string{"english": a}))
		got := r.SHA256Digests()
		if !reflect.DeepEqual(got["english"], a) || !reflect.DeepEqual(got["multilingual"], b) {
			t.Fatalf("table = %v", got)
		}
	})
	t.Run("an explicit nil masks a flat environment map for that checkpoint only", func(t *testing.T) {
		flat := `{"` + cfgFile + `":"` + hexOf("x") + `"}`
		opt := router.WithSHA256Digests(map[string]map[string]string{"english": nil})
		if err := loadVia(t, flat, nil, "english", opt); err != nil {
			t.Fatalf("english = %v", err)
		}
		if err := loadVia(t, flat, nil, "multilingual", opt); !errors.Is(err, agent.ErrDigestMismatch) {
			t.Fatalf("multilingual keeps the environment: %v", err)
		}
	})
	t.Run("a later option replaces an earlier one per name", func(t *testing.T) {
		r := newDigestRouter(t, "",
			router.WithSHA256Digests(map[string]map[string]string{"english": a, "multilingual": a}),
			router.WithSHA256Digests(map[string]map[string]string{"english": b}))
		got := r.SHA256Digests()
		if !reflect.DeepEqual(got["english"], b) || !reflect.DeepEqual(got["multilingual"], a) {
			t.Fatalf("table = %v", got)
		}
	})
	t.Run("the caller's map is copied", func(t *testing.T) {
		in := map[string]string{"f": "1"}
		r := newDigestRouter(t, "", router.WithSHA256Digests(map[string]map[string]string{"english": in}))
		in["f"] = "2"
		r.SHA256Digests()["english"]["f"] = "3"
		if got := r.SHA256Digests()["english"]["f"]; got != "1" {
			t.Fatalf("f = %q", got)
		}
	})
}

func TestExpectedSHA256Merge(t *testing.T) {
	shared := map[string]string{"tokenizer.json": "d", "model.safetensors": "c"}
	per := map[string]string{"model.safetensors": "b"}
	r := newDigestRouter(t, "", router.WithSHA256Digests(map[string]map[string]string{"english": per, "multilingual": nil}))
	cases := []struct {
		name   string
		shared map[string]string
		want   map[string]string
	}{
		{"english", nil, per},
		{"english", shared, map[string]string{"tokenizer.json": "d", "model.safetensors": "b"}}, // per-checkpoint wins
		{"multilingual", shared, shared},           // nil entry is {} and keeps shared
		{"multilingual", nil, map[string]string{}}, // nil entry still masks
		{"typed-decisions", nil, nil},              // nothing says anything
		{"typed-decisions", map[string]string{}, map[string]string{}},
		{"typed-decisions", shared, shared},
	}
	for _, c := range cases {
		got, err := r.ExpectedSHA256(c.name, c.shared)
		if err != nil || (got == nil) != (c.want == nil) || !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s %v = %#v, %v, want %#v", c.name, c.shared, got, err, c.want)
		}
	}
	got, _ := r.ExpectedSHA256("english", shared)
	got["x"] = "y"
	if _, leaked := shared["x"]; leaked {
		t.Fatal("the result must be a fresh map")
	}
	only, _ := r.ExpectedSHA256("english", nil) // no shared pin: the entry itself is tempting to return
	only["x"] = "y"
	if _, leaked := r.SHA256Digests()["english"]["x"]; leaked {
		t.Fatal("the result must not alias the table")
	}
	var ue *router.UnknownModelError
	if _, err := r.ExpectedSHA256("nope", nil); !errors.As(err, &ue) {
		t.Fatalf("unknown name: %v", err)
	}
	if got, err := r.ExpectedSHA256(" EN ", nil); err != nil || !reflect.DeepEqual(got, per) {
		t.Fatalf("alias: %v %v", got, err)
	}
}

func TestSharedAndPerCheckpointPinsBothVerify(t *testing.T) {
	good, bad := hexOf(`{}`), hexOf("x")
	// shared pins the config wrongly, the per-checkpoint entry corrects it: loads
	opt := router.WithSHA256Digests(map[string]map[string]string{"english": {cfgFile: good}})
	if err := loadVia(t, "", map[string]string{cfgFile: bad}, "english", opt); err != nil {
		t.Fatalf("override: %v", err)
	}
	// no per-checkpoint entry: shared alone is verified
	if err := loadVia(t, "", map[string]string{cfgFile: bad}, "multilingual", opt); !errors.Is(err, agent.ErrDigestMismatch) {
		t.Fatalf("shared alone: %v", err)
	}
	// a file only the per-checkpoint entry names is verified too
	opt = router.WithSHA256Digests(map[string]map[string]string{"english": {cfgFile: bad}})
	if err := loadVia(t, "", map[string]string{"tokenizer/tokenizer.json": hexOf("whatever")}, "english", opt); !errors.Is(err, agent.ErrDigestMismatch) {
		t.Fatalf("per-checkpoint file: %v", err)
	}
}

func TestDigestsConcurrentUse(t *testing.T) {
	r := newDigestRouter(t, `{"english":{"a":"1"}}`)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				got, err := r.ExpectedSHA256("english", map[string]string{"s": "t"})
				if err != nil || got["a"] != "1" || got["s"] != "t" {
					t.Errorf("got %v, %v", got, err)
					return
				}
				got["a"] = "mutated"
				r.SHA256Digests()["english"]["a"] = "mutated"
			}
		}()
	}
	wg.Wait()
	if got := r.SHA256Digests()["english"]["a"]; got != "1" {
		t.Fatalf("a = %q", got)
	}
}

// Python's json.loads reads NaN, Infinity and -Infinity; encoding/json does not. Read as "not
// JSON", the whole variable would name nothing and every nested pin in it would vanish.
func TestDigestsEnvNonFiniteNumbersAreNotDropped(t *testing.T) {
	for _, c := range []struct{ env, want string }{
		{`{"english":{"model.safetensors":"` + hexOf("a") + `"},"multilingual":{"a":NaN}}`, `the digest of "a" in "multilingual" must be a string`},
		{`{"english":{"a":-Infinity}}`, `must be a string`},
		{`{"english":NaN}`, "mixes the two"},
		{`{"english":{"a":"b"},"x":Infinity}`, "mixes the two"},
		{`{"a":-Infinity}`, "x"}, // flat-shaped but not all strings: refused as mixed
	} {
		t.Setenv("LAYA_SHA256_DIGESTS", c.env)
		_, err := router.New()
		if err == nil || (c.want != "x" && !strings.Contains(err.Error(), c.want)) {
			t.Errorf("%s: err = %v", c.env, err)
		}
	}
	// The same words inside strings, and a value that is not a bare token, change nothing.
	r := newDigestRouter(t, `{"english":{"a":"NaN"},"multilingual":{"Infinity":"-Infinity"}}`)
	if got := r.SHA256Digests(); got["english"]["a"] != "NaN" || got["multilingual"]["Infinity"] != "-Infinity" {
		t.Errorf("strings were rewritten: %v", got)
	}
	for _, env := range []string{`NaN`, `[NaN]`, `{"a":NaNx}`, `{"a":1NaN}`} { // not objects / not JSON: names nothing
		r := newDigestRouter(t, env)
		if len(r.SHA256Digests()) != 0 {
			t.Errorf("%s: table = %v", env, r.SHA256Digests())
		}
	}
}

// Python keeps whichever of two spellings comes last in dict order; Go refuses, so no pin can be
// dropped by an ordering the caller cannot see.
func TestDigestsArgumentTwoSpellingsRefused(t *testing.T) {
	t.Setenv("LAYA_SHA256_DIGESTS", "")
	x := map[string]string{"model.safetensors": hexOf("x")}
	y := map[string]string{"model.safetensors": hexOf("y")}
	_, err := router.New(router.WithSHA256Digests(map[string]map[string]string{"english": x, "en": y}))
	if err == nil || !strings.Contains(err.Error(), "same checkpoint") {
		t.Fatalf("err = %v", err)
	}
	// Separate options may name a checkpoint again: the later one replaces the earlier.
	r, err := router.New(router.WithSHA256Digests(map[string]map[string]string{"english": x}),
		router.WithSHA256Digests(map[string]map[string]string{"en": y}))
	if err != nil || r.SHA256Digests()["english"]["model.safetensors"] != hexOf("y") {
		t.Fatalf("later option must replace: %v %v", err, r.SHA256Digests())
	}
}

// The rewrite must stay out of strings (an escaped quote does not end one) and must not touch a
// token that is glued to other text, which Python reads as not JSON.
func TestDigestsEnvNonFiniteRewriteBounds(t *testing.T) {
	r := newDigestRouter(t, `{"english":{"a":"x\":NaN, y"}}`)
	if got := r.SHA256Digests()["english"]["a"]; got != `x":NaN, y` {
		t.Errorf("a string holding an escaped quote and NaN was rewritten: %q", got)
	}
	r = newDigestRouter(t, `{"english":{"a":"\\"},"multilingual":{"b":":,Infinity"}}`)
	if got := r.SHA256Digests()["multilingual"]["b"]; got != ":,Infinity" {
		t.Errorf("escaped backslash then string: %q", got)
	}
	t.Setenv("LAYA_SHA256_DIGESTS", `{"english":{"a":"\\"},"x":NaN}`)
	if _, err := router.New(); err == nil || !strings.Contains(err.Error(), "mixes the two") {
		t.Errorf("a bare NaN after an escaped backslash must be refused as mixed: %v", err)
	}
	for _, env := range []string{`{"english":NaNe5}`, `{"a":Infinity.5}`, `{"english":-Infinity5}`, `{"english":{"a":"q"},"b":NaN1}`} {
		r := newDigestRouter(t, env) // glued: not JSON, names nothing
		if len(r.SHA256Digests()) != 0 {
			t.Errorf("%s: table = %v", env, r.SHA256Digests())
		}
	}
}
