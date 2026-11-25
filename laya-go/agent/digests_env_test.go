package agent_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
)

func TestDigestsFromEnv(t *testing.T) {
	cp := toyCheckpoint(`{}`)
	good, bad := hexOf(`{}`), hexOf("x")
	open := func(env string, opts ...agent.Option) error {
		t.Setenv("LAYA_SHA256_DIGESTS", env)
		a, err := agent.Open(cp, &fakeSession{}, opts...)
		if err == nil {
			a.Close()
		}
		return err
	}
	if err := open(` {"rl_agent_config.json":"` + good + `"} `); err != nil {
		t.Fatal(err)
	}
	if err := open(""); err != nil {
		t.Fatal(err)
	}
	if err := open(`{"onnx":"` + bad + `","onnx_path":"` + bad + `"}`); err != nil {
		t.Fatalf("graph keys belong to onnx.Open: %v", err)
	}
	if err := open(`{"rl_agent_config.json":"` + bad + `"}`); !errors.Is(err, agent.ErrDigestMismatch) {
		t.Fatalf("env mismatch = %v", err)
	}
	if err := open(`{"rl_agent_config.json":"`+bad+`"}`, agent.WithExpectedSHA256(map[string]string{})); err != nil {
		t.Fatalf("an explicit option, even empty, ignores the variable: %v", err)
	}
	for _, v := range []string{"nope", `["a"]`, `{"rl_agent_config.json":1}`} {
		err := open(v)
		var ce *agent.ConfigError
		if !errors.As(err, &ce) || ce.Field != "LAYA_SHA256_DIGESTS" {
			t.Errorf("%q: err = %v", v, err)
		}
	}
}

func TestTemperatureByOptionsFalsyIsEmpty(t *testing.T) {
	mk := func(v any) map[string]map[string]any {
		return map[string]map[string]any{"zh": {"temperature": []any{2.0, 1.5, 3.0}, "temperature_by_options": v}}
	}
	want := withTempsFrom(t, mk(map[string]any{}))
	for _, v := range []any{false, 0, 0.0, "", []any{}, nil} {
		if got := withTempsFrom(t, mk(v)); got != want {
			t.Errorf("%#v: %s != %s", v, got, want)
		}
	}
}

func withTempsFrom(t *testing.T, langs map[string]map[string]any) string {
	t.Helper()
	return withTemps(t, langs, "zh")
}

func TestDuplicatePrimarySubtagIsRefused(t *testing.T) {
	langs := map[string]map[string]any{"PT": {"temperature": []any{2.0, 2.0, 2.0}}, "pt-BR": {"temperature": []any{3.0, 3.0, 3.0}}}
	_, err := agent.Open(toyCheckpoint(`{}`), &fakeSession{}, agent.WithLangTemperatures(langs))
	var ce *agent.ConfigError
	if !errors.As(err, &ce) || !strings.Contains(err.Error(), "PT") || !strings.Contains(err.Error(), "pt-BR") {
		t.Fatalf("err = %v", err)
	}
}
