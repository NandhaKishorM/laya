package agent_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io/fs"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
)

func withTemps(t *testing.T, langs map[string]map[string]any, tag string) string {
	t.Helper()
	s := &fakeSession{}
	a, err := agent.Open(toyCheckpoint(`{}`), s, agent.WithLangTemperatures(langs))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	var opts []agent.CallOption
	if tag != "" {
		opts = append(opts, agent.WithLanguage(tag))
	}
	r, err := a.Predict(context.Background(), "state", questions(t), opts...)
	if err != nil {
		t.Fatal(err)
	}
	return jsonOf(t, r)
}

func zh() map[string]map[string]any {
	return map[string]map[string]any{"zh": {"temperature": []any{2.0, 1.5, 3.0}, "temperature_by_options": map[string]any{"choice:2": 2.5}}}
}

func TestLangTemperaturesSelectByPrimarySubtag(t *testing.T) {
	plain := withTemps(t, zh(), "")
	over := withTemps(t, zh(), "zh")
	if plain == over {
		t.Fatal("the zh override changed nothing")
	}
	for _, tag := range []string{"zh-CN", "ZH", "Zh-TW"} {
		if got := withTemps(t, zh(), tag); got != over {
			t.Errorf("%q did not select the zh override", tag)
		}
	}
	for _, tag := range []string{"en", "yo-NG", "zhx", "z", "zh_Hans"} {
		if got := withTemps(t, zh(), tag); got != plain {
			t.Errorf("%q selected the zh override", tag)
		}
	}
	if got := withTemps(t, nil, "zh"); got != plain {
		t.Error("no overrides must change nothing")
	}
}

func TestLangTemperaturesNilEntryInherits(t *testing.T) {
	plain := withTemps(t, nil, "")
	for name, langs := range map[string]map[string]map[string]any{
		"nil entry":       {"zh": nil},
		"nil temperature": {"zh": {"temperature": nil}},
		"empty entry":     {"zh": {}},
	} {
		if got := withTemps(t, langs, "zh"); got != plain {
			t.Errorf("%s: did not inherit the checkpoint's temperatures", name)
		}
	}
}

func TestLangTemperaturesAreRefusedAtOpen(t *testing.T) {
	for name, langs := range map[string]map[string]map[string]any{
		"not a list":     {"zh": {"temperature": "hot"}},
		"short list":     {"zh": {"temperature": []any{1.0}}},
		"bad by_options": {"zh": {"temperature_by_options": []any{1}}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := agent.Open(toyCheckpoint(`{}`), &fakeSession{}, agent.WithLangTemperatures(langs))
			var ce *agent.ConfigError
			if !errors.As(err, &ce) || ce.Field != "lang_temperatures" {
				t.Fatalf("err = %v, want a *ConfigError on lang_temperatures", err)
			}
		})
	}
}

// A value that is not a number is not refused: clamp_temperature reads it as 1.0, as Python does.
func TestLangTemperaturesNonNumbersReadAsOne(t *testing.T) {
	ones := withTemps(t, map[string]map[string]any{"zh": {"temperature": []any{1.0, 1.0, 1.0}, "temperature_by_options": map[string]any{"choice:2": 1.0}}}, "zh")
	if ones == withTemps(t, zh(), "zh") {
		t.Fatal("the control does not differ from the override")
	}
	got := withTemps(t, map[string]map[string]any{"zh": {"temperature": []any{"a", true, nil}, "temperature_by_options": map[string]any{"choice:2": "x"}}}, "zh")
	if got != ones {
		t.Error("a non-number was not read as 1.0")
	}
}

func hexOf(b string) string { s := sha256.Sum256([]byte(b)); return hex.EncodeToString(s[:]) }

func TestExpectedSHA256(t *testing.T) {
	cp := toyCheckpoint(`{}`)
	good := hexOf(`{}`)
	open := func(m map[string]string, fsys fs.FS) error {
		s := &fakeSession{}
		a, err := agent.Open(fsys, s, agent.WithExpectedSHA256(m))
		if err == nil {
			a.Close()
		} else if s.closes != 1 {
			t.Errorf("the session was closed %d times on a refused open", s.closes)
		}
		return err
	}
	if err := open(map[string]string{"rl_agent_config.json": good}, cp); err != nil {
		t.Fatal(err)
	}
	if err := open(nil, cp); err != nil {
		t.Fatal(err)
	}
	err := open(map[string]string{"rl_agent_config.json": hexOf("x")}, cp)
	if !errors.Is(err, agent.ErrDigestMismatch) || !strings.Contains(err.Error(), "rl_agent_config.json") || !strings.Contains(err.Error(), good) {
		t.Fatalf("mismatch err = %v", err)
	}
	if err := open(map[string]string{"gone.onnx": good}, cp); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("missing err = %v", err)
	}
	for _, p := range []string{"../x", "/etc/passwd", "a/../../x", "C:\\x"} {
		if err := open(map[string]string{p: good}, cp); !errors.Is(err, agent.ErrUnsafeDigestPath) {
			t.Errorf("%q: err = %v", p, err)
		}
	}
}

func TestDigestsAreVerifiedBeforeTheConfigIsParsed(t *testing.T) {
	cp := toyCheckpoint(`{not json`)
	_, err := agent.Open(cp, &fakeSession{}, agent.WithExpectedSHA256(map[string]string{"rl_agent_config.json": hexOf("x")}))
	if !errors.Is(err, agent.ErrDigestMismatch) {
		t.Fatalf("err = %v, want the digest mismatch first", err)
	}
}
