package sequence_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

// TestStatesGolden holds ParseState and SerializeState to json.loads and serialize_state on 2,000
// random JSON texts: int and float spellings of every size, NaN and Infinity, repeated keys,
// escapes, whitespace. The texts Python rejects are rejected, and so are the lone surrogates it
// accepts and cannot encode.
func TestStatesGolden(t *testing.T) {
	root := fixtures.Obj(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json"), "states"))
	cases := fixtures.List(t, root.Must("cases"))
	if len(cases) != 2000 {
		t.Fatalf("the golden holds %d states", len(cases))
	}
	tok := toy(t)
	bad := 0
	for i, c := range cases {
		rec := fixtures.Obj(t, c)
		text := fixtures.Str(t, rec.Must("text"))
		state, err := sequence.ParseState([]byte(text))
		if err != nil {
			t.Errorf("state %d %.60q: %v", i, text, err)
			bad++
			continue
		}
		got, err := sequence.SerializeState(state)
		if want := fixtures.Str(t, rec.Must("serialized")); err != nil || got != want {
			t.Errorf("state %d %.60q\n got %.100q (%v)\nwant %.100q", i, text, got, err, want)
			bad++
			continue
		}
		enc, err := sequence.EncodeState(tok, state)
		wantKeep := sequence.KeepStart
		if fixtures.Bool(t, rec.Must("list")) {
			wantKeep = sequence.KeepEnd
		}
		if err != nil || enc.Keep != wantKeep {
			t.Errorf("state %d %.60q: keep %q (%v), want %q", i, text, enc.Keep, err, wantKeep)
			bad++
		}
		if bad > 10 {
			t.Fatal("too many differences")
		}
	}
	for _, text := range fixtures.Strings(t, root.Must("python_rejects")) {
		if _, err := sequence.ParseState([]byte(text)); err == nil {
			t.Errorf("accepted %.60q, which Python rejects", text)
		}
	}
	for _, text := range fixtures.Strings(t, root.Must("go_refuses")) {
		if _, err := sequence.ParseState([]byte(text)); err == nil || !strings.Contains(err.Error(), "lone surrogate") {
			t.Errorf("%.60q: err = %v, want a lone-surrogate refusal", text, err)
		}
	}
}

// TestMarkerRefusalsGolden holds Build's refusal to Agent._encode_state's: the same questions and
// budgets succeed or fail together, with the same text.
func TestMarkerRefusalsGolden(t *testing.T) {
	golden := fixtures.Obj(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json"), "marker_refusals"))
	for _, name := range checkpoints {
		tok := checkpoint(t, name)
		cases := fixtures.List(t, fixtures.Obj(t, fixtures.Field(t, golden, name)).Must("cases"))
		if len(cases) != 42 {
			t.Fatalf("%s: %d cases", name, len(cases))
		}
		refused := 0
		for _, c := range cases {
			rec := fixtures.Obj(t, c)
			options, words := fixtures.Int(t, rec.Must("options")), fixtures.Int(t, rec.Must("words"))
			b := sequence.Budget{MaxLen: fixtures.Int(t, rec.Must("max_len")), HeadMaxLen: fixtures.Int(t, rec.Must("head_max_len"))}
			t.Run(fmt.Sprintf("%s/%d options x %d words/%d,%d", name, options, words, b.MaxLen, b.HeadMaxLen), func(t *testing.T) {
				opts := make([]schema.Option, options)
				for i := range opts {
					parts := make([]string, words)
					for j := range parts {
						parts[j] = fmt.Sprintf("w%d_%d", i, j)
					}
					opts[i] = schema.Option{Label: fmt.Sprintf("opt%03d", i), Criterion: strings.Join(parts, " ")}
				}
				q := schema.MustQuestion(schema.Choice("Pick one.", opts...))
				st, err := sequence.EncodeState(tok, "state text")
				if err != nil {
					t.Fatal(err)
				}
				_, err = sequence.Build(tok, q, st, b)
				var me *sequence.MarkersError
				want := rec.Must("error")
				switch {
				case want == nil && err != nil:
					t.Errorf("refused: %v", err)
				case want != nil && !errors.As(err, &me):
					t.Errorf("not refused; Python says %q", want)
				case want != nil:
					refused++
					me.ID = "it's"
					if me.Error() != want.(string) {
						t.Errorf("text\n got %q\nwant %q", me.Error(), want)
					}
				}
			})
		}
		if refused < 5 {
			t.Errorf("%s: only %d refusals", name, refused)
		}
	}
}
