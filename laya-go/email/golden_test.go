package email

import (
	"reflect"
	"testing"
	"unicode"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/goldens"
)

// Every case below was answered by laya/email.py: the cleaned bodies, the budgets, and the
// states.

func TestCleanBodyMatchesPython(t *testing.T) {
	cases := fixtures.List(t, fixtures.Load(t, "email.json").Must("cleaned"))
	if len(cases) < 90 {
		t.Fatalf("golden holds %d cleaned cases, want the whole corpus", len(cases))
	}
	for _, c := range cases {
		name := fixtures.Str(t, fixtures.Field(t, c, "name"))
		t.Run(name, func(t *testing.T) {
			body := fixtures.Str(t, fixtures.Field(t, c, "body"))
			want := fixtures.Str(t, fixtures.Field(t, c, "result"))
			if got := CleanBody(body, DefaultMaxChars); got != want {
				t.Errorf("CleanBody = %q\n  python    = %q", got, want)
			}
		})
	}
}

func TestBudgetsMatchPython(t *testing.T) {
	for _, c := range fixtures.List(t, fixtures.Load(t, "email.json").Must("budgets")) {
		name := fixtures.Str(t, fixtures.Field(t, c, "name"))
		t.Run(name, func(t *testing.T) {
			body := fixtures.Str(t, fixtures.Field(t, c, "body"))
			n := fixtures.Int(t, fixtures.Field(t, c, "max_chars"))
			want := fixtures.Str(t, fixtures.Field(t, c, "result"))
			if got := CleanBody(body, n); got != want {
				t.Errorf("CleanBody(maxChars=%d) = %q\n  python = %q", n, got, want)
			}
		})
	}
}

func TestStateMatchesPython(t *testing.T) {
	for _, c := range fixtures.List(t, fixtures.Load(t, "email.json").Must("states")) {
		name := fixtures.Str(t, fixtures.Field(t, c, "name"))
		t.Run(name, func(t *testing.T) {
			opts := []Option{WithMaxChars(fixtures.Int(t, fixtures.Field(t, c, "max_chars")))}
			if sender := fixtures.Field(t, c, "sender"); sender != nil {
				opts = append(opts, WithSender(fixtures.Str(t, sender)))
			}
			if !fixtures.Bool(t, fixtures.Field(t, c, "clean")) {
				opts = append(opts, WithoutCleaning())
			}
			for _, m := range fixtures.Obj(t, fixtures.Field(t, c, "extra")) {
				opts = append(opts, WithField(m.Key, goldens.Value(m.Value)))
			}
			got, err := State(fixtures.Str(t, fixtures.Field(t, c, "subject")),
				fixtures.Str(t, fixtures.Field(t, c, "body")), opts...)
			if err != nil {
				t.Fatal(err)
			}
			if want := goldens.Value(fixtures.Field(t, c, "result")); !reflect.DeepEqual(any(got), want) {
				t.Errorf("State = %#v\n  python = %#v", got, want)
			}
		})
	}
}

// The patterns run on text folded the way re.I folds it, which is not Go's (?i): Python also
// matches dotless i and long s. This holds the fold to CPython over every code point, for every
// letter the patterns spell.
func TestFoldMatchesReIgnoreCase(t *testing.T) {
	g := fixtures.Obj(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json"), "email_fold"))
	alphabet := fixtures.Str(t, g.Must("alphabet"))
	want := map[rune]map[rune]bool{}
	for _, l := range alphabet {
		want[l] = map[rune]bool{l: true}
	}
	for _, m := range fixtures.Obj(t, g.Must("orbit")) {
		for _, cp := range fixtures.List(t, m.Value) {
			want[[]rune(m.Key)[0]][rune(fixtures.Int(t, cp))] = true
		}
	}
	got := map[rune]map[rune]bool{}
	for _, l := range alphabet {
		got[l] = map[rune]bool{}
	}
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if r >= 0xD800 && r <= 0xDFFF {
			continue
		}
		if set, ok := got[fold(r)]; ok {
			set[r] = true
		}
	}
	for l := range want {
		if !reflect.DeepEqual(got[l], want[l]) {
			t.Errorf("code points folding to %q: got %v, python %v", l, got[l], want[l])
		}
	}
	// every non-ASCII letter a pattern spells has to be one the golden covers
	for r := range placeholders {
		if _, ok := want[r]; !ok {
			t.Errorf("pattern spells %q, which the fold golden does not cover", r)
		}
	}
}

func TestCleanBodyFuzzMatchesPython(t *testing.T) {
	cases := fixtures.List(t, fixtures.Obj(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json"), "email_fuzz")).Must("cases"))
	if len(cases) != 1500 {
		t.Fatalf("golden holds %d fuzz cases, want 1500", len(cases))
	}
	bad := 0
	for i, c := range cases {
		body := fixtures.Str(t, fixtures.Field(t, c, "body"))
		n := fixtures.Int(t, fixtures.Field(t, c, "max_chars"))
		want := fixtures.Str(t, fixtures.Field(t, c, "result"))
		if got := CleanBody(body, n); got != want {
			if bad++; bad <= 5 {
				t.Errorf("case %d (maxChars %d): CleanBody(%q) = %q, python %q", i, n, body, got, want)
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d fuzz cases differ from Python", bad, len(cases))
	}
}
