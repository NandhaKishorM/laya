package email

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/lang"
)

func TestStateOptions(t *testing.T) {
	t.Run("order and drops", func(t *testing.T) {
		got, err := State("  Refund \t", "Please refund.", WithSender("ana@x.com"),
			WithField("priority", "high"), WithField("thread_id", nil), WithField("ticket", 42))
		if err != nil {
			t.Fatal(err)
		}
		want := lang.Object{{Key: "subject", Value: "Refund"}, {Key: "body", Value: "Please refund."},
			{Key: "from", Value: "ana@x.com"}, {Key: "priority", Value: "high"}, {Key: "ticket", Value: 42}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("State = %#v, want %#v", got, want)
		}
	})
	t.Run("empty sender adds no field", func(t *testing.T) {
		got, _ := State("s", "b", WithSender(""))
		if len(got) != 2 {
			t.Errorf("State = %#v, want subject and body only", got)
		}
	})
	t.Run("a field named from replaces the sender in place", func(t *testing.T) {
		got, _ := State("s", "b", WithSender("ana@x.com"), WithField("x", 1), WithField("from", "bo@x.com"))
		want := lang.Object{{Key: "subject", Value: "s"}, {Key: "body", Value: "b"},
			{Key: "from", Value: "bo@x.com"}, {Key: "x", Value: 1}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("State = %#v, want %#v", got, want)
		}
	})
	t.Run("reserved names are refused", func(t *testing.T) {
		for _, key := range reserved {
			if _, err := State("s", "b", WithField(key, "x")); err == nil {
				t.Errorf("WithField(%q) was accepted", key)
			}
		}
	})
	t.Run("without cleaning the body is whole", func(t *testing.T) {
		body := strings.Repeat("y", 50) + "\n\nSent from my iPhone"
		got, _ := State("s", body, WithoutCleaning(), WithMaxChars(5))
		if got[1].Value != body {
			t.Errorf("body = %q, want it untouched", got[1].Value)
		}
	})
	t.Run("budget reaches the cleaner", func(t *testing.T) {
		got, _ := State("s", strings.Repeat("y", 50), WithMaxChars(7))
		if got[1].Value != "yyyyyyy" {
			t.Errorf("body = %q", got[1].Value)
		}
	})
}

// RE2 has no catastrophic case, but a very long line must still be bounded by the budget: a
// quarter-megabyte line starting like an attribution is the shape that would find one.
func TestLongLinesStayFast(t *testing.T) {
	for name, line := range map[string]string{
		"attribution head": "On " + strings.Repeat("x", 250_000),
		"tail candidate":   strings.Repeat("a@b ", 60_000) + ":",
		"footer words":     strings.Repeat("This email is confidential ", 6_000),
		"spaces":           strings.Repeat(" ", 250_000) + "x",
	} {
		t.Run(name, func(t *testing.T) {
			start := time.Now()
			CleanBody("first\n"+line, 60_000)
			if d := time.Since(start); d > 30*time.Second {
				t.Errorf("took %v", d)
			}
		})
	}
}

func TestCutRunes(t *testing.T) {
	text := []rune("abcdef")
	for limit, want := range map[int64]string{0: "", 3: "abc", 6: "abcdef", 99: "abcdef", -1: "abcde", -6: "", -9: ""} {
		if got := string(cutRunes(text, limit)); got != want {
			t.Errorf("cutRunes(%d) = %q, want %q", limit, got, want)
		}
	}
}

// A literal space in a marker is U+0020 alone in Python, and the footer's `using [a-z]...` is
// ASCII alone; the other spaces and digits are \s and \d only. Each input below is kept whole.
func TestLiteralSpacesAndDigitsStayASCII(t *testing.T) {
	for _, body := range []string{
		"Hi\nI need help.\nEm seg, 1 de jan de 2024 escreveu:\nold",
		"Hi\nI need help.\nOn　Monday 5 Jan 2026 Ana wrote:\nold",
		"Hi\nI need help.\n-----Original Message-----\nold",
		"a\nb\nc\nd\ne\nThank you John\nsig",
		"a\nb\nc\nd\ne\nSent　from my iPhone\nsig",
		"Hi\nI need help.\nEm\x0eseg, 1 de jan de 2024 escreveu:\nold",
		"a\nb\nc\nd\ne\nThank\x0eyou John\nsig",
		"a\nb\nc\nd\ne\nSent from my iPhone using x\x0f\nsig",
		"a\nb\nc\nd\ne\nSent from my iPhone using x٣\nsig",
		"a\nb\nc\nd\ne\nSent from my iPhone using x１\nsig",
	} {
		if got := CleanBody(body, DefaultMaxChars); got != body {
			t.Errorf("CleanBody(%q) = %q, Python keeps it whole", body, got)
		}
	}
}

// helpers whose edge is a non-ASCII space: the trailing colon guard and the quote-line test.
func TestNonASCIISpaceAtLineEdges(t *testing.T) {
	for body, want := range map[string]string{
		"Hi\nI need help.\nana@x.com> escreveu: \nold":   "Hi\nI need help.",
		"Hi\nI need help.\nbob@x.com> wrote:\u3000\nold": "Hi\nI need help.",
		"Hi\nI need help.\n > quoted\nmore":              "Hi\nI need help.\nmore",
		"Hi\nI need help.\n\u3000>old\n\u3000>old2":      "Hi\nI need help.",
	} {
		if got := CleanBody(body, DefaultMaxChars); got != want {
			t.Errorf("CleanBody(%q) = %q, want %q", body, got, want)
		}
	}
}
