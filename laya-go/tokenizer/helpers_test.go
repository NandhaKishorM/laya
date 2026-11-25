package tokenizer

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
)

var (
	loadMu sync.Mutex
	loaded = map[string]*Tokenizer{}
)

// checkpoint loads a shipped checkpoint's tokenizer once per test binary. Without
// LAYA_CHECKPOINTS the test is skipped, and CI fails the run on any skip, so a missing
// checkpoint cannot pass as green.
func checkpoint(tb testing.TB, name string) *Tokenizer {
	tb.Helper()
	root := fixtures.Checkpoints()
	if root == "" {
		tb.Skip("set LAYA_CHECKPOINTS to a directory holding the english and multilingual checkpoints")
	}
	loadMu.Lock()
	defer loadMu.Unlock()
	if tok, ok := loaded[name]; ok {
		return tok
	}
	tok, err := LoadFS(os.DirFS(root), name)
	if err != nil {
		tb.Fatalf("load the %s tokenizer: %v", name, err)
	}
	loaded[name] = tok
	return tok
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// firstDifference describes where two id sequences part, in a line short enough to read.
func firstDifference(got, want []int32) string {
	n := len(got)
	if len(want) < n {
		n = len(want)
	}
	for i := 0; i < n; i++ {
		if got[i] != want[i] {
			return fmt.Sprintf("first difference at %d: got %d, want %d (lengths %d and %d)", i, got[i], want[i], len(got), len(want))
		}
	}
	return fmt.Sprintf("one is a prefix of the other (lengths %d and %d)", len(got), len(want))
}

func describe(text string) string {
	if text == "" {
		return "<empty>"
	}
	out := ""
	n := 0
	for _, r := range text {
		if n == 18 {
			break
		}
		if r < 0x20 || r > 0x7E {
			out += fmt.Sprintf("U+%04X", r)
		} else {
			out += string(r)
		}
		n++
	}
	return out
}

func writeFile(tb testing.TB, path, content string) {
	tb.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		tb.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		tb.Fatal(err)
	}
}
