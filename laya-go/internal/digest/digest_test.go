package digest

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io/fs"
	"strings"
	"testing"
	"testing/fstest"
	"testing/iotest"
)

func hexOf(b string) string { s := sha256.Sum256([]byte(b)); return hex.EncodeToString(s[:]) }

func TestFile(t *testing.T) {
	fsys := fstest.MapFS{"a/b.bin": {Data: []byte("graph")}, "d/x": {Data: []byte("x")}}
	good := hexOf("graph")
	for _, c := range []struct {
		name, rel, want string
		is              error
	}{
		{"match", "a/b.bin", good, nil},
		{"upper-case digest", "a/b.bin", strings.ToUpper(good), nil},
		{"padded digest", "a/b.bin", " " + good + "\n", nil},
		{"mismatch", "a/b.bin", hexOf("other"), ErrMismatch},
		{"missing", "nope", good, fs.ErrNotExist},
		{"directory", "d", good, fs.ErrNotExist},
		{"parent", "../x", good, ErrUnsafePath},
		{"inner parent", "a/../b", good, ErrUnsafePath},
		{"trailing parent", "a/..", good, ErrUnsafePath},
		{"bare parent", "..", good, ErrUnsafePath},
		{"absolute", "/etc/passwd", good, ErrUnsafePath},
		{"drive", "C:\\x", good, ErrUnsafePath},
		{"backslash parent", "..\\x", good, ErrUnsafePath},
		{"dot", ".", good, ErrUnsafePath},
		{"empty", "", good, ErrUnsafePath},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := File(fsys, c.rel, c.want)
			if c.is == nil {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if !errors.Is(err, c.is) {
				t.Fatalf("err = %v, want %v", err, c.is)
			}
		})
	}
}

func TestMismatchNamesTheFileAndBothDigests(t *testing.T) {
	err := File(fstest.MapFS{"w.onnx": {Data: []byte("a")}}, "w.onnx", "ff")
	for _, s := range []string{"w.onnx", "ff", hexOf("a")} {
		if err == nil || !strings.Contains(err.Error(), s) {
			t.Fatalf("%v does not name %q", err, s)
		}
	}
}

func TestSumStreamsAndReportsReadErrors(t *testing.T) {
	big := strings.Repeat("0123456789abcdef", 1<<16)
	got, err := Sum(iotest.OneByteReader(strings.NewReader(big[:5000])))
	if err != nil || got != hexOf(big[:5000]) {
		t.Fatalf("got %s, %v", got, err)
	}
	if _, err := Sum(iotest.ErrReader(errors.New("boom"))); err == nil {
		t.Fatal("a read error was swallowed")
	}
}
