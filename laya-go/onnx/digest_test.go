package onnx

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func stubGraph(t *testing.T) (path, sum string) {
	t.Helper()
	path = filepath.Join(t.TempDir(), "laya.onnx")
	if err := os.WriteFile(path, []byte("not a graph"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := sha256.Sum256([]byte("not a graph"))
	return path, hex.EncodeToString(s[:])
}

func TestVerifyGraph(t *testing.T) {
	path, sum := stubGraph(t)
	for name, c := range map[string]struct {
		want string
		is   error
	}{
		"match":      {sum, nil},
		"upper":      {strings.ToUpper(sum), nil},
		"mismatch":   {strings.Repeat("0", 64), ErrDigestMismatch},
		"empty want": {"", ErrDigestMismatch},
	} {
		t.Run(name, func(t *testing.T) {
			var o options
			WithExpectedSHA256(c.want)(&o)
			err := verifyGraph(path, o)
			if c.is == nil && err != nil || c.is != nil && !errors.Is(err, c.is) {
				t.Fatalf("err = %v, want %v", err, c.is)
			}
			if c.is != nil && (!strings.Contains(err.Error(), path) || !strings.Contains(err.Error(), sum)) {
				t.Errorf("%v does not name the file and the actual digest", err)
			}
		})
	}
	if err := verifyGraph(path, options{}); err != nil {
		t.Fatalf("no expectation must verify nothing: %v", err)
	}
	var o options
	WithExpectedSHA256(sum)(&o)
	if err := verifyGraph(filepath.Join(t.TempDir(), "gone.onnx"), o); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("missing graph = %v", err)
	}
}
