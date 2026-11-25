package digest

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
)

func TestFileRefusesTrailingSlash(t *testing.T) {
	fsys := fstest.MapFS{"a/b.bin": {Data: []byte("graph")}}
	for _, rel := range []string{"a/b.bin/", "a/b.bin/."} {
		if err := File(fsys, rel, hexOf("graph")); !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("%s: err = %v", rel, err)
		}
	}
}

// swapFS reports a regular file at Stat, then opens a device: the path swapped between the two.
type swapFS struct{ fstest.MapFS }

type deviceFile struct{ fs.File }

func (d deviceFile) Stat() (fs.FileInfo, error) {
	st, err := d.File.Stat()
	return deviceInfo{st}, err
}

type deviceInfo struct{ fs.FileInfo }

func (deviceInfo) Mode() fs.FileMode { return fs.ModeDevice | fs.ModeCharDevice }

func (s swapFS) Open(name string) (fs.File, error) {
	f, err := s.MapFS.Open(name)
	if err != nil {
		return nil, err
	}
	return deviceFile{f}, nil
}

func (s swapFS) Stat(name string) (fs.FileInfo, error) { return s.MapFS.Stat(name) }

func TestFileRefusesADeviceSwappedInAfterStat(t *testing.T) {
	fsys := swapFS{fstest.MapFS{"g.onnx": {Data: []byte("graph")}}}
	if err := File(fsys, "g.onnx", hexOf("graph")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v", err)
	}
}

func TestFileRefusesNonRegularFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.Symlink("/dev/zero", filepath.Join(dir, "zero.onnx")); err != nil {
		t.Fatal(err)
	}
	err := File(os.DirFS(dir), "zero.onnx", hexOf("x"))
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v", err)
	}
}

func TestFromEnv(t *testing.T) {
	for _, c := range []struct {
		name, val string
		want      map[string]string
		bad       bool
	}{
		{"unset", "", nil, false},
		{"blank", "  \n", nil, false},
		{"object", ` {"a":"b"} `, map[string]string{"a": "b"}, false},
		{"not json", "nope", nil, true},
		{"array", `["a"]`, nil, true},
		{"non-string", `{"a":1}`, nil, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv(Env, c.val)
			got, err := FromEnv()
			if c.bad != (err != nil) {
				t.Fatalf("err = %v", err)
			}
			if len(got) != len(c.want) || got["a"] != c.want["a"] {
				t.Fatalf("got %v", got)
			}
		})
	}
}
