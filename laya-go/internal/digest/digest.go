// Package digest checks SHA-256 digests the way laya's verify_digests does: streamed, so a
// multi-gigabyte graph is never read whole, and compared case-insensitively.
package digest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"strings"
)

// ErrMismatch is wrapped by the error for a file whose digest is not the expected one.
var ErrMismatch = errors.New("SHA-256 mismatch")

// ErrUnsafePath is wrapped by the error for a name that is absolute or escapes its directory.
var ErrUnsafePath = errors.New("unsafe path in expected digests")

// Sum streams r through SHA-256 and returns the lower-case hex digest.
func Sum(r io.Reader) (string, error) {
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Compare is nil when got equals want, ignoring case and surrounding space.
func Compare(name, got, want string) error {
	if strings.EqualFold(got, strings.TrimSpace(want)) {
		return nil
	}
	return fmt.Errorf("laya: %w for %s: expected %s, got %s. The artifact does not match the reviewed "+
		"digest; refusing to load it", ErrMismatch, name, want, got)
}

// Safe returns rel as a clean slash path under its directory, or an error wrapping ErrUnsafePath
// when it is empty, absolute (POSIX or a Windows drive or UNC root) or climbs out with "..".
func Safe(rel string) (string, error) {
	n := strings.ReplaceAll(rel, "\\", "/")
	drive := len(n) >= 2 && n[1] == ':' && (n[0]|0x20 >= 'a' && n[0]|0x20 <= 'z')
	if strings.HasPrefix(n, "/") || drive {
		return "", fmt.Errorf("laya: %w: absolute path %q", ErrUnsafePath, rel)
	}
	if n == "" || n == ".." || strings.HasPrefix(n, "../") || strings.Contains(n, "/../") || strings.HasSuffix(n, "/..") {
		return "", fmt.Errorf("laya: %w: %q", ErrUnsafePath, rel)
	}
	if n = path.Clean(n); n == "." {
		return "", fmt.Errorf("laya: %w: %q", ErrUnsafePath, rel)
	}
	return n, nil
}

// File hashes the regular file rel in fsys and compares it with want. A file that is absent, a
// directory, a device or a FIFO, or a name ending in "/", is an error wrapping fs.ErrNotExist.
// Python raises FileNotFoundError for "a/.." and "."; here they are ErrUnsafePath (both refuse).
func File(fsys fs.FS, rel, want string) error {
	n, err := Safe(rel)
	if err != nil {
		return err
	}
	if r := strings.ReplaceAll(rel, "\\", "/"); strings.HasSuffix(r, "/") || strings.HasSuffix(r, "/.") { // "x/", "x/." name a directory, never the file x
		return fmt.Errorf("laya: cannot verify %q: not a file: %w", rel, fs.ErrNotExist)
	}
	// Stat before Open: opening a FIFO blocks, and a device never ends. Python's isfile refuses both.
	if st, err := fs.Stat(fsys, n); err != nil {
		return fmt.Errorf("laya: cannot verify %q: %w", rel, err)
	} else if !st.Mode().IsRegular() {
		return fmt.Errorf("laya: cannot verify %q: not a regular file: %w", rel, fs.ErrNotExist)
	}
	f, err := fsys.Open(n)
	if err != nil {
		return fmt.Errorf("laya: cannot verify %q: %w", rel, err)
	}
	defer f.Close()
	// Again after Open: the path can be swapped for a device between Stat and Open.
	if st, err := f.Stat(); err != nil {
		return fmt.Errorf("laya: cannot verify %q: %w", rel, err)
	} else if !st.Mode().IsRegular() {
		return fmt.Errorf("laya: cannot verify %q: not a regular file: %w", rel, fs.ErrNotExist)
	}
	got, err := Sum(f)
	if err != nil {
		return fmt.Errorf("laya: cannot verify %q: %w", rel, err)
	}
	return Compare(rel, got, want)
}

// Env is the environment variable that carries expected digests when none are given in code.
const Env = "LAYA_SHA256_DIGESTS"

// FromEnv reads Env as a JSON object of artifact -> hex digest, as laya's verify_digests does when
// no digests are given. A blank or unset variable is no expectation (nil). Anything that is not
// an object of strings is an error.
func FromEnv() (map[string]string, error) {
	raw := strings.TrimSpace(os.Getenv(Env))
	if raw == "" {
		return nil, nil
	}
	var v any
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		return nil, fmt.Errorf("%s must be a JSON object of artifact->sha256: %w", Env, err)
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be a JSON object of artifact->sha256, got %T", Env, v)
	}
	out := make(map[string]string, len(obj))
	for k, x := range obj {
		s, ok := x.(string)
		if !ok {
			return nil, fmt.Errorf("%s: the digest of %q must be a string, got %T", Env, k, x)
		}
		out[k] = s
	}
	return out, nil
}
