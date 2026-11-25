package onnx

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/digest"
)

// LibraryEnv names the environment variable that holds the path of the ONNX Runtime shared
// library when WithLibrary is not used.
const LibraryEnv = "LAYA_ONNXRUNTIME_LIBRARY"

// ErrNoCGO is returned by Open when the package was built with CGO_ENABLED=0.
var ErrNoCGO = errors.New("onnx: built without cgo")

// ErrRuntime wraps every failure to load or start ONNX Runtime, with the path tried and a hint.
var ErrRuntime = errors.New("onnx: cannot load ONNX Runtime")

// ErrRuntimeConflict is returned when Open is given a WithLibrary path other than the one
// already loaded. Only an explicit WithLibrary is checked: LibraryEnv and the default name are
// used only to load the first runtime. An ONNX Runtime the application initialized itself is
// adopted as is, and then no library option is checked or used.
var ErrRuntimeConflict = errors.New("onnx: ONNX Runtime already loaded from another path")

// Option adjusts Open.
type Option func(*options)

type options struct {
	library      string
	intra, inter int
	sha256       string
	checkSHA     bool
}

// WithLibrary sets the ONNX Runtime shared library to load. It overrides LibraryEnv. If a
// runtime is already loaded from another path, Open fails with ErrRuntimeConflict.
func WithLibrary(path string) Option { return func(o *options) { o.library = path } }

// WithIntraOpThreads sets the threads one operator may use. 0 keeps the runtime default.
func WithIntraOpThreads(n int) Option { return func(o *options) { o.intra = n } }

// WithInterOpThreads sets the operators that may run at once. 0 keeps the runtime default. A
// value above 0 also selects ONNX Runtime's parallel execution mode, which ignores the count
// otherwise.
func WithInterOpThreads(n int) Option { return func(o *options) { o.inter = n } }

// ErrDigestMismatch is wrapped by the error Open returns for a graph whose SHA-256 is not the
// expected one.
var ErrDigestMismatch = digest.ErrMismatch

// WithExpectedSHA256 makes Open hash the graph file, as a stream, and refuse it unless its hex
// SHA-256 is hexDigest, before ONNX Runtime loads it. The error wraps ErrDigestMismatch and names
// the file and both digests. Without this option Open reads the "onnx" or "onnx_path" key of the
// LAYA_SHA256_DIGESTS JSON object, as Python does. Only the graph is hashed, not its .data sidecar, as in Python.
func WithExpectedSHA256(hexDigest string) Option {
	return func(o *options) { o.sha256, o.checkSHA = hexDigest, true }
}

// verifyGraph is the check WithExpectedSHA256 asks for, or nil without it.
func verifyGraph(path string, o options) error {
	want := o.sha256
	if !o.checkSHA {
		env, err := digest.FromEnv()
		if err != nil {
			return fmt.Errorf("onnx: %w", err)
		}
		a, hasA := env["onnx"]
		b, hasB := env["onnx_path"]
		switch {
		case hasA && hasB && !strings.EqualFold(strings.TrimSpace(a), strings.TrimSpace(b)):
			return fmt.Errorf("onnx: %s gives two different digests for the graph (onnx and onnx_path)", digest.Env)
		case hasA:
			want = a
		case hasB:
			want = b
		default:
			return nil
		}
	}
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("onnx: cannot verify %s: %w", path, err)
	}
	defer f.Close()
	got, err := digest.Sum(f)
	if err != nil {
		return fmt.Errorf("onnx: cannot verify %s: %w", path, err)
	}
	return digest.Compare(path, got, want)
}

// resolve applies the options and rejects values that cannot be honored.
func resolve(opts []Option) (options, error) {
	var o options
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	if o.intra < 0 || o.inter < 0 {
		return o, fmt.Errorf("onnx: thread counts must not be negative (intra %d, inter %d)", o.intra, o.inter)
	}
	return o, nil
}

// libraryPath picks the shared library: the explicit path, else env, else the platform's name
// for the loader to search for.
func libraryPath(explicit, env string) string {
	if explicit != "" {
		return explicit
	}
	if env != "" {
		return env
	}
	return defaultLibrary(runtime.GOOS)
}

func defaultLibrary(goos string) string {
	switch goos {
	case "darwin":
		return "libonnxruntime.dylib"
	case "windows":
		return "onnxruntime.dll"
	}
	return "libonnxruntime.so"
}

// samePath reports whether a and b name one file.
func samePath(a, b string) bool {
	if a == b {
		return true
	}
	resolve := func(p string) string {
		p = filepath.Clean(p)
		if r, err := filepath.EvalSymlinks(p); err == nil {
			return r
		}
		return p
	}
	return resolve(a) == resolve(b)
}

// loadError wraps a failure to start the runtime from path as ErrRuntime, with a hint.
func loadError(path string, err error) error {
	hint := ""
	if strings.Contains(err.Error(), "API base") {
		hint = " (needs ONNX Runtime 1.29 or newer; set WithLibrary or " + LibraryEnv + " to its shared library)"
	}
	return fmt.Errorf("%w: %q: %v%s", ErrRuntime, path, err, hint)
}
