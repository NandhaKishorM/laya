//go:build cgo

package onnx

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// The binding's environment is process-global and its init and destroy are unsynchronized, so
// this file is the only place that calls them, behind one mutex.
var rt struct {
	mu      sync.Mutex
	refs    int    // sessions holding the environment
	owned   bool   // we initialized it, so the last release destroys it
	path    string // the library we loaded, when owned
	version string
}

// acquire takes a reference on the runtime, starting it on first use, and returns its version.
func acquire(explicit string) (string, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.refs > 0 {
		if rt.owned && explicit != "" && !samePath(explicit, rt.path) {
			return "", fmt.Errorf("%w: %q is loaded, %q was requested", ErrRuntimeConflict, rt.path, explicit)
		}
		rt.refs++
		return rt.version, nil
	}
	if ort.IsInitialized() {
		rt.owned, rt.path, rt.version = false, "", ort.GetVersion()
		rt.refs = 1
		return rt.version, nil
	}
	path := libraryPath(explicit, os.Getenv(LibraryEnv))
	ort.SetSharedLibraryPath(path)
	if err := ort.InitializeEnvironment(); err != nil {
		return "", loadError(path, err)
	}
	rt.owned, rt.path, rt.version, rt.refs = true, path, ort.GetVersion(), 1
	return rt.version, nil
}

// release drops a reference and unloads the runtime when it was the last and we loaded it.
func release() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.refs--
	if rt.refs > 0 {
		return nil
	}
	owned := rt.owned
	rt.owned, rt.path, rt.version = false, "", ""
	if owned {
		return ort.DestroyEnvironment()
	}
	return nil
}

// references reports the sessions holding the runtime, for tests.
func references() int {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.refs
}
