// Package onnx runs a laya graph on ONNX Runtime and implements infer.Session, so the pure-Go
// agent package can answer questions with it.
//
// It is a separate module because it needs cgo and a native ONNX Runtime shared library
// (1.29 or newer, which the caller supplies). It requires the core laya-go module and
// github.com/yalue/onnxruntime_go; until the core module is tagged it builds only inside the
// laya repository, where go.mod points at it with a replace directive. Open takes the fused laya.onnx, whose .data
// sidecar must sit beside it, and runs it on the CPU provider:
//
//	sess, err := onnx.Open("laya.onnx", onnx.WithLibrary("/opt/onnxruntime/libonnxruntime.so.1.30.0"))
//	a, err := agent.Open(os.DirFS("checkpoint"), sess)
//
// The library is found through WithLibrary, else $LAYA_ONNXRUNTIME_LIBRARY, else the platform's
// default name on the loader's search path. A process holds one ONNX Runtime: sessions share it,
// and it is unloaded when the last one closes. An ONNX Runtime the application initialized itself
// is adopted instead: the library options are then ignored and it is never unloaded. Only an
// explicit WithLibrary is checked against a runtime already loaded. Built with CGO_ENABLED=0 the
// package still compiles, and Open returns ErrNoCGO.
//
// Notes on the binding: it wraps Go slices as tensors without copying or pinning them, and
// on an error path of a run it can leak an output tensor. Neither is mitigated here.
package onnx
