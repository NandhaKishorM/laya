//go:build !cgo

package onnx

import (
	"context"

	"github.com/NandhaKishorM/laya/laya-go/infer"
)

// Session is the ONNX Runtime session. Without cgo it cannot be opened.
type Session struct{}

var _ infer.Session = (*Session)(nil)

// Open returns ErrNoCGO: this build has no ONNX Runtime binding.
func Open(graphPath string, opts ...Option) (*Session, error) { return nil, ErrNoCGO }

// Run returns ErrNoCGO.
func (s *Session) Run(ctx context.Context, b infer.Batch) (infer.Output, error) {
	return infer.Output{}, ErrNoCGO
}

// Close does nothing.
func (s *Session) Close() error { return nil }

// RuntimeVersion returns "".
func (s *Session) RuntimeVersion() string { return "" }
