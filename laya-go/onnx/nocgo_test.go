//go:build !cgo

package onnx

import (
	"context"
	"errors"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/infer"
)

func TestOpenWithoutCGO(t *testing.T) {
	s, err := Open("laya.onnx", WithLibrary("/x"))
	if !errors.Is(err, ErrNoCGO) || s != nil {
		t.Fatalf("Open = %v, %v; want ErrNoCGO", s, err)
	}
	var stub Session
	if _, err := stub.Run(context.Background(), infer.Batch{}); !errors.Is(err, ErrNoCGO) {
		t.Errorf("Run = %v", err)
	}
	if stub.Close() != nil || stub.RuntimeVersion() != "" {
		t.Error("the stub's Close and RuntimeVersion are not inert")
	}
}
