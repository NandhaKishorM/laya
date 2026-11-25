// Router routes three requests to a model tier with the router preset, as Python example 29 does.
//
//	go run ./router <checkpoint-dir> <graph-dir-or-laya.onnx> [onnxruntime-library]
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/onnx"
	"github.com/NandhaKishorM/laya/laya-go/presets"
	"github.com/NandhaKishorM/laya/laya-go/samples/internal/gateway"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "router:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: router <checkpoint-dir> <graph-dir-or-laya.onnx> [onnxruntime-library]")
	}
	graph := args[1]
	if fi, err := os.Stat(graph); err == nil && fi.IsDir() {
		graph = filepath.Join(graph, "laya.onnx")
	}
	var opts []onnx.Option
	if len(args) > 2 {
		opts = append(opts, onnx.WithLibrary(args[2]))
	}
	sess, err := onnx.Open(graph, opts...)
	if err != nil {
		return err
	}
	// The agent owns the session; closing the agent closes it.
	a, err := agent.Open(os.DirFS(args[0]), sess)
	if err != nil {
		_ = sess.Close()
		return err
	}
	defer a.Close()

	qs := presets.Router()
	ctx := context.Background()
	results := make([]gateway.Router, len(gateway.Requests))
	for i, req := range gateway.Requests {
		// The state is an ordered object: the model sees keys in the order given.
		pred, err := a.Predict(ctx, lang.Object{{Key: "request", Value: req.Text}}, qs)
		if err != nil {
			return fmt.Errorf("%s: %w", req.Label, err)
		}
		if results[i], err = gateway.Read(pred.Answers); err != nil {
			return fmt.Errorf("%s: %w", req.Label, err)
		}
	}
	gateway.Report(os.Stdout, results)
	return nil
}
