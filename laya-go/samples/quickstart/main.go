// Quickstart asks four typed questions about one support email, in English and Hindi, and prints
// each answer by its type.
//
//	go run ./quickstart <checkpoint-dir> <graph-dir-or-laya.onnx> [onnxruntime-library]
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/onnx"
	"github.com/NandhaKishorM/laya/laya-go/samples/internal/demo"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "quickstart:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: quickstart <checkpoint-dir> <graph-dir-or-laya.onnx> [onnxruntime-library]")
	}
	graph := args[1]
	if fi, err := os.Stat(graph); err == nil && fi.IsDir() {
		graph = filepath.Join(graph, "laya.onnx")
	}
	var opts []onnx.Option
	if len(args) > 2 {
		opts = append(opts, onnx.WithLibrary(args[2]))
	}
	qs, err := demo.Questions()
	if err != nil {
		return err
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

	ctx := context.Background()
	states := []struct {
		label string
		state lang.Object
	}{
		{"English", demo.State("Charged twice for order 8812",
			"I was billed two times for the same order this morning. Please refund one of the charges. I have attached both receipts.")},
		{"Hindi", demo.State("ऑर्डर 8812 का दो बार बिल",
			"मुझे उसी ऑर्डर के लिए दो बार बिल किया गया। कृपया रिफंड करें।")},
	}
	for _, s := range states {
		pred, err := a.Predict(ctx, s.state, qs)
		if err != nil {
			return fmt.Errorf("%s: %w", s.label, err)
		}
		report(s.label, pred)
	}
	return nil
}

// report prints every answer by its concrete type: a choice is not a score.
func report(label string, p schema.Prediction) {
	fmt.Printf("\n== %s ==\n", label)
	for _, ans := range p.Answers {
		switch v := ans.Answer.(type) {
		case *schema.ChoiceAnswer:
			fmt.Printf("  %-12s %-22s confidence %.3f\n", ans.ID, v.Choice, v.Confidence)
		case *schema.ScoreAnswer:
			fmt.Printf("  %-12s %-22.3f confidence %.3f\n", ans.ID, v.Score, v.Confidence)
		case *schema.NoulAnswer:
			fmt.Printf("  %-12s %-22.3f confidence %.3f\n", ans.ID, v.Noul, v.Confidence)
		}
	}
	// Usage says whether the model saw the whole state.
	fmt.Printf("  usage        %d input tokens, %d state tokens dropped, truncated=%t\n",
		p.Usage.InputTokens, p.Usage.StateTokensDropped, p.Usage.Truncated)
}
