// Benchmark measures what the SDK costs the only way a Go measurement means anything: warm, best
// of N, with the cold first call reported on its own instead of averaged into the rest.
//
//	go run ./benchmark                                  # model-free half only
//	go run ./benchmark <checkpoint-dir> <graph> [lib]   # also the forward pass
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/onnx"
	"github.com/NandhaKishorM/laya/laya-go/samples/internal/demo"
)

const (
	warmup      = 20 // model-free calls are microseconds
	rounds      = 5
	modelWarmup = 1 // a forward pass is tens of milliseconds
	modelRounds = 2
	batchSize   = 8
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "benchmark:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	short := "I cannot log in and need a password reset."
	long := strings.Repeat(short, 90) // about 3.8 KB

	fmt.Println("model-free (no checkpoint, no graph)")
	cold, err := once(func() error { lang.Analyze(short); return nil })
	if err != nil {
		return err
	}
	fmt.Printf("  %-34s %8.3f ms   <- cold\n", "detect, first ever call", ms(cold))
	for _, c := range []struct{ label, text string }{{"detect, 42-char state", short}, {"detect, 3.8 KB state", long}} {
		d, err := best(warmup, rounds, func() error { lang.Analyze(c.text); return nil })
		if err != nil {
			return err
		}
		report(c.label, d)
	}

	if len(args) < 2 {
		fmt.Println("\npass <checkpoint-dir> <graph> to also measure the forward pass")
		return nil
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
	a, err := agent.Open(os.DirFS(args[0]), sess)
	if err != nil {
		_ = sess.Close()
		return err
	}
	defer a.Close()

	ctx := context.Background()
	state := demo.State("Charged twice for order 8812",
		"I was billed two times this morning. Please refund one charge.")
	states := make([]any, batchSize)
	for i := range states {
		states[i] = state
	}
	longState := demo.State("Charged twice", strings.Repeat("I was billed two times this morning. ", 60))

	fmt.Printf("\nforward pass (%d questions per state)\n", len(qs.IDs()))
	c, err := once(func() error { _, err := a.Predict(ctx, state, qs); return err })
	if err != nil {
		return err
	}
	fmt.Printf("  %-34s %8.3f ms   <- cold, includes the first run of the session\n", "Predict, first ever call", ms(c))
	one, err := best(modelWarmup, modelRounds, func() error { _, err := a.Predict(ctx, state, qs); return err })
	if err != nil {
		return err
	}
	report("Predict, 1 state", one)
	batch, err := best(modelWarmup, modelRounds, func() error { _, err := a.PredictBatch(ctx, states, qs); return err })
	if err != nil {
		return err
	}
	report(fmt.Sprintf("PredictBatch, %d states", batchSize), batch)
	report("  ... per state", batch/batchSize)
	long2, err := best(0, 1, func() error { _, err := a.PredictLong(ctx, longState, qs); return err })
	if err != nil {
		return err
	}
	report("PredictLong, multi-window state", long2)
	fmt.Println("  (PredictLong is one timed call: every window is a forward pass, and a sample must finish)")

	fmt.Printf("\nModel-free figures are the best of %d after %d warmup calls; model-backed ones the best of\n"+
		"%d after %d. The cold figures are single calls. Best, not mean: the mean is a mean over how\n"+
		"much other work the machine was doing. On CPU the per-state batch figure is expected to be close\n"+
		"to the single-state one: the matmuls dominate and do not amortize.\n",
		rounds, warmup, modelRounds, modelWarmup)
	return nil
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func report(label string, d time.Duration) { fmt.Printf("  %-34s %8.3f ms\n", label, ms(d)) }

// once times a single call.
func once(f func() error) (time.Duration, error) {
	start := time.Now()
	if err := f(); err != nil {
		return 0, err
	}
	return time.Since(start), nil
}

// best is the fastest of n timed calls after warm untimed ones.
func best(warm, n int, f func() error) (time.Duration, error) {
	for i := 0; i < warm; i++ {
		if err := f(); err != nil {
			return 0, err
		}
	}
	b := time.Duration(1<<63 - 1)
	for i := 0; i < n; i++ {
		start := time.Now()
		if err := f(); err != nil {
			return 0, err
		}
		if d := time.Since(start); d < b {
			b = d
		}
	}
	return b, nil
}
