package sequence_test

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// The examples that run use a tiny byte-level tokenizer built in memory, so their ids are bytes
// and a few merges. With a real checkpoint the code is the same:
//
//	tok, err := tokenizer.LoadFS(os.DirFS("checkpoints/english"), ".")
func exampleTokenizer() *tokenizer.Tokenizer {
	spec, config := fixtures.ToyTokenizer()
	tok, err := tokenizer.Parse(spec, config)
	if err != nil {
		log.Fatal(err)
	}
	return tok
}

func ExampleBuild() {
	tok := exampleTokenizer()
	q := schema.MustQuestion(schema.ChoiceOf("Which team?", "billing", "technical"))
	st, err := sequence.EncodeState(tok, map[string]any{"subject": "Charged twice", "body": "the in"})
	if err != nil {
		log.Fatal(err)
	}
	seq, err := sequence.Build(tok, q, st, sequence.Budget{MaxLen: 128, HeadMaxLen: 64})
	if err != nil {
		log.Fatal(err)
	}
	// [CLS] choice question: Which team? [SEP] [MASK] billing [MASK] technical [SEP] state [SEP]
	fmt.Println(len(seq.IDs), seq.Markers)
	fmt.Println(tok.Decode(seq.IDs, false))
	fmt.Printf("%+v\n", seq.Truncation)
	// Output:
	// 95 [30 38]
	// [CLS]choice question: Which team?[SEP][MASK] billing[MASK] technical[SEP]{"body": "the in", "subject": "Charged twice"}[SEP]
	// {StateTokens:44 StateTokensUsed:44 StateTokensDropped:0 Truncated:false}
}

// A state longer than the room is cut without an error, and the Truncation says so.
func ExampleBuild_truncation() {
	tok := exampleTokenizer()
	q := schema.MustQuestion(schema.Noul("The customer asks for a refund."))
	long := sequence.State{IDs: tok.Encode(strings.Repeat("the in ", 100)), Keep: sequence.KeepEnd}
	seq, err := sequence.Build(tok, q, long, sequence.Budget{MaxLen: 128, HeadMaxLen: 64})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(seq.IDs), seq.Truncation.Truncated, seq.Truncation.StateTokensDropped)
	// Output: 128 true 440
}

func ExampleStateRoom() {
	tok := exampleTokenizer()
	b := sequence.Budget{MaxLen: 512, HeadMaxLen: 192}
	few := schema.MustQuestion(schema.Noul("The customer asks for a refund."))
	labels := make([]string, 30)
	for i := range labels {
		labels[i] = fmt.Sprintf("option number %d with a longer description", i)
	}
	many := schema.MustQuestion(schema.ChoiceOf("Which one?", labels...))
	for _, q := range []schema.Question{few, many} {
		room, err := sequence.StateRoom(tok, q, b)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(q.NumOptions(), "options leave", room, "state tokens")
	}
	// Output:
	// 2 options leave 392 state tokens
	// 30 options leave 331 state tokens
}

func ExamplePlanWindows() {
	tok := exampleTokenizer()
	b := sequence.Budget{MaxLen: 512, HeadMaxLen: 192}
	q := schema.MustQuestion(schema.Noul("The contract has an auto-renewal clause."))
	plan, err := sequence.PlanWindows(tok, []schema.Question{q}, b, sequence.WindowRequest{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(plan.Window, plan.Stride, plan.Room, plan.Clamped())
	fmt.Println(sequence.BatchCap(40, plan.Window, b.DefaultWindow(), 0))

	// A window wider than the room is clamped, and the warning says why.
	plan, err = sequence.PlanWindows(tok, []schema.Question{q}, b, sequence.WindowRequest{Window: 10000})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(plan.Window, plan.Warnings[0])
	// Output:
	// 312 156 383 false
	// 0 false
	// 383 laya: predict_long: window=10000 is wider than the 383 state tokens these questions leave inside max_len=512, so every window would be truncated to 383 on the way to the model; scanning with window=383 instead
}

func ExampleBatchCap() {
	// 413 windows of 23 tokens, where the default 312-token window would have made 29.
	fmt.Println(sequence.BatchCap(413, 23, 312, 0))
	fmt.Println(sequence.BatchCap(413, 23, 312, 8)) // an explicit batch size always wins
	fmt.Println(sequence.BatchCap(20, 312, 312, 0)) // not clamped: Python's None, one pass
	// Output:
	// 30 true
	// 8 true
	// 0 false
}

// ExampleEncodeState_checkpoint is compiled and not run: it needs a checkpoint directory.
func ExampleEncodeState_checkpoint() {
	tok, err := tokenizer.LoadFS(os.DirFS("checkpoints/english"), ".")
	if err != nil {
		log.Fatal(err)
	}
	turns := []any{"Hi, I was charged twice.", "Let me check that for you.", "Thanks, any news?"}
	st, err := sequence.EncodeState(tok, turns) // a list keeps its end when it is cut
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(st.IDs), st.Keep)
}
