package agent_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// Example is compiled and not run: it needs a checkpoint and a native session.
func Example() {
	var session infer.Session // from the onnx package
	a, err := agent.Open(os.DirFS("checkpoints/english"), session)
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()

	qs, err := schema.NewQuestions(
		schema.Ask("intent", schema.MustQuestion(schema.ChoiceOf("What does the customer want?", "status", "refund"))),
		schema.Ask("angry", schema.MustQuestion(schema.Noul("Is the customer angry?"))),
	)
	if err != nil {
		log.Fatal(err)
	}
	preds, err := a.PredictBatch(context.Background(), []any{"I was charged twice", "Where is my order?"}, qs,
		agent.WithBatchSize(2), agent.WithSortByLength(true), agent.WithLanguage("en"))
	if err != nil {
		log.Fatal(err)
	}
	for _, p := range preds {
		intent, _ := p.Answers.Get("intent")
		fmt.Println(intent.(*schema.ChoiceAnswer).Choice, p.Usage.InputTokens, p.Usage.Truncated)
	}
}

// ExampleWithCalibration is compiled and not run: it writes a calibration payload and installs it.
func ExampleWithCalibration() {
	var session infer.Session // from the onnx package
	payload, err := agent.CalibrationPayload{Temperature: [3]float64{1.5, 2, 0.75}}.MarshalJSON()
	if err != nil {
		log.Fatal(err)
	}
	a, err := agent.Open(os.DirFS("checkpoints/english"), session, agent.WithCalibration(payload))
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()
}
