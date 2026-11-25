package onnx_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/onnx"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// Open a fused graph, hand it to the agent, and ask one question. The agent closes the session.
func ExampleOpen() {
	sess, err := onnx.Open("laya.onnx", onnx.WithLibrary("/opt/onnxruntime/lib/libonnxruntime.so.1.30.0"))
	if err != nil {
		log.Fatal(err)
	}
	a, err := agent.Open(os.DirFS("checkpoint"), sess)
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()

	qs, err := schema.ParseQuestions([]byte(`{"urgent": {"type": "noul", "instructions": "Is this urgent?"}}`))
	if err != nil {
		log.Fatal(err)
	}
	pred, err := a.Predict(context.Background(), "The site is down.", qs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(pred.Answers)
}
