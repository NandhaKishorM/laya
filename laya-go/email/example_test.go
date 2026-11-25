package email_test

import (
	"fmt"

	"github.com/NandhaKishorM/laya/laya-go/email"
)

func ExampleCleanBody() {
	body := "Please refund the duplicate charge.\n\n" +
		"Atenciosamente,\nAna\n\n" +
		"Em seg., 5 de jan. de 2026, Bruno <bruno@x.com> escreveu:\n> the quoted request"
	fmt.Printf("%q\n", email.CleanBody(body, email.DefaultMaxChars))
	// Output: "Please refund the duplicate charge."
}

func ExampleState() {
	state, err := email.State("  Refund request ", "I was charged twice.\n\nSent from my iPhone",
		email.WithSender("ana@x.com"), email.WithField("ticket", 42), email.WithField("thread_id", nil))
	if err != nil {
		panic(err)
	}
	for _, m := range state {
		fmt.Printf("%s=%v\n", m.Key, m.Value)
	}
	// Output:
	// subject=Refund request
	// body=I was charged twice.
	// from=ana@x.com
	// ticket=42
}
