package tokenizer_test

import (
	"fmt"
	"log"
	"os"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// The examples that run use a tiny byte-level tokenizer built in memory. A real checkpoint loads
// the same way: tokenizer.LoadFS(os.DirFS("checkpoints/english"), ".") reads
// tokenizer/tokenizer.json and tokenizer_config.json beside it.

func ExampleParse() {
	spec, config := fixtures.ToyTokenizer()
	tok, err := tokenizer.Parse(spec, config)
	if err != nil {
		log.Fatal(err)
	}
	ids := tok.Encode("the thin man")
	fmt.Println(ids)
	fmt.Println(tok.Decode(ids, true))
	cls, _ := tok.CLS()
	mask, _ := tok.Mask()
	fmt.Println(cls, mask, tok.MaskToken())
	// Output:
	// [260 101 32 260 261 32 109 97 110]
	// the thin man
	// 257 259 [MASK]
}

func ExampleTokenizer_EncodeLimit() {
	spec, config := fixtures.ToyTokenizer()
	tok, err := tokenizer.Parse(spec, config)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(tok.EncodeLimit("the thin man", 4)) // the first four ids, without tokenizing the rest
	// Output: [260 101 32 260]
}

// ExampleLoadFS is compiled and not run: it needs a checkpoint directory.
func ExampleLoadFS() {
	tok, err := tokenizer.LoadFS(os.DirFS("checkpoints/english"), ".")
	if err != nil {
		log.Fatal(err)
	}
	ids := tok.Encode("I cannot log in.")
	fmt.Println(ids, tok.Decode(ids, true))
}
