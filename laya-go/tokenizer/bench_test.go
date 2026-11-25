package tokenizer

import (
	"strings"
	"testing"
)

const benchSentence = "The customer reports that the dashboard will not load after the update, " +
	"and we were charged twice in March; please refund the duplicate payment today. "

func benchEncode(b *testing.B, name, text string) {
	tok := checkpoint(b, name)
	b.SetBytes(int64(len(text)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if len(tok.Encode(text)) == 0 {
			b.Fatal("no ids")
		}
	}
}

// BenchmarkEncode measures the whole path: added-token split, normalization, pre-tokenization
// and BPE. Run with LAYA_CHECKPOINTS set: go test -bench Encode -run '^$' ./tokenizer
func BenchmarkEncode(b *testing.B) {
	b.Run("english/sentence", func(b *testing.B) { benchEncode(b, "english", benchSentence) })
	b.Run("english/4KB", func(b *testing.B) { benchEncode(b, "english", strings.Repeat(benchSentence, 26)) })
	b.Run("multilingual/sentence", func(b *testing.B) { benchEncode(b, "multilingual", benchSentence) })
	b.Run("multilingual/cjk", func(b *testing.B) {
		benchEncode(b, "multilingual", strings.Repeat("我们被收取了两次费用，请退还重复的款项。", 8))
	})
}

func BenchmarkNormalizeNFC(b *testing.B) {
	text := strings.Repeat("café café สวัสดี ", 20)
	b.SetBytes(int64(len(text)))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = normalize(text, formNFC)
	}
}
