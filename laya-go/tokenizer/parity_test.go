package tokenizer

import (
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
)

var shipped = [...]string{"english", "multilingual"}

// Every corpus entry is byte-identical to the ids Python's tokenizer gives, on both checkpoints.
func TestCorpusIDsMatchPython(t *testing.T) {
	fx := fixtures.Load(t, "tokenizer_ids.json")
	corpusText := fixtures.Obj(t, fx.Must("corpus_text"))
	total := 0
	for _, name := range shipped {
		side := fixtures.Obj(t, fx.Must(name))
		if _, skipped := side.Get("skipped"); skipped {
			t.Fatalf("the golden for %s is a skip marker; regenerate it with the checkpoints present", name)
		}
		tok := checkpoint(t, name)
		t.Run(name+"/identity", func(t *testing.T) {
			id := fixtures.Field(t, side, "identity")
			if got := fixtures.Str(t, fixtures.Field(t, id, "model_type")); got != "BPE" {
				t.Fatalf("model_type = %q", got)
			}
			want := fixtures.Int(t, fixtures.Field(t, id, "vocab_size"))
			if extra, ok := fixtures.Obj(t, id).Get("added_tokens_outside_vocab"); ok && extra != nil {
				want += fixtures.Int(t, extra)
			}
			if tok.VocabSize() != want {
				t.Errorf("VocabSize = %d, want the vocabulary plus added tokens (%d)", tok.VocabSize(), want)
			}
		})
		t.Run(name+"/specials", func(t *testing.T) {
			sp := fixtures.Field(t, side, "special_ids")
			for role, get := range map[string]func() (int32, bool){
				"cls": tok.CLS, "sep": tok.SEP, "mask": tok.Mask, "pad": tok.Pad,
			} {
				got, ok := get()
				if want := int32(fixtures.Int(t, fixtures.Field(t, sp, role))); !ok || got != want {
					t.Errorf("%s id = %d (found %v), want %d", role, got, ok, want)
				}
			}
			if got, want := tok.MaskToken(), fixtures.Str(t, fixtures.Field(t, sp, "mask_token")); got != want {
				t.Errorf("MaskToken = %q, want %q", got, want)
			}
		})
		for _, entry := range fixtures.Obj(t, fixtures.Field(t, side, "corpus")) {
			total++
			text := fixtures.Str(t, corpusText.Must(entry.Key))
			want := fixtures.Ints(t, entry.Value)
			t.Run(name+"/"+entry.Key, func(t *testing.T) {
				if got := tok.Encode(text); !equalIDs(got, want) {
					t.Errorf("ids for %q: %s", entry.Key, firstDifference(got, want))
				}
			})
		}
		t.Run(name+"/option-truncation-48", func(t *testing.T) {
			trunc := fixtures.Field(t, side, "option_truncation")
			text := " " + fixtures.Str(t, fixtures.Field(t, trunc, "text"))
			uncapped := fixtures.Ints(t, fixtures.Field(t, trunc, "uncapped"))
			capped := fixtures.Ints(t, fixtures.Field(t, trunc, "capped_48"))
			if got := tok.EncodeLimit(text, -1); !equalIDs(got, uncapped) {
				t.Errorf("uncapped: %s", firstDifference(got, uncapped))
			}
			if got := tok.EncodeLimit(text, 48); !equalIDs(got, capped) {
				t.Errorf("capped at 48: %s", firstDifference(got, capped))
			}
		})
	}
	if total < 70 {
		t.Errorf("only %d corpus entries ran, expected the whole corpus on both checkpoints", total)
	}
}

func TestEncodeLimitEdges(t *testing.T) {
	tok := checkpoint(t, "english")
	if got := tok.EncodeLimit("hello world", 0); len(got) != 0 {
		t.Errorf("limit 0 gave %v", got)
	}
	full := tok.Encode("hello world, this is a longer sentence")
	for n := 1; n <= len(full); n++ {
		if got := tok.EncodeLimit("hello world, this is a longer sentence", n); !equalIDs(got, full[:n]) {
			t.Errorf("limit %d is not a prefix of the full encoding", n)
		}
	}
	if got := tok.EncodeLimit("hello", 100); !equalIDs(got, tok.Encode("hello")) {
		t.Error("a limit above the length changes nothing")
	}
}
