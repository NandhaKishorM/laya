package tokenizer

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
)

// go_extra.json holds what the crate answers over far more input than the corpus: every code
// point, and random strings from a fixed seed. See scripts/gen_go_fixtures.py.

// pyBool spells a bool the way Python formats it, which is how the goldens key their cases.
func pyBool(b bool) string {
	if b {
		return "True"
	}
	return "False"
}

func extra(t *testing.T) fixtures.Object { t.Helper(); return fixtures.Load(t, "go_extra.json") }

func codePointDigest(rows func(yield func(string))) string {
	h := sha256.New()
	rows(func(s string) { h.Write([]byte(s)) })
	return hex.EncodeToString(h.Sum(nil))
}

func mapped(cp rune, text string) string {
	var parts []string
	for _, r := range text {
		parts = append(parts, strconv.Itoa(int(r)))
	}
	return strconv.Itoa(int(cp)) + ":" + strings.Join(parts, ",") + ";"
}

func TestNormalizationMatchesTheCrateOverAllUnicode(t *testing.T) {
	sweep := fixtures.Obj(t, fixtures.Field(t, extra(t).Must("normalize"), "sweep"))
	for name, form := range map[string]normForm{"NFC": formNFC, "NFD": formNFD, "NFKC": formNFKC, "NFKD": formNFKD} {
		t.Run(name, func(t *testing.T) {
			got := codePointDigest(func(yield func(string)) {
				for r := rune(0); r < 0x110000; r++ {
					if r >= 0xD800 && r <= 0xDFFF {
						continue
					}
					yield(mapped(r, normalize(string(r), form)))
				}
			})
			if want := fixtures.Str(t, sweep.Must(name)); got != want {
				t.Errorf("%s of some single code point differs from the crate", name)
			}
		})
	}
}

func TestNormalizationMatchesTheCrateOnRandomStrings(t *testing.T) {
	cases := fixtures.List(t, fixtures.Field(t, extra(t).Must("normalize"), "cases"))
	for i, raw := range cases {
		text := fixtures.Str(t, fixtures.Field(t, raw, "text"))
		for name, form := range map[string]normForm{"nfc": formNFC, "nfd": formNFD, "nfkc": formNFKC, "nfkd": formNFKD} {
			if got, want := normalize(text, form), fixtures.Str(t, fixtures.Field(t, raw, name)); got != want {
				t.Errorf("case %d: %s(%+q) = %+q, want %+q", i, name, text, got, want)
				return
			}
		}
	}
	if len(cases) < 1000 {
		t.Errorf("only %d cases", len(cases))
	}
}

func TestLowercaseMatchesTheCrateOverAllUnicode(t *testing.T) {
	want := fixtures.Str(t, fixtures.Field(t, extra(t).Must("lowercase"), "sweep"))
	got := codePointDigest(func(yield func(string)) {
		for r := rune(0); r < 0x110000; r++ {
			if r >= 0xD800 && r <= 0xDFFF {
				continue
			}
			yield(mapped(r, lowercase(string(r))))
		}
	})
	if got != want {
		t.Error("lowercase of some code point differs from the crate")
	}
}

// A single_word added token is refused beside a word character. Which characters those are is
// the crate's rule, not Java's alphanumeric test and not Go's: it counts combining marks and
// joiners and not the other-number category. Swept over every code point on both sides.
func TestSingleWordFlankMatchesTheCrate(t *testing.T) {
	sw := fixtures.Obj(t, extra(t).Must("single_word"))
	spec := `{"model":{"type":"BPE","vocab":{"a":0,"<unk>":2},"merges":[],"unk_token":"<unk>"},
	  "added_tokens":[{"content":"TOK","id":1,"single_word":true,"special":true}]}`
	tok, err := Parse([]byte(spec), nil)
	if err != nil {
		t.Fatal(err)
	}
	standalone := func(s string) string {
		for _, id := range tok.Encode(s) {
			if id == 1 {
				return "1"
			}
		}
		return "0"
	}
	for side, check := range map[string]func(string) string{
		"left_standalone":  func(c string) string { return standalone(c + "TOK") },
		"right_standalone": func(c string) string { return standalone("TOK" + c) },
	} {
		got := codePointDigest(func(yield func(string)) {
			for r := rune(0); r < 0x110000; r++ {
				if r >= 0xD800 && r <= 0xDFFF {
					yield("0")
					continue
				}
				yield(check(string(r)))
			}
		})
		if want := fixtures.Str(t, sw.Must(side)); got != want {
			t.Errorf("%s: the characters that stop a single_word token differ from the crate's", side)
		}
	}
}

func TestPreTokenizersMatchTheCrate(t *testing.T) {
	pre := fixtures.Obj(t, extra(t).Must("pretokenize"))
	texts := fixtures.Strings(t, pre.Must("texts"))
	check := func(name string, p preTokenizer, want []any) {
		t.Run(name, func(t *testing.T) {
			if len(want) != len(texts) {
				t.Fatalf("%d outputs for %d texts", len(want), len(texts))
			}
			for i, text := range texts {
				got := p.preTokenize(text)
				if w := fixtures.Strings(t, want[i]); strings.Join(got, "\x00") != strings.Join(w, "\x00") || len(got) != len(w) {
					t.Errorf("preTokenize(%+q) = %q, want %q", text, got, w)
					return
				}
			}
		})
	}
	for _, prefix := range []bool{false, true} {
		for _, useRegex := range []bool{true, false} {
			key := "add_prefix_space=" + pyBool(prefix) + ",use_regex=" + pyBool(useRegex)
			check("ByteLevel/"+key, byteLevel{addPrefixSpace: prefix, useRegex: useRegex},
				fixtures.List(t, fixtures.Field(t, pre.Must("byte_level"), key)))
		}
	}
	for _, scheme := range []string{"always", "first", "never"} {
		for _, split := range []bool{true, false} {
			key := scheme + ",split=" + pyBool(split)
			p, err := parseMetaspace(node{"replacement": "▁", "prepend_scheme": scheme, "split": split})
			if err != nil {
				t.Fatal(err)
			}
			check("Metaspace/"+key, p, fixtures.List(t, fixtures.Field(t, pre.Must("metaspace"), key)))
		}
	}
}

func TestNormalizerSpecsMatchTheCrate(t *testing.T) {
	fam := fixtures.Obj(t, extra(t).Must("normalizers"))
	inputs := fixtures.Strings(t, fam.Must("inputs"))
	for i, entry := range fixtures.List(t, fam.Must("specs")) {
		raw := fixtures.Marshal(t, fixtures.Field(t, entry, "spec"))
		t.Run(strconv.Itoa(i)+"/"+string(raw), func(t *testing.T) {
			n, err := parseNode(raw, "normalizer")
			if err != nil {
				t.Fatal(err)
			}
			norm, err := parseNormalizer(n)
			if err != nil {
				t.Fatal(err)
			}
			want := fixtures.Strings(t, fixtures.Field(t, entry, "outputs"))
			for j, in := range inputs {
				if got := norm(in); got != want[j] {
					t.Errorf("%+q -> %+q, want %+q", in, got, want[j])
				}
			}
		})
	}
}

func TestEncodeFuzzMatchesTheCrate(t *testing.T) {
	fam := fixtures.Obj(t, extra(t).Must("encode_fuzz"))
	for _, name := range shipped {
		tok := checkpoint(t, name)
		side := fixtures.Field(t, fam, name)
		texts := fixtures.Strings(t, fixtures.Field(t, side, "texts"))
		ids := fixtures.List(t, fixtures.Field(t, side, "ids"))
		t.Run(name, func(t *testing.T) {
			bad := 0
			for i, text := range texts {
				want := fixtures.Ints(t, ids[i])
				if got := tok.Encode(text); !equalIDs(got, want) {
					t.Errorf("Encode(%+q): %s", text, firstDifference(got, want))
					if bad++; bad == 5 {
						return
					}
				}
			}
			if len(texts) < 500 {
				t.Errorf("only %d texts", len(texts))
			}
		})
	}
}

func TestDecodeFuzzMatchesTheCrate(t *testing.T) {
	fam := fixtures.Obj(t, extra(t).Must("decode_fuzz"))
	for _, name := range shipped {
		tok := checkpoint(t, name)
		cases := fixtures.List(t, fixtures.Field(t, fixtures.Field(t, fam, name), "cases"))
		t.Run(name, func(t *testing.T) {
			bad := 0
			for _, c := range cases {
				ids := fixtures.Ints(t, fixtures.Field(t, c, "ids"))
				for mode, skip := range map[string]bool{"skipping": true, "keeping": false} {
					if got, want := tok.Decode(ids, skip), fixtures.Str(t, fixtures.Field(t, c, mode)); got != want {
						t.Errorf("Decode(%v, skip=%v) = %+q, want %+q", ids, skip, got, want)
						if bad++; bad == 5 {
							return
						}
					}
				}
			}
		})
	}
}

// Short pieces merge by rescanning and long ones with a heap; they must agree.
func TestHeapAndRescanMergeAgree(t *testing.T) {
	for _, name := range shipped {
		tok := checkpoint(t, name)
		texts := fixtures.Strings(t, fixtures.Field(t, fixtures.Field(t, extra(t).Must("encode_fuzz"), name), "texts"))
		checked := 0
		for _, text := range texts {
			for _, piece := range tok.preTokenizer.preTokenize(tok.normalizer(text)) {
				var a, b []symbol
				for i := 0; i < len(piece); {
					_, w := utf8.DecodeRuneInString(piece[i:])
					id, ok := tok.bpe.vocab[piece[i:i+w]]
					if !ok {
						id = -1
					}
					a = append(a, symbol{int32(i), int32(i + w), id})
					i += w
				}
				b = append(b, a...)
				ra, rb := tok.bpe.mergeByRescan(a), tok.bpe.mergeWithHeap(b)
				if len(ra) != len(rb) {
					t.Fatalf("%s: %+q merges to %d symbols by rescan and %d by heap", name, piece, len(ra), len(rb))
				}
				for i := range ra {
					if ra[i] != rb[i] {
						t.Fatalf("%s: %+q differs at symbol %d", name, piece, i)
					}
				}
				checked++
			}
		}
		if checked < 1000 {
			t.Errorf("%s: only %d pieces compared", name, checked)
		}
	}
}

// A Python str is always valid, so invalid bytes reach the tokenizer the way
// bytes.decode('utf-8', 'replace') would deliver them: one U+FFFD per maximal invalid subpart.
func TestInvalidUTF8EncodesLikePython(t *testing.T) {
	fam := fixtures.Obj(t, extra(t).Must("invalid_utf8"))
	hexes := fixtures.Strings(t, fam.Must("hex"))
	ids := fixtures.Obj(t, fam.Must("ids"))
	for _, name := range shipped {
		tok := checkpoint(t, name)
		want := fixtures.List(t, ids.Must(name))
		for i, w := range want {
			raw, _ := hex.DecodeString(hexes[i])
			if got, wi := tok.Encode(string(raw)), fixtures.Ints(t, w); !equalIDs(got, wi) {
				t.Fatalf("%s Encode(%x): %s", name, raw, firstDifference(got, wi))
			}
		}
		if len(want) < 600 {
			t.Errorf("only %d cases", len(want))
		}
	}
}

func TestIDsFarBeyondTheVocabularyAreRefused(t *testing.T) {
	for _, id := range []string{"2147483647", "1000000000", "70000"} {
		spec := `{"model":{"type":"BPE","vocab":{"a":0,"<unk>":` + id + `},"merges":[],"unk_token":"<unk>"}}`
		if _, err := Parse([]byte(spec), nil); err == nil || !strings.Contains(err.Error(), "sparse") {
			t.Errorf("id %s: want a sparse-id refusal, got %v", id, err)
		}
	}
	spec := `{"model":{"type":"BPE","vocab":{"a":0,"<unk>":9000},"merges":[],"unk_token":"<unk>"}}`
	if _, err := Parse([]byte(spec), nil); err != nil {
		t.Errorf("a gap well inside the bound is allowed: %v", err)
	}
}

// A loaded Tokenizer is read-only, so any number of goroutines may share it.
func TestConcurrentUse(t *testing.T) {
	fam := fixtures.Obj(t, extra(t).Must("encode_fuzz"))
	for _, name := range shipped {
		tok := checkpoint(t, name)
		side := fixtures.Field(t, fam, name)
		texts := fixtures.Strings(t, fixtures.Field(t, side, "texts"))
		ids := fixtures.List(t, fixtures.Field(t, side, "ids"))
		want := make([][]int32, len(texts))
		for i := range texts {
			want[i] = fixtures.Ints(t, ids[i])
		}
		var wg sync.WaitGroup
		errs := make(chan string, 8)
		for g := 0; g < 8; g++ {
			wg.Add(1)
			go func(g int) {
				defer wg.Done()
				for i := g; i < len(texts); i += 1 + g%3 {
					got := tok.Encode(texts[i])
					if !equalIDs(got, want[i]) {
						errs <- fmt.Sprintf("%s goroutine %d text %d: %s", name, g, i, firstDifference(got, want[i]))
						return
					}
					_ = tok.Decode(got, true)
				}
			}(g)
		}
		wg.Wait()
		close(errs)
		for e := range errs {
			t.Error(e)
		}
	}
}
