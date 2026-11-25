package tokenizer

import (
	"errors"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

func decodeFamily(t *testing.T) fixtures.Object {
	t.Helper()
	return fixtures.Load(t, "decode_text.json")
}

// Every recorded decode matches, whole and sliced: decoding a slice of ids is where a per-token
// decoder differs from a whole-text one, so the slices are the point of this family.
func TestDecodeMatchesPython(t *testing.T) {
	by := fixtures.Obj(t, decodeFamily(t).Must("by"))
	slices := 0
	for _, name := range shipped {
		entry := fixtures.Field(t, by, name)
		if _, skipped := fixtures.Obj(t, entry).Get("skipped"); skipped {
			t.Fatalf("the decode golden for %s is a skip marker", name)
		}
		tok := checkpoint(t, name)
		cases := fixtures.List(t, fixtures.Field(t, entry, "cases"))
		if len(cases) < 23 {
			t.Fatalf("the decode corpus lost cases: %d of at least 23", len(cases))
		}
		for _, raw := range cases {
			text := fixtures.Str(t, fixtures.Field(t, raw, "text"))
			ids := fixtures.Ints(t, fixtures.Field(t, raw, "ids"))
			label := name + "/" + describe(text)
			t.Run(label+"/ids", func(t *testing.T) {
				if got := tok.Encode(text); !equalIDs(got, ids) {
					t.Errorf("%s", firstDifference(got, ids))
				}
			})
			t.Run(label+"/whole", func(t *testing.T) {
				if got, want := tok.Decode(ids, true), fixtures.Str(t, fixtures.Field(t, raw, "decoded")); got != want {
					t.Errorf("decode = %q, want %q", got, want)
				}
			})
			sl := fixtures.List(t, fixtures.Field(t, raw, "slices"))
			slices += len(sl)
			if len(sl) == 0 {
				t.Run(label+"/no-ids-to-slice", func(t *testing.T) {
					if text != "" || len(ids) != 0 {
						t.Errorf("only the empty text should tokenize to nothing, but %q did", text)
					}
				})
			} else {
				t.Run(label+"/slices", func(t *testing.T) {
					for _, s := range sl {
						start := fixtures.Int(t, fixtures.Field(t, s, "start"))
						end := min(start+fixtures.Int(t, fixtures.Field(t, s, "size")), len(ids))
						want := fixtures.Str(t, fixtures.Field(t, s, "decoded"))
						if got := tok.Decode(ids[start:end], true); got != want {
							t.Errorf("slice [%d,%d) = %q, want %q", start, end, got, want)
						}
					}
				})
			}
			t.Run(label+"/specials", func(t *testing.T) {
				sp := fixtures.Field(t, raw, "with_specials")
				with := fixtures.Ints(t, fixtures.Field(t, sp, "ids"))
				if got, want := tok.Decode(with, true), fixtures.Str(t, fixtures.Field(t, sp, "skipping_specials")); got != want {
					t.Errorf("skipping specials = %q, want %q", got, want)
				}
				if got, want := tok.Decode(with, false), fixtures.Str(t, fixtures.Field(t, sp, "kept")); got != want {
					t.Errorf("keeping specials = %q, want %q", got, want)
				}
			})
		}
	}
	if slices < 400 {
		t.Errorf("the slice cases are the point of this family and there are only %d", slices)
	}
}

func TestRecordedDecoderChains(t *testing.T) {
	by := fixtures.Obj(t, decodeFamily(t).Must("by"))
	en := fixtures.Field(t, fixtures.Field(t, by, "english"), "decoder")
	if got := fixtures.Str(t, fixtures.Field(t, en, "type")); got != "ByteLevel" {
		t.Errorf("the english decoder is %q now", got)
	}
	ml := fixtures.Field(t, fixtures.Field(t, by, "multilingual"), "decoder")
	var types []string
	for _, step := range fixtures.List(t, fixtures.Field(t, ml, "decoders")) {
		types = append(types, fixtures.Str(t, fixtures.Field(t, step, "type")))
	}
	if strings.Join(types, ",") != "Replace,ByteFallback,Fuse" {
		t.Errorf("the multilingual decoder chain is %v now", types)
	}
}

func fromSpec(t *testing.T, spec any) *Tokenizer {
	t.Helper()
	tok, err := Parse(fixtures.Marshal(t, spec), nil)
	if err != nil {
		t.Fatalf("load the recorded spec: %v", err)
	}
	return tok
}

// The decoder shapes no shipped checkpoint declares are recorded against the crate directly.
func TestSyntheticDecoderShapes(t *testing.T) {
	shapes := fixtures.Obj(t, decodeFamily(t).Must("synthetic_decoders"))
	if len(shapes) < 5 {
		t.Fatalf("five shapes were recorded and only %d are here", len(shapes))
	}
	for _, shape := range shapes {
		t.Run(shape.Key, func(t *testing.T) {
			tok := fromSpec(t, fixtures.Field(t, shape.Value, "spec"))
			cases := fixtures.List(t, fixtures.Field(t, shape.Value, "cases"))
			if len(cases) == 0 {
				t.Fatal("no cases recorded")
			}
			for _, c := range cases {
				ids := fixtures.Ints(t, fixtures.Field(t, c, "ids"))
				if got, want := tok.Decode(ids, false), fixtures.Str(t, fixtures.Field(t, c, "decoded")); got != want {
					t.Errorf("ids %v decode to %q, want %q", ids, got, want)
				}
			}
			// The crate panics on these; this port returns empty text and must not.
			if panics, ok := fixtures.Obj(t, shape.Value).Get("crate_panics"); ok {
				for _, p := range fixtures.List(t, panics) {
					ids := fixtures.Ints(t, p)
					if got := tok.Decode(ids, false); got != "" {
						t.Errorf("ids %v decode to %q, want empty text", ids, got)
					}
				}
			}
		})
	}
}

// ByteLevel falls back to raw text per token, not per character: a token holding one character
// outside the alphabet is kept whole.
func TestByteLevelFallsBackPerToken(t *testing.T) {
	scope := fixtures.Obj(t, decodeFamily(t).Must("mixed_token"))
	if got := fixtures.Str(t, scope.Must("scope")); got != "token" {
		t.Fatalf("the crate's fallback scope is %q now; this port assumes per token", got)
	}
	tok := fromSpec(t, scope.Must("spec"))
	for _, c := range fixtures.List(t, scope.Must("cases")) {
		ids := fixtures.Ints(t, fixtures.Field(t, c, "ids"))
		if got, want := tok.Decode(ids, false), fixtures.Str(t, fixtures.Field(t, c, "decoded")); got != want {
			t.Errorf("%s: decode = %q, want %q", fixtures.Str(t, fixtures.Field(t, c, "why")), got, want)
		}
	}
}

func TestDecodeEdges(t *testing.T) {
	tok := checkpoint(t, "english")
	if got := tok.Decode(nil, true); got != "" {
		t.Errorf("no ids decode to %q", got)
	}
	if got := tok.Decode([]int32{1 << 30}, true); got != "" {
		t.Errorf("an id outside the vocabulary decodes to %q", got)
	}
	a := tok.Encode("a")[0]
	if got := tok.Decode([]int32{1 << 30, a, -1}, true); got != "a" {
		t.Errorf("unknown and negative ids are skipped, got %q", got)
	}
	if got := tok.Decode([]int32{50275}, true); got != "   " {
		t.Errorf("the literal-whitespace added token must survive decoding, got %q", got)
	}
	for _, s := range []string{" ", "   "} {
		if got := tok.Decode(tok.Encode(s), true); got != s {
			t.Errorf("%q does not round-trip: %q", s, got)
		}
	}
}

func TestMultilingualDecodeIsLossyByContract(t *testing.T) {
	ml := checkpoint(t, "multilingual")
	if got := ml.Decode(ml.Encode("Hello world"), true); got != " Hello world" {
		t.Errorf("Metaspace prepends a marker the decoder turns into a space, got %q", got)
	}
	// U+F0000 is four bytes of byte fallback; two of them are not a character.
	ids := ml.Encode("\U000F0000")
	if len(ids) < 3 {
		t.Fatalf("expected a Metaspace marker plus a byte run, got %d ids", len(ids))
	}
	if got := ml.Decode(ids[:3], true); got != " ��" {
		t.Errorf("two bytes of a four-byte character give one U+FFFD each, got %q", got)
	}
	if got := ml.Decode(ids, true); got != " \U000F0000" {
		t.Errorf("the whole byte run decodes to the character, got %q", got)
	}
}

func TestDecoderRefusals(t *testing.T) {
	for name, spec := range map[string]string{
		"an unimplemented type":     `{"type":"Metaspace"}`,
		"a Regex Replace":           `{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "}`,
		"a Replace with no pattern": `{"type":"Replace","content":" "}`,
		"a Sequence with no steps":  `{"type":"Sequence"}`,
		"a Sequence of non-objects": `{"type":"Sequence","decoders":["ByteLevel"]}`,
		"a Strip of two characters": `{"type":"Strip","content":"ab"}`,
		"a decoder with no type":    `{}`,
	} {
		t.Run(name, func(t *testing.T) {
			n, err := parseNode([]byte(spec), "decoder")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := parseDecoder(n); err == nil {
				t.Error("must be refused, not guessed at")
			}
		})
	}
	n, _ := parseNode([]byte(`{"type":"Metaspace"}`), "decoder")
	if _, err := parseDecoder(n); !errors.Is(err, ErrUnsupported) {
		t.Errorf("an unimplemented decoder must wrap ErrUnsupported, got %v", err)
	}
	if got := decodeTokens(nil, []string{"ab", "cd"}); got != "ab cd" {
		t.Errorf("no decoder joins with a space, got %q", got)
	}
}

func TestByteTableRoundTrips(t *testing.T) {
	for b := 0; b < 256; b++ {
		if got := byteAlphabet.byteFor(byteAlphabet.toRune[b]); got != b {
			t.Errorf("byte 0x%02X does not survive the round trip (%d)", b, got)
		}
	}
	if byteAlphabet.byteFor(' ') >= 0 {
		t.Error("a literal space must NOT be in the alphabet: 0x20 is lifted to U+0120")
	}
	if byteAlphabet.byteFor(-1) != -1 || byteAlphabet.byteFor(0x10000) != -1 {
		t.Error("a rune outside the alphabet stands for no byte")
	}
}

func TestLossyUTF8ReplacesMaximalSubparts(t *testing.T) {
	for _, c := range []struct {
		in   []byte
		want string
	}{
		{[]byte("ok"), "ok"},
		{[]byte{0xE2, 0x82}, "�"},         // a truncated 3-byte sequence is one subpart
		{[]byte{0xE2, 0x82, 'a'}, "�a"},   // and stops at the byte that breaks it
		{[]byte{0xF0, 0x9F, 0x98}, "�"},   // truncated emoji
		{[]byte{0xC0, 0x80}, "��"},        // C0 never starts a sequence
		{[]byte{0xED, 0xA0, 0x80}, "���"}, // an encoded surrogate: ED then A0 breaks it
		{[]byte{0xFF, 'x', 0x80}, "�x�"},
		{[]byte{0xF4, 0x90, 0x80, 0x80}, "����"}, // above U+10FFFF
	} {
		if got := unicodedata.Lossy(c.in); got != c.want {
			t.Errorf("unicodedata.Lossy(% X) = %+q, want %+q", c.in, got, c.want)
		}
	}
}
