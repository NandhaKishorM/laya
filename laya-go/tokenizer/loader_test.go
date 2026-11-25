package tokenizer

import (
	"encoding/json"
	"errors"
	"os"
	"strconv"
	"strings"
	"testing"
)

// A tiny byte-level tokenizer built in the test, so the loader is checked without a checkpoint.
func byteLevelSpec(extra string) string {
	vocab := make([]string, 0, 260)
	for b := 0; b < 256; b++ {
		key, _ := json.Marshal(string(byteAlphabet.toRune[b]))
		vocab = append(vocab, string(key)+`:`+strconv.Itoa(b))
	}
	vocab = append(vocab, `"ab":256`)
	return `{"model":{"type":"BPE","vocab":{` + strings.Join(vocab, ",") + `},"merges":["a b"]},` +
		`"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"use_regex":true}` + extra + `}`
}

func TestSmallTokenizerRoundTrips(t *testing.T) {
	tok, err := Parse([]byte(byteLevelSpec(`,"decoder":{"type":"ByteLevel"}`)), nil)
	if err != nil {
		t.Fatal(err)
	}
	ids := tok.Encode("ab ab")
	if got := tok.Decode(ids, true); got != "ab ab" {
		t.Errorf("round trip = %q (ids %v)", got, ids)
	}
	if ids[0] != 256 {
		t.Errorf("the a b merge must fire, got %v", ids)
	}
	if tok.VocabSize() != 257 || tok.HighestID() != 256 {
		t.Errorf("VocabSize = %d, HighestID = %d", tok.VocabSize(), tok.HighestID())
	}
	if s, ok := tok.IDToToken(256); !ok || s != "ab" {
		t.Errorf("IDToToken(256) = %q, %v", s, ok)
	}
	if _, ok := tok.IDToToken(-1); ok {
		t.Error("a negative id has no token")
	}
	if id, ok := tok.TokenToID("ab"); !ok || id != 256 {
		t.Errorf("TokenToID = %d, %v", id, ok)
	}
}

func TestLoaderRefuses(t *testing.T) {
	good := byteLevelSpec("")
	for name, c := range map[string]struct {
		spec        string
		unsupported bool
	}{
		"a non-BPE model":                              {strings.Replace(good, `"BPE"`, `"WordPiece"`, 1), true},
		"an unknown normalizer":                        {strings.Replace(good, `"normalizer":null`, `"normalizer":{"type":"BertNormalizer"}`, 1), true},
		"an unknown pre-tokenizer":                     {strings.Replace(good, `"pre_tokenizer":{"type":"ByteLevel"`, `"pre_tokenizer":{"type":"Whitespace"`, 1), true},
		"an unknown decoder":                           {byteLevelSpec(`,"decoder":{"type":"WordPiece"}`), true},
		"a merge naming a symbol the vocabulary lacks": {strings.Replace(good, `"a b"`, `"a zz"`, 1), false},
		"a merge whose result is missing":              {strings.Replace(good, `"a b"`, `"a c"`, 1), false},
		"a merge that is not a pair":                   {strings.Replace(good, `"a b"`, `["a"]`, 1), false},
		"a space-joined merge with an empty half":      {strings.Replace(good, `"a b"`, `"a "`, 1), false},
		"an added token with no id":                    {byteLevelSpec(`,"added_tokens":[{"content":"x"}]`), false},
		"two added tokens sharing an id":               {byteLevelSpec(`,"added_tokens":[{"content":"x","id":300},{"content":"y","id":300}]`), false},
		"a vocabulary giving one id to two tokens":     {strings.Replace(good, `"ab":256`, `"ab":0`, 1), false},
		"no vocabulary":                                {`{"model":{"type":"BPE","merges":[]}}`, false},
		"not JSON":                                     {`{`, false},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := Parse([]byte(c.spec), nil)
			if err == nil {
				t.Fatal("must be refused, not loaded")
			}
			if got := errors.Is(err, ErrUnsupported); got != c.unsupported {
				t.Errorf("errors.Is(err, ErrUnsupported) = %v for %v", got, err)
			}
		})
	}
}

func TestLoaderRefusesALossyTokenizer(t *testing.T) {
	// No unk_token, no byte fallback, and not byte-level: a character outside the vocabulary
	// would vanish. The same vocabulary missing alphabet symbols, byte-level, is refused too.
	lossy := `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`
	if _, err := Parse([]byte(lossy), nil); !errors.Is(err, ErrUnsupported) {
		t.Errorf("a non-byte-level tokenizer that can drop text: %v", err)
	}
	partial := `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"pre_tokenizer":{"type":"ByteLevel"}}`
	if _, err := Parse([]byte(partial), nil); err == nil || !strings.Contains(err.Error(), "absent from its vocabulary") {
		t.Errorf("a byte-level tokenizer missing alphabet symbols: %v", err)
	}
	withUnk := `{"model":{"type":"BPE","vocab":{"a":0,"<unk>":1},"merges":[],"unk_token":"<unk>"}}`
	// Expected ids are what the crate returns for the same model.
	for _, c := range []struct {
		fuse bool
		text string
		want []int32
	}{
		{false, "azz a", []int32{0, 1, 1, 1, 0}},
		{false, "zzz", []int32{1, 1, 1}},
		{false, "zz a zz", []int32{1, 1, 1, 0, 1, 1, 1}},
		{true, "azz a", []int32{0, 1, 0}},
		{true, "zzz", []int32{1}},
		{true, "zz a zz", []int32{1, 0, 1}},
	} {
		spec := withUnk
		if c.fuse {
			spec = strings.Replace(withUnk, `"unk_token"`, `"fuse_unk":true,"unk_token"`, 1)
		}
		tok, err := Parse([]byte(spec), nil)
		if err != nil {
			t.Fatal(err)
		}
		if got := tok.Encode(c.text); !equalIDs(got, c.want) {
			t.Errorf("fuse_unk=%v Encode(%q) = %v, want %v", c.fuse, c.text, got, c.want)
		}
	}
}

func TestAddedTokens(t *testing.T) {
	spec := byteLevelSpec(`,"added_tokens":[` +
		`{"content":"<S>","id":300,"special":true,"normalized":false},` +
		`{"content":"<SS>","id":301,"special":true,"normalized":false},` +
		`{"content":"[M]","id":302,"special":true,"lstrip":true,"rstrip":true},` +
		`{"content":"word","id":303,"single_word":true},` +
		`{"content":"éclair","id":304,"normalized":true},` +
		`{"content":"§","id":305},{"content":"!","id":306}],` +
		`"decoder":{"type":"ByteLevel"}`)
	tok, err := Parse([]byte(spec), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("<SS><S>"); !equalIDs(got, []int32{301, 300}) {
		t.Errorf("the longest token wins at a position, got %v", got)
	}
	if got := tok.Encode("a   [M]   b"); !equalIDs(got, []int32{97, 302, 98}) {
		t.Errorf("lstrip and rstrip absorb the white space around a token, got %v", got)
	}
	if got := tok.Encode("word"); !equalIDs(got, []int32{303}) {
		t.Errorf("a single_word token alone, got %v", got)
	}
	if got := tok.Encode("swords word."); len(got) < 3 || got[len(got)-2] != 303 {
		t.Errorf("a single_word token inside a word is not matched but a free one is, got %v", got)
	}
	if got := tok.Encode("éclair"); !equalIDs(got, []int32{304}) {
		t.Errorf("a normalized token matches after normalization, got %v", got)
	}
	if got := tok.Encode("a\u00a7b!"); !equalIDs(got, []int32{97, 305, 98, 306}) {
		t.Errorf("one-character tokens, ASCII and not, got %v", got)
	}
	if got := tok.Encode("<S"); !equalIDs(got, []int32{60, 83}) {
		t.Errorf("a prefix of a token is not the token, got %v", got)
	}
	if got := tok.Decode([]int32{300, 97, 301}, true); got != "a" {
		t.Errorf("special tokens are skipped on request, got %q", got)
	}
	if got := tok.Decode([]int32{300, 97}, false); got != "<S>a" {
		t.Errorf("and kept otherwise, got %q", got)
	}
	if id, ok := tok.SpecialID("<S>"); !ok || id != 300 {
		t.Errorf("SpecialID = %d, %v", id, ok)
	}
	if _, ok := tok.SpecialID("word"); ok {
		t.Error("an added token that is not special is not in the special table")
	}
	if n := len(tok.SpecialTokens()); n != 3 {
		t.Errorf("%d special tokens", n)
	}
}

func TestRolesFromTokenizerConfig(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir+"/tokenizer.json", byteLevelSpec(`,"added_tokens":[{"content":"<c>","id":300,"special":true},{"content":"<m>","id":301,"special":true}]`))
	writeFile(t, dir+"/tokenizer_config.json", `{"cls_token":"<c>","mask_token":{"content":"<m>"},"pad_token":null}`)
	tok, err := LoadFS(os.DirFS(dir), ".")
	if err != nil {
		t.Fatal(err)
	}
	if id, ok := tok.CLS(); !ok || id != 300 {
		t.Errorf("CLS = %d, %v", id, ok)
	}
	if id, ok := tok.Mask(); !ok || id != 301 || tok.MaskToken() != "<m>" {
		t.Errorf("Mask = %d, %v, %q", id, ok, tok.MaskToken())
	}
	if _, ok := tok.Pad(); ok {
		t.Error("a null role is absent")
	}
	if _, ok := tok.SEP(); ok {
		t.Error("an unnamed role is absent")
	}
	writeFile(t, dir+"/tokenizer_config.json", `{"cls_token":"<nope>"}`)
	if _, err := LoadFS(os.DirFS(dir), "."); err == nil {
		t.Error("a role naming a token the vocabulary lacks must be refused")
	}
	writeFile(t, dir+"/tokenizer_config.json", `{"cls_token":5}`)
	if _, err := LoadFS(os.DirFS(dir), "."); err == nil {
		t.Error("a role that is not a string or an object must be refused")
	}
	// Load prefers <dir>/tokenizer/ when it holds a tokenizer.json.
	model := t.TempDir()
	writeFile(t, model+"/tokenizer/tokenizer.json", byteLevelSpec(""))
	if _, err := LoadFS(os.DirFS(model), "."); err != nil {
		t.Errorf("LoadFS(model dir): %v", err)
	}
	sub := t.TempDir()
	writeFile(t, sub+"/models/m/tokenizer.json", byteLevelSpec(""))
	if _, err := LoadFS(os.DirFS(sub), "models/m"); err != nil {
		t.Errorf("LoadFS(sub directory): %v", err)
	}
	if _, err := LoadFS(os.DirFS(t.TempDir()), "."); err == nil {
		t.Error("a directory with no tokenizer must be an error")
	}
}

func TestNormalizerAndPreTokenizerRefusals(t *testing.T) {
	for name, spec := range map[string]string{
		"an unknown normalizer":          `{"type":"BertNormalizer"}`,
		"a typeless normalizer":          `{}`,
		"a Replace with no content":      `{"type":"Replace","pattern":{"String":"a"}}`,
		"a Replace of the empty string":  `{"type":"Replace","pattern":{"String":""},"content":"x"}`,
		"a Replace with no pattern kind": `{"type":"Replace","pattern":{},"content":"x"}`,
		"a bad regex":                    `{"type":"Replace","pattern":{"Regex":"("},"content":"x"}`,
		"a Sequence with no steps":       `{"type":"Sequence"}`,
		"a Sequence of non-objects":      `{"type":"Sequence","normalizers":[1]}`,
	} {
		t.Run(name, func(t *testing.T) {
			n, err := parseNode([]byte(spec), "normalizer")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := parseNormalizer(n); err == nil {
				t.Error("must be refused")
			}
		})
	}
	// The reference reads \s, \d, \w and \b as Unicode; Go's RE2 reads them as ASCII.
	for _, expr := range []string{`\\s+`, `a\\bb`, `[\\d]`, `\\w`, `\\p{L}`} {
		n, _ := parseNode([]byte(`{"type":"Replace","pattern":{"Regex":"`+expr+`"},"content":"x"}`), "normalizer")
		if _, err := parseNormalizer(n); !errors.Is(err, ErrUnsupported) {
			t.Errorf("the regex %s must be refused as ErrUnsupported, got %v", expr, err)
		}
	}
	n, _ := parseNode([]byte(`{"type":"Replace","pattern":{"Regex":"\\\\\\\\s"},"content":"x"}`), "normalizer")
	if norm, err := parseNormalizer(n); err != nil || norm(`a\\s`) != `ax` {
		t.Errorf("an escaped backslash before s is a literal, not a class: %v", err)
	}
	for name, spec := range map[string]string{
		"an unknown pre-tokenizer":    `{"type":"Whitespace"}`,
		"a typeless pre-tokenizer":    `{}`,
		"a two-character Metaspace":   `{"type":"Metaspace","replacement":"ab"}`,
		"an unknown Metaspace scheme": `{"type":"Metaspace","prepend_scheme":"sometimes"}`,
		"a Sequence of non-objects":   `{"type":"Sequence","pretokenizers":[1]}`,
	} {
		t.Run(name, func(t *testing.T) {
			n, _ := parseNode([]byte(spec), "pre_tokenizer")
			if _, err := parsePreTokenizer(n); err == nil {
				t.Error("must be refused")
			}
		})
	}
	n, _ = parseNode([]byte(`{"type":"Whitespace"}`), "pre_tokenizer")
	if _, err := parsePreTokenizer(n); !errors.Is(err, ErrUnsupported) {
		t.Errorf("an unimplemented pre-tokenizer must wrap ErrUnsupported, got %v", err)
	}
}

func TestPreTokenizerSequence(t *testing.T) {
	n, _ := parseNode([]byte(`{"type":"Sequence","pretokenizers":[{"type":"Metaspace","prepend_scheme":"never"},{"type":"ByteLevel","add_prefix_space":false}]}`), "pre_tokenizer")
	p, err := parsePreTokenizer(n)
	if err != nil {
		t.Fatal(err)
	}
	// The crate gives ['ab', 'âĸģ', 'cd'] for the same sequence.
	if got := strings.Join(p.preTokenize("ab cd"), "|"); got != "ab|\u00e2\u0138\u0123|cd" {
		t.Errorf("sequence = %q", got)
	}
	if got := (wholeText{}).preTokenize(""); got != nil {
		t.Errorf("empty text has no pieces, got %v", got)
	}
}

func TestPreTokenizerCornerCases(t *testing.T) {
	p := byteLevel{addPrefixSpace: false, useRegex: true}
	for _, c := range []struct {
		in   string
		want []string
	}{
		// An inner white-space run gives its last character back to the next piece.
		{"a  b\t\tc\n\nd\r\ne   f", []string{"a", "Ġ", "Ġb", "ĉ", "ĉ", "c", "Ċ", "Ċ", "d", "č", "Ċ", "e", "ĠĠ", "Ġf"}},
		{"abc123", []string{"abc", "123"}},
		{"123abc", []string{"123", "abc"}},
		{"a 1 b", []string{"a", "Ġ1", "Ġb"}},
		{"x's value", []string{"x", "'s", "Ġvalue"}},
		{"", nil},
	} {
		got := p.preTokenize(c.in)
		if strings.Join(got, "|") != strings.Join(c.want, "|") || len(got) != len(c.want) {
			t.Errorf("preTokenize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
