package tokenizer

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// decoder turns the token strings of a decode back into text. Each step maps a list of pieces
// to a list of pieces, and the final text is the pieces joined with nothing between them.
type decoder interface {
	transform(tokens []string) []string
}

// decodeTokens runs the chain. With no decoder section the reference joins tokens with a space.
func decodeTokens(d decoder, tokens []string) string {
	if d == nil {
		return strings.Join(tokens, " ")
	}
	return strings.Join(d.transform(tokens), "")
}

func parseDecoder(n node) (decoder, error) {
	if n == nil {
		return nil, nil
	}
	kind, err := n.str("type", "")
	if err != nil {
		return nil, err
	}
	if kind == "" {
		return nil, fmt.Errorf("a decoder node must declare a type")
	}
	switch kind {
	case "ByteLevel":
		return byteLevelDecoder{}, nil
	case "Replace":
		return parseReplaceDecoder(n)
	case "ByteFallback":
		return byteFallbackDecoder{}, nil
	case "Fuse":
		return fuseDecoder{}, nil
	case "Strip":
		return parseStripDecoder(n)
	case "Sequence":
		steps, ok := n["decoders"].([]any)
		if !ok {
			return nil, fmt.Errorf("a Sequence decoder needs a decoders array")
		}
		chain := make(sequenceDecoder, 0, len(steps))
		for _, step := range steps {
			sn, err := asNode(step, "every Sequence decoder step")
			if err != nil {
				return nil, err
			}
			d, err := parseDecoder(sn)
			if err != nil {
				return nil, err
			}
			chain = append(chain, d)
		}
		return chain, nil
	}
	return nil, unsupported("the decoder %q is not implemented (ByteLevel, Replace, ByteFallback, "+
		"Fuse, Strip and Sequence are); refusing to load rather than decode it wrongly", kind)
}

type sequenceDecoder []decoder

func (s sequenceDecoder) transform(tokens []string) []string {
	current := append([]string(nil), tokens...)
	for _, step := range s {
		current = step.transform(current)
	}
	return current
}

// byteLevelDecoder maps the stand-in alphabet back to bytes. A token holding a character outside
// the alphabet is taken as plain text, so it survives; the whole is then decoded as lossy UTF-8.
type byteLevelDecoder struct{}

func (byteLevelDecoder) transform(tokens []string) []string {
	var raw []byte
	for _, token := range tokens {
		mapped, ok := mapToBytes(token)
		if !ok {
			mapped = []byte(token)
		}
		raw = append(raw, mapped...)
	}
	return []string{unicodedata.Lossy(raw)}
}

func mapToBytes(token string) ([]byte, bool) {
	out := make([]byte, 0, len(token))
	for _, r := range token {
		b := byteAlphabet.byteFor(r)
		if b < 0 {
			return nil, false
		}
		out = append(out, byte(b))
	}
	return out, true
}

type replaceDecoder struct{ pattern, content string }

func parseReplaceDecoder(n node) (decoder, error) {
	var literal string
	switch p := n["pattern"].(type) {
	case map[string]any:
		s, isString := p["String"].(string)
		if !isString {
			if _, regex := p["Regex"]; regex {
				return nil, unsupported("only a literal String pattern is implemented in a Replace " +
					"decoder; this checkpoint declares a Regex one")
			}
			return nil, fmt.Errorf("a Replace decoder needs a String pattern")
		}
		literal = s
	case string:
		literal = p
	default:
		return nil, fmt.Errorf("a Replace decoder needs a String pattern")
	}
	if literal == "" {
		return nil, fmt.Errorf("a Replace decoder cannot match the empty string")
	}
	content, err := n.str("content", "")
	if err != nil {
		return nil, err
	}
	return replaceDecoder{literal, content}, nil
}

func (r replaceDecoder) transform(tokens []string) []string {
	out := make([]string, len(tokens))
	for i, t := range tokens {
		out[i] = strings.ReplaceAll(t, r.pattern, r.content)
	}
	return out
}

// byteFallbackDecoder reads runs of <0xNN> tokens as bytes. A run that is valid UTF-8 becomes
// one piece; one that is not becomes a U+FFFD per byte.
type byteFallbackDecoder struct{}

func (byteFallbackDecoder) transform(tokens []string) []string {
	var out []string
	var run []byte
	flush := func() {
		if len(run) == 0 {
			return
		}
		if utf8.Valid(run) {
			out = append(out, string(run))
		} else {
			for range run {
				out = append(out, "�")
			}
		}
		run = run[:0]
	}
	for _, token := range tokens {
		if b, ok := byteToken(token); ok {
			run = append(run, b)
			continue
		}
		flush()
		out = append(out, token)
	}
	flush()
	return out
}

// byteToken parses "<0xNN>". Like Rust's from_str_radix it accepts lower-case digits and a
// leading '+' on the high digit, in which case only the low digit is read.
func byteToken(token string) (byte, bool) {
	if len(token) != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>' {
		return 0, false
	}
	high, low := token[3], token[4]
	if high == '+' {
		l := hexValue(low)
		return byte(l), l >= 0
	}
	h, l := hexValue(high), hexValue(low)
	if h < 0 || l < 0 {
		return 0, false
	}
	return byte(h<<4 | l), true
}

func hexValue(c byte) int {
	switch {
	case c >= '0' && c <= '9':
		return int(c - '0')
	case c >= 'A' && c <= 'F':
		return int(c-'A') + 10
	case c >= 'a' && c <= 'f':
		return int(c-'a') + 10
	}
	return -1
}

type fuseDecoder struct{}

func (fuseDecoder) transform(tokens []string) []string {
	return []string{strings.Join(tokens, "")}
}

type stripDecoder struct {
	content     rune
	start, stop int
}

func parseStripDecoder(n node) (decoder, error) {
	c, err := n.str("content", " ")
	if err != nil {
		return nil, err
	}
	if utf8.RuneCountInString(c) != 1 {
		return nil, fmt.Errorf("a Strip decoder's content must be one character, got %q", c)
	}
	start, err := n.integer("start", 0)
	if err != nil {
		return nil, err
	}
	stop, err := n.integer("stop", 0)
	if err != nil {
		return nil, err
	}
	r, _ := utf8.DecodeRuneInString(c)
	return stripDecoder{r, start, stop}, nil
}

func (s stripDecoder) transform(tokens []string) []string {
	out := make([]string, len(tokens))
	for i, t := range tokens {
		for n := 0; n < s.start && strings.HasPrefix(t, string(s.content)); n++ {
			t = t[utf8.RuneLen(s.content):]
		}
		for n := 0; n < s.stop && strings.HasSuffix(t, string(s.content)); n++ {
			t = t[:len(t)-utf8.RuneLen(s.content)]
		}
		out[i] = t
	}
	return out
}
