package tokenizer

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// preTokenizer splits normalized text into the pieces BPE runs on.
type preTokenizer interface {
	preTokenize(text string) []string
}

type wholeText struct{}

func (wholeText) preTokenize(text string) []string {
	if text == "" {
		return nil
	}
	return []string{text}
}

type sequence []preTokenizer

func (s sequence) preTokenize(text string) []string {
	current := []string{text}
	for _, step := range s {
		next := make([]string, 0, len(current))
		for _, piece := range current {
			next = append(next, step.preTokenize(piece)...)
		}
		current = next
	}
	return current
}

func parsePreTokenizer(n node) (preTokenizer, error) {
	if n == nil {
		return wholeText{}, nil
	}
	kind, err := n.str("type", "")
	if err != nil {
		return nil, err
	}
	if kind == "" {
		return nil, fmt.Errorf("a pre-tokenizer must declare a type")
	}
	switch kind {
	case "ByteLevel":
		prefix, err := n.boolean("add_prefix_space", true)
		if err != nil {
			return nil, err
		}
		useRegex, err := n.boolean("use_regex", true)
		if err != nil {
			return nil, err
		}
		return byteLevel{addPrefixSpace: prefix, useRegex: useRegex}, nil
	case "Metaspace":
		return parseMetaspace(n)
	case "Sequence":
		children, err := n.list("pretokenizers")
		if err != nil {
			return nil, err
		}
		steps := make(sequence, 0, len(children))
		for _, child := range children {
			cn, err := asNode(child, "a pre-tokenizer in a Sequence")
			if err != nil {
				return nil, err
			}
			step, err := parsePreTokenizer(cn)
			if err != nil {
				return nil, err
			}
			steps = append(steps, step)
		}
		return steps, nil
	}
	return nil, unsupported("the pre-tokenizer %q is not implemented; refusing to load rather "+
		"than produce ids that are wrong in a way no answer-level test would localise", kind)
}

// alphabet is the byte-level stand-in alphabet: every byte maps to a printable character, the
// printable ASCII and Latin-1 bytes to themselves and the other 68 to U+0100 upward.
type alphabet struct {
	toRune [256]rune
	toByte [0x144]int16 // indexed by rune; -1 where the rune stands for no byte
}

var byteAlphabet = newAlphabet()

func newAlphabet() *alphabet {
	a := &alphabet{}
	for i := range a.toByte {
		a.toByte[i] = -1
	}
	lifted := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			a.toRune[b] = rune(b)
		} else {
			a.toRune[b] = rune(256 + lifted)
			lifted++
		}
		a.toByte[a.toRune[b]] = int16(b)
	}
	return a
}

// byteFor returns the byte r stands for, or -1.
func (a *alphabet) byteFor(r rune) int {
	if r < 0 || int(r) >= len(a.toByte) {
		return -1
	}
	return int(a.toByte[r])
}

// encode spells piece's UTF-8 bytes in the stand-in alphabet.
func (a *alphabet) encode(piece string) string {
	out := make([]byte, 0, len(piece)+len(piece)/4)
	for i := 0; i < len(piece); i++ {
		out = utf8.AppendRune(out, a.toRune[piece[i]])
	}
	return string(out)
}

var contractions = [...]string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}

// byteLevel is the GPT-2 pre-tokenizer: the pattern
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// written as a scanner, because the standard regexp package has no lookahead.
type byteLevel struct {
	addPrefixSpace bool
	useRegex       bool
}

func (p byteLevel) preTokenize(text string) []string {
	if text == "" {
		return nil
	}
	source := text
	if p.addPrefixSpace && !strings.HasPrefix(source, " ") {
		source = " " + source
	}
	if !p.useRegex {
		if source == "" {
			return nil
		}
		return []string{byteAlphabet.encode(source)}
	}
	out := make([]string, 0, len(source)/4+1)
	for at := 0; at < len(source); {
		end := cut(source, at)
		out = append(out, byteAlphabet.encode(source[at:end]))
		at = end
	}
	return out
}

// class reports whether r belongs to branch 0 (letters), 1 (numbers) or 2 (anything else that
// is not white space).
func class(r rune, branch int) bool {
	switch branch {
	case 0:
		return isLetter(r)
	case 1:
		return isNumber(r)
	}
	return !isWhiteSpace(r) && !isLetter(r) && !isNumber(r)
}

func runEnd(source string, at, branch int) int {
	for at < len(source) {
		r, w := utf8.DecodeRuneInString(source[at:])
		if !class(r, branch) {
			break
		}
		at += w
	}
	return at
}

// cut returns where the piece starting at from ends. It always advances.
func cut(source string, from int) int {
	for _, c := range contractions {
		if strings.HasPrefix(source[from:], c) {
			return from + len(c)
		}
	}
	first, width := utf8.DecodeRuneInString(source[from:])
	for branch := 0; branch < 3; branch++ {
		if first == ' ' {
			if after := from + 1; after < len(source) {
				if r, _ := utf8.DecodeRuneInString(source[after:]); class(r, branch) {
					return runEnd(source, after, branch)
				}
			}
		} else if class(first, branch) {
			return runEnd(source, from, branch)
		}
	}
	if isWhiteSpace(first) {
		end := from
		for end < len(source) {
			r, w := utf8.DecodeRuneInString(source[end:])
			if !isWhiteSpace(r) {
				break
			}
			end += w
		}
		if end >= len(source) {
			return end
		}
		// \s+(?!\S) gives the last white-space character back to the next piece.
		_, w := lastRune(source[:end])
		if end-w > from {
			return end - w
		}
		return end // a single space: \s+(?!\S) fails and \s+ takes it
	}
	return from + width
}

// metaspace swaps spaces for a marker character and splits before each marker.
type metaspace struct {
	replacement  string
	prependFirst bool // prepend_scheme "first" or "always"
	split        bool
}

func parseMetaspace(n node) (preTokenizer, error) {
	replacement, err := n.str("replacement", "▁")
	if err != nil {
		return nil, err
	}
	if utf8.RuneCountInString(replacement) != 1 {
		return nil, fmt.Errorf("a Metaspace replacement must be exactly one code point, got %q", replacement)
	}
	scheme, err := n.str("prepend_scheme", "always")
	if err != nil {
		return nil, err
	}
	split, err := n.boolean("split", true)
	if err != nil {
		return nil, err
	}
	switch scheme {
	case "always", "first":
		return metaspace{replacement, true, split}, nil
	case "never":
		return metaspace{replacement, false, split}, nil
	}
	return nil, fmt.Errorf("unknown Metaspace prepend_scheme %q", scheme)
}

func (m metaspace) preTokenize(text string) []string {
	if text == "" {
		return nil
	}
	source := strings.ReplaceAll(text, " ", m.replacement)
	if m.prependFirst && !strings.HasPrefix(source, m.replacement) {
		source = m.replacement + source
	}
	if !m.split {
		return []string{source}
	}
	var out []string
	start := 0
	_, from := utf8.DecodeRuneInString(source) // a marker at the very start does not split
	for {
		next := strings.Index(source[from:], m.replacement)
		if next < 0 {
			break
		}
		at := from + next
		out = append(out, source[start:at])
		start, from = at, at+len(m.replacement)
	}
	if start < len(source) {
		out = append(out, source[start:])
	}
	return out
}
