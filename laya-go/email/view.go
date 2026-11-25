package email

import (
	"strings"
	"unicode"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// fold is the relation re.I applies to a str pattern: simple lowercase, plus the two
// equivalences sre adds that lowercase does not give (dotless i with i, long s with s). A test
// holds it to CPython over every code point for every letter the patterns use.
func fold(r rune) rune {
	switch r {
	case 0x131:
		return 'i'
	case 0x17f:
		return 's'
	}
	return unicode.ToLower(r)
}

// placeholders maps each non-ASCII character a pattern spells to the uppercase letter standing
// for it in the view. Folded text never holds an uppercase ASCII letter, so none can collide.
var placeholders = map[rune]byte{}

// translate rewrites a pattern for the view: every non-ASCII character becomes its placeholder.
func translate(pattern string) string {
	var b strings.Builder
	for _, r := range pattern {
		if r < 0x80 {
			b.WriteRune(r)
			continue
		}
		p, ok := placeholders[r]
		if !ok {
			p = byte('A' + len(placeholders))
			if p > 'Y' {
				panic("email: more literal characters than placeholders")
			}
			placeholders[r] = p
		}
		b.WriteByte(p)
	}
	return b.String()
}

// Non-ASCII spaces and digits get their own bytes: a literal ' ' in a pattern is U+0020 alone
// in Python, and `[0-9]` in the device footer's `using [a-z][a-z0-9_.+-]*` is ASCII alone,
// while Python's \s and \d take the others.
const (
	nonASCIISpace = 0x0e
	nonASCIIDigit = 0x0f
)

// wordStandIn is the view's character for a word character that is not a digit and that no
// pattern spells; 'Z' because folded text never holds an uppercase ASCII letter.
const wordStandIn = 'Z'

// viewByte is the view of one character.
func viewByte(r rune) byte {
	f := fold(r)
	if f < 0x80 {
		switch c := byte(f); {
		case c <= 8, c >= 0x0e && c <= 0x1b, c == 0x7f:
			return 1 // not a space, digit or word character; keeps 0x0e and 0x0f free
		default:
			return c
		}
	}
	if p, ok := placeholders[f]; ok {
		return p
	}
	switch {
	case unicodedata.IsSpace(r):
		return nonASCIISpace
	case unicodedata.IsDigit(r):
		return nonASCIIDigit
	case unicodedata.IsPythonWord(r):
		return wordStandIn
	}
	return 1
}

// viewOf is the view of text: one byte per character, so a byte offset in it is a character
// offset in runes.
func viewOf(runes []rune) string {
	out := make([]byte, len(runes))
	for i, r := range runes {
		out[i] = viewByte(r)
	}
	return string(out)
}

// disclaimerViewOf is the view for the disclaimer pattern, which has no \s or \d of its own but
// does use \b: a non-ASCII digit has to read as a word character there. A non-ASCII space stays
// the control byte it is, which is non-word and matches no literal.
func disclaimerViewOf(runes []rune) string {
	out := []byte(viewOf(runes))
	for i, c := range out {
		switch c {
		case nonASCIIDigit:
			out[i] = '0'
		}
	}
	return string(out)
}

func hasDigit(view string) bool {
	return strings.ContainsAny(view, "0123456789\x0f")
}
