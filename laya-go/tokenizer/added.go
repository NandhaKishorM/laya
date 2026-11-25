package tokenizer

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// addedToken is an entry of tokenizer.json's added_tokens: text that is matched before any
// normalization or splitting and mapped straight to an id.
type addedToken struct {
	content    string
	id         int32
	special    bool
	lstrip     bool
	rstrip     bool
	singleWord bool
	normalized bool
}

// span is a stretch of text that is either plain, or exactly one added token.
type span struct {
	text    string
	id      int32
	isToken bool
}

// bucket holds the added tokens that start with one character, longest first, which is the
// leftmost-longest match the reference uses. second records which bytes follow that character
// in any of them, so the common miss (a space before a letter, say, when dozens of tokens are
// runs of spaces) is rejected without trying each one.
type bucket struct {
	tokens []*addedToken
	second [256]bool
	// whole is set when a token is that single character, which has no second byte to test.
	whole bool
}

func (b *bucket) add(t *addedToken, firstWidth int) {
	b.tokens = append(b.tokens, t)
	if len(t.content) == firstWidth {
		b.whole = true
	} else {
		b.second[t.content[firstWidth]] = true
	}
}

// addedVocab finds added tokens in text.
type addedVocab struct {
	ascii [utf8.RuneSelf]bucket
	other map[rune]*bucket
	empty bool
}

func newAddedVocab(tokens []addedToken) (*addedVocab, error) {
	v := &addedVocab{other: map[rune]*bucket{}, empty: len(tokens) == 0}
	for i := range tokens {
		t := &tokens[i]
		if t.content == "" {
			return nil, fmt.Errorf("an added token cannot have empty content")
		}
		r, w := utf8.DecodeRuneInString(t.content)
		if r < utf8.RuneSelf {
			v.ascii[r].add(t, w)
		} else {
			b := v.other[r]
			if b == nil {
				b = &bucket{}
				v.other[r] = b
			}
			b.add(t, w)
		}
	}
	longestFirst := func(b *bucket) {
		sort.Slice(b.tokens, func(i, j int) bool {
			if len(b.tokens[i].content) != len(b.tokens[j].content) {
				return len(b.tokens[i].content) > len(b.tokens[j].content)
			}
			return b.tokens[i].content < b.tokens[j].content
		})
	}
	for i := range v.ascii {
		longestFirst(&v.ascii[i])
	}
	for _, b := range v.other {
		longestFirst(b)
	}
	return v, nil
}

func (v *addedVocab) bucket(r rune) *bucket {
	if r < utf8.RuneSelf {
		return &v.ascii[r]
	}
	return v.other[r]
}

// split cuts text around its added tokens. lstrip and rstrip tokens absorb the white space on
// that side of them.
func (v *addedVocab) split(text string) []span {
	if text == "" {
		return nil
	}
	if v.empty {
		return []span{{text: text}}
	}
	var out []span
	plainFrom := 0
	for at := 0; at < len(text); {
		r, w := utf8.DecodeRuneInString(text[at:])
		hit := v.longestAt(text, at, r, w)
		if hit == nil {
			at += w
			continue
		}
		start, stop := at, at+len(hit.content)
		if hit.lstrip {
			for start > plainFrom {
				prev, pw := utf8.DecodeLastRuneInString(text[:start])
				if !isWhiteSpace(prev) {
					break
				}
				start -= pw
			}
		}
		if hit.rstrip {
			for stop < len(text) {
				next, nw := utf8.DecodeRuneInString(text[stop:])
				if !isWhiteSpace(next) {
					break
				}
				stop += nw
			}
		}
		if start > plainFrom {
			out = append(out, span{text: text[plainFrom:start]})
		}
		out = append(out, span{text: hit.content, id: hit.id, isToken: true})
		plainFrom, at = stop, stop
	}
	if plainFrom < len(text) {
		out = append(out, span{text: text[plainFrom:]})
	}
	return out
}

func (v *addedVocab) longestAt(text string, at int, first rune, width int) *addedToken {
	b := v.bucket(first)
	if b == nil || len(b.tokens) == 0 {
		return nil
	}
	if !b.whole {
		if next := at + width; next >= len(text) || !b.second[text[next]] {
			return nil
		}
	}
	for _, t := range b.tokens {
		if !strings.HasPrefix(text[at:], t.content) {
			continue
		}
		if t.singleWord && !standsAlone(text, at, at+len(t.content)) {
			continue
		}
		return t
	}
	return nil
}

// standsAlone reports that no word character touches either side of text[start:stop].
func standsAlone(text string, start, stop int) bool {
	if start > 0 {
		if r, _ := utf8.DecodeLastRuneInString(text[:start]); isWordChar(r) {
			return false
		}
	}
	if stop < len(text) {
		if r, _ := utf8.DecodeRuneInString(text[stop:]); isWordChar(r) {
			return false
		}
	}
	return true
}
