package email

import (
	"math"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// DefaultMaxChars is the length CleanBody cuts a body to unless told otherwise.
const DefaultMaxChars = 3000

// cutRunes is Python's text[:limit] for a limit that may be negative, which cuts from the end.
func cutRunes(text []rune, limit int64) []rune {
	n := int64(len(text))
	end := n + limit
	if limit >= 0 {
		end = min(limit, n)
	}
	if end <= 0 {
		return nil
	}
	return text[:end]
}

// times4 is max_chars * 4, saturating where Python's integers would not.
func times4(n int) int64 {
	const limit = math.MaxInt64 / 4
	switch {
	case int64(n) > limit:
		return math.MaxInt64
	case int64(n) < -limit:
		return math.MinInt64
	}
	return int64(n) * 4
}

// CleanBody removes quoted email history, signatures, device footers and disclaimers from a
// body, so the model reads the new message and not the thread under it.
//
// maxChars is a slice budget, not a floor, and it is applied twice: as a maxChars*4 pre-cut that
// bounds the work of the rules, and as the cut of the result. The rules see only the first
// 4*maxChars characters, so a footer that begins past that is never recognized. 0 gives the empty
// string, and a negative value cuts from the end: 5*|maxChars| characters come off the body, and
// nothing is left once that reaches its length. That is the reference's behavior, recorded in
// the shared goldens, so "no limit" means a budget larger than the body, never 0 or a negative.
func CleanBody(body string, maxChars int) string {
	body = strings.ReplaceAll(body, "\r\n", "\n")
	body = strings.ReplaceAll(body, "\r", "\n")
	body = strings.ReplaceAll(body, `\n`, "\n")
	text := []rune(body)
	if limit := times4(maxChars); int64(len(text)) > limit {
		text = cutRunes(text, limit)
	}

	var lines [][]rune
	var views []string
	src := splitRunes(text, '\n')
	srcViews := make([]string, len(src))
	for i := range src {
		srcViews[i] = viewOf(src[i])
	}
	for i, line := range src {
		view := srcViews[i]
		if len(lines) > 0 && matchesQuoteHeader(view) {
			break
		}
		if len(lines) > 0 && headerFromName.MatchString(view) && i+1 < len(src) &&
			headerNext.MatchString(srcViews[i+1]) {
			break
		}
		if len(lines) > 0 && endsWithColon(view) && attributionTail.MatchString(view) {
			if last := views[len(views)-1]; attributionHead.MatchString(last) && hasDigit(last) {
				lines, views = lines[:len(lines)-1], views[:len(views)-1]
			}
			break
		}
		if first := firstNonSpace(view); first < len(view) && view[first] == '>' {
			continue
		}
		kept := trimRight(line)
		lines = append(lines, kept)
		views = append(views, view[:len(kept)])
	}

	cut := len(lines)
	start := max(1, min(int(float64(len(lines))*0.6), len(lines)-8))
	for i := start; i < len(lines); i++ {
		width := widthStripped(lines[i])
		if (width <= 40 && isSignature(lines[i], views[i])) ||
			(width <= 60 && deviceFooter.MatchString(views[i])) {
			cut = i
			break
		}
	}
	lines = lines[:cut]

	var joined []rune
	for i, line := range lines {
		if i > 0 {
			joined = append(joined, '\n')
		}
		joined = append(joined, line...)
	}
	var out []rune
	for _, paragraph := range splitOn(joined, blankLine.FindAllStringIndex(viewOf(joined), -1)) {
		cleaned := trimSpace(stripDisclaimer(paragraph))
		if len(cleaned) == 0 {
			continue
		}
		if len(out) > 0 {
			out = append(out, '\n', '\n')
		}
		out = append(out, cleaned...)
	}
	out = collapseSpaces(out)
	return string(cutRunes(out, int64(maxChars)))
}

func matchesQuoteHeader(view string) bool {
	for _, h := range quoteHeaders {
		if h.matches(view) {
			return true
		}
	}
	return false
}

// endsWithColon is a necessary condition for attributionTail, which ends in `:` and spaces. It
// keeps a very long line from being scanned by the pattern's `.{0,120}` for nothing.
func endsWithColon(view string) bool {
	return strings.HasSuffix(strings.TrimRight(view, " \t\n\v\f\r\x1c\x1d\x1e\x1f\x0e"), ":")
}

func firstNonSpace(view string) int {
	for i := 0; i < len(view); i++ {
		if view[i] != ' ' && view[i] != nonASCIISpace && !(view[i] >= '\t' && view[i] <= '\r') && !(view[i] >= 0x1c && view[i] <= 0x1f) {
			return i
		}
	}
	return len(view)
}

func isSignature(line []rune, view string) bool {
	return signoffDashes.MatchString(view) || isEnglishSignoff(line, view) || signoffRomance.MatchString(view)
}

// splitRunes is Python's str.split(sep) for a one-character separator.
func splitRunes(text []rune, sep rune) [][]rune {
	var out [][]rune
	start := 0
	for i, r := range text {
		if r == sep {
			out = append(out, text[start:i])
			start = i + 1
		}
	}
	return append(out, text[start:])
}

// splitOn cuts text around the [start, end) matches, as re.split does.
func splitOn(text []rune, matches [][]int) [][]rune {
	out := make([][]rune, 0, len(matches)+1)
	prev := 0
	for _, m := range matches {
		out = append(out, text[prev:m[0]])
		prev = m[1]
	}
	return append(out, text[prev:])
}

func trimRight(s []rune) []rune {
	for len(s) > 0 && unicodedata.IsSpace(s[len(s)-1]) {
		s = s[:len(s)-1]
	}
	return s
}

func trimSpace(s []rune) []rune {
	for len(s) > 0 && unicodedata.IsSpace(s[0]) {
		s = s[1:]
	}
	return trimRight(s)
}

func widthStripped(s []rune) int { return len(trimSpace(s)) }

// collapseSpaces is re.sub(r"[ \t]+", " ", text).
func collapseSpaces(s []rune) []rune {
	out := make([]rune, 0, len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' || s[i] == '\t' {
			for i+1 < len(s) && (s[i+1] == ' ' || s[i+1] == '\t') {
				i++
			}
			out = append(out, ' ')
			continue
		}
		out = append(out, s[i])
	}
	return out
}

// isDisclaimer is _DISCLAIMER.search.
func isDisclaimer(text []rune) bool { return disclaimer.MatchString(disclaimerViewOf(text)) }

// startsNewSentence is true when the first letter is uppercase. Uncased scripts never start a
// new piece, so wrapped boilerplate in them still drops whole.
func startsNewSentence(line []rune) bool {
	for _, r := range line {
		if unicodedata.IsAlpha(r) {
			return unicodedata.IsUpper(r)
		}
	}
	return false
}

// splitFusedLines splits a boilerplate sentence at sentence-starting newlines: an unpunctuated
// request line glued to a disclaimer line splits, a lowercase continuation does not.
func splitFusedLines(sentence []rune) [][]rune {
	hasNewline := false
	for _, r := range sentence {
		if r == '\n' {
			hasNewline = true
			break
		}
	}
	if !hasNewline {
		return [][]rune{sentence}
	}
	var pieces [][]rune
	var buf []rune
	for _, raw := range splitRunes(sentence, '\n') {
		line := trimSpace(raw)
		if len(line) == 0 {
			continue
		}
		if len(buf) > 0 && startsNewSentence(line) {
			pieces = append(pieces, buf)
			buf = append([]rune(nil), line...)
		} else {
			if len(buf) > 0 {
				buf = append(buf, ' ')
			}
			buf = append(buf, line...)
		}
	}
	if len(buf) > 0 {
		pieces = append(pieces, buf)
	}
	return pieces
}

// splitSentences is re.split(r"(?<=[.!?])\s+", text): a run of spaces right after . ! or ?.
func splitSentences(text []rune) [][]rune {
	var out [][]rune
	start := 0
	for i := 1; i < len(text); i++ {
		if !unicodedata.IsSpace(text[i]) || !isStop(text[i-1]) {
			continue
		}
		end := i
		for end < len(text) && unicodedata.IsSpace(text[end]) {
			end++
		}
		out = append(out, text[start:i])
		start = end
		i = end - 1
	}
	return append(out, text[start:])
}

func isStop(r rune) bool { return r == '.' || r == '!' || r == '?' }

// stripDisclaimer drops boilerplate from one paragraph. A paragraph goes whole only when every
// sentence in it is boilerplate; otherwise only the boilerplate sentences go, since a footer that
// runs on without a blank line must not take the sender's request with it.
func stripDisclaimer(paragraph []rune) []rune {
	if !isDisclaimer(paragraph) {
		return paragraph
	}
	var pieces [][]rune
	for _, part := range splitSentences(paragraph) {
		part = trimSpace(part)
		if len(part) == 0 {
			continue
		}
		if isDisclaimer(part) {
			pieces = append(pieces, splitFusedLines(part)...)
		} else {
			pieces = append(pieces, part)
		}
	}
	var out []rune
	for _, piece := range pieces {
		if isDisclaimer(piece) {
			continue
		}
		if len(out) > 0 {
			out = append(out, ' ')
		}
		out = append(out, piece...)
	}
	return out
}

// dropMarks removes the combining marks that ride on a preceding character. A mark that opens
// the string or follows a space has no base and stays, so it still separates tokens.
func dropMarks(text []rune) []rune {
	kept := make([]rune, 0, len(text))
	for _, r := range text {
		if unicodedata.IsMark(r) && len(kept) > 0 && !unicodedata.IsSpace(kept[len(kept)-1]) {
			continue
		}
		kept = append(kept, r)
	}
	return kept
}

// tailIsNameOnly is _SIGNOFF_TAIL plus the per-token case test, as one scan:
// ^[\s,;:!.]*(?:[^\W\d_][\w'-]*[\s,.]*){0,3}$, with every token opening on a letter that is not
// lowercase (Lu, Lt or Lo).
func tailIsNameOnly(tail []rune) bool {
	i := skip(tail, 0, ",;:!.")
	for token := 0; token < 3 && i < len(tail); token++ {
		if !unicodedata.IsWord(tail[i]) {
			break
		}
		if !unicodedata.IsInitial(tail[i]) {
			return false
		}
		i++
		for i < len(tail) && (unicodedata.IsPythonWord(tail[i]) || tail[i] == '\'' || tail[i] == '-') {
			i++
		}
		i = skip(tail, i, ",.")
	}
	return i == len(tail)
}

// skip advances over Python whitespace and any of extra.
func skip(text []rune, from int, extra string) int {
	i := from
	for i < len(text) && (unicodedata.IsSpace(text[i]) || strings.ContainsRune(extra, text[i])) {
		i++
	}
	return i
}

// isEnglishSignoff is true when a closing word is followed by nothing but punctuation and a
// short name.
func isEnglishSignoff(line []rune, view string) bool {
	loc := signoffHead.FindStringIndex(view)
	if loc == nil {
		return false
	}
	return tailIsNameOnly(dropMarks(line[loc[1]:]))
}
