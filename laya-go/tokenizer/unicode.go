package tokenizer

import "unicode/utf8"

// isWhiteSpace is the Unicode White_Space property, which is what the reference's \s and
// char::is_whitespace mean. It has been stable for decades, so it is spelled out rather than
// taken from a Unicode release.
func isWhiteSpace(r rune) bool {
	switch r {
	case '\t', '\n', 0x0B, '\f', '\r', ' ', 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
		return true
	}
	return r >= 0x2000 && r <= 0x200A
}

// inRanges reports whether r lies in one of the inclusive [lo, hi] pairs of a sorted table.
func inRanges(pairs []rune, r rune) bool {
	lo, hi := 0, len(pairs)/2-1
	for lo <= hi {
		mid := int(uint(lo+hi) >> 1)
		switch {
		case r < pairs[mid*2]:
			hi = mid - 1
		case r > pairs[mid*2+1]:
			lo = mid + 1
		default:
			return true
		}
	}
	return false
}

func isLetter(r rune) bool {
	if r < 0x80 {
		return 'a' <= r|0x20 && r|0x20 <= 'z'
	}
	return inRanges(letter[:], r)
}

func isNumber(r rune) bool {
	if r < 0x80 {
		return '0' <= r && r <= '9'
	}
	return inRanges(number[:], r)
}

func isWordChar(r rune) bool {
	return inRanges(wordChar[:], r)
}

func lastRune(s string) (rune, int) { return utf8.DecodeLastRuneInString(s) }
