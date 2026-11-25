package tokenizer

import "sort"

// Unicode normalization (UAX #15), driven by tables recorded from the reference crate.

const (
	hangulSBase  = 0xAC00
	hangulLBase  = 0x1100
	hangulVBase  = 0x1161
	hangulTBase  = 0x11A7
	hangulLCount = 19
	hangulVCount = 21
	hangulTCount = 28
	hangulNCount = hangulVCount * hangulTCount
	hangulSCount = hangulLCount * hangulNCount
)

type normForm struct{ compat, compose bool }

var (
	formNFC  = normForm{compat: false, compose: true}
	formNFD  = normForm{compat: false, compose: false}
	formNFKC = normForm{compat: true, compose: true}
	formNFKD = normForm{compat: true, compose: false}
)

// normalize returns s in the given form.
func normalize(s string, f normForm) string {
	ascii, below := true, true
	for _, r := range s {
		if r >= 0x80 {
			ascii = false
		}
		if r >= 0x300 {
			below = false
			break
		}
	}
	// ASCII has no decomposition in any form; and no character below U+0300 can combine with
	// anything before it, so a string of them is already NFC.
	if ascii || (below && f.compose && !f.compat) {
		return s
	}
	runes := make([]rune, 0, len(s))
	for _, r := range s {
		runes = decompose(runes, r, f.compat)
	}
	reorder(runes)
	if f.compose {
		runes = composeRunes(runes)
	}
	return string(runes)
}

// lookupPool finds r in a sorted key table and returns its slice of the pool.
func lookupPool(keys []rune, offsets []uint32, pool []rune, r rune) ([]rune, bool) {
	i := sort.Search(len(keys), func(i int) bool { return keys[i] >= r })
	if i == len(keys) || keys[i] != r {
		return nil, false
	}
	return pool[offsets[i]:offsets[i+1]], true
}

func decompose(dst []rune, r rune, compat bool) []rune {
	if r < 0xA0 || (r < 0xC0 && !compat) { // nothing below U+00C0 decomposes canonically
		return append(dst, r)
	}
	if r >= hangulSBase && r < hangulSBase+hangulSCount {
		s := r - hangulSBase
		dst = append(dst, hangulLBase+s/hangulNCount, hangulVBase+(s%hangulNCount)/hangulTCount)
		if t := s % hangulTCount; t != 0 {
			dst = append(dst, hangulTBase+t)
		}
		return dst
	}
	if compat {
		if d, ok := lookupPool(compatKeys[:], compatOffsets[:], compatPool[:], r); ok {
			return append(dst, d...)
		}
	}
	if d, ok := lookupPool(canonKeys[:], canonOffsets[:], canonPool[:], r); ok {
		return append(dst, d...)
	}
	return append(dst, r)
}

// combiningClass is the canonical combining class of r.
func combiningClass(r rune) uint8 {
	if r < 0x300 {
		return 0
	}
	n := len(cccRanges) / 3
	i := sort.Search(n, func(i int) bool { return cccRanges[i*3+1] >= r })
	if i < n && cccRanges[i*3] <= r {
		return uint8(cccRanges[i*3+2])
	}
	return 0
}

// reorder is the canonical ordering algorithm: a stable sort of each run of marks by class.
func reorder(runes []rune) {
	for i := 0; i < len(runes); {
		if combiningClass(runes[i]) == 0 {
			i++
			continue
		}
		j := i
		for j < len(runes) && combiningClass(runes[j]) != 0 {
			j++
		}
		run := runes[i:j]
		sort.SliceStable(run, func(a, b int) bool { return combiningClass(run[a]) < combiningClass(run[b]) })
		i = j
	}
}

func composePair(first, second rune) (rune, bool) {
	if first >= hangulLBase && first < hangulLBase+hangulLCount &&
		second >= hangulVBase && second < hangulVBase+hangulVCount {
		return hangulSBase + ((first-hangulLBase)*hangulVCount+(second-hangulVBase))*hangulTCount, true
	}
	if first >= hangulSBase && first < hangulSBase+hangulSCount && (first-hangulSBase)%hangulTCount == 0 &&
		second > hangulTBase && second < hangulTBase+hangulTCount {
		return first + (second - hangulTBase), true
	}
	key := uint64(first)<<32 | uint64(second)
	i := sort.Search(len(composeKeys), func(i int) bool { return composeKeys[i] >= key })
	if i < len(composeKeys) && composeKeys[i] == key {
		return composeValues[i], true
	}
	return 0, false
}

// composeRunes composes in place: each character joins the last starter unless a character of
// the same or higher class sits between them.
func composeRunes(runes []rune) []rune {
	out := runes[:0]
	starter := -1
	lastClass := uint8(0)
	for _, r := range runes {
		cc := combiningClass(r)
		if starter >= 0 && (lastClass < cc || lastClass == 0) {
			if c, ok := composePair(out[starter], r); ok {
				out[starter] = c
				continue
			}
		}
		if cc == 0 {
			starter, lastClass = len(out), 0
		} else {
			lastClass = cc
		}
		out = append(out, r)
	}
	return out
}
