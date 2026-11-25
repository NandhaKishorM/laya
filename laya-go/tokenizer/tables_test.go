package tokenizer

import "testing"

func TestWhiteSpaceIsTheUnicodeSet(t *testing.T) {
	n := 0
	for r := rune(0); r < 0x110000; r++ {
		if !isWhiteSpace(r) {
			continue
		}
		n++
		if isLetter(r) || isNumber(r) {
			t.Errorf("U+%04X is white space and must be neither a letter nor a number", r)
		}
	}
	if n != 25 {
		t.Errorf("the Unicode White_Space set is 25 code points, got %d", n)
	}
	for _, r := range []rune{0x1C, 0x1D, 0x1E, 0x1F, 0x180E, 0x200B, 0x200C, 0x200D, 0xFEFF} {
		if isWhiteSpace(r) {
			t.Errorf("U+%04X is not White_Space (it is a control or a format character)", r)
		}
	}
}

func TestLetterAndNumberClassesAreDisjoint(t *testing.T) {
	for r := rune(0); r < 0x110000; r++ {
		if isLetter(r) && isNumber(r) {
			t.Fatalf("U+%04X is in both classes", r)
		}
	}
	for r := '0'; r <= '9'; r++ {
		if !isNumber(r) || isLetter(r) {
			t.Errorf("%q must be a number only", r)
		}
	}
	for _, r := range "azAZ" {
		if !isLetter(r) || isNumber(r) {
			t.Errorf("%q must be a letter only", r)
		}
	}
	for _, r := range "@[`{" {
		if isLetter(r) {
			t.Errorf("%q sits next to the letter ranges but is not one", r)
		}
	}
}

// The classes are the reference's, which is newer than any one Go release's: these blocks were
// added in Unicode 15.1 and 16.0.
func TestClassesCarryTheReferencesUnicode(t *testing.T) {
	for _, block := range [][2]rune{
		{0x2EBF0, 0x2EE5D}, // CJK Unified Ideographs Extension I
		{0x105C0, 0x105F3}, // Todhri
		{0x10D4A, 0x10D65}, // Garay
		{0x113A0, 0x113B2}, // Tulu-Tigalari
		{0x16100, 0x1611D}, // Gurung Khema
	} {
		for r := block[0]; r <= block[1]; r++ {
			if !isLetter(r) {
				t.Errorf("U+%04X is a letter to the reference", r)
			}
		}
	}
}

func TestWordCharFollowsTheCrateNotAlphanumeric(t *testing.T) {
	// A combining mark and a joiner are word characters to the reference; the other-number
	// category and a space are not.
	for r, want := range map[rune]bool{'a': true, '_': true, '5': true, 0x300: true, 0x200D: true,
		0xB2: false, 0xBD: false, ' ': false, '-': false} {
		if got := isWordChar(r); got != want {
			t.Errorf("isWordChar(U+%04X) = %v, want %v", r, got, want)
		}
	}
}
