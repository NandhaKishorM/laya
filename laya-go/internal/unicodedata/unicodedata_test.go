package unicodedata

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
)

const maxRune = 0x110000

func digests(t *testing.T) fixtures.Object {
	t.Helper()
	return fixtures.Obj(t, fixtures.Load(t, "lang_detect.json").Must("unicode_digests"))
}

func predicateDigest(pred func(rune) bool) string {
	h := sha256.New()
	one, zero := []byte{'1'}, []byte{'0'}
	for r := rune(0); r < maxRune; r++ {
		if pred(r) {
			h.Write(one)
		} else {
			h.Write(zero)
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func TestVersion(t *testing.T) {
	want := fixtures.Str(t, fixtures.Load(t, "lang_detect.json").Must("unicode_version"))
	if Version != want {
		t.Errorf("Version = %q, goldens recorded at %q", Version, want)
	}
}

// Each predicate is compared with CPython over every code point, surrogates included: Python
// reports all of them false, and so must a port. A digest, not a table, so the test still proves
// the whole of Unicode after a table edit or a CPython bump.
func TestPredicatesMatchCPythonEverywhere(t *testing.T) {
	d := digests(t)
	for name, pred := range map[string]func(rune) bool{
		"alpha":     IsAlpha,
		"word":      IsWord,
		"combining": IsCombining,
		"digit":     IsDigit,
		"upper":     IsUpper,
		"lower":     IsLower,
		"space":     IsSpace,
		"mark":      IsMark,
		"initial":   IsInitial,
	} {
		t.Run(name, func(t *testing.T) {
			want := fixtures.Str(t, d.Must(name))
			if got := predicateDigest(pred); got != want {
				t.Errorf("Is%s no longer agrees with CPython over all of Unicode; regenerate "+
					"with scripts/gen_tables.py and read the diff", name)
			}
		})
	}
}

func TestLowerMatchesCPythonEverywhere(t *testing.T) {
	want := fixtures.Str(t, digests(t).Must("python_lower"))
	h := sha256.New()
	for r := rune(0); r < maxRune; r++ {
		if r >= 0xD800 && r <= 0xDFFF {
			continue
		}
		var parts []string
		for _, c := range Lower(string(r)) {
			parts = append(parts, strconv.Itoa(int(c)))
		}
		h.Write([]byte(strconv.Itoa(int(r)) + ":" + strings.Join(parts, ",") + ";"))
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		t.Error("Lower disagrees with str.lower() somewhere in Unicode")
	}
}

func TestPrintableHasASCIIAnchor(t *testing.T) {
	// Not in the lang goldens (repr uses it); pinned here by the shape Python guarantees.
	for r := rune(0x20); r < 0x7F; r++ {
		if !IsPrintable(r) {
			t.Errorf("U+%04X is printable ASCII", r)
		}
	}
	for _, r := range []rune{0, 0x7F, 0xA0, 0xAD, 0x200B, 0xFEFF, 0xE000, 0x378} {
		if IsPrintable(r) {
			t.Errorf("U+%04X is not printable in Python", r)
		}
	}
}

func TestStripAndFields(t *testing.T) {
	if got := Strip("  \t x y \n"); got != "x y" {
		t.Errorf("Strip = %q", got)
	}
	if got := Fields("  a b \n c　"); strings.Join(got, "|") != "a|b|c" {
		t.Errorf("Fields = %q", got)
	}
	if !IsBlank(" \t") || IsBlank(" x ") || !IsBlank("") {
		t.Error("IsBlank")
	}
}

// str.lower() is context-dependent for a capital sigma: final at the end of a word, medial
// elsewhere, skipping case-ignorable characters on both sides. The single-character digest above
// cannot see that, so the rule is held to CPython on strings built around it.
func TestLowerFinalSigmaMatchesCPython(t *testing.T) {
	cases := fixtures.List(t, fixtures.Field(t, fixtures.Load(t, "go_extra.json").Must("python_lower"), "cases"))
	if len(cases) < 1000 {
		t.Fatalf("only %d cases", len(cases))
	}
	finals := 0
	for _, c := range cases {
		text, want := fixtures.Str(t, fixtures.Field(t, c, "text")), fixtures.Str(t, fixtures.Field(t, c, "lower"))
		if got := Lower(text); got != want {
			t.Errorf("Lower(%+q) = %+q, want %+q", text, got, want)
		}
		if strings.Contains(want, "ς") {
			finals++
		}
	}
	if finals < 100 {
		t.Errorf("only %d cases lowercase a sigma to its final form, so the rule is barely exercised", finals)
	}
}

// Invalid UTF-8 becomes U+FFFD one per maximal invalid subpart, exactly as CPython's
// bytes.decode('utf-8', 'replace') does; strings.ToValidUTF8 collapses runs and is not this.
func TestToValidMatchesCPython(t *testing.T) {
	fam := fixtures.Obj(t, fixtures.Load(t, "go_extra.json").Must("invalid_utf8"))
	hexes := fixtures.Strings(t, fam.Must("hex"))
	decoded := fixtures.Strings(t, fam.Must("decoded"))
	changed := 0
	for i, h := range hexes {
		raw, err := hex.DecodeString(h)
		if err != nil {
			t.Fatal(err)
		}
		if got := ToValid(string(raw)); got != decoded[i] {
			t.Fatalf("ToValid(% x) = %+q, want %+q", raw, got, decoded[i])
		}
		if got := Lossy(raw); got != decoded[i] {
			t.Fatalf("Lossy(% x) = %+q, want %+q", raw, got, decoded[i])
		}
		if decoded[i] != string(raw) {
			changed++
		}
	}
	if changed < 1000 {
		t.Errorf("only %d of %d cases were invalid, so the replacement rule is barely exercised", changed, len(hexes))
	}
	if ToValid("\xff\xfe") != "��" || ToValid("\xe2\x82") != "�" {
		t.Error("\\xff\\xfe is two replacements and \\xe2\\x82 is one")
	}
}
