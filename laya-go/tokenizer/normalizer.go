package tokenizer

import (
	"fmt"
	"regexp"
	"strings"
)

// normalizer rewrites the text before it is split and tokenized.
type normalizer func(string) string

func identity(s string) string { return s }

// parseNormalizer builds the normalizer a tokenizer.json section describes; nil means none.
func parseNormalizer(n node) (normalizer, error) {
	if n == nil {
		return identity, nil
	}
	kind, err := n.str("type", "")
	if err != nil {
		return nil, err
	}
	if kind == "" {
		return nil, fmt.Errorf("a normalizer must declare a type")
	}
	switch kind {
	case "NFC":
		return func(s string) string { return normalize(s, formNFC) }, nil
	case "NFD":
		return func(s string) string { return normalize(s, formNFD) }, nil
	case "NFKC":
		return func(s string) string { return normalize(s, formNFKC) }, nil
	case "NFKD":
		return func(s string) string { return normalize(s, formNFKD) }, nil
	case "Replace":
		return parseReplace(n)
	case "Lowercase":
		return lowercase, nil
	case "Prepend":
		prefix, err := n.str("prepend", "")
		if err != nil {
			return nil, err
		}
		return func(s string) string {
			if s == "" { // the reference prepends to non-empty text only
				return s
			}
			return prefix + s
		}, nil
	case "Strip":
		left, err := n.boolean("strip_left", true)
		if err != nil {
			return nil, err
		}
		right, err := n.boolean("strip_right", true)
		if err != nil {
			return nil, err
		}
		return func(s string) string { return strip(s, left, right) }, nil
	case "Sequence":
		children, err := n.list("normalizers")
		if err != nil {
			return nil, err
		}
		steps := make([]normalizer, 0, len(children))
		for _, child := range children {
			cn, err := asNode(child, "a normalizer in a Sequence")
			if err != nil {
				return nil, err
			}
			step, err := parseNormalizer(cn)
			if err != nil {
				return nil, err
			}
			steps = append(steps, step)
		}
		return func(s string) string {
			for _, step := range steps {
				s = step(s)
			}
			return s
		}, nil
	}
	return nil, unsupported("the normalizer %q is not implemented; refusing to load rather than "+
		"produce ids that are wrong in a way no answer-level test would localise", kind)
}

func parseReplace(n node) (normalizer, error) {
	pattern, err := n.object("pattern")
	if err != nil {
		return nil, err
	}
	content, ok := n["content"].(string)
	if !ok {
		return nil, fmt.Errorf("a Replace normalizer must declare content")
	}
	if literal, ok := pattern["String"].(string); ok {
		if literal == "" {
			return nil, fmt.Errorf("a Replace normalizer cannot match the empty string")
		}
		return func(s string) string { return strings.ReplaceAll(s, literal, content) }, nil
	}
	expr, ok := pattern["Regex"].(string)
	if !ok {
		return nil, fmt.Errorf(`a Replace normalizer's pattern must be {"String": ...} or {"Regex": ...}`)
	}
	if class, bad := dialectDiffers(expr); bad {
		return nil, unsupported(`the Replace regex %q uses \%c, whose meaning differs between the `+
			`reference's regex engine (Unicode-aware) and Go's RE2 (ASCII-only for \s \d \w \b), so `+
			"it would normalize differently; refusing to load", expr, class)
	}
	re, err := regexp.Compile(expr)
	if err != nil {
		return nil, fmt.Errorf("a Replace normalizer's regex: %w", err)
	}
	return func(s string) string { return re.ReplaceAllLiteralString(s, content) }, nil
}

// dialectDiffers finds an escape whose meaning is not the same in the reference's regex engine
// and in Go's: the Perl classes and word boundaries, which the reference reads as Unicode and
// Go as ASCII, and the \p properties, which follow each side's own Unicode release.
func dialectDiffers(expr string) (byte, bool) {
	for i := 0; i+1 < len(expr); i++ {
		if expr[i] != '\\' {
			continue
		}
		i++
		if strings.IndexByte("sSdDwWbBpP", expr[i]) >= 0 {
			return expr[i], true
		}
	}
	return 0, false
}

// lowercase lowercases character by character, as the reference does: unlike str.lower() it has
// no context rule for a final sigma.
func lowercase(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		b.WriteString(lowerOf(r))
	}
	return b.String()
}

func lowerOf(r rune) string {
	if r < 0x80 {
		if 'A' <= r && r <= 'Z' {
			r += 32
		}
		return string(r)
	}
	if multi, ok := lowerMulti[r]; ok {
		return string(multi)
	}
	lo, hi := 0, len(lowerFrom)-1
	for lo <= hi {
		mid := int(uint(lo+hi) >> 1)
		switch {
		case r < lowerFrom[mid]:
			hi = mid - 1
		case r > lowerFrom[mid]:
			lo = mid + 1
		default:
			return string(lowerTo[mid])
		}
	}
	return string(r)
}

func strip(s string, left, right bool) string {
	if left {
		s = strings.TrimLeftFunc(s, isWhiteSpace)
	}
	if right {
		s = strings.TrimRightFunc(s, isWhiteSpace)
	}
	return s
}
