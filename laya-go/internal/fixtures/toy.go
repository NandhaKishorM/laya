package fixtures

import "encoding/json"

// Toy special-token ids of the tokenizer ToyTokenizer returns.
const (
	ToyPad  = 256
	ToyCLS  = 257
	ToySEP  = 258
	ToyMask = 259
)

// ToyTokenizer returns the tokenizer.json and tokenizer_config.json of a tiny byte-level BPE
// tokenizer: one id per byte (0-255), [PAD] [CLS] [SEP] [MASK] at 256-259, and the merges
// "t h" (260) and "i n" (261). Examples and unit tests use it where no checkpoint is at hand.
func ToyTokenizer() (tokenizerJSON, configJSON []byte) {
	// GPT-2's bytes_to_unicode: printable bytes stand for themselves, the rest move above U+0100.
	var printable []int
	for b := '!'; b <= '~'; b++ {
		printable = append(printable, int(b))
	}
	for b := 0xA1; b <= 0xFF; b++ {
		if b != 0xAD {
			printable = append(printable, b)
		}
	}
	runes := make(map[int]rune, 256)
	for _, b := range printable {
		runes[b] = rune(b)
	}
	next := rune(256)
	for b := 0; b < 256; b++ {
		if _, ok := runes[b]; !ok {
			runes[b] = next
			next++
		}
	}
	vocab := map[string]int{"[PAD]": ToyPad, "[CLS]": ToyCLS, "[SEP]": ToySEP, "[MASK]": ToyMask, "th": 260, "in": 261}
	for b := 0; b < 256; b++ {
		vocab[string(runes[b])] = b
	}
	type addedToken struct {
		ID         int    `json:"id"`
		Content    string `json:"content"`
		SingleWord bool   `json:"single_word"`
		Lstrip     bool   `json:"lstrip"`
		Rstrip     bool   `json:"rstrip"`
		Normalized bool   `json:"normalized"`
		Special    bool   `json:"special"`
	}
	var added []addedToken
	for i, name := range []string{"[PAD]", "[CLS]", "[SEP]", "[MASK]"} {
		added = append(added, addedToken{ID: ToyPad + i, Content: name, Special: true})
	}
	// Marshaled as a whole, so no token text is ever spliced into JSON by hand.
	spec, err := json.Marshal(map[string]any{
		"added_tokens":  added,
		"normalizer":    nil,
		"pre_tokenizer": map[string]any{"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
		"decoder":       map[string]any{"type": "ByteLevel"},
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{"t h", "i n"}},
	})
	if err != nil {
		panic(err) // unreachable: every value above marshals
	}
	config := `{"cls_token":"[CLS]","sep_token":"[SEP]","mask_token":"[MASK]","pad_token":"[PAD]"}`
	return spec, []byte(config)
}
