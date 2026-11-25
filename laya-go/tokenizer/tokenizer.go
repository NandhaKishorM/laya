package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"path"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// roleKeys are the special-token roles tokenizer_config.json can name, in the order they are read.
var roleKeys = [...]string{"cls_token", "sep_token", "mask_token", "pad_token", "unk_token"}

// Tokenizer encodes text to token ids and back. It is immutable once loaded and safe for
// concurrent use by multiple goroutines.
type Tokenizer struct {
	normalizer    normalizer
	preTokenizer  preTokenizer
	decoder       decoder
	rawAdded      *addedVocab
	normalAdded   *addedVocab
	bpe           *bpe
	ids           map[string]int32
	tokens        []string
	specials      map[string]int32
	isSpecial     map[int32]struct{}
	roles         map[string]int32
	maskToken     string
	byteLevelOnly bool
}

// rawMerge is one entry of model.merges, which is either "a b" or ["a", "b"].
type rawMerge [2]string

func (m *rawMerge) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && data[0] == '"' {
		var joined string
		if err := json.Unmarshal(data, &joined); err != nil {
			return err
		}
		for i := 1; i+1 < len(joined); i++ {
			if joined[i] == ' ' {
				*m = rawMerge{joined[:i], joined[i+1:]}
				return nil
			}
		}
		return fmt.Errorf("a space-joined merge must have two non-empty halves, got %q", joined)
	}
	var pair []string
	if err := json.Unmarshal(data, &pair); err != nil || len(pair) != 2 {
		return fmt.Errorf("a merge must be two strings or one space-joined string")
	}
	*m = rawMerge{pair[0], pair[1]}
	return nil
}

type rawAdded struct {
	Content    string `json:"content"`
	ID         *int32 `json:"id"`
	Special    bool   `json:"special"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	SingleWord bool   `json:"single_word"`
	Normalized bool   `json:"normalized"`
}

type rawDocument struct {
	Model struct {
		Type         string           `json:"type"`
		Vocab        map[string]int32 `json:"vocab"`
		Merges       []rawMerge       `json:"merges"`
		UnkToken     *string          `json:"unk_token"`
		FuseUnk      *bool            `json:"fuse_unk"`
		IgnoreMerges *bool            `json:"ignore_merges"`
		ByteFallback *bool            `json:"byte_fallback"`
	} `json:"model"`
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Decoder      json.RawMessage `json:"decoder"`
	AddedTokens  []rawAdded      `json:"added_tokens"`
}

func flag(p *bool) bool { return p != nil && *p }

// LoadFS reads the tokenizer of a model directory in fsys: <dir>/tokenizer/tokenizer.json when
// that exists, else <dir>/tokenizer.json, with the tokenizer_config.json beside whichever is used
// (it names the CLS, SEP, MASK and PAD tokens, and may be absent). Use os.DirFS for a directory.
func LoadFS(fsys fs.FS, dir string) (*Tokenizer, error) {
	where := dir
	if info, err := fs.Stat(fsys, path.Join(dir, "tokenizer", "tokenizer.json")); err == nil && info.Mode().IsRegular() {
		where = path.Join(dir, "tokenizer")
	}
	tokenizerJSON, err := fs.ReadFile(fsys, path.Join(where, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	configJSON, err := fs.ReadFile(fsys, path.Join(where, "tokenizer_config.json"))
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}
	return Parse(tokenizerJSON, configJSON)
}

// Parse builds a Tokenizer from the bytes of a tokenizer.json and, optionally, of its
// tokenizer_config.json (nil for none).
func Parse(tokenizerJSON, configJSON []byte) (*Tokenizer, error) {
	var config map[string]any
	if configJSON != nil {
		if err := json.Unmarshal(configJSON, &config); err != nil {
			return nil, fmt.Errorf("tokenizer_config.json must be a JSON object: %w", err)
		}
	}
	return parse(tokenizerJSON, config)
}

func parse(data []byte, config map[string]any) (*Tokenizer, error) {
	var doc rawDocument
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if doc.Model.Type != "BPE" {
		return nil, unsupported("this package implements the BPE tokenizer model; the checkpoint "+
			"declares %q. Refusing to load rather than mis-tokenize silently", doc.Model.Type)
	}
	t := &Tokenizer{}
	var err error
	nodes := [3]node{}
	for i, raw := range [3]json.RawMessage{doc.Normalizer, doc.PreTokenizer, doc.Decoder} {
		what := [3]string{"normalizer", "pre_tokenizer", "decoder"}[i]
		if nodes[i], err = parseNode(raw, what); err != nil {
			return nil, err
		}
	}
	if t.normalizer, err = parseNormalizer(nodes[0]); err != nil {
		return nil, err
	}
	if t.preTokenizer, err = parsePreTokenizer(nodes[1]); err != nil {
		return nil, err
	}
	if t.decoder, err = parseDecoder(nodes[2]); err != nil {
		return nil, err
	}
	_, t.byteLevelOnly = t.preTokenizer.(byteLevel)

	added, err := readAddedTokens(doc.AddedTokens)
	if err != nil {
		return nil, err
	}
	vocab := doc.Model.Vocab
	if vocab == nil {
		return nil, fmt.Errorf("model.vocab must be an object")
	}
	t.isSpecial = map[int32]struct{}{}
	t.specials = map[string]int32{}
	for _, tok := range added {
		if tok.special {
			t.isSpecial[tok.id] = struct{}{}
			t.specials[tok.content] = tok.id
		}
		// The added-token id wins over the vocabulary's, as in the reference.
		vocab[tok.content] = tok.id
	}
	t.ids = vocab

	highest := int32(0)
	for _, id := range vocab {
		if id < 0 {
			return nil, fmt.Errorf("the vocabulary holds a negative id %d", id)
		}
		if id > highest {
			highest = id
		}
	}
	// Ids may have gaps (added tokens sit outside the vocabulary), but a dense table sized by a
	// hostile id would be gigabytes, so an id far beyond the token count is refused.
	if limit := int64(len(vocab))*4 + 1<<16; int64(highest) >= limit {
		return nil, fmt.Errorf("the highest token id %d is far beyond the %d tokens; ids this sparse "+
			"are refused rather than allocated", highest, len(vocab))
	}
	t.tokens = make([]string, highest+1)
	for tok, id := range vocab {
		if owner := t.tokens[id]; owner != "" && owner != tok {
			return nil, fmt.Errorf("the vocabulary gives id %d to both %q and %q", id, owner, tok)
		}
		t.tokens[id] = tok
	}

	t.bpe, err = newBPE(vocab, mergePairs(doc.Model.Merges), doc.Model.UnkToken,
		flag(doc.Model.FuseUnk), flag(doc.Model.IgnoreMerges), flag(doc.Model.ByteFallback))
	if err != nil {
		return nil, err
	}
	if err := t.checkEveryCharacterIsRepresentable(); err != nil {
		return nil, err
	}

	var onRaw, onNormalized []addedToken
	for _, tok := range added {
		if tok.normalized {
			onNormalized = append(onNormalized, tok)
		} else {
			onRaw = append(onRaw, tok)
		}
	}
	if t.rawAdded, err = newAddedVocab(onRaw); err != nil {
		return nil, err
	}
	if t.normalAdded, err = newAddedVocab(onNormalized); err != nil {
		return nil, err
	}
	if err := t.readRoles(config); err != nil {
		return nil, err
	}
	return t, nil
}

func mergePairs(raw []rawMerge) [][2]string {
	out := make([][2]string, len(raw))
	for i, m := range raw {
		out[i] = m
	}
	return out
}

func readAddedTokens(raw []rawAdded) ([]addedToken, error) {
	out := make([]addedToken, 0, len(raw))
	seen := map[int32]string{}
	for _, r := range raw {
		if r.Content == "" {
			return nil, fmt.Errorf("an added_tokens entry must declare content")
		}
		if r.ID == nil || *r.ID < 0 {
			return nil, fmt.Errorf("added token %q has no non-negative id", r.Content)
		}
		if clash, ok := seen[*r.ID]; ok && clash != r.Content {
			return nil, fmt.Errorf("added token id %d is claimed by both %q and %q", *r.ID, clash, r.Content)
		}
		seen[*r.ID] = r.Content
		out = append(out, addedToken{r.Content, *r.ID, r.Special, r.LStrip, r.RStrip, r.SingleWord, r.Normalized})
	}
	return out, nil
}

// checkEveryCharacterIsRepresentable refuses a tokenizer that could drop text without trace: one
// with neither an unknown token nor byte fallback must be byte-level with the whole alphabet.
func (t *Tokenizer) checkEveryCharacterIsRepresentable() error {
	if t.bpe.hasUnk || t.bpe.byteIDs != nil {
		return nil
	}
	if !t.byteLevelOnly {
		return unsupported("this checkpoint declares no unk_token and no byte_fallback, and its " +
			"pre-tokenizer is not byte-level, so a character outside the vocabulary would be " +
			"dropped without trace. Refusing to load")
	}
	missing, reachable := 0, 0
	first := ""
	for b := 0; b < 256; b++ {
		if b == 0xC0 || b == 0xC1 || b >= 0xF5 { // bytes UTF-8 never contains
			continue
		}
		reachable++
		symbol := string(byteAlphabet.toRune[b])
		if _, ok := t.ids[symbol]; !ok {
			if missing == 0 {
				first = symbol
			}
			missing++
		}
	}
	if missing > 0 {
		return fmt.Errorf("this checkpoint declares no unk_token and no byte_fallback, and %d of the %d "+
			"byte-level alphabet symbols that text can reach are absent from its vocabulary (first: %q), "+
			"so some inputs would lose text silently. Refusing to load", missing, reachable, first)
	}
	return nil
}

func (t *Tokenizer) readRoles(config map[string]any) error {
	t.roles = map[string]int32{}
	if config == nil {
		return nil
	}
	for _, key := range roleKeys {
		value, ok := config[key]
		if !ok || value == nil {
			continue
		}
		var content string
		switch v := value.(type) {
		case string:
			content = v
		case map[string]any:
			c, isString := v["content"].(string)
			if !isString {
				return fmt.Errorf("%s in tokenizer_config.json must be a string or an object with a content field", key)
			}
			content = c
		default:
			return fmt.Errorf("%s in tokenizer_config.json must be a string or an object with a content field, found %T", key, value)
		}
		id, ok := t.ids[content]
		if !ok {
			return fmt.Errorf("tokenizer_config.json names %s as %s, but neither the vocabulary nor "+
				"the added-token table has it", content, key)
		}
		t.roles[key] = id
		if key == "mask_token" {
			t.maskToken = content
		}
	}
	return nil
}

// Encode returns the token ids of text.
func (t *Tokenizer) Encode(text string) []int32 {
	return t.EncodeLimit(text, -1)
}

// EncodeLimit is Encode truncated to its first limit ids; a negative limit means no limit. It
// stops tokenizing once the limit is reached.
func (t *Tokenizer) EncodeLimit(text string, limit int) []int32 {
	if limit == 0 {
		return []int32{}
	}
	text = unicodedata.ToValid(text) // a Python str is always valid; see unicodedata.ToValid
	capacity := len(text)/3 + 16
	if limit > 0 && limit < capacity {
		capacity = limit
	}
	out := make([]int32, 0, capacity)
	full := func() bool { return limit >= 0 && len(out) >= limit }
	for _, sp := range t.rawAdded.split(text) {
		if full() {
			break
		}
		if sp.isToken {
			out = append(out, sp.id)
			continue
		}
		normalized := t.normalizer(sp.text)
		if t.normalAdded.empty {
			out = t.encodePlain(normalized, out, limit)
			continue
		}
		for _, inner := range t.normalAdded.split(normalized) {
			if full() {
				break
			}
			if inner.isToken {
				out = append(out, inner.id)
			} else {
				out = t.encodePlain(inner.text, out, limit)
			}
		}
	}
	if limit >= 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func (t *Tokenizer) encodePlain(text string, out []int32, limit int) []int32 {
	for _, piece := range t.preTokenizer.preTokenize(text) {
		if limit >= 0 && len(out) >= limit {
			return out
		}
		out = t.bpe.encodePiece(piece, out)
	}
	return out
}

// Decode turns ids back into text. Ids the vocabulary does not hold are skipped, as the
// reference skips them; with skipSpecialTokens the added tokens marked special are skipped too.
func (t *Tokenizer) Decode(ids []int32, skipSpecialTokens bool) string {
	pieces := make([]string, 0, len(ids))
	for _, id := range ids {
		if skipSpecialTokens {
			if _, special := t.isSpecial[id]; special {
				continue
			}
		}
		if token, ok := t.IDToToken(id); ok {
			pieces = append(pieces, token)
		}
	}
	return decodeTokens(t.decoder, pieces)
}

// VocabSize is the number of tokens: the model vocabulary plus added tokens outside it.
func (t *Tokenizer) VocabSize() int { return len(t.ids) }

// HighestID is the largest token id.
func (t *Tokenizer) HighestID() int32 { return int32(len(t.tokens) - 1) }

// TokenToID returns the id of a token string.
func (t *Tokenizer) TokenToID(token string) (int32, bool) {
	id, ok := t.ids[token]
	return id, ok
}

// IDToToken returns the string of a token id.
func (t *Tokenizer) IDToToken(id int32) (string, bool) {
	if id < 0 || int(id) >= len(t.tokens) || t.tokens[id] == "" {
		return "", false
	}
	return t.tokens[id], true
}

// SpecialID returns the id of an added token marked special.
func (t *Tokenizer) SpecialID(content string) (int32, bool) {
	id, ok := t.specials[content]
	return id, ok
}

// SpecialTokens returns a copy of the special-token table.
func (t *Tokenizer) SpecialTokens() map[string]int32 {
	out := make(map[string]int32, len(t.specials))
	for k, v := range t.specials {
		out[k] = v
	}
	return out
}

// CLS is the classifier-start token id named by tokenizer_config.json.
func (t *Tokenizer) CLS() (int32, bool) { id, ok := t.roles["cls_token"]; return id, ok }

// SEP is the separator token id.
func (t *Tokenizer) SEP() (int32, bool) { id, ok := t.roles["sep_token"]; return id, ok }

// Mask is the mask token id.
func (t *Tokenizer) Mask() (int32, bool) { id, ok := t.roles["mask_token"]; return id, ok }

// Pad is the padding token id.
func (t *Tokenizer) Pad() (int32, bool) { id, ok := t.roles["pad_token"]; return id, ok }

// MaskToken is the mask token's text, or "" when none is configured.
func (t *Tokenizer) MaskToken() string { return t.maskToken }
