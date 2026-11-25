package sequence

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// The head-budget literals of laya/common.py's build_head, which Python does not name.
const (
	// optionTokenCap is the tokens one option keeps after its [MASK], before any budget.
	optionTokenCap = 48
	// minOptionBudget is the head room the options must leave before they are re-capped.
	minOptionBudget = 16
	// minTokensPerOption is the floor on a re-capped option, its [MASK] included.
	minTokensPerOption = 4
	// minInstructionTokens is the floor on the instruction, "<type> question: " included.
	minInstructionTokens = 8
)

// ErrMissingSpecialToken is wrapped when the tokenizer names no CLS, SEP or mask token, which a
// sequence cannot be built without.
var ErrMissingSpecialToken = errors.New("tokenizer names no required special token")

// ErrUnsupportedState is wrapped when a state holds a value json.dumps would not write the same
// way: a struct, a channel, a non-string map key, invalid UTF-8.
var ErrUnsupportedState = errors.New("state cannot be serialized as laya serializes it")

// Budget is a checkpoint's sequence budget, from its rl_agent_config.json.
type Budget struct {
	// MaxLen is the whole sequence: 512 on english, 1024 on multilingual.
	MaxLen int
	// HeadMaxLen is the question half: 192 on english, 256 on multilingual.
	HeadMaxLen int
}

// DefaultWindow is the scan window when none is asked for: max(64, MaxLen - HeadMaxLen - 8).
func (b Budget) DefaultWindow() int {
	return max(64, b.MaxLen-b.HeadMaxLen-8)
}

// Keep says which end of an oversized state survives.
type Keep string

const (
	// KeepStart keeps the beginning of the state; the zero Keep means the same.
	KeepStart Keep = "start"
	// KeepEnd keeps the end: what laya does for a list state, a conversation's turns, whose
	// latest turn matters most.
	KeepEnd Keep = "end"
)

// State is a state tokenized once, to be paired with every question of a request.
type State struct {
	// IDs are the state's token ids, without special tokens.
	IDs []int32
	// Keep is the end that survives a cut.
	Keep Keep
}

// SerializeState is laya's serialize_state: a string passes through, anything else becomes
// json.dumps(state, ensure_ascii=False). nil is the text "null", as in Python. A Go map is
// written in sorted key order; use lang.Object (or ParseState) for an order-sensitive state. A
// json.Number is written as Python's json.loads and json.dumps would write the literal: an int
// literal stays an int of any size (up to Python's 4300 digits), and a float literal takes
// Python's float repr (1e+16, 100.0, 1e-05), with an overflow written as Infinity.
func SerializeState(state any) (string, error) {
	if s, ok := state.(string); ok {
		return s, nil
	}
	if _, isNumber := state.(json.Number); !isNumber {
		if rv := reflect.ValueOf(state); rv.IsValid() && rv.Kind() == reflect.String {
			return rv.String(), nil // a named string type is still a str
		}
	}
	text, err := pyvalue.Dumps(state)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrUnsupportedState, err)
	}
	return text, nil
}

// ParseState reads a JSON document as Python's json.loads reads it, for use as a state: objects
// keep their key order (a repeated key keeps its first place and takes its last value) as
// lang.Object, and numbers stay json.Number, so that 2 serializes as 2 and 2.0 as 2.0. NaN,
// Infinity and -Infinity are accepted, as Python accepts them. A lone surrogate escape, a raw
// control character in a string, invalid UTF-8 and an int literal of more than 4300 digits are
// refused. data is the text json.loads would be given as a str, in UTF-8: a byte order mark is
// refused as Python refuses one in a str, and UTF-16 or UTF-32 input must be decoded first.
func ParseState(data []byte) (any, error) {
	v, err := pyvalue.Parse(data)
	if err != nil {
		return nil, fmt.Errorf("sequence: state: %w", err)
	}
	return public(v), nil
}

// public maps the decoder's pyjson.Object onto the exported lang.Object.
func public(v any) any {
	switch x := v.(type) {
	case pyjson.Object:
		out := make(lang.Object, len(x))
		for i, m := range x {
			out[i] = lang.Member{Key: m.Key.(string), Value: public(m.Value)}
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = public(item)
		}
		return out
	}
	return v
}

// EncodeState serializes state, scrubs the mask token from it and tokenizes it. Keep is KeepEnd
// for a slice or array state (not []byte) and KeepStart otherwise, as laya's predict decides.
func EncodeState(tok *tokenizer.Tokenizer, state any) (State, error) {
	sp, err := specials(tok)
	if err != nil {
		return State{}, err
	}
	text, err := SerializeState(state)
	if err != nil {
		return State{}, err
	}
	keep := KeepStart
	if isList(state) {
		keep = KeepEnd
	}
	return State{IDs: tok.Encode(strings.ReplaceAll(text, sp.maskToken, " ")), Keep: keep}, nil
}

// isList reports whether state is a Python list: a slice or array that is not bytes and not one
// of the ordered-object types, which are slices of members in Go but dicts in Python.
func isList(state any) bool {
	switch state.(type) {
	case lang.Object, pyjson.Object:
		return false
	}
	rv := reflect.ValueOf(state)
	if !rv.IsValid() || (rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array) {
		return false
	}
	return rv.Type().Elem().Kind() != reflect.Uint8
}

// special is the tokenizer's CLS, SEP and mask, resolved once per call.
type special struct {
	cls, sep, mask int32
	maskToken      string
}

func specials(tok *tokenizer.Tokenizer) (special, error) {
	if tok == nil {
		return special{}, errors.New("sequence: nil tokenizer")
	}
	var sp special
	var ok1, ok2, ok3 bool
	sp.cls, ok1 = tok.CLS()
	sp.sep, ok2 = tok.SEP()
	sp.mask, ok3 = tok.Mask()
	sp.maskToken = tok.MaskToken()
	switch {
	case !ok1:
		return special{}, fmt.Errorf("%w: no CLS token", ErrMissingSpecialToken)
	case !ok2:
		return special{}, fmt.Errorf("%w: no SEP token", ErrMissingSpecialToken)
	case !ok3 || sp.maskToken == "":
		return special{}, fmt.Errorf("%w: no mask token", ErrMissingSpecialToken)
	}
	return sp, nil
}

// Stats is what the head budget did to a question's options.
type Stats struct {
	// Options is how many options the question defines.
	Options int `json:"options"`
	// OptionsDistinct is how many still have a token span of their own. Fewer than Options
	// means the cap made two options the same tokens, which nothing downstream can see.
	OptionsDistinct int `json:"options_distinct"`
	// TokensPerOption is the cap applied to each option, or nil when none was.
	TokensPerOption *int `json:"tokens_per_option"`
}

// Collapsed reports that some options lost their distinct token span.
func (s Stats) Collapsed() bool { return s.OptionsDistinct < s.Options }

// Truncation is what the state budget dropped.
type Truncation struct {
	StateTokens        int  `json:"state_tokens"`
	StateTokensUsed    int  `json:"state_tokens_used"`
	StateTokensDropped int  `json:"state_tokens_dropped"`
	Truncated          bool `json:"truncated"`
}

// Head is the question half of a sequence, with no state.
type Head struct {
	// IDs run from [CLS] to the [SEP] closing the options.
	IDs []int32
	// Markers are the [MASK] positions, one per option slot, in presentation order.
	Markers []int
	Stats   Stats
}

// Sequence is one question paired with one state: one row of a batch.
type Sequence struct {
	// IDs are at most Budget.MaxLen token ids.
	IDs []int32
	// Markers are the [MASK] positions below MaxLen, one per option slot in presentation order:
	// slot s holds option q.OptionOrder()[s].
	Markers    []int
	Stats      Stats
	Truncation Truncation
}

// BuildHead is laya's build_head: [CLS] <type> question: <instructions> [SEP] [MASK] opt0 ...
// [SEP], with the options in q's OptionOrder.
func BuildHead(tok *tokenizer.Tokenizer, q schema.Question, headMaxLen int) (Head, error) {
	sp, err := specials(tok)
	if err != nil {
		return Head{}, err
	}
	if !q.Type().Valid() || q.NumOptions() == 0 {
		return Head{}, errors.New("sequence: the zero Question has no head; build one with schema.Choice, ChoiceOf, Score, Noul or NoulWith")
	}
	opts := q.RenderedOptions()
	order := q.OptionOrder()
	if order == nil {
		order = make([]int, len(opts))
		for i := range order {
			order[i] = i
		}
	}
	ins := strings.ReplaceAll(q.Instructions(), sp.maskToken, " ")
	headIDs := tok.Encode(string(q.Type()) + " question: " + ins)
	spans := make([][]int32, len(order))
	total := 0
	for slot, i := range order {
		// The cap is applied by the tokenizer, not after the fact, so a long description is
		// not tokenized in full to keep 48 tokens of it.
		text := " " + strings.ReplaceAll(opts[i], sp.maskToken, " ")
		span := append(make([]int32, 0, optionTokenCap+1), sp.mask)
		spans[slot] = append(span, tok.EncodeLimit(text, optionTokenCap)...)
		total += len(spans[slot])
	}
	budget := headMaxLen - total
	var perOption *int
	if budget < minOptionBudget {
		per := max(minTokensPerOption, floorDiv(headMaxLen-minOptionBudget, max(1, len(spans))))
		perOption = &per
		total = 0
		for slot, span := range spans {
			if len(span) > per {
				span = span[:per]
				spans[slot] = span
			}
			total += len(span)
		}
		budget = headMaxLen - total
	}
	if keep := max(minInstructionTokens, budget); keep < len(headIDs) {
		headIDs = headIDs[:keep]
	}
	ids := make([]int32, 0, 2+len(headIDs)+total+1)
	ids = append(ids, sp.cls)
	ids = append(ids, headIDs...)
	ids = append(ids, sp.sep)
	markers := make([]int, len(spans))
	distinct := make(map[string]struct{}, len(spans))
	for slot, span := range spans {
		markers[slot] = len(ids)
		ids = append(ids, span...)
		distinct[spanKey(span)] = struct{}{}
	}
	ids = append(ids, sp.sep)
	return Head{IDs: ids, Markers: markers, Stats: Stats{
		Options: len(spans), OptionsDistinct: len(distinct), TokensPerOption: perOption,
	}}, nil
}

// floorDiv is Python's a // b for b > 0.
func floorDiv(a, b int) int {
	q := a / b
	if a%b != 0 && a < 0 {
		q--
	}
	return q
}

// spanKey is a comparable key for a token span.
func spanKey(span []int32) string {
	b := make([]byte, 0, 4*len(span))
	for _, id := range span {
		b = append(b, byte(id), byte(id>>8), byte(id>>16), byte(id>>24))
	}
	return string(b)
}

func checkBudget(b Budget) error {
	if b.MaxLen <= 0 || b.HeadMaxLen <= 0 {
		return fmt.Errorf("sequence: budget needs a positive MaxLen and HeadMaxLen, got %d and %d", b.MaxLen, b.HeadMaxLen)
	}
	return nil
}

// MarkersError is returned by Build when the question's option markers do not all fit in
// MaxLen: the head is clamped to MaxLen, so the model would be asked about fewer options than the
// question defines. build_sequence drops them silently; the check that refuses is the agent's
// (Agent._encode_state), and Build carries it so no caller can forget it. ID is empty as Build
// returns it; set it to the question's id and Error is the agent's message, with its prefix.
type MarkersError struct {
	// ID is the question id, or "".
	ID string
	// Kept is how many markers fit below MaxLen, Total how many options the question defines.
	Kept, Total int
	// MaxLen and HeadMaxLen are the budget that was asked for.
	MaxLen, HeadMaxLen int
}

// Error is laya's message: "question 'id': only 3 of its 5 option markers fit in max_len=24 with
// head_max_len=16 spent on the question; lower head_max_len, raise max_len, or use fewer options".
func (e *MarkersError) Error() string {
	prefix := ""
	if e.ID != "" {
		prefix = "question " + pyjson.Repr(e.ID) + ": "
	}
	return fmt.Sprintf("%sonly %d of its %d option markers fit in max_len=%d with head_max_len=%d "+
		"spent on the question; lower head_max_len, raise max_len, or use fewer options",
		prefix, e.Kept, e.Total, e.MaxLen, e.HeadMaxLen)
}

// Build is laya's build_sequence: the head, then as much of state as the room after it allows,
// then [SEP], clamped to b.MaxLen. It is also the agent's marker check: when an option's marker
// falls past MaxLen it returns a *MarkersError instead of a sequence that asks about fewer
// options than were defined. Memory is linear in the question's total option text, as it is in
// Python: each option is tokenized to at most 48 tokens, but its rendered text is held whole.
func Build(tok *tokenizer.Tokenizer, q schema.Question, state State, b Budget) (Sequence, error) {
	if err := checkBudget(b); err != nil {
		return Sequence{}, err
	}
	head, err := BuildHead(tok, q, b.HeadMaxLen)
	if err != nil {
		return Sequence{}, err
	}
	sep, _ := tok.SEP() // BuildHead found it
	room := max(0, b.MaxLen-len(head.IDs)-1)
	// Not state[len-room:] by arithmetic on a zero room: that is the whole state in Python's
	// slice spelling, and none of it here.
	kept := state.IDs
	if len(kept) > room {
		if state.Keep == KeepEnd {
			kept = kept[len(kept)-room:]
		} else {
			kept = kept[:room]
		}
	}
	ids := make([]int32, 0, min(b.MaxLen, len(head.IDs)+len(kept)+1))
	ids = append(ids, head.IDs...)
	ids = append(ids, kept...)
	ids = append(ids, sep)
	if len(ids) > b.MaxLen {
		ids = ids[:b.MaxLen]
	}
	markers := make([]int, 0, len(head.Markers))
	for _, m := range head.Markers {
		if m < b.MaxLen {
			markers = append(markers, m)
		}
	}
	if len(markers) != len(head.Markers) {
		return Sequence{}, &MarkersError{Kept: len(markers), Total: len(head.Markers), MaxLen: b.MaxLen, HeadMaxLen: b.HeadMaxLen}
	}
	return Sequence{IDs: ids, Markers: markers, Stats: head.Stats, Truncation: Truncation{
		StateTokens:        len(state.IDs),
		StateTokensUsed:    len(kept),
		StateTokensDropped: len(state.IDs) - len(kept),
		Truncated:          len(kept) < len(state.IDs),
	}}, nil
}

// StateRoom is how many state tokens q leaves inside b, which is what Build keeps: laya's
// state_room. It is a property of the tokenizer as much as the question: one 88-option question
// leaves 150 tokens on english and 614 on multilingual.
func StateRoom(tok *tokenizer.Tokenizer, q schema.Question, b Budget) (int, error) {
	if err := checkBudget(b); err != nil {
		return 0, err
	}
	head, err := BuildHead(tok, q, b.HeadMaxLen)
	if err != nil {
		return 0, err
	}
	return max(0, b.MaxLen-len(head.IDs)-1), nil // -1 for the [SEP] that closes the state
}
