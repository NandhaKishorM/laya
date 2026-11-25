package schema

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
)

// Answer is one typed answer: a *ChoiceAnswer, *ScoreAnswer or *NoulAnswer. Switch on the
// concrete type, or on Type, to read the type's own fields; Base reads the fields all three share.
type Answer interface {
	// Type is the question type the answer is for.
	Type() Type
	// Base is the confidences and action every answer carries.
	Base() Common

	isAnswer()
}

// Common is what every answer carries. It is embedded in each answer type, so its fields read
// directly off a concrete answer (ans.AnswerConfidence).
//
// Two confidences, deliberately. AnswerConfidence is max(p), the probability mass on the
// reported answer: the quantity temperature scaling fits and every calibration figure in laya is
// computed on. Confidence is per type: normalized entropy for a choice or a score, max(p, 1-p)
// for a noul. They are on different scales and carry different guarantees, so they must not be
// compared against the same threshold. AnswerConfidence is calibrated only once the temperatures
// have been fitted on held-out data for that checkpoint and option count; the shipped
// checkpoints are over-confident.
type Common struct {
	// Confidence is the per-type confidence; see the note on Common.
	Confidence float64
	// AnswerConfidence is max(p), or the binning map's value for it when one is installed and
	// no language override applied.
	AnswerConfidence float64
	// Action is the head's second output.
	Action Action
	// Window names the window that decided the answer in a PredictLong scan of several windows;
	// nil everywhere else, where laya writes no "window" key.
	Window *Window
	// LowConfidence is laya's "low_confidence": true only on an answer a confidence gate flagged;
	// false is written as no key, as laya removes it.
	LowConfidence bool
	// Abstention is the gate's verdict on the answer, "passed", "abstained" or "unevaluated";
	// empty when no gate ran, where laya writes no "abstention" key.
	Abstention string
	// AbstentionThreshold is the threshold the answer was gated at; nil exactly when Abstention
	// is empty.
	AbstentionThreshold *float64
}

// Window is where in the tokenized state the deciding window sat. The probabilities of an
// answer that carries one are that window's, not a calibrated number for the whole state.
type Window struct {
	// Index counts the windows from 0.
	Index int `json:"index"`
	// TokenStart is the first state token of the window.
	TokenStart int `json:"token_start"`
	// TokenEnd is one past its last, clamped to the end of the state.
	TokenEnd int `json:"token_end"`
	// Count is how many windows the scan ran.
	Count int `json:"count"`
}

// Base returns c, so that every answer type satisfies Answer through the embedded Common.
func (c Common) Base() Common { return c }

// Action is the head's second output, nested as laya nests it under "action".
type Action struct {
	ActProbability float64 `json:"act_probability"`
}

// MarshalJSON writes {"act_probability": p}, with p spelled as Python spells a float.
func (a Action) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteString(`{"act_probability":`)
	if err := writeFloat(&b, a.ActProbability); err != nil {
		return nil, err
	}
	b.WriteByte('}')
	return []byte(b.String()), nil
}

// Probability is one option's probability, keyed by its label.
type Probability struct {
	Label Label
	P     float64
}

// Probabilities are an answer's per-option probabilities in label-index order. They marshal as
// one JSON object in that order, as laya's dict does.
type Probabilities []Probability

// Get is the probability of label. Labels read back from JSON are text, as the keys of a JSON
// object are: after a round trip, look an int, float or bool label up by Text(label.Key()).
func (ps Probabilities) Get(label Label) (float64, bool) {
	for _, p := range ps {
		if p.Label == label {
			return p.P, true
		}
	}
	return 0, false
}

// MarshalJSON writes {"label": p, ...} in order.
func (ps Probabilities) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	if err := writeProbabilities(&b, ps); err != nil {
		return nil, err
	}
	return []byte(b.String()), nil
}

// UnmarshalJSON reads {"label": p, ...} in document order.
func (ps *Probabilities) UnmarshalJSON(data []byte) error {
	v, err := pyvalue.Parse(data)
	if err != nil {
		return fmt.Errorf("schema: probabilities: %w", err)
	}
	out, err := probabilitiesFrom(v)
	if err != nil {
		return err
	}
	*ps = out
	return nil
}

// ChoiceAnswer picks one labelled option.
type ChoiceAnswer struct {
	// Choice is the label with the highest probability; the first such label on a tie, as
	// numpy.argmax breaks it. It keeps its type: the int 1 is reported as 1, not "1".
	Choice Label
	// Probabilities has one entry per option, in label order whatever the option order was.
	Probabilities Probabilities
	// Common.Confidence is 1 - H(p)/log(k), clipped to [0, 1]; 1 for a single option.
	Common
}

// ScoreAnswer rates on an ordered scale.
type ScoreAnswer struct {
	// Score is the expected level index, sum(i * p[i]), not the most likely level.
	Score float64
	// Legend is the text the model was shown for each level, index 0 first.
	Legend []string
	// Probabilities are keyed "0", "1", ... by level index.
	Probabilities Probabilities
	// Common.Confidence is 1 - H(p)/log(k), clipped to [0, 1].
	Common
}

// NoulAnswer says whether a statement holds.
type NoulAnswer struct {
	// Noul is the probability that the statement holds: p[1], the true side.
	Noul float64
	// Common.Confidence is max(Noul, 1-Noul).
	Common
}

func (*ChoiceAnswer) Type() Type { return TypeChoice }
func (*ScoreAnswer) Type() Type  { return TypeScore }
func (*NoulAnswer) Type() Type   { return TypeNoul }

func (*ChoiceAnswer) isAnswer() {}
func (*ScoreAnswer) isAnswer()  {}
func (*NoulAnswer) isAnswer()   {}

// MarshalJSON writes laya's answer dict: type, choice, probabilities, confidence,
// answer_confidence, action.
func (a ChoiceAnswer) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteString(`{"type":"choice","choice":`)
	choice, err := a.Choice.MarshalJSON()
	if err != nil {
		return nil, err
	}
	b.Write(choice)
	b.WriteString(`,"probabilities":`)
	if err := writeProbabilities(&b, a.Probabilities); err != nil {
		return nil, err
	}
	return finish(&b, a.Common)
}

// MarshalJSON writes laya's answer dict: type, score, legend ({"0": text, ...}), probabilities,
// confidence, answer_confidence, action.
func (a ScoreAnswer) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteString(`{"type":"score","score":`)
	if err := writeFloat(&b, a.Score); err != nil {
		return nil, err
	}
	b.WriteString(`,"legend":{`)
	for i, text := range a.Legend {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`"` + strconv.Itoa(i) + `":`)
		if err := writeString(&b, text); err != nil {
			return nil, err
		}
	}
	b.WriteString(`},"probabilities":`)
	if err := writeProbabilities(&b, a.Probabilities); err != nil {
		return nil, err
	}
	return finish(&b, a.Common)
}

// MarshalJSON writes laya's answer dict: type, noul, confidence, answer_confidence, action.
func (a NoulAnswer) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteString(`{"type":"noul","noul":`)
	if err := writeFloat(&b, a.Noul); err != nil {
		return nil, err
	}
	return finish(&b, a.Common)
}

// finish appends the fields every answer ends with and closes the object.
func finish(b *strings.Builder, c Common) ([]byte, error) {
	b.WriteString(`,"confidence":`)
	if err := writeFloat(b, c.Confidence); err != nil {
		return nil, err
	}
	b.WriteString(`,"answer_confidence":`)
	if err := writeFloat(b, c.AnswerConfidence); err != nil {
		return nil, err
	}
	act, err := c.Action.MarshalJSON()
	if err != nil {
		return nil, err
	}
	b.WriteString(`,"action":`)
	b.Write(act)
	if w := c.Window; w != nil {
		fmt.Fprintf(b, `,"window":{"index":%d,"token_start":%d,"token_end":%d,"count":%d}`,
			w.Index, w.TokenStart, w.TokenEnd, w.Count)
	}
	if c.LowConfidence {
		b.WriteString(`,"low_confidence":true`)
	}
	if c.Abstention != "" {
		b.WriteString(`,"abstention":`)
		if err := writeString(b, c.Abstention); err != nil {
			return nil, err
		}
	}
	if t := c.AbstentionThreshold; t != nil {
		b.WriteString(`,"abstention_threshold":`)
		if err := writeFloat(b, *t); err != nil {
			return nil, err
		}
	}
	b.WriteByte('}')
	return []byte(b.String()), nil
}

// writeFloat writes f as Python's json.dumps does (1.0, 0.5, 1e-05). JSON has no NaN or
// infinity, so those are an error rather than text a strict parser would refuse.
func writeFloat(b *strings.Builder, f float64) error {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return fmt.Errorf("schema: cannot marshal %v as JSON", f)
	}
	b.WriteString(pyjson.ReprFloat(f))
	return nil
}

func writeString(b *strings.Builder, s string) error {
	text, err := pyjson.Dumps(s)
	if err != nil {
		return err
	}
	b.WriteString(text)
	return nil
}

func writeProbabilities(b *strings.Builder, ps Probabilities) error {
	b.WriteByte('{')
	for i, p := range ps {
		if i > 0 {
			b.WriteByte(',')
		}
		if err := writeString(b, p.Label.Key()); err != nil {
			return err
		}
		b.WriteByte(':')
		if err := writeFloat(b, p.P); err != nil {
			return err
		}
	}
	b.WriteByte('}')
	return nil
}

// UnmarshalAnswer reads one answer dict, choosing the concrete type from its "type" key.
func UnmarshalAnswer(data []byte) (Answer, error) {
	v, err := pyvalue.Parse(data)
	if err != nil {
		return nil, fmt.Errorf("schema: answer: %w", err)
	}
	return answerFrom(v)
}

func answerFrom(v any) (Answer, error) {
	obj, ok := v.(pyjson.Object)
	if !ok {
		return nil, fmt.Errorf("schema: an answer is a JSON object, got %s", pyvalue.TypeName(v))
	}
	typ, err := fieldString(obj, "type")
	if err != nil {
		return nil, err
	}
	var common Common
	if common.Confidence, err = fieldFloat(obj, "confidence"); err != nil {
		return nil, err
	}
	if common.AnswerConfidence, err = fieldFloat(obj, "answer_confidence"); err != nil {
		return nil, err
	}
	act, err := field(obj, "action")
	if err != nil {
		return nil, err
	}
	actObj, ok := act.(pyjson.Object)
	if !ok {
		return nil, errors.New(`schema: answer "action" must be an object`)
	}
	if common.Action.ActProbability, err = fieldFloat(actObj, "act_probability"); err != nil {
		return nil, err
	}
	if raw, ok := optional(obj, "window"); ok {
		w, err := windowFrom(raw)
		if err != nil {
			return nil, err
		}
		common.Window = w
	}
	if err := gateFrom(obj, &common); err != nil {
		return nil, err
	}
	switch Type(typ) {
	case TypeChoice:
		a := &ChoiceAnswer{Common: common}
		choice, err := field(obj, "choice")
		if err != nil {
			return nil, err
		}
		if a.Choice, err = LabelOf(choice); err != nil || choice == nil {
			return nil, fmt.Errorf(`schema: answer "choice" must be a string, bool or number, got %s`, pyvalue.TypeName(choice))
		}
		if a.Probabilities, err = fieldProbabilities(obj); err != nil {
			return nil, err
		}
		return a, nil
	case TypeScore:
		a := &ScoreAnswer{Common: common}
		if a.Score, err = fieldFloat(obj, "score"); err != nil {
			return nil, err
		}
		legend, err := field(obj, "legend")
		if err != nil {
			return nil, err
		}
		lobj, ok := legend.(pyjson.Object)
		if !ok {
			return nil, errors.New(`schema: answer "legend" must be an object`)
		}
		a.Legend = make([]string, len(lobj))
		for i, m := range lobj {
			text, ok := m.Value.(string)
			if !ok || m.Key != strconv.Itoa(i) {
				return nil, fmt.Errorf(`schema: answer "legend" must map "0", "1", ... to text, got key %v`, m.Key)
			}
			a.Legend[i] = text
		}
		if a.Probabilities, err = fieldProbabilities(obj); err != nil {
			return nil, err
		}
		return a, nil
	case TypeNoul:
		a := &NoulAnswer{Common: common}
		if a.Noul, err = fieldFloat(obj, "noul"); err != nil {
			return nil, err
		}
		return a, nil
	}
	return nil, fmt.Errorf("schema: unknown answer type %s", pyvalue.Repr(typ))
}

func field(obj pyjson.Object, key string) (any, error) {
	for _, m := range obj {
		if m.Key == key {
			return m.Value, nil
		}
	}
	return nil, fmt.Errorf("schema: answer has no %q", key)
}

func optional(obj pyjson.Object, key string) (any, bool) {
	for _, m := range obj {
		if m.Key == key {
			return m.Value, true
		}
	}
	return nil, false
}

// gateFrom reads the keys a confidence gate writes. An absent key is no gate; low_confidence is
// true or absent, as laya writes it.
func gateFrom(obj pyjson.Object, c *Common) error {
	if raw, ok := optional(obj, "low_confidence"); ok {
		if b, isBool := raw.(bool); !isBool || !b {
			return errors.New(`schema: answer "low_confidence" must be true when present`)
		}
		c.LowConfidence = true
	}
	if raw, ok := optional(obj, "abstention"); ok {
		s, isStr := raw.(string)
		if !isStr || (s != "passed" && s != "abstained" && s != "unevaluated") {
			return fmt.Errorf(`schema: answer "abstention" must be one of "passed", "abstained", "unevaluated", got %s`, pyvalue.Repr(raw))
		}
		c.Abstention = s
	}
	if _, ok := optional(obj, "abstention_threshold"); ok {
		t, err := fieldFloat(obj, "abstention_threshold")
		if err != nil {
			return err
		}
		c.AbstentionThreshold = &t
	}
	if (c.Abstention == "") != (c.AbstentionThreshold == nil) {
		return errors.New(`schema: answer "abstention" and "abstention_threshold" come together`)
	}
	return nil
}

func windowFrom(v any) (*Window, error) {
	obj, ok := v.(pyjson.Object)
	if !ok {
		return nil, errors.New(`schema: answer "window" must be an object`)
	}
	w := &Window{}
	for key, dst := range map[string]*int{"index": &w.Index, "token_start": &w.TokenStart,
		"token_end": &w.TokenEnd, "count": &w.Count} {
		f, err := fieldFloat(obj, key)
		if err != nil || f != math.Trunc(f) {
			return nil, fmt.Errorf(`schema: answer window %q must be an integer`, key)
		}
		*dst = int(f)
	}
	if w.Count < 1 || w.Index < 0 || w.Index >= w.Count || w.TokenStart < 0 || w.TokenEnd < w.TokenStart {
		return nil, errors.New(`schema: answer "window" is out of range`)
	}
	return w, nil
}

func fieldString(obj pyjson.Object, key string) (string, error) {
	v, err := field(obj, key)
	if err != nil {
		return "", err
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("schema: answer %q must be a string, got %s", key, pyvalue.TypeName(v))
	}
	return s, nil
}

func fieldFloat(obj pyjson.Object, key string) (float64, error) {
	v, err := field(obj, key)
	if err != nil {
		return 0, err
	}
	return toFloat(v, key)
}

func toFloat(v any, what string) (float64, error) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, fmt.Errorf("schema: %q must be a number, got %s", what, pyvalue.TypeName(v))
	}
	f, err := n.Float64()
	if err != nil {
		return 0, fmt.Errorf("schema: %q is not a float64: %v", what, err)
	}
	return f, nil
}

func fieldProbabilities(obj pyjson.Object) (Probabilities, error) {
	v, err := field(obj, "probabilities")
	if err != nil {
		return nil, err
	}
	return probabilitiesFrom(v)
}

func probabilitiesFrom(v any) (Probabilities, error) {
	obj, ok := v.(pyjson.Object)
	if !ok {
		return nil, fmt.Errorf("schema: probabilities must be an object, got %s", pyvalue.TypeName(v))
	}
	out := make(Probabilities, len(obj))
	for i, m := range obj {
		p, err := toFloat(m.Value, m.Key.(string))
		if err != nil {
			return nil, err
		}
		out[i] = Probability{Label: Text(m.Key.(string)), P: p}
	}
	return out, nil
}
