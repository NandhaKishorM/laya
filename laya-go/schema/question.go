package schema

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// Type is a question type. Its Code is the integer the decision head receives, and its string is
// the first word of the sequence and the prefix of a temperature bucket ("choice:3-5").
type Type string

const (
	// TypeChoice picks one labelled option; the answer reports every option's probability.
	TypeChoice Type = "choice"
	// TypeScore rates on an ordered scale; the answer is the expected level, not the likeliest.
	TypeScore Type = "score"
	// TypeNoul asks whether a statement holds; the answer is the probability that it does.
	TypeNoul Type = "noul"
)

// Code is the integer the head receives for t: choice 0, score 1, noul 2 (laya's QTYPES).
// It is -1 for a value that is not one of the three types.
func (t Type) Code() int {
	switch t {
	case TypeChoice:
		return 0
	case TypeScore:
		return 1
	case TypeNoul:
		return 2
	}
	return -1
}

// Valid reports whether t is one of the three question types.
func (t Type) Valid() bool { return t.Code() >= 0 }

// ErrInvalidQuestion is wrapped by every error that refuses a question definition. The concrete
// error is a *QuestionError.
var ErrInvalidQuestion = errors.New("invalid question")

// QuestionError refuses one question definition and says what to fix.
type QuestionError struct {
	// ID is the question id, or "" when the question was refused before it was given one.
	ID string
	// Reason is laya's own wording for the refusal, without the "question 'id': " prefix.
	Reason string
}

// Error is "question 'id': reason" (the id spelled as Python's repr spells it), or the reason
// alone when there is no id.
func (e *QuestionError) Error() string {
	if e.ID == "" {
		return e.Reason
	}
	return "question " + pyjson.Repr(e.ID) + ": " + e.Reason
}

// Unwrap returns ErrInvalidQuestion.
func (e *QuestionError) Unwrap() error { return ErrInvalidQuestion }

func refuse(format string, args ...any) *QuestionError {
	return &QuestionError{Reason: fmt.Sprintf(format, args...)}
}

// Reasons laya's Agent._check_question gives, word for word. Python spells these in one function;
// the typed builders and the wire-form reader both reach them, so they live in one place.
const (
	reasonNoInstructions   = "no 'instructions'; add the text the model should answer"
	reasonNilInstructions  = "'instructions' must not be None; add the text the model should answer"
	reasonBlankIns         = "'instructions' must not be empty; add the text the model should answer"
	reasonChoiceShape      = "a choice question takes 'criteria' as a dict of label -> description, or a list of labels"
	reasonChoiceEmpty      = "a choice question needs at least one criterion"
	reasonScoreShape       = "a score question takes 'criteria' as a list of level descriptions, index 0 first"
	reasonScoreEmpty       = "a score question needs at least one level"
	reasonNoulShape        = "a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it"
	reasonLabelsOnlyNoul   = "'labels' is only supported for noul questions"
	reasonNoulLabels       = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
	reasonNoulKeysFormat   = "a noul question takes 'criteria' keyed only 'true'/'false' (either or both, and omitted is fine), got %s. Those keys are the option texts the model reads; any other key was silently dropped and replaced with the defaults. If you want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set 'labels' instead."
	reasonOrderFormat      = "'option_order' must be a permutation of range(%d) -- one slot per option, each option once -- got %s"
	reasonLabelTypeFormat  = "choice label %d is a %s; a label is rendered as option text and used as the answer key, so it must be a scalar (a string, number or bool), got %s"
	reasonLabelNullFormat  = "choice label %d is null; a label is rendered as option text and used as the answer key, so it must be a string, number or bool -- a null label renders as the text \"None\" while its answer key is \"null\""
	reasonLabelRepeatFmt   = "choice label %d (%s) repeats label %d; the labels are the answer keys, so every option needs its own (1, 1.0 and True are one key)"
	reasonScoreNullFormat  = "score level %d is null; give every level a description, index 0 first"
	reasonTypeFormat       = "unknown type %s; use one of ['choice', 'noul', 'score']"
	reasonDefinitionFormat = "definition must be a dict, got %s"

	defaultFalseText = "no, the statement does not hold"
	defaultTrueText  = "yes, the statement holds"
)

// Option is one choice option: a label, which is the answer key, and an optional criterion,
// which describes it to the model.
type Option struct {
	// Label names the option. It is rendered as option text and keys the answer's
	// probabilities, and it must be unique within the question. An empty label is accepted, as
	// it is in Python.
	Label string
	// LabelValue, when not nil, is the label instead of Label (which must then be empty): a bool,
	// an integer, a float or a json.Number, as Python allows. 1 and "1" are different labels, and
	// 1, 1.0 and true are one (Python keys a dict that way, so Choice refuses the repeat).
	LabelValue any
	// Criterion describes the option: a string, or a structured value rendered as JSON. nil and
	// "" mean "no description", and the option renders as its label alone.
	Criterion any
}

// NoulSpec is the optional half of a noul question.
type NoulSpec struct {
	// True and False describe the two sides. nil or "" keeps the default text: "yes, the
	// statement holds" and "no, the statement does not hold".
	True, False any
	// TrueLabel and FalseLabel rename the sides in the rendered options. Both empty keeps "true"
	// and "false". Otherwise both must be non-blank after Python's str.strip, which is applied,
	// and they must differ.
	TrueLabel, FalseLabel string
}

// criterion is a description snapshotted when the question is built, so a Question never sees
// its caller's later changes to a map or slice it was given.
type criterion struct {
	set  bool   // described: not nil and not ""
	text string // render_criterion of the value
	raw  string // the value as JSON, for the wire form
}

type choiceOption struct {
	label Label
	crit  criterion
}

// Question is one decision to put to the model. The zero value is not a valid question; build
// one with Choice, ChoiceOf, Score, Noul or NoulWith. A Question is immutable: WithOptionOrder
// returns a copy.
type Question struct {
	typ          Type
	instructions string
	options      []choiceOption // choice
	levels       []criterion    // score
	noul         [2]criterion   // noul: false side, true side
	falseLabel   string
	trueLabel    string
	customLabels bool
	order        []int // nil: label order
}

// newCriterion snapshots one description. nil and "" are no description at all.
func newCriterion(v any) (criterion, error) {
	if v == nil {
		return criterion{}, nil
	}
	if s, ok := v.(string); ok {
		if s == "" {
			return criterion{}, nil
		}
		raw, err := pyjson.Dumps(s)
		if err != nil {
			return criterion{}, err
		}
		return criterion{set: true, text: s, raw: raw}, nil
	}
	text, err := pyvalue.Dumps(v)
	if err != nil {
		return criterion{}, err
	}
	if text == "null" { // a typed nil pointer: no description, as nil is
		return criterion{}, nil
	}
	return criterion{set: true, text: text, raw: text}, nil
}

// checkInstructions returns the reason the instruction text is refused, or "".
func checkInstructions(ins string) string {
	if unicodedata.IsBlank(ins) {
		return reasonBlankIns
	}
	if !utf8.ValidString(ins) {
		return "'instructions' is not valid UTF-8; a Python str always is"
	}
	return ""
}

// checkLabelsDistinct is the duplicate check of _check_question over labels, whose repr text
// is given by repr.
func checkLabelsDistinct(n int, key func(i int) string, repr func(i int) string) string {
	first := make(map[string]int, n)
	for i := 0; i < n; i++ {
		k := key(i)
		if j, seen := first[k]; seen {
			return fmt.Sprintf(reasonLabelRepeatFmt, i, repr(i), j)
		}
		first[k] = i
	}
	return ""
}

// Choice is a question that picks one of options, presented in the order given.
//
// instructions must not be blank (Python's str.strip decides blank). A structured instruction
// is the JSON text sequence.SerializeState spells for it, as laya's _to_internal does.
func Choice(instructions string, options ...Option) (Question, error) {
	if r := checkInstructions(instructions); r != "" {
		return Question{}, &QuestionError{Reason: r}
	}
	if len(options) == 0 {
		return Question{}, &QuestionError{Reason: reasonChoiceEmpty}
	}
	labels := make([]Label, len(options))
	for i, o := range options {
		switch {
		case o.LabelValue != nil && o.Label != "":
			return Question{}, refuse("choice label %d sets both Label and LabelValue; set one", i)
		case o.LabelValue != nil:
			l, err := LabelOf(o.LabelValue)
			if err != nil {
				return Question{}, refuse("choice label %d: %v", i, err)
			}
			labels[i] = l
		case !utf8.ValidString(o.Label):
			return Question{}, refuse("choice label %d is not valid UTF-8; a Python str always is", i)
		default:
			labels[i] = Text(o.Label)
		}
	}
	if r := checkLabelsDistinct(len(options),
		func(i int) string { return labels[i].eqKey() },
		func(i int) string { return labels[i].repr() }); r != "" {
		return Question{}, &QuestionError{Reason: r}
	}
	opts := make([]choiceOption, len(options))
	for i, o := range options {
		c, err := newCriterion(o.Criterion)
		if err != nil {
			return Question{}, refuse("choice option %s: criterion cannot be rendered: %v", labels[i].repr(), err)
		}
		opts[i] = choiceOption{label: labels[i], crit: c}
	}
	return Question{typ: TypeChoice, instructions: instructions, options: opts}, nil
}

// ChoiceOf is Choice with labels and no descriptions: laya's list form of choice criteria.
func ChoiceOf(instructions string, labels ...string) (Question, error) {
	options := make([]Option, len(labels))
	for i, l := range labels {
		options[i] = Option{Label: l}
	}
	return Choice(instructions, options...)
}

// Score is a question that rates on the ordered levels given, index 0 first. A level is a string
// or a structured value and must not be nil; it renders as "level <i>: <criterion>".
func Score(instructions string, levels ...any) (Question, error) {
	if r := checkInstructions(instructions); r != "" {
		return Question{}, &QuestionError{Reason: r}
	}
	if len(levels) == 0 {
		return Question{}, &QuestionError{Reason: reasonScoreEmpty}
	}
	out := make([]criterion, len(levels))
	for i, v := range levels {
		c, err := newCriterion(v)
		if err != nil {
			return Question{}, refuse("score level %d cannot be rendered: %v", i, err)
		}
		if v == nil || (!c.set && v != "") {
			return Question{}, refuse(reasonScoreNullFormat, i)
		}
		if !c.set { // the empty string is a level: it renders as "level i: "
			c = criterion{set: true, raw: `""`}
		}
		out[i] = c
	}
	return Question{typ: TypeScore, instructions: instructions, levels: out}, nil
}

// Noul asks whether the statement in instructions holds, with the default option texts.
func Noul(instructions string) (Question, error) { return NoulWith(instructions, NoulSpec{}) }

// NoulWith is Noul with per-side descriptions and labels.
func NoulWith(instructions string, spec NoulSpec) (Question, error) {
	if r := checkInstructions(instructions); r != "" {
		return Question{}, &QuestionError{Reason: r}
	}
	q := Question{typ: TypeNoul, instructions: instructions, falseLabel: "false", trueLabel: "true"}
	for i, v := range [2]any{spec.False, spec.True} {
		c, err := newCriterion(v)
		if err != nil {
			return Question{}, refuse("noul %s description cannot be rendered: %v", [2]string{"false", "true"}[i], err)
		}
		q.noul[i] = c
	}
	if spec.TrueLabel != "" || spec.FalseLabel != "" {
		f, t := unicodedata.Strip(spec.FalseLabel), unicodedata.Strip(spec.TrueLabel)
		if f == "" || t == "" || f == t || !utf8.ValidString(f) || !utf8.ValidString(t) {
			return Question{}, &QuestionError{Reason: reasonNoulLabels}
		}
		q.falseLabel, q.trueLabel, q.customLabels = f, t, true
	}
	return q, nil
}

// MustQuestion returns q, or panics with err. It is for questions written as literals in source,
// as regexp.MustCompile is for patterns.
func MustQuestion(q Question, err error) Question {
	if err != nil {
		panic(err)
	}
	return q
}

// NumOptions is how many options the question defines: the choices, the levels, or 2.
func (q Question) NumOptions() int {
	switch q.typ {
	case TypeChoice:
		return len(q.options)
	case TypeScore:
		return len(q.levels)
	case TypeNoul:
		return 2
	}
	return 0
}

func reprInts(order []int) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, v := range order {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(strconv.Itoa(v))
	}
	b.WriteByte(']')
	return b.String()
}

// WithOptionOrder returns a copy of q that presents its options in order: slot s shows option
// order[s]. order must be a permutation of 0..NumOptions()-1. Answers are still reported in label
// order; only the input changes. A nil order restores label order.
func (q Question) WithOptionOrder(order []int) (Question, error) {
	if !q.typ.Valid() {
		return Question{}, refuse("the zero Question has no options to order; build one with Choice, ChoiceOf, Score, Noul or NoulWith")
	}
	if order == nil {
		q.order = nil
		return q, nil
	}
	n := q.NumOptions()
	seen := make([]bool, n)
	ok := len(order) == n
	for _, v := range order {
		if !ok {
			break
		}
		if v < 0 || v >= n || seen[v] {
			ok = false
			break
		}
		seen[v] = true
	}
	if !ok {
		return Question{}, refuse(reasonOrderFormat, n, reprInts(order))
	}
	q.order = append([]int(nil), order...)
	return q, nil
}

// Type is the question's type.
func (q Question) Type() Type { return q.typ }

// Instructions is the instruction text as given.
func (q Question) Instructions() string { return q.instructions }

// OptionOrder is the presentation order set by WithOptionOrder, or nil for label order. The
// returned slice is a copy.
func (q Question) OptionOrder() []int {
	if q.order == nil {
		return nil
	}
	return append([]int(nil), q.order...)
}

// Labels are the labels as text, str() of each, in label-index order: the choice labels, the
// level indices "0", "1", ..., or the noul's false and true labels. This is the spelling in the
// rendered prompt; AnswerLabels are the answer keys.
func (q Question) Labels() []string {
	switch q.typ {
	case TypeChoice:
		out := make([]string, len(q.options))
		for i, o := range q.options {
			out[i] = o.label.String()
		}
		return out
	case TypeScore:
		out := make([]string, len(q.levels))
		for i := range out {
			out[i] = strconv.Itoa(i)
		}
		return out
	case TypeNoul:
		return []string{q.falseLabel, q.trueLabel}
	}
	return nil
}

// AnswerLabels are the labels an answer is keyed by, in label-index order, with their types: the
// choice labels as given, the level indices "0", "1", ... as text, or the noul's two labels.
func (q Question) AnswerLabels() []Label {
	switch q.typ {
	case TypeChoice:
		out := make([]Label, len(q.options))
		for i, o := range q.options {
			out[i] = o.label
		}
		return out
	case TypeScore:
		out := make([]Label, len(q.levels))
		for i := range out {
			out[i] = Text(strconv.Itoa(i))
		}
		return out
	case TypeNoul:
		return []Label{Text(q.falseLabel), Text(q.trueLabel)}
	}
	return nil
}

// RenderedOptions are the option texts the model reads, in label-index order: laya's
// render_options. Everything downstream is token ids derived from these strings, so this is
// the text a port must match character for character. It does not apply OptionOrder.
func (q Question) RenderedOptions() []string {
	switch q.typ {
	case TypeChoice:
		out := make([]string, len(q.options))
		for i, o := range q.options {
			if o.crit.set {
				out[i] = o.label.String() + ": " + o.crit.text
			} else {
				out[i] = o.label.String()
			}
		}
		return out
	case TypeScore:
		out := make([]string, len(q.levels))
		for i, l := range q.levels {
			out[i] = "level " + strconv.Itoa(i) + ": " + l.text
		}
		return out
	case TypeNoul:
		f, t := defaultFalseText, defaultTrueText
		if q.noul[0].set {
			f = q.noul[0].text
		}
		if q.noul[1].set {
			t = q.noul[1].text
		}
		return []string{q.falseLabel + ": " + f, q.trueLabel + ": " + t}
	}
	return nil
}

// Legend maps each level index of a score question to the text the model was shown for it,
// index 0 first. It is nil for a choice or noul.
func (q Question) Legend() []string {
	if q.typ != TypeScore {
		return nil
	}
	out := make([]string, len(q.levels))
	for i, l := range q.levels {
		out[i] = l.text
	}
	return out
}

// MarshalJSON writes the Python wire form: type, instructions, then the criteria, labels and
// option_order the question has, in that order. A plain noul has no criteria key.
func (q Question) MarshalJSON() ([]byte, error) {
	if !q.typ.Valid() {
		return nil, errors.New("schema: cannot marshal the zero Question")
	}
	ins, err := pyjson.Dumps(q.instructions)
	if err != nil {
		return nil, err
	}
	var b strings.Builder
	b.WriteString(`{"type":"` + string(q.typ) + `","instructions":` + ins)
	switch q.typ {
	case TypeChoice:
		described := false
		for _, o := range q.options {
			described = described || o.crit.set
		}
		if !described {
			b.WriteString(`,"criteria":[`)
			for i, o := range q.options {
				if i > 0 {
					b.WriteByte(',')
				}
				label, err := o.label.MarshalJSON()
				if err != nil {
					return nil, err
				}
				b.Write(label)
			}
			b.WriteByte(']')
			break
		}
		b.WriteString(`,"criteria":{`)
		for i, o := range q.options {
			if i > 0 {
				b.WriteByte(',')
			}
			label, err := pyjson.Dumps(o.label.Key())
			if err != nil {
				return nil, err
			}
			raw := "null"
			if o.crit.set {
				raw = o.crit.raw
			}
			b.WriteString(label + ":" + raw)
		}
		b.WriteByte('}')
	case TypeScore:
		b.WriteString(`,"criteria":[`)
		for i, l := range q.levels {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(l.raw)
		}
		b.WriteByte(']')
	case TypeNoul:
		if q.noul[0].set || q.noul[1].set {
			b.WriteString(`,"criteria":{`)
			sep := ""
			for i, key := range [2]string{"false", "true"} {
				if q.noul[i].set {
					b.WriteString(sep + `"` + key + `":` + q.noul[i].raw)
					sep = ","
				}
			}
			b.WriteByte('}')
		}
		if q.customLabels {
			f, _ := pyjson.Dumps(q.falseLabel)
			t, _ := pyjson.Dumps(q.trueLabel)
			b.WriteString(`,"labels":{"false":` + f + `,"true":` + t + `}`)
		}
	}
	if q.order != nil {
		order, _ := json.Marshal(q.order)
		b.WriteString(`,"option_order":` + string(order))
	}
	b.WriteByte('}')
	return []byte(b.String()), nil
}

// UnmarshalJSON reads the Python wire form and validates it as laya's _check_question does,
// keeping the criteria in document order.
func (q *Question) UnmarshalJSON(data []byte) error {
	v, err := pyvalue.Parse(data)
	if err != nil {
		return fmt.Errorf("schema: question: %w", err)
	}
	parsed, qerr := fromWire(v)
	if qerr != nil {
		return qerr
	}
	*q = parsed
	return nil
}

// RenderCriterion is laya's render_criterion: a string passes through, anything else becomes
// json.dumps(value, ensure_ascii=False, separators=(", ", ": ")). A value Python would stringify
// with default=str is refused instead, so no Go type name reaches the model's input.
func RenderCriterion(value any) (string, error) {
	if s, ok := value.(string); ok {
		return s, nil
	}
	text, err := pyvalue.Dumps(value)
	if err != nil {
		return "", fmt.Errorf("schema: criterion: %w", err)
	}
	return text, nil
}

// fromWire reads one definition in the Python wire form, in the order _check_question checks it.
// The caller names the question in the error.
func fromWire(v any) (Question, *QuestionError) {
	def, ok := v.(pyjson.Object)
	if !ok {
		return Question{}, refuse(reasonDefinitionFormat, pyvalue.TypeName(v))
	}
	get := func(key string) (any, bool) {
		for _, m := range def {
			if m.Key == key {
				return m.Value, true
			}
		}
		return nil, false
	}
	tv, _ := get("type")
	ts, isStr := tv.(string)
	typ := Type(ts)
	if !isStr || !typ.Valid() {
		return Question{}, refuse(reasonTypeFormat, pyvalue.Repr(tv))
	}
	ins, qerr := wireInstructions(get)
	if qerr != nil {
		return Question{}, qerr
	}
	q := Question{typ: typ, instructions: ins}
	crit, _ := get("criteria")
	switch typ {
	case TypeChoice:
		q.options, qerr = wireChoice(crit)
	case TypeScore:
		q.levels, qerr = wireScore(crit)
	default:
		q.noul, qerr = wireNoul(crit)
		q.falseLabel, q.trueLabel = "false", "true"
	}
	if qerr != nil {
		return Question{}, qerr
	}
	if order, present := get("option_order"); present {
		n, bad := q.NumOptions(), true
		if list, ok := order.([]any); ok && len(list) == n {
			ints, ok := wireOrder(list, n)
			if ok {
				q.order, bad = ints, false
			}
		}
		if bad {
			return Question{}, refuse(reasonOrderFormat, n, pyvalue.Repr(order))
		}
	}
	if labels, present := get("labels"); present {
		if typ != TypeNoul {
			return Question{}, &QuestionError{Reason: reasonLabelsOnlyNoul}
		}
		if qerr := wireLabels(&q, labels); qerr != nil {
			return Question{}, qerr
		}
	}
	return q, nil
}

func wireInstructions(get func(string) (any, bool)) (string, *QuestionError) {
	ins, present := get("instructions")
	if !present {
		return "", &QuestionError{Reason: reasonNoInstructions}
	}
	switch x := ins.(type) {
	case nil:
		return "", &QuestionError{Reason: reasonNilInstructions}
	case string:
		if r := checkInstructions(x); r != "" {
			return "", &QuestionError{Reason: r}
		}
		return x, nil
	case []any:
		if len(x) == 0 {
			return "", &QuestionError{Reason: reasonBlankIns}
		}
	case pyjson.Object:
		if len(x) == 0 {
			return "", &QuestionError{Reason: reasonBlankIns}
		}
	}
	text, err := pyvalue.Dumps(ins) // _to_internal: json.dumps(ins, ensure_ascii=False)
	if err != nil {
		return "", refuse("'instructions' cannot be written as JSON: %v", err)
	}
	return text, nil
}

func wireChoice(crit any) ([]choiceOption, *QuestionError) {
	var labels []Label
	var values []any
	switch x := crit.(type) {
	case pyjson.Object:
		for _, m := range x {
			labels = append(labels, Text(m.Key.(string)))
			values = append(values, m.Value)
		}
	case []any:
		for i, label := range x {
			switch label.(type) {
			case string, bool, json.Number, float64:
			case nil:
				return nil, refuse(reasonLabelNullFormat, i)
			default:
				return nil, refuse(reasonLabelTypeFormat, i, pyvalue.TypeName(label), pyvalue.Repr(label))
			}
			l, err := LabelOf(label)
			if err != nil {
				return nil, refuse("choice label %d: %v", i, err)
			}
			labels = append(labels, l)
			values = append(values, nil)
		}
		if r := checkLabelsDistinct(len(labels),
			func(i int) string { return labels[i].eqKey() }, func(i int) string { return labels[i].repr() }); r != "" {
			return nil, &QuestionError{Reason: r}
		}
	default:
		return nil, &QuestionError{Reason: reasonChoiceShape}
	}
	if len(labels) == 0 {
		return nil, &QuestionError{Reason: reasonChoiceEmpty}
	}
	opts := make([]choiceOption, len(labels))
	for i, label := range labels {
		c, err := newCriterion(values[i])
		if err != nil {
			return nil, refuse("choice option %s: criterion cannot be rendered: %v", label.repr(), err)
		}
		opts[i] = choiceOption{label: label, crit: c}
	}
	return opts, nil
}

func wireScore(crit any) ([]criterion, *QuestionError) {
	list, ok := crit.([]any)
	if !ok {
		return nil, &QuestionError{Reason: reasonScoreShape}
	}
	if len(list) == 0 {
		return nil, &QuestionError{Reason: reasonScoreEmpty}
	}
	out := make([]criterion, len(list))
	for i, v := range list {
		if v == nil {
			return nil, refuse(reasonScoreNullFormat, i)
		}
		c, err := newCriterion(v)
		if err != nil {
			return nil, refuse("score level %d cannot be rendered: %v", i, err)
		}
		if !c.set {
			c = criterion{set: true, raw: `""`}
		}
		out[i] = c
	}
	return out, nil
}

func wireNoul(crit any) ([2]criterion, *QuestionError) {
	var out [2]criterion
	if crit == nil {
		return out, nil
	}
	obj, ok := crit.(pyjson.Object)
	if !ok {
		return out, &QuestionError{Reason: reasonNoulShape}
	}
	// _to_internal: {str(k).lower(): v}; a later key of the same spelling wins.
	sides := map[string]any{}
	for _, m := range obj {
		sides[unicodedata.Lower(m.Key.(string))] = m.Value
	}
	var unknown []string
	for k := range sides {
		if k != "true" && k != "false" {
			unknown = append(unknown, k)
		}
	}
	if len(unknown) > 0 {
		keys := make([]string, 0, len(sides))
		for k := range sides {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		items := make([]any, len(keys))
		for i, k := range keys {
			items[i] = k
		}
		return out, refuse(reasonNoulKeysFormat, pyvalue.Repr(items))
	}
	for i, key := range [2]string{"false", "true"} {
		c, err := newCriterion(sides[key])
		if err != nil {
			return out, refuse("noul %s description cannot be rendered: %v", key, err)
		}
		out[i] = c
	}
	return out, nil
}

// wireOrder reads option_order the way _check_question counts it: only true ints (not bools)
// take part, and they must be exactly 0..n-1.
func wireOrder(list []any, n int) ([]int, bool) {
	seen := make([]bool, n)
	out := make([]int, 0, len(list))
	for _, v := range list {
		num, ok := v.(json.Number)
		if !ok || strings.ContainsAny(string(num), ".eE") {
			return nil, false
		}
		i, err := strconv.Atoi(string(num))
		if err != nil || i < 0 || i >= n || seen[i] {
			return nil, false
		}
		seen[i] = true
		out = append(out, i)
	}
	return out, len(out) == n
}

func wireLabels(q *Question, labels any) *QuestionError {
	if labels == nil { // _resolve_noul_labels(None) is the default pair
		return nil
	}
	obj, ok := labels.(pyjson.Object)
	if !ok || len(obj) != 2 {
		return refuse("%s", reasonNoulLabels)
	}
	var f, t string
	var haveF, haveT bool
	for _, m := range obj {
		s, isStr := m.Value.(string)
		switch m.Key {
		case "false":
			f, haveF = s, isStr
		case "true":
			t, haveT = s, isStr
		default:
			return refuse("%s", reasonNoulLabels)
		}
	}
	f, t = unicodedata.Strip(f), unicodedata.Strip(t)
	if !haveF || !haveT || f == "" || t == "" || f == t {
		return refuse("%s", reasonNoulLabels)
	}
	q.falseLabel, q.trueLabel, q.customLabels = f, t, true
	return nil
}

// Subset returns the choice question reduced to the options at indices, in the order given: the
// criteria of laya's _subset_criteria. Labels and descriptions are kept as they were. The
// option order is not kept, because an order has one slot per option and no longer fits; callers
// that care compare OptionOrder before and after. indices must be non-empty, in range and
// distinct.
func (q Question) Subset(indices []int) (Question, error) {
	if q.typ != TypeChoice {
		return Question{}, refuse("only a choice question has options to subset, not %s", q.typ)
	}
	if len(indices) == 0 {
		return Question{}, &QuestionError{Reason: reasonChoiceEmpty}
	}
	seen := make([]bool, len(q.options))
	opts := make([]choiceOption, len(indices))
	for i, ix := range indices {
		if ix < 0 || ix >= len(q.options) || seen[ix] {
			return Question{}, refuse("subset index %d at position %d is out of range or repeated", ix, i)
		}
		seen[ix] = true
		opts[i] = q.options[ix]
	}
	return Question{typ: TypeChoice, instructions: q.instructions, options: opts}, nil
}
