package schema

import (
	"fmt"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// Entry is one question of a request, with its id.
type Entry struct {
	ID       string
	Question Question
}

// Ask pairs a question with its id, for NewQuestions.
func Ask(id string, q Question) Entry { return Entry{ID: id, Question: q} }

// Questions is the ordered set of questions one request asks, keyed by id: the Go form of
// laya's questions dict. Answers come back in this order. The zero value is an empty set.
type Questions struct {
	entries []Entry
	index   map[string]int
}

func checkID(id string) *QuestionError {
	if unicodedata.IsBlank(id) {
		return &QuestionError{Reason: "question id must be a non-empty string, got " + pyjson.Repr(id)}
	}
	return nil
}

// NewQuestions builds a set from entries in the order given. It refuses a blank id, a repeated
// id and a zero Question.
func NewQuestions(entries ...Entry) (Questions, error) {
	qs := Questions{entries: make([]Entry, 0, len(entries)), index: make(map[string]int, len(entries))}
	for _, e := range entries {
		if err := checkID(e.ID); err != nil {
			return Questions{}, err
		}
		if !e.Question.typ.Valid() {
			return Questions{}, &QuestionError{ID: e.ID,
				Reason: "zero Question; build one with Choice, ChoiceOf, Score, Noul or NoulWith"}
		}
		if _, dup := qs.index[e.ID]; dup {
			return Questions{}, &QuestionError{ID: e.ID, Reason: repeatedID}
		}
		qs.index[e.ID] = len(qs.entries)
		qs.entries = append(qs.entries, e)
	}
	return qs, nil
}

const repeatedID = "id is given twice; answers are keyed by id, so every question needs its own"

// ParseQuestions reads a questions object in the Python wire form, {"id": {"type": ...}, ...},
// keeping document order and validating every question as laya's _check_question does. A repeated
// id keeps its first position and takes its last value, as Python's json.loads resolves it, and
// that value is then validated like any other (a null one is "definition must be a dict, got
// NoneType"). NaN and Infinity are accepted as Python accepts them.
func ParseQuestions(data []byte) (Questions, error) {
	v, err := pyvalue.Parse(data)
	if err != nil {
		return Questions{}, fmt.Errorf("schema: questions: %w", err)
	}
	members, ok := v.(pyjson.Object)
	if !ok {
		return Questions{}, fmt.Errorf("schema: questions must be a JSON object, got %s", pyvalue.TypeName(v))
	}
	qs := Questions{entries: make([]Entry, 0, len(members)), index: make(map[string]int, len(members))}
	for _, m := range members {
		id := m.Key.(string)
		if err := checkID(id); err != nil {
			return Questions{}, err
		}
		q, qerr := fromWire(m.Value)
		if qerr != nil {
			qerr.ID = id
			return Questions{}, qerr
		}
		qs.index[id] = len(qs.entries)
		qs.entries = append(qs.entries, Entry{ID: id, Question: q})
	}
	return qs, nil
}

// Len is the number of questions.
func (qs Questions) Len() int { return len(qs.entries) }

// At is the i-th question and its id, in request order.
func (qs Questions) At(i int) Entry { return qs.entries[i] }

// Get is the question with id, if the set has one.
func (qs Questions) Get(id string) (Question, bool) {
	i, ok := qs.index[id]
	if !ok {
		return Question{}, false
	}
	return qs.entries[i].Question, true
}

// IDs are the question ids in request order.
func (qs Questions) IDs() []string {
	out := make([]string, len(qs.entries))
	for i, e := range qs.entries {
		out[i] = e.ID
	}
	return out
}

// List is the questions in request order, without their ids.
func (qs Questions) List() []Question {
	out := make([]Question, len(qs.entries))
	for i, e := range qs.entries {
		out[i] = e.Question
	}
	return out
}

// MarshalJSON writes the set as one object in request order.
func (qs Questions) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteByte('{')
	for i, e := range qs.entries {
		if i > 0 {
			b.WriteByte(',')
		}
		id, err := pyjson.Dumps(e.ID)
		if err != nil {
			return nil, err
		}
		q, err := e.Question.MarshalJSON()
		if err != nil {
			return nil, err
		}
		b.WriteString(id + ":")
		b.Write(q)
	}
	b.WriteByte('}')
	return []byte(b.String()), nil
}

// UnmarshalJSON is ParseQuestions.
func (qs *Questions) UnmarshalJSON(data []byte) error {
	parsed, err := ParseQuestions(data)
	if err != nil {
		return err
	}
	*qs = parsed
	return nil
}
