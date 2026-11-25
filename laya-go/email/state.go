package email

import (
	"fmt"
	"slices"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
	"github.com/NandhaKishorM/laya/laya-go/lang"
)

type stateConfig struct {
	sender   string
	clean    bool
	maxChars int
	extra    []lang.Member
}

// Option adjusts State.
type Option func(*stateConfig)

// WithSender stores the sender under "from". An empty sender adds no field.
func WithSender(sender string) Option { return func(c *stateConfig) { c.sender = sender } }

// WithoutCleaning keeps the body as given, whole, instead of running it through CleanBody.
func WithoutCleaning() Option { return func(c *stateConfig) { c.clean = false } }

// WithMaxChars sets the budget CleanBody cuts the body to (DefaultMaxChars unless set). It is a
// slice budget, so 0 gives the model a subject and an empty body and a negative value cuts from
// the end, which is where a late request lives; raise it for a long message, and read CleanBody
// before passing anything below the body's length. Ignored with WithoutCleaning.
func WithMaxChars(n int) Option { return func(c *stateConfig) { c.maxChars = n } }

// WithField adds a field the model reads, after the subject, body and sender. A nil value is
// dropped rather than stored (untyped nil only; a typed nil pointer is kept, as a value), so a
// state with a nil thread id is the state without one: a stored
// nil would put "thread_id": null in the prompt. A key already present is replaced in place.
func WithField(key string, value any) Option {
	return func(c *stateConfig) { c.extra = append(c.extra, lang.Member{Key: key, Value: value}) }
}

// reserved are the names of the reference's parameters. Python cannot put them in a state as
// keyword arguments, so State refuses them rather than letting a typo become an input mutation.
var reserved = []string{"subject", "body", "sender", "clean", "max_chars"}

// State builds the state for classifying a message: the trimmed subject, the body (cleaned unless
// WithoutCleaning), the sender under "from" when there is one, and any fields added with
// WithField, in the order given. It fails only when a field is named like one of the reference's
// parameters.
func State(subject, body string, opts ...Option) (lang.Object, error) {
	c := stateConfig{clean: true, maxChars: DefaultMaxChars}
	for _, opt := range opts {
		opt(&c)
	}
	for _, m := range c.extra {
		if slices.Contains(reserved, m.Key) {
			return nil, fmt.Errorf("email: field %q names a parameter of State; pass it as an option instead", m.Key)
		}
	}
	if c.clean {
		body = CleanBody(body, c.maxChars)
	}
	state := lang.Object{
		{Key: "subject", Value: unicodedata.Strip(subject)},
		{Key: "body", Value: body},
	}
	if c.sender != "" {
		state = append(state, lang.Member{Key: "from", Value: c.sender})
	}
	for _, m := range c.extra {
		if m.Value == nil {
			continue
		}
		if i := slices.IndexFunc(state, func(x lang.Member) bool { return x.Key == m.Key }); i >= 0 {
			state[i].Value = m.Value
		} else {
			state = append(state, m)
		}
	}
	return state, nil
}
