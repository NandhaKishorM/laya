package agent

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// WithWindow sets the state tokens per window of a PredictLong scan; 0 or less means the room
// the questions leave. A window wider than that room is cut to it, with a warning. Predict and
// PredictBatch ignore it.
func WithWindow(tokens int) CallOption { return func(c *call) { c.window = tokens } }

// WithStride sets the tokens a PredictLong scan steps between windows; 0 or less means half the
// window. Predict and PredictBatch ignore it.
func WithStride(tokens int) CallOption { return func(c *call) { c.stride = tokens } }

// LongPrediction is PredictLong's result: a Prediction, and the warnings laya would have raised.
type LongPrediction struct {
	schema.Prediction
	// Warnings are laya's RuntimeWarning texts for a window or stride the scan had to reduce,
	// verbatim; empty when nothing was reduced. A scan that got slower says why here.
	Warnings []string
}

// PredictLong answers the questions about a state that may be longer than one sequence, as
// ONNXAgent.predict_long does with aggregate="auto". Predict truncates such a state and drops the
// rest; this splits its tokens into overlapping windows, scores every window as a state of its
// own, and combines the answers per question:
//   - a noul takes the window with the highest probability of true;
//   - a choice or a score takes the window with the highest AnswerConfidence;
//   - a tie goes to the earliest window.
//
// The deciding window's answer is returned whole, with Common.Window naming it. Its
// probabilities are that window's, not a calibrated number for the state. A state that fits the
// room the questions leave (or the explicit window) is answered as Predict would answer it, with
// Usage.Windows 1 and no Window on the answers.
//
// Across several windows Usage sums the numeric fields, so Truncated reflects TruncatedWindows,
// a count; TruncatedQuestions is the last window's list; collapsed options are merged by
// question; OutputTokens is 0. Usage.Windows is the window count.
//
// WithWindow, WithStride, WithBatchSize (windows per run; by default the scan bounds a run as
// laya's window_batch_cap does) and WithLanguage apply. A *StateError names a window, not the
// state. Errors are those of PredictBatch, and sequence.ErrNoRoom and sequence.ErrStrideGap when
// the questions leave no room or the stride leaves gaps. WithMinConfidence and
// WithMinConfidenceMap are refused with an error before any work, as Python raises TypeError for
// min_confidence; WithHooks, WithOnPredictStart and WithOnPredictEnd are refused the same way: PredictLong runs no hooks, as in
// Python, and Go has no calibration.
func (a *Agent) PredictLong(ctx context.Context, state any, qs schema.Questions, opts ...CallOption) (LongPrediction, error) {
	var c call
	for _, opt := range opts {
		if opt != nil {
			opt(&c)
		}
	}
	if c.gated {
		// Python's predict_long has no min_confidence argument (TypeError); ignoring the option
		// would return ungated answers to a caller who asked for a gate.
		return LongPrediction{}, errors.New("agent: PredictLong takes no confidence gate; use Predict or PredictBatch with WithMinConfidence")
	}
	if c.hooked {
		return LongPrediction{}, errors.New("agent: PredictLong runs no hooks; use Predict or PredictBatch with WithHooks")
	}
	if err := ctx.Err(); err != nil {
		return LongPrediction{}, err
	}
	if err := a.enter(); err != nil {
		return LongPrediction{}, err
	}
	defer a.leave()
	if isNil(state) {
		return LongPrediction{}, fmt.Errorf("agent: state is nil; pass a string, a map or a slice: %w", sequence.ErrUnsupportedState)
	}
	budget := sequence.Budget{MaxLen: a.cfg.MaxLen, HeadMaxLen: a.cfg.HeadMaxLen}
	questions := make([]schema.Question, qs.Len())
	for i := range questions {
		questions[i] = qs.At(i).Question
	}
	plan, err := sequence.PlanWindows(a.tok, questions, budget, sequence.WindowRequest{Window: c.window, Stride: c.stride})
	if err != nil {
		return LongPrediction{}, err
	}
	st, err := sequence.EncodeState(a.tok, state)
	if err != nil {
		return LongPrediction{}, err
	}
	ids := st.IDs
	fits := plan.Room
	if c.window > 0 {
		fits = plan.Window
	}
	if len(ids) <= fits {
		c.batch, c.sort = 0, false
		preds, err := a.predictStates(ctx, []any{state}, qs, c)
		if err != nil {
			if se, ok := err.(*StateError); ok {
				err = se.Err
			}
			return LongPrediction{}, err
		}
		p := preds[0]
		p.Usage.Windows = 1
		return LongPrediction{Prediction: p, Warnings: plan.Warnings}, nil
	}

	texts, starts := splitWindows(a.tok, ids, plan)
	// Python's window_batch_cap: an explicit batch size, else a bound when capping the window
	// multiplied the window count.
	c.sort = false
	c.batch, _ = sequence.BatchCap(len(texts), plan.Window, max(64, budget.MaxLen-budget.HeadMaxLen-8), c.batch)
	states := make([]any, len(texts))
	for i, t := range texts {
		states[i] = t
	}
	results, err := a.predictStates(ctx, states, qs, c)
	if err != nil {
		return LongPrediction{}, err
	}
	p := aggregate(qs, results, starts, plan.Window, len(ids))
	return LongPrediction{Prediction: p, Warnings: plan.Warnings}, nil
}

// splitWindows is predict_long's loop: each window of plan.Window tokens, decoded back to text so
// it is scored as an ordinary state, and the token it starts at; it stops at the window that
// reaches the end.
func splitWindows(tok *tokenizer.Tokenizer, ids []int32, plan sequence.WindowPlan) (texts []string, starts []int) {
	for i := 0; i < len(ids); i += plan.Stride {
		end := min(i+plan.Window, len(ids))
		texts = append(texts, tok.Decode(ids[i:end], false))
		starts = append(starts, i)
		if end >= len(ids) {
			break
		}
	}
	return texts, starts
}

// aggregate combines the per-window predictions of one scan. starts are the first token of each
// window, window the effective window, and stateTokens the tokenized state's length, which clamps
// the last window's end.
func aggregate(qs schema.Questions, results []schema.Prediction, starts []int, window, stateTokens int) schema.Prediction {
	answers := make(schema.Answers, qs.Len())
	for r := range answers {
		best, bestKey := 0, decidingKey(results[0].Answers[r].Answer)
		for j := 1; j < len(results); j++ {
			// Strictly greater, so a tie keeps the earliest window, as Python's max does.
			if key := decidingKey(results[j].Answers[r].Answer); key > bestKey {
				best, bestKey = j, key
			}
		}
		w := &schema.Window{Index: best, TokenStart: starts[best],
			TokenEnd: min(starts[best]+window, stateTokens), Count: len(results)}
		answers[r] = schema.Answered{ID: results[best].Answers[r].ID, Answer: withWindow(results[best].Answers[r].Answer, w)}
	}
	u := schema.Usage{TruncatedQuestions: []string{}, Windows: len(results)}
	for _, r := range results {
		u.InputTokens += r.Usage.InputTokens
		u.StateTokens += r.Usage.StateTokens
		u.StateTokensDropped += r.Usage.StateTokensDropped
		if r.Usage.Truncated {
			u.TruncatedWindows++
		}
		u.TruncatedQuestions = slices.Clone(r.Usage.TruncatedQuestions)
		for _, c := range r.Usage.CollapsedOptions {
			u.CollapsedOptions = mergeCollapsed(u.CollapsedOptions, c)
		}
	}
	u.Truncated = u.TruncatedWindows > 0
	if u.TruncatedQuestions == nil {
		u.TruncatedQuestions = []string{}
	}
	return schema.Prediction{Model: schema.Model, Answers: answers, Usage: u}
}

// decidingKey is what picks the window: P(true) for a noul, AnswerConfidence for the rest.
func decidingKey(a schema.Answer) float64 {
	if n, ok := a.(*schema.NoulAnswer); ok {
		return n.Noul
	}
	return a.Base().AnswerConfidence
}

// withWindow returns a copy of a that names the window that decided it.
func withWindow(a schema.Answer, w *schema.Window) schema.Answer {
	switch x := a.(type) {
	case *schema.ChoiceAnswer:
		c := *x
		c.Window = w
		return &c
	case *schema.ScoreAnswer:
		c := *x
		c.Window = w
		return &c
	case *schema.NoulAnswer:
		c := *x
		c.Window = w
		return &c
	}
	return a
}

// mergeCollapsed is Python's {**prev, **val} for one question: a later window's report replaces
// an earlier one in place, and a new question is appended.
func mergeCollapsed(all schema.CollapsedQuestions, c schema.CollapsedQuestion) schema.CollapsedQuestions {
	if i := slices.IndexFunc(all, func(x schema.CollapsedQuestion) bool { return x.ID == c.ID }); i >= 0 {
		all[i] = c
		return all
	}
	return append(all, c)
}
