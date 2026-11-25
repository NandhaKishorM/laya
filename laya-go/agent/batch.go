package agent

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/NandhaKishorM/laya/laya-go/internal/collate"
	"github.com/NandhaKishorM/laya/laya-go/internal/decode"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

// RunError reports a failed model run: the session's error, an output that does not fit the
// batch, or a logit that cannot be decoded.
type RunError struct {
	// Rows, Length and Markers are the dimensions of the batch that was run.
	Rows, Length, Markers int
	Err                   error
}

// Error names the batch and the cause.
func (e *RunError) Error() string {
	return fmt.Sprintf("agent: running %d rows of %d tokens with %d marker slots: %v", e.Rows, e.Length, e.Markers, e.Err)
}

// Unwrap returns the cause.
func (e *RunError) Unwrap() error { return e.Err }

// lookaheadBatches is how many batches of states are encoded and sorted together.
const lookaheadBatches = 8

// predictChunks is ONNXAgent._infer_batch: encode, collate, run and decode in chunks of c.batch states.
// With c.sort the states are ordered by their longest question row inside a window of eight
// chunks, so a chunk pads to a shorter length; results stay in input order.
func (a *Agent) predictChunks(ctx context.Context, states []any, qs schema.Questions, c call) ([]schema.Prediction, error) {
	chunk := len(states)
	if c.batch > 0 {
		chunk = min(c.batch, len(states))
	}
	reorder := c.sort && 1 < chunk && chunk < len(states)
	window := chunk
	if reorder {
		window = chunk * lookaheadBatches
	}
	results := make([]schema.Prediction, len(states))
	for start := 0; start < len(states); start += window {
		part := states[start:min(start+window, len(states))]
		encoded := make([][]sequence.Sequence, len(part))
		for i, state := range part {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			var err error
			if encoded[i], err = a.encode(state, qs, c); err != nil {
				return nil, &StateError{Index: start + i, Err: err}
			}
		}
		order := make([]int, len(part))
		for i := range order {
			order[i] = i
		}
		if reorder {
			longest := make([]int, len(part))
			for i, seqs := range encoded {
				for _, s := range seqs {
					longest[i] = max(longest[i], len(s.IDs))
				}
			}
			sort.SliceStable(order, func(x, y int) bool { return longest[order[x]] < longest[order[y]] })
		}
		for off := 0; off < len(order); off += chunk {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if a.closed.Load() {
				return nil, ErrClosed
			}
			idx := order[off:min(off+chunk, len(order))]
			preds, err := a.run(ctx, qs, encoded, idx, c.lang)
			if err != nil {
				return nil, err
			}
			for j, i := range idx {
				results[start+i] = preds[j]
			}
		}
	}
	return results, nil
}

// encode is one state's question rows, tokenizing the state once for all of them.
func (a *Agent) encode(state any, qs schema.Questions, c call) ([]sequence.Sequence, error) {
	st, err := sequence.EncodeState(a.tok, state)
	if err != nil {
		return nil, err
	}
	budget := sequence.Budget{MaxLen: a.cfg.MaxLen, HeadMaxLen: a.cfg.HeadMaxLen}
	if c.maxLen > 0 {
		budget.MaxLen = c.maxLen
	}
	if c.headMaxLen > 0 {
		budget.HeadMaxLen = c.headMaxLen
	}
	seqs := make([]sequence.Sequence, qs.Len())
	for i := range seqs {
		e := qs.At(i)
		if seqs[i], err = sequence.Build(a.tok, e.Question, st, budget); err != nil {
			var me *sequence.MarkersError
			if errors.As(err, &me) {
				me.ID = e.ID
			}
			return nil, err
		}
	}
	return seqs, nil
}

// run collates the states at idx into one batch, runs it and decodes it: one prediction per
// state, in idx order.
func (a *Agent) run(ctx context.Context, qs schema.Questions, encoded [][]sequence.Sequence, idx []int, lang string) ([]schema.Prediction, error) {
	rows := make([]collate.Row, 0, len(idx)*qs.Len())
	for _, i := range idx {
		for r, s := range encoded[i] {
			rows = append(rows, collate.Row{IDs: s.IDs, Markers: s.Markers, Type: qs.At(r).Question.Type()})
		}
	}
	b, err := collate.Collate(rows, a.padID)
	if err != nil {
		return nil, fmt.Errorf("agent: %w", err) // unreachable through the API: questions are valid when built
	}
	fail := func(err error) error { return &RunError{Rows: b.Rows, Length: b.Length, Markers: b.Markers, Err: err} }
	out, err := a.sess.Run(ctx, b)
	if err != nil {
		return nil, fail(err)
	}
	if err := out.Check(b); err != nil {
		return nil, fail(err)
	}
	preds := make([]schema.Prediction, len(idx))
	row := 0
	for j, i := range idx {
		seqs := encoded[i]
		answers := make(schema.Answers, len(seqs))
		for r, s := range seqs {
			at, e := row+r, qs.At(r)
			act, err := decode.ActionProbabilities(out.ActLogits[at*2 : at*2+2])
			if err != nil {
				return nil, fail(fmt.Errorf("action logits of question %s: %w", e.ID, err))
			}
			ans, err := decode.Decode(e.Question, decode.Row{
				Logits: out.Logits[at*b.Markers : (at+1)*b.Markers], K: len(s.Markers), Act: act,
			}, lang, a.temps)
			if err != nil {
				return nil, fail(fmt.Errorf("question %s: %w", e.ID, err))
			}
			answers[r] = schema.Answered{ID: e.ID, Answer: ans}
		}
		n := len(seqs)
		preds[j] = schema.Prediction{Model: schema.Model, Answers: answers,
			Usage: usage(qs, seqs, b.AttentionMask[row*b.Length:(row+n)*b.Length])}
		row += n
	}
	return preds, nil
}
