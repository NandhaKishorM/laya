package shortlist

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

// DefaultK is the number of labels kept when the caller does not choose: DEFAULT_SHORTLIST_K.
const DefaultK = 20

var (
	// ErrInvalidArgument is wrapped by every refusal of a caller's argument: k, a cache size, a
	// question that is not a choice, a nil predictor or embedder.
	ErrInvalidArgument = errors.New("shortlist: invalid argument")
	// ErrEmbedding is wrapped when an Embedder returns something that is not one row per text.
	ErrEmbedding = errors.New("shortlist: invalid embedder output")
)

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidArgument, fmt.Sprintf(format, args...))
}

// Embedder maps texts to one vector each, all of one width. It is called with the query text first
// and then one string per option, in option order. A NaN or infinite component is read as 0.
type Embedder func(ctx context.Context, texts []string) ([][]float64, error)

// Predictor is what Predict calls once: *agent.Agent.
type Predictor interface {
	Predict(ctx context.Context, state any, qs schema.Questions, opts ...agent.CallOption) (schema.Prediction, error)
}

// Ranking is what shortlisting did to one choice question.
type Ranking struct {
	// Labels are the kept labels: in rank order, or in option order when Passthrough is set. It
	// always holds min(K, N) labels.
	Labels []schema.Label
	// Indices are the positions of Labels among the question's options.
	Indices []int
	// Scores is the signed cosine of each kept label, negative included, or nil when nothing was
	// dropped (Passthrough).
	Scores []float64
	// K is the cap the caller asked for; N is the question's option count before narrowing.
	K, N int
	// Passthrough is set when K >= N: no ranking ran and the Embedder was not called.
	Passthrough bool
}

// Ranked is the Ranking of one question.
type Ranked struct {
	ID string
	Ranking
}

// Shortlisted is the result of Predict.
type Shortlisted struct {
	// Prediction is the model's answer to the reduced questions. Probabilities on a shortlisted
	// choice are over the kept labels only.
	Prediction schema.Prediction
	// Rankings has an entry for each choice question, in question order; a question that is not a
	// choice has none, since no ranking ran for it.
	Rankings []Ranked
	// Warnings has one entry for each question whose option_order was dropped, because an order
	// has one slot per option and no longer fits the kept labels.
	Warnings []string
}

// Ranking returns the ranking of question id.
func (s Shortlisted) Ranking(id string) (Ranking, bool) {
	for _, r := range s.Rankings {
		if r.ID == id {
			return r.Ranking, true
		}
	}
	return Ranking{}, false
}

// QueryText is the text embedded for the state: the instructions and a newline before the
// serialized state, or the state alone when instructions is empty.
func QueryText(state any, instructions string) (string, error) {
	body, err := sequence.SerializeState(state)
	if err != nil {
		return "", err
	}
	if instructions == "" {
		return body, nil
	}
	return instructions + "\n" + body, nil
}

// Choice returns the top-k labels of choice question q for state. See Rank.
func Choice(ctx context.Context, state any, q schema.Question, embed Embedder, k int) ([]schema.Label, error) {
	r, err := Rank(ctx, state, q, embed, k)
	return r.Labels, err
}

// Rank ranks the options of choice question q against state and keeps the top k. The query is
// q's instructions and the state.
//
// When k is at least the number of options, every label is returned in option order and embed is
// not called. Otherwise embed is called once.
func Rank(ctx context.Context, state any, q schema.Question, embed Embedder, k int) (Ranking, error) {
	return RankWithInstructions(ctx, state, q, embed, k, q.Instructions())
}

// RankWithInstructions is Rank with the instruction text of the query given: "" embeds the state
// alone. The option texts are always q's own.
func RankWithInstructions(ctx context.Context, state any, q schema.Question, embed Embedder, k int, instructions string) (Ranking, error) {
	if q.Type() != schema.TypeChoice {
		return Ranking{}, invalid("only a choice question has labels to shortlist, not %q", q.Type())
	}
	if err := checkK(k); err != nil {
		return Ranking{}, err
	}
	labels := q.AnswerLabels()
	n := len(labels)
	if k >= n {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		return Ranking{Labels: labels, Indices: idx, K: k, N: n, Passthrough: true}, nil
	}
	query, err := QueryText(state, instructions)
	if err != nil {
		return Ranking{}, err
	}
	texts := append([]string{query}, q.RenderedOptions()...)
	matrix, err := embeddings(ctx, embed, texts)
	if err != nil {
		return Ranking{}, err
	}
	sims := cosine(matrix[0], matrix[1:])
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	// A stable sort by descending score, NaN last: np.argsort(-sims, kind="mergesort").
	sort.SliceStable(order, func(a, b int) bool {
		x, y := sims[order[a]], sims[order[b]]
		if math.IsNaN(y) {
			return !math.IsNaN(x)
		}
		return x > y // false for NaN x
	})
	order = order[:k]
	out := Ranking{Labels: make([]schema.Label, k), Indices: order, Scores: make([]float64, k), K: k, N: n}
	for i, ix := range order {
		out.Labels[i] = labels[ix]
		out.Scores[i] = sims[ix]
	}
	return out, nil
}

// Predict shortlists each choice question of qs, then calls p.Predict once on the reduced set.
//
// Non-choice questions are forwarded unchanged. A choice with at most k labels is forwarded
// unchanged and does not call embed. qs is not changed. A narrowed choice that carried an option
// order loses it, with one entry in Warnings: remapping the order onto the kept labels would move
// options between slots silently. The call options are forwarded to p.Predict.
func Predict(ctx context.Context, p Predictor, state any, qs schema.Questions, embed Embedder, k int, opts ...agent.CallOption) (Shortlisted, error) {
	if p == nil {
		return Shortlisted{}, invalid("predictor must not be nil")
	}
	if err := checkK(k); err != nil {
		return Shortlisted{}, err
	}
	var out Shortlisted
	entries := make([]schema.Entry, 0, qs.Len())
	for i := 0; i < qs.Len(); i++ {
		e := qs.At(i)
		if e.Question.Type() != schema.TypeChoice {
			entries = append(entries, e)
			continue
		}
		r, err := Rank(ctx, state, e.Question, embed, k)
		if err != nil {
			return Shortlisted{}, fmt.Errorf("question %s: %w", pyjson.Repr(e.ID), err)
		}
		out.Rankings = append(out.Rankings, Ranked{ID: e.ID, Ranking: r})
		if r.Passthrough {
			entries = append(entries, e)
			continue
		}
		narrowed, err := e.Question.Subset(r.Indices)
		if err != nil {
			return Shortlisted{}, fmt.Errorf("question %s: %w", pyjson.Repr(e.ID), err)
		}
		if order := e.Question.OptionOrder(); order != nil {
			out.Warnings = append(out.Warnings, fmt.Sprintf(
				"laya.shortlist: question %s: 'option_order' (%d slots) was dropped because the label "+
					"set was narrowed to %d; the order no longer fits the kept options and remapping "+
					"it onto them would move options between slots.", pyjson.Repr(e.ID), len(order), len(r.Labels)))
		}
		entries = append(entries, schema.Ask(e.ID, narrowed))
	}
	reduced, err := schema.NewQuestions(entries...)
	if err != nil {
		return Shortlisted{}, err
	}
	out.Prediction, err = p.Predict(ctx, state, reduced, opts...)
	if err != nil {
		return Shortlisted{}, err
	}
	return out, nil
}

func checkK(k int) error {
	if k < 1 {
		return invalid("k must be a positive integer, got %d", k)
	}
	return nil
}

// checkRows validates one embedder reply and returns it as a fresh matrix with every non-finite
// component read as 0: the checks of laya's _embeddings.
func checkRows(raw [][]float64, want int) ([][]float64, error) {
	if len(raw) != want || want == 0 {
		return nil, fmt.Errorf("%w: embedder must return %d rows, got %d", ErrEmbedding, want, len(raw))
	}
	width := len(raw[0])
	if width < 1 {
		return nil, fmt.Errorf("%w: embedder returned an empty row at index 0", ErrEmbedding)
	}
	out := make([][]float64, want)
	for i, row := range raw {
		if len(row) != width {
			return nil, fmt.Errorf("%w: embedder returned rows of different widths: %d then %d at index %d",
				ErrEmbedding, width, len(row), i)
		}
		out[i] = make([]float64, width)
		for j, v := range row {
			if !math.IsNaN(v) && !math.IsInf(v, 0) {
				out[i][j] = v
			}
		}
	}
	return out, nil
}

func embeddings(ctx context.Context, embed Embedder, texts []string) ([][]float64, error) {
	if embed == nil {
		return nil, invalid("embedder must not be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	raw, err := embed(ctx, append([]string(nil), texts...))
	if err != nil {
		return nil, fmt.Errorf("embedder: %w", err)
	}
	return checkRows(raw, len(texts))
}

// cosine is the clipped cosine of query against each doc, 0 for a zero vector, as laya's _cosine.
// A pair whose norms overflow gives NaN, as it does there.
func cosine(query []float64, docs [][]float64) []float64 {
	sims := make([]float64, len(docs))
	qn := norm(query)
	if qn == 0 {
		return sims
	}
	for i, d := range docs {
		denom := norm(d) * qn
		if !(denom > 0) {
			continue
		}
		var dot float64
		for j := range query {
			dot += float64(d[j] * query[j]) // float64() blocks a fused multiply-add (arm64)
		}
		v := dot / denom
		if !math.IsNaN(v) {
			v = math.Max(-1, math.Min(1, v))
		}
		sims[i] = v
	}
	return sims
}

func norm(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += float64(x * x)
	}
	return math.Sqrt(s)
}
