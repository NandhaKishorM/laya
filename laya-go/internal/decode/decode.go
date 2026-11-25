// Package decode turns the head's logit rows into typed answers: the port of
// Agent._decode_answers in laya/agent.py, with the temperature and rounding rules it applies.
//
// The steps, in order: slice the row to the question's option count k (the row may be wider),
// divide by the temperature its bucket names, softmax with the maximum subtracted, put the row
// back in label order (undo OptionOrder), derive the type's fields, and round every number to
// four decimals as Python's round(x, 4) does.
//
// Python divides the float32 logits by the temperature in float32, and so does Decode (the quotient
// is rounded to float32 first). Everything after that is float64 here, where Python computes exp
// and the sum in float32: the line laya-ts, laya-dotnet and laya-java hold. The choice is the first
// maximum of those probabilities, so an exact tie falls to the lower index as numpy.argmax does.
// Between options that are not tied but within about 1e-6 of each other Python's own choice is
// not stable: it depends on the last bit of numpy's float32 exp, a SIMD routine that differs
// between CPUs. The fuzz in the golden (go_extra.json, answers_fuzz) counts those rows.
package decode

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// ErrShape is wrapped when a logit row is narrower than the option count, or the action row is
// empty.
var ErrShape = errors.New("logits do not fit the question")

// Row is one question's slice of the graph's outputs.
type Row struct {
	// Logits is the head's option row, at least K wide; padding columns past K are ignored.
	Logits []float32
	// K is how many option markers the question's sequence kept.
	K int
	// Act is the row of ActionProbabilities; Act[0] is reported as the act probability.
	Act []float32
}

// Decode is one question's answer. lang is the request's language tag ("" for none); its
// primary subtag, lower-cased, selects a language override in t. A binning map, when t has one,
// replaces AnswerConfidence unless a language override applied.
func Decode(q schema.Question, row Row, lang string, t *Temperatures) (schema.Answer, error) {
	typ := q.Type()
	if !typ.Valid() {
		return nil, fmt.Errorf("%w: the zero Question has no answer", ErrShape)
	}
	k := row.K
	if k != q.NumOptions() {
		return nil, fmt.Errorf("%w: %d option markers for a question of %d options", ErrShape, k, q.NumOptions())
	}
	if len(row.Logits) < k {
		return nil, fmt.Errorf("%w: %d logits for %d options", ErrShape, len(row.Logits), k)
	}
	if len(row.Act) == 0 {
		return nil, fmt.Errorf("%w: no action probability", ErrShape)
	}
	for _, x := range row.Logits[:k] {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return nil, fmt.Errorf("%w: non-finite logit %v", ErrShape, x)
		}
	}
	if math.IsNaN(float64(row.Act[0])) || math.IsInf(float64(row.Act[0]), 0) {
		return nil, fmt.Errorf("%w: non-finite action probability %v", ErrShape, row.Act[0])
	}
	temperature, override := t.For(typ, k, lang)

	// softmax(logits[:k] / temperature), the maximum subtracted so a large logit cannot overflow.
	z := make([]float32, k)
	peak := float32(math.Inf(-1))
	for i := range z {
		z[i] = row.Logits[i] / float32(temperature) // float32 division, as numpy does it
		if math.IsInf(float64(z[i]), 1) {
			// a finite logit can overflow float32 once divided, and inf - inf is NaN in every field
			return nil, fmt.Errorf("%w: logit %v over temperature %v is not finite", ErrShape, row.Logits[i], temperature)
		}
		peak = max(peak, z[i])
	}
	p := make([]float64, k)
	sum := 0.0
	for i := range z {
		p[i] = math.Exp(float64(z[i]) - float64(peak))
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	// The row is in slot order; everything below indexes by option.
	if order := q.OptionOrder(); len(order) == len(p) {
		canonical := make([]float64, k)
		for slot, option := range order {
			canonical[option] = p[slot]
		}
		p = canonical
	}

	common := schema.Common{
		AnswerConfidence: Round4(binned(t, typ, k, answerConfidence(p), override)),
		Action:           schema.Action{ActProbability: Round4(float64(row.Act[0]))},
	}
	switch typ {
	case schema.TypeChoice:
		common.Confidence = Round4(entropyConfidence(p))
		labels := q.AnswerLabels()
		best := 0
		for i := range p {
			if p[i] > p[best] { // strict: the first maximum wins, as numpy.argmax breaks a tie
				best = i
			}
		}
		return &schema.ChoiceAnswer{Choice: labels[best], Probabilities: probabilities(labels, p), Common: common}, nil
	case schema.TypeScore:
		common.Confidence = Round4(entropyConfidence(p))
		expected := 0.0
		for i, v := range p {
			expected += float64(i) * v
		}
		labels := q.AnswerLabels()
		return &schema.ScoreAnswer{Score: Round4(expected), Legend: q.Legend(),
			Probabilities: probabilities(labels, p), Common: common}, nil
	}
	if k < 2 {
		return nil, fmt.Errorf("%w: a noul question needs two options", ErrShape)
	}
	yes := p[1]
	common.Confidence = Round4(math.Max(yes, 1-yes))
	return &schema.NoulAnswer{Noul: Round4(yes), Common: common}, nil
}

func probabilities(labels []schema.Label, p []float64) schema.Probabilities {
	out := make(schema.Probabilities, len(p))
	for i, v := range p {
		out[i] = schema.Probability{Label: labels[i], P: Round4(v)}
	}
	return out
}

// answerConfidence is max(p), clipped to [0, 1]: the mass on the reported answer.
func answerConfidence(p []float64) float64 {
	if len(p) < 1 {
		return 1
	}
	top := p[0]
	for _, v := range p[1:] {
		top = math.Max(top, v)
	}
	return math.Min(1, math.Max(0, top))
}

// entropyConfidence is 1 - H(p)/log(k), clipped to [0, 1]; 1 for fewer than two options. The
// probabilities are clipped to [1e-12, 1] inside the log only.
func entropyConfidence(p []float64) float64 {
	if len(p) < 2 {
		return 1
	}
	h := 0.0
	for _, v := range p {
		h -= v * math.Log(math.Min(1, math.Max(1e-12, v)))
	}
	return math.Min(1, math.Max(0, 1-h/math.Log(float64(len(p)))))
}

// binned applies the histogram-binning map, but only without a language override.
func binned(t *Temperatures, typ schema.Type, k int, confidence float64, override bool) float64 {
	if t == nil || override {
		return confidence
	}
	return ApplyBinning(confidence, Bucket(typ, k), t.Binning)
}

// ApplyBinning is laya's apply_binning_map: confidence recalibrated through the bucket's bins, or
// unchanged when the map has no entry for the bucket.
func ApplyBinning(confidence float64, bucket string, m map[string]Bins) float64 {
	bins, ok := m[bucket]
	if !ok || len(bins.Values) == 0 {
		return confidence
	}
	n := len(bins.Values)
	return bins.Values[min(n-1, max(0, int(confidence*float64(n))))]
}

// ActionProbabilities is a softmax over one row of the graph's act_logits. The decoder is given
// probabilities, never the raw logit, which is unbounded and only looks like a probability.
func ActionProbabilities(actLogits []float32) ([]float32, error) {
	if len(actLogits) == 0 {
		return nil, fmt.Errorf("%w: an empty action row", ErrShape)
	}
	peak := math.Inf(-1)
	for _, x := range actLogits {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return nil, fmt.Errorf("%w: non-finite action logit %s", ErrShape, strconv.FormatFloat(float64(x), 'g', -1, 32))
		}
		peak = math.Max(peak, float64(x))
	}
	exps := make([]float64, len(actLogits))
	sum := 0.0
	for i, x := range actLogits {
		exps[i] = math.Exp(float64(x) - peak)
		sum += exps[i]
	}
	out := make([]float32, len(actLogits))
	for i, e := range exps {
		out[i] = float32(e / sum)
	}
	return out, nil
}
