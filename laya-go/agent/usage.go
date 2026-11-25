package agent

import (
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
)

// usage is one state's report, from its question rows (seqs, in question order) and the slice of
// the attention mask those rows occupy in the batch.
func usage(qs schema.Questions, seqs []sequence.Sequence, mask []int64) schema.Usage {
	var u schema.Usage
	for _, m := range mask {
		u.InputTokens += int(m)
	}
	u.StateTokens = seqs[0].Truncation.StateTokens
	u.TruncatedQuestions = []string{}
	for i, s := range seqs {
		id := qs.At(i).ID
		// The questions share one state, not one head budget, so the worst case is reported.
		u.StateTokensDropped = max(u.StateTokensDropped, s.Truncation.StateTokensDropped)
		if s.Truncation.Truncated {
			u.TruncatedQuestions = append(u.TruncatedQuestions, id)
		}
		if s.Stats.Collapsed() {
			u.CollapsedOptions = append(u.CollapsedOptions, schema.CollapsedQuestion{ID: id,
				CollapsedOptions: schema.CollapsedOptions{Total: s.Stats.Options,
					Distinct: s.Stats.OptionsDistinct, TokensPerOption: s.Stats.TokensPerOption}})
		}
	}
	u.Truncated = u.StateTokensDropped > 0
	return u
}
