// Package sequence composes a question and its state into the token sequence the encoder
// reads, and sizes the windows a long state is scanned in. It is the port of build_sequence,
// build_head, state_room, window_budget and window_batch_cap in laya/common.py, and it is pure
// Go: a caller can measure what a request will drop before any model is loaded.
//
// A sequence is
//
//	[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
//
// and the position of each option's [MASK] is that option's marker: the head reads one logit
// there. A marker that is off by one still yields a plausible answer, which is why the goldens
// record markers rather than inferring them from ids.
//
// # Budgets
//
// Two rules decide what survives when a question does not fit, step for step as common.py:
//
//   - Options yield nothing and the instruction yields everything. Each option is capped at 48
//     tokens by the tokenizer; if the options then leave the head fewer than 16 tokens, every
//     option is cut to max(4, (headMaxLen-16)/k) tokens (Stats.TokensPerOption). The instruction
//     takes what is left, never fewer than 8 tokens.
//   - The state takes only the room the head leaves, maxLen - len(head) - 1, silently. The
//     start is kept, or the end for a list state (a conversation's turns). Truncation reports
//     the cut, because a caller cannot reconstruct it: the budget is in tokens and moves per
//     checkpoint and per question.
//
// The mask token's text is replaced by a space in the instruction, every option and the state,
// so caller text cannot plant option markers.
//
// # Windows
//
// A state longer than one sequence is scanned in overlapping token windows (PlanWindows). The
// window is capped at the smallest StateRoom of the questions asked, since a wider window would
// be re-truncated on its way to the model while its reported span says it was read. Every clamp
// is a decision the caller cannot see, so WindowPlan.Warnings carries laya's RuntimeWarning text
// verbatim; this package does not log.
//
// Everything here is safe for concurrent use.
package sequence
