// Package shortlist narrows a high-cardinality choice question before the model sees it: the port
// of laya.shortlist's embedding shortlist.
//
// Choice options share one head_max_len budget, so a large label set leaves only a few tokens per
// label. Predict embeds the state and each option with a caller-supplied Embedder, keeps the top
// k, and runs a single Predict on the reduced questions. It adds no second model pass.
//
//	out, err := shortlist.Predict(ctx, a, state, qs, embed, 20)
//	pred, ranked := out.Prediction, out.Rankings
//
// Ranking is the signed cosine of the vectors the Embedder returns, a stable sort: a tie keeps the
// earlier label, and a label that scores 0 outranks an earlier one that scored negative.
//
// # Scope
//
// Ported: Rank, Choice, Predict, QueryText and Cached (cached_embed_fn). Not ported:
// predict_tournament, which is a separate feature, and embed_fn_from_agent, which needs the
// checkpoint's encoder; the ONNX session in this port exposes only the decision head.
//
// # Differences from Python
//
// The reference takes dict question definitions and forwards extra keywords; here a question is a
// schema.Question and call options are agent.CallOption. Python's RuntimeWarning for a dropped
// option_order is a string in Shortlisted.Warnings, with the same wording. An Embedder returns
// [][]float64, so Python's "not an array" and "wrong number of dimensions" refusals reduce to a
// wrong row count, an empty row, or rows of different widths. Cached stores rows as float32, as
// Python does.
package shortlist
