// Package agent answers typed questions about a state with a laya checkpoint: the port of
// ONNXAgent.predict and predict_batch. It is pure Go. The model runs behind an infer.Session,
// which a native backend supplies, so this package links no runtime.
//
//	a, err := agent.Open(os.DirFS("checkpoints/english"), session)
//	pred, err := a.Predict(ctx, "I was charged twice", questions)
//
// A call encodes the state once, builds one row per question, collates the rows of up to
// WithBatchSize states into one batch, runs it, and decodes the logits into the typed answers
// and the usage report of laya's predict().
//
// # Order and batching
//
// Answers follow the question order and results follow the input order, whatever the batch size
// or sorting. Batch shape does change padding, and with it a probability's last digits, so
// results are compared with a tolerance across shapes, not byte for byte.
//
// # Differences from Python
//
// With no questions, Python writes a usage of two keys (input_tokens, output_tokens); this package
// writes six: input_tokens, output_tokens, state_tokens, state_tokens_dropped, truncated and
// truncated_questions. The seventh, options, is omitted when empty.
//
// A nil slice or map is an empty list or dict, as in Python; a nil state is refused, as there.
//
// max_len and head_max_len are checked at load and must be positive integers; Python reads them
// with cfg.get and checks nothing until a call uses them.
//
// A temperature that is not a list of exactly 3 numbers is refused at Open. Python's ONNXAgent
// loads a list of the wrong length, a string or a dict and fails later or falls back (a scalar
// such as 1.5 raises a TypeError at load); this package refuses all of them.
//
// A NaN or infinite logit is refused, including a -Inf for a real option, where Python would
// answer probability 0 for that option. A model that emits one is broken.
//
// # Scope
//
// Shipped: hooks, the min_confidence gate, per-language temperatures (WithLangTemperatures,
// selected by WithLanguage) and digest checks (WithExpectedSHA256, LAYA_SHA256_DIGESTS) and the calibration
// payload (WithCalibration, CalibrationPayload; the fitters are not here). Not here yet: the router. PredictLong ports predict_long for aggregate="auto"
// and applies neither the gate nor hooks. Without overrides WithLanguage does nothing.
package agent
