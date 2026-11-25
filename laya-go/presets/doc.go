// Package presets holds ready-made question sets for common decision workflows: the port of
// laya.presets. Each set is generated from the Python reference by a golden test, word for word,
// because one changed word is a different question put to the model.
//
// A preset says which part of the state it reads by naming it in backticks ("What does the
// customer want in `message`?"); StateField reads that name back out.
//
//	qs := presets.Triage()
//	field, _ := presets.StateField(qs) // "message"
//	pred, err := a.Predict(ctx, map[string]any{field: text}, qs)
//
// # Differences from Python
//
// A Questions value is immutable, so a preset needs no copy: to drop or add a question, build a
// new set from its entries with schema.NewQuestions. Python's email_questions takes a dict of
// categories; EmailWith takes the options in the order to offer them, since a Go map has no order.
// StateField takes typed questions, which always carry instructions; Python's raises on a
// definition without them, and such a definition cannot be built here.
package presets
