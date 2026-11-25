// Package infer is the seam between laya's agent and a model runtime: the five tensors the
// exported graph takes, the two it returns, and the Session that runs one batch of them.
//
// It is pure Go and imports nothing outside the standard library, so the agent, the router and
// the tests link no runtime. A native backend implements Session in its own module; a test
// implements it with a function.
//
// The tensor layout is the graph's contract: every slice is flattened row-major, one row per
// (state, question) pair, and Markers is never below 2 because the exported head takes topk(2).
package infer
