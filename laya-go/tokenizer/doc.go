// Package tokenizer is a pure-Go reader and runner for Hugging Face tokenizer.json files. It
// reproduces the ids the reference `tokenizers` crate gives for the BPE tokenizers laya ships:
// the normalizer, the pre-tokenizer, the added-token splitter, BPE with its byte fallback, and
// the decoder chain.
//
// Anything the loader does not implement is refused with an error wrapping ErrUnsupported
// rather than approximated, because an approximated tokenizer changes ids in a way no
// answer-level test localises. A loaded Tokenizer is immutable and safe for concurrent use.
//
// The Unicode behaviour that decides ids (letter and number classes, word characters beside a
// single_word token, lowercase, and normalization) is compiled in from the reference, see
// scripts/gen_tables.py, so ids do not depend on the Go toolchain's own Unicode release.
package tokenizer
