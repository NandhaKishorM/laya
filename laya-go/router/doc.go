// Package router decides which Laya checkpoint should read a request. It is the routing half of
// laya/router.py: Route and RouteBatch look at the state, the questions and the caller's hints and
// name a checkpoint and the reason, without loading or running anything. OnRoute hooks (see
// hooks.go) may observe or replace the decision. Loading and routed inference are separate
// changes.
//
// # Loading
//
// A Router built with WithAgentFactory also owns the models' lifecycle, as laya's Router does:
// Load and Lease build a checkpoint on first use, concurrent callers share one build, a build that
// fails is not cached, and at most WithMaxLoaded (default 2) stay resident, the least recently used
// leaving first. Lease is the addition Go needs: an evicted agent closes only when its last Lease
// is closed, so a prediction in flight never meets a closed session. Attach hands in an agent the
// caller built, Preload builds checkpoints up front, Unload, UnloadAll and Close free them.
// OnLoad and OnEvict hooks run after every internal lock is released.
//
// The checkpoints are English, Multilingual and TypedDecisions. The order of the evidence is
// fixed, first match wins:
//
//	ForModel > ForTask > typed-decisions workflow (opt-in) > ForLang > ForLangGuess
//	  > WithLangGuess on the Router > language and script detection > the default checkpoint
//
// The default is Multilingual, not English: text whose language cannot be identified takes it, as
// text with no letters does. WithDefault(English) restores the English default for a deployment
// that is mostly English.
//
// Reason strings are part of the contract. They quote with Python's repr, round percentages half
// to even and cut at 60 code points, so a decision here reads exactly as the Python router's.
//
// # Digests
//
// WithSHA256Digests pins checkpoints to the SHA-256 of their files, and a nested
// LAYA_SHA256_DIGESTS ({model: {artifact: digest}}) does the same from the environment, read by New;
// a flat one is left to the agent. ExpectedSHA256 merges those with a pin the factory holds for
// every checkpoint and returns the map for agent.WithExpectedSHA256, nil meaning pass none. The
// Router verifies nothing itself.
//
// A Router is safe for concurrent use. Its routing is fixed once built; only its hook list changes
// (AddHook, RemoveHook, InstallHooks).
//
// # Differences from Python
//
// A name is a string: Python's str(name) of a non-string has no counterpart. Router.register and
// the custom checkpoints are not ported, so the Router knows the three built-in names. A hint
// callable is a LangHint returning "" to abstain. explain=True is not ported: lang.Analysis has no
// diagnostic field. Loading needs an AgentFactory: there is no hub download. WithMaxLoaded below 1
// is an error, where Python clamps it to 1. Attach of a name the Router does not know is an error,
// where Python registers it. Load returns an agent that closes on eviction; Lease keeps it open.
// Attach and Preload raise MaxLoaded as in Python. Hooks run concurrently as in Python; WithSerialHooks
// runs one at a time (and then a hook must not call Route on its own Router). A hook failure
// names the hook's Go type. Digests: SHA256Digests is a copy, where Python's sha256_digests is a
// public dict that can be edited after construction; a non-string digest in a nested
// LAYA_SHA256_DIGESTS is an error from New, where Python hands it to the agent; two spellings of one
// checkpoint in one WithSHA256Digests map are an error from New (Python keeps whichever comes last),
// and several unknown names report the first in name order, where Python uses
// the dict's order. Not
// ported, because the factory owns them: agent_kwargs and its reserved-name check (a factory passes
// typed agent options), revision and revisions (no hub download), preload=True (New has no context,
// call Preload), per-checkpoint lang_temperatures (a factory passes agent.WithLangTemperatures).
package router
