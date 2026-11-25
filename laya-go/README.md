# laya-go

A pure-Go port of [laya](../README.md). This module is standard library only: no cgo, no third-party
modules. Running the model needs the nested `onnx` module, which uses cgo.

The foundation (language detection, questions, sequences, the tokenizer) and the predict path
(`infer`, `agent`) are here, and the `onnx` module runs the graph. The router's pure routing half
(`router`) and its `on_route` hooks are here; loading and routed inference come in later changes.

| Package | What it is |
| --- | --- |
| `lang` | Script and language detection, the port of `laya/lang.py`: `DetectScript`, `ScriptProfile`, `LatinProfileOf`, `GuessLatinLanguage`, `Analyze`, `DetectLanguage`, `IsEnglish`. |
| `presets` | Ready-made question sets (`Triage`, `Email`, `Guard`, `Moderation`, `Router`), the port of `laya/presets.py`, and `StateField`, which reads the field a set asks about. |
| `shortlist` | Embedding shortlist for high-cardinality choice questions, the port of `laya/shortlist.py`: `Rank`, `Predict`, `Cached`. `predict_tournament` is not ported. |

| `email` | Mail-body cleaning and state building, the port of `laya/email.py`: `CleanBody` strips quoted history, signatures, device footers and disclaimers (English, Portuguese, Spanish, French); `State` builds the state, with `WithSender`, `WithoutCleaning`, `WithMaxChars` and `WithField`. |
| `schema` | Questions (`Choice`, `ChoiceOf`, `Score`, `Noul`, `NoulWith`, `Questions`) and results (`Answer`, `Prediction`, `Usage`) every other package shares. Questions are refused with laya's own wording and read or written in the Python wire form. |
| `sequence` | `build_sequence`, `build_head`, `state_room`, `window_budget` and `window_batch_cap`: `EncodeState`, `Build`, `StateRoom`, `PlanWindows`, `BatchCap`. |
| `infer` | The seam to a model runtime: `Batch` (the graph's five inputs), `Output` (its two), `Session`, and the shape checks `Batch.Validate` and `Output.Check`. |
| `agent` | `ONNXAgent.predict` and `predict_batch` over an `infer.Session`: `Open`, `Predict`, `PredictBatch`, `LoadConfig`, `WithCalibration` (installs a `calibrate` payload at `Open`) and `CalibrationPayload` (writes one), with batch size, sort-by-length and context cancellation between runs. |
| `router` | Which checkpoint reads a request, the port of the routing half of `laya/router.py`: `New`, `Route`, `RouteBatch`, `NormaliseName`, `EnglishFromCode`, `MatchTypedDecisionsWorkflow`, `QuestionSchema`. Decisions carry laya's reason strings. `OnRoute` hooks (`AddHook`, `WithInstalledHooks`, `ForHooks`) may observe or replace a decision, under the agent's hook dispatcher. With `WithAgentFactory` it also loads: `Load`, `Lease` (an evicted agent closes after its last lease), `Attach`, `Preload`, `Unload`, `UnloadAll`, `Close`, LRU `WithMaxLoaded`, single-flight builds, `OnLoad`/`OnEvict` hooks. `WithSHA256Digests`, a nested `LAYA_SHA256_DIGESTS` and `ExpectedSHA256` give each checkpoint its own digest pins. |
| `onnx` | A nested module (cgo, its own `go.mod`): `onnx.Open` runs a fused `laya.onnx` on ONNX Runtime and implements `infer.Session`, for `agent.Open`. Needs an ONNX Runtime 1.29+ shared library; `CGO_ENABLED=0` builds and `Open` returns `ErrNoCGO`. |
| `tokenizer` | A reader and runner for Hugging Face `tokenizer.json` (BPE): normalizer, pre-tokenizer, added tokens, byte fallback, decoder. |
| `internal/unicodedata` | The character predicates `laya/lang.py` leans on, recorded from CPython. |
| `internal/pyjson` | `json.dumps` and `repr` spelled as CPython spells them. |
| `internal/pyvalue` | `json.loads` with order and number spelling kept, and the Go shapes a state or criterion can take, mapped onto `pyjson`. |
| `internal/decode` | Logit rows to typed answers: `Agent._decode_answers`, its temperatures, binning and `round(x, 4)`. |
| `internal/collate` | Sequences to the five flattened tensors the graph takes. |

## Quickstart

```go
a := lang.Analyze("Ich kann mich nicht in mein Konto einloggen")
fmt.Println(a.Script, a.Language, a.English) // latin de false

lang.IsEnglish(map[string][]string{"msg": {"No puedo entrar en mi cuenta"}}) // false
lang.Analyze("") // English: true, Script: lang.ScriptUnknown
```

A `lang` state is a string, bytes, or any map, slice or array of them, nested to six levels, as
`laya/lang.py` reads one; a struct, a number or nil reads as no text. A Go map has no insertion
order, so its values are read in sorted key order (like `fmt` for one key type; mixed key types by
kind, then type name), and a verdict can differ from a Python dict with the same items when the
4000-character budget truncates. Use `lang.Object`, which decodes JSON in document order, for an
order-sensitive state. `Analyze` reads every leaf, so its cost grows with the number of leaves.

## Layout

- No root package: the module path ends in `laya-go`, so tags are `laya-go/vX.Y.Z`.
- Every package in this module is pure Go: `lang`, `schema`, `sequence`, `tokenizer`, `infer` and
  `agent`, `presets` and `shortlist` now, and `router` later.

- Every package in this module is pure Go: `lang`, `email`, `schema`, `sequence`, `tokenizer`, `infer` and
  `agent` now, and `router`, `presets` and `shortlist` later.

  `agent` and `router` now, and `presets` and `shortlist` later.
- cgo lives only in `onnx`, a nested module with its own `go.mod`, so `go test ./...` here never
  compiles it. `onnx` requires this module and `github.com/yalue/onnxruntime_go`; until this module has
  a release tag, `onnx` builds only inside this repository (its `go.mod` points here with `replace`). Importing `lang` or `agent` never compiles cgo.
- `tokenizer` is public (`LoadFS`, `Parse`); `internal/unicodedata`, `internal/pyjson`,
  `internal/pyvalue`, `internal/decode` and `internal/collate` stay internal, and
  `internal/fixtures` and `internal/goldens` exist for the parity tests alone. `infer.Batch` is the
  public name of what `collate` builds.

## Parity

Python is the source of truth. The tests compare against goldens recorded from it, and the Unicode
behaviour that decides an answer is compiled in from the reference instead of taken from the Go
toolchain, whose Unicode release changes between versions.

```sh
python laya-go/scripts/gen_tables.py            # regenerate the compiled tables
python laya-go/scripts/gen_tables.py --check    # fail if a committed table is stale
python laya-go/scripts/gen_go_fixtures.py       # regenerate laya-go/fixtures/go_extra.json
```

Run them under Python 3.12, the version the lane pins. `go_extra.json` records the interpreter's
Unicode version (15.0.0), and `--check` under an interpreter with another one (3.11 has 14.0.0)
reports the Unicode-dependent families as stale.

Tests read goldens from `$LAYA_GO_FIXTURES` (default: `laya-java/fixtures`, plus the committed
`laya-go/fixtures/go_extra.json`), and the tokenizer tests need `$LAYA_CHECKPOINTS` to point at a
directory holding the `english` and `multilingual` checkpoints. Without it they skip, and the CI
lane fails a run in which anything skipped.

```sh
cd laya-go
LAYA_CHECKPOINTS=/path/to/checkpoints go test ./...
LAYA_CHECKPOINTS=/path/to/checkpoints go test -bench Encode -run '^$' ./tokenizer
```

CI (`.github/workflows/go.yml`) regenerates every golden from Python at the commit under test into
a scratch directory, points the tests at it, and fails on any skip.
