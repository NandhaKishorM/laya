# Changelog

All notable changes to the `laya-onnx` crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged `laya-rust-v<version>`.

## [0.1.0] - unreleased

First release: a port of the Laya Python SDK, checked against goldens and routing probes
regenerated from laya (Python) 0.4.14 for the supported paths (see the known gaps below).

### Added

- The abstention gate, ported from `laya/confidence.py`: `PredictOptions::min_confidence` with
  `LayaEngine::predict_with_options` and `LayaRouter::predict_with_options`, and
  `BatchOptions::min_confidence` for `predict_batch` on the engine and the router (one threshold
  for the whole batch, applied to every result and no part of the router's grouping).
  `MinConfidence` is a scalar or a per-bucket map, built with `MinConfidence::scalar` and
  `MinConfidence::map`, which refuse what `check_min_confidence` refuses with Python's `ValueError`
  text (`LayaError::InvalidMinConfidence`); the router validates before it routes or loads, and the
  engine before it looks at the states. With a threshold set, `Answer::{low_confidence, abstention,
  abstention_threshold}` report the gate (`GATE_STATES`: passed, abstained, unevaluated); without
  one they report nothing. `Answer::jev_confidence` is Python's `jev_confidence` and
  `bucket_keys()` its `bucket_keys`. `predict_long` takes no threshold, as in Python. Checked
  against `routing/confidence_probe.json`, which Python's own `confidence.py` writes. Two API
  notes: `BatchOptions` no longer derives `Eq` (it now holds floats), and `LayaError` is not
  `#[non_exhaustive]`, so an exhaustive `match` on it outside the crate needs an arm for
  `InvalidMinConfidence`.
- `LangGuess` and `LayaRouter::{route_with_lang_guess, predict_with_lang_guess,
  predict_long_with_lang_guess, with_lang_guess}`: Python's `lang_guess` hint (a language code or a
  callback taking the state), per call or installed on the router, checked after an explicit `lang`
  and before detection. A hint that names no language falls through to detection.
- `LanguageAnalysis::mixed_segment`, the line or field the segment scan singled out in a state that
  reads English as a whole, and the matching routing reason ("Latin script, mostly English, but a
  line or field reads as ..."). `LanguageAnalysis` is now `#[non_exhaustive]`, so a field added
  later does not break callers again; read it, do not construct it.
- `LayaEmail::{clean_body_signed, state_with_max_chars}`: `max_chars` as Python takes it, including
  the `max_chars * 4` pre-cut and the specified results for `0` and negative budgets. `clean_body`
  and `state` are unchanged and keep the 3000-character default.
- `LayaEngine::predict_long` and `LayaRouter::predict_long` (`aggregate="auto"`): scan a state
  longer than one sequence in overlapping token windows and keep the most confident window per
  question, with `Answer::window()` attribution, `LayaResult::scan_usage()` and summed
  `state_usage()`. Matches Python's `ONNXAgent.predict_long` and the .NET SDK; hooks, per-language
  temperatures are not part of it, and, as in Python, it takes no `min_confidence`. New `LongOptions`, `LayaLongPredictor`,
  `LayaError::{UnsupportedAggregate, InvalidWindow, NotSupported}` and a default
  `LayaTokenizer::decode`. `LayaError` is not `#[non_exhaustive]`, so an exhaustive `match` on it
  outside the crate needs arms for the three new variants.
- `LayaEngine::predict_batch` (and `predict_batch_serialize`) and `LayaRouter::{predict_batch,
  route_batch}`: Python's `ONNXAgent.predict_batch` and `Router.predict_batch` / `route_batch`. One
  result per state in input order, `batch_size` states per forward pass (`None`, `0` and, in
  Python, negative values send them all in one), and `sort_by_length` windows of eight batches
  that apply only with a `batch_size` above one and below the state count. The router groups
  requests by checkpoint and then by question schema, restores request order, attaches each
  request's `routing`, and honours each request's `model`, `task`, `lang` and `lang_guess`;
  `route_batch` loads nothing. New `BatchOptions`, `BatchRequest` (both `#[non_exhaustive]`, with
  chained setters, so hooks, `lang` and per-request token budgets can be added
  later), `LayaBatchPredictor` with the provided method `LayaPredictor::as_batch_predictor` (a
  default of `None`, so existing implementors keep compiling) and `LayaError::InvalidState`, raised
  for a `null` state as Python raises `TypeError`. Hooks, per-call `lang` and
  `max_len` / `head_max_len` and the router's `diagnostics` report are not part of it. `LayaError`
  is not `#[non_exhaustive]`, so an exhaustive `match` on it outside the crate needs an arm for the
  new variant.
- Behaviour change: `LayaEngine::predict` and `predict_serialize` now act as Python's `system_one`,
  a one-state `predict_batch`. A `null` state returns `LayaError::InvalidState` (Python's
  `TypeError`) instead of being scored as the text `"null"`, and an empty `QuestionSet` returns an
  empty result with zero usage instead of `LayaError::InvalidQuestion`. `predict_long` refuses a
  `null` state the same way, before it checks the aggregate mode, as Python does.
- `predict` now runs through a shared multi-state core that scores several states in one forward
  pass; its results are unchanged.
- Advisory parity CI on every push and PR: regenerate fixtures from the current Python checkout,
  test all checkpoints, and reject skipped tests. No golden cache or crate publication.
- Python float-repr parity probes, including exact half-even decimal ties and random bit patterns.
- `LayaResult::state_usage()` reports state-token truncation across question budgets, matching
  Python 0.4.0 without changing the existing `Usage` totals or result constructor.
- Engine score legends render structured criteria as strings, matching Python 0.4.0.
- `NoMarkers` errors include actual/requested marker counts and the total sequence budget,
  matching Python's actionable truncation diagnostic.
- State usage also reports option-span collisions after clipping, including the applied cap.
- `LayaEngine`: `choice`, `score` and `noul` predictions over the exported ONNX checkpoints
  (english, multilingual, typed-decisions); all questions for a state run in one batched forward pass.
- Loads both ONNX layouts: the fused `model.onnx` and the split `encoder.onnx` + `head.onnx`.
- Answers report `confidence` and `answer_confidence` (added in Python 0.3.21).
- Python-compatible sequence building, JSON serialization and calibration, checked against goldens
  recorded from the Python SDK.
- Checkpoint router, language detection, email state and option shortlisting.
- Opt-in model download from Hugging Face (Cargo feature `download`).
- Samples: quickstart, routing and benchmark.

### Fixed

- `Calibration::round4` is now exactly Python's `round(x, 4)`: it rounds the exact binary value half
  to even in integer arithmetic (it used to scale by 10 000 in floating point, which differed on
  about 2% of probabilities, e.g. `0.00625` gave `0.0062` where Python gives `0.0063`) and keeps
  the sign of a value that rounds to zero.
- `predict_long` emits Python's two warnings (window capped, stride reduced), and loading a
  checkpoint whose temperatures were clamped emits Python's warning, each with Python's text
  through the `log` facade (`warn`, target `laya`); `log` is a new dependency.
- A string in a checkpoint's `temperature` or `temperature_by_options` is read as Python's `float(str)` reads it (surrounding whitespace, any-script digits, single underscores, `inf`/`nan`) before clamping, instead of being replaced by 1.0.
- A checkpoint whose temperature is a JSON integer too large for a float fails to load with `LayaError::InvalidConfig("OverflowError: int too large to convert to float")`, as Python's `clamp_temperature` raises; a float literal such as `1e400` is read as infinity and replaced by 1.0, as in Python.
- The bare JSON constants `NaN`, `Infinity` and `-Infinity`, which Python's `json.loads` reads as floats, are accepted in `rl_agent_config.json` and `tokenizer_config.json` (a temperature of `NaN` clamps to 1.0 with Python's warning).
- Checkpoint JSON (`rl_agent_config.json`, `tokenizer_config.json`) is read by a reader that follows Python's `json.loads`: integers keep their literal (the temperature warning prints `0` for `-0` and big integers as Python does), `NaN`/`Infinity`/`-Infinity` are numbers only in value position, and nothing is encoded in-band. `serde_json` now uses `float_roundtrip` so JSON floats equal Python's `float()` bit for bit. An identical warning is emitted once per process, as Python's default filter does.
- Config loading matches Python's `ONNXAgent.__init__` and `json.loads` further: `option_layout` other than `sequential` is refused with Python's message; an integer literal of more than 4300 digits is refused with Python's `ValueError` text; `max_len`/`head_max_len` read `true`/`false` as 1/0 and `-0` as 0, and a float, string, `null`, list or object is an error (Python fails at the first prediction; this port refuses the config when it is loaded, with the same `TypeError` text, `head_max_len` first). A negative value is refused too, since Python slices from the end for it. The JSON reader no longer re-validates the rest of the document for every character of a string.
- Config JSON errors read like Python's `JSONDecodeError` (`Expecting ',' delimiter: line 1 column 7 (char 6)`, checked on 1,500 generated malformed texts), a file is read in text mode (`\r\n` becomes `\n` first), and `temperature` / `temperature_by_options` of the wrong JSON type fail while loading with Python's `TypeError` / `AttributeError` text (a string iterates its characters, a dict its keys).
- A `temperature` list shorter than three loads, and `predict` fails with `IndexError: list index out of range` for a question whose type index it lacks (choice 0, score 1, noul 2), as Python does when it evaluates `self.temperature[qtype]` (even when a `temperature_by_options` bucket matches); entries past the third are ignored.
- `LangGuess` callbacks receive the state as Python passes it (`GuessInput::Value`, or the original
  bytes as `GuessInput::Bytes` from `LayaRouter::route_bytes`).
- Email cleaning matches Python's `re` where the `regex` crate differs: `\b` follows Python's `\w`
  (superscripts and other `No` digits are word characters; combining marks, joiners and connector
  punctuation other than `_` are not) and `re.I` folds U+0130 and U+0131 to `i`.
- Added `LanguageDetection::{state_text_bytes, analyse_bytes, is_english_bytes}` and
  `LayaRouter::route_bytes`: Python's `lang`/`router` accept a `bytes` state, decode it as strict
  UTF-8 and read invalid UTF-8 as no text.

- `LayaRouterOptions::max_loaded` defaults to 2, as Python's `Router(max_loaded=2)` does, so a
  router that alternates two languages no longer reloads a checkpoint on every request.
- Language detection follows Python 0.4.11: the line-by-line and field-by-field scan of a state that
  reads English as a whole, the Swedish and Nordic rules (Swedish, Banglish and Azerbaijani word
  lists, short Swedish requests, the Swedish-Danish overlap marker), the English rescue of a single
  accented loanword, acronym-shaped and all-caps lines no longer named foreign, identifier and code
  lines skipped, fullwidth and IPA letters counted as Latin (cutoff U+02B0) and letters no script
  range claims counted as "other". Python's `\w`, `\s`, `isupper` and `unicodedata.combining` are
  reproduced character for character rather than through the `regex` crate.
- A `lang` of `C`, `POSIX`, `C.UTF-8`, `und`, `zxx`, `mul` or a blank string no longer pins a
  checkpoint: it falls through to `lang_guess` and detection. `en_US` and `en_US.UTF-8` now read as
  English instead of routing to the multilingual checkpoint.
- Routing reasons quote caller text like Python's `%r`, escaping non-printable characters.
- Email cleaning follows Python 0.4.11: nine quote-header patterns (Portuguese, Spanish and French
  attributions that carry a date, Outlook headers in four languages, bare `From:`/`De:` headers
  confirmed by the `Sent:`/`Enviado:`/`Date:` line after them), the wrapped-attribution tail,
  Portuguese/Spanish/French sign-offs, the English sign-off name rule, device and mail-app footers,
  and Portuguese/Spanish/French disclaimers. A bare `From:` line that opens a sentence is kept.
- Language detection follows Python 0.4.14: the English stop list counts question words, auxiliaries,
  pronouns and connectives (116 words, up from 34), so a short request such as "Why hasn't my card
  been delivered?" is read as English instead of staying undecided and taking the multilingual
  checkpoint. The opt-in `explain` diagnostics Python added in 0.4.14 are not ported.
- Dependency: `unicode-normalization` (canonical combining classes for `unicodedata.combining`).

### Known gaps

- Language detection, routing and email cleaning follow the regenerated Python 0.4.14 probes; a
  later Python change needs the probes regenerated.
- Not yet ported: histogram-binning calibration and `lang_temperatures`.
