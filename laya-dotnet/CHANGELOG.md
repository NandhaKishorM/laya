# Changelog

All notable changes to the `Laya.Onnx` NuGet package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged `laya-dotnet-v<version>`.

## [0.1.0] - unreleased

First release: a port of the Laya Python SDK, tested against laya (Python) 0.4.11 (routing
goldens: 0.4.12).

### Added

- `LayaEngine.PredictBatch` (`ILayaBatchPredictor`), `LayaBatchOptions` (`BatchSize`, `SortByLength`),
  `LayaRouter.PredictBatch` and `LayaRouter.RouteBatch` over `LayaBatchRequest`, ported from Python
  `ONNXAgent.predict_batch` and `Router.predict_batch` / `route_batch`: shared forward passes, `batch_size`
  chunking, `sort_by_length` inside windows of eight batches (effective only for `1 < batch_size < len(states)`),
  routing grouped by checkpoint then question schema with input order restored, and per-request `lang_guess`.
  Hooks and the router `diagnostics` dictionary are not ported yet. Existing signatures are unchanged.
- Per-language temperatures and per-call overrides, ported from Python `lang_temperatures` /
  `resolve_lang_temperatures` and the `lang`, `max_len` and `head_max_len` arguments: `LayaOptions.LangTemperatures`
  (a table of `LayaLangTemperature`, resolved when the engine is created, with Python's key normalisation, clamping,
  warning and `ValueError` text; a router hands it to every engine it builds), `Lang`, `MaxLen` and `HeadMaxLen` on
  `LayaPredictOptions` and `LayaBatchOptions`, `MaxLen` and `HeadMaxLen` on `LayaBatchRequest`, and
  `LayaEngine.PredictLong(state, questions, LayaLongOptions)` with `Lang`. A language the table holds replaces both
  the checkpoint's temperature and its per-bucket map. A router forwards its `lang` argument, else the language it
  detected, to the checkpoint, and `PredictBatch` splits a checkpoint's requests by token budget and, for a
  checkpoint with a table, by language. Token budgets are unchecked as in Python: `0`, a negative value or one too
  small for the options' markers throws the "option markers fit" `ArgumentException`. New interfaces
  `ILayaCallPredictor`, `ILayaLongLangPredictor` and `ILayaLangTemperatures` let a router pass these on; existing
  signatures are unchanged.
- Calibration files and histogram binning, ported from Python's `calibration=` / `load_calibration`
  (`apply_calibration_payload`, `apply_binning_map`): `LayaOptions.Calibration`, the path of a JSON file read when the engine
  is created, after the checkpoint's temperatures and the `LangTemperatures` table. Its `temperature` and
  `temperature_by_options` replace the checkpoint's (clamped, with Python's warning), and its `binning_map` recalibrates
  `AnswerConfidence` alone (`int(p * bins)` of the unrounded `max(p)`, then rounded to four places; a bucket it lacks passes
  through; skipped for a call whose `Lang` the table holds), so `MinConfidence` and the `PredictLong` window choice see the
  binned value. A file Python refuses throws `InvalidDataException` carrying Python's exception type and text. A version 2 file's
  checkpoint-identity warning is not emitted: the SDK has no equivalent of Python's `model_id_or_path` string; temperatures,
  binning and errors are unaffected. A router hands the path to every engine it builds. Existing
  signatures are unchanged.
- Abstention gate, ported from Python `laya/confidence.py` (`min_confidence`, `jev_confidence`): `MinConfidence`
  (`Of` for one threshold, `ForBuckets` for a per-option-count map, `BucketKeys`), `LayaPredictOptions`
  with new `LayaEngine.Predict(state, questions, options)` and `LayaRouter.Predict(..., langGuess, options)`
  overloads, `LayaBatchOptions.MinConfidence` for engine and router batches, and on every `Answer`
  `LowConfidence`, `Abstention` (`GateStates`: `passed`, `abstained`, `unevaluated`; `null` when ungated),
  `AbstentionThreshold` and `JevConfidence()`. Invalid thresholds throw `ArgumentException` with Python's
  `ValueError` text. `PredictLong` takes none, as in Python. Existing signatures are unchanged.
- Behaviour change, to match Python (`system_one` is `predict_batch([state])`): `LayaEngine.Predict` and
  `LayaRouter.Predict` / `PredictLong` refuse a null state with `ArgumentNullException` carrying Python's
  text (`state must not be None; pass a string, dict, or list`) instead of answering the text `"null"`,
  and `LayaEngine.Predict` with an empty `QuestionSet` returns an empty answer set with zero usage instead
  of throwing. `LayaEngine.Predict` now reports the usage diagnostics too, as `PredictBatch([state])[0]` does. An unknown model name now carries Python's exact message.
- `LangGuess` (a language code or a callback of the state), Python's `lang_guess`: per call through new
  `LayaRouter.Route`, `Predict` and `PredictLong` overloads, and for every request through
  `LayaRouterOptions.LangGuess`. Precedence is explicit `lang`, then `lang_guess`, then detection;
  a hint that names no language falls through. Existing signatures are unchanged.
- `LayaEmail.State` overload taking Python's `max_chars`, consumed (never a state field) and applied only
  when cleaning.
- `LayaConfig.TemperatureRawValues` / `TemperatureByOptionsRawValues` and
  `Calibration.ClampTemperature(object?)` / `TemperatureRejected`: Python's `clamp_temperature` and
  `temperature_rejected` on the JSON value itself.
- `LayaEngine.PredictLong` and `LayaRouter.PredictLong`, ported from Python
  `ONNXAgent.predict_long`: overlapping token windows, batched inference, deciding-window
  attribution, per-type aggregation and summed scan diagnostics.
- `LayaEngine`: `choice`, `score` and `noul` predictions over the exported ONNX checkpoints
  (english, multilingual, typed-decisions); all questions for a state run in one batched forward pass.
- Loads both ONNX layouts: the fused `model.onnx` and the split `encoder.onnx` + `head.onnx`.
- Answers report `confidence` and `answer_confidence` (added in Python 0.3.21).
- Python-compatible sequence building, JSON serialization and calibration, checked against goldens
  recorded from the Python SDK.
- Checkpoint router, language detection, email state and option shortlisting.
- Opt-in model download from Hugging Face.
- Samples: quickstart, routing and benchmark.

### Fixed

- Language detection: a line or state written in capitals is no longer named a language on acronym-shaped
  evidence (`MON DES EST LA`, `STORES LOS ANGELES LAS VEGAS`), as Python's `_shouted` rules; combining
  marks count with their script only when they have a combining class, so Devanagari matras read as the
  script; the per-field read falls through to the diacritic rule when a named language is disbelieved; the
  `\w`/`\d`/`\s`/`isupper`/`strip` classes follow Python, including for characters outside the BMP.
- Checkpoint config JSON is read by a reader that follows CPython's `json.loads` (`PythonJsonReader`): the bare
  `NaN`, `Infinity` and `-Infinity` are accepted in value position only (`{NaN: 1}`, `nan`, `+Infinity`, `01`,
  trailing commas, comments and single quotes stay errors), integer and float literals keep their Python type
  (`-0` is the int `0`, `7.0` a float, a 30-digit int prints in full), floats are bit-identical to `float()`,
  and no marker text stands in for a token. `tokenizer_config.json` is read the same way. A temperature given as
  an integer too large for a float fails the load with Python's `OverflowError: int too large to convert to
  float` (`InvalidDataException`), while a float literal such as `1e400` reads as infinity and falls back to 1.0.
- A warning is traced once per process for the same text and call site, as Python's default filter does
  (`stacklevel`): the site is the caller of `LayaEngine.Create` / `FromDirectory` (temperature warning) or of
  `PredictLong` (window and stride warnings), through the router the router's own call. The `predict_long`
  window and stride warnings now carry Python's texts.
- `rl_agent_config.json` and `tokenizer_config.json` are read as strict UTF-8 and a BOM is rejected, as
  Python's `json.load(open(path))` does on a UTF-8 locale (it opens them without an `encoding`, so on a default
  Windows install it would use the ANSI code page; the two agree on ASCII files).
- Config and detection follow Python's loader for the cases the review found: JSON errors carry Python's exact text
  (`Expecting value: line L column C (char P)`, the integer-digit limit text), nesting up to CPython's limit loads,
  config files are read in text mode so CRLF and a lone CR count as one newline; `temperature` is iterated as Python
  does (a string yields its characters, a dict its keys, a number or null raises `TypeError`), `temperature_by_options`
  must be a mapping (`AttributeError`), `option_layout` other than `sequential` is refused, a `temperature` list shorter
  than three fails at the first prediction that needs the missing entry (`IndexError`), and a wrong-typed `max_len` /
  `head_max_len` is refused when the config is loaded, with the `TypeError` text Python raises at the first prediction
  (a float, string, null, list or object; a negative integer is refused with its own message); integers `>= 0` and
  booleans are accepted. See "Differences from Python" in the README. `str.lower()` is exact (final sigma,
  U+0130), only lists and tuples are walked for text (a set holds none and is not JSON serializable), and any
  `IReadOnlyDictionary` is walked as a mapping.
- **Behaviour change for callers passing lazy sequences as a state:** only an `IList` (arrays, `List<T>`) or a tuple is
  serialized as a JSON array and walked for language detection; a set, `Queue<T>`, LINQ iterator or any other
  non-list enumerable now throws `JsonException` when serialized (and holds no text for detection), as Python's
  `json.dumps` rejects a set or generator. Pass `ToList()` / `ToArray()`. A tuple is no longer left-truncated when
  over budget (Python truncates from the left only for a `list`).
- `Calibration.ClampTemperature(object?)` and `TemperatureRejected` accept every boxed numeric type (`int`, `short`,
  `byte`, `sbyte`, `ushort`, `uint`, `ulong`, `float`, `decimal`, `Half`, `nint`, `BigInteger`, `Int128`), converted as
  Python's `float()` rounds.
- Email patterns see astral decimal digits (Unicode Nd, such as the mathematical double-struck digits) as digits, so a
  reply header whose date is written with them is recognised as in Python, while astral numbers that are not decimal
  digits stay word characters only.
- Case-insensitive email matching also treats the long s with `s` and the Kelvin sign with `k`, as `re.I` does.
- Email patterns count and bound code points, not UTF-16 units, so characters outside the BMP next to a
  keyword or inside a `.{0,300}` window behave as in Python.
- `byte[]` states are decoded as strict UTF-8 for detection, and invalid UTF-8 is no text, as in Python;
  serializing one for a prediction throws, as Python's `TypeError` does.
- Route: a blank, whitespace, `C`, `C.UTF-8`, `POSIX`, `und`, `zxx`, `mul` or `.` `lang` falls through to
  detection, and locale codes (`en_US`, `en_US.UTF-8`, ` EN `) normalise as Python does; reasons quote
  caller text with Python's `repr` (non-printable characters escaped).
- Email cleaning: a negative `maxChars` slices as Python does (including the `max_chars * 4` pre-cut);
  U+001C..U+001F count as whitespace, `\b` follows Python's `\w`, and case-insensitive matching treats
  `i`, `I`, `ı` and `İ` as one letter, as `re.I` does.
- Checkpoint temperatures are read as Python does: a numeric string (`"2.5"`, `" 2.5 "`, `"1e1"`) is a
  number, and `true`, `null`, lists, dictionaries and non-numeric strings fall back to 1.0. The load
  warning is one message with Python's text, listing each entry as `name=repr -> applied`.
- `LayaRouterOptions.MaxLoaded` defaults to 2, as Python's `Router(max_loaded=2)` does, so a
  router that alternates two languages no longer reloads a checkpoint on every request.
- Undetermined Latin text uses the router's configured default, including its route reason.
- Language detection counts unlisted scripts, distinguishes mixed-language fields and lines
  (`MixedSegment`), and applies current script, stopword and loanword rules.
- Email cleaning preserves unpunctuated requests before disclaimers, distinguishes prose from
  signatures and reply headers, and handles Portuguese, Spanish and French client markers.
- Oversized questions report the surviving option markers and the actual sequence budget.
- Score legends return the same rendered criterion text shown to the model, including JSON for
  structured levels.
- Golden fixtures regenerated from current Python for all three checkpoints and routing.
- `Calibration.Round4` is exactly Python's `round(x, 4)`: it rounds the exact binary value half to even
  (the old scale-and-round differed on about 2% of values, e.g. `0.00625` and `0.03125`), and keeps
  the sign of a zero result.

### Known gaps

- Python-only opt-in features are not yet ported: hooks, question option ordering, digest controls and
  calibration binning (`load_calibration`). Batching is ported without hooks.
