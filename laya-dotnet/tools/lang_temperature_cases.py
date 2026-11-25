"""Synthetic inputs for the model-free `lang_temperatures` probe, `routing/lang_temperatures_probe.json`.

`dump_routing_golden.build_lang_temperatures_probe` feeds each of these to Python's own
`laya.common.resolve_lang_temperatures` and records the resolved table, or the exception it raised, plus
every warning it emitted (text and category), so a port is checked against the shipping code on the same
inputs without loading a model. This list is SDK-neutral: it is the shared corpus for the .NET and Rust
ports, and each port keeps a verbatim copy.

A case is `(label, pairs, base, typed)`:

* `pairs` is the table as an ordered list of `(key, entry)` pairs, or None for no table. An entry is None
  (inherit), or a dict with optional `temperature` (a list, or None to inherit) and `temperature_by_options`.
  A key that is not a string is written as the JSON value it is.
* `base` is the checkpoint's own `temperature` as `rl_agent_config.json` holds it: what an entry without its
  own temperature inherits, and what the error text prints for a checkpoint whose list is not three values.
* `typed` is whether a typed SDK can state the case: every key a string, every entry null or an object whose
  `temperature` is null or a list of numbers and whose `temperature_by_options` is null or an object of
  numbers, every number written as a float (a JSON `2` is a Python int, which prints as `2` where a typed
  port's float prints `2.0` in a warning). The other cases pin Python's own refusals and spellings, and a
  typed port skips them.

Non-finite floats are written by the dumper as the strings "nan", "inf" and "-inf" (a typed port reads them
back as the numbers); no case uses those strings as text.
"""

NAN = float("nan")
INF = float("inf")

BASE = [1.0, 1.0, 1.0]
FITTED = [1.6, 1.25, 1.98]

PROBE_CASES = [
    # --- accepted shapes ---------------------------------------------------------------------
    ("no table", None, BASE, True),
    ("empty table", [], BASE, True),
    ("own temperature", [("de", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("own temperature and buckets",
     [("de", {"temperature": [2.0, 3.0, 1.5], "temperature_by_options": {"choice:3-5": 2.5, "noul:2": 0.7}})],
     BASE, True),
    ("region tag", [("de-AT", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("uppercase key", [("DE", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("mixed case region", [("Pt-BR", {"temperature": [1.2, 1.3, 1.4]})], BASE, True),
    ("underscore is not a separator", [("de_DE", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("only the first dash splits", [("zh-Hant-TW", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("empty key", [("", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("dash only", [("-", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("leading dash", [("-de", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("full lowercase of a capital I with dot", [("İ", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("final sigma", [("ΣΣ", {"temperature": [2.0, 3.0, 1.5]})], BASE, True),
    ("last duplicate wins",
     [("de", {"temperature": [2.0, 3.0, 1.5]}), ("DE-at", {"temperature": [4.0, 4.5, 3.5]})], BASE, True),
    ("last duplicate wins over a null entry",
     [("de", {"temperature": [2.0, 3.0, 1.5]}), ("de-CH", None)], FITTED, True),
    ("two languages",
     [("de", {"temperature": [2.0, 3.0, 1.5]}), ("fr", {"temperature_by_options": {"score:3-5": 2.0}})],
     FITTED, True),
    # --- inheriting the checkpoint's temperature ------------------------------------------------
    ("null entry inherits", [("fr", None)], FITTED, True),
    ("empty entry inherits", [("fr", {})], FITTED, True),
    ("null temperature inherits", [("fr", {"temperature": None})], FITTED, True),
    ("null temperature keeps own buckets",
     [("fr", {"temperature": None, "temperature_by_options": {"choice:2": 2.0}})], FITTED, True),
    ("only buckets inherits temperature", [("fr", {"temperature_by_options": {"noul:2": 0.7}})], FITTED, True),
    ("empty buckets", [("fr", {"temperature": [1.0, 2.0, 3.0], "temperature_by_options": {}})], BASE, True),
    ("null buckets", [("fr", {"temperature": [1.0, 2.0, 3.0], "temperature_by_options": None})], BASE, True),
    ("inherited out-of-range values are clamped without a warning of their own",
     [("fr", None)], [0.3, 6.0, 1.0], True),
    ("bucket names are not validated", [("fr", {"temperature_by_options": {"bogus": 2.0, "": 3.0}})], BASE, True),
    # --- clamping and the warning ---------------------------------------------------------------
    ("clamp high and low", [("de", {"temperature": [9.0, 0.1, 5.0]})], BASE, True),
    ("clamp boundaries are accepted", [("de", {"temperature": [0.5, 5.0, 1.0]})], BASE, True),
    ("clamp a bucket",
     [("de", {"temperature_by_options": {"choice:3-5": 99.0, "noul:2": 0.01, "score:3-5": 2.0}})], BASE, True),
    ("clamp temperature and buckets in one warning",
     [("de", {"temperature": [9.0, 1.0, 0.0], "temperature_by_options": {"choice:2": -1.0}})], BASE, True),
    ("non-finite values", [("de", {"temperature": [NAN, 1.0, INF]})], BASE, True),
    ("negative infinity", [("de", {"temperature": [float("-inf"), 1.0, 1.0]})], BASE, True),
    ("negative zero", [("de", {"temperature": [-0.0, 1.0, 1.0]})], BASE, True),
    ("a warning per language",
     [("de", {"temperature": [9.0, 1.0, 1.0]}), ("fr", {"temperature": [1.0, 0.1, 1.0]})], BASE, True),
    ("a key's own spelling is named in the warning",
     [("DE-at", {"temperature": [9.0, 1.0, 1.0]})], BASE, True),
    ("a quote in the key", [("it's", {"temperature": [9.0, 1.0, 1.0]})], BASE, True),
    ("a non-ASCII key", [("dé", {"temperature": [9.0, 1.0, 1.0]})], BASE, True),
    ("a warning for the replaced entry of a later duplicate",
     [("de", {"temperature": [9.0, 1.0, 1.0]}), ("DE", {"temperature": [1.0, 1.0, 1.0]})], BASE, True),
    # --- refusals -------------------------------------------------------------------------------
    ("two values", [("de", {"temperature": [1.0, 2.0]})], BASE, True),
    ("four values", [("de", {"temperature": [1.0, 2.0, 3.0, 4.0]})], BASE, True),
    ("no values", [("de", {"temperature": []})], BASE, True),
    ("two values with a buckets map", [("de", {"temperature": [1.0, 2.0], "temperature_by_options": {"noul:2": 2.0}})],
     BASE, True),
    ("non-finite values in a refused list", [("de", {"temperature": [NAN, INF]})], BASE, True),
    ("an inherited list of two values", [("de", None)], [1.0, 2.0], True),
    ("an inherited list of four values", [("de", {"temperature_by_options": {"noul:2": 2.0}})], [1.0, 2.0, 3.0, 4.0], True),
    ("an inherited empty list", [("de", {})], [], True),
    ("an inherited list holding a string", [("de", None)], [1.0, "x", 2.0], False),
    ("an inherited string", [("de", None)], "1.5", True),
    ("an inherited object", [("de", None)], {"a": 1.0}, True),
    ("an own list needs no valid base", [("de", {"temperature": [2.0, 3.0, 1.5]})], [1.0], True),
    ("a refusal after a warning",
     [("de", {"temperature": [9.0, 1.0, 1.0]}), ("fr", {"temperature": [1.0]})], BASE, True),
    ("a refusal before a later entry",
     [("fr", {"temperature": [1.0]}), ("de", {"temperature": [9.0, 1.0, 1.0]})], BASE, True),
    ("the refused language is named as spelled", [("DE-at", {"temperature": [1.0]})], BASE, True),
    ("a quote in a refused key", [("it's", {"temperature": [1.0]})], BASE, True),
    # --- shapes only Python can state -----------------------------------------------------------
    ("integers", [("de", {"temperature": [2, 3, 1]})], BASE, False),
    ("integers that the clamp moves", [("de", {"temperature": [99, 0.01, 1]})], BASE, False),
    ("a bool in the list", [("de", {"temperature": [True, 1.5, 2.0]})], BASE, False),
    ("false in the list", [("de", {"temperature": [False, 1.5, 2.0]})], BASE, False),
    ("a numeric string", [("de", {"temperature": ["2.5", 1.5, 2.0]})], BASE, False),
    ("a string that is no number", [("de", {"temperature": ["abc", 1.5, 2.0]})], BASE, False),
    ("a null in the list", [("de", {"temperature": [None, 1.5, 2.0]})], BASE, False),
    ("a nested list in the list", [("de", {"temperature": [[1.0], 1.5, 2.0]})], BASE, False),
    ("a bucket value that is a bool", [("de", {"temperature_by_options": {"choice:2": True}})], BASE, False),
    ("a bucket value that is null", [("de", {"temperature_by_options": {"choice:2": None}})], BASE, False),
    ("a bucket value that is a string", [("de", {"temperature_by_options": {"choice:2": "2.5"}})], BASE, False),
    ("a bucket value that is an integer", [("de", {"temperature_by_options": {"choice:2": 7}})], BASE, False),
    ("a temperature that is a number", [("de", {"temperature": 2})], BASE, False),
    ("a temperature that is a string", [("de", {"temperature": "abc"})], BASE, False),
    ("a temperature that is an object", [("de", {"temperature": {"a": 1.0}})], BASE, False),
    ("a temperature that is false", [("de", {"temperature": False})], BASE, False),
    ("an entry that is a list", [("de", [1.0, 2.0, 3.0])], BASE, False),
    ("an entry that is a string", [("de", "x")], BASE, False),
    ("an entry that is a number", [("de", 2.0)], BASE, False),
    ("an entry that is false", [("de", False)], BASE, False),
    ("buckets that are a list", [("de", {"temperature_by_options": [1.0]})], BASE, False),
    ("buckets that are a string", [("de", {"temperature_by_options": "x"})], BASE, False),
    ("buckets that are a number", [("de", {"temperature_by_options": 3})], BASE, False),
    ("buckets that are an empty list", [("de", {"temperature_by_options": []})], BASE, False),
    ("buckets that are zero", [("de", {"temperature_by_options": 0})], BASE, False),
    ("buckets that are an empty string", [("de", {"temperature_by_options": ""})], BASE, False),
    ("a key that is a number", [(1, {"temperature": [1.0, 1.0, 1.0]})], BASE, False),
    ("a key that is null", [(None, {"temperature": [1.0, 1.0, 1.0]})], BASE, False),
    ("a base that is a number", [("de", None)], 2.0, False),
    ("a base that is null", [("de", None)], None, False),
]


# ---------------------------------------------------------------------------------------------
# `routing/router_forwarding_probe.json`: what a Router hands the agent it routed to. Python's own
# `Router.predict`, `Router.predict_long` and `Router.predict_batch` run over stub agents that record the
# keyword arguments they received (no model, no checkpoint). Requests carry only JSON: `lang_guess` is a
# language code. `has_lang_temperatures` says which stub agent carries a non-empty `lang_temperatures`
# (a Router splits a batch by language only for such an agent); `stub_takes_lang: false` is a `predict_long`
# with no `lang` parameter. The States are plain text, so the detection is the one `lang_probe.json` pins.
# ---------------------------------------------------------------------------------------------
ROUTER_EN = "I was charged twice, please refund."
ROUTER_DE = "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung die nicht korrekt ist"
ROUTER_FR = "Le client a ete facture deux fois et demande un remboursement pour cette facture qui est incorrecte"
ROUTER_QUESTIONS = {"q": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}
ROUTER_OTHER_QUESTIONS = {"q": {"type": "noul", "instructions": "Is the customer angry?"}}

# Router.predict(state, questions, **kwargs)
ROUTER_PREDICT_CASES = [
    ("the detected language of English text is forwarded", {"state": ROUTER_EN}),
    ("the detected language of German text is forwarded", {"state": ROUTER_DE}),
    ("an explicit language wins over detection and goes unnormalised", {"state": ROUTER_DE, "lang": "en-GB"}),
    ("an explicit language routes and is forwarded", {"state": ROUTER_EN, "lang": "DE-at"}),
    ("an empty language is forwarded as it is", {"state": ROUTER_DE, "lang": ""}),
    ("a language hint leaves no detection", {"state": ROUTER_EN, "lang_guess": "de"}),
    ("an explicit model leaves no detection", {"state": ROUTER_EN, "model": "ml"}),
    ("text with no letters has no language", {"state": "12345"}),
    ("max_len is forwarded alone", {"state": ROUTER_EN, "max_len": 100}),
    ("head_max_len is forwarded alone", {"state": ROUTER_EN, "head_max_len": 64}),
    ("both budgets are forwarded", {"state": ROUTER_DE, "max_len": 300, "head_max_len": 64}),
    ("zero budgets are forwarded", {"state": ROUTER_EN, "max_len": 0, "head_max_len": 0}),
    ("negative budgets are forwarded", {"state": ROUTER_EN, "max_len": -1, "head_max_len": -2}),
    ("a language and budgets together", {"state": ROUTER_EN, "lang": "fr", "max_len": 200, "head_max_len": 96}),
]

# Router.predict_long(state, questions, **kwargs) over an agent whose predict_long takes `lang` or not.
ROUTER_LONG_CASES = [
    ("the detected language reaches the scan", {"state": ROUTER_DE}),
    ("an explicit language reaches the scan", {"state": ROUTER_EN, "lang": "de-AT"}),
    ("a language hint leaves no detection", {"state": ROUTER_DE, "lang_guess": "en"}),
    ("an empty language is forwarded", {"state": ROUTER_DE, "lang": ""}),
    ("no letters, no language", {"state": "12345"}),
    ("a scan with no lang parameter runs without it and warns about the detected language",
     {"state": ROUTER_DE, "stub_takes_lang": False}),
    ("a scan with no lang parameter warns about the requested language",
     {"state": ROUTER_EN, "lang": "fr", "stub_takes_lang": False}),
    ("a requested language longer than 32 characters is cut in the warning",
     {"state": ROUTER_EN, "lang": "x" * 40, "stub_takes_lang": False}),
    ("a requested language of exactly 32 characters is not cut",
     {"state": ROUTER_EN, "lang": "y" * 32, "stub_takes_lang": False}),
    ("a requested empty language still warns", {"state": ROUTER_EN, "lang": "", "stub_takes_lang": False}),
    ("no language, no warning", {"state": "12345", "stub_takes_lang": False}),
    ("a scan with no lang parameter and a hint warns about nothing",
     {"state": ROUTER_DE, "lang_guess": "en", "stub_takes_lang": False}),
]

# Router.predict_batch(requests): `has_lang_temperatures` per stub agent, then the requests.
_R = ROUTER_QUESTIONS
ROUTER_BATCH_CASES = [
    ("no agent has per-language temperatures, so a language never reaches or splits",
     {"english": False, "multilingual": False},
     [{"state": ROUTER_EN, "questions": _R, "lang": "en"},
      {"state": ROUTER_EN + " Again.", "questions": _R},
      {"state": ROUTER_DE, "questions": _R, "lang": "de"},
      {"state": ROUTER_DE + " Noch einmal.", "questions": _R}]),
    ("a language splits the groups of an agent that has temperatures",
     {"english": False, "multilingual": True},
     [{"state": ROUTER_DE, "questions": _R},
      {"state": ROUTER_DE + " 2", "questions": _R, "lang": "de"},
      {"state": ROUTER_DE + " 3", "questions": _R, "lang": "DE"},
      {"state": ROUTER_DE + " 4", "questions": _R, "lang": "de-AT"},
      {"state": ROUTER_DE + " 5", "questions": _R}]),
    ("the key is the language as written, not normalised, and an empty one is a key",
     {"english": True, "multilingual": True},
     [{"state": ROUTER_EN, "questions": _R, "lang": "en-US"},
      {"state": ROUTER_EN + " 2", "questions": _R, "lang": "EN-us"},
      {"state": ROUTER_EN + " 3", "questions": _R, "lang": ""},
      {"state": ROUTER_EN + " 4", "questions": _R, "lang": ""},
      {"state": ROUTER_EN + " 5", "questions": _R, "lang": "en-US"}]),
    ("a hint leaves no detection, so its request has no language of its own",
     {"english": True, "multilingual": False},
     [{"state": ROUTER_EN, "questions": _R},
      {"state": ROUTER_EN + " 2", "questions": _R, "lang_guess": "en"},
      {"state": ROUTER_EN + " 3", "questions": _R, "lang_guess": "en"},
      {"state": ROUTER_EN + " 4", "questions": _R}]),
    ("an explicit model leaves no detection either",
     {"english": False, "multilingual": True},
     [{"state": ROUTER_DE, "questions": _R, "model": "ml"},
      {"state": ROUTER_DE + " 2", "questions": _R},
      {"state": ROUTER_DE + " 3", "questions": _R, "model": "ml", "lang": "de"}]),
    ("token budgets split the groups whatever the agent",
     {"english": False, "multilingual": False},
     [{"state": ROUTER_EN, "questions": _R},
      {"state": ROUTER_EN + " 2", "questions": _R, "max_len": 100},
      {"state": ROUTER_EN + " 3", "questions": _R, "head_max_len": 100},
      {"state": ROUTER_EN + " 4", "questions": _R, "max_len": 100, "head_max_len": 100},
      {"state": ROUTER_EN + " 5", "questions": _R, "max_len": 100},
      {"state": ROUTER_EN + " 6", "questions": _R},
      {"state": ROUTER_EN + " 7", "questions": _R, "max_len": 0},
      {"state": ROUTER_EN + " 8", "questions": _R, "max_len": 0, "head_max_len": 0},
      {"state": ROUTER_EN + " 9", "questions": _R, "max_len": -1}]),
    ("schema, budgets, language and checkpoint together, in first-seen order",
     {"english": True, "multilingual": True},
     [{"state": ROUTER_EN, "questions": _R},
      {"state": ROUTER_DE, "questions": _R, "max_len": 150},
      {"state": ROUTER_EN + " 2", "questions": ROUTER_OTHER_QUESTIONS},
      {"state": ROUTER_DE + " 2", "questions": _R, "max_len": 150},
      {"state": ROUTER_EN + " 3", "questions": _R, "lang": "en"},
      {"state": ROUTER_FR, "questions": _R, "lang": "fr", "max_len": 150},
      {"state": ROUTER_EN + " 4", "questions": _R},
      {"state": ROUTER_DE + " 3", "questions": _R, "max_len": 150, "lang": "de"},
      {"state": ROUTER_EN + " 5", "questions": ROUTER_OTHER_QUESTIONS, "max_len": 90, "head_max_len": 40}]),
]
