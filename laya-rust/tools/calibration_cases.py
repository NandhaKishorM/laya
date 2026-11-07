"""Synthetic inputs for the model-free calibration probe, `routing/calibration_probe.json`.

`dump_routing_golden.build_calibration_probe` feeds each payload to Python's own
`laya.calibrate.apply_calibration_payload` and each lookup to `laya.calibrate.apply_binning_map`, and records
what was installed, or the exception it raised, plus every warning it emitted (text and category), so a port is
checked against the shipping code on the same inputs without loading a model. This list is SDK-neutral: it is
the shared corpus for the .NET and Rust ports, and each port keeps a verbatim copy.

A payload case is `(label, text, identity)`. `text` is the JSON document, written as text because a payload may
hold what Python's `json.load` accepts and a typed value cannot state (`NaN`, `Infinity`, an integer past
float range, a duplicate key); a port reads it with its Python-json reader. `identity` names an entry of
`IDENTITIES`: the `model_id_or_path`, `subfolder` and parsed `rl_agent_config.json` of the agent the payload is
applied to, which a version 2 file is compared against. An SDK that has no equivalent of
Python's `model_id_or_path` (the .NET and Rust SDKs have none) does not emit that comparison's warning: its tests skip the
recorded warnings whose text starts "calibration was fitted for model_id_or_path=" and assert every other one.

A lookup case is `(label, map name, bucket, confidence)` for `apply_binning_map`; the name picks an entry of `MAPS`,
which are already validated maps, so only a valid one is written. A non-finite `confidence` is written as "nan"/"inf"/"-inf".
"""
import json

# The agent a payload is applied to. `cfg` is a checkpoint config: its two temperature fields are not part of the
# identity a calibration file records.
CFG = {
    "encoder": "jhu-clsp/mmBERT-base",
    "max_len": 1024,
    "head_max_len": 192,
    "option_layout": "sequential",
    "temperature": [1.2, 1.3, 1.4],
    "temperature_by_options": {"choice:2": 1.1},
}
IDENTITIES = {
    "multilingual": {"model_id_or_path": "convaiinnovations/laya", "subfolder": "multilingual", "cfg": CFG},
    "english": {"model_id_or_path": "convaiinnovations/laya", "subfolder": None, "cfg": CFG},
    "local": {"model_id_or_path": "onnx/multilingual", "subfolder": None, "cfg": CFG},
}

# The config a version 2 file records: the checkpoint config without the two temperature fields.
RECORDED_CFG = {k: v for k, v in CFG.items() if k not in ("temperature", "temperature_by_options")}

T3 = [1.5, 2.0, 1.25]
MAP4 = {"bins": 4, "values": [0.1, 0.4, 0.6, 0.9]}


def _p(label, payload, identity="multilingual"):
    return (label, json.dumps(payload), identity)


def _raw(label, text, identity="multilingual"):
    return (label, text, identity)


def _v2(**extra):
    """A version 2 payload recorded for the multilingual checkpoint, with `extra` replacing or adding fields."""
    body = {"version": 2, "temperature": T3, "temperature_by_options": {"choice:3-5": 2.5},
            "model_id_or_path": "convaiinnovations/laya", "subfolder": "multilingual", "config": RECORDED_CFG}
    body.update(extra)
    return body


def _bin(entry, other=None):
    body = {"temperature": T3, "binning_map": {"choice:3-5": entry}}
    if other is not None:
        body["binning_map"]["noul:2"] = other
    return body


PAYLOAD_CASES = [
    # --- accepted ------------------------------------------------------------------------------
    _p("temperatures only", {"temperature": T3}),
    _p("temperatures and buckets", {"temperature": T3, "temperature_by_options": {"choice:3-5": 2.5, "noul:2": 0.8}}),
    _p("null buckets", {"temperature": T3, "temperature_by_options": None}),
    _p("empty buckets", {"temperature": T3, "temperature_by_options": {}}),
    _p("bucket names are not validated", {"temperature": T3, "temperature_by_options": {"bogus": 2.0, "": 3.0}}),
    _p("integers are numbers", {"temperature": [2, 3, 1], "temperature_by_options": {"noul:2": 2}}),
    _p("version 1", {"version": 1, "temperature": T3}),
    _p("null version is 1", {"version": None, "temperature": T3}),
    _p("a binning map", {"temperature": T3, "binning_map": {
        "choice:3-5": MAP4, "noul:2": {"bins": 2, "values": [0.2, 0.8]}, "score:2": {"bins": 1, "values": [0.5]}}}),
    _p("null binning map clears", {"temperature": T3, "binning_map": None}),
    _p("empty binning map is kept", {"temperature": T3, "binning_map": {}}),
    _p("values at the bounds", _bin({"bins": 3, "values": [0.0, 1.0, 0.5]})),
    _p("integer values", _bin({"bins": 2, "values": [0, 1]})),
    _p("a bucket name is not validated", {"temperature": T3, "binning_map": {"nonsense": MAP4}}),
    _p("a quote in a bucket name", {"temperature": T3, "binning_map": {"it's": MAP4}}),
    # --- clamping and its warning ---------------------------------------------------------------
    _p("clamp high and low", {"temperature": [9.0, 0.1, 1.0]}),
    _p("clamp boundaries are accepted", {"temperature": [0.5, 5.0, 1.0]}),
    _p("clamp a bucket", {"temperature": T3, "temperature_by_options": {"choice:3-5": 99.0, "noul:2": 0.01, "score:3-5": 2.0}}),
    _p("clamp temperatures and buckets in one warning",
       {"temperature": [9.0, 1.0, 0.0], "temperature_by_options": {"choice:2": -1.0, "noul:2": 7.5}}),
    _p("negative zero", {"temperature": [-0.0, 1.0, 1.0]}),
    _raw("non-finite temperatures", '{"temperature": [NaN, 1.0, Infinity], "temperature_by_options": {"noul:2": -Infinity}}'),
    _p("a bool is not a temperature", {"temperature": [True, 1.5, False]}),
    _p("a numeric string is read as float() reads it", {"temperature": ["2.5", " 1e1 ", "abc"]}),
    _p("null and nested values are 1.0", {"temperature": [None, [1.0], {"a": 1.0}]}),
    _p("bucket values that are not numbers",
       {"temperature": T3, "temperature_by_options": {"choice:2": True, "noul:2": None, "score:2": "2.5", "choice:6-10": 7}}),
    _p("a quote in a clamped bucket name", {"temperature": T3, "temperature_by_options": {"it's": 99.0}}),
    # --- version and the checkpoint it records ---------------------------------------------------
    _p("version 2 for this checkpoint", _v2()),
    _p("version 2 written as a float", _v2(version=2.0)),
    _p("version 2 as a string", _v2(version="2")),
    _p("version 2 as a padded string", _v2(version=" 2 ")),
    _p("version 2 as a signed string", _v2(version="+2")),
    _p("version 2 with an underscore", _v2(version="0_2")),
    _p("version 2 in fullwidth digits", _v2(version="２")),
    _p("version 3 is checked too", _v2(version=3)),
    _p("a huge version is checked", _v2(version=10 ** 40)),
    _p("version 1 is never checked", _v2(version=1, model_id_or_path="elsewhere")),
    _p("version 0 is never checked", _v2(version=0, model_id_or_path="elsewhere")),
    _p("negative version is never checked", _v2(version=-3, model_id_or_path="elsewhere")),
    _p("version 2 for another model", _v2(model_id_or_path="convaiinnovations/laya-multilingual")),
    _p("version 2 for another subfolder", _v2(subfolder="typed-decisions")),
    _p("version 2 recorded without a subfolder", _v2(subfolder=None)),
    _p("version 2 with another config", _v2(config=dict(RECORDED_CFG, max_len=512))),
    _p("version 2 with an extra config key", _v2(config=dict(RECORDED_CFG, extra=1))),
    _p("version 2 with a missing config key", _v2(config={k: v for k, v in RECORDED_CFG.items() if k != "encoder"})),
    _p("version 2 with a float where the config has an integer", _v2(config=dict(RECORDED_CFG, max_len=1024.0))),
    _p("version 2 with a config that is a list", _v2(config=[1, 2])),
    _p("version 2 with no config", {k: v for k, v in _v2().items() if k != "config"}),
    _p("version 2 with a null config", _v2(config=None)),
    _p("version 2 with nothing recorded", {"version": 2, "temperature": T3}),
    _p("version 2 with a model that is not a string", _v2(model_id_or_path=7)),
    _p("version 2 with a model that is a list", _v2(model_id_or_path=["a", "b"], subfolder={"k": 1.5})),
    _p("version 2 with a quote in the model", _v2(model_id_or_path="it's")),
    _p("version 2 for the English checkpoint", _v2(subfolder=None), "english"),
    _p("version 2 for the English checkpoint recorded with a subfolder", _v2(), "english"),
    _p("version 2 for a local directory", _v2(model_id_or_path="onnx/multilingual", subfolder=None), "local"),
    _p("version 2 for a local directory recorded for the hub", _v2(), "local"),
    _p("an identity warning and a clamp warning", _v2(temperature=[9.0, 1.0, 1.0], model_id_or_path="x")),
    # --- refusals: the payload ------------------------------------------------------------------
    _p("payload is a list", [1, 2, 3]),
    _p("payload is a number", 3),
    _p("payload is a float", 1.5),
    _p("payload is a string", "abc"),
    _p("payload is null", None),
    _p("payload is true", True),
    _p("payload is empty", {}),
    # --- refusals: version ----------------------------------------------------------------------
    _p("version is a bool", {"version": True, "temperature": T3}),
    _p("version is false", {"version": False, "temperature": T3}),
    _p("version is a list", {"version": [2], "temperature": T3}),
    _p("version is an object", {"version": {"v": 2}, "temperature": T3}),
    _p("version is not whole", {"version": 1.9, "temperature": T3}),
    _p("version is not whole and large", {"version": 2.5, "temperature": T3}),
    _p("version is a word", {"version": "abc", "temperature": T3}),
    _p("version is an empty string", {"version": "", "temperature": T3}),
    _p("version is a decimal string", {"version": "1.5", "temperature": T3}),
    _p("version is a double underscore string", {"version": "1__2", "temperature": T3}),
    _p("version has a quote", {"version": "it's", "temperature": T3}),
    _raw("version is NaN", '{"version": NaN, "temperature": [1.0, 1.0, 1.0]}'),
    _raw("version is Infinity", '{"version": Infinity, "temperature": [1.0, 1.0, 1.0]}'),
    _raw("version is an overflowing float", '{"version": 1e999, "temperature": [1.0, 1.0, 1.0]}'),
    _raw("version is a huge whole float", '{"version": 1e300, "temperature": [1.0, 1.0, 1.0]}'),
    # --- refusals: temperatures -----------------------------------------------------------------
    _p("temperature is missing", {"version": 1}),
    _p("temperature is null", {"temperature": None}),
    _p("temperature has two values", {"temperature": [1.0, 2.0]}),
    _p("temperature has four values", {"temperature": [1.0, 2.0, 3.0, 4.0]}),
    _p("temperature is empty", {"temperature": []}),
    _p("temperature is a number", {"temperature": 2.0}),
    _p("temperature is a string of three", {"temperature": "abc"}),
    _p("temperature is an object of three", {"temperature": {"a": 1, "b": 2, "c": 3}}),
    _p("buckets are a list", {"temperature": T3, "temperature_by_options": [1.0]}),
    _p("buckets are a number", {"temperature": T3, "temperature_by_options": 3}),
    _p("buckets are a float", {"temperature": T3, "temperature_by_options": 1.5}),
    _p("buckets are a string", {"temperature": T3, "temperature_by_options": "x"}),
    _p("buckets are false", {"temperature": T3, "temperature_by_options": False}),
    _p("buckets are an empty list", {"temperature": T3, "temperature_by_options": []}),
    # --- refusals: the binning map --------------------------------------------------------------
    _p("binning map is a list", {"temperature": T3, "binning_map": []}),
    _p("binning map is a number", {"temperature": T3, "binning_map": 3}),
    _p("binning map is a string", {"temperature": T3, "binning_map": "x"}),
    _p("binning map is false", {"temperature": T3, "binning_map": False}),
    _p("entry is a list", {"temperature": T3, "binning_map": {"choice:3-5": [0.1, 0.9]}}),
    _p("entry is null", {"temperature": T3, "binning_map": {"choice:3-5": None}}),
    _p("entry is a number", {"temperature": T3, "binning_map": {"choice:3-5": 0.5}}),
    _p("entry is a string", {"temperature": T3, "binning_map": {"choice:3-5": "x"}}),
    _p("entry is empty", {"temperature": T3, "binning_map": {"choice:3-5": {}}}),
    _p("entry is true", {"temperature": T3, "binning_map": {"choice:3-5": True}}),
    _p("a quote in a refused bucket name", {"temperature": T3, "binning_map": {"it's": 3}}),
    _p("bins is missing", _bin({"values": [0.1, 0.9]})),
    _p("bins is null", _bin({"bins": None, "values": [0.1, 0.9]})),
    _p("bins is zero", _bin({"bins": 0, "values": []})),
    _p("bins is negative", _bin({"bins": -2, "values": [0.1, 0.9]})),
    _p("bins is a whole float", _bin({"bins": 2.0, "values": [0.1, 0.9]})),
    _p("bins is a float", _bin({"bins": 2.5, "values": [0.1, 0.9]})),
    _p("bins is a string", _bin({"bins": "2", "values": [0.1, 0.9]})),
    _p("bins is true", _bin({"bins": True, "values": [0.1]})),
    _p("bins is a list", _bin({"bins": [2], "values": [0.1, 0.9]})),
    _p("bins is a huge integer", _bin({"bins": 10 ** 30, "values": [0.1, 0.9]})),
    _p("bins is a negative huge integer", _bin({"bins": -10 ** 30, "values": [0.1, 0.9]})),
    _raw("bins is NaN", '{"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": NaN, "values": [0.5]}}}'),
    _p("values is missing", _bin({"bins": 2})),
    _p("values is null", _bin({"bins": 2, "values": None})),
    _p("values is too short", _bin({"bins": 3, "values": [0.1, 0.9]})),
    _p("values is too long", _bin({"bins": 1, "values": [0.1, 0.9]})),
    _p("values is a string", _bin({"bins": 3, "values": "abc"})),
    _p("values is an object of the right size", _bin({"bins": 2, "values": {"a": 0.1, "b": 0.9}})),
    _p("values is a number", _bin({"bins": 1, "values": 0.5})),
    _p("a value is null", _bin({"bins": 2, "values": [0.1, None]})),
    _p("a value is a string", _bin({"bins": 2, "values": [0.1, "0.9"]})),
    _p("a value is a word", _bin({"bins": 2, "values": [0.1, "abc"]})),
    _p("a value is true", _bin({"bins": 2, "values": [0.1, True]})),
    _p("a value is false", _bin({"bins": 2, "values": [False, 0.5]})),
    _p("a value is a list", _bin({"bins": 2, "values": [0.1, [0.9]]})),
    _p("a value is an object", _bin({"bins": 2, "values": [{"v": 0.1}, 0.5]})),
    _raw("a value is NaN", '{"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 2, "values": [0.1, NaN]}}}'),
    _raw("a value is Infinity",
         '{"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 2, "values": [0.1, Infinity]}}}'),
    _raw("a value is negative Infinity",
         '{"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [-Infinity]}}}'),
    _raw("a value overflows to Infinity",
         '{"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [1e999]}}}'),
    _p("a value is below zero", _bin({"bins": 2, "values": [-0.1, 0.5]})),
    _p("a value is above one", _bin({"bins": 2, "values": [0.5, 1.0000001]})),
    _p("a value is an integer above one", _bin({"bins": 2, "values": [0.5, 2]})),
    _p("a value is -0.0", _bin({"bins": 1, "values": [-0.0]})),
    _p("a value is a huge integer", _bin({"bins": 2, "values": [0.5, 10 ** 400]})),
    _p("a huge integer value after a bad one", _bin({"bins": 3, "values": [None, 10 ** 400, 0.5]})),
    _p("a bad value after a huge integer", _bin({"bins": 3, "values": [10 ** 400, None, 0.5]})),
    _p("a string value before a list value", _bin({"bins": 2, "values": ["0.5", [1]]})),
    _p("an out-of-range value before a finite one", _bin({"bins": 2, "values": [2.0, 0.5]})),
    _raw("an out-of-range value before a non-finite one",
         '{"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 2, "values": [2.0, NaN]}}}'),
    _p("the second entry is the bad one", _bin(MAP4, {"bins": 2, "values": [0.1, 5.0]})),
    _p("the first entry is the bad one", _bin({"bins": 2, "values": [0.1, 5.0]}, {"bins": 2, "values": [0.2, "x"]})),
    _p("a refusal beats a temperature warning",
       {"temperature": [9.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [3.0]}}}),
    _p("a refusal beats an identity warning", _v2(binning_map={"a": {"bins": 1, "values": [3.0]}})),
    _raw("a duplicate key keeps its last value",
         '{"temperature": [1.0, 2.0, 3.0], "temperature": [4.0, 5.0, 1.5], "binning_map": {"a": {"bins": 1, "values": [0.5]}, '
         '"a": {"bins": 2, "values": [0.1, 0.9]}}}'),
    # --- an integer too large for a float -------------------------------------------------------
    _p("a temperature that is a huge integer", {"temperature": [10 ** 400, 1.0, 1.0]}),
    _p("a bucket temperature that is a huge integer", {"temperature": T3, "temperature_by_options": {"noul:2": 10 ** 400}}),
    _p("a huge integer after an identity warning", _v2(temperature=[10 ** 400, 1.0, 1.0], model_id_or_path="x")),
]

BIN4 = {"bins": 4, "values": [0.1, 0.4, 0.6, 0.9]}
BIN1 = {"bins": 1, "values": [0.5]}
BIN10 = {"bins": 10, "values": [round(i / 10 + 0.05, 2) for i in range(10)]}
BIN100 = {"bins": 100, "values": [round((i * 37 % 100) / 100, 2) for i in range(100)]}
MAP = {"choice:3-5": BIN4, "noul:2": {"bins": 2, "values": [0.25, 0.75]}, "score:2": BIN1, "choice:6-10": BIN10,
       "choice:11+": BIN100}

MAPS = {"main": MAP, "empty": {}}

LOOKUP_CASES = (
    # bin edges and the ends
    [("choice:3-5 at %r" % c, "main", "choice:3-5", c)
     for c in (0.0, 1e-300, 0.1, 0.2499999999999999, 0.25, 0.2500000000000001, 0.3, 0.4999999999999999, 0.5,
               0.5000000000000001, 0.6, 0.7499999999999999, 0.75, 0.7500000000000001, 0.9, 0.9999999999999999, 1.0)]
    # outside [0, 1]: clipped by the bin index, not rejected
    + [("choice:3-5 at %r" % c, "main", "choice:3-5", c) for c in (-0.0, -0.1, -1e300, 1.0000000000000002, 1.5, 7.0, 1e300)]
    # one bin takes everything
    + [("one bin at %r" % c, "main", "score:2", c) for c in (0.0, 0.5, 1.0)]
    # a bucket the map was not fit for passes through, with whatever the caller holds
    + [("miss at %r" % c, "main", "score:3-5", c) for c in (0.0, 0.123456789, 0.5, 1.0, 1.5, -0.25)]
    + [("miss on an empty bucket name", "main", "", 0.4), ("miss on a nearly right name", "main", "choice:3-5 ", 0.4),
       ("miss on another case", "main", "CHOICE:3-5", 0.4), ("empty map", "empty", "choice:3-5", 0.4)]
    # ten bins: float products that fall just under a whole number are truncated, not rounded
    + [("ten bins at %r" % c, "main", "choice:6-10", c) for c in (0.1, 0.2, 0.3, 0.7, 0.57, 0.9, 0.95, 0.99, 0.999999)]
    + [("hundred bins at %r" % c, "main", "choice:11+", c) for c in (0.01, 0.07, 0.29, 0.57, 0.58, 0.99, 1.0)]
    + [("two bins at %r" % c, "main", "noul:2", c) for c in (0.5, 0.7499999999999999, 0.75, 1.0)]
    # a value float() takes and int() refuses
    + [("nan", "main", "choice:3-5", float("nan")), ("inf", "main", "choice:3-5", float("inf")),
       ("-inf", "main", "choice:3-5", float("-inf")), ("nan on a miss", "main", "score:3-5", float("nan")),
       ("inf on a miss", "main", "score:3-5", float("inf"))]
)
