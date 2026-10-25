"""Record golden parity vectors for the Rust SDK from the shipping Python code path.

The point of this script is that it does *not* re-derive anything. It builds a real
`laya.Agent` via `Agent.__new__` and fills in only the attributes `system_one` reads,
installing an ONNX-backed callable as `self.model`. The genuine `Agent.system_one`,
`build_sequence`, `render_options`, `render_criterion` and `confidence_from_probs` then run
unmodified, so every number written here is what Python actually answers -- not a
paraphrase of it. It also means no 1.3 GB safetensors checkpoint is needed: the exported
ONNX graph supplies the forward pass.

Long cases run the genuine `ONNXAgent.predict_long`, sharing that same verified session. Batch cases
run the genuine `ONNXAgent.predict_batch` the same way.

Before writing anything the script checks option logits and unrounded action probabilities
against the PyTorch reference vectors bundled in `fixtures.npz`. Raw action logits are
retained in the goldens and the self-check report for diagnostics.

Usage:
    # Explicit paths (original interface, still works):
    python laya-rust/tools/dump_golden.py --onnx onnx/multilingual --out laya-rust/tests/golden/multilingual

    # Shorthand: derive both from the checkpoint name:
    python laya-rust/tools/dump_golden.py --checkpoint multilingual
    python laya-rust/tools/dump_golden.py --checkpoint english
    python laya-rust/tools/dump_golden.py --checkpoint typed-decisions
    python laya-rust/tools/dump_golden.py --checkpoint all    # runs all three in sequence
"""

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, REPO)
sys.path.insert(0, TOOLS)

from golden_cases import (  # noqa: E402
    BATCH_CASES,
    BATCH_ERROR_CASES,
    CASES,
    ERROR_CASES,
    LONG_CASES,
    LONG_ERROR_CASES,
    MIN_CONFIDENCE_CASES,
    MIN_CONFIDENCE_ERROR_CASES,
    SHORTLIST_E2E_CASES,
)
from repr_cases import build_repr_probe  # noqa: E402

import hashing_embedder  # noqa: E402

from laya.agent import Agent, _fix_tokenizer_config  # noqa: E402
from laya.common import (  # noqa: E402
    QTYPES,
    build_sequence,
    clamp_temperature,
    render_criterion,
    render_options,
    serialize_state,
)

NP_DTYPES = {
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
    "tensor(float)": np.float32,
}

# Canonical checkpoint names and their onnx subdirectory names.
CHECKPOINTS = {
    "english": "english",
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}

# Strings the port's tokenizer must encode identically with add_special_tokens=False. Anything
# that diverges here shifts every marker position downstream, so this probe is the cheapest
# place to catch a tokenizer mismatch -- it needs no model at all.
TOKENIZER_PROBE = [
    "",
    " ",
    "  ",
    "\n",
    "\t",
    "a",
    " a",
    "hello world",
    " hello world",
    "hello  world",
    " leading and trailing ",
    "choice question: Which department should handle this?",
    "score question: How urgent is this request?",
    "noul question: The customer threatens to cancel the account.",
    " billing: invoices, payments, double charges, refunds",
    " level 0: not urgent",
    " false: no, the statement does not hold",
    " true: yes, the statement holds",
    "▁",
    "▁▁",
    "a▁b",
    "CamelCase and_snake_case and-kebab-case",
    "1234567890 3.14159 -0.5 1e21",
    "punctuation: !?.,;:'\"()[]{}<>/\\|@#$%^&*~`+=_",
    "naïve résumé Müller",
    "我这个月被重复扣款了两次",
    "こんにちは世界",
    "여보세요",
    "тест русского",
    "تم خصم المبلغ مرتين",
    "שלום עולם",
    "हिन्दी पाठ",
    "\U0001f600 \U0001f680 ❤️",
    "zero​width​space",
    "<mask>",
    "<bos> <eos> <pad> <unk>",
    "a" * 300,
]

# Values fed through each json.dumps dialect, recorded as the exact strings Python produces.
# No other language's default JSON serializer reproduces all three dialects, so each port's
# serializer is checked against these rather than against a second reimplementation.
JSON_PROBE = [
    "plain string",
    {"a": 1},
    {"a": 1, "b": "two"},
    {"nested": {"x": [1, 2, 3]}},
    [1, 2, 3],
    ["a", "b"],
    [],
    {},
    0,
    0.0,
    1.0,
    2.0,
    -0.125,
    0.5,
    1e21,
    1e-7,
    0.3333333333333333,
    # Pins Python's repr thresholds for switching to exponent notation, which differ from
    # other languages' default float formatting and are the one place a float can silently
    # change the token stream.
    1e15,
    1e16,
    1e17,
    1e-4,
    1e-5,
    123456789012345.6,
    -0.0,
    5e-324,
    1.7976931348623157e308,
    1234567890123456789,
    True,
    False,
    None,
    "quote \" backslash \\ slash /",
    "html < > & ' chars",
    "newline \n tab \t return \r",
    "control chars " + chr(1) + " " + chr(31),
    "accents naïve résumé Müller",
    "我这个月",
    "\U0001f600 emoji",
    {"mixed": "Müller & co <urgent>", "n": 2.0, "flag": True, "none": None},
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
]


class OnnxForward:
    """Stands in for `DecisionModel.forward`, backed by the exported graph.

    `system_one` hands it torch tensors and expects torch tensors back, so this converts at
    the boundary and keeps a copy of the last call for the goldens.
    """

    def __init__(self, session: ort.InferenceSession):
        self.sess = session
        self.dtypes = {i.name: NP_DTYPES[i.type] for i in session.get_inputs()}
        self.last = None
        self.padded_markers = False

    # The tracer baked `TopK k=2` into the graph: Python picks the top-2 branch at runtime
    # (common.py:122) and the export only ever saw >=2 markers, so a batch whose questions all
    # have a single option makes TopK fail. Padding the marker axis to 2 with marker_mask=False
    # is exactly equivalent, not merely close: masked_fill puts -1e4 in the pad column, which
    # underflows softmax to 0.0, so p == [1.0, 0.0] -- precisely the [top1, 0.0] Python
    # substitutes -- and k = marker_mask.sum().clamp(min=2) == 2 either way. Every port's
    # Collator pads identically, which is why the recorded tensors are the padded ones.
    MIN_MARKERS = 2

    def _pad_markers(self, marker_pos, marker_mask):
        k = marker_pos.shape[1]
        if k >= self.MIN_MARKERS:
            return marker_pos, marker_mask, False
        pad = self.MIN_MARKERS - k
        marker_pos = torch.cat([marker_pos, torch.zeros((marker_pos.shape[0], pad), dtype=marker_pos.dtype)], 1)
        marker_mask = torch.cat([marker_mask, torch.zeros((marker_mask.shape[0], pad), dtype=torch.bool)], 1)
        return marker_pos, marker_mask, True

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        marker_pos, marker_mask, padded = self._pad_markers(marker_pos, marker_mask)
        self.padded_markers = padded
        supplied = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": qtype,
        }
        missing = set(self.dtypes) - set(supplied)
        if missing:
            raise RuntimeError("graph expects unknown inputs %s" % sorted(missing))
        feeds = {n: t.cpu().numpy().astype(self.dtypes[n]) for n, t in supplied.items()}
        logits, act_logits = self.sess.run(["logits", "act_logits"], feeds)
        self.last = {"feeds": feeds, "logits": logits, "act_logits": act_logits}
        return torch.from_numpy(logits), torch.from_numpy(act_logits)


def build_agent(onnx_dir: str):
    """A real Agent whose forward pass is the ONNX graph rather than torch weights."""
    from transformers import AutoTokenizer

    _fix_tokenizer_config(onnx_dir)
    with open(os.path.join(onnx_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    tok = AutoTokenizer.from_pretrained(os.path.join(onnx_dir, "tokenizer"))
    sess = ort.InferenceSession(
        os.path.join(onnx_dir, "model.onnx"),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )

    agent = Agent.__new__(Agent)
    agent.cfg = cfg
    agent.tok = tok
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.model = OnnxForward(sess)
    # Same clamping Agent.__init__ applies; the goldens must reflect shipped behaviour.
    agent.temperature = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
    agent.temperature_by_options = {
        k: clamp_temperature(v) for k, v in cfg.get("temperature_by_options", {}).items()
    }
    return agent


def self_check(agent: Agent, onnx_dir: str) -> dict:
    """Refuse to emit goldens unless the session reproduces the bundled PyTorch reference."""
    path = os.path.join(onnx_dir, "fixtures.npz")
    if not os.path.exists(path):
        raise SystemExit("missing %s -- cannot verify the ONNX session, refusing to emit goldens" % path)

    z = np.load(path)
    groups = sorted({k.split("/")[0] for k in z.files})
    report = {}
    for g in groups:
        need = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
        if not all("%s/%s" % (g, n) in z.files for n in need):
            continue
        logits, act = agent.model(*[torch.from_numpy(z["%s/%s" % (g, n)]) for n in need])
        ref_logits = z["%s/ref_logits" % g]
        ref_act = z["%s/ref_act" % g]
        for actual, reference in [(logits.numpy(), ref_logits), (act.numpy(), ref_act)]:
            if actual.shape != reference.shape or not (
                np.isfinite(actual).all() and np.isfinite(reference).all()
            ):
                raise SystemExit("invalid ONNX/reference outputs for %r -- refusing to emit goldens" % g)
        d_logits = float(np.abs(logits.numpy() - ref_logits).max())
        d_act = float(np.abs(act.numpy() - ref_act).max())
        # Agent._forward consumes softmax(act), not raw act logits. Check every
        # action probability before rounding: saturated logits in the thousands
        # can drift across CPU kernels without changing that signal at all.
        # Do not mask action columns: they are classes, not padded option slots.
        act_probs = torch.softmax(act.float(), -1).numpy()
        ref_probs = torch.softmax(torch.from_numpy(ref_act).float(), -1).numpy()
        d_act_probs = float(np.abs(act_probs - ref_probs).max())
        report[g] = {"logits_maxdiff": d_logits, "act_maxdiff": d_act,
                     "act_probs_maxdiff": d_act_probs}
        if d_logits > 1e-3 or d_act_probs > 1e-4:
            raise SystemExit(
                "ONNX session does not match the PyTorch reference for %r "
                "(logits %.3e, act %.3e, act_probs %.3e) -- refusing to emit goldens"
                % (g, d_logits, d_act, d_act_probs)
            )
        print("  self-check %-14s logits %.3e  act %.3e  act_probs %.3e  OK"
              % (g, d_logits, d_act, d_act_probs))
    if not report:
        raise SystemExit("fixtures.npz contained no usable reference groups")
    return report


def describe_questions(agent: Agent, state, questions: dict) -> list:
    """Per-question tokenisation, recomputed exactly as system_one does it.

    Recorded separately from the answers so a port's SequenceBuilder can be checked with no
    model present -- tokenisation is where a port silently drifts.
    """
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    mask_tok = agent.tok.mask_token
    out = []
    for qid, qdef in questions.items():
        q = Agent._to_internal(qdef)
        seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        out.append(
            {
                "id": qid,
                "type": q["t"],
                "qtype": QTYPES[q["t"]],
                "instructions": q["ins"],
                "head_text": "%s question: %s" % (q["t"], str(q["ins"]).replace(mask_tok, " ")),
                "options": render_options(q),
                "input_ids": [int(i) for i in seq],
                "markers": [int(m) for m in markers],
            }
        )
    return out


def record_case(agent: Agent, case: dict) -> dict:
    state, questions = case["state"], case["questions"]
    per_question = describe_questions(agent, state, questions)
    options = case.get("options")
    result = agent.system_one(state, questions, **(options or {}))
    last = agent.model.last
    rec = {
        "name": case["name"],
        "state": state,
        "serialized_state": serialize_state(state),
        "questions": questions,
        "per_question": per_question,
        "batch": {
            "input_ids": last["feeds"]["input_ids"].tolist(),
            "attention_mask": last["feeds"]["attention_mask"].tolist(),
            "marker_pos": last["feeds"]["marker_pos"].tolist(),
            "marker_mask": last["feeds"]["marker_mask"].tolist(),
            "qtype": last["feeds"]["qtype"].tolist(),
        },
        "marker_axis_padded": agent.model.padded_markers,
        "logits": last["logits"].tolist(),
        "act_logits": last["act_logits"].tolist(),
        "result": result,
    }
    if options:
        rec["options"] = options
    return rec


def record_shortlist_case(agent: Agent, case: dict) -> dict:
    """Shortlist end to end: predict_shortlist ranks `case["questions"]`'s choice question(s)
    down to `case["k"]` with the deterministic hashing embedder, then runs the real checkpoint
    on the reduced criteria. `per_question`/`batch`/`logits`/`act_logits` describe the reduced
    question set actually scored (not the original 40-option one), because that is what a
    port's SequenceBuilder needs to check itself against -- reconstructed the same way
    `predict_shortlist` builds it internally (`_subset_criteria` on the kept labels), which is
    guaranteed to match what the model actually saw since `agent.model.last` (captured by
    `OnnxForward.__call__`) reflects that same reduced call.
    """
    from laya.shortlist import _subset_criteria, predict_shortlist

    state, questions, k = case["state"], case["questions"], case["k"]
    result = predict_shortlist(agent, state, questions, hashing_embedder.embed, k=k)
    shortlist_meta = result["shortlist"]

    reduced = {}
    for qid, qdef in questions.items():
        meta = shortlist_meta.get(qid)
        if meta is not None and not meta["passthrough"]:
            updated = dict(qdef)
            updated["criteria"] = _subset_criteria(qdef["criteria"], meta["labels"])
            reduced[qid] = updated
        else:
            reduced[qid] = qdef

    per_question = describe_questions(agent, state, reduced)
    last = agent.model.last
    return {
        "name": case["name"],
        "state": state,
        "serialized_state": serialize_state(state),
        "questions": questions,
        "shortlist_k": k,
        "reduced_questions": reduced,
        "per_question": per_question,
        "batch": {
            "input_ids": last["feeds"]["input_ids"].tolist(),
            "attention_mask": last["feeds"]["attention_mask"].tolist(),
            "marker_pos": last["feeds"]["marker_pos"].tolist(),
            "marker_mask": last["feeds"]["marker_mask"].tolist(),
            "qtype": last["feeds"]["qtype"].tolist(),
        },
        "marker_axis_padded": agent.model.padded_markers,
        "logits": last["logits"].tolist(),
        "act_logits": last["act_logits"].tolist(),
        "result": result,
    }


def record_error_case(agent: Agent, case: dict) -> dict:
    options = case.get("options")
    try:
        agent.system_one(case["state"], case["questions"], **(options or {}))
    except ValueError as e:
        rec = {
            "name": case["name"],
            "state": case["state"],
            "questions": case["questions"],
            "error_type": "ValueError",
            "error_message": str(e),
        }
        if options:
            rec["options"] = options
        return rec
    raise SystemExit("case %r was expected to raise ValueError but did not" % case["name"])


def record_long_case(agent: Agent, case: dict, expects_error: bool = False) -> dict:
    """Run the genuine ONNXAgent scan, sharing the already verified session/tokenizer."""
    from laya.common import encode_text, window_budget
    from laya.onnx_agent import ONNXAgent

    onnx = ONNXAgent.__new__(ONNXAgent)
    onnx.cfg, onnx.tok, onnx.session = agent.cfg, agent.tok, agent.model.sess
    onnx.temperature = agent.temperature
    onnx.temperature_by_options = agent.temperature_by_options
    onnx.lang_temperatures = {}
    options = case.get("options", {})
    state, questions = case.get("state"), case["questions"]
    rec = {"name": case["name"], "state": state, "questions": questions, "options": options}
    if expects_error:
        try:
            onnx.predict_long(state, questions, **options)
        except ValueError as e:
            rec.update(error_type="ValueError", error_message=str(e))
            return rec
        raise SystemExit("long case %r was expected to raise ValueError but did not" % case["name"])

    internal = [onnx._to_internal(q) for q in questions.values()]
    budget, step, room = window_budget(
        onnx.tok, internal, onnx.cfg.get("max_len", 512), onnx.cfg.get("head_max_len", 192),
        window=options.get("window"), stride=options.get("stride"),
    )
    # predict_long's one-pass bound: an explicit window, else the room the questions leave.
    single = budget if (options.get("window") or 0) > 0 else room
    if case.get("state_at_room") and room <= budget:
        raise SystemExit("long case %r needs room (%d) > window (%d)" % (case["name"], room, budget))
    target = budget if case.get("state_at_limit") else room if case.get("state_at_room") else None
    if target is not None:
        state = " ".join(["a"] * target)
        rec["state"] = state
    state_ids = encode_text(
        onnx.tok, serialize_state(state).replace(onnx.tok.mask_token, " "), add_special_tokens=False,
    )["input_ids"]
    if target is not None and len(state_ids) != target:
        raise SystemExit("boundary state is %d tokens, expected exactly %d" % (len(state_ids), target))
    rec.update(state_tokens=len(state_ids), effective_window=budget, effective_stride=step,
               effective_room=room)
    # Record decoded windows too: this pins cleanup, special tokens and boundary drift independently
    # of rounded model scores. The single-window fast path must not decode/re-encode the state.
    windows = []
    if len(state_ids) > single:
        for start in range(0, len(state_ids), step):
            windows.append({"token_start": start, "token_end": min(start + budget, len(state_ids)),
                            "text": onnx.tok.decode(state_ids[start:start + budget])})
            if start + budget >= len(state_ids):
                break
    rec["windows"] = windows
    rec["result"] = onnx.predict_long(state, questions, **options)
    return rec


class RecordingSession:
    """Forwards to the verified ONNX session and notes each run's input shape, so a batch case
    can record how `predict_batch` actually chunked and ordered its states."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.runs = []

    def run(self, names, feeds):
        self.runs.append([int(d) for d in feeds["input_ids"].shape])
        return self.session.run(names, feeds)


def record_batch_case(agent: Agent, case: dict, expects_error: bool = False) -> dict:
    """Run the genuine ONNXAgent.predict_batch, sharing the already verified session/tokenizer."""
    from laya.onnx_agent import ONNXAgent

    session = RecordingSession(agent.model.sess)
    onnx = ONNXAgent.__new__(ONNXAgent)
    onnx.cfg, onnx.tok, onnx.session = agent.cfg, agent.tok, session
    onnx.temperature = agent.temperature
    onnx.temperature_by_options = agent.temperature_by_options
    onnx.lang_temperatures = {}
    onnx.hooks = []
    onnx.model_id = "golden"
    states, questions = case["states"], case["questions"]
    options = case.get("options", {})
    rec = {"name": case["name"], "states": states, "questions": questions, "options": options}
    if expects_error:
        expected = {"TypeError": TypeError, "ValueError": ValueError}[case["error_type"]]
        try:
            onnx.predict_batch(states, questions, **options)
        except expected as e:
            rec.update(error_type=type(e).__name__, error_message=str(e))
            return rec
        raise SystemExit("batch case %r was expected to raise %s but did not" % (case["name"], case["error_type"]))

    # Each state's widest question row: the key sort_by_length orders by. Recorded separately from
    # the results so the C# chunk planning can be checked against Python's own token counts.
    widths = []
    if questions and states:
        ids = list(questions.keys())
        internal = {qid: onnx._to_internal(questions[qid]) for qid in ids}
        for state in states:
            items = onnx._encode_state(state, ids, internal, onnx.cfg.get("max_len", 512),
                                       onnx.cfg.get("head_max_len", 192))
            widths.append(max(len(item["ids"]) for item in items))
    rec["state_widths"] = widths
    session.runs.clear()
    rec["results"] = onnx.predict_batch(states, questions, **options)
    rec["runs"] = list(session.runs)
    return rec


def dump(obj, path: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    print("  wrote %-28s %8.1f KB" % (os.path.basename(path), os.path.getsize(path) / 1024.0))


def run_checkpoint(onnx_dir: str, out_dir: str) -> int:
    """Generate and write golden data for one checkpoint. Returns 0 on success."""
    onnx_dir = os.path.abspath(onnx_dir)
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(onnx_dir):
        raise SystemExit("no such ONNX directory: %s" % onnx_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("loading %s" % onnx_dir)
    agent = build_agent(onnx_dir)
    check = self_check(agent, onnx_dir)

    tok = agent.tok
    meta = {
        "generator": "tools/dump_golden.py",
        "onnx_dir": os.path.basename(onnx_dir),
        "laya_version": __import__("laya").__version__,
        "onnxruntime_version": ort.__version__,
        "transformers_version": __import__("transformers").__version__,
        "config": agent.cfg,
        "max_len": agent.cfg.get("max_len", 512),
        "head_max_len": agent.cfg.get("head_max_len", 192),
        "temperature": agent.temperature,
        "temperature_by_options": agent.temperature_by_options,
        "qtypes": QTYPES,
        "special_tokens": {
            "pad_id": tok.pad_token_id,
            "cls_id": tok.cls_token_id,
            "sep_id": tok.sep_token_id,
            "mask_id": tok.mask_token_id,
            "unk_id": tok.unk_token_id,
            "mask_token": tok.mask_token,
            "cls_token": tok.cls_token,
            "sep_token": tok.sep_token,
            "pad_token": tok.pad_token,
        },
        "onnx_self_check": check,
    }
    dump(meta, os.path.join(out_dir, "meta.json"))

    dump(
        [
            {"text": s, "input_ids": [int(i) for i in tok(s, add_special_tokens=False)["input_ids"]]}
            for s in TOKENIZER_PROBE
        ],
        os.path.join(out_dir, "tokenizer_probe.json"),
    )

    # Every dialect for every value, so a port's test can assert the right one is used where it
    # matters and see exactly how the three differ.
    dump(
        [
            {
                "value": v,
                # serialize_state accepts any value at runtime: a string passes through, and
                # everything else reaches json.dumps. Record the actual output unconditionally
                # so the golden matches what Python would write for every probe value.
                "state_dialect": serialize_state(v),
                "criterion_dialect": render_criterion(v),
                "instructions_dialect": v if isinstance(v, str) else json.dumps(v, ensure_ascii=False),
            }
            for v in JSON_PROBE
        ],
        os.path.join(out_dir, "json_probe.json"),
    )

    index = []
    for case in CASES:
        rec = record_case(agent, case)
        dump(rec, os.path.join(out_dir, "case_%s.json" % case["name"]))
        index.append({"name": case["name"], "file": "case_%s.json" % case["name"], "expects_error": False})
    for case in ERROR_CASES:
        rec = record_error_case(agent, case)
        dump(rec, os.path.join(out_dir, "case_%s.json" % case["name"]))
        index.append({"name": case["name"], "file": "case_%s.json" % case["name"], "expects_error": True})

    # Appended last, after every pre-existing case, so the index entries above keep their
    # exact prior order and index.json's diff is a pure append.
    for case in SHORTLIST_E2E_CASES:
        rec = record_shortlist_case(agent, case)
        dump(rec, os.path.join(out_dir, "case_%s.json" % case["name"]))
        index.append({
            "name": case["name"], "file": "case_%s.json" % case["name"],
            "expects_error": False, "kind": "shortlist",
        })

    for cases, expects_error in ((LONG_CASES, False), (LONG_ERROR_CASES, True)):
        for case in cases:
            rec = record_long_case(agent, case, expects_error)
            dump(rec, os.path.join(out_dir, "case_%s.json" % case["name"]))
            index.append({
                "name": case["name"], "file": "case_%s.json" % case["name"],
                "expects_error": expects_error, "kind": "long",
            })

    for cases, expects_error in ((BATCH_CASES, False), (BATCH_ERROR_CASES, True)):
        for case in cases:
            rec = record_batch_case(agent, case, expects_error)
            dump(rec, os.path.join(out_dir, "case_%s.json" % case["name"]))
            index.append({
                "name": case["name"], "file": "case_%s.json" % case["name"],
                "expects_error": expects_error, "kind": "batch",
            })

    # Appended after the batch cases, so index.json's diff stays a pure append. No `kind`: these are
    # standard system_one cases whose recorded `options` the parity tests forward.
    for cases, expects_error in ((MIN_CONFIDENCE_CASES, False), (MIN_CONFIDENCE_ERROR_CASES, True)):
        for case in cases:
            rec = (record_error_case if expects_error else record_case)(agent, case)
            dump(rec, os.path.join(out_dir, "case_%s.json" % case["name"]))
            index.append({"name": case["name"], "file": "case_%s.json" % case["name"],
                          "expects_error": expects_error})

    dump(index, os.path.join(out_dir, "index.json"))
    dump(build_repr_probe(), os.path.join(out_dir, "repr_probe.json"))
    print("done: %d cases -> %s" % (len(index), out_dir))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Original interface: explicit paths.
    ap.add_argument(
        "--onnx",
        default=None,
        help="ONNX artifact directory (e.g. onnx/multilingual). "
             "When --checkpoint is given, defaults to onnx/<checkpoint>.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output directory for the golden files. "
             "When --checkpoint is given, defaults to laya-rust/tests/golden/<checkpoint>.",
    )
    # Shorthand: derive both from the checkpoint name.
    ap.add_argument(
        "--checkpoint",
        choices=list(CHECKPOINTS) + ["all"],
        default=None,
        metavar="{%s,all}" % ",".join(CHECKPOINTS),
        help=(
            "Checkpoint name. Derives --onnx and --out automatically. "
            "Pass 'all' to regenerate all three in sequence."
        ),
    )
    args = ap.parse_args()

    golden_root = os.path.join(REPO, "laya-rust", "tests", "golden")

    if args.checkpoint == "all":
        # Regenerate all three checkpoints in sequence.
        for name in CHECKPOINTS:
            print("\n=== checkpoint: %s ===" % name)
            onnx_dir = args.onnx or os.path.join(REPO, "onnx", name)
            out_dir = args.out or os.path.join(golden_root, name)
            run_checkpoint(onnx_dir, out_dir)
        return 0

    if args.checkpoint is not None:
        name = args.checkpoint
        onnx_dir = args.onnx or os.path.join(REPO, "onnx", name)
        out_dir = args.out or os.path.join(golden_root, name)
    else:
        # Legacy: --onnx and --out must both be provided (or default to multilingual).
        onnx_dir = args.onnx or os.path.join(REPO, "onnx", "multilingual")
        out_dir = args.out or os.path.join(golden_root, "multilingual")

    return run_checkpoint(onnx_dir, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
