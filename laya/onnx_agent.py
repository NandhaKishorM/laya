import json
import os
import threading
import time
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np

from laya.hooks import (
    HookRegistry, PredictContext, aggregate_usage, compose_hooks, dispatch, normalise_hooks,
    validate_timeout,
)
from laya.revisions import resolve_revision, snapshot_revision, verify_digests
from laya.common import (
    QTYPES,
    answer_confidence,
    build_sequence,
    collapsed_options,
    collate_items,
    confidence_from_probs,
    encode_text,
    render_options,
    serialize_state,
    temp_bucket,
    TEMP_MIN,
    TEMP_MAX,
    clamp_temperature,
)


class ONNXAgent(HookRegistry):
    """System 1 decision model runtime via ONNX: fast CPU-optimized decisions."""

    # Opt-in hooks; defaults keep a hand-built instance working and make an unset hook a no-op.
    # `hooks`/`_hooks_mutex` come from HookRegistry.
    hooks_raise = True
    hooks_concurrent = True
    hooks_timeout = None
    _hooks_lock = None
    model_id = None

    def __init__(
        self,
        model_id_or_path: str,
        onnx_path: str = "laya.onnx",
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        expected_sha256: Optional[Dict[str, str]] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: bool = True,
        hooks_concurrent: bool = True,
        hooks_timeout: Optional[float] = None,
        lang_temperatures: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Load a Laya agent backed by ONNX Runtime.

        Args:
            model_id_or_path: HuggingFace Hub ID or local path to the original PyTorch checkpoint
                              (used to load the tokenizer and config).
            onnx_path: Path to the exported .onnx file.
            subfolder: Optional subfolder if downloading from a repo bundle.
            revision: Optional Hub revision (commit SHA/branch/tag). When omitted,
                      huggingface_hub's normal default and existing offline cache are used.
            expected_sha256: Optional {path relative to the checkpoint dir: hexdigest}
                      verified before any checkpoint file is parsed; opt-in, and applies
                      to local directories too. A missing artifact raises
                      `FileNotFoundError` and a digest mismatch raises `ValueError`; either
                      error refuses the load.
            hooks (HookArg): Opt-in prediction hooks; see `laya.hooks`.
            on_predict_start (PredictHookArg): An opt-in start hook, run before inference.
            on_predict_end (PredictHookArg): An opt-in end hook, run after inference.
            hooks_raise: When False, a failing hook warns and inference continues.
            hooks_concurrent: When False, hooks are serialised with a lock.
            hooks_timeout: Bounds each hook call in seconds; None means no limit.
            lang_temperatures: Optional per-language temperature overrides, keyed by language
                               code, each `{"temperature": [3 floats], "temperature_by_options": {}}`.
                               Applied when a `lang=` is passed to `system_one`/`predict`, mirroring
                               the PyTorch `Agent`; a cross-backend swap otherwise loses calibration.
        """
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
        self.hooks_raise = bool(hooks_raise)
        self.hooks_concurrent = bool(hooks_concurrent)
        self.hooks_timeout = None if hooks_timeout is None else validate_timeout(hooks_timeout)
        self._hooks_lock = threading.RLock() if not hooks_concurrent else None
        self._hooks_mutex = threading.Lock()
        self.model_id = model_id_or_path

        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = model_id_or_path
        self.revision: Optional[str] = None
        if not os.path.exists(model_dir):
            if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
                raise FileNotFoundError(
                    f"Local model path not found: {model_id_or_path!r}."
                )
            from huggingface_hub import snapshot_download

            revision = resolve_revision(model_id_or_path, revision)
            prefix = f"{subfolder}/" if subfolder else ""
            kw = {
                "allow_patterns": [prefix + name for name in (
                    "rl_agent_config.json", "tokenizer/*", "encoder/*",
                )],
            }
            if revision:
                kw["revision"] = revision
            model_dir = snapshot_download(model_id_or_path, **kw)
            self.revision = snapshot_revision(model_dir) or revision

        if subfolder:
            model_dir = os.path.join(model_dir, subfolder)
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(
                    f"Subfolder {subfolder!r} not found in {model_id_or_path!r}."
                )

        # Verify integrity before any file in the checkpoint is parsed or executed.
        verify_digests(model_dir, expected_sha256, onnx_path=onnx_path)

        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Incompatible model: {model_id_or_path!r} does not contain 'rl_agent_config.json'."
            )

        with open(cfg_path) as f:
            self.cfg = json.load(f)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path!r}. Please run export_onnx.py first."
            )

        # Load Tokenizer.  Keep this compatibility fix in sync with Agent: checkpoints
        # produced by newer Transformers versions can contain TokenizersBackend or a list-valued
        # extra_special_tokens field that older loaders cannot parse.
        from .agent import _fix_tokenizer_config
        _fix_tokenizer_config(model_dir)
        tok_dir = os.path.join(model_dir, "tokenizer")
        self.tok = AutoTokenizer.from_pretrained(tok_dir if os.path.exists(tok_dir) else self.cfg.get("encoder"))

        # Initialize ONNX Runtime Session (auto-detect GPU if available)
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        # Enable ONNX Runtime's full graph optimization (operator fusion, constant folding).
        # It is functionally neutral and free at inference time; without it ORT runs the
        # unoptimized graph. Measured ~1.45x on CPU / ~1.75x on GPU vs eager torch with it on.
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        except Exception:
            # A provider can be listed by onnxruntime yet still fail to initialize
            # (missing CUDA libraries, unsupported driver, mismatched DLLs).  Keep
            # the CPU runtime usable instead of making model construction fail.
            if providers == ["CPUExecutionProvider"]:
                raise
            warnings.warn(
                "laya ONNX: CUDAExecutionProvider initialization failed; falling back to CPUExecutionProvider.",
                RuntimeWarning, stacklevel=2,
            )
            self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {k: clamp_temperature(v)
                                       for k, v in self.temperature_by_options_raw.items()}
        # Per-language temperature overrides, built exactly as the PyTorch Agent does so a caller
        # can hand the same `lang_temperatures` to either backend and read the same confidence.
        self.lang_temperatures = {}
        for l, lcfg in (lang_temperatures or {}).items():
            norm_l = l.split("-")[0].lower()
            t_raw = lcfg.get("temperature", self.temperature_raw)
            if len(t_raw) != 3:
                raise ValueError("Language override %r temperature must be a list of 3 floats" % l)
            tbo_raw = lcfg.get("temperature_by_options", {})
            self.lang_temperatures[norm_l] = {
                "temperature": [clamp_temperature(t) for t in t_raw],
                "temperature_by_options": {k: clamp_temperature(v) for k, v in tbo_raw.items()},
            }
        entries = [(k, v, self.temperature_by_options[k]) for k, v in self.temperature_by_options_raw.items()]
        entries += [("temperature[%d]" % i, t, self.temperature[i]) for i, t in enumerate(self.temperature_raw)]
        rejected = []
        for name, raw, applied in entries:
            try:
                if float(raw) == applied:
                    continue
            except (TypeError, ValueError):
                # Invalid entries already have a neutral fallback; diagnostics must not
                # repeat the failed conversion or prevent the checkpoint from loading.
                pass
            rejected.append("%s=%r -> %g" % (name, raw, applied))
        if rejected:
            warnings.warn(
                "laya ONNX: this checkpoint ships temperatures outside [%g, %g] which would distort "
                "confidence; clamping %s. Treat confidence from the affected buckets as uncalibrated."
                % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
                RuntimeWarning, stacklevel=2)

    @staticmethod
    def _to_internal(qdef: Dict) -> Dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        elif t == "noul" and isinstance(crit, dict):
            # Normalize boolean literal keys to string keys ("true"/"false")
            crit = {str(k).lower(): v for k, v in crit.items()}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            ins = json.dumps(ins, ensure_ascii=False)
        q = {"t": t, "ins": ins, "crit": crit}
        if "labels" in qdef:
            q["labels"] = qdef["labels"]
        return q

    def system_one(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]],
                   lang: Optional[str] = None,
                   hooks=None, on_predict_start=None, on_predict_end=None,
                   hooks_raise: Optional[bool] = None,
                   hooks_timeout: Optional[float] = None,
                   max_len: Optional[int] = None,
                   head_max_len: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate typed questions, running any opt-in hooks around the inference.

        `lang` selects a per-language temperature override (see `lang_temperatures`), matching the
        PyTorch `Agent.system_one` signature so either backend is a drop-in for the other.
        """
        active = compose_hooks(self.hooks, hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)
        ctx = PredictContext(states=[state], questions=questions, model=self.model_id, agent=self,
                             max_len=max_len, head_max_len=head_max_len)
        try:
            dispatch(active, "on_predict_start", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            if ctx.results is None:
                overrides = {}
                if ctx.max_len is not None:
                    overrides["max_len"] = ctx.max_len
                if ctx.head_max_len is not None:
                    overrides["head_max_len"] = ctx.head_max_len
                ctx.results = [self._infer(ctx.states[0], ctx.questions, lang=lang, **overrides)]
        except BaseException as exc:
            ctx.error = exc
            try:
                dispatch(active, "on_error", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                exc.__context__ = hook_exc
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000.0
            if ctx.results is not None:
                ctx.usage = aggregate_usage(ctx.results)
            try:
                dispatch(active, "on_predict_end", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                else:
                    raise
        return ctx.results[0]

    def _infer(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]],
               max_len: Optional[int] = None, head_max_len: Optional[int] = None,
               lang: Optional[str] = None) -> Dict[str, Any]:
        from .agent import Agent as _Agent

        ids = list(questions.keys())
        if not ids:
            return {
                "model": "laya-rl-agent-onnx",
                "answers": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        for qid in ids:
            _Agent._check_question(qid, questions[qid])
        items = []
        max_len = self.cfg.get("max_len", 512) if max_len is None else max_len
        head_max_len = self.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len

        # Tokenize the shared state once and reuse it across questions, instead of
        # re-serializing and re-tokenizing the same document inside build_sequence per
        # question (the PyTorch Agent already does this via `state_ids`).
        truncate_left = isinstance(state, list)
        state_ids = encode_text(
            self.tok,
            serialize_state(state).replace(self.tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]

        for qid in ids:
            q = self._to_internal(questions[qid])
            seq, markers, stats = build_sequence(
                self.tok, state, q, max_len, head_max_len,
                truncate_left=truncate_left, state_ids=state_ids, return_stats=True,
            )
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "options": stats})

        b = collate_items([items], self.tok.pad_token_id)
        
        # Prepare ONNX inputs as numpy arrays
        ort_inputs = {
            "input_ids": b["input_ids"].numpy().astype(np.int64),
            "attention_mask": b["attention_mask"].numpy().astype(np.int64),
            "marker_pos": b["marker_pos"].numpy().astype(np.int64),
            "marker_mask": b["marker_mask"].numpy().astype(bool),
            "qtype": b["qtype"].numpy().astype(np.int64),
        }

        # Run ONNX inference
        ort_outs = self.session.run(["logits", "act_logits"], ort_inputs)
        logits = ort_outs[0]
        act_logits = ort_outs[1]

        # Compute softmax for actions manually in numpy
        act_exp = np.exp(act_logits - np.max(act_logits, axis=-1, keepdims=True))
        act = act_exp / np.sum(act_exp, axis=-1, keepdims=True)

        answers = {}
        n_tokens = int(b["attention_mask"].sum())

        for r, qid in enumerate(ids):
            q = self._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            if lang and lang.split("-")[0].lower() in self.lang_temperatures:
                l_cfg = self.lang_temperatures[lang.split("-")[0].lower()]
                t_scale = l_cfg["temperature_by_options"].get(temp_bucket(qt, k), l_cfg["temperature"][qt])
            z = logits[r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()

            conf_score = round(confidence_from_probs(p, k), 4)
            # `answer_confidence` is the calibrated max(p) confidence, reported on every question
            # type so a caller can gate across types on one number -- matching the PyTorch Agent,
            # whose output ONNX callers otherwise cannot read (KeyError on cross-backend swap).
            ans_conf = round(answer_confidence(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf_score,
                    "answer_confidence": ans_conf,
                    "action": ext,
                }
            elif q["t"] == "score":
                exp_score = float((np.arange(k) * p).sum())
                answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf_score,
                    "answer_confidence": ans_conf,
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    "answer_confidence": ans_conf,
                    "action": ext,
                }

        usage = {"input_tokens": n_tokens, "output_tokens": 0}
        collapsed = collapsed_options(ids, items)
        if collapsed:
            usage["options"] = collapsed
        return {
            "model": "laya-rl-agent-onnx",
            "answers": answers,
            "usage": usage,
        }

    def decide(self, state: Union[str, dict, list], schema: Any = None, *,
               questions: Optional[Dict[str, Dict[str, Any]]] = None, return_details: bool = False,
               **predict_kwargs) -> Any:
        """Answer `state` against a schema (JSON schema or pydantic model) and return typed values.

        See `laya.structured`. Pass exactly one of `schema` or `questions`; extra keyword arguments
        are forwarded to `predict` / `system_one`.
        """
        from laya.structured import decide as _decide
        return _decide(self, state, schema, questions=questions,
                       return_details=return_details, **predict_kwargs)

    predict = system_one
