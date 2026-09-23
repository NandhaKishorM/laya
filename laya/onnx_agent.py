import json
import os
import threading
import time
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np

from laya.hooks import HookRegistry, PredictContext, aggregate_usage, dispatch, normalise_hooks
from laya.common import (
    QTYPES,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
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
    _hooks_lock = None
    model_id = None

    def __init__(
        self,
        model_id_or_path: str,
        onnx_path: str = "laya.onnx",
        subfolder: Optional[str] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: bool = True,
        hooks_concurrent: bool = True,
    ):
        """Load a Laya agent backed by ONNX Runtime.

        Args:
            model_id_or_path: HuggingFace Hub ID or local path to the original PyTorch checkpoint
                              (used to load the tokenizer and config).
            onnx_path: Path to the exported .onnx file.
            subfolder: Optional subfolder if downloading from a repo bundle.
            hooks, on_predict_start, on_predict_end: Opt-in prediction hooks; see `laya.hooks`.
            hooks_raise: When False, a failing hook warns and inference continues.
            hooks_concurrent: When False, hooks are serialised with a lock.
        """
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
        self.hooks_raise = bool(hooks_raise)
        self.hooks_concurrent = bool(hooks_concurrent)
        self._hooks_lock = threading.RLock() if not hooks_concurrent else None
        self._hooks_mutex = threading.Lock()
        self.model_id = model_id_or_path

        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = model_id_or_path
        if not os.path.exists(model_dir):
            if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
                raise FileNotFoundError(
                    f"Local model path not found: {model_id_or_path!r}."
                )
            from huggingface_hub import snapshot_download

            prefix = f"{subfolder}/" if subfolder else ""
            kw = {
                "allow_patterns": [prefix + name for name in (
                    "rl_agent_config.json", "tokenizer/*", "encoder/*",
                )],
            }
            model_dir = snapshot_download(model_id_or_path, **kw)

        if subfolder:
            model_dir = os.path.join(model_dir, subfolder)
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(
                    f"Subfolder {subfolder!r} not found in {model_id_or_path!r}."
                )

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

        # Load Tokenizer
        tok_dir = os.path.join(model_dir, "tokenizer")
        self.tok = AutoTokenizer.from_pretrained(tok_dir if os.path.exists(tok_dir) else self.cfg.get("encoder"))

        # Initialize ONNX Runtime Session (auto-detect GPU if available)
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {k: clamp_temperature(v)
                                       for k, v in self.temperature_by_options_raw.items()}
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
                   hooks=None, on_predict_start=None, on_predict_end=None,
                   hooks_raise: Optional[bool] = None,
                   max_len: Optional[int] = None,
                   head_max_len: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate typed questions, running any opt-in hooks around the inference."""
        active = list(self.hooks) + normalise_hooks(hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        ctx = PredictContext(states=[state], questions=questions, model=self.model_id, agent=self,
                             max_len=max_len, head_max_len=head_max_len)
        try:
            dispatch(active, "on_predict_start", ctx, raise_errors=raise_errors, lock=self._hooks_lock)
            if ctx.results is None:
                overrides = {}
                if ctx.max_len is not None:
                    overrides["max_len"] = ctx.max_len
                if ctx.head_max_len is not None:
                    overrides["head_max_len"] = ctx.head_max_len
                ctx.results = [self._infer(ctx.states[0], ctx.questions, **overrides)]
        except BaseException as exc:
            ctx.error = exc
            try:
                dispatch(active, "on_error", ctx, raise_errors=raise_errors, lock=self._hooks_lock)
            except BaseException as hook_exc:
                exc.__context__ = hook_exc
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000.0
            if ctx.results is not None:
                ctx.usage = aggregate_usage(ctx.results)
            try:
                dispatch(active, "on_predict_end", ctx, raise_errors=raise_errors, lock=self._hooks_lock)
            except BaseException as hook_exc:
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                else:
                    raise
        return ctx.results[0]

    def _infer(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]],
               max_len: Optional[int] = None, head_max_len: Optional[int] = None) -> Dict[str, Any]:
        from .agent import Agent as _Agent

        ids = list(questions.keys())
        for qid in ids:
            _Agent._check_question(qid, questions[qid])
        items = []
        max_len = self.cfg.get("max_len", 512) if max_len is None else max_len
        head_max_len = self.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len

        for qid in ids:
            q = self._to_internal(questions[qid])
            seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

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
            z = logits[r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()

            conf_score = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf_score,
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
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    "action": ext,
                }

        return {
            "model": "laya-rl-agent-onnx",
            "answers": answers,
            "usage": {"input_tokens": n_tokens, "output_tokens": 0},
        }

    predict = system_one
