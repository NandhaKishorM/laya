import json
import os
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Union

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
from laya.confidence import check_min_confidence, flag_low_confidence


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
                   head_max_len: Optional[int] = None,
                   min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """Evaluate typed questions across state in one ONNX Runtime session run.

        `lang` selects a per-language temperature override (see `lang_temperatures`), matching the
        PyTorch `Agent.system_one` signature so either backend is a drop-in for the other.

        Defined in terms of `predict_batch`, exactly as the PyTorch `Agent.system_one` is, so the
        single-state and batched paths cannot drift apart.

        Args:
            state: Text string, JSON dict, or conversation turn list.
            questions: Question definitions, with the shapes `Agent.system_one` accepts.
            lang: Per-language temperature override (see `lang_temperatures`).
            hooks (HookArg): Per-call hooks, appended after any installed on the agent.
            on_predict_start (PredictHookArg): A per-call start hook. It may rewrite the
                    state/questions or call `ctx.skip(...)` to short-circuit inference.
            on_predict_end (PredictHookArg): A per-call end hook. It may rewrite the results.
            hooks_raise: Override the agent's `hooks_raise` for this call.
            hooks_timeout: Override the agent's `hooks_timeout` for this call.
            max_len: Override the config's `max_len` for this call.
            head_max_len: Override the config's `head_max_len` for this call.
            min_confidence: Opt-in abstention threshold on `answer_confidence` (#361); an answer
                    below it is returned flagged with `low_confidence: True`.

        Returns:
            Dictionary with answers, probabilities, calibrated confidence, and token usage.

        To score many states at once, see `predict_batch`, which shares session runs across them.
        """
        return self.predict_batch([state], questions, lang=lang, hooks=hooks,
                                  on_predict_start=on_predict_start,
                                  on_predict_end=on_predict_end, hooks_raise=hooks_raise,
                                  hooks_timeout=hooks_timeout,
                                  max_len=max_len, head_max_len=head_max_len,
                                  min_confidence=min_confidence)[0]

    def predict_batch(self, states: List[Union[str, dict, list]], questions: Dict[str, Dict[str, Any]],
                      batch_size: Optional[int] = None, lang: Optional[str] = None,
                      hooks=None, on_predict_start=None, on_predict_end=None,
                      hooks_raise: Optional[bool] = None,
                      hooks_timeout: Optional[float] = None,
                      max_len: Optional[int] = None,
                      head_max_len: Optional[int] = None,
                      min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """Evaluate the same questions over many states, sharing ONNX Runtime session runs.

        The throughput path, mirroring `laya.agent.Agent.predict_batch`: `system_one` collates one
        state's question rows per session run, so N states cost N runs. `predict_batch` collates
        several states' rows into one run -- or `ceil(len(states) / batch_size)` of them -- which is
        where ONNX Runtime's own parallelism pays off on CPU.

        Args:
            states: A list of states (each a text string, JSON dict, or conversation turn list).
                    The same `questions` are evaluated against every state.
            questions: Question definitions, exactly as accepted by `system_one`.
            batch_size: Optional cap on states per session run. `None` sends them all in one run;
                        set it to bound peak memory when batching many or long states.
            lang: Per-language temperature override applied to every state; see
                    `lang_temperatures`.
            hooks (HookArg): Per-call hooks, appended after any installed on the agent.
            on_predict_start (PredictHookArg): A per-call start hook. It may rewrite the
                    states/questions or call `ctx.skip(...)` to short-circuit inference.
            on_predict_end (PredictHookArg): A per-call end hook. It may rewrite the results.
            hooks_raise: Override the agent's `hooks_raise` for this call.
            hooks_timeout: Override the agent's `hooks_timeout` for this call.
            max_len: Override the config's `max_len` for this call.
            head_max_len: Override the config's `head_max_len` for this call.
            min_confidence: Opt-in abstention threshold on `answer_confidence` (#361); answers
                    below it are returned flagged with `low_confidence: True`.

        Returns:
            A list of per-state result dicts, each identical in shape to `system_one`'s output and
            aligned with `states` by index.
        """
        mc = check_min_confidence(min_confidence) if min_confidence is not None else None
        active = compose_hooks(self.hooks, hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)
        ctx = PredictContext(states=states, questions=questions, model=self.model_id, agent=self,
                             max_len=max_len, head_max_len=head_max_len)
        try:
            dispatch(active, "on_predict_start", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            states, questions = ctx.states, ctx.questions
            if ctx.results is None:
                # A start hook may have normalised a bare string/dict into a list; only the value
                # that survives the hook is validated, as Agent.predict_batch does.
                if isinstance(states, (str, bytes, dict)):
                    raise TypeError(
                        "predict_batch expects a list of states; pass a single state to predict()/system_one()."
                    )
                states = list(states)
                if not states:
                    ctx.results = []
                else:
                    overrides: Dict[str, int] = {}
                    if ctx.max_len is not None:
                        overrides["max_len"] = ctx.max_len
                    if ctx.head_max_len is not None:
                        overrides["head_max_len"] = ctx.head_max_len
                    ctx.results = self._infer_batch(states, questions, lang=lang,
                                                    batch_size=batch_size, **overrides)
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
                if mc is not None:
                    flag_low_confidence(ctx.results, mc)
            try:
                dispatch(active, "on_predict_end", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                else:
                    raise
        return ctx.results

    def predict_long(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]],
                     window: Optional[int] = None, stride: Optional[int] = None,
                     aggregate: str = "auto", batch_size: Optional[int] = None,
                     lang: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate questions over a state longer than the context window, scanning it in
        overlapping windows and aggregating per question.

        The ONNX port of `laya.agent.Agent.predict_long`, with the same aggregation rules:
        `system_one` truncates a state that exceeds `max_len` to a single window, silently
        dropping the rest. `predict_long` tokenizes the state once, splits it into overlapping
        token windows, scores every window through `predict_batch` -- so the windows share ONNX
        Runtime session runs rather than costing one each -- and combines the per-window answers:

          * noul  -> P(true) is the max over windows (the statement holds if any window supports it)
          * choice-> the answer from the single most-confident window, so a localized signal isn't
                     out-voted by the many neutral windows a long document is mostly made of
          * score -> the level from the most-confident window, likewise

        The returned probability/confidence is the deciding window's, **not a calibrated number for
        the whole document**, for the same reasons the PyTorch docstring gives. Each answer carries
        `answer["window"]` -- the deciding window's `index`, `token_start`/`token_end` into the
        tokenized state, and the window `count`.

        A state that already fits one window is passed straight to `system_one` (identical output).

        Args:
            state: Text string, JSON dict, or conversation turn list.
            questions: Question definitions, exactly as accepted by `system_one`.
            window: State tokens per window. Defaults to the per-question state budget
                    (`max_len - head_max_len - 8`). Smaller windows isolate a localized signal
                    better at the cost of more windows, as in `Agent.predict_long`.
            stride: Token step between windows. Defaults to `window // 2` (50% overlap).
            aggregate: "auto" (the per-type rules above) is the only mode for now.
            batch_size: Cap on windows per session run, to bound peak memory on very long states.
            lang: Per-language temperature selection, as in `system_one`.

        Returns a single result dict, the same shape as `system_one`, with `usage["windows"]`
        added.
        """
        if aggregate != "auto":
            raise ValueError("predict_long: only aggregate='auto' is supported")
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        budget = window if (window and window > 0) else max(64, max_len - head_max_len - 8)

        state_ids = encode_text(
            self.tok,
            serialize_state(state).replace(self.tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        # Fits in one window: identical to a plain call, no windowing overhead.
        if len(state_ids) <= budget:
            return self.system_one(state, questions, lang=lang)

        step = stride if (stride and stride > 0) else max(1, budget // 2)
        windows, starts = [], []
        i, n = 0, len(state_ids)
        while i < n:
            # Decode each token window back to text so predict_batch re-tokenizes it as a normal
            # state; the 50% default overlap absorbs any boundary drift on re-tokenization.
            windows.append(self.tok.decode(state_ids[i:i + budget]))
            starts.append(i)
            if i + budget >= n:
                break
            i += step

        results = self.predict_batch(windows, questions, batch_size=batch_size, lang=lang)

        ids = list(questions.keys())
        internal = {qid: self._to_internal(questions[qid]) for qid in ids}
        answers = {}
        for qid in ids:
            per = [r["answers"][qid] for r in results]
            if internal[qid]["t"] == "noul":
                # Evidence anywhere: the strongest window decides, carrying its own P(true),
                # confidence and act so the fields stay mutually consistent.
                best = max(range(len(per)), key=lambda j: float(per[j]["noul"]))
            else:
                # choice / score: the most-confident window wins, preserving a localized signal
                # that averaging over a mostly-neutral document would drown.
                best = max(range(len(per)), key=lambda j: float(per[j]["answer_confidence"]))
            ans = per[best]
            # Name the window that decided; the probability is that window's, not the document's.
            ans["window"] = {"index": best, "token_start": starts[best],
                             "token_end": min(starts[best] + budget, len(state_ids)),
                             "count": len(windows)}
            answers[qid] = ans
        # Aggregate usage generically so fields predict_batch may grow later are propagated
        # rather than silently dropped, then record the window count.
        usage: Dict[str, Any] = {}
        for r in results:
            for key, val in r["usage"].items():
                usage[key] = (usage.get(key, 0) + val) if isinstance(val, (int, float)) else val
        usage["output_tokens"] = 0
        usage["windows"] = len(windows)
        return {"model": "laya-rl-agent-onnx", "answers": answers, "usage": usage}

    def _infer(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]],
               max_len: Optional[int] = None, head_max_len: Optional[int] = None,
               lang: Optional[str] = None) -> Dict[str, Any]:
        """Answer one state without hooks: one session run, the `predict_batch([state])` case."""
        return self._infer_batch([state], questions, max_len=max_len, head_max_len=head_max_len,
                                 lang=lang)[0]

    def _encode_state(self, state: Union[str, dict, list], ids: List[str], internal: Dict[str, Any],
                      max_len: int, head_max_len: int) -> List[Dict[str, Any]]:
        """This state's question rows, tokenizing the state once for all of them."""
        truncate_left = isinstance(state, list)
        # Tokenize the shared state once and reuse it across questions, instead of
        # re-serializing and re-tokenizing the same document inside build_sequence per
        # question (the PyTorch Agent already does this via `state_ids`).
        state_ids = encode_text(
            self.tok,
            serialize_state(state).replace(self.tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]

        items = []
        for qid in ids:
            q = internal[qid]
            seq, markers = build_sequence(
                self.tok, state, q, max_len, head_max_len,
                truncate_left=truncate_left, state_ids=state_ids,
            )
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        return items

    def _decode_answers(self, logits, act, items: List[Dict[str, Any]], ids: List[str],
                        internal: Dict[str, Any], offset: int,
                        lang: Optional[str] = None) -> Dict[str, Any]:
        """Decode the `len(ids)` head rows starting at `offset` into this state's answers."""
        answers = {}
        for r, qid in enumerate(ids):
            q = internal[qid]
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            if lang and lang.split("-")[0].lower() in self.lang_temperatures:
                l_cfg = self.lang_temperatures[lang.split("-")[0].lower()]
                t_scale = l_cfg["temperature_by_options"].get(temp_bucket(qt, k), l_cfg["temperature"][qt])
            z = logits[offset + r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()

            conf_score = round(confidence_from_probs(p, k), 4)
            # `answer_confidence` is the calibrated max(p) confidence, reported on every question
            # type so a caller can gate across types on one number -- matching the PyTorch Agent,
            # whose output ONNX callers otherwise cannot read (KeyError on cross-backend swap).
            ans_conf = round(answer_confidence(p, k), 4)
            ext = {"act_probability": round(float(act[offset + r, 0]), 4)}

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
        return answers

    def _infer_batch(self, states: List[Union[str, dict, list]], questions: Dict[str, Dict[str, Any]],
                     max_len: Optional[int] = None, head_max_len: Optional[int] = None,
                     lang: Optional[str] = None,
                     batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Validate once, then encode, collate, run and decode in chunks of `batch_size` states."""
        from .agent import Agent as _Agent

        ids = list(questions.keys())
        if not ids:
            return [{"model": "laya-rl-agent-onnx", "answers": {},
                     "usage": {"input_tokens": 0, "output_tokens": 0}} for _ in states]
        for qid in ids:
            _Agent._check_question(qid, questions[qid])
        internal = {qid: self._to_internal(questions[qid]) for qid in ids}
        max_len = self.cfg.get("max_len", 512) if max_len is None else max_len
        head_max_len = self.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len
        chunk = batch_size if (batch_size and batch_size > 0) else len(states)

        results: List[Dict[str, Any]] = []
        for start in range(0, len(states), chunk):
            part = states[start:start + chunk]
            encoded = [self._encode_state(st, ids, internal, max_len, head_max_len) for st in part]

            b = collate_items(encoded, self.tok.pad_token_id)

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
            att = b["attention_mask"]

            row = 0
            for index, items in enumerate(encoded):
                nrows = len(items)
                n_tokens = int(att[row:row + nrows].sum())
                results.append({
                    "model": "laya-rl-agent-onnx",
                    "answers": self._decode_answers(logits, act, items, ids, internal, row, lang=lang),
                    "usage": {"input_tokens": n_tokens, "output_tokens": 0},
                })
                row += nrows
        return results

    def decide(self, state: Union[str, dict, list], schema: Any = None, *,
               questions: Optional[Dict[str, Dict[str, Any]]] = None, return_details: bool = False,
               min_confidence: Optional[float] = None,
               **predict_kwargs) -> Any:
        """Answer `state` against a schema (JSON schema or pydantic model) and return typed values.

        See `laya.structured`. Pass exactly one of `schema` or `questions`; extra keyword arguments
        are forwarded to `predict` / `system_one`.
        """
        from laya.structured import decide as _decide
        return _decide(self, state, schema, questions=questions,
                       return_details=return_details, min_confidence=min_confidence, **predict_kwargs)

    predict = system_one
