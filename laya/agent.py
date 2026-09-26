"""High-level inference runtime for laya System 1 decision models."""
import json
import os
import tempfile
import threading
import time
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    amp_dtype,
    build_model,
    build_sequence,
    collapsed_options,
    clamp_temperature,
    collate_items,
    answer_confidence,
    confidence_from_probs,
    _resolve_noul_labels,
    encode_text,
    render_options,
    serialize_state,
    temp_bucket,
)
from .hooks import (
    HookRegistry, PredictContext, aggregate_usage, compose_hooks, dispatch, normalise_hooks,
    validate_timeout,
)
from .revisions import resolve_revision, snapshot_revision, verify_digests


def _fix_tokenizer_config(path: str):
    """Ensure tokenizer_config.json can be loaded across all transformers versions."""
    cfg_file = os.path.join(path, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return
    try:
        with open(cfg_file) as f:
            tcfg = json.load(f)
        changed = False
        if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            changed = True
        # Checkpoints built on the mmBERT/Gemma tokenizer store extra_special_tokens as a list;
        # transformers expects a mapping and raises "'list' object has no attribute 'keys'",
        # which makes AutoTokenizer -- and so the whole model -- fail to load.
        extra = tcfg.get("extra_special_tokens")
        if isinstance(extra, list):
            tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
            changed = True
        if changed:
            # HuggingFace snapshots are symlinks into a shared blob store, so writing through the
            # link would truncate a file shared with other revisions and processes, race
            # concurrent loads, and desync the hub's cache metadata. Write to a temporary file in
            # the same directory and `os.replace` it into place: the snapshot entry becomes a
            # regular file and is swapped atomically, so a concurrent load never sees a missing
            # or half-written config.
            cfg_dir = os.path.dirname(cfg_file)
            fd, tmp_file = tempfile.mkstemp(dir=cfg_dir, prefix=".tokenizer_config.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(tcfg, f, indent=2)
                # mkstemp creates the file 0600; keep the mode the cache file had so a shared
                # cache stays readable to the same users as before.
                try:
                    os.chmod(tmp_file, os.stat(cfg_file).st_mode & 0o777)
                except OSError:
                    pass
                os.replace(tmp_file, cfg_file)
            except BaseException:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass
                raise
    except Exception as e:
        # Do not swallow this silently: if the patch did not apply, AutoTokenizer may fail later
        # with a confusing error and no hint that the config was the cause.
        warnings.warn(
            "laya: could not patch %s (%s); the tokenizer may fail to load with this "
            "transformers version." % (cfg_file, e),
            RuntimeWarning, stacklevel=2)


def _verify_compatibility(model: torch.nn.Module, cfg: Dict, weights: Dict[str, torch.Tensor], model_id: str):
    """Verify that the loaded checkpoint weights and config strictly match the expected architecture."""
    # 1. Verify required configuration attributes
    required_cfg = ["encoder", "head_layers"]
    missing_cfg = [k for k in required_cfg if k not in cfg]
    if missing_cfg:
        raise ValueError(
            f"Incompatible model config for {model_id!r}: missing configuration keys {missing_cfg}. "
            f"Ensure this is a valid RL Agent decision model."
        )

    # 2. Check for required component prefixes
    required_prefixes = ("encoder.", "type_emb.", "scorer.", "act_head.")
    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in weights.keys()):
            raise ValueError(
                f"Incompatible model weights for {model_id!r}: checkpoint is missing '{prefix}' parameters. "
                f"Expected an RL Agent decision model with encoder and decision heads."
            )

    # 3. Check for parameter shape mismatches
    model_sd = model.state_dict()
    shape_mismatches = []
    missing_keys = []

    for name, param in model.named_parameters():
        if name not in weights:
            missing_keys.append(name)
        elif tuple(weights[name].shape) != tuple(param.shape):
            shape_mismatches.append(f"  - {name}: expected {tuple(param.shape)}, found {tuple(weights[name].shape)}")

    if shape_mismatches:
        err_details = "\n".join(shape_mismatches[:5])
        if len(shape_mismatches) > 5:
            err_details += f"\n  ... and {len(shape_mismatches) - 5} more mismatched layers."
        raise ValueError(
            f"Model architecture mismatch for {model_id!r}:\n{err_details}\n"
            f"The checkpoint weights do not match the configured model architecture."
        )

    if missing_keys:
        raise ValueError(
            f"Model weights incomplete for {model_id!r}: missing {len(missing_keys)} parameter tensors "
            f"(e.g. {missing_keys[:3]})."
        )


_TOKENIZERS: Dict[tuple, Any] = {}
_TOKENIZERS_LOCK = threading.Lock()

# The per-inference CPU fallback rewrites shared runtime state (device, dtype, amp) and moves the
# model while other threads may be running their own forward, so the demotion and the restore are
# serialised. A second request that hits OOM waits here and re-demotes only if it needs to.
_OOM_FALLBACK_LOCK = threading.Lock()


def _load_tokenizer(tok_dir: str, cfg: Dict) -> Any:
    """Tokenizer for a checkpoint, parsed once per process.

    Parsing `tokenizer.json` is not free and `huggingface_hub` caches only the download, not
    the parsed object: the multilingual checkpoint ships a 34 MB, 256k-vocabulary file that
    costs seconds to load, several times the cost of applying its weights. A tokenizer is
    read-only during inference, so one instance is shared by every Agent that wants the same
    directory -- including an Agent the Router has rebuilt after eviction.

    Keyed on the tokenizer directory and its mtime, so a re-download or a config rewritten by
    `_fix_tokenizer_config()` still produces a fresh parse. Non-directory sources (a hub id)
    are not cached, so a caller cannot pin a stale remote revision.
    """
    from transformers import AutoTokenizer

    if not os.path.isdir(tok_dir):
        return AutoTokenizer.from_pretrained(cfg.get("encoder"))

    try:
        stamp = (os.path.abspath(tok_dir), os.path.getmtime(os.path.join(tok_dir, "tokenizer_config.json")))
    except OSError:
        return AutoTokenizer.from_pretrained(tok_dir)

    with _TOKENIZERS_LOCK:
        cached = _TOKENIZERS.get(stamp)
        if cached is not None:
            return cached
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        _TOKENIZERS[stamp] = tokenizer
        return tokenizer


def _amp_context(device, dtype, enabled: bool):
    """Autocast context for the forward pass, or a no-op when mixed precision is not in use.

    Entering `torch.autocast` on a device torch has no autocast backend for raises even with
    `enabled=False` ("User specified an unsupported autocast device_type mps"), which broke
    every `predict()` on Apple Silicon on some torch builds. Only enter it when we use it;
    the `enabled` flag is set per device by the load-time policy, which skips XPU on builds
    without an XPU autocast backend.
    """
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


MPS_AMP_MIN_ROWS_DEFAULT = 5


def _mps_amp_min_rows() -> int:
    """Rows at which MPS fp16 autocast starts to pay off. Override with LAYA_MPS_AMP_MIN_ROWS."""
    raw = os.environ.get("LAYA_MPS_AMP_MIN_ROWS", "")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return MPS_AMP_MIN_ROWS_DEFAULT


def _cuda_amp_dtype(checkpoint_default: Optional[str]) -> torch.dtype:
    """Autocast dtype on CUDA at compute capability >= 8: the checkpoint's `amp_dtype` (bf16 for the
    shipped checkpoints), or LAYA_CUDA_AMP=fp16|bf16 when set. fp16 stays 2-10x closer to the fp32
    forward than bf16 on every shipped checkpoint at the same speed; anything else is ignored."""
    raw = os.environ.get("LAYA_CUDA_AMP", "").lower()
    if raw in ("fp16", "float16"):
        return torch.float16
    if raw in ("bf16", "bfloat16"):
        return torch.bfloat16
    return amp_dtype(checkpoint_default)


class Agent(HookRegistry):
    """System 1 decision model runtime: fast, non-autoregressive, calibrated decisions."""

    # Hooks are opt-in. `hooks`/`_hooks_mutex` defaults come from HookRegistry; the rest keep a
    # hand-built instance (`Agent.__new__` in tests) working and make an unset hook a no-op.
    hooks_raise = True
    hooks_concurrent = True
    hooks_timeout = None
    _hooks_lock = None
    model_id = None
    # Autocast is chosen per device in __init__; this default covers instances built
    # without it (for example a hand-constructed runtime in tests).
    amp_enabled = False
    mps_amp_min_rows = MPS_AMP_MIN_ROWS_DEFAULT
    # The stock forward is also used by lightweight runtimes built with __new__ in tests.
    _fast = None

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
        fast: bool = False,
        compile: bool = False,
        revision: Optional[str] = None,
        expected_sha256: Optional[Dict[str, str]] = None,
        lang_temperatures: Optional[Dict[str, Dict[str, Any]]] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: bool = True,
        hooks_concurrent: bool = True,
        hooks_timeout: Optional[float] = None,
    ):
        """Load a Laya checkpoint.

        `revision` optionally pins the Hub download to an explicit commit SHA/branch/tag;
        when omitted, huggingface_hub's normal default and existing offline cache are used.
        `expected_sha256` ({path relative to the checkpoint dir: hexdigest})
        verifies artifact integrity before any weight is parsed or executed; it is opt-in
        and applies to local directories too. A missing artifact raises `FileNotFoundError`
        and a digest mismatch raises `ValueError`; either error refuses the load.

        `fast=True` swaps the encoder/head forward for the TileLang fast path (CUDA only, needs
        `pip install laya[fast]`); see `Agent.accelerate`.

        `subfolder` selects one checkpoint from a repo that bundles several, e.g.
        `Agent("convaiinnovations/laya", subfolder="multilingual")`. Only that subfolder is
        downloaded, so bundling does not cost every user the whole family.

        `hooks` / `on_predict_start` / `on_predict_end` observe or shape every prediction; see
        `laya.hooks`. `hooks_raise=False` warns and continues when a hook fails,
        `hooks_concurrent=False` serialises hooks that are not safe to run in parallel, and
        `hooks_timeout` bounds each hook call in seconds (None means no limit).
        """
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
        self.hooks_raise = bool(hooks_raise)
        self.hooks_concurrent = bool(hooks_concurrent)
        self.hooks_timeout = None if hooks_timeout is None else validate_timeout(hooks_timeout)
        self._hooks_lock = threading.RLock() if not hooks_concurrent else None
        self._hooks_mutex = threading.Lock()
        self.model_id = model_id_or_path

        from safetensors.torch import load_file

        model_dir = model_id_or_path
        self.revision: Optional[str] = None
        if not os.path.exists(model_dir):
            if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
                raise FileNotFoundError(
                    f"Local model path not found: {model_id_or_path!r}. "
                    f"Check that the directory exists and that training saved the model successfully."
                )
            from huggingface_hub import snapshot_download

            # Restrict root checkpoints too: the default repo also contains sibling
            # checkpoints, which an unfiltered snapshot would unnecessarily download.
            revision = resolve_revision(model_id_or_path, revision)
            prefix = f"{subfolder}/" if subfolder else ""
            kw = {
                "token": token or os.environ.get("HF_TOKEN") or None,
                "allow_patterns": [prefix + name for name in (
                    "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*",
                )],
            }
            if revision:
                kw["revision"] = revision
            model_dir = snapshot_download(model_id_or_path, **kw)
            # The cache layout records which commit the snapshot points at.
            self.revision = snapshot_revision(model_dir) or revision

        if subfolder:
            model_dir = os.path.join(model_dir, subfolder)
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(
                    f"Subfolder {subfolder!r} not found in {model_id_or_path!r}."
                )

        # Verify integrity before any file in the checkpoint is parsed or executed.
        verify_digests(model_dir, expected_sha256)

        _fix_tokenizer_config(model_dir)

        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Incompatible model: {model_id_or_path!r} does not contain 'rl_agent_config.json'. "
                f"That file ships with the weights of a Laya checkpoint, so load one of those "
                f"(e.g. 'convaiinnovations/laya') or a directory your own training run wrote."
            )

        with open(cfg_path) as f:
            self.cfg = json.load(f)

        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Incompatible model: 'model.safetensors' not found in {model_id_or_path!r}."
            )

        # 1. Device resolution with automatic fallback
        if device is not None:
            target_device = torch.device(device)
            if target_device.type == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            elif target_device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                print("Warning: MPS requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            elif target_device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                print("Warning: XPU requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = target_device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = torch.device("xpu")
            else:
                self.device = torch.device("cpu")

        tok_dir = os.path.join(model_dir, "tokenizer")
        self.tok = _load_tokenizer(tok_dir, self.cfg)

        enc_dir = os.path.join(model_dir, "encoder")
        # build_model skips initialisation of every parameter; the checkpoint supplies them all.
        self.model = build_model(self.cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None,
                                 pretrained=False)

        # Load weights and verify architectural compatibility
        weights = load_file(weights_path)
        _verify_compatibility(self.model, self.cfg, weights, model_id_or_path)

        self.model.load_state_dict(weights, strict=True)

        # ModernBERT's reference_compile defaults to "auto" and will torch.compile the encoder.
        # That is a loss for the batch sizes Laya runs (a handful of questions per call) and can
        # hang on some platforms, so keep the eager path unless explicitly requested.
        try:
            self.model.encoder.config.reference_compile = compile
        except Exception:
            pass
            
        # The TileLang fast path replaces the forward itself, so it takes precedence over compile.
        if compile and not fast:
            self.model = torch.compile(self.model)

        # Keep what the checkpoint shipped for inspection, but only ever apply clamped values:
        # some buckets are fitted to sharpen rather than soften (see clamp_temperature).
        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {k: clamp_temperature(v)
                                       for k, v in self.temperature_by_options_raw.items()}

        self.lang_temperatures = {}
        for l, cfg in (lang_temperatures or {}).items():
            norm_l = l.split("-")[0].lower()
            t_raw = cfg.get("temperature", self.temperature_raw)
            if len(t_raw) != 3:
                raise ValueError("Language override %r temperature must be a list of 3 floats" % l)
            tbo_raw = cfg.get("temperature_by_options", {})
            self.lang_temperatures[norm_l] = {
                "temperature": [clamp_temperature(t) for t in t_raw],
                "temperature_by_options": {k: clamp_temperature(v) for k, v in tbo_raw.items()}
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
                "laya: this checkpoint ships invalid temperatures or values outside [%g, %g]; "
                "using %s. Treat confidence from the affected entries as uncalibrated."
                % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
                RuntimeWarning, stacklevel=2)
        # Autocast policy. CUDA, MPS and XPU all support fp16/bf16 autocast and the shipped
        # checkpoints are trained in reduced precision; on CUDA the checkpoint's `amp_dtype`
        # (bf16) is the default and LAYA_CUDA_AMP=fp16|bf16 overrides it. CPU bf16 is only a win
        # on hardware with native BF16, so it stays opt-in via LAYA_CPU_AMP=bf16. MPS fp16 is slower than fp32 on
        # a single small row (autocast overhead dominates) and only wins once the batch has
        # several rows, so it is gated per call by `mps_amp_min_rows` (default 5, override with
        # LAYA_MPS_AMP_MIN_ROWS) rather than enabled unconditionally. XPU autocast supports
        # bf16/fp16 only, and entering it on a torch build without an XPU autocast backend
        # raises on every predict (the failure #273 fixed for MPS), so it is only enabled on
        # builds that have one.
        self.dtype = torch.float32
        self.amp_enabled = False
        self.mps_amp_min_rows = _mps_amp_min_rows()
        if self.device.type == "cuda":
            self.amp_enabled = True
            if torch.cuda.get_device_capability(self.device)[0] < 8:
                self.dtype = torch.float16
            else:
                self.dtype = _cuda_amp_dtype(self.cfg.get("amp_dtype", "fp16"))
        elif self.device.type == "mps":
            self.amp_enabled = True
            self.dtype = torch.float16
        elif self.device.type == "xpu":
            # Device selection above already requires torch.xpu for an xpu device; the probe
            # also guards hand-built agents on builds without an XPU autocast backend.
            if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
                self.amp_enabled = True
                self.dtype = torch.bfloat16
        elif self.device.type == "cpu":
            if os.environ.get("LAYA_CPU_AMP", "").lower() in ("bf16", "bfloat16"):
                self.amp_enabled = True
                self.dtype = torch.bfloat16

        self._fast = None

        # 2. Place on device with graceful fallback to CPU on memory error
        fell_back_from = fell_back_why = None
        try:
            self.model.to(self.device).eval()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if self.device.type != "cpu":
                # Record what actually went wrong: the reason matters more than the symptom,
                # and it is the only place the underlying exception is ever surfaced.
                fell_back_from, fell_back_why = self.device, e
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.amp_enabled = False
                self.model.to(self.device).eval()
            else:
                raise e

        if fast:
            self.accelerate()

        if fell_back_from is not None:
            print(
                "\n[laya] Warning: could not place the model on %s, so it is running on CPU.\n"
                "  Reason: %s\n"
                "  Inference will be roughly 10-15x slower (~200-500 ms rather than ~35 ms).\n"
                "  If this is a newer NVIDIA GPU (Blackwell / RTX 50-series), your PyTorch build\n"
                "  may not support its CUDA architecture:\n"
                "    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128\n"
                "  See https://pytorch.org/get-started/locally/\n"
                % (fell_back_from, fell_back_why), flush=True)

    def accelerate(self, use_graphs: bool = True, strict: bool = False):
        """Replace the model forward with the TileLang fast path (fused GEMM/GEGLU/LayerNorm/RoPE kernels,
        sliding-window flash attention, 16-bit resident weights, CUDA graphs per shape bucket).

        The fast path runs in the agent's autocast dtype at the time of the call (bf16 or fp16), so it
        matches the stock forward it replaces within rounding (see benchmarks/parity_fast.py). After
        changing `agent.dtype`, call `deaccelerate()` then `accelerate()` to rebuild it. Returns True if
        enabled. With `strict=False` any failure (no CUDA, tilelang missing) leaves the stock path in place.
        """
        if self._fast is not None:
            return True
        if self.device.type != "cuda":
            if strict:
                raise RuntimeError("laya fast path needs a CUDA device")
            return False
        last = None
        for _attempt in range(2):  # tilelang's JIT cache has been seen to fail once, then succeed
            try:
                from .fast import FastLaya
                fast_dtype = self.dtype if self.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
                self._fast = FastLaya(self.model, max_len=self.cfg.get("max_len", 512), use_graphs=use_graphs,
                                      dtype=fast_dtype)
                break
            except Exception as e:  # tilelang missing / unsupported arch
                last = e
        if self._fast is None:
            if strict:
                raise last
            print("Warning: laya fast path unavailable (%s); using the stock forward." % last)
            return False
        self._stock_forward = self.model.forward
        self.model.forward = self._fast.forward
        return True

    def deaccelerate(self):
        """Restore the stock forward."""
        if self._fast is not None:
            self.model.forward = self._stock_forward
            self._fast = None

    def _restore_runtime(self, device, dtype, amp_enabled: bool, had_fast: bool) -> None:
        """Undo the scoped CPU fallback of `_infer`, best effort.

        A model that no longer fits `device` after the CPU retry stays demoted: raising out of a
        request that already succeeded would trade a silent slowdown for a crash, the worse
        failure of the two.
        """
        try:
            self.model.to(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # The move can fail partway through, leaving parameters split between devices, so
            # finish the demotion before giving up: every later call must find one device,
            # not a mix of both.
            self.model.to(torch.device("cpu"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Warning: could not move the model back to %s after the CPU retry (%s); "
                  "staying on CPU." % (device, e))
            return
        self.device, self.dtype, self.amp_enabled = device, dtype, amp_enabled
        if had_fast:
            self.accelerate()

    @staticmethod
    def _check_question(qid: str, qdef: Any) -> None:
        """Reject a question that cannot be answered, naming it and what to fix.

        `render_options` reads `criteria` in the shape the question's type expects and the decision
        head needs at least one option, so a malformed definition used to surface from three frames
        down as something that names neither the question nor the problem: `AttributeError:
        'NoneType' object has no attribute 'items'`, `KeyError: 'bool'`, or a `selected index k out
        of range` raised inside the model for a question that ended up with no options at all.
        """
        if not isinstance(qdef, dict):
            raise ValueError("question %r: definition must be a dict, got %s"
                             % (qid, type(qdef).__name__))
        t = qdef.get("type")
        if t not in QTYPES:
            raise ValueError("question %r: unknown type %r; use one of %s" % (qid, t, sorted(QTYPES)))
        if "instructions" not in qdef:
            raise ValueError("question %r: no 'instructions'; add the text the model should answer" % (qid,))
        crit = qdef.get("criteria")
        if t == "choice":
            if not isinstance(crit, (dict, list)):
                raise ValueError("question %r: a choice question takes 'criteria' as a dict of "
                                 "label -> description, or a list of labels" % (qid,))
            if not crit:
                raise ValueError("question %r: a choice question needs at least one criterion" % (qid,))
            # A label is used as a dict key when a list of labels is normalised, so a list, dict
            # or set label raised `TypeError: unhashable type: 'list'` from three frames down --
            # which names neither the question nor the label, and which `serve` cannot classify
            # as a caller error, so over HTTP it became a 500 "inference failed" instead of a 422.
            # Labels are rendered as option text, so a nested structure has no meaning here.
            for i, label in enumerate(crit if isinstance(crit, list) else crit.keys()):
                if isinstance(label, (list, dict, set, bytearray)):
                    raise ValueError(
                        "question %r: choice label %d is a %s; a label is rendered as option text "
                        "and used as the answer key, so it must be a scalar (a string, number or "
                        "None), got %r" % (qid, i, type(label).__name__, label))
        elif t == "score":
            if not isinstance(crit, list):
                raise ValueError("question %r: a score question takes 'criteria' as a list of level "
                                 "descriptions, index 0 first" % (qid,))
            if not crit:
                raise ValueError("question %r: a score question needs at least one level" % (qid,))
            if None in crit:
                raise ValueError("question %r: score level %d is null; give every level a description, "
                                 "index 0 first" % (qid, crit.index(None)))
        elif crit is not None and not isinstance(crit, dict):
            raise ValueError("question %r: a noul question takes 'criteria' as a dict with optional "
                             "'true'/'false' descriptions, or omits it" % (qid,))
        elif isinstance(crit, dict):
            # `render_options` reads these two descriptions out by name -- `crit.get("false")` and
            # `crit.get("true")` -- so a dict keyed any other way is not a noul description at all.
            # It used to be substituted with the default pair without a word, so a caller saw their
            # descriptions accepted and never reach the model (#156). `labels` just below has
            # rejected the same mistake since #163; this is the same rule on the other parameter,
            # and a noul is a boolean question either way, so those are the only two keys it can have.
            keys = {str(k).lower() for k in crit}
            if not keys <= {"true", "false"}:
                raise ValueError(
                    "question %r: a noul question takes 'criteria' keyed only 'true'/'false' (either "
                    "or both, and omitted is fine), got %s. Those keys are the option texts the model "
                    "reads; any other key was silently dropped and replaced with the defaults. If you "
                    "want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set "
                    "'labels' instead." % (qid, sorted(keys)))
        if "labels" in qdef:
            if t != "noul":
                raise ValueError("question %r: 'labels' is only supported for noul questions" % (qid,))
            try:
                _resolve_noul_labels(qdef["labels"])
            except ValueError as e:
                raise ValueError("question %r: %s" % (qid, e)) from e

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
            # `ensure_ascii=False`, matching `serialize_state` and `render_criterion` in
            # common.py and the instructions path in shortlist.py. The default escaped
            # non-ASCII to literal `\uXXXX`, which the tokenizer then read as escape text:
            # on the English checkpoint one German question answered noul=0.1652 as a dict
            # and noul=0.2650 as the identical plain string.
            ins = json.dumps(ins, ensure_ascii=False)
        q = {"t": t, "ins": ins, "crit": crit}
        if "labels" in qdef:
            q["labels"] = qdef["labels"]
        return q

    def _encode_state(self, state: Union[str, dict, list], ids: List[str], internal: Dict[str, Dict],
                      max_len: Optional[int] = None, head_max_len: Optional[int] = None) -> List[Dict]:
        """Tokenize one state against every (already validated + normalized) question.

        `max_len` / `head_max_len` override the agent config for this call (a start hook may set
        `ctx.max_len` / `ctx.head_max_len`).
        """
        max_len = self.cfg.get("max_len", 512) if max_len is None else max_len
        head_max_len = self.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len
        # A chronological conversation list is serialized newest-last, so the default
        # right-truncation (st[:room]) would silently drop the newest turn. Truncate
        # from the left for lists so the most recent intent is preserved.
        truncate_left = isinstance(state, list)
        # Tokenize the shared state once. The ids are identical for every question, so
        # re-serializing and re-tokenizing it inside build_sequence per question was pure
        # duplicated work. Tokenize in full and let build_sequence slice per question, so
        # left-truncation for conversation lists keeps its meaning.
        state_ids = encode_text(
            self.tok,
            serialize_state(state).replace(self.tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        items = []
        for qid in ids:
            q = internal[qid]
            seq, markers, stats = build_sequence(self.tok, state, q, max_len, head_max_len,
                                                 truncate_left=truncate_left, state_ids=state_ids,
                                                 return_stats=True)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "options": stats})
        return items

    def _amp_enabled_for(self, rows: int) -> bool:
        """Whether to autocast a forward with `rows` question rows.

        MPS fp16 loses to fp32 on a single small row and wins once the batch grows, so it is
        only enabled at or above `mps_amp_min_rows`. Other devices are unaffected.
        """
        if not self.amp_enabled:
            return False
        if self.device.type == "mps" and rows < self.mps_amp_min_rows:
            return False
        return True

    def _infer(self, b: Dict):
        """Run the forward pass under autocast, degrading gracefully on OOM or unsupported autocast."""
        use_amp = self._amp_enabled_for(b["input_ids"].shape[0])

        if self._fast is not None and b["input_ids"].shape[1] > self._fast.max_len:
            raise ValueError(
                "the CUDA fast path was built for max_len=%d; this request needs %d tokens. "
                "Use max_len <= %d or load without fast=True."
                % (self._fast.max_len, b["input_ids"].shape[1], self._fast.max_len)
            )

        def run():
            # Recomputed inside run() so a fallback that disables amp (or moves to CPU) takes
            # effect on the retry. A disabled gate never enters torch.autocast at all.
            enabled = self._amp_enabled_for(b["input_ids"].shape[0])
            with _amp_context(self.device, self.dtype, enabled):
                return self.model(
                    b["input_ids"].to(self.device),
                    b["attention_mask"].to(self.device),
                    b["marker_pos"].to(self.device),
                    b["marker_mask"].to(self.device),
                    b["qtype"].to(self.device),
                )

        try:
            return run()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            low = str(e).lower()
            # An OOM is `torch.cuda.OutOfMemoryError` or says so in its message. The bare
            # "cuda" substring used to be accepted too, so any CUDA-shaped RuntimeError (a
            # shape, assert or kernel error) silently and permanently demoted the agent.
            if self.device.type != "cpu" and (isinstance(e, torch.cuda.OutOfMemoryError) or "memory" in low):
                print("Warning: GPU memory exceeded during inference. Retrying this request on CPU...")
                # Scoped, not permanent: the demotion used to rewrite device/dtype/amp and move
                # the model for the life of the process, so one oversized request left every
                # later call ~10-15x slower on CPU. Demote under the lock, answer this request
                # on CPU, then put the runtime back the way it was.
                with _OOM_FALLBACK_LOCK:
                    held_device, held_dtype, held_amp = self.device, self.dtype, self.amp_enabled
                    had_fast = self._fast is not None
                    # FastLaya keeps copied CUDA weights and replaces model.forward.  Move
                    # the model first without that replacement, or the retry would still
                    # execute on the failed CUDA fast path.
                    self.deaccelerate()
                    self.device = torch.device("cpu")
                    self.dtype = torch.float32
                    self.amp_enabled = False
                    self.model.to(self.device)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        return run()
                    finally:
                        self._restore_runtime(held_device, held_dtype, held_amp, had_fast)
            if use_amp and self.device.type in ("mps", "cpu"):
                # Not every MPS/CPU build implements autocast for every op. Drop to full
                # precision once rather than failing the request.
                self.amp_enabled = False
                self.dtype = torch.float32
                return run()
            raise

    def _forward(self, b: Dict):
        """Run the model on a collated batch, with the GPU->CPU OOM fallback, and return numpy outputs."""
        logits, act = self._infer(b)
        return logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy()

    def _decode_answers(self, logits, act, items: List[Dict], ids: List[str],
                        internal: Dict[str, Dict], offset: int, lang: Optional[str] = None) -> Dict[str, Any]:
        """Turn one state's logit rows (starting at `offset`) into typed answers."""
        answers = {}
        for j, qid in enumerate(ids):
            r = offset + j
            q = internal[qid]
            k = len(items[j]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            if lang and lang.split("-")[0].lower() in self.lang_temperatures:
                l_cfg = self.lang_temperatures[lang.split("-")[0].lower()]
                t_scale = l_cfg["temperature_by_options"].get(temp_bucket(qt, k), l_cfg["temperature"][qt])
            z = logits[r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()

            # `confidence` means one thing for `noul` (max(p)) and another for `choice` and
            # `score` (normalized entropy), and only the first is the quantity temperature
            # scaling fits and ECE measures. Rather than change one underneath existing
            # callers, report both: `answer_confidence` is the calibrated one, on every
            # question type, so a caller can gate across types on a single number.
            ans_conf = round(answer_confidence(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": round(confidence_from_probs(p, k), 4),
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
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "answer_confidence": ans_conf,
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    # identical here: over two options max(p_true, 1 - p_true) is max(p)
                    "answer_confidence": ans_conf,
                    "action": ext,
                }
        return answers

    @torch.no_grad()
    def predict_batch(self, states: List[Union[str, dict, list]], questions: Dict[str, Dict[str, Any]],
                      batch_size: Optional[int] = None, lang: Optional[str] = None,
                      hooks=None,
                      on_predict_start=None, on_predict_end=None,
                      hooks_raise: Optional[bool] = None,
                      hooks_timeout: Optional[float] = None,
                      max_len: Optional[int] = None,
                      head_max_len: Optional[int] = None,
                      sort_by_length: bool = False) -> List[Dict[str, Any]]:
        """Evaluate the same questions over many states, packing them into shared forward passes.

        This is the throughput path. `system_one`/`predict` handle one state per forward pass; on a
        GPU that leaves most of the batch dimension idle. `predict_batch` collates several states'
        question rows into one tensor, so a call that would take N sequential forward passes takes
        one (or `ceil(len(states) / batch_size)`), which is several times faster per decision on GPU.

        Args:
            states: A list of states (each a text string, JSON dict, or conversation turn list).
                    The same `questions` are evaluated against every state.
            questions: Question definitions, exactly as accepted by `system_one`.
            batch_size: Optional cap on states per forward pass. `None` sends them all in one pass;
                        set it to bound peak memory when batching many or long states.
            hooks (HookArg): Per-call hooks, appended after any installed on the Agent.
                    See `laya.hooks`.
            on_predict_start (PredictHookArg): A per-call start hook. It may rewrite the
                    state/questions or call `ctx.skip(...)` to short-circuit inference.
            on_predict_end (PredictHookArg): A per-call end hook. It may rewrite the results.
            hooks_raise: Override the Agent's `hooks_raise` for this call.
            hooks_timeout: Override the Agent's `hooks_timeout` for this call.
            max_len: Override the agent config's `max_len` for this call. A start hook may also
                    set `ctx.max_len` to shape the token budget.
            head_max_len: Override the agent config's `head_max_len` for this call. A start hook
                    may also set `ctx.head_max_len`.
            sort_by_length: Group similarly sized encoded states within windows of eight batches
                    to reduce padding. Requires an explicit `batch_size` greater than one and
                    smaller than the number of states; otherwise it has no effect. Results retain
                    input order. This buffers up to eight batches of tokenized states instead of
                    one. Changing batch shapes can slightly change floating-point predictions.

        Returns:
            A list of per-state result dicts, each identical in shape to `system_one`'s output and
            aligned with `states` by index.
        """
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
                # that survives the hook is validated.
                if isinstance(states, (str, bytes, dict)):
                    raise TypeError(
                        "predict_batch expects a list of states; pass a single state to predict()/system_one()."
                    )
                states = list(states)
                if not states:
                    ctx.results = []
                else:
                    ids = list(questions.keys())
                    # Empty questions: empty answers, zero usage, no tokenization or forward.
                    if not ids:
                        ctx.results = [
                            {"model": "laya-rl-agent", "answers": {},
                             "usage": {"input_tokens": 0, "output_tokens": 0}} for _ in states
                        ]
                    else:
                        # Validate + normalize each question once (state-independent).
                        for qid in ids:
                            self._check_question(qid, questions[qid])
                        internal = {qid: self._to_internal(questions[qid]) for qid in ids}
                        chunk = batch_size if (batch_size and batch_size > 0) else len(states)

                        # Per-call token-budget overrides (a start hook may have set them).
                        overrides: Dict[str, int] = {}
                        if ctx.max_len is not None:
                            overrides["max_len"] = ctx.max_len
                        if ctx.head_max_len is not None:
                            overrides["head_max_len"] = ctx.head_max_len

                        results: List[Dict[str, Any]] = []
                        reorder = sort_by_length and 1 < chunk < len(states)
                        # Bound tokenized lookahead independently of the full input size. Use
                        # actual post-truncation lengths, with each state's questions kept together.
                        window = chunk * 8 if reorder else chunk
                        for start in range(0, len(states), window):
                            part = states[start:start + window]
                            encoded = [self._encode_state(st, ids, internal, **overrides) for st in part]
                            order = list(range(len(encoded)))
                            if reorder:
                                order.sort(key=lambda i: max(len(item["ids"]) for item in encoded[i]))
                            window_results = [None] * len(encoded)
                            for offset in range(0, len(order), chunk):
                                indices = order[offset:offset + chunk]
                                per_state_items = [encoded[i] for i in indices]

                                b = collate_items(per_state_items, self.tok.pad_token_id)
                                logits, act = self._forward(b)
                                att = b["attention_mask"]

                                row = 0
                                for index, items in zip(indices, per_state_items):
                                    nrows = len(items)
                                    n_tokens = int(att[row:row + nrows].sum())
                                    answers = self._decode_answers(logits, act, items, ids, internal, row,
                                                                  **({"lang": lang} if lang else {}))
                                    usage = {"input_tokens": n_tokens, "output_tokens": 0}
                                    # Only when a question actually lost options to the head
                                    # budget: an answer chosen from 42 distinguishable spans of
                                    # 58 has a ceiling the caller cannot otherwise see, and a
                                    # key that is always present would be noise on the
                                    # overwhelming majority of requests that never collapse.
                                    collapsed = collapsed_options(ids, items)
                                    if collapsed:
                                        usage["options"] = collapsed
                                    window_results[index] = {
                                        "model": "laya-rl-agent",
                                        "answers": answers,
                                        "usage": usage,
                                    }
                                    row += nrows
                            results.extend(window_results)
                        ctx.results = results
        except BaseException as exc:
            ctx.error = exc
            try:
                dispatch(active, "on_error", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                # A failing on_error hook must not hide the failure that triggered it.
                exc.__context__ = hook_exc
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000.0
            if ctx.results is not None:
                ctx.usage = aggregate_usage(ctx.results)
            try:
                dispatch(active, "on_predict_end", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                # End hooks run on the failure path too; do not let one mask the real error.
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

        `system_one`/`predict` truncate a state that exceeds `max_len` to a single window (the
        first, or for a conversation list the last), silently dropping the rest. `predict_long`
        tokenizes the state once, splits it into overlapping token windows, scores every window in
        shared forward passes (via `predict_batch`), and combines the per-window answers:

          * noul  -> P(true) is the max over windows (the statement holds if any window supports it)
          * choice-> the answer from the single most-confident window, so a localized signal isn't
                     out-voted by the many neutral windows a long document is mostly made of
                     (averaging drowns it -- the neutral majority dominates)
          * score -> the level from the most-confident window, likewise

        The returned probability/confidence is the deciding window's, **not a calibrated number for
        the whole document**: a `noul` max over many windows drifts up with the window count even
        with no signal, and `choice` can land on a confidently-neutral window when nothing in the
        document is decisive. Each answer therefore carries `answer["window"]` — the deciding
        window's `index`, `token_start`/`token_end` into the tokenized state, and the window `count`
        — so a caller can inspect the span the answer came from rather than trust the raw number.

        A state that already fits one window is passed straight to `system_one` (identical output).

        Args:
            window: state tokens per window. Defaults to the per-question state budget
                    (`max_len - head_max_len - 8`) -- the most a window can hold for every question.
                    A smaller window isolates a localized signal better (a short deciding span is a
                    larger fraction of its window, so that window classifies it clearly), at the
                    cost of more windows; the large default favors context and throughput. `noul`
                    is robust to this, `choice`/`score` benefit from a smaller window when the
                    deciding span is a small part of a long, otherwise-neutral document.
            stride: token step between windows. Defaults to `window // 2` (50% overlap), so a span
                    near a boundary still lands whole inside some window.
            aggregate: "auto" (the per-type rules above) is the only mode for now.
            batch_size: cap on windows per forward pass, to bound memory on very long states.
            lang: per-language temperature selection, as in `system_one`.

        Returns a single result dict, the same shape as `system_one`, with `usage["windows"]` added.
        """
        if aggregate != "auto":
            raise ValueError("predict_long: only aggregate='auto' is supported")
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        budget = window if (window and window > 0) else max(64, max_len - head_max_len - 8)

        state_ids = self.tok(serialize_state(state).replace(self.tok.mask_token, " "),
                             add_special_tokens=False)["input_ids"]
        # Fits in one window: identical to a plain call, no windowing overhead.
        if len(state_ids) <= budget:
            return self.system_one(state, questions, lang=lang)

        step = stride if (stride and stride > 0) else max(1, budget // 2)
        windows, starts = [], []
        i, n = 0, len(state_ids)
        while i < n:
            # Decode each token window back to text so predict_batch re-tokenizes it as a normal
            # state. For BPE tokenizers the re-tokenized boundaries can shift by a token or two vs
            # this split; harmless for aggregation since the 50% default overlap absorbs it.
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
                # Evidence anywhere: the strongest window decides. Its own P(true) and confidence
                # (and act) are carried through, so the fields stay mutually consistent.
                best = max(range(len(per)), key=lambda j: float(per[j]["noul"]))
            else:
                # choice / score: the most-confident window wins. Averaging over a long, mostly
                # neutral document lets the neutral majority out-vote the one window that saw the
                # deciding span; the single most-confident window preserves a localized signal.
                best = max(range(len(per)), key=lambda j: float(per[j]["answer_confidence"]))
            ans = per[best]
            # Name the window that decided, so a caller can check the deciding span itself. The
            # probability here is that window's, NOT a document-level calibrated number.
            ans["window"] = {"index": best, "token_start": starts[best],
                             "token_end": min(starts[best] + budget, len(state_ids)),
                             "count": len(windows)}
            answers[qid] = ans
        # Aggregate usage generically so fields predict_batch may grow later (e.g. the fallback
        # counters from #351) are propagated, not silently dropped: sum numeric fields across
        # windows, carry any non-numeric field through, then record the window count.
        usage: Dict[str, Any] = {}
        for r in results:
            for key, val in r["usage"].items():
                usage[key] = (usage.get(key, 0) + val) if isinstance(val, (int, float)) else val
        usage["output_tokens"] = 0
        usage["windows"] = len(windows)
        return {"model": "laya-rl-agent", "answers": answers, "usage": usage}

    @torch.no_grad()
    def system_one(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]], lang: Optional[str] = None,
                   hooks=None, on_predict_start=None, on_predict_end=None,
                   hooks_raise: Optional[bool] = None,
                   hooks_timeout: Optional[float] = None,
                   max_len: Optional[int] = None,
                   head_max_len: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate typed questions across state in a single, parallel forward pass.

        Args:
            state: Text string, JSON dict, or conversation turn list.
            questions: Dictionary mapping question_id -> question definition.
                - choice: {"type": "choice", "instructions": "...", "criteria": {"optA": "...", ...}}
                - score:  {"type": "score",  "instructions": "...", "criteria": ["lvl0", "lvl1", ...]}
                - noul:   {"type": "noul", "instructions": "...",
                           "criteria": {"false": "...", "true": "..."},
                           "labels": {"false": "B", "true": "A"}}

                  Noul criteria and labels are optional. Labels only control the text shown to the
                  model; their keys retain false/true semantics, and the returned `noul` value is
                  always P(true). Labels default to false/true for compatibility.

        Returns:
            Dictionary with answers, probabilities, calibrated confidence, and token usage.
            Empty questions return empty answers and zero token usage without tokenization
            or a model forward pass.

            When the head budget leaves two options with the same token span, `usage` carries
            an `options` entry for each question it happened to -- `total`, `distinct` and
            `tokens_per_option` -- because an answer chosen among 42 distinguishable spans of
            58 has a ceiling that is the budget's and not the model's. Questions whose options
            all survive are absent, so a request that collapses nothing is unchanged.

        To score many states at once, see `predict_batch`, which shares forward passes across them.
        """
        return self.predict_batch([state], questions, lang=lang, hooks=hooks,
                                  on_predict_start=on_predict_start,
                                  on_predict_end=on_predict_end, hooks_raise=hooks_raise,
                                  hooks_timeout=hooks_timeout,
                                  max_len=max_len, head_max_len=head_max_len)[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        except Exception:
            pass
        return False

    def decide(self, state: Union[str, dict, list], schema: Any = None, *,
               questions: Optional[Dict[str, Any]] = None, return_details: bool = False,
               **predict_kwargs) -> Any:
        """Answer `state` against a schema (JSON schema or pydantic model) and return typed values.

        See `laya.structured`. Pass exactly one of `schema` or `questions`; extra keyword arguments
        are forwarded to `predict` / `system_one`.
        """
        from .structured import decide as _decide
        return _decide(self, state, schema, questions=questions,
                       return_details=return_details, **predict_kwargs)

    def __repr__(self) -> str:
        return "Agent(model_id=%r, device=%s)" % (self.model_id, getattr(self, "device", None))

    predict = system_one


RLAgent = Agent


def load(model_id_or_path: str = "convaiinnovations/laya", device: Optional[str] = None,
         token: Optional[str] = None, subfolder: Optional[str] = None, fast: bool = False,
         revision: Optional[str] = None, expected_sha256: Optional[Dict[str, str]] = None,
         lang_temperatures: Optional[Dict[str, Dict[str, Any]]] = None,
         hooks=None, on_predict_start=None, on_predict_end=None,
         hooks_raise: bool = True, hooks_concurrent: bool = True,
         hooks_timeout: Optional[float] = None) -> Agent:
    """Load a Laya agent.

    `subfolder` picks one checkpoint out of a repo that bundles several:

        laya.load("convaiinnovations/laya")                           # English (repo root)
        laya.load("convaiinnovations/laya", subfolder="multilingual")
        laya.load("convaiinnovations/laya", fast=True)                # TileLang GPU fast path

    `revision`/`expected_sha256` pin and verify the downloaded artifacts; see `Agent`.
    `hooks` / `on_predict_start` / `on_predict_end` observe or shape every prediction; see
    `laya.hooks`.
    """
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder, fast=fast,
                 revision=revision, expected_sha256=expected_sha256,
                 lang_temperatures=lang_temperatures,
                 hooks=hooks, on_predict_start=on_predict_start, on_predict_end=on_predict_end,
                 hooks_raise=hooks_raise, hooks_concurrent=hooks_concurrent,
                 hooks_timeout=hooks_timeout)
