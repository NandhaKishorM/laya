"""High-level inference runtime for laya System 1 decision models."""
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .common import (
    QTYPES,
    amp_dtype,
    build_model,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
)
from .errors import (
    ConfigurationError,
    DeviceError,
    IncompatibleModelError,
    InferenceError,
    InvalidCriteriaError,
    ModelNotFoundError,
    OptionsTooLongError,
    QuestionError,
    UnknownQuestionTypeError,
)
from .types import (
    Action,
    ChoiceAnswer,
    NoulAnswer,
    Response,
    ScoreAnswer,
    Truncation,
    Usage,
)

logger = logging.getLogger("laya")


def _fix_tokenizer_config(path: str):
    """Ensure tokenizer_config.json can be loaded across all transformers versions."""
    cfg_file = os.path.join(path, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return
    try:
        with open(cfg_file) as f:
            tcfg = json.load(f)
        if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            with open(cfg_file, "w") as f:
                json.dump(tcfg, f, indent=2)
    except Exception:
        pass


def _verify_compatibility(model: torch.nn.Module, cfg: Dict, weights: Dict[str, torch.Tensor], model_id: str):
    """Verify that the loaded checkpoint weights and config strictly match the expected architecture."""
    # 1. Verify required configuration attributes
    required_cfg = ["encoder", "head_layers"]
    missing_cfg = [k for k in required_cfg if k not in cfg]
    if missing_cfg:
        raise ConfigurationError(
            f"Incompatible model config for {model_id!r}: missing configuration keys {missing_cfg}. "
            f"Ensure this is a valid RL Agent decision model."
        )

    # 2. Check for required component prefixes
    required_prefixes = ("encoder.", "type_emb.", "scorer.", "act_head.")
    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in weights.keys()):
            raise IncompatibleModelError(
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
        raise IncompatibleModelError(
            f"Model architecture mismatch for {model_id!r}:\n{err_details}\n"
            f"The checkpoint weights do not match the configured model architecture."
        )

    if missing_keys:
        raise IncompatibleModelError(
            f"Model weights incomplete for {model_id!r}: missing {len(missing_keys)} parameter tensors "
            f"(e.g. {missing_keys[:3]})."
        )


class Agent:
    """System 1 decision model runtime: fast, non-autoregressive, calibrated decisions."""

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        token: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        truncate_left: bool = False,
    ):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        model_dir = model_id_or_path
        if not os.path.exists(model_dir):
            if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
                raise ModelNotFoundError(
                    f"Local model path not found: {model_id_or_path!r}. "
                    f"Check that the directory exists and that training saved the model successfully."
                )
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(model_id_or_path, token=token or os.environ.get("HF_TOKEN"))

        _fix_tokenizer_config(model_dir)

        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise ModelNotFoundError(
                f"Incompatible model: {model_id_or_path!r} does not contain 'rl_agent_config.json'. "
                f"Make sure you are loading a compatible RL Agent model (e.g. 'convaiinnovations/rl-agent')."
            )

        with open(cfg_path) as f:
            self.cfg = json.load(f)

        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise ModelNotFoundError(
                f"Incompatible model: 'model.safetensors' not found in {model_id_or_path!r}."
            )

        # 1. Device resolution with automatic fallback
        if device is not None:
            target_device = torch.device(device)
            if target_device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("laya: CUDA requested but not available; falling back to CPU.")
                self.device = torch.device("cpu")
            elif target_device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("laya: MPS requested but not available; falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = target_device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        tok_dir = os.path.join(model_dir, "tokenizer")
        self.tok = AutoTokenizer.from_pretrained(tok_dir if os.path.exists(tok_dir) else self.cfg.get("encoder"))

        enc_dir = os.path.join(model_dir, "encoder")
        self.model = build_model(self.cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)

        # Load weights and verify architectural compatibility
        weights = load_file(weights_path)
        _verify_compatibility(self.model, self.cfg, weights, model_id_or_path)

        self.model.load_state_dict(weights, strict=True)

        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})

        # Context window. Defaults come from the checkpoint config; callers may raise them,
        # but note the encoder was fine-tuned at cfg["max_len"] -- going far beyond that is
        # untested and will degrade calibration until the model is retrained at the new length.
        self.max_len = int(max_len if max_len is not None else self.cfg.get("max_len", 512))
        self.head_max_len = int(
            head_max_len if head_max_len is not None else self.cfg.get("head_max_len", 192)
        )
        self.truncate_left = bool(truncate_left)

        trained_len = int(self.cfg.get("max_len", 512))
        if self.max_len > trained_len:
            logger.warning(
                "laya: max_len=%d exceeds the checkpoint's trained max_len=%d. Sequence building "
                "will work, but accuracy and calibration beyond the trained length are unverified.",
                self.max_len, trained_len,
            )
        self.dtype = amp_dtype(self.cfg.get("amp_dtype", "fp16"))

        if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] < 8:
            self.dtype = torch.float16
        elif self.device.type in ("cpu", "mps"):
            self.dtype = torch.float32

        # 2. Place on device with graceful fallback to CPU on memory error
        try:
            self.model.to(self.device).eval()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if self.device.type != "cpu":
                logger.warning(
                    "laya: could not place model on %s (%s); falling back to CPU.", self.device, e
                )
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.model.to(self.device).eval()
            else:
                raise DeviceError("failed to place model on CPU: %s" % e) from e

    @staticmethod
    def _to_internal(qdef: Dict, qid: str = "?") -> Dict:
        if not isinstance(qdef, dict):
            raise QuestionError("question %r must be a dict, got %s" % (qid, type(qdef).__name__))
        if "type" not in qdef:
            raise QuestionError("question %r is missing required key 'type'" % qid)
        if "instructions" not in qdef:
            raise QuestionError("question %r is missing required key 'instructions'" % qid)

        t = qdef["type"]
        if t not in QTYPES:
            raise UnknownQuestionTypeError(
                "question %r has unknown type %r; expected one of %s" % (qid, t, ", ".join(sorted(QTYPES)))
            )

        crit = qdef.get("criteria")
        if t == "choice":
            if isinstance(crit, list):
                crit = {c: None for c in crit}
            if not isinstance(crit, dict) or not crit:
                raise InvalidCriteriaError(
                    "choice question %r needs non-empty 'criteria' as a dict of option -> description "
                    "(or a list of option names)" % qid
                )
        elif t == "score":
            if not isinstance(crit, (list, tuple)) or len(crit) < 2:
                raise InvalidCriteriaError(
                    "score question %r needs 'criteria' as a list of at least 2 ordered levels" % qid
                )
            crit = list(crit)
        elif crit is not None and not isinstance(crit, dict):
            raise InvalidCriteriaError(
                "noul question %r criteria must be a dict with 'true'/'false' keys, got %s"
                % (qid, type(crit).__name__)
            )

        ins = qdef["instructions"]
        if not isinstance(ins, str):
            ins = json.dumps(ins, ensure_ascii=False)
        return {"t": t, "ins": ins, "crit": crit}

    @torch.no_grad()
    def system_one(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate typed questions across state in a single, parallel forward pass.

        Args:
            state: Text string, JSON dict, or conversation turn list.
            questions: Dictionary mapping question_id -> question definition.
                - choice: {"type": "choice", "instructions": "...", "criteria": {"optA": "...", ...}}
                - score:  {"type": "score",  "instructions": "...", "criteria": ["lvl0", "lvl1", ...]}
                - noul:   {"type": "noul",   "instructions": "..."}

        Returns:
            Dictionary with answers, probabilities, calibrated confidence, and token usage.
        """
        if not questions:
            raise QuestionError("questions must not be empty")

        request_id = "req_" + uuid.uuid4().hex[:16]
        ids = list(questions.keys())
        items = []
        internal = {}
        truncated: List[Truncation] = []
        max_len = self.max_len
        head_max_len = self.head_max_len

        for qid in ids:
            q = self._to_internal(questions[qid], qid)
            internal[qid] = q
            events = []
            seq, markers = build_sequence(
                self.tok,
                state,
                q,
                max_len,
                head_max_len,
                truncate_left=self.truncate_left,
                on_truncate=lambda kept, total: events.append((kept, total)),
            )
            if events:
                kept, total = events[0]
                truncated.append(Truncation(question_id=qid, kept_tokens=kept, state_tokens=total))
                logger.warning(
                    "laya[%s]: state truncated for question %r: kept %d of %d tokens "
                    "(max_len=%d, truncate_left=%s). %d tokens were dropped.",
                    request_id, qid, kept, total, max_len, self.truncate_left, total - kept,
                )
            n_opts = len(render_options(q))
            if len(markers) != n_opts:
                raise OptionsTooLongError(
                    "question %r: %d options do not fit head_max_len=%d (only %d fit). "
                    "Use fewer or shorter options, or raise 'head_max_len' in the model config."
                    % (qid, n_opts, head_max_len, len(markers))
                )
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        b = collate_items([items], self.tok.pad_token_id)
        use_amp = self.device.type == "cuda"

        try:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=use_amp):
                logits, act = self.model(
                    b["input_ids"].to(self.device),
                    b["attention_mask"].to(self.device),
                    b["marker_pos"].to(self.device),
                    b["marker_mask"].to(self.device),
                    b["qtype"].to(self.device),
                )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if self.device.type != "cpu" and ("memory" in str(e).lower() or "cuda" in str(e).lower()):
                logger.warning(
                    "laya[%s]: GPU memory exceeded during inference; falling back to CPU.", request_id
                )
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.model.to(self.device)
                logits, act = self.model(
                    b["input_ids"].to(self.device),
                    b["attention_mask"].to(self.device),
                    b["marker_pos"].to(self.device),
                    b["marker_mask"].to(self.device),
                    b["qtype"].to(self.device),
                )
            else:
                raise InferenceError("forward pass failed: %s" % e) from e

        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()

        answers = {}
        n_tokens = int(b["attention_mask"].sum())

        for r, qid in enumerate(ids):
            q = internal[qid]
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[r, :k] / max(1e-3, float(t_scale))
            p = np.exp(z - z.max())
            p = p / p.sum()

            conf_score = round(confidence_from_probs(p, k), 4)
            ext = Action(act_probability=round(float(act[r, 0]), 4))

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = ChoiceAnswer(
                    choice=keys[int(p.argmax())],
                    probabilities={kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    confidence=conf_score,
                    action=ext,
                )
            elif q["t"] == "score":
                exp_score = float((np.arange(k) * p).sum())
                answers[qid] = ScoreAnswer(
                    score=round(exp_score, 4),
                    legend={str(i): c for i, c in enumerate(q["crit"])},
                    probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    confidence=conf_score,
                    action=ext,
                )
            else:
                answers[qid] = NoulAnswer(
                    noul=round(float(p[1]), 4),
                    confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    action=ext,
                )

        return Response(
            answers=answers,
            usage=Usage(input_tokens=n_tokens, output_tokens=0),
            request_id=request_id,
            truncated=truncated,
        )

    predict = system_one


RLAgent = Agent


def load(model_id_or_path: str = "convaiinnovations/laya", device: Optional[str] = None, token: Optional[str] = None) -> Agent:
    """Helper function to load a Laya agent model."""
    return Agent(model_id_or_path, device=device, token=token)
