"""Explicitly opt into a local numeric-comparison specialist without changing Agent."""
import copy
import hashlib
import json
from pathlib import Path
import threading

from ._comparison_data import OPERATORS, QUESTIONS, VERDICTS, collate, encode, validate_state


def comparison_questions():
    """Return a fresh copy of the specialist's supported three-way question schema."""
    return copy.deepcopy(QUESTIONS)


def _batch_size(value):
    if value is None:
        return 8
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("batch_size must be a positive integer or None")
    return value


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ComparisonSpecialist:
    """Two frozen local networks with learned bindings and exact unit comparisons.

    Load a compatible bundle with ``from_pretrained``. A normal LAYA checkpoint
    is not a specialist bundle. Probabilities are uncalibrated, and correct
    arithmetic does not guarantee correct language interpretation or binding.
    No hosted inference, automatic download, or implicit CPU fallback is used.
    """
    def __init__(self, operator, operands, tokenizer, *, device=None,
                 operator_precision="fp32", operand_precision="bf16"):
        import torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("ComparisonSpecialist supports CPU and CUDA devices")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if operator_precision not in ("fp32", "bf16") or operand_precision not in ("fp32", "bf16"):
            raise ValueError("Precisions must be fp32 or bf16")
        if self.device.type == "cuda" and "bf16" in (operator_precision, operand_precision):
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("This specialist requires BF16-capable CUDA hardware")
        if getattr(operator, "variant", None) not in ("flat", "local_program"):
            raise ValueError("Unsupported comparison operator architecture")
        if not getattr(tokenizer, "is_fast", False) or any(getattr(tokenizer, name, None) is None for name in
                ("cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id")):
            raise ValueError("Comparison bundles require a fast tokenizer with CLS, SEP, PAD and MASK tokens")
        self.operator = operator.to(device=self.device, dtype=torch.float32).eval()
        self.operands = operands.to(device=self.device, dtype=torch.float32).eval()
        self.tokenizer = tokenizer
        self.operator_precision, self.operand_precision = operator_precision, operand_precision
        self._lock = threading.RLock()

    @classmethod
    def from_pretrained(cls, directory, *, device=None, expected_sha256=None):
        """Load a local safetensors bundle after verifying its complete manifest.

        ``directory`` must already exist. ``expected_sha256`` optionally supplies
        independently trusted relative-file digests, as for ``laya.Agent``. The
        bundled manifest detects changed files; it does not authenticate a producer.
        """
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from ._comparison_model import OperandNetwork, OperatorNetwork
        from .common import build_model
        from .revisions import verify_digests

        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError("Comparison bundle must be an existing local directory")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        required = {"comparison_config.json", "model.safetensors", "operator_encoder/config.json",
                    "operand_encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
        if not isinstance(manifest, dict) or not required.issubset(manifest):
            raise ValueError("Incomplete comparison bundle manifest")
        for name in manifest:
            if not isinstance(name, str) or not (root / name).resolve().is_relative_to(root):
                raise ValueError("Comparison manifest path escapes its directory")
        files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        if files != set(manifest) | {"manifest.json"}:
            raise ValueError("Unmanifested or missing comparison bundle files")
        verify_digests(str(root), manifest)
        if expected_sha256 is not None:
            verify_digests(str(root), expected_sha256)
        config = json.loads((root / "comparison_config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("format") != "laya-comparison-v1" or config.get("variant") not in ("flat", "local_program"):
            raise ValueError("Unsupported comparison bundle format or architecture")
        if any(config.get(key) not in ("fp32", "bf16") for key in ("operator_precision", "operand_precision")):
            raise ValueError("Invalid comparison precision")
        for role in ("operator", "operand"):
            settings = config.get(role, {})
            if not isinstance(settings, dict):
                raise ValueError("Invalid comparison network configuration")
            width, layers = settings.get("width"), settings.get("head_layers")
            if isinstance(width, bool) or not isinstance(width, int) or width < 4 or width > 1024 or width % 4:
                raise ValueError("Invalid comparison network width")
            if isinstance(layers, bool) or not isinstance(layers, int) or not 0 <= layers <= 32:
                raise ValueError("Invalid comparison head layer count")
        # Constructors overwrite every tensor below; do not perturb the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            networks = []
            for role in ("operator", "operand"):
                base = build_model(config[role], encoder_dir=str(root / (role + "_encoder")), pretrained=False)
                if hasattr(base.encoder.config, "reference_compile"):
                    base.encoder.config.reference_compile = False
                network = (OperatorNetwork(base, config["variant"], config[role]["width"]) if role == "operator"
                           else OperandNetwork(base, config[role]["width"]))
                networks.append(network)
            weights = load_file(str(root / "model.safetensors"), device="cpu")
            expected = {role + "." + name for role, net in zip(("operator", "operand"), networks)
                        for name in net.state_dict()}
            if set(weights) != expected:
                raise ValueError("Comparison weights do not exactly match the architecture")
            for role, net in zip(("operator", "operand"), networks):
                net.load_state_dict({key[len(role) + 1:]: value for key, value in weights.items()
                                     if key.startswith(role + ".")}, strict=True)
        tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True, trust_remote_code=False)
        return cls(*networks, tok, device=device, operator_precision=config["operator_precision"],
                   operand_precision=config["operand_precision"])

    def save_pretrained(self, directory):
        """Write a portable local bundle into a new directory, never overwrite one.

        The manifest is written last. A failed partial export cannot be loaded.
        Bundle weights and tokenizer files belong outside source control.
        """
        from safetensors.torch import save_file
        root = Path(directory).expanduser()
        with self._lock:
            root.mkdir(parents=True, exist_ok=False)
            config = {"format": "laya-comparison-v1", "variant": self.operator.variant,
                      "operator_precision": self.operator_precision, "operand_precision": self.operand_precision}
            weights = {}
            for role, net in (("operator", self.operator), ("operand", self.operands)):
                folder = root / (role + "_encoder")
                folder.mkdir()
                encoder_config = net.encoder.config.to_dict()
                encoder_config.pop("_name_or_path", None)
                encoder_config.pop("_commit_hash", None)
                (folder / "config.json").write_text(json.dumps(encoder_config, indent=2) + "\n", encoding="utf-8")
                config[role] = {"width": net.width, "head_layers": 0 if net.head is None else len(net.head.layers)}
                weights.update({role + "." + key: value.detach().cpu().contiguous()
                                for key, value in net.state_dict().items()})
            self.tokenizer.save_pretrained(str(root / "tokenizer"))
            save_file(weights, str(root / "model.safetensors"))
            (root / "comparison_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            manifest = {path.relative_to(root).as_posix(): _digest(path) for path in root.rglob("*") if path.is_file()}
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def predict_batch(self, states, *, batch_size=8):
        """Evaluate numeric-comparison states, rejecting unsupported or truncated inputs."""
        import torch
        from .common import _TOKENIZE_LOCK
        size = _batch_size(batch_size)
        if not isinstance(states, list):
            raise TypeError("states must be a list")
        validated = [validate_state(state) for state in states]
        with self._lock, torch.inference_mode():
            if self.operator.training or self.operands.training:
                raise RuntimeError("Comparison inference requires both networks in evaluation mode")
            # Validate/tokenize the complete call before evaluating its first batch.
            with _TOKENIZE_LOCK:
                rows = [encode(state, self.tokenizer) for state in validated]
            results = []
            for start in range(0, len(rows), size):
                group = rows[start:start + size]
                op, operands = collate(group, self.tokenizer.pad_token_id)
                op = {key: value.to(self.device) for key, value in op.items()}
                operands = {key: value.to(self.device) for key, value in operands.items()}
                with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda" and self.operator_precision == "bf16"):
                    probability = self._probabilities(self.operator(op), (len(group), 6))
                with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda" and self.operand_precision == "bf16"):
                    logits, pointer_logits, gates = self.operands(operands, probability)
                verdict = self._probabilities(logits, (len(group), 3)).cpu()
                pointers = self._probabilities(pointer_logits, (len(group), 2, 8)).cpu()
                if tuple(gates.shape) != (len(group),) or not torch.isfinite(gates).all() or ((gates < 0) | (gates > 1)).any():
                    raise FloatingPointError("Invalid comparison mixture gate")
                probability = probability.cpu()
                for i, row in enumerate(group):
                    p = verdict[i].tolist()
                    index = int(verdict[i].argmax())
                    indices = pointers[i].argmax(-1).tolist()
                    fields = validated[start + i]["fields"]
                    results.append({"model": "laya-comparison",
                                    "answers": {"verdict": {"type": "choice", "choice": VERDICTS[index],
                                                             "probabilities": dict(zip(VERDICTS, p)),
                                                             "answer_confidence": max(p), "calibrated": False}},
                                    "usage": {"input_tokens": len(row["op_ids"]) + len(row["ids"]), "output_tokens": 0},
                                    "specialist": {"backend": self.operator.variant, "calibrated": False,
                                                   "operator": OPERATORS[int(probability[i].argmax())],
                                                   "operator_probabilities": probability[i].tolist(),
                                                   "operand_indices": indices,
                                                   "operand_names": [fields[j]["name"] if j < len(fields) else None for j in indices],
                                                   "operand_probabilities": pointers[i].tolist(), "gate": float(gates[i]),
                                                   "operator_precision": self.operator_precision if self.device.type == "cuda" else "fp32",
                                                   "operand_precision": self.operand_precision if self.device.type == "cuda" else "fp32"}})
            return results

    @staticmethod
    def _probabilities(logits, shape):
        import torch
        if tuple(logits.shape) != shape:
            raise ValueError("Invalid comparison network output shape")
        p = logits.float().softmax(-1)
        if not torch.isfinite(p).all():
            raise FloatingPointError("Nonfinite comparison probabilities")
        return p

    def predict(self, state):
        """Evaluate one numeric-comparison state."""
        return self.predict_batch([state], batch_size=1)[0]


class ComparisonAgent:
    """Wrap a base runner with explicitly selected comparison inference.

    The default path delegates directly to the base, including its hooks and
    keyword arguments. Enabled requests use the specialist's fixed question
    contract, not the base hooks or action head. Unsupported enabled options
    raise instead of being silently ignored. This is not an automatic router.
    """
    def __init__(self, base, specialist):
        self.base, self.specialist = base, specialist

    def predict_batch(self, states, questions=None, *, comparison=False, batch_size=None, **predict_kwargs):
        """Delegate normally, or explicitly evaluate a batch of comparison requests."""
        if not isinstance(comparison, bool):
            raise TypeError("comparison must be an explicit bool")
        if not comparison:
            return self.base.predict_batch(states, questions, batch_size=batch_size, **predict_kwargs)
        if predict_kwargs:
            raise TypeError("Enabled comparisons do not support base prediction options or hooks")
        if questions is not None and questions != comparison_questions():
            raise ValueError("Enabled comparisons require comparison_questions()")
        return self.specialist.predict_batch(states, batch_size=_batch_size(batch_size))

    def predict(self, state, questions=None, *, comparison=False, **predict_kwargs):
        """Delegate one request, or explicitly evaluate a numeric comparison."""
        if not isinstance(comparison, bool):
            raise TypeError("comparison must be an explicit bool")
        if not comparison:
            return self.base.predict(state, questions, **predict_kwargs)
        return self.predict_batch([state], questions, comparison=True, batch_size=1, **predict_kwargs)[0]

    predict_many = predict_batch
    system_one = predict
