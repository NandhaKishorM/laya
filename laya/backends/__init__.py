"""Inference backends: one contract, several ways to run `DecisionModel.forward`.

    laya.load(..., backend="auto" | "eager" | "compile" | "tilelang" | "onnx")

| backend    | what it is                                                        | needs                    |
|------------|-------------------------------------------------------------------|--------------------------|
| `eager`    | the stock PyTorch forward under autocast (the default)            | nothing                  |
| `compile`  | `torch.compile(dynamic=True, mode="reduce-overhead")` + warm-up   | CUDA                     |
| `tilelang` | fused TileLang kernels, bf16 weights, one CUDA graph per bucket   | CUDA, `laya[fast]`, bf16 |
| `onnx`     | ONNX Runtime via `laya.onnx_agent.ONNXAgent` (the explicit CPU option, its own class) | `laya[onnx]`, an exported model |

`auto` picks per device: CUDA -> `tilelang` when tilelang imports and the encoder is a
ModernBERT-family model (the shipped checkpoints), else `compile`; CPU and MPS -> `eager`.
Every backend honours `agent.dtype`, and one that cannot run falls back to `eager` with one
`RuntimeWarning`; `strict=True` raises instead. See `laya.backends.base` for the contract and
the shared dtype and padding policies.
"""
from .base import Backend, BackendUnavailable, amp_context, bucket_shape, choose_dtype, pad_batch, warn_fallback

NAMES = ("auto", "eager", "compile", "tilelang", "onnx")
AUTO = "auto"


def normalise(name):
    """Accept the documented spellings; anything else is a ValueError naming the choices."""
    key = str(name).strip().lower().replace("_", "-") if name is not None else "eager"
    aliases = {"fast": "tilelang", "torch": "eager", "torch.compile": "compile", "inductor": "compile"}
    key = aliases.get(key, key)
    if key not in NAMES:
        raise ValueError("unknown backend %r; use one of %s" % (name, ", ".join(NAMES)))
    return key


def tilelang_available() -> bool:
    try:
        import tilelang  # noqa: F401
        return True
    except Exception:
        return False


def is_modernbert(model) -> bool:
    cfg = getattr(getattr(model, "encoder", None), "config", None)
    return getattr(cfg, "model_type", None) == "modernbert"


def auto_policy(agent) -> str:
    """The backend `auto` resolves to for this agent (device, dtype, encoder, installed extras)."""
    if agent.device.type == "cuda":
        import torch
        if tilelang_available() and is_modernbert(agent.model) and agent.dtype == torch.bfloat16:
            return "tilelang"
        return "compile"
    return "eager"


def make_backend(name: str, agent, **options) -> Backend:
    """Build (not install) the backend `name` for `agent`. `auto` is resolved here."""
    key = normalise(name)
    if key == AUTO:
        key = auto_policy(agent)
    if key == "eager":
        from .eager import EagerBackend
        return EagerBackend(agent, **options)
    if key == "compile":
        from .compile import CompileBackend
        return CompileBackend(agent, **options)
    if key == "tilelang":
        try:
            from .tilelang import TileLangBackend
        except ImportError as e:
            raise BackendUnavailable("tilelang is not installed (pip install laya[fast]): %s" % e)
        return TileLangBackend(agent, **options)
    raise BackendUnavailable("backend %r is not a forward replacement; use laya.load(..., backend='onnx') "
                             "or laya.onnx_agent.ONNXAgent directly" % key)


def install(agent, name: str, strict: bool = False, **options) -> Backend:
    """Install `name` on `agent`, falling back to eager with one warning unless `strict`.

    Returns the backend that ended up installed. The agent's previous backend is uninstalled
    first, so switching is always eager -> new and never stacks two replacements.
    """
    from .eager import EagerBackend

    current = getattr(agent, "_backend", None)
    if current is not None:
        current.uninstall()
        agent._backend = None
    try:
        backend = make_backend(name, agent, **options)
        backend.install()
    except BackendUnavailable as e:
        if strict:
            raise
        warn_fallback(normalise(name), e)
        backend = EagerBackend(agent)
        backend.install()
    agent._backend = backend
    return backend


__all__ = ["Backend", "BackendUnavailable", "NAMES", "amp_context", "auto_policy", "bucket_shape",
           "choose_dtype", "install", "make_backend", "normalise", "pad_batch", "warn_fallback"]
