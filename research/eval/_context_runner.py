"""CPU reference runner; imported only by the context-robustness run command."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import re
import subprocess

from .context_robustness import digest, file_digest, require


def runtime_settings(agent):
    return {"device": str(agent.device), "backend": agent.backend, "dtype": str(agent.dtype),
            "amp_enabled": agent.amp_enabled, "config": agent.cfg,
            "temperature": list(agent.temperature),
            "temperature_by_options": dict(agent.temperature_by_options),
            "lang_temperatures": getattr(agent, "lang_temperatures", None),
            "binning_map": agent.binning_map}


def source_identity():
    import laya
    root = Path(laya.__file__).parent
    sources = {p.relative_to(root).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(root.rglob("*.py"))}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"commit": commit, "core_source_sha256": digest(sources)}


class _Runner:
    def __init__(self, agent, provenance):
        self.agent = agent
        self.provenance = provenance
        self.runtime_hash = digest(runtime_settings(agent))

    def predict(self, state, questions, **settings):
        require(digest(runtime_settings(self.agent)) == self.runtime_hash, "runtime settings changed")
        result = self.agent.predict(state, questions, **settings)
        require(digest(runtime_settings(self.agent)) == self.runtime_hash, "runtime settings changed during prediction")
        return result


@contextmanager
def cpu_runner(model, revision=None, subfolder=None):
    local = Path(model).is_dir()
    if local:
        require(revision is None, "local paths use artifact hashes, not a requested Hub revision")
    else:
        require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
                "remote models require an explicit 40-character commit SHA")
    from laya import Agent
    with Agent(model_id_or_path=model, revision=revision, subfolder=subfolder,
               device="cpu", backend="eager") as agent:
        settings = runtime_settings(agent)
        require(settings["device"] == "cpu" and settings["backend"] == "eager"
                and settings["amp_enabled"] is False and settings["dtype"] == "torch.float32",
                "reference protocol requires CPU eager without AMP")
        if not local:
            require(agent.revision == revision, "resolved model revision differs from requested SHA")
        model_dir = Path(agent.tok.name_or_path).parent
        require((model_dir / "model.safetensors").is_file()
                and (model_dir / "rl_agent_config.json").is_file(), "cannot identify loaded checkpoint artifacts")
        files = {p.relative_to(model_dir).as_posix(): file_digest(p) for p in sorted(model_dir.rglob("*"))
                 if p.is_file() and (p.relative_to(model_dir).as_posix() in ("model.safetensors", "rl_agent_config.json")
                                     or p.relative_to(model_dir).parts[0] in ("encoder", "tokenizer"))}
        provenance = {"model": model, "subfolder": subfolder, "requested_revision": revision,
                      "resolved_revision": agent.revision, "checkpoint_files_sha256": files,
                      "runtime": settings, "source": source_identity(),
                      "python": platform.python_version(), "platform": platform.platform(),
                      "packages": {name: importlib.metadata.version(name) for name in
                                   ("torch", "numpy", "transformers", "tokenizers", "safetensors", "huggingface-hub")}}
        yield _Runner(agent, provenance)
        require(source_identity()["core_source_sha256"] == provenance["source"]["core_source_sha256"],
                "core source changed during collection")
