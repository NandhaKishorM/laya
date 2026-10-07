"""The stock PyTorch forward: no padding, no graphs, the reference every other backend is measured against."""
from .base import Backend


class EagerBackend(Backend):
    name = "eager"

    def install(self) -> None:
        # Nothing to swap: `model.forward` is already the eager forward. Marking it installed keeps
        # the lifecycle uniform for the Agent.
        self.installed = True

    def uninstall(self) -> None:
        self.installed = False
