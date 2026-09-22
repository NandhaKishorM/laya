"""Skip ModernBERT random init on Windows + Python 3.14.

``transformers`` 5.17 segfaults inside ``PreTrainedModel.initialize_weights``
while constructing a ModernBERT module on Windows 11 with Python 3.14 and
torch 2.14 (issue #123). ``Agent`` loads the checkpoint with
``load_state_dict(strict=True)`` immediately afterwards, so those initialized
values are never read. The verified workaround is to replace
``initialize_weights`` with a no-op before the encoder is built.

``build_model`` does that automatically, and only while the encoder is being
constructed, when ``platform.system() == "Windows"`` and
``sys.version_info >= (3, 14)``. ``install`` / ``guard`` are there for callers
who build a ModernBERT module themselves. ``tie_weights`` still runs:
``init_weights`` is left alone.
"""

import platform
import sys
import threading
from contextlib import contextmanager

_lock = threading.Lock()
_holds = 0
_original = None


def _noop_initialize_weights(self, *args, **kwargs):
    """Replacement for ``PreTrainedModel.initialize_weights``. Returns None."""
    return None


def _affected(system: str, version_info) -> bool:
    return system == "Windows" and version_info >= (3, 14)


def needs_guard() -> bool:
    """True on Windows with Python 3.14 or newer, where encoder init segfaults."""
    return _affected(platform.system(), sys.version_info)


def installed() -> bool:
    """True while at least one ``install`` / ``guard`` hold is active."""
    return _holds > 0


def install(*, force: bool = False) -> bool:
    """Take one hold on the no-op. Holds nest; pair each True with ``restore``.

    Returns False when this platform is unaffected and ``force`` is false, or
    when ``PreTrainedModel.initialize_weights`` is absent. A second call on an
    affected platform takes another hold and also returns True.
    """
    global _holds, _original
    with _lock:
        if not force and not needs_guard():
            return False
        if _holds == 0:
            import transformers.modeling_utils as modeling_utils

            current = getattr(modeling_utils.PreTrainedModel, "initialize_weights", None)
            if current is None:
                return False
            _original = current
            modeling_utils.PreTrainedModel.initialize_weights = _noop_initialize_weights
        _holds += 1
        return True


def restore() -> bool:
    """Drop one hold. The original method is put back when the last hold drops."""
    global _holds, _original
    with _lock:
        if _holds == 0:
            return False
        _holds -= 1
        if _holds == 0:
            import transformers.modeling_utils as modeling_utils

            modeling_utils.PreTrainedModel.initialize_weights = _original
            _original = None
        return True


@contextmanager
def guard(*, force: bool = False):
    """Hold the no-op for the duration of the block, then drop that hold.

    Yields True when this block took a hold. On an unaffected platform the
    yield is False and ``initialize_weights`` is left as it was.
    """
    held = install(force=force)
    try:
        yield held
    finally:
        if held:
            restore()
