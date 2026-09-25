"""Opt-in workaround for the Windows + Python 3.14 segfault (issue #123).

On Windows 11 with Python 3.14 and torch 2.14, building a ModernBERT model via
``transformers`` crashes the interpreter inside
``PreTrainedModel.initialize_weights`` -- so ``Agent(...)`` cannot finish
loading. Basic torch ops, tokenizers and safetensors loading are unaffected;
only the weight-initialization pass dies.

Usage (before importing/loading laya)::

    import laya.win_patch  # noqa: F401  (applies itself on matching platforms)
    from laya import Router
    ...

This is safe for laya because ``Agent.__init__`` immediately calls
``load_state_dict(strict=True)`` with the checkpoint, so the skipped random
initial values are never used. The patch is a no-op on all other platforms.
"""

import platform
import sys

_APPLIES = sys.version_info >= (3, 14) and platform.system() == "Windows"


def apply() -> bool:
    """Replace ``PreTrainedModel.initialize_weights`` with a no-op.

    Returns True if the patch was applied, False if this platform is
    unaffected (nothing is changed in that case).
    """
    if not _APPLIES:
        return False
    try:
        import transformers.modeling_utils as _mu
    except ImportError:
        return False
    _mu.PreTrainedModel.initialize_weights = lambda self: None
    return True


applied = apply()
