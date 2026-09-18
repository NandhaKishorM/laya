"""Typed exception hierarchy for laya.

Every error raised by laya derives from :class:`LayaError`, so callers can catch the
whole library with one except clause. Errors that used to be raised as bare
``ValueError`` / ``FileNotFoundError`` / ``RuntimeError`` keep those as secondary bases,
so existing ``except ValueError:`` code continues to work.
"""


class LayaError(Exception):
    """Base class for every error raised by laya."""


class ConfigurationError(LayaError, ValueError):
    """The model config is missing keys or holds an unusable value."""


class ModelNotFoundError(LayaError, FileNotFoundError):
    """The model directory, config or weights file could not be found."""


class IncompatibleModelError(LayaError, ValueError):
    """Checkpoint weights do not match the configured architecture."""


class QuestionError(LayaError, ValueError):
    """A question definition is malformed."""


class UnknownQuestionTypeError(QuestionError):
    """Question ``type`` is not one of choice / score / noul."""


class InvalidCriteriaError(QuestionError):
    """Question ``criteria`` is missing, empty or of the wrong shape for its type."""


class OptionsTooLongError(QuestionError):
    """Rendered options do not fit within ``head_max_len``."""


class InferenceError(LayaError, RuntimeError):
    """The forward pass failed."""


class DeviceError(InferenceError):
    """The model could not be placed or run on the requested device."""


__all__ = [
    "LayaError",
    "ConfigurationError",
    "ModelNotFoundError",
    "IncompatibleModelError",
    "QuestionError",
    "UnknownQuestionTypeError",
    "InvalidCriteriaError",
    "OptionsTooLongError",
    "InferenceError",
    "DeviceError",
]
