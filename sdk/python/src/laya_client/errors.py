"""Failures callers can distinguish without importing the inference engine."""
from typing import Any


class LayaError(Exception):
    """Base class for client errors."""


class LayaValidationError(LayaError, ValueError):
    """An invalid client setting or request, rejected before sending it."""


class LayaAPIError(LayaError):
    """A non-2xx HTTP response; detail preserves the server's JSON value."""

    def __init__(self, status_code: int, detail: Any = None):
        super().__init__("Laya returned HTTP %d" % status_code)
        self.status_code = status_code
        self.detail = detail


class LayaConnectionError(LayaError):
    """The request or response could not be transported."""


class LayaTimeoutError(LayaConnectionError):
    """A blocking I/O operation timed out; server inference may still finish."""


class LayaResponseError(LayaError):
    """A successful HTTP response did not satisfy the prediction contract."""
