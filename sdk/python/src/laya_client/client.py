"""Synchronous requests to the existing self-hosted HTTP API."""
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional, cast

from ._validation import batch_prediction, expect, json_value, prediction, probability, questions_value, timeout_value
from .errors import LayaAPIError, LayaConnectionError, LayaResponseError, LayaTimeoutError, LayaValidationError
from .types import BatchPrediction, Health, MinConfidence, Prediction, Questions, State


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _invalid_constant(value):
    raise ValueError("Non-finite JSON number")


class LayaClient:
    """Call a self-hosted Laya server without installing the inference engine.

    timeout is a blocking I/O timeout in seconds, not a total request deadline.
    Calls are synchronous. No inference request is retried automatically.
    """

    def __init__(self, *, base_url: str = "http://127.0.0.1:8000",
                 api_key: Optional[str] = None, timeout: float = 120.0):
        try:
            if (not isinstance(base_url, str) or any(ord(c) <= 32 or ord(c) == 127 for c in base_url)
                    or "?" in base_url or "#" in base_url):
                raise ValueError()
            base_url.encode("ascii")
            url = urllib.parse.urlsplit(base_url)
            if (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
                    or url.password is not None or url.query or url.fragment):
                raise ValueError()
            url.port  # Validate the port even though urllib would otherwise defer it.
        except (ValueError, UnicodeError):
            raise LayaValidationError(
                "base_url must be an ASCII HTTP(S) URL without credentials, whitespace, query or fragment"
            ) from None
        if api_key is not None:
            if not isinstance(api_key, str) or not api_key or any(ord(c) < 33 or ord(c) > 126 for c in api_key):
                raise LayaValidationError("api_key must be a nonempty token containing printable ASCII without spaces")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_value(timeout)

    def predict(self, state: State, questions: Questions, *, model: Optional[str] = None,
                task: Optional[str] = None, lang: Optional[str] = None, lang_guess: Optional[str] = None,
                max_len: Optional[int] = None, head_max_len: Optional[int] = None,
                min_confidence: Optional[MinConfidence] = None, timeout: Optional[float] = None) -> Prediction:
        """Predict choice, score and noul answers; omitted controls inherit server settings."""
        if not isinstance(state, (str, dict, list)):
            raise LayaValidationError("state must be text, a JSON object or an array")
        body = self._prediction_body(questions, model, task, lang, lang_guess, max_len, head_max_len, min_confidence)
        body["state"] = state
        result = self._request("/v1/systemone", body, timeout)
        prediction(result, questions)
        return cast(Prediction, result)

    def predict_batch(self, states: list[State], questions: Questions, *, model: Optional[str] = None,
                      task: Optional[str] = None, lang: Optional[str] = None, lang_guess: Optional[str] = None,
                      max_len: Optional[int] = None, head_max_len: Optional[int] = None,
                      min_confidence: Optional[MinConfidence] = None, batch_size: Optional[int] = None,
                      sort_by_length: Optional[bool] = None, timeout: Optional[float] = None) -> BatchPrediction:
        """Predict a nonempty list of states with shared questions, preserving input order.

        Sends one batch request. Server limits and unsupported endpoints raise LayaAPIError;
        there is no automatic splitting or fallback to single predictions.
        """
        if not isinstance(states, list) or not states:
            raise LayaValidationError("states must be a nonempty list")
        for i, state in enumerate(states):
            if not isinstance(state, (str, dict, list)):
                raise LayaValidationError("states[%d] must be text, a JSON object or an array" % i)
        body = self._prediction_body(questions, model, task, lang, lang_guess, max_len, head_max_len, min_confidence)
        body["states"] = states
        if batch_size is not None:
            if type(batch_size) is not int or batch_size <= 0:
                raise LayaValidationError("batch_size must be a positive integer")
            body["batch_size"] = batch_size
        if sort_by_length is not None:
            if type(sort_by_length) is not bool:
                raise LayaValidationError("sort_by_length must be a boolean")
            body["sort_by_length"] = sort_by_length
        result = self._request("/v1/systemone/batch", body, timeout)
        batch_prediction(result, questions, len(states))
        return cast(BatchPrediction, result)

    @staticmethod
    def _prediction_body(questions, model, task, lang, lang_guess, max_len, head_max_len, min_confidence):
        questions_value(questions)
        body: dict[str, Any] = {"questions": questions}
        for key, value in (("model", model), ("task", task), ("lang", lang), ("lang_guess", lang_guess)):
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise LayaValidationError(key + " must be a nonempty string")
                body[key] = value
        for key, value in (("max_len", max_len), ("head_max_len", head_max_len)):
            if value is not None:
                if type(value) is not int or value <= 0:
                    raise LayaValidationError(key + " must be a positive integer")
                body[key] = value
        if min_confidence is not None:
            valid_map = (isinstance(min_confidence, dict) and bool(min_confidence)
                         and all(isinstance(k, str) for k in min_confidence)
                         and all(probability(v) for v in min_confidence.values()))
            if not probability(min_confidence) and not valid_map:
                raise LayaValidationError("min_confidence must be a threshold in [0, 1] or a nonempty threshold map")
            body["min_confidence"] = min_confidence
        return body

    def health(self, *, timeout: Optional[float] = None) -> Health:
        """Read liveness; a minimal response does not prove authentication or model readiness."""
        result = self._request("/health", None, timeout)
        expect(isinstance(result, dict) and result.get("status") == "ok", "health")
        return cast(Health, result)

    def _request(self, path: str, body: Optional[dict], timeout: Optional[float]) -> Any:
        seconds = self._timeout if timeout is None else timeout_value(timeout)
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            try:
                json_value(body)
                data = json.dumps(body, allow_nan=False).encode("utf-8")
            except (ValueError, TypeError, RecursionError) as exc:
                raise LayaValidationError("request must contain finite, acyclic JSON values") from exc
            headers["Content-Type"] = "application/json"
        if self._api_key is not None:
            headers["Authorization"] = "Bearer " + self._api_key
        req = urllib.request.Request(self._base_url + path, data=data, headers=headers,
                                     method="GET" if body is None else "POST")
        try:
            # A private opener avoids global changes; redirects must not replay a POST or forward its token.
            with urllib.request.build_opener(_NoRedirect()).open(req, timeout=seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            with exc:
                try:
                    raw = exc.read(65536)
                    try:
                        payload = json.loads(raw, parse_constant=_invalid_constant)
                        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
                    except (ValueError, UnicodeError, RecursionError):
                        detail = raw.decode("utf-8", errors="replace")
                except (OSError, http.client.HTTPException):
                    detail = None
            raise LayaAPIError(exc.code, detail) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise LayaTimeoutError("Laya request timed out; server inference may still finish") from exc
            raise LayaConnectionError("Could not reach the Laya server") from exc
        except TimeoutError as exc:
            raise LayaTimeoutError("Laya request timed out; server inference may still finish") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise LayaConnectionError("Laya response could not be read") from exc
        try:
            return json.loads(raw, parse_constant=_invalid_constant)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise LayaResponseError("Server returned invalid JSON") from exc
