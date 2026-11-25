"""Qualify one authenticated HTTP inference against an already-running cached server.

Run with --help for required provenance and device checks. This command neither starts
the service nor controls its downloads: configure HF_HUB_OFFLINE=1 on the server first.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a bearer credential to a redirected destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class SmokeFailure(ValueError):
    """A safe, fixed diagnostic that contains no credential or server error text."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def check_answers(result: Dict[str, Any], questions: Dict[str, Any]) -> None:
    """Check the example's answer types and bounds without asserting model accuracy."""
    require(isinstance(result, dict) and isinstance(result.get("model"), str), "missing response model")
    answers = result.get("answers")
    require(isinstance(answers, dict) and set(answers) == set(questions), "answer keys differ from request")
    for name, question in questions.items():
        answer = answers[name]
        kind = question["type"]
        require(isinstance(answer, dict) and answer.get("type") == kind, "answer type differs from request")
        if kind == "noul":
            require(probability(answer.get("noul")), "invalid noul probability")
            continue
        criteria = question["criteria"]
        keys = set(criteria) if kind == "choice" else {str(i) for i in range(len(criteria))}
        probs = answer.get("probabilities")
        require(isinstance(probs, dict) and set(probs) == keys, "probability keys differ from criteria")
        require(all(probability(p) for p in probs.values()), "invalid probabilities")
        require(abs(sum(probs.values()) - 1) <= 0.001 * len(keys), "probabilities do not sum to one")
        require(probability(answer.get("confidence")), "invalid confidence")
        if kind == "choice":
            require(answer.get("choice") in keys, "choice is not a requested option")
        elif kind == "score":
            score = answer.get("score")
            require(isinstance(score, (int, float)) and not isinstance(score, bool)
                    and math.isfinite(score) and 0 <= score <= len(criteria) - 1, "invalid score")
        else:
            raise SmokeFailure("smoke supports choice, score and noul questions")


def check_checkpoint(health: Dict[str, Any], checkpoint: str, revision: str, device: str) -> None:
    """A live process or configured GPU preference is not inference qualification."""
    require(health.get("status") == "ok", "HTTP health is not ok")
    require(checkpoint in health.get("loaded", []), "requested checkpoint is not resident")
    require(health.get("device_is_preference") is False, "device is only a configured preference")
    revisions = health.get("revisions")
    devices = health.get("checkpoint_devices")
    fallbacks = health.get("cpu_fallbacks")
    require(all(isinstance(value, dict) for value in (revisions, devices, fallbacks)), "missing checkpoint evidence")
    require(revisions.get(checkpoint) == revision, "checkpoint revision differs from pin")
    actual = devices.get(checkpoint)
    require(actual == device or (device == "cuda" and isinstance(actual, str) and actual.startswith("cuda:")),
            "checkpoint device differs from requested device")
    fallback = fallbacks.get(checkpoint)
    require(isinstance(fallback, dict) and fallback.get("count") == 0
            and fallback.get("last_reason") is None, "checkpoint has CPU fallback history or missing evidence")


def qualify(args: argparse.Namespace, report: Dict[str, Any]) -> None:
    parsed = urllib.parse.urlsplit(args.url)
    require(not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            "URL must not contain credentials, a query or fragment")
    loopback = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    require(parsed.scheme == "https" or (parsed.scheme == "http" and loopback), "HTTP requires a loopback host")
    require(bool(parsed.hostname), "URL needs a host")
    for value in (args.source_revision, args.checkpoint_revision):
        require(re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None, "revisions must be full commit SHAs")
    require(math.isfinite(args.timeout) and args.timeout > 0, "timeout must be finite and positive")
    key = Path(args.api_key_file).read_text(encoding="utf-8-sig").strip()
    require(bool(key) and not any(c in key for c in "\r\n\0"), "invalid API key file")
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    questions = request.get("questions")
    require("state" in request and isinstance(questions, dict) and bool(questions), "request needs state and questions")
    request["model"] = args.checkpoint
    # Ignore environment proxies for loopback; redirects are refused for every destination.
    handlers = [NoRedirect()]
    if loopback:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)

    def call(path: str, body: Any = None, authenticated: bool = True):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + key
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(args.url.rstrip("/") + path, data=data, headers=headers)
        try:
            with opener.open(req, timeout=args.timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            # Error bodies may contain operator details; retain the status, not the body.
            return error.code, None

    status, _ = call("/v1/systemone", request, authenticated=False)
    report["unauthenticated_status"] = status
    require(status == 401, "server did not reject an unauthenticated inference request")
    status, before = call("/health")
    require(status == 200 and isinstance(before, dict) and "loaded" in before, "authenticated health details unavailable")
    report["health_before"] = before
    started = time.perf_counter()
    status, result = call("/v1/systemone", request)
    report["inference_status"] = status
    report["inference_seconds"] = round(time.perf_counter() - started, 4)
    require(status == 200, "inference did not return HTTP 200")
    check_answers(result, questions)
    if "routing" in result:
        require(isinstance(result["routing"], dict) and result["routing"].get("model") == args.checkpoint,
                "response routed to a different checkpoint")
    report["response"] = result
    status, after = call("/health")
    require(status == 200 and isinstance(after, dict), "post-inference health details unavailable")
    report["health_after"] = after
    check_checkpoint(after, args.checkpoint, args.checkpoint_revision, args.device)
    report["inference_qualified"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--request", default=str(Path(__file__).resolve().parents[1] / "examples/docker/request.json"))
    parser.add_argument("--source-revision", required=True, help="operator-verified server source SHA; HTTP does not expose it")
    parser.add_argument("--checkpoint", default="english", help="explicit registered checkpoint name")
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--device", required=True, help="expected actual checkpoint device, e.g. cpu or cuda:0")
    parser.add_argument("--timeout", type=float, default=120, help="seconds per HTTP request; no inference retries")
    parser.add_argument("--output", type=Path, help="JSON report destination; default stdout")
    args = parser.parse_args()
    report = {"inference_qualified": False, "declared_server_source_revision": args.source_revision,
              "expected_checkpoint_revision": args.checkpoint_revision, "checkpoint": args.checkpoint,
              "expected_device": args.device, "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        qualify(args, report)
    except (ValueError, KeyError, TypeError, OSError, urllib.error.URLError) as error:
        # Never print arbitrary connection/file errors: a supplied URL/path may hold a secret.
        report["error"] = str(error) if isinstance(error, SmokeFailure) else type(error).__name__
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if report["inference_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
