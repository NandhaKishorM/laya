"""Optional TypeSafe adapter for Router.attach; independent of Laya's local engines."""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("refusing redirect with Monid credentials")


class MonidAgent:
    def __init__(self, *, model="jev-latest", timeout=60):
        self.api_key = os.environ.get("MONID_API_KEY")
        if not self.api_key:
            raise ValueError("MONID_API_KEY is required for MonidAgent")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.timeout = timeout

    def system_one(self, state, questions, *, lang=None):
        deadline = time.monotonic() + self.timeout
        opener = urllib.request.build_opener(_NoRedirect())
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json; charset=utf-8"}
        payload = {"provider": "typesafe", "endpoint": "/systemone",
                   "input": {"body": {"state": state, "questions": questions, "model": self.model}}}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        def request(url, data=None):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Monid run timed out")
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with opener.open(req, timeout=remaining) as response:
                    return response.status, json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Monid server error {e.code}: {body}") from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"Failed to connect to Monid: {e}") from e

        status, run = request("https://api.monid.ai/v1/run", data)
        if status == 202:
            run_id = urllib.parse.quote(run["runId"], safe="")
            while True:
                _, run = request(f"https://api.monid.ai/v1/runs/{run_id}")
                if run.get("status") in ("COMPLETED", "FAILED", "TIMED_OUT", "STOPPED", "BLOCKED"):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Monid run timed out")
                time.sleep(min(2, remaining))
        if run.get("status") not in (None, "COMPLETED"):
            raise RuntimeError(f"Monid run {run.get('status')}")
        provider_response = run.get("providerResponse") or {}
        if provider_response.get("httpStatus", 200) >= 400:
            # Match the remote Laya integrations' provider HTTP error handling.
            code = provider_response["httpStatus"]
            body = provider_response.get("error")
            raise RuntimeError(f"Laya server error {code}: {body}")
        output = run["output"]
        if not isinstance(output, dict):
            raise RuntimeError("Monid returned a non-object decision response")
        return output

    predict = system_one


def attach_monid(router, **options):
    if os.environ.get("MONID_API_KEY"):
        router.attach("monid", MonidAgent(**options))
    return router
