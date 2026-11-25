"""Offline checks for false qualification and the smoke command's exit status.

Run: python scripts/check_http_inference_smoke.py
The HTTP fixture tests the client; it is not model-inference evidence.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest

from http_inference_smoke import check_answers, check_checkpoint, SmokeFailure


PIN = "1" * 40
HEALTH = {"status": "ok", "loaded": ["english"], "revisions": {"english": PIN},
          "device_is_preference": False, "checkpoint_devices": {"english": "cpu"},
          "cpu_fallbacks": {"english": {"count": 0, "last_reason": None}}}
QUESTION = {"team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "money", "technical": "bugs"}}}
ANSWER = {"model": "laya-rl-agent", "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 0.9,
             "probabilities": {"billing": 0.9, "technical": 0.1}}}}


class Checks(unittest.TestCase):
    def test_live_empty_service_is_not_qualified(self):
        empty = {"status": "ok", "loaded": [], "device_is_preference": True}
        with self.assertRaises(SmokeFailure):
            check_checkpoint(empty, "english", PIN, "cpu")

    def test_revision_device_and_fallback_must_match(self):
        for field, value in [("revisions", {"english": "2" * 40}),
                             ("checkpoint_devices", {"english": "cpu"}),
                             ("cpu_fallbacks", {"english": {"count": 1, "last_reason": "OOM"}})]:
            health = copy.deepcopy(HEALTH)
            health[field] = value
            with self.subTest(field=field), self.assertRaises(SmokeFailure):
                check_checkpoint(health, "english", PIN, "cuda" if field == "checkpoint_devices" else "cpu")

    def test_wrong_keys_option_and_nonfinite_values_are_rejected(self):
        for change in [{"choice": "invented"}, {"probabilities": {"billing": float("nan"), "technical": 0.1}},
                       {"probabilities": {"billing": 0.9}}, {"confidence": True}]:
            answer = copy.deepcopy(ANSWER)
            answer["answers"]["team"].update(change)
            with self.subTest(change=change), self.assertRaises(SmokeFailure):
                check_answers(answer, QUESTION)

    def test_cli_requires_real_success_and_returns_nonzero_on_http_error(self):
        class Handler(BaseHTTPRequestHandler):
            fail = False

            def log_message(self, *args):
                pass

            def respond(self, status, body):
                self.send_response(status)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_GET(self):
                self.respond(200, HEALTH)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.headers.get("Authorization") != "Bearer fixture":
                    self.respond(401, {})
                    return
                assert body["model"] == "english"
                assert body["questions"] == QUESTION
                self.respond(500 if self.fail else 200, {} if self.fail else ANSWER)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                key = Path(directory) / "key"
                key.write_text("fixture")
                request = Path(directory) / "request.json"
                request.write_text(json.dumps({"state": "hello", "questions": QUESTION}))
                command = [sys.executable, str(Path(__file__).with_name("http_inference_smoke.py")),
                           "--url", "http://127.0.0.1:%s" % server.server_port, "--api-key-file", str(key),
                           "--request", str(request), "--source-revision", PIN,
                           "--checkpoint-revision", PIN, "--device", "cpu"]
                success = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(success.returncode, 0, success.stdout)
                self.assertTrue(json.loads(success.stdout)["inference_qualified"])
                Handler.fail = True
                failure = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(failure.returncode, 1)
                self.assertFalse(json.loads(failure.stdout)["inference_qualified"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
