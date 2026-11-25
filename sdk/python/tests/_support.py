import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "technical": "help"}},
    "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "medium", "high"]},
    "refund": {"type": "noul", "instructions": "Refund?"},
}
PREDICTION = {
    "model": "custom-checkpoint",
    "answers": {
        "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "technical": 0.2},
                 "confidence": 0.278},
        "urgency": {"type": "score", "score": 1.7, "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                    "confidence": 0.418, "legend": {"0": "low", "1": "medium", "2": "high"}},
        "refund": {"type": "noul", "noul": 0.7},
    },
    "usage": {"input_tokens": 123, "output_tokens": 0},
}


class Server:
    def __init__(self, payload=None, *, raw=None, status=200, headers=None, slow=False, truncated=False):
        self.payload = copy.deepcopy(PREDICTION if payload is None else payload)
        self.raw, self.status = raw, status
        self.headers = headers or {}
        self.slow, self.truncated = slow, truncated
        self.requests = []
        self.release = threading.Event()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def handle_request(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fixture.requests.append((self.command, self.path, dict(self.headers), body))
                if fixture.slow:
                    fixture.release.wait(5)
                raw = fixture.raw
                if raw is None:
                    raw = json.dumps(fixture.payload).encode("utf-8")
                self.send_response(fixture.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw) + (10 if fixture.truncated else 0)))
                for key, value in fixture.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                self.close_connection = True

            do_GET = do_POST = handle_request

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_port
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.release.set()
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()
