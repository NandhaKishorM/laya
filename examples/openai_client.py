#!/usr/bin/env python3
"""Call the OpenAI-compatible Laya server using only the standard library.

Start the server first, then:

    python examples/openai_client.py                       # http://127.0.0.1:8000
    python examples/openai_client.py http://host:8000 KEY

The same request through the OpenAI SDK (``pip install openai``) is shown in a comment below.
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
API_KEY = sys.argv[2] if len(sys.argv) > 2 else ""


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    if API_KEY:
        headers["authorization"] = "Bearer " + API_KEY
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get(path):
    headers = {"authorization": "Bearer " + API_KEY} if API_KEY else {}
    request = urllib.request.Request(BASE + path, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


TOOLS = [{
    "type": "function",
    "function": {
        "name": "route_ticket",
        "description": "Route a support ticket to the right team",
        "parameters": {
            "type": "object",
            "properties": {
                "department": {"type": "string", "enum": ["billing", "support", "sales"]},
                "urgent": {"type": "boolean"},
            },
        },
    },
}]

if __name__ == "__main__":
    print("models:", [model["id"] for model in get("/v1/models")["data"]])

    response = post("/v1/chat/completions", {
        "model": "laya",
        "messages": [{"role": "user", "content": "The site is down and this is urgent"}],
        "tools": TOOLS,
    })
    call = response["choices"][0]["message"]["tool_calls"][0]["function"]
    print("tool:", call["name"])
    print("arguments:", json.loads(call["arguments"]))
    print("usage:", response["usage"])

    # Through the OpenAI SDK:
    #   from openai import OpenAI
    #   client = OpenAI(base_url=BASE + "/v1", api_key=API_KEY or "not-needed")
    #   completion = client.chat.completions.create(
    #       model="laya", messages=[...], tools=[{"type": "function", "function": TOOLS[0]["function"]}])
    #   print(completion.choices[0].message.tool_calls)
