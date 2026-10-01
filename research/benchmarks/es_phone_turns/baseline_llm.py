"""The baseline the router has to beat: the language model the voice agent uses today,
asked the way the voice agent asks it.

The request mirrors a production voice agent: its system prompt, its two tools
(`transfer_call` with the destinations as an enum, `end_call`), streaming, reasoning off,
120 tokens, temperature 0.3. A turn with no tool call is `ninguno`: the agent keeps talking.

Standard library only. Needs an Ollama server with the model already pulled.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import metrics
import prompts

HERE = Path(__file__).parent
# The two answers that are not a destination: one is a tool of its own, the other is no
# tool at all.
HANG_UP, KEEP_TALKING = "colgar", "ninguno"
SYSTEM_PROMPT = (
    "Sos la recepcionista. UNA frase corta y hablada. Derivá al área que corresponda.")


def destinations(dataset):
    return {label: text for label, text in prompts.MENUS[dataset]["es"].items()
            if label not in (HANG_UP, KEEP_TALKING)}


def tools(dataset):
    menu = destinations(dataset)
    described = [f"{label} ({text})" for label, text in menu.items()]
    return [
        {"type": "function", "function": {
            "name": "transfer_call",
            "description": (
                "Pasa la llamada a otra persona o área. Úsala apenas quien llama pida hablar "
                "con alguien, o cuando la consulta no sea algo que puedas resolver. "
                "Destinos disponibles: " + "; ".join(described) + "."),
            "parameters": {"type": "object", "properties": {"destination": {
                "type": "string",
                "description": "La etiqueta exacta del destino, tal como aparece en la lista.",
                "enum": list(menu)}}, "required": ["destination"]}}},
        {"type": "function", "function": {
            "name": "end_call",
            "description": (
                "Termina la llamada. Úsala solo cuando ya te despediste y no queda nada "
                "pendiente."),
            "parameters": {"type": "object", "properties": {"reason": {
                "type": "string",
                "description": "En pocas palabras, por qué termina la llamada."}},
                "required": ["reason"]}}},
    ]


def ask(url, model, text, dataset):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": text}],
        "tools": tools(dataset), "stream": True, "think": False, "keep_alive": "30m",
        "options": {"num_predict": 120, "temperature": 0.3},
    }).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    started = time.perf_counter()
    first_ms = decided_ms = None
    calls, spoken, tokens_in, tokens_out = [], "", 0, 0
    with urllib.request.urlopen(request, timeout=120) as response:
        for line in response:
            if not line.strip():
                continue
            chunk = json.loads(line)
            now = (time.perf_counter() - started) * 1000
            message = chunk.get("message") or {}
            if first_ms is None and (message.get("content") or message.get("tool_calls")):
                first_ms = now
            if message.get("tool_calls") and decided_ms is None:
                decided_ms = now
            calls += message.get("tool_calls") or []
            spoken += message.get("content") or ""
            if chunk.get("done"):
                tokens_in = chunk.get("prompt_eval_count", 0)
                tokens_out = chunk.get("eval_count", 0)
    total_ms = (time.perf_counter() - started) * 1000

    predicted, argument = "ninguno", None
    for call in calls:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if function.get("name") == "transfer_call":
            argument = arguments.get("destination")
            wanted = str(argument or "").strip().lower()
            # An invented destination is not a transfer the agent can make.
            predicted = wanted if wanted in destinations(dataset) else "invalid"
            break
        if function.get("name") == "end_call":
            predicted = "colgar"
            break

    return {
        "predicted": predicted, "argument": argument, "spoken": spoken.strip(),
        "first_ms": first_ms, "decided_ms": decided_ms, "latency_ms": total_ms,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:11434/api/chat")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--dataset", default="cases", choices=sorted(prompts.MENUS))
    args = parser.parse_args()

    cases = [json.loads(line) for line in
             (HERE / "data" / f"{args.dataset}.jsonl").read_text(encoding="utf-8").splitlines()
             if line]

    # Warm: the first call loads the model.
    ask(args.url, args.model, cases[0]["text"], args.dataset)

    records = []
    for case in cases:
        answer = ask(args.url, args.model, case["text"], args.dataset)
        records.append({
            "id": case["id"], "text": case["text"], "gold": case["gold"], "tags": case["tags"],
            # A language model gives no probability for its decision: every call it makes,
            # it makes. 1.0 keeps the record comparable and makes the point visible.
            "confidence": 1.0, **answer})

    summary = metrics.summarize(records, prompts.labels(args.dataset))
    headline = [r for r in records if "borderline" not in r["tags"]]
    summary["first_ms"] = {
        "p50": metrics.percentile([r["first_ms"] for r in headline if r["first_ms"]], 50),
        "p95": metrics.percentile([r["first_ms"] for r in headline if r["first_ms"]], 95)}
    summary["tokens"] = {
        "in_mean": sum(r["tokens_in"] for r in headline) / len(headline),
        "out_mean": sum(r["tokens_out"] for r in headline) / len(headline)}
    summary["invalid_destinations"] = sum(r["predicted"] == "invalid" for r in headline)

    name = args.model.replace(":", "-").replace("/", "-")
    (HERE / "results" / f"llm-{name}__{args.dataset}.json").write_text(json.dumps(
        {"meta": {"model": args.model, "dataset": args.dataset,
                  "system_prompt": SYSTEM_PROMPT},
         "summary": summary, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{args.model} / {args.dataset}: accuracy {summary['accuracy']:.4f}  macro-F1 {summary['macro_f1']:.4f}  "
          f"wrong actions {summary['wrong_actions']}  invalid {summary['invalid_destinations']}  "
          f"first {summary['first_ms']['p50']:.0f} ms  total p50 {summary['latency_ms']['p50']:.0f} ms  "
          f"tokens in/out {summary['tokens']['in_mean']:.0f}/{summary['tokens']['out_mean']:.0f}")


if __name__ == "__main__":
    main()
