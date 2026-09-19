"""
Laya CLI — command-line interface for the Laya decision engine.

Provides shell-level access to all three decision primitives and a
built-in server launcher. Mirrors the TypeSafe CLI surface so teams
already using ``typesafe`` CLI can switch without retraining.

Commands
--------
laya serve      Launch the REST API server (wraps uvicorn)
laya decide     Single choice classification from the shell
laya judge      Noul (boolean probability) from the shell
laya rate       Ordinal score from the shell
laya health     Ping a running Laya server

Install
-------
    pip install "laya[server]"

Usage
-----
    laya serve --port 8000 --model convaiinnovations/laya

    laya decide "Server disk space at 99%" \\
        -c storage_alert -c network_alert -c auth_alert

    laya judge "Payment declined on checkout" \\
        -i "Is this a payment failure?"

    laya rate "Server is dead across all nodes" \\
        -l cosmetic -l minor -l critical \\
        -i "Rate the severity:"

    laya health --url http://localhost:8000
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import click


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bold(text: str) -> str:
    return click.style(text, bold=True)


def _color(text: str, fg: str) -> str:
    return click.style(text, fg=fg)


def _load_agent(model: str, device: Optional[str]):
    """Import and load agent — deferred so CLI starts instantly."""
    try:
        import laya
        return laya.load(model, device=device)
    except ImportError as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


def _print_answer(qid: str, answer: dict) -> None:
    """Pretty-print a single answer dict."""
    t = answer.get("type", "unknown")
    conf = answer.get("confidence", None)
    act = answer.get("action", {}).get("act_probability", None)

    click.echo()
    click.echo(_bold(f"  {qid}"))

    if t == "choice":
        click.echo(f"    choice     : {_color(answer['choice'], 'cyan')}")
        click.echo(f"    confidence : {answer['confidence']:.4f}")
        click.echo("    probs      :")
        for label, p in answer["probabilities"].items():
            bar = "█" * int(p * 20)
            click.echo(f"      {label:<20} {p:.4f}  {bar}")

    elif t == "score":
        score_val = answer["score"]
        legend = answer.get("legend", {})
        levels = len(legend)
        lo = legend.get("0", "0")
        hi = legend.get(str(levels - 1), str(levels - 1))
        click.echo(f"    score      : {_color(f'{score_val:.4f}', 'yellow')}  ({lo} → {hi})")
        click.echo(f"    confidence : {answer['confidence']:.4f}")
        click.echo("    probs      :")
        for idx, p in answer["probabilities"].items():
            label = legend.get(idx, idx)
            bar = "█" * int(p * 20)
            click.echo(f"      [{idx}] {label:<18} {p:.4f}  {bar}")

    elif t == "noul":
        noul_val = answer["noul"]
        icon = _color("TRUE ", "green") if noul_val >= 0.5 else _color("FALSE", "red")
        click.echo(f"    noul       : {noul_val:.4f}  → {icon}")
        click.echo(f"    confidence : {answer['confidence']:.4f}")

    if act is not None:
        act_str = _color(f"{act:.4f}", "green" if act >= 0.7 else "yellow")
        click.echo(f"    act_prob   : {act_str}")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="laya")
def main():
    """
    \b
    Laya — Fast, non-autoregressive System 1 decision engine.
    Open-source drop-in for TypeSafe Jev.

    \b
    Quickstart:
      laya serve --port 8000
      laya decide "ticket text" -c billing -c technical -c sales
      laya judge  "ticket text" -i "Is this urgent?"
      laya rate   "ticket text" -l low -l medium -l high -i "Urgency:"
    """


# ---------------------------------------------------------------------------
# laya serve
# ---------------------------------------------------------------------------

@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port.")
@click.option("--model", default=os.environ.get("LAYA_MODEL", "convaiinnovations/laya"),
              show_default=True, help="HuggingFace model ID or local path.")
@click.option("--device", default=None, help="cuda | cpu | mps (default: auto).")
@click.option("--workers", default=4, show_default=True, type=int, help="Thread-pool workers.")
@click.option("--reload", is_flag=True, default=False, help="Hot-reload for development.")
@click.option("--log-level", default="info", show_default=True,
              type=click.Choice(["debug", "info", "warning", "error"]),
              help="Uvicorn log level.")
def serve(host, port, model, device, workers, reload, log_level):
    """
    Launch the Laya REST API server.

    \b
    Exposes:
      POST /v1/decide          Single-state inference (OpenAI-compatible)
      POST /v1/systemone       Drop-in for TypeSafe Jev SDK
      POST /v1/decide/batch    Batch inference (up to 256 states)
      GET  /v1/models          OpenRouter-compatible model list
      GET  /health             Liveness probe

    \b
    Examples:
      laya serve --port 8000
      laya serve --model convaiinnovations/laya --device cuda
      LAYA_MODEL=my/model laya serve
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(click.style(
            'uvicorn not found. Install it with: pip install "laya[server]"', fg="red"
        ), err=True)
        sys.exit(1)

    # Propagate CLI args to environment so the server lifespan picks them up.
    os.environ["LAYA_MODEL"] = model
    if device:
        os.environ["LAYA_DEVICE"] = device
    os.environ["LAYA_WORKERS"] = str(workers)
    os.environ["LAYA_LOG_LEVEL"] = log_level

    click.echo(_bold(f"\n  🔷 Laya server starting on http://{host}:{port}"))
    click.echo(f"     model  : {model}")
    click.echo(f"     device : {device or 'auto'}")
    click.echo(f"     docs   : http://{host}:{port}/docs\n")

    uvicorn.run(
        "laya.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# laya decide  (choice)
# ---------------------------------------------------------------------------

@main.command()
@click.argument("state")
@click.option("-c", "--choice", "choices", multiple=True, required=True,
              help="A candidate label. Repeat for each option: -c billing -c technical")
@click.option("-d", "--description", "descs", multiple=True,
              help="Optional description per label (same order as -c).")
@click.option("-i", "--instructions", default="Classify the input.",
              show_default=True, help="Natural-language question to ask.")
@click.option("--model", default=os.environ.get("LAYA_MODEL", "convaiinnovations/laya"),
              help="Model ID or path.")
@click.option("--device", default=None, help="cuda | cpu | mps.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
def decide(state, choices, descs, instructions, model, device, as_json):
    """
    Classify STATE into one of the provided choice labels.

    \b
    Examples:
      laya decide "Disk at 99%" -c storage_alert -c network_alert -c auth_alert
      laya decide "Refund me now" -c billing -c technical -c sales --json
    """
    agent = _load_agent(model, device)
    criteria = {c: (descs[i] if i < len(descs) else None) for i, c in enumerate(choices)}
    result = agent.system_one(state, {
        "answer": {"type": "choice", "instructions": instructions, "criteria": criteria}
    })
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(_bold("\n  Laya · decide"))
        click.echo(f"  state : {state[:80]}{'…' if len(state) > 80 else ''}")
        _print_answer("answer", result["answers"]["answer"])
        click.echo()


# ---------------------------------------------------------------------------
# laya judge  (noul)
# ---------------------------------------------------------------------------

@main.command()
@click.argument("state")
@click.option("-i", "--instructions", required=True,
              help="Yes/no statement to evaluate. E.g. 'Is this a payment failure?'")
@click.option("--model", default=os.environ.get("LAYA_MODEL", "convaiinnovations/laya"),
              help="Model ID or path.")
@click.option("--device", default=None, help="cuda | cpu | mps.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
def judge(state, instructions, model, device, as_json):
    """
    Evaluate a boolean/noul question about STATE.

    Returns calibrated P(true) ∈ [0, 1].

    \b
    Examples:
      laya judge "Payment declined on checkout" -i "Is this a payment failure?"
      laya judge "Cancel my subscription" -i "Is the customer churning?"
    """
    agent = _load_agent(model, device)
    result = agent.system_one(state, {
        "answer": {"type": "noul", "instructions": instructions}
    })
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(_bold("\n  Laya · judge"))
        click.echo(f"  state : {state[:80]}{'…' if len(state) > 80 else ''}")
        _print_answer("answer", result["answers"]["answer"])
        click.echo()


# ---------------------------------------------------------------------------
# laya rate  (score)
# ---------------------------------------------------------------------------

@main.command()
@click.argument("state")
@click.option("-l", "--level", "levels", multiple=True, required=True,
              help="Ordered rubric level (lowest first). Repeat: -l low -l medium -l high")
@click.option("-i", "--instructions", default="Rate the following input.",
              show_default=True, help="Question to ask about the state.")
@click.option("--model", default=os.environ.get("LAYA_MODEL", "convaiinnovations/laya"),
              help="Model ID or path.")
@click.option("--device", default=None, help="cuda | cpu | mps.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
def rate(state, levels, instructions, model, device, as_json):
    """
    Score STATE on an ordered rubric (lowest → highest).

    \b
    Examples:
      laya rate "Server is down" -l cosmetic -l minor -l critical -i "Rate severity:"
      laya rate "Refund me NOW" -l calm -l annoyed -l furious -i "Customer mood?"
    """
    agent = _load_agent(model, device)
    result = agent.system_one(state, {
        "answer": {"type": "score", "instructions": instructions, "criteria": list(levels)}
    })
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(_bold("\n  Laya · rate"))
        click.echo(f"  state : {state[:80]}{'…' if len(state) > 80 else ''}")
        _print_answer("answer", result["answers"]["answer"])
        click.echo()


# ---------------------------------------------------------------------------
# laya health
# ---------------------------------------------------------------------------

@main.command()
@click.option("--url", default="http://localhost:8000", show_default=True,
              help="Base URL of a running Laya server.")
def health(url):
    """
    Ping a running Laya server and display its status.

    \b
    Example:
      laya health --url http://localhost:8000
    """
    try:
        import httpx
    except ImportError:
        click.echo(click.style('httpx not found. Install with: pip install httpx', fg="red"), err=True)
        sys.exit(1)

    target = url.rstrip("/") + "/health"
    try:
        resp = httpx.get(target, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        click.echo(_bold("\n  Laya server — healthy ✅"))
        for k, v in data.items():
            click.echo(f"  {k:<10}: {v}")
        click.echo()
    except httpx.ConnectError:
        click.echo(click.style(f"\n  ❌  Cannot connect to {target}", fg="red"), err=True)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        click.echo(click.style(f"\n  ❌  Server returned {e.response.status_code}", fg="red"), err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
