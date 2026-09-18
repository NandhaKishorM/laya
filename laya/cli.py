"""Command-line interface for Laya."""

import json
import os
import sys
import click

from .client import decide as api_decide
from .client import judge as api_judge
from .client import rate as api_rate
from .client import system_one as api_system_one


@click.group()
@click.version_option(version="0.1.6", prog_name="laya")
def main():
    """Laya - Fast, non-autoregressive System 1 decision engine."""
    pass


@main.command()
@click.option("--host", default="0.0.0.0", help="Host interface to bind on.")
@click.option("--port", default=8000, type=int, help="Port to listen on.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
@click.option("--model", default="convaiinnovations/laya", help="Hugging Face model checkpoint.")
def serve(host: str, port: int, reload: bool, model: str):
    """Start the Laya TypeSafe-compatible HTTP server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn is required to run the server. Run 'pip install laya[server]'.", err=True)
        sys.exit(1)

    os.environ["LAYA_MODEL_ID"] = model
    click.echo(f"Starting Laya System One Decision Server [{model}] on http://{host}:{port}")
    uvicorn.run("laya.server:app", host=host, port=port, reload=reload)


@main.command()
@click.argument("text")
@click.option(
    "-c",
    "--choices",
    required=True,
    help="Comma-separated choices (e.g. 'billing,bug_report,feature_request').",
)
@click.option(
    "-i",
    "--instructions",
    default="Which option best describes the input?",
    help="Instructions for classification.",
)
def decide(text: str, choices: str, instructions: str):
    """Classify input text among discrete choices (Choice)."""
    opts = [c.strip() for c in choices.split(",") if c.strip()]
    if not opts:
        click.echo("Error: At least one choice must be provided.", err=True)
        sys.exit(1)

    ans = api_decide(state=text, choices=opts, instructions=instructions)
    click.echo(json.dumps(ans, indent=2))


@main.command()
@click.argument("text")
@click.option(
    "-i",
    "--instructions",
    required=True,
    help="Boolean judgment question (e.g. 'Is the server down?').",
)
@click.option(
    "--pos",
    default="",
    help="Explicit criteria description for True condition.",
)
@click.option(
    "--neg",
    default="",
    help="Explicit criteria description for False condition.",
)
def judge(text: str, instructions: str, pos: str, neg: str):
    """Evaluate a yes/no judgment (Noul) and return the probability."""
    crit = {}
    if pos:
        crit["true"] = pos
    if neg:
        crit["false"] = neg

    prob = api_judge(state=text, instructions=instructions, criteria=crit or None)
    click.echo(
        json.dumps(
            {
                "type": "noul",
                "instructions": instructions,
                "noul": prob,
            },
            indent=2,
        )
    )


@main.command()
@click.argument("text")
@click.option(
    "-l",
    "--levels",
    required=True,
    help="Comma-separated descriptions of ordered levels from 0 to N-1.",
)
@click.option(
    "-i",
    "--instructions",
    default="Rate where the state falls on this scale:",
    help="Instructions for rating.",
)
def rate(text: str, levels: str, instructions: str):
    """Rate text on an ordered multi-level scale (Score)."""
    lvl_list = [lvl.strip() for lvl in levels.split(",") if lvl.strip()]
    if len(lvl_list) < 2:
        click.echo("Error: At least two levels must be provided.", err=True)
        sys.exit(1)

    ans = api_rate(state=text, criteria=lvl_list, instructions=instructions)
    click.echo(json.dumps(ans, indent=2))


@main.command()
@click.argument("request_file", type=click.Path(exists=True))
def eval(request_file: str):
    """Evaluate a JSON request file containing state and questions."""
    with open(request_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    state = data.get("state")
    questions = data.get("questions")
    model = data.get("model", "laya-latest")

    if state is None or questions is None:
        click.echo("Error: JSON must contain 'state' and 'questions' fields.", err=True)
        sys.exit(1)

    resp = api_system_one(state=state, questions=questions, model=model)
    click.echo(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()
