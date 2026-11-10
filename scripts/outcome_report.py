"""Static outcome report: the local dashboard as a file, not a service.

Reads a decision log (`OutcomeLogHook` SQLite) and writes one self-contained HTML
file: decision volume, label coverage by source, confidence and abstention with the
review queue, and -- when given a fitted payload -- its temperatures and thresholds.
No server, no JavaScript, no new dependencies: the standard library plus inline SVG,
so the report opens from `file://`, over SSH, or from a nightly job with nothing to
supervise. Regenerate rather than refresh; the data moves on calibration cadence,
not seconds.

Run: python scripts/outcome_report.py --db decisions.db --out report.html
     python scripts/outcome_report.py --db decisions.db --out report.html \\
         --payload calibration.json
"""
import argparse
import html
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # repository root; the hook lives in <root>/laya

from laya.outcome_labels import export_review_queue  # noqa: E402
from laya.outcome_log import OutcomeLogHook  # noqa: E402


def _bar(label, value, maximum, width=200):
    """One labelled horizontal bar as inline SVG, values escaped, no JavaScript."""
    frac = (value / maximum) if maximum > 0 else 0.0
    text = "%s: %s" % (html.escape(str(label)), html.escape(str(value)))
    return (
        '<div class="row"><span class="lab">%s</span>'
        '<svg width="%d" height="14" role="img" aria-label="%s">'
        '<rect width="%d" height="14"/></svg> '
        '<span class="val">%s</span></div>'
        % (html.escape(str(label)), width, text, max(0, int(width * frac)),
           html.escape(str(value)))
    )


def _table(headers, rows):
    head = "".join("<th>%s</th>" % html.escape(h) for h in headers)
    body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % html.escape(str(c)) for c in r)
                   for r in rows)
    return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (head, body)


def _day(ts):
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError, OSError):
        return "unknown"


def build_report(db_path, payload_path=None):
    """Collect every panel's data. Pure reads; raises on an unreadable database."""
    hook = OutcomeLogHook(db_path)
    try:
        decisions = hook.decisions()
        outcomes = hook.outcomes()
        review = export_review_queue(hook)
    finally:
        hook.close()
    by_qid: dict = {}
    by_day: dict = {}
    by_model: dict = {}
    confidences = []
    abstained = 0
    for row in decisions:
        if row.get("qid") is None:
            continue
        by_qid[row["qid"]] = by_qid.get(row["qid"], 0) + 1
        by_day[_day(row.get("ts"))] = by_day.get(_day(row.get("ts")), 0) + 1
        by_model[row.get("model") or "unknown"] = by_model.get(row.get("model")
                                                               or "unknown", 0) + 1
        if isinstance(row.get("confidence"), (int, float)) \
                and not isinstance(row.get("confidence"), bool):
            confidences.append(row["confidence"])
        if row.get("abstention") not in (None, "passed"):
            abstained += 1
    labels = {}
    for row in outcomes:
        entry = labels.setdefault(row["qid"], {"n": 0, "sources": set()})
        entry["n"] += 1
        entry["sources"].add(row["source"])
    coverage = []
    for qid in sorted(by_qid):
        lab = labels.get(qid, {"n": 0, "sources": set()})
        coverage.append({"qid": qid, "answers": by_qid[qid], "labeled": lab["n"],
                         "sources": ", ".join(sorted(lab["sources"]))})
    payload = None
    if payload_path is not None:
        with open(payload_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    return {"n_decisions": len([r for r in decisions if r.get("qid") is not None]),
            "n_outcomes": len(outcomes),
            "by_qid": sorted(by_qid.items()),
            "by_day": sorted(by_day.items()),
            "by_model": sorted(by_model.items()),
            "mean_confidence": (sum(confidences) / len(confidences) if confidences else None),
            "abstained": abstained,
            "coverage": coverage,
            "review": review,
            "payload": payload,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def render_html(report, db_path):
    """Assemble the page. Every number derives from `build_report`; nothing is fetched."""
    parts = ["<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
             "<title>Outcome report</title><style>",
             "body{font-family:sans-serif;max-width:60em;margin:2em auto;padding:0 1em}",
             "table{border-collapse:collapse;margin:1em 0}th,td{border:1px solid #ccc;"
             "padding:.25em .6em;text-align:left}",
             ".row{display:flex;align-items:center;gap:.5em;margin:.15em 0}",
             ".lab{display:inline-block;width:14em;overflow:hidden;text-overflow:ellipsis;"
             "white-space:nowrap}svg rect{fill:#4472c4}",
             "</style></head><body>",
             "<h1>Outcome report</h1>",
             "<p>Source: %s. Generated: %s.</p>"
             % (html.escape(str(db_path)), html.escape(report["generated_at"])),
             "<h2>Volume</h2><p>%d logged answers, %d labels.</p>"
             % (report["n_decisions"], report["n_outcomes"])]
    peak = max([n for _, n in report["by_qid"]] or [0])
    parts.append("<h3>By question</h3>")
    parts.extend(_bar(qid, n, peak) for qid, n in report["by_qid"])
    if report["by_day"]:
        parts.append("<h3>By day</h3>")
        peak_day = max(n for _, n in report["by_day"])
        parts.extend(_bar(day, n, peak_day) for day, n in report["by_day"])
    parts.append(_table(["checkpoint", "answers"],
                        [(m, n) for m, n in report["by_model"]]))
    parts.append("<h2>Calibration state</h2>")
    mean = report["mean_confidence"]
    parts.append("<p>Mean confidence: %s. Non-passed abstentions: %d.</p>"
                 % ("%.3f" % mean if mean is not None else "n/a", report["abstained"]))
    parts.append("<h3>Label coverage</h3>")
    parts.append(_table(["question", "answers", "labeled", "sources"],
                        [(c["qid"], c["answers"], c["labeled"], c["sources"])
                         for c in report["coverage"]]))
    parts.append("<h2>Review queue (%d)</h2>" % len(report["review"]))
    parts.append(_table(["run", "question", "reason", "confidence", "abstention"],
                        [(e["run_id"], e["qid"], e["reason"], e["confidence"],
                          e["abstention"]) for e in report["review"][:200]]))
    payload = report["payload"]
    parts.append("<h2>Fitted payload</h2>")
    if not isinstance(payload, dict):
        parts.append("<p>No payload supplied; pass <code>--payload</code> to show one.</p>")
    else:
        temps = payload.get("temperature") or []
        parts.append(_table(["type", "temperature"],
                            [("choice", temps[0] if len(temps) > 0 else "?"),
                             ("score", temps[1] if len(temps) > 1 else "?"),
                             ("noul", temps[2] if len(temps) > 2 else "?")]))
        by_opts = payload.get("temperature_by_options") or {}
        if by_opts:
            parts.append(_table(["bucket", "temperature"], sorted(by_opts.items())))
        parts.append("<p>Version: %s.</p>" % html.escape(str(payload.get("version", "?"))))
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="decision log SQLite file")
    parser.add_argument("--out", required=True, help="HTML file to write")
    parser.add_argument("--payload", default=None,
                        help="calibration_payload JSON to show; omit for log-only")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.db, args.payload)
    except (sqlite3.Error, OSError, ValueError) as exc:
        print("outcome_report: cannot read %s: %s" % (args.db, exc), file=sys.stderr)
        return 1
    try:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(render_html(report, args.db))
    except OSError as exc:
        print("outcome_report: cannot write %s: %s" % (args.out, exc), file=sys.stderr)
        return 1
    print("wrote %s (%d answers, %d labels)" % (args.out, report["n_decisions"],
                                                report["n_outcomes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
