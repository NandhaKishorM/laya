import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_zev_rs_public_comparison_artifact_exists():
    payload = json.loads((REPO / "research" / "results" / "zev_rs_public_comparison.json").read_text())

    assert payload["source"] == "https://bhubbard.github.io/zev-rs/"
    assert payload["zev_default"]["accuracy"] > 0.68
    assert payload["zev_apfel"]["accuracy"] > payload["zev_default"]["accuracy"]


def test_compare_script_runs_and_reports_summary():
    result = subprocess.run(
        [sys.executable, "research/scripts/compare_zev_rs.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Zev RS" in result.stdout
    assert "Laya" in result.stdout
