# tests/test_ci_workflow.py
"""CI runs the suite on macOS, Windows and Linux (spec 2026-10-09 A4)."""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_suite_runs_on_three_systems():
    flow = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
    job = flow["jobs"]["test"]
    assert job["strategy"]["matrix"]["os"] == ["macos-latest", "windows-latest", "ubuntu-latest"]
    assert job["runs-on"] == "${{ matrix.os }}"
    assert job["strategy"]["fail-fast"] is False
    commands = " ".join(step.get("run", "") for step in job["steps"])
    assert "python tools/no_radiance.py" in commands
    assert job["env"]["UV_PYTHON"] == "3.12"
