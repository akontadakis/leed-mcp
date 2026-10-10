# tests/test_cli.py
"""Tests for leed_daylight.cli: the ``leed-daylight`` command (CLI spec 3).

The dispatcher reads only the verb and hands the rest of the line to that
verb's own module ``main(rest, prog=...)``. A stub module put in
``sys.modules`` stands in for a real verb, so these tests never run a real
command.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
import types
import pathlib

import pytest

from leed_daylight import cli
from leed_daylight.constants import TOOL_VERSION

ROOT = pathlib.Path(__file__).resolve().parents[1]

VERB_ORDER = ["init", "scan", "materials", "rooms", "prepare", "run", "option2", "views", "report",
              "atlas", "calculator", "package"]


def _stub(monkeypatch, verb, result=0, raises=None):
    calls = []

    def main(argv=None, prog=None):
        calls.append((list(argv), prog))
        if raises is not None:
            raise raises
        return result

    module = types.ModuleType(f"leed_daylight.drivers.{verb}")
    module.main = main
    monkeypatch.setitem(sys.modules, f"leed_daylight.drivers.{verb}", module)
    return calls


def test_the_verbs_are_in_work_order():
    assert [verb for verb, _ in cli.VERBS] == VERB_ORDER


@pytest.mark.parametrize("argv", [[], ["-h"], ["--help"]])
def test_help_lists_every_verb_in_order(argv, capsys):
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: leed-daylight <verb> [arguments]")
    positions = [out.index(f"\n  {verb} ") for verb in VERB_ORDER]
    assert positions == sorted(positions)


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == f"leed-daylight {TOOL_VERSION}\n"


def test_an_unknown_verb_returns_2(capsys):
    assert cli.main(["simulate"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: unknown verb 'simulate'; known verbs are init, scan, materials,")


def test_an_option_before_the_verb_returns_2(capsys):
    assert cli.main(["--case", "base", "run"]) == 2
    assert "unknown verb '--case'" in capsys.readouterr().err


def test_a_verb_gets_the_rest_of_the_line_and_its_prog(monkeypatch):
    calls = _stub(monkeypatch, "run", result=0)
    assert cli.main(["run", "office_a", "--case", "base"]) == 0
    assert calls == [(["office_a", "--case", "base"], "leed-daylight run")]


def test_the_verb_exit_code_is_returned(monkeypatch):
    _stub(monkeypatch, "report", result=1)
    assert cli.main(["report", "office_a"]) == 1


def test_ctrl_c_prints_how_to_resume_and_returns_130(monkeypatch, capsys):
    _stub(monkeypatch, "run", raises=KeyboardInterrupt())
    assert cli.main(["run", "office_a"]) == 130
    assert capsys.readouterr().err == "stopped; run the same command again to resume\n"


def test_a_verb_help_exit_passes_through(monkeypatch):
    _stub(monkeypatch, "views", raises=SystemExit(0))
    with pytest.raises(SystemExit) as info:
        cli.main(["views", "--help"])
    assert info.value.code == 0


def test_real_verb_help_names_the_new_command(capsys):
    with pytest.raises(SystemExit):
        cli.main(["init", "--help"])
    assert capsys.readouterr().out.startswith("usage: leed-daylight init ")


def test_help_imports_no_verb_module():
    code = (
        "import sys; from leed_daylight import cli; cli.main(['--help']); "
        "print(sorted(m for m in sys.modules if m.startswith('leed_daylight.drivers.')))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_python_dash_m_runs_the_command():
    done = subprocess.run(
        [sys.executable, "-m", "leed_daylight", "--version"], capture_output=True, text=True, cwd=ROOT,
    )
    assert done.returncode == 0
    assert done.stdout == f"leed-daylight {TOOL_VERSION}\n"


def test_the_console_script_is_declared():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["project"]["scripts"] == {
        "leed-daylight": "leed_daylight.cli:main",
        "leed-mcp": "leed_mcp.server:main",
    }


def test_ctrl_c_during_the_verb_import_is_also_caught(monkeypatch, capsys):
    """M5: KeyboardInterrupt raised while importing the verb's module (not
    only while the verb's own main() runs) must still print the resume
    message and return 130, not propagate."""
    real_import_module = cli.importlib.import_module

    def flaky_import(name, *args, **kwargs):
        if name == "leed_daylight.drivers.run":
            raise KeyboardInterrupt()
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", flaky_import)
    assert cli.main(["run", "office_a"]) == 130
    assert capsys.readouterr().err == "stopped; run the same command again to resume\n"
