# tests/test_cli_end_record.py
"""The job end record (wizard spec 8.1): with ``LEED_JOB_RECORD`` set,
``cli.main`` writes the verb's exit code and end time to that path however
the verb ends, so a finished job is told from a crashed one. The verb here
is ``scan`` with its ``main`` replaced; nothing real runs.
"""
from __future__ import annotations

import json
import re

import pytest

import leed_daylight.drivers.scan as scan_module
from leed_daylight import cli
from leed_daylight.constants import JOB_RECORD_ENV


@pytest.fixture
def record(tmp_path, monkeypatch):
    path = tmp_path / "job.end.json"
    monkeypatch.setenv(JOB_RECORD_ENV, str(path))
    return path


def _verb(monkeypatch, behaviour):
    monkeypatch.setattr(scan_module, "main", lambda rest, prog=None: behaviour())


def _read(path):
    data = json.loads(path.read_text())
    assert set(data) == {"exit_code", "ended"}
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", data["ended"])
    return data["exit_code"]


def test_the_variable_names_the_end_record():
    assert JOB_RECORD_ENV == "LEED_JOB_RECORD"


@pytest.mark.parametrize("code", [0, 1, 3])
def test_a_returned_code_is_recorded(record, monkeypatch, code):
    _verb(monkeypatch, lambda: code)
    assert cli.main(["scan", "folder"]) == code
    assert _read(record) == code
    assert not record.with_name(record.name + ".part").exists()


def test_ctrl_c_is_recorded_as_130(record, monkeypatch, capsys):
    def interrupted():
        raise KeyboardInterrupt

    _verb(monkeypatch, interrupted)
    assert cli.main(["scan", "folder"]) == 130
    assert _read(record) == 130
    assert "stopped; run the same command again to resume" in capsys.readouterr().err


@pytest.mark.parametrize("code, recorded", [(2, 2), (0, 0), (None, 0), ("a message", 1)])
def test_a_system_exit_is_recorded_and_raised_again(record, monkeypatch, code, recorded):
    def leaves():
        raise SystemExit(code)

    _verb(monkeypatch, leaves)
    with pytest.raises(SystemExit) as raised:
        cli.main(["scan", "folder"])
    assert raised.value.code == code
    assert _read(record) == recorded


def test_the_argument_parser_s_usage_error_is_recorded(record):
    with pytest.raises(SystemExit):
        cli.main(["scan"])
    assert _read(record) == 2


def test_a_crash_is_recorded_as_1_and_raised_again(record, monkeypatch):
    def crashes():
        raise RuntimeError("invented crash")

    _verb(monkeypatch, crashes)
    with pytest.raises(RuntimeError, match="invented crash"):
        cli.main(["scan", "folder"])
    assert _read(record) == 1


def test_an_unknown_verb_is_recorded(record):
    assert cli.main(["nonsense"]) == cli.USAGE_ERROR
    assert _read(record) == cli.USAGE_ERROR


def test_no_record_without_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(JOB_RECORD_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    _verb(monkeypatch, lambda: 0)
    assert cli.main(["scan", "folder"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_a_record_that_cannot_be_written_only_warns(tmp_path, monkeypatch, capsys):
    path = tmp_path / "missing" / "job.end.json"
    monkeypatch.setenv(JOB_RECORD_ENV, str(path))
    _verb(monkeypatch, lambda: 0)
    assert cli.main(["scan", "folder"]) == 0
    err = capsys.readouterr().err
    assert err.startswith(f"warning: could not write the job end record {path}: ")
    assert err.count("\n") == 1
    assert not path.parent.exists()
