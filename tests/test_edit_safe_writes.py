# tests/test_wizard_safe_writes.py
"""No write of the wizard follows a planted symbolic link out of the
project folder (security review H1, L5): the safe save's ``.part`` files,
the backups under ``_run/backup``, ``_run/wizard.log``, the job log,
the job record and its pruning, and the end record ``cli.main`` writes.
Each new file is created exclusively; a ``.part`` file, target or ``_run``
sub-folder that is a link is refused, and nothing is written outside.
``recent.json`` and ``wizard.log`` are readable by their owner only.

Every folder is built fresh in ``tmp_path``; the "outside" files are in a
sibling folder of the project, never a real file of the user.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib

import pytest

import leed_daylight.drivers.scan as scan_module
from conftest import symlink_or_skip
from job_fakes import FakeClock, FakeLauncher, FakeSignaller, fixed_now
from leed_daylight import cli
from leed_daylight.constants import JOB_RECORD_ENV
from leed_daylight.drivers.init import init_folder
from leed_daylight.edit import files
from leed_daylight.edit.files import Served, add_recent, save_files
from leed_daylight.engine.jobs import CURRENT, JOBS_DIR, Jobs

NOON = datetime.datetime(2026, 1, 1, 12, 0, 0)
OLD = b"modifier,primitive_type,role,note\nwall_paint,plastic,wall,\n"
NEW = b"modifier,primitive_type,role,note\nwall_paint,plastic,ceiling,\n"
KEEP = b"a file outside the project, never to be changed\n"


@pytest.fixture
def outside(tmp_path):
    """A folder beside the project, with one file the attack aims at."""
    path = tmp_path / "outside"
    path.mkdir()
    (path / "victim.txt").write_bytes(KEEP)
    return path


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "project"
    path.mkdir()
    (path / "materials.csv").write_bytes(OLD)
    return path


def _untouched(outside: pathlib.Path) -> None:
    assert (outside / "victim.txt").read_bytes() == KEEP
    assert sorted(p.name for p in outside.iterdir()) == ["victim.txt"]


def _save(folder, data=NEW):
    served = Served()
    served.remember(folder, "materials.csv")
    return save_files(folder, {"materials.csv": data}, check=lambda parts: None, served=served, now=NOON)


# --- The safe save ------------------------------------------------------------

def test_a_planted_part_link_is_refused_and_nothing_outside_is_written(folder, outside):
    symlink_or_skip(folder / "materials.csv.part", outside / "victim.txt")
    with pytest.raises(OSError, match="materials.csv.part: is a symbolic link"):
        _save(folder)
    _untouched(outside)
    assert (folder / "materials.csv").read_bytes() == OLD
    assert not (folder / "materials.csv").is_symlink()


def test_a_planted_backup_folder_link_is_refused_and_nothing_outside_is_written(folder, outside):
    (folder / "_run").mkdir()
    symlink_or_skip(folder / "_run" / "backup", outside)
    with pytest.raises(OSError, match=r"_run[\\/]backup: is a symbolic link"):
        _save(folder)
    _untouched(outside)
    assert (folder / "materials.csv").read_bytes() == OLD
    assert not list(folder.glob("*.part"))


def test_a_planted_run_folder_link_is_refused_for_the_backup(folder, outside):
    symlink_or_skip(folder / "_run", outside)
    with pytest.raises(OSError, match="_run: is a symbolic link"):
        _save(folder)
    _untouched(outside)
    assert (folder / "materials.csv").read_bytes() == OLD


def test_a_leftover_part_file_is_replaced_by_the_new_one(folder):
    (folder / "materials.csv.part").write_bytes(b"left from an earlier crash\n")
    assert _save(folder) == ("materials.csv",)
    assert (folder / "materials.csv").read_bytes() == NEW
    assert not list(folder.glob("*.part"))


def test_a_planted_part_and_backup_link_are_refused_through_save_files(tmp_path, outside):
    """Security review H1, without the page: a planted ``project.yaml.part``
    link and a planted ``_run/backup`` link, then a save."""
    project = init_folder(tmp_path / "received")
    before = (project / "project.yaml").read_bytes()
    symlink_or_skip(project / "project.yaml.part", outside / "victim.txt")
    (project / "_run").mkdir(exist_ok=True)
    symlink_or_skip(project / "_run" / "backup", outside)
    served = Served()
    served.remember(project, "project.yaml")
    with pytest.raises(OSError, match="is a symbolic link"):
        save_files(project, {"project.yaml": before + b"# edited\n"}, check=lambda parts: None,
                   served=served, now=NOON)
    _untouched(outside)
    assert not (project / "project.yaml").is_symlink()
    assert (project / "project.yaml").read_bytes() == before


# --- The recent file ----------------------------------------------------------

def test_a_planted_recent_file_link_is_refused(tmp_path, outside):
    recent = tmp_path / "home" / ".leed-daylight" / "recent.json"
    recent.parent.mkdir(parents=True)
    symlink_or_skip(recent, outside / "victim.txt")
    with pytest.raises(OSError, match="recent.json: is a symbolic link"):
        add_recent(recent, tmp_path / "received")
    _untouched(outside)
    assert recent.is_symlink()


def test_a_planted_recent_part_link_is_not_followed(tmp_path, outside):
    recent = tmp_path / "home" / ".leed-daylight" / "recent.json"
    recent.parent.mkdir(parents=True)
    symlink_or_skip(pathlib.Path(f"{recent}.part"), outside / "victim.txt")
    with pytest.raises(OSError, match="is a symbolic link"):
        add_recent(recent, tmp_path / "received")
    _untouched(outside)


# --- Jobs -------------------------------------------------------------------

def _jobs(folder, launcher):
    return Jobs(folder, launcher=launcher, now=fixed_now("2026-01-01T14:02:05"),
                signaller=FakeSignaller(FakeClock()))


def test_a_planted_jobs_folder_link_refuses_the_start(folder, outside):
    (folder / "_run").mkdir()
    symlink_or_skip(folder / "_run" / "jobs", outside)
    launcher = FakeLauncher()
    with pytest.raises(OSError, match="jobs: is a symbolic link"):
        _jobs(folder, launcher).start("run")
    assert launcher.calls == []
    _untouched(outside)


def test_a_planted_record_part_link_refuses_the_start_before_any_launch(folder, outside):
    jobs_dir = folder.joinpath(*JOBS_DIR)
    jobs_dir.mkdir(parents=True)
    symlink_or_skip(jobs_dir / f"{CURRENT}.part", outside / "victim.txt")
    launcher = FakeLauncher()
    with pytest.raises(OSError, match=f"{CURRENT}.part: is a symbolic link"):
        _jobs(folder, launcher).start("run")
    assert launcher.calls == []
    _untouched(outside)


def test_a_planted_record_link_refuses_the_start_before_any_launch(folder, outside):
    jobs_dir = folder.joinpath(*JOBS_DIR)
    jobs_dir.mkdir(parents=True)
    symlink_or_skip(jobs_dir / CURRENT, outside / "victim.txt")
    launcher = FakeLauncher()
    with pytest.raises(OSError, match=f"{CURRENT}: is a symbolic link"):
        _jobs(folder, launcher).start("run")
    assert launcher.calls == []
    _untouched(outside)


def test_the_job_record_names_paths_relative_to_the_folder_even_through_a_linked_folder(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    symlink_or_skip(tmp_path / "alias", real)
    record = _jobs(tmp_path / "alias", FakeLauncher()).start("run")
    assert record.log == "_run/jobs/20260101-140205_run.log"
    assert record.end_record == "_run/jobs/20260101-140205_run.end.json"


def test_a_planted_log_link_is_never_written_through(folder, outside):
    jobs_dir = folder.joinpath(*JOBS_DIR)
    jobs_dir.mkdir(parents=True)
    symlink_or_skip(jobs_dir / "20260101-140205_run.log", outside / "victim.txt")
    launcher = FakeLauncher(["a line"])
    record = _jobs(folder, launcher).start("run")
    assert record.log != "_run/jobs/20260101-140205_run.log"
    _untouched(outside)


# --- The end record -----------------------------------------------------------

def test_the_end_record_never_follows_a_planted_part_link(tmp_path, outside, monkeypatch, capsys):
    record = tmp_path / "job.end.json"
    symlink_or_skip(pathlib.Path(f"{record}.part"), outside / "victim.txt")
    monkeypatch.setenv(JOB_RECORD_ENV, str(record))
    monkeypatch.setattr(scan_module, "main", lambda rest, prog=None: 0)
    assert cli.main(["scan", "folder"]) == 0
    assert "warning: could not write the job end record" in capsys.readouterr().err
    _untouched(outside)
    assert not record.exists()


def test_a_leftover_end_record_part_file_is_replaced(tmp_path, monkeypatch):
    record = tmp_path / "job.end.json"
    pathlib.Path(f"{record}.part").write_text("left from an earlier crash", newline="\n")
    monkeypatch.setenv(JOB_RECORD_ENV, str(record))
    monkeypatch.setattr(scan_module, "main", lambda rest, prog=None: 0)
    assert cli.main(["scan", "folder"]) == 0
    assert json.loads(record.read_text())["exit_code"] == 0
    assert not pathlib.Path(f"{record}.part").exists()
