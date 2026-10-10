# tests/test_lock_commands.py
"""The folder lock in the seven writing verbs (wizard spec 8.2).

``prepare``, ``atlas``, ``run``, ``option2``, ``views``, ``report`` and
``package`` take ``<folder>/.leed-daylight.lock`` in ``main()``, after the
argument and folder checks and before any write, from the terminal as from
the wizard; ``scan``, ``materials`` and ``rooms`` take none. A second
command on a locked folder is refused with the holder's verb, start and
process. Every folder is built fresh in ``tmp_path``. No Radiance runs:
wherever a refused verb could reach its work, the work is replaced by one
that fails the test.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import types

import pytest

from deliver_folder import build_deliver_project
from leed_daylight.engine import jobs
from leed_daylight.engine.jobs import LOCK_FILE, LockInfo, process_start_time
from option2_folder import build_option2_project

#: verb -> the ``*_folder`` function its ``main()`` calls.
LOCKING = {
    "prepare": "prepare_folder",
    "atlas": "atlas_folder",
    "run": "run_folder",
    "option2": "option2_folder",
    "views": "run_views",
    "report": "report_folder",
    "package": "package_folder",
}
NOT_LOCKING = ("scan", "materials", "rooms")


def _driver(verb: str) -> types.ModuleType:
    return importlib.import_module(f"leed_daylight.drivers.{verb}")


def _hold(folder, verb: str = "run") -> LockInfo:
    """A lock naming this very process, as a job started from the wizard would hold it."""
    info = LockInfo(os.getpid(), process_start_time(os.getpid()), verb, "2026-01-01T14:02:00")
    (folder / LOCK_FILE).write_text(json.dumps(
        {"pid": info.pid, "start_time": info.start_time, "verb": info.verb, "started": info.started}
    ), newline="\n")
    return info


def _no_work(*args, **kwargs):
    pytest.fail("the work ran under a held lock")


def test_run_refuses_while_a_job_holds_the_lock(tmp_path, write_example, write_wea, capsys, monkeypatch):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    _hold(folder)
    monkeypatch.setattr(_driver("run"), "run_folder", _no_work)
    before = (folder / LOCK_FILE).read_bytes()

    assert _driver("run").main([str(folder)]) == 1

    assert capsys.readouterr().err == (
        f"error: run is already working on this folder (started 14:02, process {os.getpid()})\n"
    )
    assert (folder / LOCK_FILE).read_bytes() == before
    assert not (folder / "_run").exists()


def test_a_refused_option2_creates_no_run_folder(tmp_path, write_example, write_wea, capsys, monkeypatch):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    _hold(folder, "views")
    monkeypatch.setattr(_driver("option2"), "option2_folder", _no_work)

    assert _driver("option2").main([str(folder)]) == 1

    assert "views is already working on this folder (started 14:02, process" in capsys.readouterr().err
    assert not (folder / "_run").exists()
    assert sorted(p.name for p in folder.iterdir()) == [
        LOCK_FILE, "credit_rules.yaml", "materials.csv", "project.yaml", "scene.rad", "spaces.csv", "weather",
    ]


@pytest.mark.parametrize("verb", sorted(LOCKING))
def test_each_writing_verb_takes_the_lock_around_its_work(verb, tmp_path, monkeypatch):
    module = _driver(verb)
    folder = tmp_path / "project"
    folder.mkdir()
    events = []

    @contextlib.contextmanager
    def spy_lock(where, name, *, err=None):
        events.append(("lock", where, name))
        yield LockInfo(1, 0.0, name, "2026-01-01T00:00:00")
        events.append(("unlock", where, name))

    def work(*args, **kwargs):
        events.append(("work", args[0]))
        return types.SimpleNamespace(blender_failed=False)

    monkeypatch.setattr(module, "folder_lock", spy_lock)
    monkeypatch.setattr(module, LOCKING[verb], work)

    assert module.main([str(folder)]) == 0

    assert events == [("lock", folder, verb), ("work", str(folder)), ("unlock", folder, verb)]


@pytest.mark.parametrize("verb", sorted(LOCKING))
def test_a_held_lock_refuses_each_writing_verb_before_its_work(verb, tmp_path, monkeypatch, capsys):
    module = _driver(verb)
    folder = tmp_path / "project"
    folder.mkdir()
    _hold(folder, "package")
    monkeypatch.setattr(module, LOCKING[verb], _no_work)

    assert module.main([str(folder)]) == 1

    assert capsys.readouterr().err.startswith("error: package is already working on this folder (started 14:02")


@pytest.mark.parametrize("verb", sorted(LOCKING))
def test_a_folder_inside_the_repository_is_refused_before_any_lock(verb, tmp_path, monkeypatch, capsys):
    module = _driver(verb)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    folder = tmp_path / "client_model"
    folder.mkdir()
    monkeypatch.setattr(jobs, "acquire_lock", lambda *a, **k: pytest.fail("a lock was taken"))
    monkeypatch.setattr(module, LOCKING[verb], _no_work)

    assert module.main([str(folder)]) == 1

    assert "inside the repository" in capsys.readouterr().err
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize("verb", sorted(LOCKING))
def test_a_missing_folder_is_refused_before_any_lock(verb, tmp_path, monkeypatch, capsys):
    module = _driver(verb)
    monkeypatch.setattr(jobs, "acquire_lock", lambda *a, **k: pytest.fail("a lock was taken"))
    monkeypatch.setattr(module, LOCKING[verb], _no_work)

    assert module.main([str(tmp_path / "missing")]) == 1

    assert "not a folder (or does not exist)" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()


def test_the_read_only_verbs_take_no_lock(tmp_path, write_example, write_wea, monkeypatch, capsys):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    _hold(folder, "run")
    monkeypatch.setattr(jobs, "acquire_lock", lambda *a, **k: pytest.fail("a lock was taken"))

    for verb in NOT_LOCKING:
        module = _driver(verb)
        assert not hasattr(module, "folder_lock"), verb
        assert module.main([str(folder)]) == 0, verb
    capsys.readouterr()


def test_report_building_the_atlas_locks_once(tmp_path, write_example, write_wea, monkeypatch, capsys):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))
    taken = []
    real = jobs.acquire_lock

    def counting(where, verb, **kwargs):
        taken.append(verb)
        return real(where, verb, **kwargs)

    monkeypatch.setattr(jobs, "acquire_lock", counting)

    assert _driver("report").main([str(folder), "--option", "option1", "--no-blender"]) == 0

    assert taken == ["report"]
    assert (folder / "_run" / "atlas").is_dir()
    assert not (folder / LOCK_FILE).exists()
    capsys.readouterr()


def test_a_finished_verb_leaves_no_lock(tmp_path, write_example, write_wea, capsys):
    folder = build_option2_project(tmp_path, write_example, write_wea)

    assert _driver("prepare").main([str(folder)]) == 0

    assert not (folder / LOCK_FILE).exists()
    assert not [p for p in folder.iterdir() if p.name.startswith(LOCK_FILE)]
    capsys.readouterr()
