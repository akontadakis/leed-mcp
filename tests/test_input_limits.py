# tests/test_input_limits.py
"""Bounded reads of the project's input files (security review L2).

``project.yaml``, the CSV files and the scene files are read only as plain
files of at most ``INPUT_FILE_MAX`` bytes, by the verbs and the edit helpers
alike; a FIFO is refused at once, never waited on; a scene file that
includes itself is one plain refusal. The limit is made small here, so no
test writes a large file.
"""
from __future__ import annotations

import os
import threading

import pytest

from leed_daylight.edit.files import sha256_or_none
from leed_daylight.project.config import check_project_draft, load_project, load_scene_settings
from leed_daylight.project.inputs import _project_yaml_digest
from leed_daylight.project.materials import load_materials
from leed_daylight.project.spaces import load_spaces
from leed_daylight.project.splits import read_splits
from leed_daylight.scene import input_file
from leed_daylight.scene.reader import XformError, read_scene

LIMIT = 64
TOO_LARGE = f"is larger than {LIMIT} bytes; an input file this large is not read"
BLOCKED_S = 5.0


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setattr(input_file, "INPUT_FILE_MAX", LIMIT)


def _big(path, line: str = "# a comment line that only fills the file\n"):
    path.write_text(line * (LIMIT // len(line) + 2))
    return path


def _within(seconds: float, func):
    box: dict = {}

    def run():
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001 - handed back to the test
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"blocked for more than {seconds} s"
    return box


@pytest.mark.parametrize("load", [load_project, load_scene_settings, check_project_draft, _project_yaml_digest],
                         ids=lambda f: f.__name__)
def test_a_project_yaml_over_the_limit_is_refused(tmp_path, small_limit, load):
    path = _big(tmp_path / "project.yaml")
    with pytest.raises(ValueError, match=TOO_LARGE):
        load(path)


@pytest.mark.parametrize("load, name", [(load_spaces, "spaces.csv"), (load_materials, "materials.csv"),
                                        (read_splits, "room_splits.csv")], ids=["spaces", "materials", "splits"])
def test_a_csv_file_over_the_limit_is_refused(tmp_path, small_limit, load, name):
    path = _big(tmp_path / name, "a,b,c,d,e,f,g,h,i,j,k,l\n")
    with pytest.raises(ValueError, match=TOO_LARGE):
        load(path)


def test_a_scene_file_over_the_limit_is_refused(tmp_path, small_limit):
    path = _big(tmp_path / "scene.rad")
    with pytest.raises(ValueError, match=TOO_LARGE):
        read_scene([path], "m")


def test_an_included_scene_file_over_the_limit_is_refused(tmp_path, small_limit):
    _big(tmp_path / "part.rad")
    (tmp_path / "scene.rad").write_text("!xform part.rad\n")
    with pytest.raises(ValueError, match=TOO_LARGE):
        read_scene([tmp_path / "scene.rad"], "m")


def test_the_edit_digest_of_a_file_over_the_limit_is_refused(tmp_path, small_limit):
    with pytest.raises(ValueError, match=TOO_LARGE):
        sha256_or_none(_big(tmp_path / "project.yaml"))


def test_a_file_at_the_limit_is_read(tmp_path, small_limit):
    path = tmp_path / "scene.rad"
    path.write_text("#" * (LIMIT - 1) + "\n")
    assert read_scene([path], "m") == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
@pytest.mark.parametrize("name", ["project.yaml", "scene.rad"])
def test_a_fifo_input_is_refused_at_once(tmp_path, name):
    os.mkfifo(tmp_path / name)
    load = {"project.yaml": lambda: load_project(tmp_path / name),
            "scene.rad": lambda: read_scene([tmp_path / name], "m")}[name]
    box = _within(BLOCKED_S, load)
    assert isinstance(box.get("error"), ValueError)
    assert str(box["error"]) == f"{tmp_path / name}: is not a plain file; it is not read"


def test_a_scene_file_that_includes_itself_is_one_plain_refusal(tmp_path):
    (tmp_path / "scene.rad").write_text("!xform scene.rad\n")
    with pytest.raises(XformError) as caught:
        read_scene([tmp_path / "scene.rad"], "m")
    assert str(caught.value) == (
        f"{tmp_path / 'scene.rad'}: the !xform includes nest too deep (over 32 levels); a scene file probably "
        "includes itself; remove that !xform"
    )


def test_two_scene_files_that_include_each_other_are_refused(tmp_path):
    (tmp_path / "a.rad").write_text("!xform b.rad\n")
    (tmp_path / "b.rad").write_text("!xform -t 1 0 0 a.rad\n")
    with pytest.raises(XformError, match="nest too deep"):
        read_scene([tmp_path / "a.rad"], "m")


def test_a_folder_is_a_directory_error_on_every_system(tmp_path, monkeypatch):
    folder = tmp_path / "spaces.csv"
    folder.mkdir()
    with pytest.raises(IsADirectoryError):
        input_file.open_input(folder)

    def refused(name, flags, *args):
        raise PermissionError(13, "Permission denied", str(name))  # what Windows gives for a folder

    monkeypatch.setattr(input_file.os, "open", refused)
    with pytest.raises(IsADirectoryError):
        input_file.open_input(folder, platform="win32")
    with pytest.raises(PermissionError) as raised:
        input_file.open_input(tmp_path / "absent.csv", platform="win32")  # a real permission problem stays one
    assert not isinstance(raised.value, IsADirectoryError)
