# tests/mcp/test_mcp_time_scene.py
"""tools/time_scene.py: the Q1 timing of scene scan and scene rooms (spec 13)."""
import importlib.util
import itertools
import pathlib
import sys

import pytest

from option2_folder import build_option2_project

ROOT = pathlib.Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("time_scene", ROOT / "tools" / "time_scene.py")
time_scene = importlib.util.module_from_spec(_spec)
sys.modules["time_scene"] = time_scene
_spec.loader.exec_module(time_scene)


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_option2_project(tmp_path, write_example, write_wea)


def _steps(seconds):
    """A clock that moves ``seconds`` at every reading."""
    counter = itertools.count()
    return lambda: next(counter) * seconds


def test_both_actions_are_timed_and_judged(folder, capsys):
    assert time_scene.main([str(folder)], clock=_steps(1.0)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["scene scan: 1.0 s, 4 materials, 1 storeys, 0 problems", "scene rooms: 1.0 s, 1 rooms",
                     "Q1: scene scan and scene rooms each took under 30 s"]
    assert (folder / "_run" / "spaces_template.csv").is_file()


def test_an_action_over_30_s_is_named(folder, capsys):
    assert time_scene.main([str(folder)], clock=_steps(40.0)) == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Q1: scene scan and scene rooms took over 30 s: spec Q1 moves it to a job")


def test_a_refused_action_says_why(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert time_scene.main([str(tmp_path / "empty")]) == 1
    assert capsys.readouterr().out.startswith("scene scan: refused (FILE_MISSING): ")


@pytest.mark.parametrize("argv", [[], ["a", "b"], ["/no/such/folder"]])
def test_a_bad_argument_is_refused(argv, capsys):
    assert time_scene.main(argv) == 2
    assert capsys.readouterr().err.startswith(("usage:", "error:"))


@pytest.mark.parametrize("folder", [ROOT, ROOT / "tools"])
def test_this_repository_or_a_folder_inside_it_is_refused(capsys, folder):
    assert time_scene.main([str(folder)]) == 2
    assert "cannot be a root" in capsys.readouterr().err
