# tests/mcp/test_mcp_run.py
"""run (spec 6.4) and the checks before every job (spec 6.4, 9, 10)."""
import argparse
import json
import os
import shutil
import sys

import pytest

from conftest import symlink_or_skip
from job_fakes import FakeLauncher
from leed_daylight.scene import input_file
from leed_mcp.adapters import project_state, run_checks
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import SPACES_CSV, build_option2_project

pytestmark = pytest.mark.anyio


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_option2_project(tmp_path, write_example, write_wea)


@pytest.fixture
def radiance(monkeypatch):
    """Every Radiance program 'found': a run passes its check, and none starts (the launcher is a fake)."""
    monkeypatch.setattr(project_state.shutil, "which", lambda name: f"/invented/bin/{name}")


async def _run(tmp_path, folder, action, launcher=None, **arguments):
    server = tool_server(tmp_path, launcher=launcher or FakeLauncher())
    return await call(server, "run", {"action": action, "folder": str(folder), **arguments})


async def test_prepare_starts_a_job_and_answers_at_once(tmp_path, folder):
    launcher = FakeLauncher()
    result = await _run(tmp_path, folder, "prepare", launcher)
    data = result.structured_content
    assert result.is_error is False and codes(result) == []
    assert (data["action"], data["state"], data["verb"], data["case"], data["folder"]) == (
        "prepare", "started", "prepare", "base_case", str(folder))
    assert data["log"].startswith(str(folder / "_run" / "jobs")) and data["log"].endswith("_prepare.log")
    assert launcher.calls[0]["argv"][-3:] == ["prepare", str(folder), "--case=base_case"]
    state = tmp_path.with_name(tmp_path.name + ".state")
    record = json.loads((state / "jobs" / f"{data['job_id']}.json").read_text())
    assert (record["folder"], record["action"], record["pid"]) == (str(folder), "prepare", os.getpid())


async def test_option1_runs_the_run_verb_with_the_case_and_processes(tmp_path, folder, radiance):
    launcher = FakeLauncher()
    result = await _run(tmp_path, folder, "option1", launcher, case="base_case", processes=1)
    assert result.structured_content["verb"] == "run"
    assert launcher.calls[0]["argv"][-5:] == ["run", str(folder), "--case=base_case", "--processes", "1"]


async def test_without_radiance_a_run_is_refused_and_names_the_programs(tmp_path, folder, monkeypatch):
    monkeypatch.setattr(project_state.shutil, "which", lambda name: None)
    result = await _run(tmp_path, folder, "option2")
    assert codes(result) == [("RADIANCE_NOT_FOUND", None)]
    assert "oconv, rtrace, gendaylit, xform not found" in result.structured_content["reason"]


async def test_views_takes_the_first_views_case_and_refuses_without_one(tmp_path, write_example, write_wea,
                                                                         radiance):
    spaces = SPACES_CSV.replace("include_base_case,reason", "include_base_case,views_scheme,reason").replace(
        ",no,,1,", ",no,,1,1,")
    folder = build_option2_project(tmp_path, write_example, write_wea, spaces_csv=spaces)
    assert (await _run(tmp_path, folder, "views")).structured_content["case"] == "scheme"
    (tmp_path / "b").mkdir()
    other = build_option2_project(tmp_path / "b", write_example, write_wea)
    refused = await _run(tmp_path, other, "views")
    assert codes(refused) == [("CASE_UNKNOWN", "case")]
    assert "no views_<case> column" in refused.structured_content["reason"]


@pytest.mark.parametrize("arguments, code, field", [
    ({"case": "other"}, "CASE_UNKNOWN", "case"),
    ({"case": "a/b"}, "INPUT_INVALID", "case"),
    ({"processes": 0}, "INPUT_INVALID", "processes"),
    ({"processes": 100000}, "INPUT_INVALID", "processes"),
    ({"tail": 5}, "UNKNOWN_ARGUMENT", "tail"),
])
async def test_a_bad_argument_is_refused_and_nothing_starts(tmp_path, folder, radiance, arguments, code, field):
    launcher = FakeLauncher()
    result = await _run(tmp_path, folder, "option1", launcher, **arguments)
    assert codes(result) == [(code, field)] and launcher.calls == []


async def test_prepare_takes_no_processes(tmp_path, folder):
    assert codes(await _run(tmp_path, folder, "prepare", processes=2)) == [("UNKNOWN_ARGUMENT", "processes")]


async def test_fill_me_left_names_the_keys(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea,
                                   mutate=lambda d: d["site"].update(latitude_N="FILL_ME"))
    result = await _run(tmp_path, folder, "prepare")
    assert codes(result) == [("FILL_ME_LEFT", "folder")] and "site.latitude_N" in result.structured_content["reason"]


async def test_a_missing_input_file_is_file_missing(tmp_path, folder):
    (folder / "spaces.csv").unlink()
    result = await _run(tmp_path, folder, "prepare")
    assert codes(result) == [("FILE_MISSING", "folder")]
    assert "spaces.csv missing" in result.structured_content["reason"]


async def test_a_project_yaml_path_outside_the_folder_is_refused_before_the_job(tmp_path, folder):
    shutil.copy(folder / "weather" / "example.wea", tmp_path / "elsewhere.wea")
    text = (folder / "project.yaml").read_text().replace("weather/example.wea", "../elsewhere.wea")
    (folder / "project.yaml").write_text(text)
    launcher = FakeLauncher()
    result = await _run(tmp_path, folder, "prepare", launcher)
    assert codes(result) == [("PATH_OUTSIDE_ROOTS", None)] and launcher.calls == []
    assert "weather.file" in result.structured_content["reason"]


async def test_a_scene_include_outside_the_folder_is_refused(tmp_path, folder):
    (tmp_path / "outside.rad").write_text((folder / "scene.rad").read_text())
    (folder / "scene.rad").write_text("!xform ../outside.rad\n")
    assert codes(await _run(tmp_path, folder, "prepare")) == [("SCENE_INCLUDE_OUTSIDE", None)]


async def test_a_linked_input_file_is_never_read(tmp_path, folder):
    (tmp_path / "real.csv").write_text((folder / "spaces.csv").read_text())
    (folder / "spaces.csv").unlink()
    symlink_or_skip(folder / "spaces.csv", tmp_path / "real.csv")
    assert codes(await _run(tmp_path, folder, "prepare")) == [("PATH_UNSAFE", None)]


async def test_a_project_the_loader_refuses_is_input_invalid(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea,
                                   mutate=lambda d: d["glazing"].update(frame_style="bogus"))
    result = await _run(tmp_path, folder, "prepare")
    assert codes(result) == [("INPUT_INVALID", None)] and "glazing.frame_style" in result.structured_content["reason"]


async def test_an_oversized_project_yaml_is_path_unsafe_not_input_invalid(tmp_path, folder, monkeypatch):
    monkeypatch.setattr(input_file, "INPUT_FILE_MAX", 10)
    launcher = FakeLauncher()
    result = await _run(tmp_path, folder, "prepare", launcher)
    assert codes(result) == [("PATH_UNSAFE", None)] and launcher.calls == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX owner, modes and process groups")
async def test_a_job_index_folder_others_can_write_is_path_unsafe(tmp_path, folder):
    state = tmp_path.with_name(tmp_path.name + ".state")
    state.mkdir()
    state.chmod(0o777)
    launcher = FakeLauncher()
    result = await _run(tmp_path, folder, "prepare", launcher)
    assert codes(result) == [("PATH_UNSAFE", None)] and "others can write it" in result.structured_content["reason"]
    assert launcher.calls == []


async def test_a_case_name_starting_with_a_dash_reaches_the_driver_as_one_argument(
        tmp_path, folder, radiance, monkeypatch):
    """argparse reads ``--case -north`` as a missing value; ``--case=-north`` is one argument."""
    monkeypatch.setattr(run_checks, "resolve_case", lambda path, case, **kw: case)
    launcher = FakeLauncher()
    await _run(tmp_path, folder, "prepare", launcher, case="-north")
    assert launcher.calls[0]["argv"][-2:] == [str(folder), "--case=-north"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    assert parser.parse_args(launcher.calls[0]["argv"][-1:]).case == "-north"
