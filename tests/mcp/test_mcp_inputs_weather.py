# tests/mcp/test_mcp_inputs_weather.py
"""inputs(action="import_weather") (spec 6.3). Radiance never runs here:
the epw2wea on PATH is a fake written by the test."""
import os
import shutil
import sys

import pytest

from conftest import symlink_or_skip, wea_text
from leed_daylight.engine.jobs import acquire_lock, release_lock
from leed_mcp.adapters import weather_import
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from stub_program import make_program

pytestmark = pytest.mark.anyio

#: A fake epw2wea: copies FAKE_WEA to its second argument, or fails or
#: hangs when told to, and writes its first argument to FAKE_ARGV when
#: set. It never reads the .epw.
FAKE = """import os, shutil, sys, time
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "fail":
    sys.stderr.write("epw2wea: invented failure\\n")
    sys.exit(2)
if mode == "hang":
    time.sleep(5)
if os.environ.get("FAKE_ARGV"):
    with open(os.environ["FAKE_ARGV"], "w") as seen:
        seen.write(sys.argv[1])
shutil.copy(os.environ["FAKE_WEA"], sys.argv[2])
"""


@pytest.fixture
def folder(tmp_path, monkeypatch):
    folder = tmp_path / "p"
    (folder / "weather").mkdir(parents=True)
    (folder / "weather" / "site.epw").write_text("invented epw\n", newline="\n")
    bin_dir = tmp_path / "bin"
    make_program(bin_dir, "epw2wea", FAKE)
    (tmp_path / "fake.wea").write_text(wea_text(), newline="\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_WEA", str(tmp_path / "fake.wea"))
    return folder


async def _import(tmp_path, folder, **arguments):
    return await call(tool_server(tmp_path), "inputs", {"action": "import_weather", "folder": str(folder),
                                                       **arguments})


async def test_an_epw_becomes_a_wea_in_weather(tmp_path, folder):
    result = await _import(tmp_path, folder, epw="weather/site.epw")
    data = result.structured_content
    assert result.is_error is False and codes(result) == []
    assert (folder / "weather" / "site.wea").read_text() == wea_text()
    assert data["path"] == str(folder / "weather" / "site.wea") and data["backup"] is None
    assert data["epw"] == str(folder / "weather" / "site.epw")
    assert [p.name for p in (folder / "_run").iterdir()] == []


async def test_a_second_import_backs_up_the_first(tmp_path, folder):
    await _import(tmp_path, folder, epw=str(folder / "weather" / "site.epw"), name="example")
    result = await _import(tmp_path, folder, epw="weather/site.epw", name="example")
    assert result.structured_content["backup"].startswith(str(folder / "_run" / "backup" / "weather" / "example.wea."))


@pytest.mark.parametrize("arguments, code, field", [
    ({"epw": "../outside.epw"}, "PATH_OUTSIDE_ROOTS", "epw"),
    ({"epw": "weather/none.epw"}, "FILE_MISSING", "epw"),
    ({"epw": "weather/site.epw", "name": "a/b"}, "INPUT_INVALID", "name"),
    ({"epw": "weather/site.epw", "name": ".hidden"}, "INPUT_INVALID", "name"),
    ({"epw": "weather/site.epw", "file": "project"}, "UNKNOWN_ARGUMENT", "file"),
])
async def test_a_bad_argument_is_refused(tmp_path, folder, arguments, code, field):
    assert codes(await _import(tmp_path, folder, **arguments)) == [(code, field)]


@pytest.mark.parametrize("inside, code", [(False, ("PATH_OUTSIDE_ROOTS", "epw")), (True, ("PATH_UNSAFE", None))])
async def test_a_linked_epw_is_not_read(tmp_path, folder, inside, code):
    real = (folder if inside else tmp_path) / "real.epw"
    real.write_text("invented epw\n", newline="\n")
    symlink_or_skip(folder / "weather" / "linked.epw", real)
    assert codes(await _import(tmp_path, folder, epw="weather/linked.epw")) == [code]


async def test_without_epw2wea_it_is_radiance_not_found(tmp_path, folder, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    assert codes(await _import(tmp_path, folder, epw="weather/site.epw")) == [("RADIANCE_NOT_FOUND", None)]


async def test_a_failing_epw2wea_is_radiance_failed_with_its_message(tmp_path, folder, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    result = await _import(tmp_path, folder, epw="weather/site.epw")
    assert codes(result) == [("RADIANCE_FAILED", None)]
    assert "exited with code 2: epw2wea: invented failure" in result.structured_content["reason"]
    assert not (folder / "weather" / "site.wea").exists() and list((folder / "_run").iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="a .bat stub's child outlives the timeout")
async def test_a_hanging_epw2wea_is_stopped(tmp_path, folder, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    monkeypatch.setattr(weather_import, "TIMEOUT_S", 0.5)
    result = await _import(tmp_path, folder, epw="weather/site.epw")
    assert codes(result) == [("RADIANCE_FAILED", None)] and "did not finish" in result.structured_content["reason"]


async def test_epw2wea_runs_on_a_copy_in_run_that_is_removed(tmp_path, folder, monkeypatch):
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.txt"))
    result = await _import(tmp_path, folder, epw="weather/site.epw")
    assert result.is_error is False
    given = (tmp_path / "argv.txt").read_text()
    assert given.startswith(str(folder / "_run" / "epw2wea-")) and given.endswith(".epw")
    assert list((folder / "_run").iterdir()) == []


async def test_an_epw_swapped_for_a_link_after_the_check_is_not_read(tmp_path, folder, monkeypatch):
    (folder / "real.epw").write_text("invented epw\n", newline="\n")
    symlink_or_skip(folder / "weather" / "linked.epw", folder / "real.epw")
    monkeypatch.setattr(weather_import, "_epw", lambda folder, epw: folder / "weather" / "linked.epw")
    result = await _import(tmp_path, folder, epw="weather/site.epw")
    assert codes(result) == [("PATH_UNSAFE", None)]
    assert not (folder / "weather" / "linked.wea").exists() and not list(folder.glob("_run/epw2wea-*"))


async def test_an_absolute_epw_through_a_linked_prefix_is_read(tmp_path, folder):
    symlink_or_skip(tmp_path / "alias", tmp_path)
    result = await _import(tmp_path, folder, epw=str(tmp_path / "alias" / "p" / "weather" / "site.epw"))
    assert result.is_error is False and codes(result) == []
    assert result.structured_content["epw"] == str(folder / "weather" / "site.epw")
    assert (folder / "weather" / "site.wea").read_text() == wea_text()


async def test_an_absolute_epw_through_a_linked_prefix_still_refuses_a_link_inside(tmp_path, folder):
    symlink_or_skip(tmp_path / "alias", tmp_path)
    (folder / "real.epw").write_text("invented epw\n", newline="\n")
    symlink_or_skip(folder / "weather" / "linked.epw", folder / "real.epw")
    result = await _import(tmp_path, folder, epw=str(tmp_path / "alias" / "p" / "weather" / "linked.epw"))
    assert codes(result) == [("PATH_UNSAFE", None)]


async def test_import_weather_is_refused_while_a_job_holds_the_lock(tmp_path, folder):
    info = acquire_lock(folder, "run")
    try:
        result = await _import(tmp_path, folder, epw="weather/site.epw")
    finally:
        release_lock(folder, info)
    assert codes(result) == [("LOCK_HELD", None)]
    assert not (folder / "weather" / "site.wea").exists()
    assert not list(folder.glob("_run/epw2wea-*"))


async def test_import_weather_is_refused_when_project_yaml_names_a_path_outside(tmp_path, folder):
    (folder / "project.yaml").write_text('weather:\n  file: "../outside.wea"\n', newline="\n")
    result = await _import(tmp_path, folder, epw="weather/site.epw")
    assert codes(result) == [("PATH_OUTSIDE_ROOTS", None)]
    assert "weather.file" in result.structured_content["reason"]
    assert not (folder / "weather" / "site.wea").exists()
