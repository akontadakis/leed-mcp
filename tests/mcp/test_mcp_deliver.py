# tests/mcp/test_mcp_deliver.py
"""deliver (spec 6.6): report, atlas, package as jobs."""
import importlib.util
import typing

import pytest

from job_fakes import FakeLauncher
from leed_daylight.deliver.data import OPTIONS, result_path
from leed_daylight.project.config import RENDERERS
from leed_mcp.adapters import run_checks
from leed_mcp.tools.deliver import Option, Renderer
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import SPACES_CSV, build_option2_project

pytestmark = pytest.mark.anyio


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    path = result_path(folder, "option1", "base_case")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    return folder


async def _deliver(tmp_path, folder, action, launcher=None, **arguments):
    server = tool_server(tmp_path, launcher=launcher or FakeLauncher())
    return await call(server, "deliver", {"action": action, "folder": str(folder), **arguments})


def test_the_published_options_and_renderers_are_the_pipelines():
    assert typing.get_args(Option) == OPTIONS and set(typing.get_args(Renderer)) == set(RENDERERS)


@pytest.mark.parametrize("action, arguments, tail", [
    ("report", {}, ["report", "{folder}", "--case=base_case", "--no-blender"]),
    ("report", {"option": "option1", "renderer": "pillow", "blender": True},
     ["report", "{folder}", "--case=base_case", "--option", "option1", "--renderer", "pillow"]),
    ("atlas", {}, ["atlas", "{folder}", "--case=base_case", "--no-blender"]),
    ("atlas", {"blender": True}, ["atlas", "{folder}", "--case=base_case"]),
])
async def test_blender_stays_off_unless_asked(tmp_path, folder, action, arguments, tail):
    launcher = FakeLauncher()
    result = await _deliver(tmp_path, folder, action, launcher, **arguments)
    assert result.is_error is False and result.structured_content["state"] == "started"
    width = len(tail)
    assert launcher.calls[0]["argv"][-width:] == [part.format(folder=folder) for part in tail]


async def test_package_needs_the_cases_report(tmp_path, folder):
    refused = await _deliver(tmp_path, folder, "package")
    assert codes(refused) == [("RESULTS_MISSING", "case")] and "no report yet" in refused.structured_content["reason"]
    manifest = folder / "_run" / "reports" / "base_case" / "report_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    launcher = FakeLauncher()
    assert (await _deliver(tmp_path, folder, "package", launcher)).is_error is False
    assert launcher.calls[0]["argv"][-3:] == ["package", str(folder), "--case=base_case"]


async def test_report_and_atlas_need_results(tmp_path, folder):
    result_path(folder, "option1", "base_case").unlink()
    assert codes(await _deliver(tmp_path, folder, "atlas")) == [("RESULTS_MISSING", "case")]
    refused = await _deliver(tmp_path, folder, "report", option="views")
    assert "no views results yet" in refused.structured_content["reason"]


@pytest.mark.parametrize("renderer, missing", [(None, "docx"), ("pillow", "PIL"), (None, "matplotlib")])
async def test_report_libs_missing_names_the_package(tmp_path, folder, monkeypatch, renderer, missing):
    real = importlib.util.find_spec
    monkeypatch.setattr(run_checks.importlib.util, "find_spec",
                        lambda name, *a: None if name == missing else real(name, *a))
    arguments = {"renderer": renderer} if renderer else {}
    result = await _deliver(tmp_path, folder, "report", **arguments)
    assert codes(result) == [("REPORT_LIBS_MISSING", None)]
    assert {"docx": "python-docx", "PIL": "Pillow", "matplotlib": "matplotlib"}[missing] in (
        result.structured_content["reason"])


async def test_the_atlas_needs_no_report_library(tmp_path, folder, monkeypatch):
    monkeypatch.setattr(run_checks.importlib.util, "find_spec", lambda name, *a: None)
    assert (await _deliver(tmp_path, folder, "atlas")).is_error is False


@pytest.mark.parametrize("arguments, code, field", [
    ({"option": "option1"}, "UNKNOWN_ARGUMENT", "option"),
    ({"processes": 2}, "UNKNOWN_ARGUMENT", "processes"),
    ({"renderer": "ink"}, "INPUT_INVALID", "renderer"),
])
async def test_an_argument_the_action_does_not_take_is_refused(tmp_path, folder, arguments, code, field):
    assert (code, field) in codes(await _deliver(tmp_path, folder, "atlas", **arguments))


async def test_a_scene_include_outside_the_folder_is_refused_before_the_atlas(tmp_path, folder):
    (tmp_path / "outside.rad").write_text((folder / "scene.rad").read_text())
    (folder / "scene.rad").write_text("!xform ../outside.rad\n")
    assert codes(await _deliver(tmp_path, folder, "atlas")) == [("SCENE_INCLUDE_OUTSIDE", None)]


async def test_deliver_never_imports_the_report_libraries():
    import subprocess
    import sys

    code = ("import sys, leed_mcp.tools.deliver, leed_mcp.server; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('docx', 'matplotlib', 'PIL', 'cairosvg')"
            " or m.endswith('deliver.narrative')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


async def test_a_views_only_case_is_accepted_as_the_pipeline_accepts_it(tmp_path, write_example, write_wea):
    spaces = SPACES_CSV.replace("include_base_case,reason", "include_base_case,views_scheme,reason").replace(
        ",no,,1,", ",no,,1,1,")
    folder = build_option2_project(tmp_path, write_example, write_wea, spaces_csv=spaces)
    path = result_path(folder, "views", "scheme")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    launcher = FakeLauncher()
    result = await _deliver(tmp_path, folder, "report", launcher, case="scheme", option="views")
    assert result.is_error is False and result.structured_content["case"] == "scheme"
    tail = ["report", str(folder), "--case=scheme", "--option", "views", "--no-blender"]
    assert launcher.calls[0]["argv"][-len(tail):] == tail


async def test_an_unknown_case_is_refused_naming_both_kinds_of_column(tmp_path, folder):
    refused = await _deliver(tmp_path, folder, "atlas", case="nowhere")
    assert codes(refused) == [("CASE_UNKNOWN", "case")]
    assert "include_<case> or views_<case>" in refused.structured_content["reason"]
