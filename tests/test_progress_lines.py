# tests/test_progress_lines.py
"""Progress lines (wizard spec 8.4): ``run``, ``option2`` and ``views``
print ``progress: <text>`` once per finished unit of work, so the wizard
can show the latest above the log. Every file they write is byte-identical
with and without them. Every run here uses a fake engine or tracer; no
Radiance runs.
"""
from __future__ import annotations

import io
import pathlib
import shutil

import pytest

import leed_daylight.drivers.option2 as option2_module
import leed_daylight.drivers.run as run_module
import leed_daylight.drivers.views as views_module
from deliver_folder import CASE, Option1Fake, Option2Fake
from option2_folder import build_option2_project, load_chain
from test_views_command import MATERIALS_CSV, RULES_YAML, SCENE_RAD, SPACES_CSV, VIEWS, FakeTracer

OPTION1_LINES = [
    "progress: residual pass",
    "progress: window group 1 of 1 (direct)",
    "progress: window group 1 of 1 (sky)",
]
OPTION2_LINES = [
    "progress: sky 1 of 4 (03-28_0900)",
    "progress: sky 2 of 4 (03-28_1500)",
    "progress: sky 3 of 4 (09-14_0900)",
    "progress: sky 4 of 4 (09-14_1500)",
]
VIEWS_LINES = ["progress: space 1 of 1"]


@pytest.fixture
def option_folder(tmp_path, write_example, write_wea):
    return build_option2_project(tmp_path, write_example, write_wea)


@pytest.fixture
def views_folder(tmp_path, write_example):
    def mutate(d):
        d["views"] = dict(VIEWS)
        return d

    base = tmp_path / "views"
    base.mkdir()
    folder = write_example(base, mutate).parent
    (folder / "scene.rad").write_text(SCENE_RAD, newline="\n")
    (folder / "materials.csv").write_text(MATERIALS_CSV, newline="\n")
    (folder / "spaces.csv").write_text(SPACES_CSV, newline="\n")
    (folder / "credit_rules.yaml").write_text(RULES_YAML, newline="\n")
    return folder


def _option1(folder, **kwargs) -> str:
    _, prepared, _, _, _ = load_chain(folder)
    out = io.StringIO()
    run_module.run_folder(folder, CASE, engine=Option1Fake(prepared), out=out, **kwargs)
    return out.getvalue()


def _option2(folder, **kwargs) -> str:
    _, prepared, _, _, _ = load_chain(folder)
    out = io.StringIO()
    option2_module.option2_folder(folder, CASE, engine=Option2Fake(prepared), out=out, **kwargs)
    return out.getvalue()


def _views(folder, **kwargs) -> str:
    out = io.StringIO()
    views_module.run_views(folder, tracer=FakeTracer(), out=out, **kwargs)
    return out.getvalue()


def _progress(printed: str) -> list[str]:
    return [line for line in printed.splitlines() if line.startswith("progress: ")]


def _files(folder: pathlib.Path) -> dict[str, bytes]:
    run = folder / "_run"
    return {p.relative_to(run).as_posix(): p.read_bytes() for p in sorted(run.rglob("*")) if p.is_file()}


def test_run_prints_a_progress_line_per_unit(option_folder):
    assert _progress(_option1(option_folder, progress=True)) == OPTION1_LINES


def test_option2_prints_a_progress_line_per_sky(option_folder):
    assert _progress(_option2(option_folder, progress=True)) == OPTION2_LINES


def test_views_prints_a_progress_line_per_space(views_folder):
    assert _progress(_views(views_folder, progress=True)) == VIEWS_LINES


def test_library_callers_see_no_progress_line(option_folder, views_folder):
    assert _progress(_option1(option_folder)) == []
    assert _progress(_option2(option_folder)) == []
    assert _progress(_views(views_folder)) == []


@pytest.mark.parametrize("run, folder_fixture", [
    (_option1, "option_folder"), (_option2, "option_folder"), (_views, "views_folder"),
])
def test_results_are_byte_identical_with_and_without_progress(request, run, folder_fixture):
    folder = request.getfixturevalue(folder_fixture)
    quiet = run(folder)
    without = _files(folder)
    shutil.rmtree(folder / "_run")
    loud = run(folder, progress=True)
    assert _files(folder) == without
    assert [line for line in loud.splitlines() if not line.startswith("progress: ")] == quiet.splitlines()


@pytest.mark.parametrize("verb, work", [
    ("run", "run_folder"), ("option2", "option2_folder"), ("views", "run_views"),
])
def test_each_main_asks_for_progress_lines(option_folder, monkeypatch, verb, work):
    module = {"run": run_module, "option2": option2_module, "views": views_module}[verb]
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return option_folder

    monkeypatch.setattr(module, work, spy)
    assert module.main([str(option_folder)]) == 0
    assert seen["progress"] is True
