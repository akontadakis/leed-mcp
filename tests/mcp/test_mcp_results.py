# tests/mcp/test_mcp_results.py
"""The results tool (spec 6.5) on the invented deliver project (fake engines)."""
import shutil

import pytest

from conftest import symlink_or_skip
from deliver_folder import CASE, build_deliver_project, edit_results
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture

pytestmark = pytest.mark.anyio


@pytest.fixture
def project(tmp_path, write_example, write_wea):
    return build_deliver_project(tmp_path, write_example, write_wea)


def _args(folder, **more):
    return {"folder": str(folder), "case": CASE, "credit": "option1", **more}


async def test_summary_gives_score_metrics_and_the_pipeline_warning(tmp_path, project):
    result = await call(tool_server(tmp_path), "results", {"action": "summary", **_args(project)})
    data = result.structured_content
    assert data["score"]["points"] == 1 and data["inputs"] == "current" and data["changed_inputs"] == []
    assert data["metrics"]["headline"] == "floor_area" and round(data["metrics"]["sda_area_pct"], 2) == 44.44
    assert data["spaces"] == 1 and data["tool_version"] == "0.1.0"
    assert codes(result) == [("PIPELINE_WARNING", "warnings")]


async def test_summary_says_which_inputs_changed_since_the_run(tmp_path, project):
    with open(project / "spaces.csv", "a") as handle:
        handle.write("\n")
    result = await call(tool_server(tmp_path), "results", {"action": "summary", **_args(project)})
    assert result.structured_content["inputs"] == "changed"
    assert result.structured_content["changed_inputs"] == ["spaces.csv"]
    assert ("INPUTS_CHANGED_SINCE_RUN", "inputs") in codes(result)


async def test_summary_of_views_carries_the_views_metrics(tmp_path, project):
    data = (await call(tool_server(tmp_path), "results",
                       {"action": "summary", **_args(project, credit="views")})).structured_content
    assert set(data["metrics"]) == {"quality_area_pct", "quality_point_pct", "type2_area_pct",
                                    "type3_area_pct", "total_area_m2", "quality_area_m2"}
    assert all(isinstance(data["metrics"][key], (int, float)) for key in ("quality_area_pct", "total_area_m2"))


async def test_a_missing_result_is_results_missing(tmp_path, project):
    result = await call(tool_server(tmp_path), "results", {"action": "summary", **_args(project, case="other")})
    assert codes(result) == [("RESULTS_MISSING", "case")]


async def test_an_unsafe_case_is_input_invalid(tmp_path, project):
    result = await call(tool_server(tmp_path), "results", {"action": "summary", **_args(project, case="../x")})
    assert codes(result) == [("INPUT_INVALID", "case")]


async def test_a_results_file_that_is_not_json_is_input_invalid(tmp_path, project):
    (project / "_run" / "option1" / CASE / "results.json").write_text("{not json", newline="\n")
    result = await call(tool_server(tmp_path), "results", {"action": "summary", **_args(project)})
    assert codes(result) == [("INPUT_INVALID", None)]


async def test_a_linked_results_folder_is_never_read(tmp_path, project):
    run = project / "_run" / "option1"
    shutil.move(run, tmp_path / "elsewhere")
    symlink_or_skip(run, tmp_path / "elsewhere")
    result = await call(tool_server(tmp_path), "results", {"action": "summary", **_args(project)})
    assert codes(result) == [("PATH_UNSAFE", None)]


async def test_spaces_pages_and_caps_the_limit(tmp_path, project):
    server = tool_server(tmp_path)
    data = (await call(server, "results", {"action": "spaces", **_args(project)})).structured_content
    assert data["total"] == 1 and data["rows"][0]["room"] == "Room North" and data["limit"] == 100
    result = await call(server, "results", {"action": "spaces", **_args(project, limit=201)})
    assert codes(result) == [("INPUT_INVALID", "limit")]


async def test_points_types_the_cells_filters_by_space_and_pages(tmp_path, project):
    server = tool_server(tmp_path)
    data = (await call(server, "results", {"action": "points", **_args(project)})).structured_content
    assert data["total"] == 54 and data["columns"][:3] == ["room", "storey", "x"]
    first = data["rows"][0]
    assert first["room"] == "Room North" and first["storey"] == 0 and first["sda_pass"] in (True, False)
    assert isinstance(first["da_pct"], float) and isinstance(first["ase_hours"], int)
    none = await call(server, "results", {"action": "points", **_args(project, space="Nowhere")})
    assert none.structured_content["total"] == 0
    page = await call(server, "results", {"action": "points", **_args(project, offset=50, limit=10)})
    assert len(page.structured_content["rows"]) == 4 and codes(page) == []
    cut = await call(server, "results", {"action": "points", **_args(project, limit=10)})
    assert codes(cut) == [("RESULT_TRUNCATED", "rows")]


async def test_compare_gives_differences_and_the_spaces_in_one_case_only(tmp_path, project):
    run = project / "_run" / "option1"
    shutil.copytree(run / CASE, run / "variant")

    def change(payload):
        payload["spaces"][0]["sda_pct"] = 50.0
        payload["spaces"].append({**payload["spaces"][0], "room": "Room Extra"})

    edit_results(project, "option1", change)
    result = await call(tool_server(tmp_path), "results", {
        "action": "compare", "folder": str(project), "credit": "option1", "case_a": "variant", "case_b": CASE})
    data = result.structured_content
    assert [s["room"] for s in data["spaces"]] == ["Room North"]
    assert round(data["spaces"][0]["difference"]["sda_pct"], 2) == 5.56
    assert data["only_in_a"] == [] and data["only_in_b"] == ["Room Extra"]
    assert data["a"]["case"] == "variant" and data["b"]["case"] == CASE


async def test_compare_takes_case_a_and_case_b_not_case(tmp_path, project):
    result = await call(tool_server(tmp_path), "results", {"action": "compare", **_args(project)})
    assert codes(result) == [("UNKNOWN_ARGUMENT", "case")]


def test_points_cells_become_json_values():
    from leed_mcp.adapters.results_files import _cell
    assert [_cell(t) for t in ("yes", "no", "", "inf", "nan", "3", "3.0", "0.762", "Room North")] == [
        True, False, None, None, None, 3, 3.0, 0.762, "Room North"]


async def test_compare_skips_hand_edited_values_that_are_not_numbers(tmp_path, project):
    run = project / "_run" / "option1"
    shutil.copytree(run / CASE, run / "variant")
    edit_results(project, "option1", lambda p: p["spaces"][0].update(sda_pct="n/a", area_m2=None))
    result = await call(tool_server(tmp_path), "results", {
        "action": "compare", "folder": str(project), "credit": "option1", "case_a": "variant", "case_b": CASE})
    difference = result.structured_content["spaces"][0]["difference"]
    assert "sda_pct" not in difference and "area_m2" not in difference and difference["points"] == 0


def _weather_outside(tmp_path, project):
    """Point project.yaml's weather file at a copy outside the project folder."""
    outside = tmp_path / "outside.wea"
    shutil.copy(project / "weather" / "example.wea", outside)
    text = (project / "project.yaml").read_text()
    (project / "project.yaml").write_text(text.replace("weather/example.wea", str(outside)), newline="\n")
    return outside


async def test_an_input_named_outside_the_folder_is_never_read_and_the_check_is_unchecked(tmp_path, project):
    outside = _weather_outside(tmp_path, project)
    result = await call(tool_server(tmp_path), "results", {"action": "summary", **_args(project)})
    data = result.structured_content
    assert data["inputs"] == "unchecked" and data["changed_inputs"] == []
    assert data["outside_inputs"] == [str(outside.resolve())]
    assert ("PATH_OUTSIDE_ROOTS", "inputs") in codes(result)


def test_a_recorded_input_outside_the_folder_is_not_read(tmp_path):
    from leed_mcp.adapters.results_files import _digest
    folder = (tmp_path / "p")
    folder.mkdir()
    (tmp_path / "secret.txt").write_text("x", newline="\n")
    assert _digest(folder.resolve(), str((tmp_path / "secret.txt").resolve())) is None
    assert _digest(folder.resolve(), "../secret.txt") is None


async def test_a_room_named_by_a_number_stays_text_and_filters(tmp_path, project):
    path = project / "_run" / "option1" / CASE / "points.csv"
    path.write_text(path.read_text().replace("Room North", "101"), newline="\n")
    data = (await call(tool_server(tmp_path), "results",
                       {"action": "points", **_args(project, space="101")})).structured_content
    assert data["total"] == 54 and data["rows"][0]["room"] == "101"


async def test_warnings_held_as_one_text_is_one_warning(tmp_path, project):
    edit_results(project, "option1", lambda p: p.update(warnings="one warning"))
    result = await call(tool_server(tmp_path), "results", {"action": "summary", **_args(project)})
    assert codes(result) == [("PIPELINE_WARNING", "warnings")]
    assert result.structured_content["advisories"][0]["message"] == "one warning"


async def test_a_score_that_is_not_a_number_is_named_in_plain_words(tmp_path, project):
    edit_results(project, "option1", lambda p: p["score"].update(points="ten"))
    result = await call(tool_server(tmp_path), "results", {"action": "summary", **_args(project)})
    assert codes(result) == [("INPUT_INVALID", None)]
    reason = result.structured_content["reason"]
    assert "score.points is not a whole number" in reason and "pydantic" not in reason


async def test_compare_loads_and_fingerprints_the_inputs_once(tmp_path, project, monkeypatch):
    from leed_mcp.adapters import results_files
    calls = []
    real_load, real_prints = results_files.load_project, results_files.input_fingerprints
    monkeypatch.setattr(results_files, "load_project", lambda *a: calls.append("load") or real_load(*a))
    monkeypatch.setattr(results_files, "input_fingerprints",
                        lambda *a: calls.append("prints") or real_prints(*a))
    run = project / "_run" / "option1"
    shutil.copytree(run / CASE, run / "variant")
    await call(tool_server(tmp_path), "results", {
        "action": "compare", "folder": str(project), "credit": "option1", "case_a": "variant", "case_b": CASE})
    assert calls == ["load", "prints"]
