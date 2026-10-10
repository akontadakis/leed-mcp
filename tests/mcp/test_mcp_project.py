# tests/mcp/test_mcp_project.py
"""The project tool (spec 6.1) on invented folders under a temporary root."""
import pytest

from conftest import symlink_or_skip
from deliver_folder import CASE, build_deliver_project
from leed_mcp.adapters import project_state
from leed_mcp.tools import project as project_tool
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import build_option2_project

pytestmark = pytest.mark.anyio


async def test_create_writes_the_template_and_lists_the_fill_me_keys(tmp_path):
    result = await call(tool_server(tmp_path), "project", {"action": "create", "folder": str(tmp_path / "new")})
    data = result.structured_content
    assert result.is_error is False
    assert data["written"] == ["project.yaml", "credit_rules.yaml", "model/", "weather/"]
    assert "scene.units" in data["fill_me"] and "site.latitude_N" in data["fill_me"]
    assert (tmp_path / "new" / "project.yaml").is_file() and (tmp_path / "new" / "model").is_dir()


async def test_create_never_overwrites_a_project(tmp_path):
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "spaces.csv").write_text("x\n")
    result = await call(tool_server(tmp_path), "project", {"action": "create", "folder": str(tmp_path / "p")})
    assert codes(result) == [("PROJECT_EXISTS", "folder")]
    assert not (tmp_path / "p" / "project.yaml").exists()


async def test_create_outside_the_roots_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    result = await call(tool_server(root), "project", {"action": "create", "folder": str(tmp_path / "x")})
    assert codes(result) == [("PATH_OUTSIDE_ROOTS", "folder")]


async def test_create_without_a_parent_folder_is_file_missing(tmp_path):
    result = await call(tool_server(tmp_path), "project", {"action": "create", "folder": str(tmp_path / "a" / "b")})
    assert codes(result) == [("FILE_MISSING", "folder")]


async def test_status_of_a_new_project_names_every_fill_me_and_the_missing_files(tmp_path):
    server = tool_server(tmp_path)
    await call(server, "project", {"action": "create", "folder": str(tmp_path / "new")})
    result = await call(server, "project", {"action": "status", "folder": str(tmp_path / "new")})
    data = result.structured_content
    present = {f["file"]: f["present"] for f in data["files"]}
    assert present == {"project.yaml": True, "materials.csv": False, "spaces.csv": False,
                       "credit_rules.yaml": True, "splits.csv": False, "glare_narrative.md": False}
    assert "scene.units" in data["fill_me"]
    assert data["cases"] is None and data["results"] == []
    assert codes(result) == [("INPUT_INVALID", "cases")]
    assert data["lock"]["state"] == "free" and data["job"]["state"] == "none"
    assert set(data["radiance_missing"]) == {"prepare", "option1", "option2", "views"}
    assert data["radiance_missing"]["prepare"] == []
    assert set(data["report_libraries"]) == {"python-docx", "matplotlib", "Pillow", "cairosvg"}


async def test_status_reports_results_and_whether_their_inputs_changed(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    server = tool_server(tmp_path)
    data = (await call(server, "project", {"action": "status", "folder": str(folder)})).structured_content
    states = {(r["case"], r["credit"]): r["inputs"] for r in data["results"]}
    assert states == {(CASE, "option1"): "current", (CASE, "option2"): "current", (CASE, "views"): "current"}
    with open(folder / "materials.csv", "a") as handle:
        handle.write("\n")
    data = (await call(server, "project", {"action": "status", "folder": str(folder)})).structured_content
    changed = {(r["case"], r["credit"]): r["changed"] for r in data["results"]}
    assert changed[(CASE, "option1")] == ["materials.csv"]


async def test_status_after_a_value_is_set_back_to_fill_me_says_project_yaml_changed(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))
    text = (folder / "project.yaml").read_text()
    (folder / "project.yaml").write_text(text.replace("latitude_N: 42.0", "latitude_N: FILL_ME"))
    data = (await call(tool_server(tmp_path), "project", {"action": "status", "folder": str(folder)})).structured_content
    assert data["fill_me"] == ["site.latitude_N"]
    assert [r["changed"] for r in data["results"] if r["credit"] == "option1"] == [["project.yaml"]]


async def test_list_finds_the_root_s_project_folders(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (tmp_path / "not_a_project").mkdir()
    data = (await call(tool_server(tmp_path), "project", {"action": "list"})).structured_content
    assert [p["folder"] for p in data["projects"]] == [str(folder.resolve())]
    assert data["projects"][0]["fill_me"] == 0 and data["projects"][0]["job"] == "none"
    assert data["projects"][0]["results"] == {"option1": [], "option2": [], "views": []}


async def test_list_is_cut_at_its_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tool, "LIST_MAX", 2)
    for name in "abc":
        (tmp_path / name).mkdir()
        (tmp_path / name / "project.yaml").write_text("scene: {}\n")
    result = await call(tool_server(tmp_path), "project", {"action": "list"})
    assert len(result.structured_content["projects"]) == 2 and result.structured_content["total"] == 3
    assert codes(result) == [("RESULT_TRUNCATED", "projects")]


async def test_list_takes_no_folder(tmp_path):
    result = await call(tool_server(tmp_path), "project", {"action": "list", "folder": str(tmp_path)})
    assert codes(result) == [("UNKNOWN_ARGUMENT", "folder")]


async def test_files_lists_kinds_skips_run_and_pages(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))
    (folder / "drawing.PNG").write_bytes(b"x")
    symlink_or_skip(folder / "link.rad", folder / "scene.rad")
    server = tool_server(tmp_path)
    data = (await call(server, "project", {"action": "files", "folder": str(folder)})).structured_content
    kinds = {f["path"]: (f["kind"], f["link"]) for f in data["files"]}
    assert kinds["scene.rad"] == ("scene", False) and kinds["weather/example.wea"] == ("weather", False)
    assert kinds["drawing.PNG"] == ("drawing", False) and kinds["link.rad"] == ("scene", True)
    assert not any(path.startswith("_run/") for path in kinds)
    page = await call(server, "project", {"action": "files", "folder": str(folder), "offset": 1, "limit": 2})
    assert len(page.structured_content["files"]) == 2 and page.structured_content["total"] == data["total"]
    assert codes(page) == [("RESULT_TRUNCATED", "files")]


async def test_files_limit_above_500_is_input_invalid(tmp_path):
    result = await call(tool_server(tmp_path), "project", {"action": "files", "folder": str(tmp_path), "limit": 501})
    assert codes(result) == [("INPUT_INVALID", "limit")]


def test_folder_in_repository_is_refused_before_anything_is_written():
    from leed_mcp.paths import REPO_ROOT
    result = project_tool._create(REPO_ROOT / "a_project")
    assert result.structured_content["advisories"][0]["code"] == "FOLDER_IN_REPOSITORY"
    assert not (REPO_ROOT / "a_project").exists()


@pytest.mark.parametrize("text", ["- a list\n", "scene: [unclosed\n"])
async def test_status_of_an_unreadable_project_yaml_answers_with_an_advisory(tmp_path, text):
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "project.yaml").write_text(text)
    result = await call(tool_server(tmp_path), "project", {"action": "status", "folder": str(tmp_path / "p")})
    assert result.is_error is False and result.structured_content["fill_me"] is None
    assert ("INPUT_INVALID", "fill_me") in codes(result)


def _laughs(levels: int) -> str:
    """A small YAML 'billion laughs': each level holds ten aliases of the one before."""
    lines = ['l0: &l0 ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x"]']
    lines += [f"l{n}: &l{n} [{', '.join([f'*l{n - 1}'] * 10)}]" for n in range(1, levels + 1)]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("text", [_laughs(6), "a: &a [*a]\n"])
async def test_status_of_an_expanding_or_recursive_project_yaml_answers_with_an_advisory(tmp_path, text):
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "project.yaml").write_text(text)
    result = await call(tool_server(tmp_path), "project", {"action": "status", "folder": str(tmp_path / "p")})
    assert result.is_error is False and result.structured_content["fill_me"] is None
    assert ("INPUT_INVALID", "fill_me") in codes(result)


async def test_status_never_reads_an_input_named_outside_the_folder(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))
    outside = tmp_path / "outside.wea"
    outside.write_bytes((folder / "weather" / "example.wea").read_bytes())
    text = (folder / "project.yaml").read_text()
    (folder / "project.yaml").write_text(text.replace("weather/example.wea", str(outside)))
    result = await call(tool_server(tmp_path), "project", {"action": "status", "folder": str(folder)})
    (state,) = [r for r in result.structured_content["results"] if r["credit"] == "option1"]
    assert state["inputs"] == "unchecked" and state["outside"] == [str(outside.resolve())]
    assert ("PATH_OUTSIDE_ROOTS", "results") in codes(result)


def _link_out(tmp_path, folder, name, text):
    """Replace ``folder/name`` with a link to a file outside every root."""
    outside = tmp_path.parent / f"{tmp_path.name}_outside_{name}"
    outside.write_text(text)
    (folder / name).unlink(missing_ok=True)
    symlink_or_skip(folder / name, outside)


async def test_status_never_reads_a_project_yaml_or_spaces_csv_that_is_a_link(tmp_path):
    root = tmp_path / "root"
    (root / "p").mkdir(parents=True)
    _link_out(tmp_path, root / "p", "project.yaml", "marker_key: FILL_ME\nother: [unclosed\n")
    _link_out(tmp_path, root / "p", "spaces.csv", "marker_column\n")
    result = await call(tool_server(root), "project", {"action": "status", "folder": str(root / "p")})
    assert result.is_error is False and result.structured_content["fill_me"] is None
    assert "marker" not in str(result.structured_content) and "unclosed" not in str(result.structured_content)
    assert ("INPUT_INVALID", "fill_me") in codes(result) and ("INPUT_INVALID", "cases") in codes(result)


async def test_a_programming_error_while_reading_project_yaml_is_not_hidden(tmp_path, monkeypatch):
    def broken(*args):
        raise TypeError("a bug")
    monkeypatch.setattr(project_tool, "fill_me_keys", broken)
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "project.yaml").write_text("scene: {}\n")
    result = await call(tool_server(tmp_path), "project", {"action": "status", "folder": str(tmp_path / "p")})
    assert codes(result) == [("INTERNAL_ERROR", None)]


async def test_status_loads_and_fingerprints_the_inputs_once(tmp_path, write_example, write_wea, monkeypatch):
    from leed_mcp.adapters import results_files
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    calls = []
    real_load, real_prints = results_files.load_project, results_files.input_fingerprints
    monkeypatch.setattr(results_files, "load_project", lambda *a: calls.append("load") or real_load(*a))
    monkeypatch.setattr(results_files, "input_fingerprints",
                        lambda *a: calls.append("prints") or real_prints(*a))
    data = (await call(tool_server(tmp_path), "project", {"action": "status", "folder": str(folder)})).structured_content
    assert len(data["results"]) == 3 and calls == ["load", "prints"]


async def test_an_input_that_cannot_be_read_counts_as_changed(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))
    (folder / "materials.csv").unlink()
    data = (await call(tool_server(tmp_path), "project", {"action": "status", "folder": str(folder)})).structured_content
    (state,) = [r for r in data["results"] if r["credit"] == "option1"]
    assert state["inputs"] == "changed" and state["changed"] == ["materials.csv"]


async def test_files_stops_listing_at_the_cap_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(project_state, "FILES_MAX", 3)
    for n in range(5):
        (tmp_path / f"f{n}.txt").write_text("x")
    result = await call(tool_server(tmp_path), "project", {"action": "files", "folder": str(tmp_path)})
    assert [f["path"] for f in result.structured_content["files"]] == ["f0.txt", "f1.txt", "f2.txt"]
    assert codes(result) == [("RESULT_TRUNCATED", "files")]


async def test_files_skips_an_entry_that_vanishes_mid_walk(tmp_path, monkeypatch):
    (tmp_path / "kept.txt").write_text("x")
    real_walk = project_state.os.walk

    def walk(*args, **kwargs):
        for top, dirs, names in real_walk(*args, **kwargs):
            yield top, dirs, [*names, "gone.txt"]

    monkeypatch.setattr(project_state.os, "walk", walk)
    result = await call(tool_server(tmp_path), "project", {"action": "files", "folder": str(tmp_path)})
    assert result.is_error is False
    assert [f["path"] for f in result.structured_content["files"]] == ["kept.txt"]


@pytest.mark.parametrize("name", ["project.yaml", "credit_rules.yaml", "model", "weather"])
async def test_create_refuses_a_dangling_link_and_writes_nothing_through_it(tmp_path, name):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "p").mkdir()
    symlink_or_skip(root / "p" / name, outside / "target")
    result = await call(tool_server(root), "project", {"action": "create", "folder": str(root / "p")})
    assert result.is_error is True and codes(result)[0][0] in {"PROJECT_EXISTS", "INPUT_INVALID"}
    assert list(outside.iterdir()) == []
    assert [p.name for p in (root / "p").iterdir()] == [name]


def test_folder_in_repository_is_refused_when_only_the_case_differs():
    from pathlib import Path

    from leed_mcp.paths import REPO_ROOT
    swapped = Path(str(REPO_ROOT).swapcase())
    if not swapped.exists():
        pytest.skip("the file system tells names that differ in case apart")
    result = project_tool._create(swapped / "a_project")
    assert result.structured_content["advisories"][0]["code"] == "FOLDER_IN_REPOSITORY"
    assert not (REPO_ROOT / "a_project").exists()
