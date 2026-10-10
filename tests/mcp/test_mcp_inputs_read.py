# tests/mcp/test_mcp_inputs_read.py
"""inputs(action="read") on every file (spec 6.3)."""
import hashlib

import pytest

from conftest import symlink_or_skip
from leed_daylight.deliver.narrative import TEMPLATE
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from merge_split_folder import build_split_project
from option2_folder import SPACES_CSV, build_option2_project

pytestmark = pytest.mark.anyio


async def _read(tmp_path, folder, file, **extra):
    return await call(tool_server(tmp_path), "inputs", {"action": "read", "folder": str(folder), "file": file,
                                                       **extra})


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def test_read_project_gives_content_fill_me_comments_and_suggestions(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea,
                                   mutate=lambda d: d["site"].update(latitude_N="FILL_ME"))
    with open(folder / "project.yaml", "a") as handle:
        handle.write("# checked with the architect\n")
    result = await _read(tmp_path, folder, "project")
    data = result.structured_content
    assert result.is_error is False and codes(result) == []
    assert data["sha256"] == _sha(folder / "project.yaml") and data["path"] == str(folder / "project.yaml")
    assert data["content"]["scene"] == {"units": "m", "files": ["scene.rad"]}
    assert data["fill_me"] == ["site.latitude_N"]
    assert data["hand_comments"] == ["# checked with the architect"]
    assert len(data["suggestions"]) == 13
    assert {"key": "grid.spacing_m", "value": 0.6, "source": "LEED v4.1: at most 0.60 m"} in data["suggestions"]


async def test_read_a_file_that_does_not_exist_gives_null(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    result = await _read(tmp_path, folder, "splits")
    data = result.structured_content
    assert result.is_error is False
    assert (data["sha256"], data["content"], data["path"]) == (None, None, str(folder / "splits.csv"))


async def test_read_materials_gives_every_cell_as_text(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    data = (await _read(tmp_path, folder, "materials")).structured_content
    assert data["content"]["columns"] == ["modifier", "primitive_type", "role", "note"]
    assert data["content"]["rows"][0] == {"modifier": "wall_paint", "primitive_type": "plastic", "role": "wall",
                                          "note": ""}


async def test_a_room_named_like_a_number_stays_text(tmp_path, write_example, write_wea):
    spaces = SPACES_CSV.replace("Room North", "101")
    folder = build_option2_project(tmp_path, write_example, write_wea, spaces_csv=spaces)
    data = (await _read(tmp_path, folder, "spaces")).structured_content
    row = data["content"]["rows"][0]
    assert row["room"] == "101" and row["area_m2"] == "24" and row["include_base_case"] == "1"


async def test_read_spaces_with_a_case_keeps_its_rows_and_refuses_an_unknown_case(tmp_path, write_example,
                                                                                   write_wea):
    spaces = SPACES_CSV + "L0,Room South,office,yes,10,3.0,6.0,0.0,no,,0,\n"
    folder = build_option2_project(tmp_path, write_example, write_wea, spaces_csv=spaces)
    data = (await _read(tmp_path, folder, "spaces", case="base_case")).structured_content
    assert [r["room"] for r in data["content"]["rows"]] == ["Room North"]
    unknown = await _read(tmp_path, folder, "spaces", case="other")
    assert codes(unknown) == [("CASE_UNKNOWN", "case")]
    assert "its cases are base_case" in unknown.structured_content["reason"]


async def test_case_is_for_spaces_only(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    assert codes(await _read(tmp_path, folder, "materials", case="base_case")) == [("UNKNOWN_ARGUMENT", "case")]


async def test_a_file_that_cannot_be_parsed_still_gives_its_sha256(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / "materials.csv").write_bytes("modifier,primitive_type,role,note\nwall_p\xe4int,plastic,wall,\n"
                                           .encode("cp1252"))
    result = await _read(tmp_path, folder, "materials")
    data = result.structured_content
    assert result.is_error is False and data["content"] is None
    assert data["sha256"] == _sha(folder / "materials.csv")
    assert codes(result) == [("INPUT_INVALID", "content")]
    assert result.structured_content["advisories"][0]["severity"] == "warn"


async def test_a_spaces_csv_the_loader_refuses_gives_its_rows_and_the_problem(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea,
                                   spaces_csv=SPACES_CSV.replace(",office,yes,24,", ",office,maybe,24,"))
    result = await _read(tmp_path, folder, "spaces")
    assert result.structured_content["content"]["rows"][0]["regularly_occupied"] == "maybe"
    assert codes(result) == [("INPUT_INVALID", "content")]
    assert "spaces.csv line 2" in result.structured_content["advisories"][0]["message"]


async def test_read_splits_gives_the_lines(tmp_path, write_example, write_wea):
    folder = build_split_project(tmp_path, write_example, write_wea)
    data = (await _read(tmp_path, folder, "splits")).structured_content
    assert data["content"] == {"lines": [{"level_z": 0.0, "points": [[4.5, -1.0], [4.5, 5.0]],
                                          "note": "invented west edge"}]}


async def test_read_credit_rules_and_refuse_one_named_outside(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    data = (await _read(tmp_path, folder, "credit_rules")).structured_content
    assert data["content"]["max_points"] == 3 and data["path"] == str(folder / "credit_rules.yaml")
    (tmp_path / "b").mkdir()
    outside = build_option2_project(tmp_path / "b", write_example, write_wea,
                                    mutate=lambda d: d["standard"].update(credit_rules="../rules.yaml"))
    assert codes(await _read(tmp_path, outside, "credit_rules")) == [("PATH_OUTSIDE_ROOTS", "file")]


async def test_read_the_glare_narrative_lists_measures_and_the_marks_outside_the_auto_blocks(
        tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    table = ("| Storey | Space | Area m2 | ASE | sDA | Glare measures |\n|---|---|---:|---:|---:|---|\n"
             "| L0 | Room North | 24.0 | 20.0 % | 60.0 % | [ADD] |\n")
    text = TEMPLATE.replace("<!-- auto:spaces -->\n", "<!-- auto:spaces -->\n" + table)
    (folder / "glare_narrative.md").write_text(text)
    data = (await _read(tmp_path, folder, "glare_narrative")).structured_content
    assert data["content"]["spaces"] == [{"space": "Room North", "measures": "[ADD]"}]
    assert [m["index"] for m in data["content"]["marks"]] == [1, 2, 3]
    assert data["content"]["marks"][0]["mark"].startswith("[CONFIRM that the spaces")
    assert data["content"]["marks"][1]["line"].startswith("[ADD the blinds or shades")
    assert data["marks_left"] == 4


async def test_a_yaml_date_nan_and_number_key_read_as_json(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    with open(folder / "project.yaml", "a") as handle:
        handle.write("extra:\n  when: 2026-03-21\n  odd: .nan\n  7: seven\n")
    data = (await _read(tmp_path, folder, "project")).structured_content
    assert data["content"]["extra"] == {"when": "2026-03-21", "odd": None, "7": "seven"}


async def test_a_linked_input_file_is_not_read(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (tmp_path / "real.csv").write_text((folder / "materials.csv").read_text())
    (folder / "materials.csv").unlink()
    symlink_or_skip(folder / "materials.csv", tmp_path / "real.csv")
    assert codes(await _read(tmp_path, folder, "materials")) == [("PATH_UNSAFE", None)]


async def test_read_without_a_file_is_argument_missing(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    result = await call(tool_server(tmp_path), "inputs", {"action": "read", "folder": str(folder)})
    assert codes(result) == [("ARGUMENT_MISSING", "file")]


@pytest.mark.parametrize("text", [
    TEMPLATE,
    TEMPLATE.replace("<!-- /auto:spaces -->\n", ""),
    TEMPLATE + "<!-- auto:results -->\n",
    "intro [ADD x]\n<!-- auto:spaces --><!-- /auto:spaces -->\n[CONFIRM y] [ADDED no]\n",
    "| L0 | A \\| B | 1.0 | 2.0 % | 3.0 % | blinds | east |\n|---|---|---:|---:|---:|---|\n| Storey | Space |\n",
])
def test_the_ported_narrative_rules_equal_the_pipelines(text):
    from leed_daylight.deliver import narrative
    from leed_mcp.adapters import narrative_file as ported
    assert ported.count_marks(text) == narrative.count_marks(text)
    assert ported.measures(text) == narrative._measures(text)
    for line in text.splitlines():
        assert ported.split_row(line, 5) == narrative._split_row(line, 5)
    for name in ported.AUTO_BLOCKS:
        assert ported.block_state(text, name) == narrative._block_state(text, name)
        if ported.block_state(text, name) == "well_formed":
            assert ported.body_span(text, name) == narrative._body_span(text, name)


def _laughs(levels: int = 9) -> str:
    lines = ['l0: &l0 ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x"]']
    lines += [f"l{n}: &l{n} [{', '.join([f'*l{n - 1}'] * 10)}]" for n in range(1, levels + 1)]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("file, name, text, words", [
    ("project", "project.yaml", _laughs(), "more than 100000 nodes"),
    ("project", "project.yaml", "a: &a [*a]\n", "recursive alias"),
    ("credit_rules", "credit_rules.yaml", _laughs(), "more than 100000 nodes"),
    ("credit_rules", "credit_rules.yaml", "a: &a {b: *a}\n", "recursive alias"),
])
async def test_an_expanding_or_recursive_yaml_reads_as_null_with_a_warning(tmp_path, write_example, write_wea,
                                                                            file, name, text, words):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / name).write_text(text)
    result = await _read(tmp_path, folder, file)
    data = result.structured_content
    assert result.is_error is False and data["content"] is None and data["sha256"] == _sha(folder / name)
    assert codes(result) == [("INPUT_INVALID", "content")]
    assert data["advisories"][0]["severity"] == "warn" and words in data["advisories"][0]["message"]


async def test_a_yaml_binary_and_set_read_as_json(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    with open(folder / "project.yaml", "a") as handle:
        handle.write("extra:\n  blob: !!binary AAEC\n  tags: !!set {b: null, a: null}\n")
    result = await _read(tmp_path, folder, "project")
    assert result.is_error is False
    assert result.structured_content["content"]["extra"] == {"blob": None, "tags": ["a", "b"]}


@pytest.mark.parametrize("text", ["", "# only a comment\n", "~\n", "null\n"])
async def test_an_empty_project_yaml_reads_as_null_with_a_warning(tmp_path, write_example, write_wea, text):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / "project.yaml").write_text(text)
    result = await _read(tmp_path, folder, "project")
    data = result.structured_content
    assert result.is_error is False and data["content"] is None and data["sha256"] == _sha(folder / "project.yaml")
    assert codes(result) == [("INPUT_INVALID", "content")]
    assert data["advisories"][0] == {"code": "INPUT_INVALID", "severity": "warn", "field": "content",
                                     "message": "project.yaml is empty; it must be a mapping at the top level"}
