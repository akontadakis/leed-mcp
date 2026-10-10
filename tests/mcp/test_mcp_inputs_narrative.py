# tests/mcp/test_mcp_inputs_narrative.py
"""inputs(action="update", file="glare_narrative") (spec 6.3)."""
import hashlib

import pytest

from leed_daylight.deliver.narrative import TEMPLATE, _measures
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture

pytestmark = pytest.mark.anyio

TABLE = ("| Storey | Space | Area m2 | ASE | sDA | Glare measures |\n|---|---|---:|---:|---:|---|\n"
         "| L0 | Room North | 24.0 | 20.0 % | 60.0 % | [ADD] |\n"
         "| L0 | Room A \\| B | 12.0 | 15.0 % | 55.0 % | [ADD] |\n")
TEXT = TEMPLATE.replace("<!-- auto:spaces -->\n", "<!-- auto:spaces -->\n" + TABLE)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def folder(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    (folder / "glare_narrative.md").write_text(TEXT)
    return folder


async def _update(tmp_path, folder, **change):
    return await call(tool_server(tmp_path), "inputs", {
        "action": "update", "folder": str(folder), "file": "glare_narrative",
        "expected_sha256": _sha(folder / "glare_narrative.md"), **change})


async def test_measures_replace_the_sixth_cell_and_report_carries_it(tmp_path, folder):
    result = await _update(tmp_path, folder, measures={"Room A | B": "manual roller blinds | east facade"})
    text = (folder / "glare_narrative.md").read_text()
    assert result.is_error is False and codes(result) == []
    assert "| L0 | Room A \\| B | 12.0 | 15.0 % | 55.0 % | manual roller blinds \\| east facade |\n" in text
    assert text.replace("manual roller blinds \\| east facade", "[ADD]") == TEXT
    start = text.index("<!-- auto:spaces -->")
    assert _measures(text[start:])["Room A | B"] == "manual roller blinds \\| east facade"
    assert result.structured_content["marks_left"] == 4


async def test_prose_replaces_one_mark_and_nothing_else(tmp_path, folder):
    result = await _update(tmp_path, folder, prose={"2": "External fins on the south facade (invented)."})
    text = (folder / "glare_narrative.md").read_text()
    assert result.is_error is False
    assert "[ADD the blinds or shades" not in text and "External fins on the south facade (invented)." in text
    assert [m["index"] for m in result.structured_content["content"]["marks"]] == [1, 2]
    assert result.structured_content["marks_left"] == 4


@pytest.mark.parametrize("change, field, words", [
    ({"measures": {"Room South": "blinds"}}, "measures", "'Room South' not in the auto:spaces table"),
    ({"measures": {"Room North": "line one\nline two"}}, "measures", "one line"),
    ({"measures": {"Room North": "<!-- /auto:spaces -->"}}, "measures", "auto-block marker"),
    ({"prose": {"9": "x"}}, "prose", "the marks outside the auto blocks are 1 to 3"),
    ({"prose": {"one": "x"}}, "prose", "not a mark index"),
    ({"prose": {"1": "ok <!-- auto:results -->"}}, "prose", "auto-block marker"),
    ({"prose": {"1": "first", "01": "second"}}, "prose", "name the same mark"),
    ({"prose": {"\u0661": "x"}}, "prose", "not a mark index"),
    ({"prose": {"\u00b2": "x"}}, "prose", "not a mark index"),
    ({"prose": {"1": "nul \x00 here"}}, "prose", "control character"),
    ({"prose": {"1": "carriage \r return"}}, "prose", "control character"),
    ({"measures": {"Room North": "soft\x0breturn"}}, "measures", "one line"),
    ({"measures": {"Room North": "line\u2028separator"}}, "measures", "one line"),
    ({"measures": {"Room North": "next\x85line"}}, "measures", "one line"),
    ({"measures": {"Room North": "group\x1dseparator"}}, "measures", "one line"),
    ({"measures": {"Room North": "nul\x00here"}}, "measures", "one line"),
])
async def test_a_refused_edit_writes_nothing(tmp_path, folder, change, field, words):
    result = await _update(tmp_path, folder, **change)
    assert codes(result) == [("INPUT_INVALID", field)] and words in result.structured_content["reason"]
    assert (folder / "glare_narrative.md").read_text() == TEXT


async def test_measures_need_a_well_formed_spaces_block(tmp_path, folder):
    (folder / "glare_narrative.md").write_text(TEXT.replace("<!-- /auto:spaces -->\n", ""))
    result = await _update(tmp_path, folder, measures={"Room North": "blinds"})
    assert codes(result) == [("INPUT_INVALID", "measures")]


async def test_without_the_file_update_is_file_missing(tmp_path, folder):
    (folder / "glare_narrative.md").unlink()
    result = await call(tool_server(tmp_path), "inputs", {"action": "update", "folder": str(folder),
                                                         "file": "glare_narrative", "prose": {"1": "x"}})
    assert codes(result) == [("FILE_MISSING", "file")]


async def test_prose_keeps_line_breaks_and_tabs_and_a_leading_zero_index(tmp_path, folder):
    result = await _update(tmp_path, folder, prose={"01": "first line\n\tsecond line (invented)"})
    assert result.is_error is False and codes(result) == []
    assert "first line\n\tsecond line (invented)" in (folder / "glare_narrative.md").read_text()


async def test_measures_keep_a_tab(tmp_path, folder):
    result = await _update(tmp_path, folder, measures={"Room North": "blinds\tinvented"})
    assert result.is_error is False and codes(result) == []
