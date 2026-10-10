# tests/test_project_materials.py
"""Task 3.3: materials.csv -- every modifier that reaches the scene gets a
role, and `glazing_interior` can never be mistaken for exterior glazing.

Both entry points (`load_materials`, `rows`) walk the same parse-and-validate
path, so a file that is bad in one way fails the same way for both.
"""
from __future__ import annotations

import csv
import pathlib
import re

import pytest

from leed_daylight.project.config import ProjectError
from leed_daylight.project.materials import (
    EXTERIOR_GLAZING_ROLES,
    ROLES,
    MaterialRow,
    declared_types,
    vlt_values,
    load_materials,
    rows,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "materials_example.csv"

HEADER = ("modifier", "primitive_type", "role", "note")


def _write(tmp_path: pathlib.Path, data_rows, header=HEADER, name="materials.csv") -> pathlib.Path:
    """Write a small materials.csv with the given header and data rows."""
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_the_example_file_loads():
    materials = load_materials(FIXTURE)

    assert len(materials) == 11
    assert materials["glass_clear"] == "glazing"
    assert materials["glass_curtain"] == "glazing_curtain_wall"
    assert materials["glass_partition"] == "glazing_interior"
    assert materials["hidden_helper"] == "ignore"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_declared_types_returns_every_non_blank_primitive_type(tmp_path):
    path = _write(tmp_path, [
        ("paint_a", "plastic", "wall", ""),
        ("glass_a", "glass", "glazing", ""),
        ("helper_a", "", "ignore", ""),
    ])
    assert declared_types(path) == {"paint_a": "plastic", "glass_a": "glass"}


def test_a_blank_role_is_refused(tmp_path):
    path = _write(tmp_path, [
        ("glass_clear", "glass", "glazing", ""),
        ("wall_paint", "plastic", "", ""),
    ])
    with pytest.raises(ProjectError, match=r"line 3.*wall_paint.*no role"):
        load_materials(path)


def test_ignore_is_an_accepted_role(tmp_path):
    path = _write(tmp_path, [("hidden_helper", "plastic", "ignore", "explicit exclusion")])

    materials = load_materials(path)

    assert materials["hidden_helper"] == "ignore"


def test_an_unknown_role_is_refused(tmp_path):
    path = _write(tmp_path, [("glass_clear", "glass", "windw", "")])

    with pytest.raises(ProjectError) as excinfo:
        load_materials(path)
    message = str(excinfo.value)
    assert "glass_clear" in message
    assert "windw" in message
    for role in sorted(ROLES):
        assert role in message


def test_a_duplicate_modifier_is_refused(tmp_path):
    path = _write(tmp_path, [
        ("glass_clear", "glass", "glazing", ""),
        ("glass_clear", "glass", "wall", ""),
    ])

    with pytest.raises(ProjectError, match=r"glass_clear.*line 2.*line 3"):
        load_materials(path)


def test_interior_glazing_is_not_exterior_glazing():
    assert "glazing_interior" not in EXTERIOR_GLAZING_ROLES


def test_a_missing_column_is_refused(tmp_path):
    path = _write(
        tmp_path,
        [("glass_clear", "glass", "glazing")],
        header=("modifier", "primitive_type", "role"),
    )

    with pytest.raises(ProjectError, match="note"):
        load_materials(path)


def test_notes_survive(tmp_path):
    path = _write(tmp_path, [("glass_clear", "glass", "glazing", "invented note for the report")])

    result = rows(path)

    assert result == [MaterialRow(
        modifier="glass_clear",
        primitive_type="glass",
        role="glazing",
        note="invented note for the report",
    )]


# ---------------------------------------------------------------------------
# Additional refusals settled by the controller rulings, beyond the table
# ---------------------------------------------------------------------------

def test_an_extra_column_is_refused(tmp_path):
    path = _write(
        tmp_path,
        [("glass_clear", "glass", "glazing", "", "extra")],
        header=("modifier", "primitive_type", "role", "note", "extra_col"),
    )

    with pytest.raises(ProjectError, match="extra_col"):
        load_materials(path)


def test_a_missing_file_is_refused(tmp_path):
    path = tmp_path / "does_not_exist.csv"

    with pytest.raises(ProjectError, match=re.escape(str(path))):
        load_materials(path)


def test_a_header_only_file_is_refused(tmp_path):
    path = _write(tmp_path, [])

    with pytest.raises(ProjectError, match="no modifiers"):
        load_materials(path)


def test_a_blank_modifier_is_refused(tmp_path):
    path = _write(tmp_path, [("", "glass", "glazing", "")])

    with pytest.raises(ProjectError, match="line 2"):
        load_materials(path)


def test_rows_and_load_materials_agree_on_size():
    assert len(rows(FIXTURE)) == len(load_materials(FIXTURE))


def test_roles_and_exterior_glazing_roles_shape():
    assert EXTERIOR_GLAZING_ROLES <= ROLES
    assert ROLES == frozenset({
        "glazing", "glazing_curtain_wall", "glazing_diffusing", "glazing_interior",
        "wall", "ceiling", "floor", "fixed_shading",
        "context", "ground", "furniture", "ignore",
    })


# ---------------------------------------------------------------------------
# Fix round 1: physical line numbers, and refusing unreadable/malformed input
# ---------------------------------------------------------------------------

def test_a_blank_line_does_not_shift_later_line_numbers(tmp_path):
    # Header (1), a good row (2), a blank physical line (3), a bad row (4).
    # A record count would report the bad row as "line 3"; the file line is 4.
    path = tmp_path / "materials.csv"
    path.write_text(
        "modifier,primitive_type,role,note\n"
        "glass_clear,glass,glazing,\n"
        "\n"
        "wall_paint,plastic,,\n",
        encoding="utf-8",
    newline="\n")

    with pytest.raises(ProjectError, match=r"line 4.*wall_paint.*no role"):
        load_materials(path)


def test_a_multiline_quoted_note_does_not_shift_later_line_numbers(tmp_path):
    # The note on line 2 spans two physical lines (2-3) because it is
    # quoted with an embedded newline; the next row starts on line 4.
    path = tmp_path / "materials.csv"
    path.write_text(
        "modifier,primitive_type,role,note\n"
        'glass_clear,glass,glazing,"line one\nline two"\n'
        "wall_paint,plastic,,\n",
        encoding="utf-8",
    newline="\n")

    with pytest.raises(ProjectError, match=r"line 4.*wall_paint.*no role"):
        load_materials(path)


def test_a_directory_path_is_refused(tmp_path):
    directory = tmp_path / "materials.csv"
    directory.mkdir()

    with pytest.raises(ProjectError, match=re.escape(str(directory))):
        load_materials(directory)


def test_a_utf8_bom_does_not_cause_a_false_missing_column(tmp_path):
    path = tmp_path / "materials.csv"
    path.write_bytes(
        b"\xef\xbb\xbf" + b"modifier,primitive_type,role,note\nglass_clear,glass,glazing,\n"
    )

    materials = load_materials(path)

    assert materials["glass_clear"] == "glazing"


def test_a_row_with_more_fields_than_the_header_is_refused(tmp_path):
    path = tmp_path / "materials.csv"
    path.write_text(
        "modifier,primitive_type,role,note\n"
        "glass_clear,glass,glazing,,surplus_value\n",
        encoding="utf-8",
    newline="\n")

    with pytest.raises(ProjectError, match=r"line 2.*5.*surplus_value"):
        load_materials(path)


def test_a_row_with_fewer_fields_than_the_header_is_a_named_refusal(tmp_path):
    path = tmp_path / "materials.csv"
    path.write_text(
        "modifier,primitive_type,role,note\n"
        "glass_clear,glass,glazing\n",
        encoding="utf-8",
    newline="\n")

    with pytest.raises(ProjectError, match="line 2"):
        load_materials(path)


# ---------------------------------------------------------------------------
# Excel-saved files: cp1252 bytes and ';' as the delimiter
# ---------------------------------------------------------------------------


def test_a_non_utf8_file_is_refused_by_name(tmp_path):
    path = tmp_path / "materials.csv"
    # "Café" in cp1252: 0xE9 is not valid UTF-8 on its own.
    path.write_bytes(
        "modifier,primitive_type,role,note\nwall_paint,plastic,wall,Caf\xe9 wall\n".encode("cp1252")
    )
    with pytest.raises(ProjectError) as excinfo:
        load_materials(path)
    message = str(excinfo.value)
    assert "materials.csv" in message and "UTF-8" in message and "line 2" in message


def test_a_semicolon_delimited_file_is_refused_as_such(tmp_path):
    path = tmp_path / "materials.csv"
    path.write_text("modifier;primitive_type;role;note\nwall_paint;plastic;wall;\n", newline="\n")
    with pytest.raises(ProjectError) as excinfo:
        load_materials(path)
    message = str(excinfo.value)
    assert "';'" in message and "comma" in message
    assert "missing column" not in message


# ---------------------------------------------------------------------------
# glazing_diffusing (skylights spec 4.1): exterior glass that scatters light
# ---------------------------------------------------------------------------

def test_glazing_diffusing_is_an_exterior_glazing_role():
    assert "glazing_diffusing" in ROLES
    assert EXTERIOR_GLAZING_ROLES == {"glazing", "glazing_curtain_wall", "glazing_diffusing"}


def test_a_diffusing_glazing_row_loads(tmp_path):
    path = _write(tmp_path, [("panel_diffuse", "trans", "glazing_diffusing", "invented panel")])

    assert load_materials(path) == {"panel_diffuse": "glazing_diffusing"}


_VLT_BASE = "modifier,primitive_type,role,note"


def _vlt_write(tmp_path, text):
    path = pathlib.Path(tmp_path) / "materials.csv"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.mark.parametrize("text", [
    f"{_VLT_BASE},vlt\nglass_a,glass,glazing,,62\nwall_a,plastic,wall,,\n",
    f"{_VLT_BASE},view,vlt\nglass_a,glass,glazing,,clear,62\nwall_a,plastic,wall,,,\n",
])
def test_vlt_is_read_in_either_column_layout(tmp_path, text):
    path = _vlt_write(tmp_path, text)
    assert vlt_values(path) == {"glass_a": 62.0}
    assert load_materials(path)["glass_a"] == "glazing"


@pytest.mark.parametrize("cell, words", [("140", "from 0 to 100"), ("abc", "a number"), ("-1", "from 0 to 100")])
def test_a_bad_vlt_is_refused(tmp_path, cell, words):
    path = _vlt_write(tmp_path, f"{_VLT_BASE},vlt\nglass_a,glass,glazing,,{cell}\n")
    with pytest.raises(ProjectError, match=words):
        load_materials(path)


def test_vlt_on_a_wall_is_refused(tmp_path):
    path = _vlt_write(tmp_path, f"{_VLT_BASE},vlt\nwall_a,plastic,wall,,50\n")
    with pytest.raises(ProjectError, match="vlt is allowed only on"):
        load_materials(path)


def test_a_vlt_between_0_and_1_is_warned_as_a_probable_fraction(tmp_path):
    from leed_daylight.project.materials import vlt_warnings

    path = _write(tmp_path, [("glass_a", "glass", "glazing", "", "0.62"), ("glass_b", "glass", "glazing", "", "62")],
                  header=HEADER + ("vlt",))

    (warning,) = vlt_warnings(path)
    assert "glass_a" in warning and "percentage" in warning and "62" in warning
    assert "glass_b" not in warning
