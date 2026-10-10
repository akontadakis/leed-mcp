# tests/test_project_points.py
"""The optional sensor points file (``grid.points_file``).

``project.yaml`` may name a CSV file of sensor points, header exactly
``space,x,y,z``, metres, one point per row. When it is set the grid is
these points instead of the tool's lattice. This file covers the key and
the reader; ``tests/test_build_grid_points.py`` covers the grid built from
it. Every name and coordinate is invented and round.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from leed_daylight.project.config import ProjectError, check_project_draft, load_project
from leed_daylight.project.inputs import input_fingerprints
from leed_daylight.project.points import POINTS_HEADER, PointsError, SpacePoints, read_points

HEADER = "space,x,y,z\n"


def _write(tmp_path, text: str | bytes, name: str = "points.csv"):
    path = tmp_path / name
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _with_points_file(tmp_path, write_example, value="points.csv", text=HEADER + "office a,1.0,1.0,0.8\n"):
    def mutate(d):
        d["grid"]["points_file"] = value
        return d

    path = write_example(tmp_path, mutate)
    if text is not None:
        (path.parent / "points.csv").write_text(text, encoding="utf-8", newline="\n")
    return path


# -- the key in project.yaml ---------------------------------------------------


def test_the_key_is_optional_and_unset_by_default(example_project_dir):
    project = load_project(example_project_dir / "project.yaml")

    assert project.grid.points_file is None


def test_the_key_is_resolved_against_the_project_folder(tmp_path, write_example):
    path = _with_points_file(tmp_path, write_example)

    project = load_project(path)

    assert project.grid.points_file == path.parent / "points.csv"


def test_a_points_file_that_does_not_exist_is_refused(tmp_path, write_example):
    path = _with_points_file(tmp_path, write_example, value="missing.csv", text=None)

    with pytest.raises(ProjectError, match=r"grid\.points_file: file does not exist"):
        load_project(path)


def test_a_points_file_that_is_not_text_is_refused(tmp_path, write_example):
    path = _with_points_file(tmp_path, write_example, value=12)

    with pytest.raises(ProjectError, match=r"grid\.points_file must be a string"):
        load_project(path)


def test_the_draft_check_lists_a_fill_me_points_file_as_unfinished(tmp_path, write_example):
    path = _with_points_file(tmp_path, write_example, value="FILL_ME")

    assert check_project_draft(path) == ((), ("grid.points_file",))


def _other_inputs(folder):
    for name in ("materials.csv", "spaces.csv"):
        (folder / name).write_text("", encoding="utf-8", newline="\n")


def test_the_points_file_is_an_input_of_the_results(tmp_path, write_example):
    path = _with_points_file(tmp_path, write_example)
    _other_inputs(path.parent)
    before = input_fingerprints(load_project(path), path.parent)

    (path.parent / "points.csv").write_text(HEADER + "office a,2.0,1.0,0.8\n", encoding="utf-8", newline="\n")
    after = input_fingerprints(load_project(path), path.parent)

    assert "points.csv" in before
    assert before["points.csv"] != after["points.csv"]


def test_without_the_key_the_fingerprints_are_as_before(example_project_dir):
    _other_inputs(example_project_dir)
    project = load_project(example_project_dir / "project.yaml")

    prints = input_fingerprints(project, example_project_dir)

    assert sorted(prints) == [
        "credit_rules.yaml", "materials.csv", "project.yaml", "scene.rad", "spaces.csv", "weather/example.wea",
    ]


# -- the reader -----------------------------------------------------------------


def test_the_names_and_values_of_the_contract():
    assert POINTS_HEADER == ("space", "x", "y", "z")
    assert issubclass(PointsError, ProjectError)


def test_a_good_file_is_read_per_space_in_file_order(tmp_path):
    path = _write(tmp_path, HEADER + "office a,1.0,1.0,0.8\nhall,5.0,1.0,0.8\noffice a,2.0,1.0,0.8\n")

    points = read_points(path)

    assert list(points) == ["office a", "hall"]
    office = points["office a"]
    assert isinstance(office, SpacePoints)
    assert office.space == "office a"
    np.testing.assert_array_equal(office.points, [[1.0, 1.0, 0.8], [2.0, 1.0, 0.8]])
    assert office.points.dtype == np.float64
    assert office.lines == (2, 4)
    assert points["hall"].lines == (3,)


def test_cells_may_carry_spaces_and_a_bom_is_stripped(tmp_path):
    path = _write(tmp_path, ("﻿" + "space, x, y, z\n office a , 1.0 , 1.0 , 0.8 \n").encode("utf-8"))

    points = read_points(path)

    np.testing.assert_array_equal(points["office a"].points, [[1.0, 1.0, 0.8]])


def test_a_blank_line_is_skipped_but_still_counted(tmp_path):
    path = _write(tmp_path, HEADER + "\noffice a,1.0,1.0,0.8\n")

    assert read_points(path)["office a"].lines == (3,)


def test_a_file_with_only_the_header_gives_no_point(tmp_path):
    assert read_points(_write(tmp_path, HEADER)) == {}


def test_an_empty_file_is_refused(tmp_path):
    with pytest.raises(PointsError, match="empty file; expected the header space,x,y,z"):
        read_points(_write(tmp_path, ""))


def test_a_wrong_header_is_refused(tmp_path):
    with pytest.raises(PointsError, match="the header must be exactly space,x,y,z; the first line has 4 field"):
        read_points(_write(tmp_path, "zone,x,y,z\n"))


def test_a_row_with_the_wrong_field_count_is_refused_naming_its_line(tmp_path):
    path = _write(tmp_path, HEADER + "office a,1.0,1.0,0.8\noffice a,1.0,1.0\n")

    with pytest.raises(PointsError, match=r"points\.csv line 3: row has 3 field\(s\), expected 4"):
        read_points(path)


@pytest.mark.parametrize("cell", ["", "wide", "nan", "inf"])
def test_a_coordinate_that_is_not_a_finite_number_is_refused(tmp_path, cell):
    path = _write(tmp_path, HEADER + f"office a,1.0,{cell},0.8\n")

    with pytest.raises(PointsError, match=r"points\.csv line 2: y .* is not a finite number"):
        read_points(path)


def test_a_point_with_no_space_is_refused(tmp_path):
    path = _write(tmp_path, HEADER + " ,1.0,1.0,0.8\n")

    with pytest.raises(PointsError, match=r"points\.csv line 2: the space is empty"):
        read_points(path)


#: An invented line that looks like a secret: a refusal must never print it whole.
SYNTHETIC_SECRET = "https://user:INVENTED0SECRET0TOKEN0000000000@example.invalid/x"


def test_a_header_refusal_does_not_quote_the_first_line(tmp_path):
    with pytest.raises(PointsError) as caught:
        read_points(_write(tmp_path, SYNTHETIC_SECRET + "\n"))

    message = str(caught.value)
    assert message == "points.csv: the header must be exactly space,x,y,z; the first line has 1 field(s)"
    assert "INVENTED0SECRET" not in message


def test_a_row_refusal_shows_at_most_a_short_escaped_excerpt(tmp_path):
    path = _write(tmp_path, HEADER + "\x1b]0;" + SYNTHETIC_SECRET + "\n")

    with pytest.raises(PointsError, match=r"points\.csv line 2: row has 1 field\(s\)") as caught:
        read_points(path)

    shown = str(caught.value).rsplit(": ", 1)[1]
    assert len(shown) <= 40
    assert "\x1b" not in str(caught.value)
    assert shown.startswith("'\\x1b]0;")
    assert "example.invalid" not in str(caught.value)


def test_a_number_refusal_shows_at_most_a_short_escaped_excerpt(tmp_path):
    path = _write(tmp_path, HEADER + f"office a,1.0,{SYNTHETIC_SECRET},0.8\n")

    with pytest.raises(PointsError, match=r"points\.csv line 2: y '.*\.\.\. is not a finite number") as caught:
        read_points(path)

    assert "example.invalid" not in str(caught.value)


def test_a_semicolon_header_shows_at_most_a_short_excerpt(tmp_path):
    with pytest.raises(PointsError, match="uses ';' as the delimiter") as caught:
        read_points(_write(tmp_path, SYNTHETIC_SECRET + ";x\n"))

    assert "example.invalid" not in str(caught.value)


def test_a_short_semicolon_header_is_not_quoted_at_all(tmp_path):
    with pytest.raises(PointsError, match="uses ';' as the delimiter") as caught:
        read_points(_write(tmp_path, "TOKEN=INVENTED0SECRET;\n"))

    message = str(caught.value)
    assert "INVENTED0SECRET" not in message
    assert "TOKEN" not in message
    assert "1 ';' on line 1" in message


def test_a_field_over_the_csv_limit_is_a_named_refusal(tmp_path):
    path = _write(tmp_path, HEADER + "office a,1.0,1.0,0.8\n" + "a" * 200_000 + ",1.0,1.0,0.8\n")

    with pytest.raises(PointsError, match=r"points\.csv line 3: cannot be read as CSV: field larger than field limit"):
        read_points(path)


def test_an_exact_duplicate_point_is_refused_naming_both_lines(tmp_path):
    path = _write(tmp_path, HEADER + "office a,1.0,1.0,0.8\nhall,5.0,1.0,0.8\n office a ,1.00,1,0.9\n")

    expected = r"points\.csv line 4: the point x 1\.0, y 1\.0 of space 'office a' is already on line 2"
    with pytest.raises(PointsError, match=expected) as caught:
        read_points(path)

    assert str(caught.value).count("office a") == 1


def test_the_same_position_in_another_space_is_not_a_duplicate(tmp_path):
    path = _write(tmp_path, HEADER + "office a,1.0,1.0,0.8\nhall,1.0,1.0,0.8\n")

    assert list(read_points(path)) == ["office a", "hall"]


def test_a_duplicate_refusal_shows_at_most_a_short_excerpt_of_the_space(tmp_path):
    row = f"{SYNTHETIC_SECRET},1.0,1.0,0.8\n"
    path = _write(tmp_path, HEADER + row + row)

    with pytest.raises(PointsError, match=r"points\.csv line 3: .* is already on line 2") as caught:
        read_points(path)

    assert "example.invalid" not in str(caught.value)


def test_the_most_rows_is_a_module_constant():
    from leed_daylight.project import points

    assert points.POINTS_MAX_ROWS == 1_000_000


def test_a_file_over_the_row_limit_is_refused_naming_the_line(tmp_path, monkeypatch):
    from leed_daylight.project import points

    monkeypatch.setattr(points, "POINTS_MAX_ROWS", 2)
    path = _write(tmp_path, HEADER + "office a,1.0,1.0,0.8\n\noffice a,2.0,1.0,0.8\noffice a,3.0,1.0,0.8\n")

    expected = r"points\.csv line 5: more than 2 points; a points file holds at most 2"
    with pytest.raises(PointsError, match=expected):
        read_points(path)


def test_a_file_at_the_row_limit_is_read(tmp_path, monkeypatch):
    from leed_daylight.project import points

    monkeypatch.setattr(points, "POINTS_MAX_ROWS", 2)
    path = _write(tmp_path, HEADER + "office a,1.0,1.0,0.8\noffice a,2.0,1.0,0.8\n")

    assert read_points(path)["office a"].lines == (2, 3)


def test_a_semicolon_file_is_refused_by_name(tmp_path):
    path = _write(tmp_path, "space;x;y;z\noffice a;1,0;1,0;0,8\n")

    with pytest.raises(PointsError, match="uses ';' as the delimiter"):
        read_points(path)


def test_render_keeps_the_key_on_a_round_trip(tmp_path, write_example):
    from leed_daylight.edit.project_yaml import render

    path = _with_points_file(tmp_path, write_example)
    raw = yaml.safe_load(path.read_text())
    path.write_text(render(raw), encoding="utf-8", newline="\n")

    assert load_project(path).grid.points_file == path.parent / "points.csv"
