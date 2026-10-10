# tests/test_project_splits.py
"""Reading splits.csv (merge-split spec 3.1).

The file is optional: header exactly ``level_z,points,note``; ``level_z`` in
metres; ``points`` two or more ``x y`` pairs in metres separated by ``;``;
``note`` free text. Row numbers count the header as line 1 and skip nothing:
a blank line is not a row but still counts. Every coordinate is invented and
round.
"""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight.project.config import ProjectError
from leed_daylight.project.splits import (
    SPLIT_TOL_M,
    SPLITS_FILE,
    SPLITS_HEADER,
    SplitError,
    SplitLine,
    load_splits,
    read_splits,
)

HEADER = "level_z,points,note\n"


def _write(tmp_path, text: str | bytes):
    path = tmp_path / SPLITS_FILE
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_the_names_and_values_of_the_contract():
    assert SPLITS_FILE == "splits.csv"
    assert SPLIT_TOL_M == 0.01
    assert SPLITS_HEADER == ("level_z", "points", "note")
    assert issubclass(SplitError, ProjectError)


def test_a_good_file_with_a_straight_and_a_bent_line_is_read(tmp_path):
    path = _write(tmp_path, HEADER + '0.0,"2.0 0.0; 2.0 4.0",east edge\n'
                  '3.5,"12.0 4.0; 12.0 9.5; 18.0 9.5",an invented bent line\n')

    lines = read_splits(path)

    assert [type(line) for line in lines] == [SplitLine, SplitLine]
    first, second = lines
    assert (first.row, first.level_z, first.note) == (2, 0.0, "east edge")
    assert first.points.dtype == np.float64
    assert first.points.tolist() == [[2.0, 0.0], [2.0, 4.0]]
    assert (second.row, second.level_z, second.note) == (3, 3.5, "an invented bent line")
    assert second.points.shape == (3, 2)
    assert second.points.tolist() == [[12.0, 4.0], [12.0, 9.5], [18.0, 9.5]]


def test_cells_and_points_may_carry_spaces(tmp_path):
    path = _write(tmp_path, HEADER + ' 3.5 ," 1 2 ;3   4 ", \n')

    (line,) = read_splits(path)

    assert line.level_z == 3.5
    assert line.points.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert line.note == ""


def test_a_file_with_only_the_header_gives_no_line(tmp_path):
    assert read_splits(_write(tmp_path, HEADER)) == ()


def test_a_utf8_bom_is_stripped(tmp_path):
    path = _write(tmp_path, ("﻿" + HEADER + '0,"0 0; 2 0",\n').encode("utf-8"))

    (line,) = read_splits(path)

    assert line.row == 2


def test_a_cp1252_file_is_refused_by_name(tmp_path):
    path = _write(tmp_path, (HEADER + '0,"0 0; 2 0",caf\xe9\n').encode("cp1252"))

    with pytest.raises(SplitError, match=r"^splits\.csv: .* is not UTF-8 text .*line 2.*CSV UTF-8"):
        read_splits(path)


def test_a_semicolon_header_is_refused(tmp_path):
    path = _write(tmp_path, "level_z;points;note\n")

    with pytest.raises(SplitError, match=r"^splits\.csv: .*uses ';' as the delimiter"):
        read_splits(path)


def test_an_empty_file_is_refused(tmp_path):
    with pytest.raises(SplitError, match=r"^splits\.csv: empty file; expected the header level_z,points,note$"):
        read_splits(_write(tmp_path, ""))


def test_a_wrong_header_is_refused(tmp_path):
    path = _write(tmp_path, "level,points,note\n")

    with pytest.raises(SplitError, match=(
        r"^splits\.csv: the header must be exactly level_z,points,note, got level,points,note$"
    )):
        read_splits(path)


def test_a_row_with_the_points_unquoted_is_refused_with_the_field_count(tmp_path):
    path = _write(tmp_path, HEADER + "0,0 0, 2 0,note\n")

    with pytest.raises(SplitError, match=(
        r"^splits\.csv line 2: row has 4 field\(s\), expected 3 \(level_z,points,note\); "
        r"put the points in double quotes"
    )):
        read_splits(path)


@pytest.mark.parametrize("value", ["", "three", "nan", "inf"])
def test_a_level_that_is_not_a_finite_number_is_refused(tmp_path, value):
    path = _write(tmp_path, HEADER + f'{value},"0 0; 2 0",\n')

    with pytest.raises(SplitError, match=(
        rf"^splits\.csv line 2: level_z {value!r} is not a finite number; "
        r"give the storey's floor level in metres$"
    )):
        read_splits(path)


@pytest.mark.parametrize("point", ["2", "2 0 1", "2 x", "2 nan", ""])
def test_a_point_that_is_not_two_finite_numbers_is_refused(tmp_path, point):
    path = _write(tmp_path, HEADER + f'0,"0 0; {point}",\n')

    with pytest.raises(SplitError, match=(
        rf"^splits\.csv line 2: point 2 {point!r} is not two finite numbers; "
        r"write each point as 'x y' in metres, points separated by ';'$"
    )):
        read_splits(path)


def test_a_line_of_one_point_is_refused(tmp_path):
    path = _write(tmp_path, HEADER + '0,"2 0",\n')

    with pytest.raises(SplitError, match=(
        r"^splits\.csv line 2: the line has 1 point; a split line needs two or more points$"
    )):
        read_splits(path)


def test_two_consecutive_points_closer_than_the_tolerance_are_refused(tmp_path):
    path = _write(tmp_path, HEADER + '0,"0 0; 2 0; 2.005 0; 2 4",\n')

    with pytest.raises(SplitError, match=(
        r"^splits\.csv line 2: points 2 and 3 are 0\.005 m apart; consecutive points "
        r"must be at least 0\.01 m apart: remove one of them$"
    )):
        read_splits(path)


def test_line_numbers_count_the_header_and_blank_lines(tmp_path):
    path = _write(tmp_path, HEADER + '0,"0 0; 2 0",first\n\n\n0,"0 1; 2 1",fourth\n0,"5 5",sixth\n')

    with pytest.raises(SplitError, match=r"^splits\.csv line 6: the line has 1 point"):
        read_splits(path)
    good = _write(tmp_path, HEADER + '0,"0 0; 2 0",first\n\n\n0,"0 1; 2 1",fourth\n')
    assert [line.row for line in read_splits(good)] == [2, 5]


def test_no_splits_csv_gives_none(tmp_path):
    assert load_splits(tmp_path) is None


def test_load_splits_reads_the_file_in_the_folder(tmp_path):
    _write(tmp_path, HEADER + '0,"0 0; 2 0",\n')

    (line,) = load_splits(tmp_path)

    assert line.points.tolist() == [[0.0, 0.0], [2.0, 0.0]]
    assert load_splits(str(tmp_path))[0].row == 2
