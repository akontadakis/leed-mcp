"""A1 references inside spreadsheet formulas (calculator spec 5.1).

``shift_formula`` moves every reference to one sheet below an inserted
block, as Excel does on an insert: rows after ``after_row`` gain ``count``,
absolute (``$``) or not, and a range whose ends straddle the insert grows.
``translate_formula`` moves only the relative parts, as a copy or a fill
does. Text inside double quotes is never touched. A reference is a cell or a
cell range, optionally after ``Sheet!`` or ``'Sheet name'!``; an external
``[n]`` prefix or ``#REF!`` is never matched.
"""
from __future__ import annotations

import re
from collections.abc import Callable

__all__ = ["col_letters", "col_number", "shift_formula", "shift_sqref", "translate_formula"]

_REF = re.compile(
    r"(?<![A-Za-z0-9_.\]!$])"
    r"(?P<sheet>(?:'(?:[^']|'')+'|[A-Za-z_][A-Za-z0-9_.]*)!)?"
    r"(?P<c1>\$?)(?P<col1>[A-Z]{1,3})(?P<r1>\$?)(?P<row1>\d+)"
    r"(?::(?P<c2>\$?)(?P<col2>[A-Z]{1,3})(?P<r2>\$?)(?P<row2>\d+))?"
    r"(?![A-Za-z0-9_(!])"
)


def col_number(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def col_letters(n: int) -> str:
    out = ""
    while n:
        n, rest = divmod(n - 1, 26)
        out = chr(65 + rest) + out
    return out


def _sheet_of(prefix: str | None) -> str | None:
    if not prefix:
        return None
    name = prefix[:-1]
    if name.startswith("'"):
        name = name[1:-1].replace("''", "'")
    return name


def _map_code(formula: str, fn: Callable[[re.Match], str]) -> str:
    """``fn`` on every reference outside string literals."""
    parts = formula.split('"')
    for i in range(0, len(parts), 2):
        parts[i] = _REF.sub(fn, parts[i])
    return '"'.join(parts)


def shift_formula(formula: str, own_sheet: str, sheet: str, after_row: int, count: int) -> str:
    """``formula`` (living on ``own_sheet``) with every reference to ``sheet``
    moved for ``count`` rows inserted after ``after_row``."""
    def row(text: str) -> int:
        n = int(text)
        return n + count if n > after_row else n

    def fix(m: re.Match) -> str:
        if (_sheet_of(m.group("sheet")) or own_sheet) != sheet:
            return m.group(0)
        text = f"{m.group('sheet') or ''}{m.group('c1')}{m.group('col1')}{m.group('r1')}{row(m.group('row1'))}"
        if m.group("col2"):
            text += f":{m.group('c2')}{m.group('col2')}{m.group('r2')}{row(m.group('row2'))}"
        return text

    return _map_code(formula, fix)


def translate_formula(formula: str, drow: int, dcol: int) -> str:
    """``formula`` with its relative parts moved by ``drow`` rows and ``dcol`` columns."""
    def one(c: str, col: str, r: str, row: str) -> str:
        col = col if c else col_letters(col_number(col) + dcol)
        row = row if r else str(int(row) + drow)
        return f"{c}{col}{r}{row}"

    def fix(m: re.Match) -> str:
        text = (m.group("sheet") or "") + one(m.group("c1"), m.group("col1"), m.group("r1"), m.group("row1"))
        if m.group("col2"):
            text += ":" + one(m.group("c2"), m.group("col2"), m.group("r2"), m.group("row2"))
        return text

    return _map_code(formula, fix)


def shift_sqref(sqref: str, after_row: int, count: int) -> str:
    """A space-separated list of cells and ranges (``sqref``, ``ref``) moved
    for ``count`` rows inserted after ``after_row``."""
    return " ".join(shift_formula(part, "", "", after_row, count) for part in sqref.split())
