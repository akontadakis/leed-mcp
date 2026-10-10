# leed_daylight/deliver/calculator/v02.py
"""USGBC "LEED v4.1 Daylight and Quality Views Calculator" V02, as read on
2026-10-09 (calculator spec 3): where each value goes, and the check that a
file is that calculator, still blank. Another version is refused by name,
never filled by guess."""
from __future__ import annotations

import dataclasses
import posixpath
import re
import types
from collections.abc import Mapping

from .xlsx_rows import Book

__all__ = ["CONTENT", "CalculatorUnsupported", "DAYLIGHT", "SHEETS", "Table", "UNITS_CELL", "UNITS_TEXT", "VIEWS",
           "check_template", "rows_needed"]

SHEETS = ("Instructions", "Daylight", "Quality Views", "Quality Views HC", "Summary D+C", "Summary HC", "Summary WDC")


@dataclasses.dataclass(frozen=True)
class Table:
    sheet: str
    #: The first data row and how many the blank calculator has.
    first_row: int
    template_rows: int
    #: ``AddA``'s row: the marker row minus 3 (spec 5.2).
    insert_after: int
    #: Value name -> column letter.
    columns: Mapping[str, str]


DAYLIGHT = Table("Daylight", 8, 8, 13, types.MappingProxyType({
    "id": "B", "description": "C", "area": "D", "sda": "F", "ase": "G", "in_band": "K", "above": "L", "shades": "M"}))
VIEWS = Table("Quality Views", 7, 10, 14, types.MappingProxyType({
    "id": "B", "description": "C", "area": "D", "sight": "H", "vlt": "I", "content": "J"}))

UNITS_CELL = ("Instructions", "J10")
UNITS_TEXT = types.MappingProxyType({"IP": "IP units", "SI": "SI units"})

#: J's dropdown items: Excel splits the literal list at every comma (spec 3).
CONTENT = types.MappingProxyType({
    "distance": "objects at least 25 feet (7.5 meters) from the exterior of the glazing",
    "nature": "nature",
    "art": "urban landmarks or art",
    "none": "other (view does not qualify)",
})

#: (sheet, cell, how, expected): ``text`` equal after stripping, ``contains``
#: in the text, ``formula`` equal, ``formula_contains`` in the formula.
_CHECKS: tuple[tuple[str, str, str, str], ...] = (
    ("Instructions", "K48", "text", "V02"),
    ("Daylight", "B5", "text", "Space ID"),
    ("Daylight", "C5", "text", "Space Description"),
    ("Daylight", "F7", "contains", "sDA300/50%"),
    ("Daylight", "G7", "contains", "ASE1000,250"),
    ("Daylight", "K7", "formula_contains", "between 300 lux and 3,000 lux"),
    ("Daylight", "L7", "formula_contains", "above 3,000 lux"),
    ("Daylight", "M7", "contains", "View-preserving"),
    ("Daylight", "A16", "text", "A"),
    ("Daylight", "D16", "formula", "SUM(D8:D15)"),
    ("Daylight", "H16", "formula", "IFERROR(SUMPRODUCT(F8:F15,D8:D15)/SUM(D8:D15),0)"),
    ("Daylight", "N16", "formula", "SUM(N8:N15)"),
    ("Quality Views", "B5", "text", "Space ID"),
    ("Quality Views", "H5", "contains", "three times the head height"),
    ("Quality Views", "I5", "contains", "(VLT)"),
    ("Quality Views", "J5", "text", "View Content"),
    ("Quality Views", "A17", "text", "A"),
    ("Quality Views", "J17", "formula", "SUM(D7:D16)"),
    ("Quality Views", "J18", "formula_contains", 'SUMIFS(D7:D16,H7:H16,"=Yes"'),
    ("Summary D+C", "H10", "formula", "Daylight!D16"),
    ("Summary D+C", "H14", "formula", "Daylight!H16"),
    ("Summary D+C", "H22", "formula", "Daylight!N16"),
    ("Summary D+C", "H41", "formula", "'Quality Views'!J17"),
    ("Summary D+C", "H43", "formula", "'Quality Views'!J18"),
)


class CalculatorUnsupported(ValueError):
    """The file is not a blank USGBC calculator V02."""


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


#: What ``insert_rows`` does not move: a table, a filter, a threaded
#: comment, a page break, a hyperlink into the workbook. On a sheet that
#: gets rows, Excel would then "repair" the file.
_UNMOVED = (("a table", re.compile(r"<tablePart\b")), ("a filter", re.compile(r"<autoFilter\b")),
            ("a threaded comment", re.compile(r"threadedComment")), ("page breaks", re.compile(r"<rowBreaks\b")),
            ("a hyperlink into the workbook", re.compile(r"<hyperlink\b[^>]*\blocation=")))


def check_template(book: Book) -> None:
    """Refuse, naming the first difference, a file that is not a blank V02."""
    sheets = book.sheets()
    names = tuple(sheets)
    for sheet in SHEETS:
        if sheet not in names:
            raise CalculatorUnsupported(f"no sheet {sheet!r} (the sheets are {', '.join(names)}); "
                                        "this is not the LEED v4.1 Daylight and Quality Views Calculator V02")
    for table in (DAYLIGHT, VIEWS):
        folder, base = posixpath.split(sheets[table.sheet])
        for part in (sheets[table.sheet], f"{folder}/_rels/{base}.rels"):
            if part not in book.parts:
                continue
            text = book.text(part)
            for what, pattern in _UNMOVED:
                if pattern.search(text):
                    raise CalculatorUnsupported(f"{part} ({table.sheet}) holds {what}, which inserted rows would "
                                                "not move; this is not the LEED v4.1 Daylight and Quality Views "
                                                "Calculator V02")
    for sheet, ref, how, expected in _CHECKS:
        cell = book.cell(sheet, ref)
        text, formula = _norm(cell.value), _norm(cell.formula)
        ok = {"text": text == expected, "contains": expected in text,
              "formula": formula == expected, "formula_contains": expected in formula}[how]
        if not ok:
            raise CalculatorUnsupported(f"{sheet}!{ref} is {cell.formula or cell.value!r}, not {expected!r}; "
                                        "this is not the LEED v4.1 Daylight and Quality Views Calculator V02")
    for table in (DAYLIGHT, VIEWS):
        for row in range(table.first_row, table.first_row + table.template_rows):
            for column in table.columns.values():
                if book.cell(table.sheet, f"{column}{row}").value not in (None, ""):
                    raise CalculatorUnsupported(f"the calculator is not blank: {table.sheet}!{column}{row} holds a "
                                                "value; use a blank copy of the calculator")


def rows_needed(table: Table, n: int) -> int:
    """How many rows to insert for ``n`` rows of data."""
    return max(0, n - table.template_rows)
