# leed_daylight/deliver/calculator/build.py
"""Fill a blank calculator V02 for one case (calculator spec 5.1).

The template is checked (``v02.check_template``), rows are inserted as
``AddA`` does, values written, and the file is set to recalculate on open.
It is written as ``.part`` and renamed, with ``calculator.json`` beside it:
the units, the template's and each results file's sha256, the sha256 of
the VLT values, the row counts,
the expected Summary values and the warnings."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import zipfile
import zlib

from ...project.materials import VLT_ROLES, load_materials, vlt_values
from ...engine.in_use import replace_explained
from ..data import CaseData, DeliverError, result_path
from . import v02
from .rows import Rows, calculator_rows
from .xlsx_rows import Book, force_recalculation, insert_rows, set_cells

__all__ = ["CALCULATOR_FILE", "CalculatorResult", "RECORD", "RECORD_FORMAT", "build_calculator", "calculator_dir", "vlt_digest"]

CALCULATOR_FILE = "LEED_v4.1_Daylight_QualityViews_{case}.xlsm"
RECORD = "calculator.json"
#: Raise when the rows a calculator holds change meaning (2: an option not on the LEED basis is warned);
#: a record with no format or an older one is stale, so the package leaves its calculator out.
RECORD_FORMAT = 2


@dataclasses.dataclass(frozen=True)
class CalculatorResult:
    path: pathlib.Path
    record: pathlib.Path
    rows: dict[str, int]
    expected: dict
    warnings: tuple[str, ...]


def calculator_dir(folder, case: str) -> pathlib.Path:
    return pathlib.Path(folder) / "_run" / "calculator" / case


#: What reading a file that is not a workbook, or not the calculator, can raise.
_UNREADABLE = (zipfile.BadZipFile, KeyError, IndexError, ValueError, zlib.error, EOFError, NotImplementedError,
               RuntimeError)


def vlt_digest(materials: pathlib.Path) -> str:
    """sha256 of materials.csv's VLT values (the only input of the calculator
    that no results file records)."""
    return hashlib.sha256(json.dumps(vlt_values(materials), sort_keys=True).encode("utf-8")).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fill(book: Book, table: v02.Table, rows: list[dict]) -> None:
    insert_rows(book, table.sheet, table.insert_after, v02.rows_needed(table, len(rows)))
    values = {}
    for i, row in enumerate(rows):
        for key, value in row.items():
            if value is not None:
                values[f"{table.columns[key]}{table.first_row + i}"] = value
    set_cells(book, table.sheet, values)


def build_calculator(case: CaseData, template: pathlib.Path) -> CalculatorResult:
    units = case.project.calculator.units
    if units is None:
        raise DeliverError("project.yaml: calculator.units is not set; set it to IP or SI, as the project is "
                           "registered with LEED")
    template = pathlib.Path(template)
    try:
        book = Book.read(template)
    except OSError as exc:
        raise DeliverError(f"cannot read {template.name}: {exc}") from None
    except _UNREADABLE as exc:
        raise v02.CalculatorUnsupported(f"{template.name} cannot be read as a workbook ({exc}); this is not the "
                                        "LEED v4.1 Daylight and Quality Views Calculator V02") from None
    try:
        v02.check_template(book)
    except v02.CalculatorUnsupported:
        raise
    except _UNREADABLE as exc:
        raise v02.CalculatorUnsupported(f"{template.name} cannot be read as the calculator ({exc}); this is not "
                                        "the LEED v4.1 Daylight and Quality Views Calculator V02") from None
    materials = case.folder / "materials.csv"
    vlts = vlt_values(materials)
    if "views" in case.options:
        missing = [m for m, role in load_materials(materials).items() if role in VLT_ROLES and m not in vlts]
        if missing:
            raise DeliverError(f"materials.csv: {', '.join(missing)} need a vlt (the glass's visible light "
                               "transmittance in percent, from the product data sheet) for the calculator's "
                               "Quality Views rows; set it in materials.csv")
    results = {option: _sha256(result_path(case.folder, option, case.case)) for option in case.options}
    rows: Rows = calculator_rows(case, units)
    set_cells(book, v02.UNITS_CELL[0], {v02.UNITS_CELL[1]: v02.UNITS_TEXT[units]})
    _fill(book, v02.DAYLIGHT, [dataclasses.asdict(r) for r in rows.daylight])
    _fill(book, v02.VIEWS, [dataclasses.asdict(r) for r in rows.views])
    force_recalculation(book)

    out_dir = calculator_dir(case.folder, case.case)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CALCULATOR_FILE.format(case=case.case)
    part = path.with_name(path.name + ".part")
    record_path = out_dir / RECORD
    record_part = record_path.with_name(RECORD + ".part")
    counts = {"daylight": len(rows.daylight), "views": len(rows.views)}
    record = {
        "format": RECORD_FORMAT, "case": case.case, "file": path.name, "units": units, "template_sha256": _sha256(template),
        "vlt_sha256": vlt_digest(materials),
        "results": results, "rows": counts, "expected": rows.expected, "warnings": list(rows.warnings),
    }
    # The record says "current": remove the old one first so a failure never leaves it beside a new file.
    record_path.unlink(missing_ok=True)
    try:
        book.write(part)
        replace_explained(part, path)
        record_part.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8", newline="\n")
        os.replace(record_part, record_path)
    finally:
        part.unlink(missing_ok=True)
        record_part.unlink(missing_ok=True)
    return CalculatorResult(path, record_path, counts, rows.expected, rows.warnings)
