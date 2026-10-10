# tests/acceptance/test_calculator_template.py
"""The real USGBC calculator V02, filled and recalculated (calculator spec 6).

Runs only when LEED_CALCULATOR_TEMPLATE names a blank copy and LibreOffice's
``soffice`` is on PATH. USGBC's file is never part of this repository."""
import os
import shutil
import subprocess
import zipfile

import pytest

from calculator_standin import add_vlt, set_units
from deliver_folder import CASE, build_deliver_project
from leed_daylight.deliver.calculator.build import build_calculator
from leed_daylight.deliver.calculator.xlsx_rows import Book
from leed_daylight.deliver.data import load_case

TEMPLATE = os.environ.get("LEED_CALCULATOR_TEMPLATE")
pytestmark = pytest.mark.skipif(not TEMPLATE or shutil.which("soffice") is None,
                                reason="set LEED_CALCULATOR_TEMPLATE to a blank V02 calculator; needs soffice")

#: Invented, round rows (no fixture value; ``test_no_fixture_value_is_hard_coded``
#: allows numbers here only in ``SYNTHETIC_`` constants): 20 Daylight rows, 12
#: more than the template's 8, every third with view-preserving shades; 25
#: Quality Views rows, 15 more than its 10, cycling through the four contents.
SYNTHETIC_ROWS = {"daylight": 20, "views": 25, "area": 10.0, "sda": 50.0, "ase": 5.0, "in_band": 8.0,
                  "above": 1.0, "shades_every": 3, "view_area": 10.0, "vlt": 60.0}
#: Summary D+C cell -> the expected value it shows once recalculated.
SYNTHETIC_SUMMARY_CELLS = {"H10": "total_area", "H14": "average_sda", "H22": "daylit_area",
                           "H41": "views_total", "H43": "views_area"}
#: The expected values are rounded to 0.1.
SYNTHETIC_TOLERANCE = 0.051
SYNTHETIC_SOFFICE_TIMEOUT_S = 300


def test_the_real_calculator_recalculates_to_the_expected_summary(tmp_path, write_example, write_wea, monkeypatch):
    import leed_daylight.deliver.calculator.build as build
    from leed_daylight.deliver.calculator.rows import DaylightRow, Rows, ViewsRow, expected_summary
    from leed_daylight.deliver.calculator.v02 import CONTENT

    with zipfile.ZipFile(TEMPLATE) as z:
        assert "xl/vbaProject.bin" in z.namelist(), "the template has no macro project; use the .xlsm calculator"
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    add_vlt(folder)
    set_units(folder)
    s = SYNTHETIC_ROWS
    daylight = tuple(DaylightRow(f"S{i}", "office", s["area"] + i, s["sda"] + i, s["ase"], s["in_band"], s["above"],
                                 "Yes" if i % s["shades_every"] == 0 else "No") for i in range(s["daylight"]))
    # Every other row has a view ("Yes"), else none ("No"); the contents cycle through all four.
    contents = tuple(CONTENT.values())
    views = tuple(ViewsRow(f"V{i}", "office", s["view_area"], "No" if i & 1 else "Yes", s["vlt"],
                           contents[i % len(contents)]) for i in range(s["views"]))
    expected = expected_summary(daylight, views, True)
    monkeypatch.setattr(build, "calculator_rows", lambda case, units: Rows(daylight, views, expected, ()))
    result = build_calculator(load_case(folder, CASE), TEMPLATE)

    with zipfile.ZipFile(TEMPLATE) as a, zipfile.ZipFile(result.path) as b:
        for name in a.namelist():
            if name == "xl/vbaProject.bin" or name.startswith("xl/ctrlProps/") or name == "xl/styles.xml":
                assert a.read(name) == b.read(name), name

    run = subprocess.run(["soffice", "--headless", "--calc", "--convert-to", "xlsx", "--outdir", str(tmp_path / "lo"),
                          str(result.path)], capture_output=True, text=True, timeout=SYNTHETIC_SOFFICE_TIMEOUT_S)
    converted = tmp_path / "lo" / result.path.with_suffix(".xlsx").name
    assert run.returncode == 0 and converted.is_file(), f"soffice failed ({run.returncode}): {run.stderr}"
    book = Book.read(converted)
    value = lambda ref: float(book.cell("Summary D+C", ref).value)
    assert value("H10") == pytest.approx(sum(r.area for r in daylight))
    assert value("H14") == pytest.approx(sum(r.sda * r.area for r in daylight) / sum(r.area for r in daylight))
    for ref, key in SYNTHETIC_SUMMARY_CELLS.items():
        assert result.expected[key] is not None, key
        assert value(ref) == pytest.approx(result.expected[key], abs=SYNTHETIC_TOLERANCE), (ref, key)
    assert book.cell("Instructions", "M53").value == "(sq m)"
