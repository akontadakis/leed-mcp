# tests/test_calculator_command.py
"""build_calculator and the calculator verb (calculator spec 5.1, 5.2)."""
import json
import pathlib
import zipfile

import pytest

from calculator_standin import MACRO, add_vlt, set_units, write_standin
from deliver_folder import CASE, build_deliver_project
from xlsx_build import assert_edited_parts_well_formed
from leed_daylight.deliver.calculator.build import build_calculator, calculator_dir
from leed_daylight.deliver.calculator.xlsx_rows import Book
from leed_daylight.deliver.data import DeliverError, load_case
from leed_daylight.drivers.calculator import main


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    add_vlt(folder)
    set_units(folder)
    return folder


def test_the_file_is_written_with_the_rows_and_the_record(folder, tmp_path):
    template = write_standin(tmp_path / "blank.xlsm")
    result = build_calculator(load_case(folder, CASE), template)
    assert result.path.parent == calculator_dir(folder, CASE)
    book = Book.read(result.path)
    edited = assert_edited_parts_well_formed(Book.read(template).parts, book.parts)
    assert {"xl/workbook.xml", "xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"} <= set(edited)
    assert book.cell("Instructions", "J10").value == "SI units"
    assert book.cell("Daylight", "B8").value == "Room North"
    assert book.cell("Daylight", "D8").value == "24"
    assert book.cell("Quality Views", "B7").value == "Room North (view, distance)"
    assert book.cell("Daylight", "D16").formula == "SUM(D8:D15)"
    assert book.cell("Daylight", "D16").value is None                 # recalculated on open
    with zipfile.ZipFile(result.path) as z:
        assert z.read("xl/vbaProject.bin") == MACRO
    record = json.loads(result.record.read_text(encoding="utf-8"))
    assert record["rows"] == {"daylight": 1, "views": 2} and record["units"] == "SI"
    assert set(record["results"]) == {"option1", "option2", "views"}


def test_twenty_and_twenty_five_rows_keep_the_summary_links(folder, tmp_path, monkeypatch):
    import leed_daylight.deliver.calculator.build as build
    from leed_daylight.deliver.calculator.rows import DaylightRow, Rows, ViewsRow

    rows = Rows(
        tuple(DaylightRow(f"S{i}", "office", 10.0, 50.0, 5.0, 8.0, 1.0, "No") for i in range(20)),
        tuple(ViewsRow(f"V{i}", "office", 10.0, "Yes", 60.0, "nature") for i in range(25)),
        {}, ())
    monkeypatch.setattr(build, "calculator_rows", lambda case, units: rows)
    template = write_standin(tmp_path / "blank.xlsm")
    result = build_calculator(load_case(folder, CASE), template)
    book = Book.read(result.path)
    edited = assert_edited_parts_well_formed(Book.read(template).parts, book.parts)
    assert {"xl/worksheets/sheet2.xml", "xl/worksheets/sheet3.xml", "xl/drawings/drawing1.xml"} <= set(edited)
    assert book.cell("Daylight", "B27").value == "S19"
    assert book.cell("Daylight", "D28").formula == "SUM(D8:D27)"
    assert book.cell("Summary D+C", "H10").formula == "Daylight!D28"
    assert book.cell("Quality Views", "B31").value == "V24"
    assert book.cell("Summary D+C", "H41").formula == "'Quality Views'!J32"


def test_the_verb_refuses_without_units(folder, tmp_path, capsys):
    path = folder / "project.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("calculator:\n  units: SI\n", ""), encoding="utf-8", newline="\n")
    code = main([str(folder), "--case", CASE, "--template", str(write_standin(tmp_path / "blank.xlsm"))])
    assert code == 1
    assert "calculator.units" in capsys.readouterr().err


def test_the_verb_refuses_another_file(folder, tmp_path, capsys):
    other = tmp_path / "other.xlsm"
    other.write_bytes(b"not a zip")
    assert main([str(folder), "--case", CASE, "--template", str(other)]) == 1
    assert "not the LEED v4.1 Daylight and Quality Views Calculator" in capsys.readouterr().err


def test_the_verb_prints_the_file_and_the_expected_summary(folder, tmp_path, capsys):
    assert main([str(folder), "--case", CASE, "--template", str(write_standin(tmp_path / "blank.xlsm"))]) == 0
    out = capsys.readouterr().out
    assert "LEED_v4.1_Daylight_QualityViews_base_case.xlsm" in out and "expected Summary" in out


def test_a_failed_write_leaves_no_part_and_no_old_record(folder, tmp_path, monkeypatch):
    template = write_standin(tmp_path / "blank.xlsm")
    first = build_calculator(load_case(folder, CASE), template)
    assert first.record.exists()

    def boom(self, path):
        pathlib.Path(path).write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(Book, "write", boom)
    with pytest.raises(OSError):
        build_calculator(load_case(folder, CASE), template)
    assert not first.record.exists()
    assert not list(calculator_dir(folder, CASE).glob("*.part"))


def test_a_missing_template_is_a_read_error(folder, tmp_path):
    with pytest.raises(DeliverError, match="cannot read nope.xlsm"):
        build_calculator(load_case(folder, CASE), tmp_path / "nope.xlsm")


from leed_daylight.deliver.calculator.build import CALCULATOR_FILE


def _package(folder):
    import io
    from leed_daylight.drivers.package import package_folder
    from leed_daylight.drivers.report import report_folder
    report_folder(folder, CASE, out=io.StringIO(), blender=False)
    return package_folder(folder, CASE, out=io.StringIO())


def test_a_current_calculator_is_packaged(folder, tmp_path):
    build_calculator(load_case(folder, CASE), write_standin(tmp_path / "blank.xlsm"))
    package = _package(folder)
    name = CALCULATOR_FILE.format(case=CASE)
    assert (package / "calculator" / name).is_file()
    assert f"calculator/{name}" in (package / "README.md").read_text(encoding="utf-8")
    assert f"calculator/{name}" in (package / "checksums.sha256").read_text(encoding="utf-8")


def test_a_calculator_older_than_the_results_is_left_out(folder, tmp_path):
    from deliver_folder import edit_results
    build_calculator(load_case(folder, CASE), write_standin(tmp_path / "blank.xlsm"))
    edit_results(folder, "option1", lambda r: r.update({"warnings": ["changed"]}))
    package = _package(folder)
    assert not (package / "calculator").exists()


def _readme(package) -> str:
    return (package / "README.md").read_text(encoding="utf-8")


def test_a_calculator_filled_in_other_units_is_left_out(folder, tmp_path):
    from leed_daylight.deliver.package import CALCULATOR_STALE
    build_calculator(load_case(folder, CASE), write_standin(tmp_path / "blank.xlsm"))
    path = folder / "project.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("units: SI", "units: IP"), encoding="utf-8", newline="\n")
    package = _package(folder)
    assert not (package / "calculator").exists()
    assert CALCULATOR_STALE in _readme(package)


def test_a_calculator_filled_with_another_vlt_is_left_out(folder, tmp_path):
    from leed_daylight.deliver.package import CALCULATOR_STALE
    build_calculator(load_case(folder, CASE), write_standin(tmp_path / "blank.xlsm"))
    path = folder / "materials.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",62\n", ",70\n"), encoding="utf-8", newline="\n")
    package = _package(folder)
    assert not (package / "calculator").exists()
    assert CALCULATOR_STALE in _readme(package)


@pytest.mark.parametrize("edit", [
    lambda record: "not json",
    lambda record: json.dumps({k: v for k, v in record.items() if k != "vlt_sha256"}),
    lambda record: json.dumps({k: v for k, v in record.items() if k != "format"}),
    lambda record: json.dumps(dict(record, format=1)),
    lambda record: json.dumps(dict(record, file="../x")),
])
def test_a_malformed_calculator_record_is_left_out(folder, tmp_path, edit):
    result = build_calculator(load_case(folder, CASE), write_standin(tmp_path / "blank.xlsm"))
    (result.path.parent.parent / "x").write_bytes(b"x")
    result.record.write_text(edit(json.loads(result.record.read_text(encoding="utf-8"))), encoding="utf-8", newline="\n")
    package = _package(folder)
    assert not (package / "calculator").exists()


def test_a_calculator_filled_before_the_record_format_is_stale(folder, tmp_path):
    from leed_daylight.deliver.calculator.build import RECORD_FORMAT
    from leed_daylight.deliver.package import CALCULATOR_STALE
    result = build_calculator(load_case(folder, CASE), write_standin(tmp_path / "blank.xlsm"))
    record = json.loads(result.record.read_text(encoding="utf-8"))
    assert record["format"] == RECORD_FORMAT
    del record["format"]
    result.record.write_text(json.dumps(record), encoding="utf-8", newline="\n")
    package = _package(folder)
    assert not (package / "calculator").exists()
    assert CALCULATOR_STALE in _readme(package)


def test_no_calculator_says_nothing_in_the_readme(folder):
    from leed_daylight.deliver.package import CALCULATOR_STALE
    assert CALCULATOR_STALE not in _readme(_package(folder))


def test_the_verb_refuses_vision_glazing_without_a_vlt(folder, tmp_path, capsys):
    path = folder / "materials.csv"
    text = path.read_text(encoding="utf-8")
    glazing = [line.split(",")[0] for line in text.splitlines() if ",glazing," in line]
    assert glazing
    path.write_text(text.replace(",62\n", ",\n"), encoding="utf-8", newline="\n")
    code = main([str(folder), "--case", CASE, "--template", str(write_standin(tmp_path / "blank.xlsm"))])
    assert code == 1
    err = capsys.readouterr().err
    assert "need a vlt" in err and all(name in err for name in glazing)
    assert not calculator_dir(folder, CASE).exists() or not list(calculator_dir(folder, CASE).glob("*.xlsm"))


def test_a_template_the_check_cannot_read_is_unsupported(folder, tmp_path, capsys):
    template = write_standin(tmp_path / "blank.xlsm")
    book = Book.read(template)
    part = book.sheets()["Daylight"]
    # A shared-string index past the end: reading the cell raises IndexError inside the check.
    book.put(part, book.text(part).replace('<c r="B5" t="s"><v>2</v>', '<c r="B5" t="s"><v>99</v>'))
    book.write(tmp_path / "broken.xlsm")
    assert main([str(folder), "--case", CASE, "--template", str(tmp_path / "broken.xlsm")]) == 1
    err = capsys.readouterr().err
    assert "not the LEED v4.1 Daylight and Quality Views Calculator V02" in err and "Traceback" not in err
