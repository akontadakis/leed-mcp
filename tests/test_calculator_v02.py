"""The V02 map and its template check (calculator spec 3, 4)."""
import pytest

from calculator_standin import write_standin
from leed_daylight.deliver.calculator import v02
from leed_daylight.deliver.calculator.xlsx_rows import Book, set_cells


def test_the_standin_passes_the_check(tmp_path):
    v02.check_template(Book.read(write_standin(tmp_path / "c.xlsm")))


@pytest.mark.parametrize("sheet, ref, value, words", [
    ("Instructions", "K48", "V03", "Instructions!K48"),
    ("Daylight", "B5", "Room", "Daylight!B5"),
    ("Quality Views", "J5", "Content", "Quality Views!J5"),
])
def test_a_changed_cell_is_named(tmp_path, sheet, ref, value, words):
    book = Book.read(write_standin(tmp_path / "c.xlsm"))
    set_cells(book, sheet, {ref: value})
    with pytest.raises(v02.CalculatorUnsupported, match=words):
        v02.check_template(book)


def test_a_missing_sheet_is_named(tmp_path):
    book = Book.read(write_standin(tmp_path / "c.xlsm"))
    book.put("xl/workbook.xml", book.text("xl/workbook.xml").replace('name="Summary WDC"', 'name="Other"'))
    with pytest.raises(v02.CalculatorUnsupported, match="Summary WDC"):
        v02.check_template(book)


def test_a_started_calculator_is_refused(tmp_path):
    book = Book.read(write_standin(tmp_path / "c.xlsm", filled=True))
    with pytest.raises(v02.CalculatorUnsupported, match=r"not blank: Daylight!B8"):
        v02.check_template(book)


@pytest.mark.parametrize("sheet, part, text, what", [
    ("Daylight", "xl/worksheets/sheet2.xml", '<tableParts count="1"><tablePart r:id="rId9"/></tableParts>', "a table"),
    ("Daylight", "xl/worksheets/sheet2.xml", '<autoFilter ref="B7:M15"/>', "a filter"),
    ("Quality Views", "xl/worksheets/sheet3.xml", '<rowBreaks count="1"><brk id="10"/></rowBreaks>', "page breaks"),
    ("Quality Views", "xl/worksheets/sheet3.xml",
     '<hyperlinks><hyperlink ref="B5" location="Summary!A1"/></hyperlinks>', "a hyperlink into the workbook"),
    ("Daylight", "xl/worksheets/_rels/sheet2.xml.rels",
     '<Relationship Id="rId9" Type="http://schemas.microsoft.com/office/2017/10/relationships/threadedComment" '
     'Target="../threadedComments/t1.xml"/>', "a threaded comment"),
])
def test_a_part_rows_would_not_move_is_refused(tmp_path, sheet, part, text, what):
    book = Book.read(write_standin(tmp_path / "c.xlsm"))
    close = "</Relationships>" if part.endswith(".rels") else "</worksheet>"
    book.put(part, book.text(part).replace(close, text + close))
    with pytest.raises(v02.CalculatorUnsupported, match=f"{part}.*{what}"):
        v02.check_template(book)


@pytest.mark.parametrize("n, extra", [(0, 0), (8, 0), (9, 1), (20, 12)])
def test_rows_needed(n, extra):
    assert v02.rows_needed(v02.DAYLIGHT, n) == extra
