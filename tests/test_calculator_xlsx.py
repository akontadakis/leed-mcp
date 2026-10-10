# tests/test_calculator_xlsx.py
"""The workbook editing under the USGBC calculator (calculator spec 5.1)."""
import pytest

from leed_daylight.deliver.calculator.xlsx_refs import (
    col_letters, col_number, shift_formula, shift_sqref, translate_formula,
)


@pytest.mark.parametrize("letters, n", [("A", 1), ("Z", 26), ("AA", 27), ("AZ", 52), ("XFD", 16384)])
def test_columns_round_trip(letters, n):
    assert col_number(letters) == n
    assert col_letters(n) == letters


@pytest.mark.parametrize("formula, expected", [
    ("SUM(D8:D15)", "SUM(D8:D18)"),             # a range straddling the insert grows
    ("D16", "D19"),                              # below: moves
    ("$D$16", "$D$19"),                          # absolute rows move too: it is an insert
    ("D13", "D13"),                              # at the insert row: stays
    ("D8:D13", "D8:D13"),                        # wholly above: stays
    ('IF(D14>=0,"D20",1)', 'IF(D17>=0,"D20",1)'),  # text in quotes is never a reference
    ("LOG10(D20)", "LOG10(D23)"),                # a function name is not a reference
])
def test_shift_in_its_own_sheet(formula, expected):
    assert shift_formula(formula, "Daylight", "Daylight", 13, 3) == expected


def test_shift_from_another_sheet_only_touches_that_sheet():
    formula = "Daylight!D16+'Quality Views'!J17+D16"
    assert shift_formula(formula, "Summary D+C", "Daylight", 13, 3) == "Daylight!D19+'Quality Views'!J17+D16"
    assert shift_formula(formula, "Summary D+C", "Quality Views", 14, 2) == "Daylight!D16+'Quality Views'!J19+D16"


def test_a_quoted_sheet_name_with_an_apostrophe():
    assert shift_formula("'Bob''s'!A20", "x", "Bob's", 13, 1) == "'Bob''s'!A21"


def test_an_external_or_broken_reference_is_left_alone():
    assert shift_formula("[1]Instructions!#REF!", "", "Instructions", 1, 5) == "[1]Instructions!#REF!"


@pytest.mark.parametrize("formula, drow, dcol, expected", [
    ("IF(D13>=0,0,1)", 2, 0, "IF(D15>=0,0,1)"),
    ("$D$13+D$13+$D13", 2, 1, "$D$13+E$13+$D15"),
    ('IF(M13="Yes", (K13+ L13), K13)', 1, 0, 'IF(M14="Yes", (K14+ L14), K14)'),
])
def test_translate_moves_relative_parts_only(formula, drow, dcol, expected):
    assert translate_formula(formula, drow, dcol) == expected


def test_shift_sqref_handles_several_ranges():
    assert shift_sqref("P8:P15 K8:K15 D16", 13, 2) == "P8:P17 K8:K17 D18"

import zipfile

from leed_daylight.deliver.calculator.xlsx_rows import Book, Cell, force_recalculation, set_cells, unshare
from xlsx_build import assert_edited_parts_well_formed, make_book

_ROWS = ('<sheetData>'
         '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" s="3"/></row>'
         '<row r="2"><c r="A2"><v>5</v></c><c r="B2" t="str"><f>A2*2</f><v>10</v></c></row>'
         '</sheetData>')


def _book(tmp_path, **kwargs):
    return Book.read(make_book(tmp_path / "in.xlsm", {"One": _ROWS}, shared=("Space ID",),
                               parts={"xl/vbaProject.bin": b"\x00macro bytes\x01"}, **kwargs))


def test_cells_read_shared_strings_numbers_and_formulas(tmp_path):
    book = _book(tmp_path)
    assert book.sheets() == {"One": "xl/worksheets/sheet1.xml"}
    assert book.cell("One", "A1") == Cell(formula=None, value="Space ID")
    assert book.cell("One", "A2") == Cell(formula=None, value="5")
    assert book.cell("One", "B2") == Cell(formula="A2*2", value="10")
    assert book.cell("One", "B1") == Cell(formula=None, value=None)
    assert book.cell("One", "Z9") == Cell(formula=None, value=None)


def test_write_keeps_every_untouched_part_byte_for_byte(tmp_path):
    book = _book(tmp_path)
    book.write(tmp_path / "out.xlsm")
    with zipfile.ZipFile(tmp_path / "in.xlsm") as a, zipfile.ZipFile(tmp_path / "out.xlsm") as b:
        assert a.namelist() == b.namelist()
        assert all(a.read(n) == b.read(n) for n in a.namelist())


def test_set_cells_writes_text_and_numbers_and_keeps_the_style(tmp_path):
    book = _book(tmp_path)
    set_cells(book, "One", {"B1": 'R&D <North> "1"', "A2": 12.5, "C2": "=not a formula"})
    assert book.cell("One", "B1") == Cell(formula=None, value='R&D <North> "1"')
    assert book.cell("One", "A2") == Cell(formula=None, value="12.5")
    assert book.cell("One", "C2") == Cell(formula=None, value="=not a formula")
    xml = book.text("xl/worksheets/sheet1.xml")
    assert '<c r="B1" s="3" t="inlineStr">' in xml
    assert xml.index('r="A2"') < xml.index('r="B2"') < xml.index('r="C2"')


def test_set_cells_drops_illegal_controls_and_keeps_x_escapes_literal(tmp_path):
    book = _book(tmp_path)
    before = dict(book.parts)
    set_cells(book, "One", {"B1": "Room\x01 A\x0bB\tC", "A2": "Lab_x0041_"})
    assert_edited_parts_well_formed(before, book.parts)
    assert book.cell("One", "B1").value == "Room AB\tC"
    # Excel reads "_x005F_" as "_", so it shows "Lab_x0041_", not "LabA".
    assert book.cell("One", "A2").value == "Lab_x005F_x0041_"


def test_set_cells_refuses_a_formula_cell(tmp_path):
    book = _book(tmp_path)
    with pytest.raises(ValueError, match="B2 holds a formula"):
        set_cells(book, "One", {"B2": 1})


def test_force_recalculation_drops_cached_values_and_the_calc_chain(tmp_path):
    book = _book(tmp_path)
    force_recalculation(book)
    assert "xl/calcChain.xml" not in book.parts
    assert "calcChain" not in book.text("[Content_Types].xml")
    assert "calcChain" not in book.text("xl/_rels/workbook.xml.rels")
    assert 'fullCalcOnLoad="1"' in book.text("xl/workbook.xml")
    assert book.cell("One", "B2") == Cell(formula="A2*2", value=None)
    book.write(tmp_path / "out.xlsm")
    assert "xl/calcChain.xml" not in zipfile.ZipFile(tmp_path / "out.xlsm").namelist()


@pytest.mark.parametrize("names", ["", '<definedNames><definedName name="X">One!$A$1</definedName></definedNames>'])
def test_force_recalculation_adds_calc_pr_in_schema_order_when_missing(tmp_path, names):
    book = _book(tmp_path, workbook_extra=names)
    book.put("xl/workbook.xml", book.text("xl/workbook.xml").replace('<calcPr calcId="152511"/>',
                                                                     "<extLst><ext uri=\"x\"/></extLst>"))
    force_recalculation(book)
    xml = book.text("xl/workbook.xml")
    anchor = "</definedNames>" if names else "</sheets>"
    assert anchor + '<calcPr fullCalcOnLoad="1"/>' in xml
    assert xml.index("<calcPr") < xml.index("<extLst>")


def test_unshare_writes_each_shared_formula_out():
    xml = ('<row r="8"><c r="A8"><f t="shared" ref="A8:A10" si="0">IF(D8&gt;=0,0,1)</f><v>0</v></c></row>'
           '<row r="9"><c r="A9"><f t="shared" si="0"/><v>0</v></c></row>')
    out = unshare(xml)
    assert '<c r="A8"><f>IF(D8&gt;=0,0,1)</f>' in out
    assert '<c r="A9"><f>IF(D9&gt;=0,0,1)</f>' in out
    assert "shared" not in out


def test_formulas_with_attributes_are_seen_and_recalculated(tmp_path):
    rows = ('<sheetData><row r="1">'
            '<c r="A1"><f ca="1">NOW()</f><v>45000</v></c>'
            '<c r="B1"><f t="array" ref="B1">SUM(C1:C2)</f><v>7</v></c>'
            '</row></sheetData>')
    book = Book.read(make_book(tmp_path / "in.xlsm", {"One": rows}))
    assert book.cell("One", "A1") == Cell(formula="NOW()", value="45000")
    assert book.cell("One", "B1") == Cell(formula="SUM(C1:C2)", value="7")
    with pytest.raises(ValueError, match="A1 holds a formula"):
        set_cells(book, "One", {"A1": 1})
    force_recalculation(book)
    xml = book.text("xl/worksheets/sheet1.xml")
    assert '<f ca="1">NOW()</f></c>' in xml
    assert '<f t="array" ref="B1">SUM(C1:C2)</f></c>' in xml
    assert "<v>" not in xml


def test_unshare_reads_a_master_with_other_attributes_first():
    xml = ('<row r="8"><c r="A8"><f ca="1" t="shared" si="0" ref="A8:A10">IF(D8&gt;=0,0,1)</f><v>0</v></c></row>'
           '<row r="9"><c r="A9"><f t="shared" si="0"/><v>0</v></c></row>')
    out = unshare(xml)
    assert '<c r="A8"><f ca="1">IF(D8&gt;=0,0,1)</f>' in out
    assert '<c r="A9"><f>IF(D9&gt;=0,0,1)</f>' in out
    assert "shared" not in out


def test_unshare_names_a_child_whose_master_is_missing():
    xml = '<row r="9"><c r="A9"><f t="shared" si="7"/><v>0</v></c></row>'
    with pytest.raises(ValueError, match="A9.*shared formula 7"):
        unshare(xml)


def test_force_recalculation_does_not_merge_neighbouring_cells(tmp_path):
    rows = ('<sheetData><row r="1">'
            '<c r="A1" t="str"><f>X1+1</f></c>'
            '<c r="B1"><f>B2*3</f></c>'
            '<c r="C1"><v>3</v></c>'
            '<c r="D1"><f>D2*2</f><v>8</v></c>'
            '</row></sheetData>')
    book = Book.read(make_book(tmp_path / "in.xlsm", {"One": rows}))
    force_recalculation(book)
    xml = book.text("xl/worksheets/sheet1.xml")
    assert '<c r="A1" t="str"><f>X1+1</f></c>' in xml
    assert '<c r="B1"><f>B2*3</f></c>' in xml
    assert '<c r="C1"><v>3</v></c>' in xml
    assert '<c r="D1"><f>D2*2</f></c>' in xml


from leed_daylight.deliver.calculator.xlsx_rows import insert_rows

_DATA = (
    '<dimension ref="A1:D6"/><sheetData>'
    '<row r="2"><c r="A2"><f t="shared" ref="A2:A3" si="0">IF(B2&gt;=0,0,1)</f><v>0</v></c><c r="B2" s="7"><v>4</v></c></row>'
    '<row r="3"><c r="A3"><f t="shared" si="0"/><v>0</v></c><c r="B3" s="7"><v>6</v></c></row>'
    '<row r="4"><c r="A4" t="s"><v>0</v></c><c r="B4"><f>SUM(B2:B3)</f><v>10</v></c></row>'
    '</sheetData>'
    '<mergeCells count="1"><mergeCell ref="C4:D4"/></mergeCells>'
    '<conditionalFormatting sqref="B2:B3"><cfRule type="expression" priority="1"><formula>B2&gt;B4</formula></cfRule></conditionalFormatting>'
    '<dataValidations count="1"><dataValidation type="list" sqref="B2:B3 D4"><formula1>"Yes, No"</formula1></dataValidation></dataValidations>'
    '<legacyDrawing r:id="rId2"/>'
    '<controls xmlns:xdr="x"><control r:id="rId3"><controlPr><anchor><from><xdr:col>1</xdr:col><xdr:row>4</xdr:row></from>'
    '<to><xdr:col>2</xdr:col><xdr:row>5</xdr:row></to></anchor></controlPr></control></controls>'
)
_SUMMARY = ('<sheetData><row r="1"><c r="A1"><f>Data!B4</f><v>10</v></c>'
            '<c r="B1"><f>COUNTIF(Data!A2:A3,0)</f><v>2</v></c><c r="C1"><f>B4</f><v>0</v></c></row></sheetData>')
_DRAWING = ('<xdr:wsDr xmlns:xdr="x"><xdr:twoCellAnchor><xdr:from><xdr:col>1</xdr:col><xdr:row>1</xdr:row></xdr:from>'
            '<xdr:to><xdr:col>2</xdr:col><xdr:row>5</xdr:row></xdr:to></xdr:twoCellAnchor></xdr:wsDr>')
_VML = ('<xml><v:shape><x:ClientData ObjectType="Note"><x:Anchor>1, 15, 3, 2, 3, 15, 6, 4</x:Anchor>'
        '<x:Row>4</x:Row><x:Column>1</x:Column></x:ClientData></v:shape></xml>')
_COMMENTS = '<comments><commentList><comment ref="B4" authorId="0"/><comment ref="B1" authorId="0"/></commentList></comments>'


def _table_book(tmp_path):
    rels = ('<Relationship Id="rId1" Type="http://x/drawing" Target="../drawings/drawing1.xml"/>'
            '<Relationship Id="rId2" Type="http://x/vmlDrawing" Target="../drawings/vmlDrawing1.vml"/>'
            '<Relationship Id="rId4" Type="http://x/comments" Target="../comments1.xml"/>'
            '<Relationship Id="rId3" Type="http://x/ctrlProp" Target="../ctrlProps/ctrlProp1.xml"/>')
    path = make_book(
        tmp_path / "t.xlsm", {"Data": _DATA, "Summary": _SUMMARY}, shared=("A",),
        workbook_extra='<definedNames><definedName name="Area">Data!$B$2:$B$3</definedName></definedNames>',
        parts={"xl/drawings/drawing1.xml": _DRAWING, "xl/drawings/vmlDrawing1.vml": _VML,
               "xl/comments1.xml": _COMMENTS, "xl/ctrlProps/ctrlProp1.xml": "<formControlPr/>"},
        sheet_rels={"Data": rels})
    return Book.read(path)


def test_insert_copies_the_row_with_its_formulas_and_styles_but_not_its_values(tmp_path):
    book = _table_book(tmp_path)
    before = dict(book.parts)
    insert_rows(book, "Data", 2, 2)
    assert "xl/worksheets/sheet1.xml" in assert_edited_parts_well_formed(before, book.parts)
    assert book.cell("Data", "A3") == Cell("IF(B3>=0,0,1)", None)
    assert book.cell("Data", "A4") == Cell("IF(B4>=0,0,1)", None)
    assert book.cell("Data", "B3") == Cell(None, None)
    assert '<c r="B4" s="7"/>' in book.text("xl/worksheets/sheet1.xml")
    assert book.cell("Data", "B5") == Cell(None, "6")                     # the old row 3
    assert book.cell("Data", "A5") == Cell("IF(B5>=0,0,1)", "0")


def test_insert_moves_totals_and_every_reference_to_them(tmp_path):
    book = _table_book(tmp_path)
    before = dict(book.parts)
    insert_rows(book, "Data", 2, 2)
    assert "xl/worksheets/sheet1.xml" in assert_edited_parts_well_formed(before, book.parts)
    assert book.cell("Data", "A6") == Cell(None, "A")
    assert book.cell("Data", "B6").formula == "SUM(B2:B5)"
    assert book.cell("Summary", "A1").formula == "Data!B6"
    assert book.cell("Summary", "B1").formula == "COUNTIF(Data!A2:A5,0)"
    assert book.cell("Summary", "C1").formula == "B4"                       # its own sheet: untouched
    assert "<definedName name=\"Area\">Data!$B$2:$B$5</definedName>" in book.text("xl/workbook.xml")


def test_insert_moves_ranges_anchors_and_comments(tmp_path):
    book = _table_book(tmp_path)
    before = dict(book.parts)
    insert_rows(book, "Data", 2, 2)
    assert "xl/worksheets/sheet1.xml" in assert_edited_parts_well_formed(before, book.parts)
    xml = book.text("xl/worksheets/sheet1.xml")
    assert '<dimension ref="A1:D8"/>' in xml
    assert '<mergeCell ref="C6:D6"/>' in xml
    assert 'sqref="B2:B5"' in xml and "<formula>B2&gt;B6</formula>" in xml
    assert 'sqref="B2:B5 D6"' in xml
    assert "<xdr:row>6</xdr:row>" in xml and "<xdr:row>7</xdr:row>" in xml    # the button, zero-based 4 and 5
    drawing = book.text("xl/drawings/drawing1.xml")
    assert "<xdr:row>1</xdr:row>" in drawing and "<xdr:row>7</xdr:row>" in drawing
    vml = book.text("xl/drawings/vmlDrawing1.vml")
    assert "<x:Anchor>1, 15, 5, 2, 3, 15, 8, 4</x:Anchor>" in vml and "<x:Row>6</x:Row>" in vml
    comments = book.text("xl/comments1.xml")
    assert '<comment ref="B6"' in comments and '<comment ref="B1"' in comments
    assert book.parts["xl/ctrlProps/ctrlProp1.xml"] == b"<formControlPr/>"


def test_an_empty_row_and_an_empty_cell_do_not_hide_the_cells_after_them(tmp_path):
    xml = ('<sheetData><row r="2"><c r="A2" s="1"/><c r="B2"><v>3</v></c></row><row r="3"/>'
           '<row r="4"><c r="A4" s="1"/><c r="B4"><v>7</v></c></row></sheetData>')
    book = Book.read(make_book(tmp_path / "e.xlsm", {"Data": xml}))
    before = dict(book.parts)
    insert_rows(book, "Data", 2, 1)
    assert assert_edited_parts_well_formed(before, book.parts)
    assert book.cell("Data", "B5") == Cell(None, "7")              # old row 4, every cell moved
    assert book.cell("Data", "B3") == Cell(None, None)             # the copy of row 2, value cleared
    assert '<row r="4"/>' in book.text("xl/worksheets/sheet1.xml")  # old empty row 3


def test_insert_of_nothing_changes_nothing(tmp_path):
    book = _table_book(tmp_path)
    before = dict(book.parts)
    insert_rows(book, "Data", 2, 0)
    assert book.parts == before


def test_formulas_with_attributes_are_shifted_and_copied_too(tmp_path):
    xml = ('<sheetData><row r="2"><c r="A2"><f ca="1">B2+1</f><v>1</v></c>'
           '<c r="B2"><f t="array" ref="B2">SUM(C2:C2)</f><v>1</v></c></row>'
           '<row r="3"><c r="A3"><f ca="1">SUM(A2:A3)</f><v>1</v></c></row></sheetData>')
    book = Book.read(make_book(tmp_path / "a.xlsm", {"Data": xml}))
    before = dict(book.parts)
    insert_rows(book, "Data", 2, 1)
    assert assert_edited_parts_well_formed(before, book.parts)
    assert book.cell("Data", "A3") == Cell("B3+1", None)            # copied, translated
    assert book.cell("Data", "A4").formula == "SUM(A2:A4)"           # moved below, range grown
    text = book.text("xl/worksheets/sheet1.xml")
    assert '<c r="A3"><f ca="1">B3+1</f></c>' in text
    assert '<f t="array" ref="B3">SUM(C3:C3)</f>' in text


def test_a_moved_array_formula_keeps_its_ref_with_its_cell(tmp_path):
    xml = ('<sheetData><row r="2"><c r="A2"><v>1</v></c></row>'
           '<row r="3"><c r="A3"><f t="array" ref="A3">SUM(B3:B3)</f><v>1</v></c></row></sheetData>')
    book = Book.read(make_book(tmp_path / "r.xlsm", {"Data": xml}))
    insert_rows(book, "Data", 2, 2)
    assert '<c r="A5"><f t="array" ref="A5">' in book.text("xl/worksheets/sheet1.xml")


def test_validation_on_another_sheet_that_names_the_target_moves(tmp_path):
    other = ('<sheetData><row r="1"><c r="A1"/></row></sheetData><dataValidations count="1">'
             '<dataValidation type="list" sqref="A1"><formula1>Data!$A$2:$A$3</formula1></dataValidation></dataValidations>')
    data = '<sheetData><row r="2"><c r="A2"><v>1</v></c></row><row r="3"><c r="A3"><v>2</v></c></row></sheetData>'
    book = Book.read(make_book(tmp_path / "o.xlsm", {"Data": data, "Other": other}))
    before = dict(book.parts)
    insert_rows(book, "Data", 2, 2)
    assert assert_edited_parts_well_formed(before, book.parts)
    assert "<formula1>Data!$A$2:$A$5</formula1>" in book.text("xl/worksheets/sheet2.xml")


def test_a_missing_source_row_leaves_the_book_untouched(tmp_path):
    book = _table_book(tmp_path)
    before = dict(book.parts)
    with pytest.raises(ValueError):
        insert_rows(book, "Data", 9, 1)
    assert book.parts == before
