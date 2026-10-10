"""A stand-in for USGBC's calculator V02, written by the tests.

It has V02's sheet names, the cells ``v02.check_template`` reads, the data
rows with their formulas, the marker rows, the totals and the Summary links,
a drawing anchored below each table and a fake macro part. It holds no
USGBC text beyond those cells. The real file is used only by
tests/acceptance/test_calculator_template.py."""
from __future__ import annotations

from xlsx_build import make_book

SHARED = ("V02", "IP units", "Space ID", "Space Description", " sDA300/50% value (%)", "ASE1000,250 value (%)",
          "For spaces with illuminance above 3,000 lux: View-preserving automated shades?",
          "Occupants are within three times the head height and have direct access to the view",
          "Visible light transmittance of the glazing (VLT) (%)", "View Content", "A", "Office 1")


def _s(text: str) -> str:
    return f'<v>{SHARED.index(text)}</v>'


def _instructions() -> str:
    return ('<sheetData>'
            f'<row r="10"><c r="J10" t="s">{_s("IP units")}</c></row>'
            f'<row r="48"><c r="K48" t="s">{_s("V02")}</c></row>'
            '<row r="49"><c r="M49"><f>IF(J10="IP units",1,2)</f><v>1</v></c></row>'
            '<row r="53"><c r="M53" t="str"><f>IF(M49=1,"(sq ft)","(sq m)")</f><v>(sq ft)</v></c></row>'
            '</sheetData>')


def _daylight(filled: bool) -> str:
    rows = [f'<row r="5"><c r="B5" t="s">{_s("Space ID")}</c><c r="C5" t="s">{_s("Space Description")}</c></row>',
            '<row r="7">'
            f'<c r="F7" t="s">{_s(" sDA300/50% value (%)")}</c><c r="G7" t="s">{_s("ASE1000,250 value (%)")}</c>'
            '<c r="K7" t="str"><f>CONCATENATE("Area with daylight illuminance levels between 300 lux and 3,000 lux at 9 a.m. and 3 p.m. ",Instructions!$M$53)</f><v>x</v></c>'
            '<c r="L7" t="str"><f>CONCATENATE("Area with daylight illuminance levels above 3,000 lux at 9 a.m. and/or 3 p.m.",Instructions!$M$53)</f><v>x</v></c>'
            f'<c r="M7" t="s">{_s("For spaces with illuminance above 3,000 lux: View-preserving automated shades?")}</c></row>']
    for r in range(8, 16):
        b = f'<c r="B{r}" s="2" t="s">{_s("Office 1")}</c>' if filled and r == 8 else f'<c r="B{r}" s="2"/>'
        rows.append(
            f'<row r="{r}"><c r="A{r}"><f>IF(D{r}&gt;=0,0,1)</f><v>0</v></c>{b}<c r="C{r}" s="2"/><c r="D{r}" s="3"/>'
            f'<c r="F{r}" s="3"/><c r="G{r}" s="3"/>'
            f'<c r="H{r}" t="str"><f>IFERROR(IF(ISBLANK(D{r}),"N/A",(IF(F{r}&gt;=55,"Yes","No"))),"No")</f><v>N/A</v></c>'
            f'<c r="I{r}" t="str"><f>IFERROR(IF(ISBLANK(G{r}),"N/A",(IF(G{r}&lt;=10,"Yes","No"))),"No")</f><v>N/A</v></c>'
            f'<c r="K{r}" s="3"/><c r="L{r}" s="3"/><c r="M{r}" s="2"/>'
            f'<c r="N{r}"><f>IF(M{r}="Yes", (K{r}+ L{r}), K{r})</f><v>0</v></c></row>')
    rows.append(f'<row r="16"><c r="A16" t="s">{_s("A")}</c><c r="D16"><f>SUM(D8:D15)</f><v>0</v></c>'
                '<c r="H16"><f>IFERROR(SUMPRODUCT(F8:F15,D8:D15)/SUM(D8:D15),0)</f><v>0</v></c>'
                '<c r="N16"><f>SUM(N8:N15)</f><v>0</v></c></row>')
    rows.append('<row r="18"><c r="H18" t="str"><f>IF(COUNTIF(H8:H15,"No"),"No",IF(COUNTIF(H8:H15,"Yes"),"Yes","N/A"))</f><v>N/A</v></c></row>')
    return ('<dimension ref="A5:N18"/><sheetData>' + "".join(rows) + '</sheetData>'
            '<dataValidations count="1"><dataValidation type="list" sqref="M8:M15"><formula1>"Yes, No"</formula1></dataValidation></dataValidations>'
            '<drawing r:id="rId1"/>')


def _views() -> str:
    rows = [f'<row r="5"><c r="B5" t="s">{_s("Space ID")}</c>'
            f'<c r="H5" t="s">{_s("Occupants are within three times the head height and have direct access to the view")}</c>'
            f'<c r="I5" t="s">{_s("Visible light transmittance of the glazing (VLT) (%)")}</c>'
            f'<c r="J5" t="s">{_s("View Content")}</c></row>']
    for r in range(7, 17):
        rows.append(f'<row r="{r}"><c r="A{r}"><f>IF(D{r}&gt;=0,0,1)</f><v>0</v></c><c r="B{r}" s="2"/><c r="C{r}" s="2"/>'
                    f'<c r="D{r}" s="3"/><c r="H{r}" s="2"/><c r="I{r}" s="3"/><c r="J{r}" s="2"/></row>')
    rows.append(f'<row r="17"><c r="A17" t="s">{_s("A")}</c><c r="J17"><f>SUM(D7:D16)</f><v>0</v></c></row>')
    rows.append('<row r="18"><c r="J18"><f>SUMIFS(D7:D16,H7:H16,"=Yes", I7:I16,"&gt;=40", J7:J16,"&lt;&gt;other (view does not qualify)")</f><v>0</v></c></row>')
    return '<dimension ref="A5:J18"/><sheetData>' + "".join(rows) + "</sheetData>"


def _summary() -> str:
    cells = {"H10": "Daylight!D16", "H14": "Daylight!H16", "H16": "Daylight!H18", "H22": "Daylight!N16",
             "H24": "IFERROR(H22/H10,0)", "H41": "'Quality Views'!J17", "H43": "'Quality Views'!J18",
             "H45": "IFERROR(H43/H41,0)"}
    rows = "".join(f'<row r="{ref[1:]}"><c r="{ref}"><f>{f.replace("&", "&amp;")}</f><v>0</v></c></row>' for ref, f in cells.items())
    return f"<sheetData>{rows}</sheetData>"


_DRAWING = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing">'
            '<xdr:twoCellAnchor><xdr:from><xdr:col>1</xdr:col><xdr:row>18</xdr:row></xdr:from>'
            '<xdr:to><xdr:col>2</xdr:col><xdr:row>19</xdr:row></xdr:to></xdr:twoCellAnchor></xdr:wsDr>')
MACRO = b"\xd0\xcf\x11\xe0 stand-in macro project"


def add_vlt(folder, value: str = "62") -> None:
    """Add a `vlt` column to a test project's materials.csv: ``value`` on
    every `glazing` row, empty elsewhere."""
    path = folder / "materials.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [lines[0] + ",vlt"] + [line + ("," + value if ",glazing," in line else ",") for line in lines[1:]]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def set_units(folder, units: str = "SI") -> None:
    """Add ``calculator: units`` to a test project's project.yaml."""
    path = folder / "project.yaml"
    path.write_text(path.read_text(encoding="utf-8") + f"calculator:\n  units: {units}\n", encoding="utf-8")


def write_standin(path, *, filled: bool = False):
    """The stand-in calculator at ``path``; ``filled`` puts a space name in
    Daylight!B8, as a calculator someone already started would have."""
    sheets = {"Instructions": _instructions(), "Daylight": _daylight(filled), "Quality Views": _views(),
              "Quality Views HC": "<sheetData/>", "Summary D+C": _summary(), "Summary HC": "<sheetData/>",
              "Summary WDC": "<sheetData/>"}
    rels = '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing1.xml"/>'
    return make_book(path, sheets, shared=SHARED, parts={"xl/vbaProject.bin": MACRO, "xl/drawings/drawing1.xml": _DRAWING},
                     sheet_rels={"Daylight": rels})
