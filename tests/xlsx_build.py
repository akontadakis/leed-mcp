# tests/xlsx_build.py
"""Tiny .xlsx/.xlsm workbooks written by hand, for the calculator's
workbook tests. Each sheet is given as the XML between <worksheet> and
</worksheet>; everything else is the least a reader needs."""
from __future__ import annotations

import pathlib
import zipfile

NS = ('xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')


def make_book(path, sheets: dict[str, str], *, shared=(), workbook_extra="", parts=None, sheet_rels=None):
    """Write a workbook to ``path``. ``sheets``: name -> inner sheet XML, in
    order. ``shared``: the shared strings, in index order. ``parts``: extra
    zip parts (name -> bytes or text). ``sheet_rels``: sheet name -> the XML
    inside <Relationships> for that sheet."""
    path = pathlib.Path(path)
    parts = dict(parts or {})
    sheet_rels = dict(sheet_rels or {})
    names = list(sheets)
    content = ['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
               '<Default Extension="xml" ContentType="application/xml"/>',
               '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.ms-excel.sheet.macroEnabled.main+xml"/>',
               '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>',
               '<Override PartName="/xl/calcChain.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.calcChain+xml"/>']
    rels = []
    sheet_entries = []
    files = {}
    for i, name in enumerate(names, start=1):
        files[f"xl/worksheets/sheet{i}.xml"] = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<worksheet {NS}>{sheets[name]}</worksheet>'
        content.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
        sheet_entries.append(f'<sheet name="{name.replace("&", "&amp;")}" sheetId="{i}" r:id="rId{i}"/>')
        if name in sheet_rels:
            files[f"xl/worksheets/_rels/sheet{i}.xml.rels"] = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{sheet_rels[name]}</Relationships>')
    n = len(names)
    rels.append(f'<Relationship Id="rId{n + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>')
    rels.append(f'<Relationship Id="rId{n + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/calcChain" Target="calcChain.xml"/>')
    files["[Content_Types].xml"] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                                    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                                    + "".join(content) + "</Types>")
    files["_rels/.rels"] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    files["xl/workbook.xml"] = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<workbook {NS}><sheets>'
                                + "".join(sheet_entries) + f"</sheets>{workbook_extra}<calcPr calcId=\"152511\"/></workbook>")
    files["xl/_rels/workbook.xml.rels"] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                                           '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                           + "".join(rels) + "</Relationships>")
    items = "".join(f"<si><t>{s}</t></si>" for s in shared)
    files["xl/sharedStrings.xml"] = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<sst {NS} '
                                     f'count="{len(shared)}" uniqueCount="{len(shared)}">{items}</sst>')
    files["xl/calcChain.xml"] = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<calcChain {NS}><c r="A1" i="1"/></calcChain>'
    files.update(parts)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def assert_edited_parts_well_formed(before: dict[str, bytes], after: dict[str, bytes]) -> list[str]:
    """Every ``.xml``/``.rels`` part that differs between ``before`` and
    ``after`` parses as XML (a read-only check; nothing is written back).
    Returns the names checked."""
    from xml.etree import ElementTree

    checked = []
    for name, data in after.items():
        if name.endswith((".xml", ".rels")) and before.get(name) != data:
            try:
                ElementTree.fromstring(data)
            except ElementTree.ParseError as exc:
                raise AssertionError(f"{name} is not well-formed XML after the edit: {exc}") from None
            checked.append(name)
    return checked
